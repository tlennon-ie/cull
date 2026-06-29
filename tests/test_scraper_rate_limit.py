"""tests/test_scraper_rate_limit.py — scraper ↔ rate_limit integration.

The pure pacing/backoff/proxy behaviour of ``rate_limit`` is covered in
``tests/test_rate_limit.py``. This suite verifies the *wiring*: that the
HTTP-making scrapers actually go through their module-level ``RateLimiter``
before each outbound ``requests`` call, splat its proxy kwargs, and react to a
429 — while staying completely unthrottled when no ``RATE_LIMIT_*`` env is set
(opt-in requirement).

Everything is deterministic: ``requests`` is mocked (no real network) and the
limiter's clock/sleep are injected via a fake clock (no real waiting). We keep
coverage focused on two representative scrapers — civitai (the throttling +
no-throttle contract, per the task) and discord (a second backend, plus the
proxy splat) — rather than re-testing all five.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_scraper_rate_limit.py -q
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

import rate_limit  # noqa: E402


# ── Fake clock (mirrors tests/test_rate_limit.py) ─────────────────────────────

class FakeClock:
    """Controllable monotonic clock + sleep that records pacing durations."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = float(start)
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        if dt < 0:
            raise AssertionError(f"negative sleep requested: {dt}")
        if dt > 0:
            self.sleeps.append(dt)
            self.t += dt


# ── Fake requests.Response ────────────────────────────────────────────────────

class FakeResponse:
    """Minimal stand-in for ``requests.Response`` for the scraper call sites."""

    def __init__(self, status_code: int = 200, json_data=None, headers=None,
                 content: bytes = b"x" * 6000, text: str = "") -> None:
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.headers = headers or {}
        self.content = content
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def json(self):
        return self._json


def _fresh_import(mod_name: str):
    """Import a scraper module with a clean slate so module-level limiter
    construction re-reads the (monkeypatched) environment."""
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(
        mod_name, PIPELINE_CODE / f"{mod_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_fake_clock_limiter(mod, source: str, clock: FakeClock):
    """Rebuild the module's limiter from the SAME env-derived config but with an
    injected fake clock, so pacing is observable without real sleeping.

    Returns the installed limiter. Reads ``mod.os.environ`` so it reflects any
    monkeypatched ``RATE_LIMIT_*`` keys the module saw at import time.
    """
    cfg = rate_limit.RateLimitConfig.from_mapping(source, mod.os.environ)
    lim = rate_limit.RateLimiter(cfg, now=clock.now, sleep=clock.sleep,
                                 rand=lambda: 0.0)
    mod._LIMITER = lim
    return lim


# ── civitai: throttling is opt-in via RATE_LIMIT_CIVITAI_* ────────────────────

def test_civitai_paces_calls_when_max_per_min_set(monkeypatch):
    """With RATE_LIMIT_CIVITAI_MAX_PER_MIN=2, the 3rd tRPC call in the same
    window must pace (the limiter sleeps on a fake clock)."""
    monkeypatch.setenv("RATE_LIMIT_CIVITAI_MAX_PER_MIN", "2")
    monkeypatch.delenv("RATE_LIMIT_CIVITAI_MIN_INTERVAL_S", raising=False)
    monkeypatch.delenv("RATE_LIMIT_CIVITAI_PROXY", raising=False)

    civitai = _fresh_import("scraper_civitai")
    clock = FakeClock()
    lim = _install_fake_clock_limiter(civitai, "civitai", clock)
    assert lim.config.max_per_min == 2  # env actually took effect

    calls: list[dict] = []

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return FakeResponse(200, json_data={"result": {"data": {"json": {"ok": 1}}}})

    monkeypatch.setattr(civitai.requests, "get", fake_get)

    # Three metadata fetches; budget is 2/min → the 3rd must wait ~60s.
    civitai.trpc_get("image.getGenerationData", {"id": 1})
    civitai.trpc_get("image.getGenerationData", {"id": 2})
    assert clock.sleeps == []  # first two fit the window

    civitai.trpc_get("image.getGenerationData", {"id": 3})

    assert len(calls) == 3
    assert clock.sleeps, "3rd call should have been paced by the limiter"
    assert sum(clock.sleeps) == pytest.approx(60.0, abs=1e-6)


def test_civitai_does_not_throttle_with_no_env(monkeypatch):
    """With NO RATE_LIMIT_* env set, behaviour is unchanged: the limiter is
    unthrottled and never sleeps regardless of call volume."""
    for key in (
        "RATE_LIMIT_CIVITAI_MAX_PER_MIN",
        "RATE_LIMIT_CIVITAI_MIN_INTERVAL_S",
        "RATE_LIMIT_CIVITAI_PROXY",
        "RATE_LIMIT_CIVITAI_MAX_BACKOFF_S",
    ):
        monkeypatch.delenv(key, raising=False)

    civitai = _fresh_import("scraper_civitai")
    clock = FakeClock()
    lim = _install_fake_clock_limiter(civitai, "civitai", clock)
    assert lim.config.max_per_min is None  # unthrottled
    assert lim.remaining() is None

    def fake_get(url, **kwargs):
        return FakeResponse(200, json_data={"result": {"data": {"json": {"ok": 1}}}})

    monkeypatch.setattr(civitai.requests, "get", fake_get)

    for i in range(25):
        civitai.trpc_get("image.getGenerationData", {"id": i})

    assert clock.sleeps == [], "no RATE_LIMIT_* env → must never pace"


def test_civitai_429_opens_backoff_and_retries(monkeypatch):
    """A 429 on the first tRPC attempt records a backoff (honouring
    Retry-After) and the retry loop pulls through to the eventual 200."""
    monkeypatch.delenv("RATE_LIMIT_CIVITAI_MAX_PER_MIN", raising=False)

    civitai = _fresh_import("scraper_civitai")
    clock = FakeClock()
    lim = _install_fake_clock_limiter(civitai, "civitai", clock)

    responses = [
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(200, json_data={"result": {"data": {"json": {"got": "it"}}}}),
    ]

    def fake_get(url, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(civitai.requests, "get", fake_get)

    out = civitai.trpc_get("image.getGenerationData", {"id": 99})

    assert out == {"got": "it"}
    # The 429 set a 7s gate; acquire() before the retry slept it out.
    assert sum(clock.sleeps) == pytest.approx(7.0, abs=1e-6)
    # Success cleared the backoff.
    assert lim.current_backoff() == 0.0


def test_civitai_proxy_splatted_into_requests(monkeypatch):
    """RATE_LIMIT_CIVITAI_PROXY flows through requests_kwargs into every get()."""
    monkeypatch.setenv("RATE_LIMIT_CIVITAI_PROXY", "http://127.0.0.1:8888")
    monkeypatch.delenv("RATE_LIMIT_CIVITAI_MAX_PER_MIN", raising=False)

    civitai = _fresh_import("scraper_civitai")
    clock = FakeClock()
    _install_fake_clock_limiter(civitai, "civitai", clock)

    seen_kwargs: list[dict] = []

    def fake_get(url, **kwargs):
        seen_kwargs.append(kwargs)
        return FakeResponse(200, json_data={"result": {"data": {"json": {}}}})

    monkeypatch.setattr(civitai.requests, "get", fake_get)

    civitai.trpc_get("image.getGenerationData", {"id": 1})

    assert seen_kwargs, "request was made"
    assert seen_kwargs[0].get("proxies") == {
        "http": "http://127.0.0.1:8888",
        "https": "http://127.0.0.1:8888",
    }


# ── discord: second representative backend ────────────────────────────────────

def test_discord_paces_downloads_when_min_interval_set(monkeypatch):
    """A second scraper proves the wiring generalises: with a min-interval set,
    consecutive download_bytes calls are paced."""
    monkeypatch.setenv("RATE_LIMIT_DISCORD_MIN_INTERVAL_S", "0.5")
    monkeypatch.setenv("RATE_LIMIT_DISCORD_MAX_PER_MIN", "1000")
    monkeypatch.setenv("DISCORD_AUTH_MODE", "user")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")

    discord = _fresh_import("scraper_discord")
    clock = FakeClock()
    lim = _install_fake_clock_limiter(discord, "discord", clock)
    assert lim.config.min_interval_s == pytest.approx(0.5)

    def fake_get(url, **kwargs):
        return FakeResponse(200, content=b"y" * 6000)

    monkeypatch.setattr(discord.requests, "get", fake_get)

    assert discord.download_bytes("http://img/1") == b"y" * 6000
    assert clock.sleeps == []  # first is free
    discord.download_bytes("http://img/2")
    assert clock.sleeps == pytest.approx([0.5])  # second paced by min-interval


def test_discord_does_not_throttle_with_no_env(monkeypatch):
    """No RATE_LIMIT_DISCORD_* env → downloads never pace."""
    for key in (
        "RATE_LIMIT_DISCORD_MAX_PER_MIN",
        "RATE_LIMIT_DISCORD_MIN_INTERVAL_S",
        "RATE_LIMIT_DISCORD_PROXY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DISCORD_AUTH_MODE", "user")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")

    discord = _fresh_import("scraper_discord")
    clock = FakeClock()
    _install_fake_clock_limiter(discord, "discord", clock)

    def fake_get(url, **kwargs):
        return FakeResponse(200, content=b"z" * 6000)

    monkeypatch.setattr(discord.requests, "get", fake_get)

    for i in range(30):
        discord.download_bytes(f"http://img/{i}")

    assert clock.sleeps == []


def test_discord_retry_after_from_json_body(monkeypatch):
    """Discord sends Retry-After in the JSON body; _retry_after_seconds reads it
    and the download backs off before returning None on the 429."""
    monkeypatch.setenv("DISCORD_AUTH_MODE", "user")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    discord = _fresh_import("scraper_discord")
    clock = FakeClock()
    lim = _install_fake_clock_limiter(discord, "discord", clock)

    def fake_get(url, **kwargs):
        return FakeResponse(429, json_data={"retry_after": 3.5})

    monkeypatch.setattr(discord.requests, "get", fake_get)

    assert discord.download_bytes("http://img/1") is None
    assert lim.current_backoff() == pytest.approx(3.5)
    assert lim.blocked_until() == pytest.approx(clock.now() + 3.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
