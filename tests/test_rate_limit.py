"""Tests for ``rate_limit`` — per-source pacing, backoff, and proxy helpers.

``rate_limit`` is a pure, deterministic module: every time/sleep dependency is
injected so tests drive a fake clock instead of touching the wall clock. It
gives cull's scrapers a single place to:

  * pace N requests per source inside a rolling window (fixed-window limiter
    with an optional hard ``min_interval_s`` floor between calls),
  * react to HTTP 429s with exponential backoff (cap + jitter) that decays on
    success, surfacing a "blocked until" monotonic timestamp the UI can show,
  * build ``requests`` keyword args (``proxies=...``) from per-source config.

Runnable:
    pytest tests/test_rate_limit.py -q
    python tests/test_rate_limit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

import rate_limit  # noqa: E402


# ── Fake clock ────────────────────────────────────────────────────────────────

class FakeClock:
    """A controllable monotonic clock + sleep.

    ``sleep(dt)`` advances the clock by ``dt`` and records the duration so tests
    can assert *how long* the limiter decided to pace, without real waiting.
    """

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

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


def _limiter(clock: FakeClock, **cfg_kwargs):
    cfg = rate_limit.RateLimitConfig(source="civitai", **cfg_kwargs)
    return rate_limit.RateLimiter(cfg, now=clock.now, sleep=clock.sleep)


# ── RateLimitConfig ───────────────────────────────────────────────────────────

def test_config_defaults_are_permissive():
    cfg = rate_limit.RateLimitConfig(source="x")
    assert cfg.source == "x"
    assert cfg.max_per_min is None or cfg.max_per_min == 0
    assert cfg.min_interval_s == 0.0
    assert cfg.proxy_url is None
    assert cfg.max_backoff_s > 0  # there is always a sane cap


def test_config_is_frozen():
    cfg = rate_limit.RateLimitConfig(source="x")
    with pytest.raises(Exception):
        cfg.source = "mutated"  # type: ignore[misc]


# ── Window pacing ─────────────────────────────────────────────────────────────

def test_paces_calls_within_window_using_fake_clock(clock):
    """With max_per_min=3 the 4th acquire in the same minute must sleep until
    the window rolls, and never touch the real clock."""
    lim = _limiter(clock, max_per_min=3)

    # First 3 are free (no sleeping) — they fit in the window.
    for _ in range(3):
        lim.acquire()
    assert clock.sleeps == []

    t_before = clock.now()
    lim.acquire()  # 4th — must wait for the oldest of the 3 to age out (60s).
    assert clock.sleeps, "4th call should have paced"
    # Total paced time lands the clock ~60s after the first call.
    assert clock.now() == pytest.approx(t_before + 60.0, abs=1e-6)


def test_window_slides_so_steady_state_matches_rate(clock):
    """Over a long run the limiter must not admit more than max_per_min in any
    rolling 60s window."""
    lim = _limiter(clock, max_per_min=5)
    admit_times: list[float] = []
    for _ in range(20):
        lim.acquire()
        admit_times.append(clock.now())

    # No 60s window contains more than 5 admissions.
    for i, t in enumerate(admit_times):
        in_window = [u for u in admit_times if t - 60.0 < u <= t]
        assert len(in_window) <= 5, f"window at idx {i} admitted {len(in_window)}"


def test_min_interval_enforced_between_calls(clock):
    """A hard min_interval floor paces even when the per-minute budget is fine."""
    lim = _limiter(clock, max_per_min=1000, min_interval_s=0.5)
    lim.acquire()
    lim.acquire()
    assert clock.sleeps == pytest.approx([0.5])


def test_min_interval_not_applied_when_enough_time_elapsed(clock):
    lim = _limiter(clock, max_per_min=1000, min_interval_s=0.5)
    lim.acquire()
    clock.advance(2.0)  # plenty of real time passes between calls
    lim.acquire()
    assert clock.sleeps == []  # no extra pacing needed


def test_no_limits_means_no_sleeping(clock):
    lim = _limiter(clock)  # all defaults → unbounded
    for _ in range(50):
        lim.acquire()
    assert clock.sleeps == []


# ── try_acquire (non-blocking) ────────────────────────────────────────────────

def test_try_acquire_returns_true_while_budget_remains(clock):
    lim = _limiter(clock, max_per_min=2)
    assert lim.try_acquire() is True
    assert lim.try_acquire() is True
    # Budget exhausted for this window → non-blocking refusal, no sleeping.
    assert lim.try_acquire() is False
    assert clock.sleeps == []


def test_try_acquire_recovers_after_window(clock):
    lim = _limiter(clock, max_per_min=2)
    assert lim.try_acquire() is True
    assert lim.try_acquire() is True
    assert lim.try_acquire() is False
    clock.advance(61.0)  # window fully rolls
    assert lim.try_acquire() is True


def test_try_acquire_respects_min_interval(clock):
    lim = _limiter(clock, max_per_min=1000, min_interval_s=1.0)
    assert lim.try_acquire() is True
    assert lim.try_acquire() is False  # too soon
    clock.advance(1.0)
    assert lim.try_acquire() is True


def test_try_acquire_blocked_during_backoff(clock):
    lim = _limiter(clock, max_per_min=1000)
    lim.note_429()
    assert lim.try_acquire() is False  # backoff gate closed
    assert clock.sleeps == []


# ── Backoff state machine ─────────────────────────────────────────────────────

def test_429_triggers_backoff_and_blocked_until(clock):
    lim = _limiter(clock, max_backoff_s=60.0)
    assert lim.current_backoff() == 0.0
    t0 = clock.now()
    lim.note_429()
    assert lim.current_backoff() > 0.0
    assert lim.blocked_until() is not None
    assert lim.blocked_until() > t0


def test_backoff_grows_exponentially(clock):
    """Successive 429s grow the backoff (monotonic, modulo jitter) up to the cap."""
    lim = _limiter(clock, base_backoff_s=1.0, max_backoff_s=100.0, jitter=0.0)
    seen = []
    for _ in range(5):
        lim.note_429()
        seen.append(lim.current_backoff())
        clock.advance(lim.current_backoff())  # serve the wait so next is comparable
    # 1, 2, 4, 8, 16 with zero jitter.
    assert seen == pytest.approx([1.0, 2.0, 4.0, 8.0, 16.0])


def test_backoff_capped(clock):
    lim = _limiter(clock, base_backoff_s=1.0, max_backoff_s=5.0, jitter=0.0)
    for _ in range(10):
        lim.note_429()
        clock.advance(lim.current_backoff())
    assert lim.current_backoff() <= 5.0


def test_backoff_jitter_stays_within_bounds(clock):
    """Jitter must never push the wait above the cap or below zero."""
    rng = lambda: 1.0  # noqa: E731 — worst case (max) jitter
    lim = rate_limit.RateLimiter(
        rate_limit.RateLimitConfig(source="x", base_backoff_s=2.0,
                                   max_backoff_s=8.0, jitter=0.5),
        now=clock.now, sleep=clock.sleep, rand=rng,
    )
    for _ in range(8):
        lim.note_429()
        b = lim.current_backoff()
        assert 0.0 <= b <= 8.0
        clock.advance(b)


def test_success_resets_backoff(clock):
    lim = _limiter(clock, base_backoff_s=1.0, max_backoff_s=100.0, jitter=0.0)
    lim.note_429()
    lim.note_429()
    assert lim.current_backoff() > 0.0
    lim.note_success()
    assert lim.current_backoff() == 0.0
    assert lim.blocked_until() is None


def test_success_without_prior_429_is_noop(clock):
    lim = _limiter(clock)
    lim.note_success()
    assert lim.current_backoff() == 0.0
    assert lim.blocked_until() is None


def test_acquire_waits_out_active_backoff(clock):
    """acquire() must block until blocked_until has passed."""
    lim = _limiter(clock, max_per_min=1000, base_backoff_s=4.0,
                   max_backoff_s=100.0, jitter=0.0)
    lim.note_429()
    blocked_until = lim.blocked_until()
    assert blocked_until is not None
    lim.acquire()
    assert clock.now() >= blocked_until
    assert sum(clock.sleeps) == pytest.approx(4.0)


def test_acquire_partial_backoff_already_elapsed(clock):
    """If some of the backoff window already passed, only wait the remainder."""
    lim = _limiter(clock, max_per_min=1000, base_backoff_s=10.0,
                   max_backoff_s=100.0, jitter=0.0)
    lim.note_429()
    clock.advance(6.0)  # 6 of 10s already gone
    lim.acquire()
    assert sum(clock.sleeps) == pytest.approx(4.0)


def test_retry_after_overrides_backoff(clock):
    """An explicit Retry-After (seconds) from the server wins over computed."""
    lim = _limiter(clock, base_backoff_s=1.0, max_backoff_s=1000.0, jitter=0.0)
    t0 = clock.now()
    lim.note_429(retry_after=30.0)
    assert lim.blocked_until() == pytest.approx(t0 + 30.0)
    assert lim.current_backoff() == pytest.approx(30.0)


# ── remaining-budget introspection (for the UI) ───────────────────────────────

def test_remaining_budget_decrements(clock):
    lim = _limiter(clock, max_per_min=5)
    assert lim.remaining() == 5
    lim.acquire()
    assert lim.remaining() == 4
    lim.acquire()
    assert lim.remaining() == 3


def test_remaining_budget_refills_as_window_slides(clock):
    lim = _limiter(clock, max_per_min=3)
    for _ in range(3):
        lim.acquire()
    assert lim.remaining() == 0
    clock.advance(61.0)
    assert lim.remaining() == 3


def test_remaining_unbounded_when_no_limit(clock):
    lim = _limiter(clock)
    assert lim.remaining() is None  # None = unbounded


def test_snapshot_exposes_ui_state(clock):
    """A single dict the dashboard can render: source, remaining, backoff,
    blocked_until, proxied."""
    lim = rate_limit.RateLimiter(
        rate_limit.RateLimitConfig(source="civitai", max_per_min=10,
                                   proxy_url="http://127.0.0.1:8888"),
        now=clock.now, sleep=clock.sleep,
    )
    lim.acquire()
    lim.note_429(retry_after=12.0)
    snap = lim.snapshot()
    assert snap["source"] == "civitai"
    assert snap["remaining"] == 9
    assert snap["backoff_s"] == pytest.approx(12.0)
    assert snap["blocked_until"] == pytest.approx(clock.now() + 12.0)
    assert snap["proxied"] is True


# ── Proxy kwargs ──────────────────────────────────────────────────────────────

def test_requests_kwargs_emits_proxies_for_both_schemes():
    cfg = rate_limit.RateLimitConfig(source="x", proxy_url="http://user:pw@10.0.0.1:3128")
    kwargs = rate_limit.requests_kwargs(cfg)
    assert kwargs == {"proxies": {
        "http": "http://user:pw@10.0.0.1:3128",
        "https": "http://user:pw@10.0.0.1:3128",
    }}


def test_requests_kwargs_empty_without_proxy():
    cfg = rate_limit.RateLimitConfig(source="x")
    assert rate_limit.requests_kwargs(cfg) == {}


def test_requests_kwargs_blank_proxy_is_ignored():
    cfg = rate_limit.RateLimitConfig(source="x", proxy_url="   ")
    assert rate_limit.requests_kwargs(cfg) == {}


def test_limiter_requests_kwargs_delegates(clock):
    lim = rate_limit.RateLimiter(
        rate_limit.RateLimitConfig(source="x", proxy_url="socks5://127.0.0.1:9050"),
        now=clock.now, sleep=clock.sleep,
    )
    assert lim.requests_kwargs() == {"proxies": {
        "http": "socks5://127.0.0.1:9050",
        "https": "socks5://127.0.0.1:9050",
    }}


# ── Config from env / mapping (per-source projection) ─────────────────────────

def test_config_from_mapping_reads_known_keys():
    cfg = rate_limit.RateLimitConfig.from_mapping("civitai", {
        "RATE_LIMIT_CIVITAI_MAX_PER_MIN": "30",
        "RATE_LIMIT_CIVITAI_MIN_INTERVAL_S": "0.5",
        "RATE_LIMIT_CIVITAI_PROXY": "http://127.0.0.1:8888",
        "RATE_LIMIT_CIVITAI_MAX_BACKOFF_S": "120",
    })
    assert cfg.source == "civitai"
    assert cfg.max_per_min == 30
    assert cfg.min_interval_s == pytest.approx(0.5)
    assert cfg.proxy_url == "http://127.0.0.1:8888"
    assert cfg.max_backoff_s == pytest.approx(120.0)


def test_config_from_mapping_defaults_when_absent():
    cfg = rate_limit.RateLimitConfig.from_mapping("x", {})
    assert cfg.source == "x"
    assert cfg.max_per_min is None
    assert cfg.proxy_url is None


def test_config_from_mapping_ignores_garbage_numbers():
    cfg = rate_limit.RateLimitConfig.from_mapping("x", {
        "RATE_LIMIT_X_MAX_PER_MIN": "not-a-number",
        "RATE_LIMIT_X_MIN_INTERVAL_S": "",
    })
    assert cfg.max_per_min is None
    assert cfg.min_interval_s == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
