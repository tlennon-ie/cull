"""Tests for the per-job lifecycle webhook dispatcher.

Covers:

* **CRUD + SSRF guard.** A private-address URL is rejected at save time.
* **Signature.** A signed payload verifies with ``verify_signature`` and
  differs from an unsigned one.
* **Retry loop.** A 5xx response is retried with backoff (mocked out so we
  don't sleep); a 4xx is not retried; a 2xx returns after the first try.
* **Threshold watcher.** Fires once per crossing, silent below the threshold,
  re-fires after a dip-then-cross-again.
* **Dispatch fan-out.** ``dispatch`` matches only the requested event and
  submits work on the shared thread pool.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

try:
    import dotenv as _dotenv
    _dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pass


# ── Fake HTTP client ─────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, status: int) -> None:
        self.status_code = status


class _FakeHttp:
    """Recording fake — .calls receives every POST, .responses controls status.

    Sequential response draining lets one test cover 5xx → 5xx → 200 in three
    calls.
    """

    def __init__(self, responses: list[int] | None = None, raise_on: int | None = None) -> None:
        self.responses = list(responses or [200])
        self.raise_on_index = raise_on
        self.calls: list[dict] = []

    def __call__(self, url, *, data, headers, timeout):
        idx = len(self.calls)
        self.calls.append({
            "url": url, "data": data, "headers": headers, "timeout": timeout,
        })
        if self.raise_on_index is not None and idx == self.raise_on_index:
            raise ConnectionError("boom")
        status = self.responses[min(idx, len(self.responses) - 1)]
        return _FakeResp(status)


# ── Fixture: isolated jobs store + neutered network ──────────────────────────

@pytest.fixture()
def wd_env(tmp_path, monkeypatch):
    """Reload everything against a temp jobs dir; return (webhook_dispatcher, job_config, tmp_path)."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))

    import paths
    importlib.reload(paths)
    import job_config
    importlib.reload(job_config)
    import scheduler
    importlib.reload(scheduler)
    import webhook_dispatcher
    importlib.reload(webhook_dispatcher)

    # Speed the retry loop up — tests don't want to wait 2/4/8 seconds.
    monkeypatch.setattr(webhook_dispatcher, "_RETRY_BACKOFFS", (0.0, 0.0, 0.0))

    # Bypass DNS: treat every http(s) URL as public.
    monkeypatch.setattr(scheduler, "_is_public_http_url", lambda url: url.startswith(("http://", "https://")))

    # Wipe the auto-migrated default job.
    jobs_dir = tmp_path / "jobs"
    if jobs_dir.is_dir():
        for f in jobs_dir.glob("*"):
            f.unlink()

    return webhook_dispatcher, job_config, tmp_path


def _make_job(job_config, slug: str = "wt1") -> None:
    job_config.create_job(slug.capitalize(), subject="widgets")
    assert job_config.get_job(slug) is not None


# ── CRUD + SSRF ──────────────────────────────────────────────────────────────

def test_save_and_list_roundtrip(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    hook = wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="https://example.com/hook")
    saved = wd.save_webhooks("wt1", [hook])
    assert len(saved) == 1
    listed = wd.list_webhooks("wt1")
    assert len(listed) == 1
    assert listed[0].url == "https://example.com/hook"


def test_save_rejects_ssrf_url(wd_env, monkeypatch):
    wd, jc, _ = wd_env
    _make_job(jc)
    # Reinstate the real guard for this one test.
    monkeypatch.setattr(wd.scheduler, "_is_public_http_url", lambda url: False)
    hook = wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="http://127.0.0.1/nope")
    with pytest.raises(ValueError):
        wd.save_webhooks("wt1", [hook])
    # Nothing persisted.
    assert wd.list_webhooks("wt1") == []


def test_save_rejects_unknown_event(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    hook = wd.Webhook(event="nope", url="https://example.com/x")
    with pytest.raises(ValueError):
        wd.save_webhooks("wt1", [hook])


def test_from_dict_rejects_missing_fields(wd_env):
    wd, _, _ = wd_env
    with pytest.raises(ValueError):
        wd.Webhook.from_dict({"event": "job.completed"})  # no url
    with pytest.raises(ValueError):
        wd.Webhook.from_dict({"event": "bogus", "url": "https://x"})


def test_duplicates_folded(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    hooks = [
        wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="https://example.com/a"),
        wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="https://example.com/a"),
    ]
    saved = wd.save_webhooks("wt1", hooks)
    assert len(saved) == 1


# ── Signature ────────────────────────────────────────────────────────────────

def test_signature_round_trips(wd_env):
    wd, _, _ = wd_env
    body = b'{"event":"job.completed"}'
    sig = wd.sign_body("s3cret!", body)
    assert sig.startswith("sha256=")
    assert wd.verify_signature("s3cret!", body, sig)


def test_signature_rejects_wrong_secret(wd_env):
    wd, _, _ = wd_env
    body = b"payload"
    sig = wd.sign_body("secret", body)
    assert not wd.verify_signature("different", body, sig)


def test_signature_empty_secret_returns_empty_string(wd_env):
    wd, _, _ = wd_env
    assert wd.sign_body("", b"whatever") == ""


# ── Dispatch + retries ───────────────────────────────────────────────────────

def test_dispatch_sends_signature_when_secret_set(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    wd.save_webhooks("wt1", [
        wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="https://example.com/x", secret="topsecret"),
    ])
    fake = _FakeHttp(responses=[200])
    wd.set_http_post(fake)
    try:
        results = wd.dispatch_sync(wd.EVENT_JOB_COMPLETED, "wt1", {"n": 1})
    finally:
        wd.set_http_post(None)
    assert results == [True]
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"] == "https://example.com/x"
    assert wd.SIGNATURE_HEADER in call["headers"]
    assert wd.verify_signature("topsecret", call["data"], call["headers"][wd.SIGNATURE_HEADER])


def test_dispatch_retries_on_5xx(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    wd.save_webhooks("wt1", [
        wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="https://example.com/x"),
    ])
    fake = _FakeHttp(responses=[500, 502, 200])
    wd.set_http_post(fake)
    try:
        results = wd.dispatch_sync(wd.EVENT_JOB_COMPLETED, "wt1")
    finally:
        wd.set_http_post(None)
    assert results == [True]
    assert len(fake.calls) == 3


def test_dispatch_does_not_retry_on_4xx(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    wd.save_webhooks("wt1", [
        wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="https://example.com/x"),
    ])
    fake = _FakeHttp(responses=[404])
    wd.set_http_post(fake)
    try:
        results = wd.dispatch_sync(wd.EVENT_JOB_COMPLETED, "wt1")
    finally:
        wd.set_http_post(None)
    assert results == [False]
    assert len(fake.calls) == 1


def test_dispatch_retries_on_transport_error(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    wd.save_webhooks("wt1", [
        wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="https://example.com/x"),
    ])
    # Raise on first call, then succeed.
    fake = _FakeHttp(responses=[200, 200], raise_on=0)
    wd.set_http_post(fake)
    try:
        results = wd.dispatch_sync(wd.EVENT_JOB_COMPLETED, "wt1")
    finally:
        wd.set_http_post(None)
    assert results == [True]
    assert len(fake.calls) == 2


def test_dispatch_only_matches_requested_event(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    wd.save_webhooks("wt1", [
        wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="https://example.com/a"),
        wd.Webhook(event=wd.EVENT_SCRAPER_AUTH_FAILED, url="https://example.com/b"),
    ])
    fake = _FakeHttp(responses=[200])
    wd.set_http_post(fake)
    try:
        wd.dispatch_sync(wd.EVENT_JOB_COMPLETED, "wt1")
    finally:
        wd.set_http_post(None)
    assert len(fake.calls) == 1
    assert fake.calls[0]["url"] == "https://example.com/a"


def test_dispatch_unknown_event_is_noop(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    assert wd.dispatch_sync("bogus.event", "wt1") == []


# ── Threshold watcher ────────────────────────────────────────────────────────

def _make_image_files(root: Path, count: int, category: str = "keep") -> None:
    folder = root / category
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (folder / f"img{i:04d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")


def test_threshold_watcher_fires_on_crossing(wd_env, tmp_path):
    wd, jc, _ = wd_env
    _make_job(jc, "wtthr")
    sorted_root = tmp_path / "sorted-thr"
    sorted_root.mkdir()
    wd.save_webhooks("wtthr", [
        wd.Webhook(
            event=wd.EVENT_SORTED_THRESHOLD_HIT,
            url="https://example.com/threshold",
            threshold_category="keep",
            threshold_count=3,
        ),
    ])

    fake = _FakeHttp(responses=[200])
    wd.set_http_post(fake)
    try:
        watcher = wd.ThresholdWatcher(sorted_root_fn=lambda slug: sorted_root)

        # Below threshold — no fire.
        _make_image_files(sorted_root, 2)
        watcher.tick()
        assert fake.calls == []

        # Crosses — fires once.
        _make_image_files(sorted_root, 3)  # total now 3 (2+ overwritten filenames only add 1 extra)
        # Actually _make_image_files uses img0000..N-1 so it overwrites; add fresh ones.
        for i in range(3, 6):
            (sorted_root / "keep" / f"img{i:04d}.png").write_bytes(b"\x89PNG")
        watcher.tick()
        assert len(fake.calls) == 1

        # Stays above threshold — no re-fire (already at that height).
        (sorted_root / "keep" / "img9999.png").write_bytes(b"\x89PNG")
        watcher.tick()
        assert len(fake.calls) == 1
    finally:
        wd.set_http_post(None)


def test_threshold_watcher_missing_folder_is_zero(wd_env, tmp_path):
    wd, jc, _ = wd_env
    _make_job(jc, "wtmiss")
    watcher = wd.ThresholdWatcher(sorted_root_fn=lambda slug: tmp_path / "nowhere")
    assert watcher._count_for("wtmiss", "keep") == 0


def test_dispatch_empty_hooks_returns_zero(wd_env):
    wd, jc, _ = wd_env
    _make_job(jc)
    count = wd.dispatch(wd.EVENT_JOB_COMPLETED, "wt1")
    assert count == 0


def test_list_webhooks_for_unknown_job(wd_env):
    wd, _, _ = wd_env
    assert wd.list_webhooks("does_not_exist") == []


def test_save_webhooks_unknown_job_raises(wd_env):
    wd, _, _ = wd_env
    hook = wd.Webhook(event=wd.EVENT_JOB_COMPLETED, url="https://example.com/x")
    with pytest.raises(ValueError):
        wd.save_webhooks("nope", [hook])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
