"""Tests for ``scheduler`` — per-job scheduling + completion webhooks/notifications.

The scheduler lets cull run a job action (``scrape`` / ``curate`` / ``export``)
on a cadence, and fire a best-effort webhook + desktop notification when work
completes.

Design constraints these tests pin down:

* ``Schedule`` is an immutable-ish dataclass {slug, cadence, action, enabled,
  last_run}. ``cadence`` is either a small cron-ish string (``@daily`` /
  ``@weekly`` / ``@hourly``) OR an interval (``every 30m`` / ``every 2h``).
* The schedule list persists to ``data/schedules.json`` (via ``paths``) with
  add/update/remove/list helpers and atomic writes.
* ``due(schedules, now)`` is PURE — ``now`` is injected, never read from the
  clock — and returns the schedules that should run.
* ``run_due(now, runner)`` calls an INJECTED ``runner`` callback so tests never
  spawn the real pipeline, and stamps ``last_run`` on the ones it ran.
* ``notify(event, payload, webhook_url=...)`` POSTs JSON via ``requests``
  (mockable) and tolerates a missing desktop-notification library.

Hermetic: no network, no real ``.env``, no wall-clock reads in assertions.

Runnable:
    pytest tests/test_scheduler.py -q
    python tests/test_scheduler.py
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise ``dotenv.load_dotenv`` so a developer's real ``.env`` can't leak
    into these hermetic tests."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


@pytest.fixture()
def scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A freshly-imported ``scheduler`` pointed at a temp data dir."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DESKTOP_NOTIFICATIONS", raising=False)
    import scheduler as _sched
    return importlib.reload(_sched)


# A couple of fixed instants used across the cadence tests. Timezone-aware so the
# arithmetic is unambiguous regardless of the machine's local zone.
T0 = datetime(2026, 6, 28, 12, 0, 0, tzinfo=timezone.utc)            # Sunday noon
T_PLUS_25H = datetime(2026, 6, 29, 13, 0, 0, tzinfo=timezone.utc)     # +25h
T_PLUS_8D = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)       # +8 days


def _sched(scheduler, **kw: Any):
    """Build a Schedule with sensible defaults for the test under inspection."""
    base = dict(slug="demo", cadence="@daily", action="scrape",
                enabled=True, last_run=None)
    base.update(kw)
    return scheduler.Schedule(**base)


# ── module smoke ──────────────────────────────────────────────────────────────

def test_module_imports(scheduler) -> None:
    for name in ("Schedule", "due", "run_due", "notify",
                 "list_schedules", "add_schedule", "update_schedule",
                 "remove_schedule"):
        assert hasattr(scheduler, name), f"missing public symbol: {name}"


def test_schedule_is_a_dataclass_with_expected_fields(scheduler) -> None:
    s = _sched(scheduler)
    assert s.slug == "demo"
    assert s.cadence == "@daily"
    assert s.action == "scrape"
    assert s.enabled is True
    assert s.last_run is None


def test_schedule_rejects_unknown_action(scheduler) -> None:
    with pytest.raises(ValueError):
        scheduler.Schedule(slug="demo", cadence="@daily", action="nonsense")


# ── cadence parsing (the small cron subset) ───────────────────────────────────

def test_parse_cadence_aliases(scheduler) -> None:
    assert scheduler.cadence_seconds("@hourly") == 3600
    assert scheduler.cadence_seconds("@daily") == 86400
    assert scheduler.cadence_seconds("@weekly") == 7 * 86400


def test_parse_cadence_interval_minutes_and_hours(scheduler) -> None:
    assert scheduler.cadence_seconds("every 30m") == 30 * 60
    assert scheduler.cadence_seconds("every 2h") == 2 * 3600
    # bare forms / spacing tolerance
    assert scheduler.cadence_seconds("30m") == 30 * 60
    assert scheduler.cadence_seconds("every  90m") == 90 * 60


def test_parse_cadence_rejects_garbage(scheduler) -> None:
    for bad in ("", "yearly", "every 5x", "abc", "every"):
        with pytest.raises(ValueError):
            scheduler.cadence_seconds(bad)


# ── due() — the pure selector ─────────────────────────────────────────────────

def test_due_selects_never_run_enabled_schedule(scheduler) -> None:
    s = _sched(scheduler, last_run=None)
    assert scheduler.due([s], now=T0) == [s]


def test_due_skips_disabled_schedule(scheduler) -> None:
    s = _sched(scheduler, enabled=False, last_run=None)
    assert scheduler.due([s], now=T0) == []


def test_due_skips_when_interval_not_elapsed(scheduler) -> None:
    # Ran at T0, daily cadence, only ~25h later would be due; check at +1h.
    just_after = datetime(2026, 6, 28, 13, 0, 0, tzinfo=timezone.utc)
    s = _sched(scheduler, cadence="@daily", last_run=T0)
    assert scheduler.due([s], now=just_after) == []


def test_due_selects_when_interval_elapsed(scheduler) -> None:
    s = _sched(scheduler, cadence="@daily", last_run=T0)
    assert scheduler.due([s], now=T_PLUS_25H) == [s]


def test_due_weekly_boundary(scheduler) -> None:
    weekly = _sched(scheduler, cadence="@weekly", last_run=T0)
    # +25h: not yet a week
    assert scheduler.due([weekly], now=T_PLUS_25H) == []
    # +8 days: past a week
    assert scheduler.due([weekly], now=T_PLUS_8D) == [weekly]


def test_due_accepts_naive_or_aware_last_run(scheduler) -> None:
    """A persisted last_run round-trips as a naive ISO string in some flows;
    due() must not explode on naive vs aware mismatches."""
    naive_last = datetime(2026, 6, 28, 12, 0, 0)  # no tzinfo
    s = _sched(scheduler, cadence="@hourly", last_run=naive_last)
    later = datetime(2026, 6, 28, 14, 0, 0)       # naive, +2h
    assert scheduler.due([s], now=later) == [s]


def test_due_is_pure_does_not_mutate_inputs(scheduler) -> None:
    s = _sched(scheduler, cadence="@daily", last_run=T0)
    before = (s.slug, s.cadence, s.last_run, s.enabled)
    scheduler.due([s], now=T_PLUS_25H)
    after = (s.slug, s.cadence, s.last_run, s.enabled)
    assert before == after


def test_due_orders_and_filters_multiple(scheduler) -> None:
    a = _sched(scheduler, slug="a", cadence="@hourly", last_run=T0)
    b = _sched(scheduler, slug="b", cadence="@weekly", last_run=T0)   # not due at +25h
    c = _sched(scheduler, slug="c", cadence="@daily", last_run=None)  # never run -> due
    d = _sched(scheduler, slug="d", enabled=False, last_run=None)     # disabled
    selected = {s.slug for s in scheduler.due([a, b, c, d], now=T_PLUS_25H)}
    assert selected == {"a", "c"}


# ── persistence: add / update / remove / list ─────────────────────────────────

def test_list_schedules_empty_on_fresh_install(scheduler) -> None:
    assert scheduler.list_schedules() == []


def test_add_schedule_persists_and_lists(scheduler) -> None:
    s = _sched(scheduler, slug="demo")
    scheduler.add_schedule(s)
    listed = scheduler.list_schedules()
    assert len(listed) == 1
    assert listed[0].slug == "demo"
    assert listed[0].cadence == "@daily"
    # File actually written under data/schedules.json
    import paths
    assert (paths.base_dir() / "schedules.json").exists()


def test_add_schedule_is_idempotent_by_slug_action(scheduler) -> None:
    """Adding the same (slug, action) twice updates rather than duplicates."""
    scheduler.add_schedule(_sched(scheduler, slug="demo", cadence="@daily"))
    scheduler.add_schedule(_sched(scheduler, slug="demo", cadence="@hourly"))
    listed = scheduler.list_schedules()
    assert len(listed) == 1
    assert listed[0].cadence == "@hourly"


def test_distinct_action_same_slug_coexist(scheduler) -> None:
    scheduler.add_schedule(_sched(scheduler, slug="demo", action="scrape"))
    scheduler.add_schedule(_sched(scheduler, slug="demo", action="export"))
    assert len(scheduler.list_schedules()) == 2


def test_update_schedule_changes_fields(scheduler) -> None:
    scheduler.add_schedule(_sched(scheduler, slug="demo", action="scrape", enabled=True))
    scheduler.update_schedule("demo", "scrape", enabled=False, cadence="@weekly")
    s = scheduler.list_schedules()[0]
    assert s.enabled is False
    assert s.cadence == "@weekly"


def test_remove_schedule(scheduler) -> None:
    scheduler.add_schedule(_sched(scheduler, slug="demo", action="scrape"))
    scheduler.add_schedule(_sched(scheduler, slug="demo", action="export"))
    removed = scheduler.remove_schedule("demo", "scrape")
    assert removed is True
    remaining = scheduler.list_schedules()
    assert len(remaining) == 1
    assert remaining[0].action == "export"


def test_remove_unknown_returns_false(scheduler) -> None:
    assert scheduler.remove_schedule("ghost", "scrape") is False


def test_persistence_round_trips_last_run(scheduler) -> None:
    scheduler.add_schedule(_sched(scheduler, slug="demo", last_run=T0))
    s = scheduler.list_schedules()[0]
    assert s.last_run is not None
    # Compare on the instant (tolerate aware/naive normalisation by the loader).
    assert s.last_run.replace(tzinfo=None) == T0.replace(tzinfo=None)


def test_persistence_tolerates_corrupt_file(scheduler) -> None:
    import paths
    p = paths.base_dir() / "schedules.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ this is not json", encoding="utf-8")
    # Should not raise — corrupt file degrades to empty list.
    assert scheduler.list_schedules() == []


# ── run_due() — drives the injected runner ────────────────────────────────────

def test_run_due_calls_runner_for_due_schedules(scheduler) -> None:
    scheduler.add_schedule(_sched(scheduler, slug="a", action="scrape", last_run=None))
    scheduler.add_schedule(_sched(scheduler, slug="b", action="export",
                                  cadence="@weekly", last_run=T0))  # not due at +25h

    calls: list[tuple[str, str]] = []

    def runner(slug: str, action: str) -> None:
        calls.append((slug, action))

    ran = scheduler.run_due(now=T_PLUS_25H, runner=runner)
    assert calls == [("a", "scrape")]
    assert [(s.slug, s.action) for s in ran] == [("a", "scrape")]


def test_run_due_stamps_last_run_and_persists(scheduler) -> None:
    scheduler.add_schedule(_sched(scheduler, slug="a", action="scrape", last_run=None))

    scheduler.run_due(now=T0, runner=lambda slug, action: None)

    # Reload from disk: last_run must now be stamped, so it is no longer due 1s later.
    one_sec_later = datetime(2026, 6, 28, 12, 0, 1, tzinfo=timezone.utc)
    assert scheduler.due(scheduler.list_schedules(), now=one_sec_later) == []
    stamped = scheduler.list_schedules()[0]
    assert stamped.last_run is not None


def test_run_due_runner_error_does_not_abort_others(scheduler) -> None:
    scheduler.add_schedule(_sched(scheduler, slug="bad", action="scrape", last_run=None))
    scheduler.add_schedule(_sched(scheduler, slug="good", action="curate", last_run=None))

    seen: list[str] = []

    def runner(slug: str, action: str) -> None:
        seen.append(slug)
        if slug == "bad":
            raise RuntimeError("boom")

    # Must not propagate the runner's exception; both should be attempted.
    ran = scheduler.run_due(now=T0, runner=runner)
    assert set(seen) == {"bad", "good"}
    # The good one ran cleanly and should be in the result.
    assert "good" in {s.slug for s in ran}


def test_run_due_no_due_returns_empty(scheduler) -> None:
    scheduler.add_schedule(_sched(scheduler, slug="a", cadence="@weekly", last_run=T0))
    calls: list[Any] = []
    ran = scheduler.run_due(now=T_PLUS_25H, runner=lambda *a: calls.append(a))
    assert ran == []
    assert calls == []


# ── notify() — webhook POST + gated desktop notification ──────────────────────

class _FakeResp:
    status_code = 200

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None


def test_notify_posts_json_to_webhook(scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    posted: dict[str, Any] = {}

    def fake_post(url, json=None, timeout=None, **kw):
        posted["url"] = url
        posted["json"] = json
        posted["timeout"] = timeout
        return _FakeResp()

    import scheduler as _s
    monkeypatch.setattr(_s.requests, "post", fake_post)

    ok = scheduler.notify(
        "job_complete",
        {"slug": "demo", "count": 5},
        webhook_url="https://example.com/hook",
    )
    assert ok is True
    assert posted["url"] == "https://example.com/hook"
    assert posted["json"]["event"] == "job_complete"
    assert posted["json"]["slug"] == "demo"
    assert posted["json"]["count"] == 5
    assert posted["timeout"] is not None  # a timeout is always supplied


def test_notify_uses_env_webhook_when_unspecified(scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://env.example/hook")
    captured: dict[str, Any] = {}

    import scheduler as _s
    monkeypatch.setattr(_s.requests, "post",
                        lambda url, **kw: captured.setdefault("url", url) or _FakeResp())

    scheduler.notify("ping", {})
    assert captured["url"] == "https://env.example/hook"


def test_notify_no_webhook_is_noop_but_truthy_for_desktop(scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no webhook configured and desktop off, notify should not POST and
    should not raise."""
    called = {"post": False}

    import scheduler as _s
    monkeypatch.setattr(_s.requests, "post",
                        lambda *a, **k: called.__setitem__("post", True) or _FakeResp())
    monkeypatch.delenv("WEBHOOK_URL", raising=False)

    # Explicit empty webhook_url + no desktop -> nothing posts, returns False.
    result = scheduler.notify("evt", {"x": 1}, webhook_url="")
    assert called["post"] is False
    assert result is False


def test_notify_tolerates_webhook_failure(scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    import scheduler as _s

    def boom(*a, **k):
        raise _s.requests.RequestException("network down")

    monkeypatch.setattr(_s.requests, "post", boom)
    # Best-effort: a webhook error must be swallowed, returning False not raising.
    result = scheduler.notify("evt", {"x": 1}, webhook_url="https://example.com/hook")
    assert result is False


def test_notify_tolerates_missing_desktop_lib(scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the optional desktop-notification library is unavailable, notify must
    still complete (and still POST the webhook)."""
    import scheduler as _s

    # Force the desktop path on, but make the helper behave as if the lib is absent.
    monkeypatch.setenv("DESKTOP_NOTIFICATIONS", "true")
    monkeypatch.setattr(_s, "_desktop_notify", lambda title, message: False)

    posted = {"n": 0}
    monkeypatch.setattr(_s.requests, "post",
                        lambda *a, **k: posted.__setitem__("n", posted["n"] + 1) or _FakeResp())

    result = scheduler.notify("evt", {"y": 2}, webhook_url="https://example.com/hook")
    # Webhook still fired; missing desktop lib did not break anything.
    assert posted["n"] == 1
    assert result is True


def test_desktop_notify_is_safe_when_lib_absent(scheduler) -> None:
    """The internal _desktop_notify must never raise even with no backend
    installed — it returns a bool."""
    out = scheduler._desktop_notify("cull", "hello")
    assert out in (True, False)


# ── payload serialisation safety ──────────────────────────────────────────────

def test_notify_payload_is_json_serialisable(scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    """The posted body must be plain JSON (no datetime objects leaking through)."""
    import scheduler as _s
    captured: dict[str, Any] = {}
    monkeypatch.setattr(_s.requests, "post",
                        lambda url, json=None, **kw: captured.update(body=json) or _FakeResp())

    scheduler.notify("evt", {"when": T0.isoformat(), "n": 3},
                     webhook_url="https://example.com/hook")
    # Round-trips through json without error.
    json.dumps(captured["body"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
