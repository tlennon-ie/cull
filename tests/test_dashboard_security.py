"""Security-regression tests for the dashboard (CodeQL: stack-trace exposure +
path injection).

Two CodeQL rules motivated these tests; both are asserted here so a regression
re-introduces a failing test rather than a silent leak:

  A) py/stack-trace-exposure — an exception caught in an endpoint must NEVER have
     its ``str(exc)`` / ``repr(exc)`` text echoed to the HTTP client. The fix
     routes every error response through a generic, fixed message (logging the
     detail server-side via ``_err`` / inline ``logger.warning``). These tests
     force the underlying call to raise with a recognisable secret-ish marker and
     assert the marker is absent from the response body, while the response is
     still the expected clean envelope + status.

  B) py/path-injection — user-supplied paths are funnelled through
     ``safe_inside``, which now uses an ``os.path.realpath`` + ``commonpath``
     normalize-then-contain pattern. These tests assert the sanitiser's
     semantics directly (inside → resolved Path, traversal/outside → None) and
     that a traversal payload to a representative file-op endpoint is a clean
     400 with nothing written outside the roots.

Fixture mirrors test_dashboard_lastmile.py: PIPELINE_BASE_DIR is set first, then
the storage modules + the dashboard are reloaded so all state lands under a temp
dir and the developer's real .env can't leak in.
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

# Neutralise dotenv so the developer's real .env can't leak into the temp store.
try:
    import dotenv as _dotenv
    _dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pass

_DOTENV_SOURCED_KEYS = (
    "PIPELINE_SLUG", "PIPELINE_TOPIC", "SCRAPER_DISABLED", "X_ACCOUNTS",
    "HF_TOKEN", "HUGGINGFACE_TOKEN",
)
for _k in _DOTENV_SOURCED_KEYS:
    os.environ.pop(_k, None)

# A marker an exception message carries so we can assert it never reaches the
# client. Picked to look like the kind of internal detail a stack trace leaks
# (an absolute filesystem path with a "secret" segment).
_SECRET_MARKER = "C:/internal/secret_stack_detail_DO_NOT_LEAK"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A Flask test client isolated under tmp_path; returns (client, dashboard, tmp_path)."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    for leaked in _DOTENV_SOURCED_KEYS:
        monkeypatch.delenv(leaked, raising=False)
    (tmp_path / ".env").write_text("BLUR_NSFW_THUMBS=true\n", encoding="utf-8")
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))

    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)

    import paths
    importlib.reload(paths)
    import job_config
    importlib.reload(job_config)
    import index_store
    importlib.reload(index_store)
    import thumb_cache
    importlib.reload(thumb_cache)

    import dashboard_enhanced
    dashboard = importlib.reload(dashboard_enhanced)
    dashboard.app.config.update(TESTING=True)

    jobs_dir = tmp_path / "jobs"
    if jobs_dir.is_dir():
        for f in jobs_dir.glob("*"):
            f.unlink()
    return dashboard.app.test_client(), dashboard, tmp_path


def _make_sorted_image(dash, category: str, source: str = "civitai", stem: str = "img") -> Path:
    """Create <sorted>/<category>/<source>/<stem>.png with a .txt + .vision.json sidecar."""
    folder = dash.PIPELINE_SORTED / category / source
    folder.mkdir(parents=True, exist_ok=True)
    img = folder / f"{stem}.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    img.with_suffix(".txt").write_text("caption", encoding="utf-8")
    img.with_name(f"{stem}.vision.json").write_text('{"category": "%s"}' % category, encoding="utf-8")
    return img


# ── A. the _err helper: generic message out, detail logged ────────────────────

def test_err_helper_returns_generic_envelope_without_exc_text(client):
    c, dash, _ = client
    boom = RuntimeError(_SECRET_MARKER)
    # jsonify needs an app context when called outside a live request.
    with dash.app.app_context():
        resp, code = dash._err("something failed", boom, 500)
        body = resp.get_json()
    assert code == 500
    assert body == {"success": False, "data": None, "error": "something failed"}
    # The exception text is never embedded in the client-facing envelope.
    assert _SECRET_MARKER not in body["error"]


def test_err_helper_logs_exception_detail_server_side(client, caplog):
    c, dash, _ = client
    import logging
    with caplog.at_level(logging.WARNING, logger="dashboard"), dash.app.app_context():
        dash._err("explode", RuntimeError(_SECRET_MARKER), 500)
    # Detail STAYS on the server (logs), where operators need it.
    assert any(_SECRET_MARKER in rec.getMessage() for rec in caplog.records)


# ── A. forced-failure endpoints must not leak the exception text ──────────────

def test_export_dataset_failure_is_generic_500(client, monkeypatch):
    """A non-ValueError raised by export_dataset returns a generic 500 envelope
    with no exception text (py/stack-trace-exposure)."""
    c, dash, _ = client
    dash.job_config.create_job("Sec1", subject="x")
    dash.job_config.activate("sec1")

    def boom(*a, **k):
        raise OSError(_SECRET_MARKER)

    monkeypatch.setattr(dash.export_profiles, "export_dataset", boom)
    r = c.post("/api/export/dataset", json={"profile": "kohya"})
    assert r.status_code == 500
    body = r.get_json()
    assert body["success"] is False and body["data"] is None
    assert _SECRET_MARKER not in (body["error"] or "")
    assert body["error"] == "dataset export failed"


def test_export_dataset_value_error_is_generic_400(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Sec1b", subject="x")
    dash.job_config.activate("sec1b")

    def boom(*a, **k):
        raise ValueError(_SECRET_MARKER)

    monkeypatch.setattr(dash.export_profiles, "export_dataset", boom)
    r = c.post("/api/export/dataset", json={"profile": "kohya"})
    assert r.status_code == 400
    body = r.get_json()
    assert _SECRET_MARKER not in (body["error"] or "")


def test_schedule_add_disk_error_does_not_leak_exc(client, monkeypatch):
    """OSError from the persist layer → generic 'could not persist', no exc text."""
    c, dash, _ = client
    dash.job_config.create_job("Sec2", subject="x")

    def boom(_sched):
        raise OSError(_SECRET_MARKER)

    monkeypatch.setattr(dash.scheduler, "add_schedule", boom)
    r = c.post("/api/schedules", json={"slug": "sec2", "cadence": "@daily", "action": "scrape"})
    assert r.status_code == 500
    body = r.get_json()
    assert body["success"] is False
    assert "could not persist" in (body["error"] or "")
    assert _SECRET_MARKER not in (body["error"] or "")


def test_vision_health_probe_failure_does_not_leak_exc(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Sec3", subject="x")

    def boom(fleet, timeout=None):
        raise RuntimeError(_SECRET_MARKER)

    monkeypatch.setattr(dash.fleet_health, "probe_fleet", boom)
    r = c.get("/api/vision/health?job=sec3")
    assert r.status_code == 500
    body = r.get_json()
    assert _SECRET_MARKER not in (body["error"] or "")


def test_duplicates_scan_failure_does_not_leak_exc(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Sec4", subject="x")

    def boom(threshold=10, slug=None):
        raise RuntimeError(_SECRET_MARKER)

    monkeypatch.setattr(dash.phash_dedup, "find_near_duplicates", boom)
    r = c.get("/api/duplicates?job=sec4")
    assert r.status_code == 500
    body = r.get_json()
    assert _SECRET_MARKER not in (body["error"] or "")


def test_caption_write_failure_does_not_leak_exc(client, monkeypatch):
    """An OSError writing the sibling .txt returns a generic 500, no exc text."""
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep")

    real_write = Path.write_text

    def boom(self, *a, **k):
        if self.suffix == ".txt":
            raise OSError(_SECRET_MARKER)
        return real_write(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", boom)
    r = c.post("/api/caption", json={"path": str(img), "text": "hello"})
    assert r.status_code == 500
    body = r.get_json()
    assert body["success"] is False
    assert _SECRET_MARKER not in (body["error"] or "")


def test_create_job_duplicate_does_not_leak_exc(client, monkeypatch):
    """job_config.create_job ValueError must surface as a generic 400, not str(exc)."""
    c, dash, _ = client

    def boom(*a, **k):
        raise ValueError(_SECRET_MARKER)

    monkeypatch.setattr(dash.job_config, "create_job", boom)
    r = c.post("/api/jobs", json={"name": "Whatever"})
    assert r.status_code == 400
    body = r.get_json()
    assert _SECRET_MARKER not in (body["error"] or "")


# ── B. safe_inside sanitiser semantics (normalize + contain) ──────────────────

def test_safe_inside_returns_resolved_path_when_inside(client, tmp_path):
    c, dash, _ = client
    root = tmp_path / "sorted"
    target = root / "Keep" / "civitai" / "a.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    out = dash.safe_inside(str(target), [root])
    assert out is not None
    # Returns the *resolved* path (realpath-normalised), and it is inside root.
    assert out == Path(os.path.realpath(str(target)))
    assert os.path.commonpath([str(out), os.path.realpath(str(root))]) == os.path.realpath(str(root))


def test_safe_inside_returns_none_for_traversal(client, tmp_path):
    c, dash, _ = client
    root = tmp_path / "sorted"
    root.mkdir(parents=True, exist_ok=True)
    evil = str(root / ".." / ".." / "etc" / "passwd.png")
    assert dash.safe_inside(evil, [root]) is None


def test_safe_inside_returns_none_for_outside_root(client, tmp_path):
    c, dash, _ = client
    root = tmp_path / "sorted"
    root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "elsewhere" / "x.png"
    assert dash.safe_inside(str(outside), [root]) is None


def test_safe_inside_resolved_path_inside_for_nonexistent_dest(client, tmp_path):
    """A not-yet-created destination INSIDE a root still resolves to a Path (move
    targets are validated before their dirs exist)."""
    c, dash, _ = client
    root = tmp_path / "queue"
    root.mkdir(parents=True, exist_ok=True)
    out = dash.safe_inside(str(root / "NewCat" / "civitai"), [root])
    assert out is not None


def test_safe_inside_handles_different_drive_gracefully(client, tmp_path):
    """commonpath raising ValueError on a different drive is treated as 'outside'."""
    c, dash, _ = client
    root = tmp_path / "sorted"
    root.mkdir(parents=True, exist_ok=True)
    # A drive letter unlikely to be the test drive; on POSIX this is just an
    # absolute path outside root. Either way → None, never an exception.
    assert dash.safe_inside("Z:/nope/x.png", [root]) is None


# ── B. traversal payloads to file-op endpoints are clean 400s ─────────────────

def test_caption_post_traversal_is_400_and_writes_nothing(client):
    c, dash, _ = client
    evil = str(dash.PIPELINE_SORTED / ".." / ".." / ".." / "etc" / "passwd.png")
    r = c.post("/api/caption", json={"path": evil, "text": "pwned"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False
    # Nothing landed outside the roots.
    assert not (dash.PIPELINE_SORTED.parent.parent / "etc" / "passwd.txt").exists()


def test_duplicates_delete_traversal_is_400(client):
    c, dash, _ = client
    r = c.post("/api/duplicates/delete", json={"path": "/etc/passwd"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_prompt_post_traversal_is_400(client):
    c, dash, _ = client
    evil = str(dash.PIPELINE_QUEUE / ".." / ".." / "etc" / "x.png")
    r = c.post("/api/prompt", json={"path": evil, "text": "x"})
    assert r.status_code == 400


def test_queue_move_traversal_target_is_400_and_no_escape(client, tmp_path):
    """A traversal *target* on the queue mover is rejected and never relocates the
    image outside the queue root (the dest is re-validated via safe_inside)."""
    c, dash, _ = client
    dash.job_config.create_job("SecMove", subject="x")
    dash.job_config.activate("secmove")
    img = dash.PIPELINE_QUEUE / "OffTopic" / "m.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"x")

    r = c.post("/api/queue/action",
               json={"path": str(img), "action": "move", "target": "../../escaped"})
    assert r.status_code == 400
    # The image stays put; no "escaped" folder was created outside the queue.
    assert img.exists()
    assert not (dash.PIPELINE_QUEUE.parent.parent / "escaped").exists()


def test_queue_move_valid_target_still_moves(client):
    """Regression guard: the safe_inside hardening must not break a normal move."""
    c, dash, _ = client
    dash.job_config.create_job("SecMove2", subject="x")
    dash.job_config.activate("secmove2")
    img = dash.PIPELINE_QUEUE / "OffTopic" / "ok.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"x")

    r = c.post("/api/queue/action",
               json={"path": str(img), "action": "move", "target": "Keep"})
    assert r.status_code == 200 and r.get_json()["success"] is True
    assert (dash.PIPELINE_QUEUE / "Keep" / "ok.png").exists()


def test_recategorize_traversal_source_is_400(client):
    c, dash, _ = client
    evil = str(dash.PIPELINE_SORTED / ".." / ".." / "etc" / "p.png")
    r = c.post("/api/gallery/recategorize", json={"path": evil, "target": "Keep"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_get_root_still_200(client):
    """The module stays importable and GET / renders (smoke)."""
    c, dash, _ = client
    r = c.get("/")
    assert r.status_code == 200


# ── C. gallery-dl test-url: cookies_file reads an arbitrary local path ────────
# (CodeQL py/path-injection) so the route must be loopback-only, mirroring the
# cookies-converter and preset-publish routes.

def test_gallery_dl_test_url_rejects_non_loopback(client):
    c, _dash, _ = client
    r = c.post(
        "/api/scrapers/gallery-dl/test-url",
        json={"url": "https://example.com/x", "cookies_file": "C:/whatever.txt"},
        environ_base={"REMOTE_ADDR": "10.0.0.5"},
    )
    assert r.status_code == 403
    assert r.get_json()["ok"] is False


def test_gallery_dl_test_url_allows_loopback(client):
    c, _dash, _ = client
    r = c.post(
        "/api/scrapers/gallery-dl/test-url",
        json={"url": "not a url"},
    )
    # Loopback is let through the gate; the malformed URL fails downstream
    # instead of a 403 from the guard.
    assert r.status_code != 403
