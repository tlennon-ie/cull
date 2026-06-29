"""Flask test-client tests for the dashboard's culling + caption-editing surface
(roadmap wave-3 §3a keyboard culling + §3d inline caption editor).

Covered here (the JS keyboard layer is browser-verified, not unit-tested):

  * POST /api/caption       — writes the sibling <stem>.txt, length-capped,
                              safe_inside-guarded (traversal / out-of-roots → 400)
  * GET  /api/caption       — returns the saved text, safe_inside-guarded
  * POST /api/gallery/recategorize — moves a SORTED image between category
                              folders, validates the target category, guards both
                              the source and the recomputed destination with
                              safe_inside, and returns {from_category,to_category}
                              so the client can build a reversible undo entry
  * the envelope shape ({success,data,error}) on all the new endpoints
  * GET / still renders 200 with the new culling/caption markup present

The fixture mirrors test_dashboard_integration.py: set PIPELINE_BASE_DIR first,
then reload the storage modules + the dashboard so all state lands under tmp_path
and the developer's real .env can't leak in.
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
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY", "HF_TOKEN", "HUGGINGFACE_TOKEN",
)
for _k in _DOTENV_SOURCED_KEYS:
    os.environ.pop(_k, None)


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


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_sorted_image(dash, category: str, source: str = "civitai", stem: str = "img") -> Path:
    """Create <sorted>/<category>/<source>/<stem>.png with a .txt + .vision.json sidecar."""
    folder = dash.PIPELINE_SORTED / category / source
    folder.mkdir(parents=True, exist_ok=True)
    img = folder / f"{stem}.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    img.with_suffix(".txt").write_text("original caption", encoding="utf-8")
    img.with_name(f"{stem}.vision.json").write_text('{"category": "%s"}' % category, encoding="utf-8")
    return img


# ── POST /api/caption ───────────────────────────────────────────────────────

def test_caption_post_writes_sibling_txt(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep")
    r = c.post("/api/caption", json={"path": str(img), "text": "a blonde woman, beach"})
    assert r.status_code == 200
    body = r.get_json()
    # Envelope shape.
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["bytes"] == len("a blonde woman, beach")
    # The .txt landed next to the image stem.
    txt = img.with_suffix(".txt")
    assert txt.read_text(encoding="utf-8") == "a blonde woman, beach"
    assert body["data"]["path"] == str(txt)


def test_caption_post_overwrites_existing(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep")
    assert img.with_suffix(".txt").read_text(encoding="utf-8") == "original caption"
    r = c.post("/api/caption", json={"path": str(img), "text": "new caption"})
    assert r.status_code == 200 and r.get_json()["success"] is True
    assert img.with_suffix(".txt").read_text(encoding="utf-8") == "new caption"


def test_caption_post_works_under_queue_root(client):
    c, dash, _ = client
    folder = dash.PIPELINE_QUEUE / "civitai"
    folder.mkdir(parents=True, exist_ok=True)
    img = folder / "q.png"
    img.write_bytes(b"x")
    r = c.post("/api/caption", json={"path": str(img), "text": "queued caption"})
    assert r.status_code == 200 and r.get_json()["success"] is True
    assert img.with_suffix(".txt").read_text(encoding="utf-8") == "queued caption"


def test_caption_post_rejects_traversal_path(client):
    c, dash, _ = client
    # A classic traversal that escapes the sorted root.
    evil = str(dash.PIPELINE_SORTED / ".." / ".." / ".." / "etc" / "passwd.png")
    r = c.post("/api/caption", json={"path": evil, "text": "pwned"})
    assert r.status_code == 400
    body = r.get_json()
    assert body["success"] is False
    assert body["data"] is None
    # Nothing was written outside the roots.
    assert not (dash.PIPELINE_SORTED.parent.parent / "etc" / "passwd.txt").exists()


def test_caption_post_rejects_path_outside_roots(client, tmp_path):
    c, dash, _ = client
    outside = tmp_path / "elsewhere" / "x.png"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"x")
    r = c.post("/api/caption", json={"path": str(outside), "text": "nope"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False
    # The sibling .txt must NOT have been created outside the pipeline roots.
    assert not outside.with_suffix(".txt").exists()


def test_caption_post_rejects_absolute_outside(client):
    c, dash, _ = client
    r = c.post("/api/caption", json={"path": "/etc/passwd", "text": "x"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_caption_post_requires_text(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep")
    r = c.post("/api/caption", json={"path": str(img)})
    assert r.status_code == 400
    body = r.get_json()
    assert body["success"] is False and body["data"] is None


def test_caption_post_rejects_non_string_text(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep")
    r = c.post("/api/caption", json={"path": str(img), "text": 1234})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_caption_post_length_capped(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep")
    huge = "x" * (dash._MAX_CAPTION_CHARS + 1)
    r = c.post("/api/caption", json={"path": str(img), "text": huge})
    assert r.status_code == 400
    body = r.get_json()
    assert body["success"] is False
    assert str(dash._MAX_CAPTION_CHARS) in body["error"]
    # The oversized write was rejected — original caption untouched.
    assert img.with_suffix(".txt").read_text(encoding="utf-8") == "original caption"


def test_caption_post_accepts_empty_string(client):
    """An empty caption is a valid clear, not a missing-field error."""
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep")
    r = c.post("/api/caption", json={"path": str(img), "text": ""})
    assert r.status_code == 200 and r.get_json()["success"] is True
    assert img.with_suffix(".txt").read_text(encoding="utf-8") == ""


# ── GET /api/caption ────────────────────────────────────────────────────────

def test_caption_get_returns_saved_text(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep")
    # Save via POST, then read back via GET.
    c.post("/api/caption", json={"path": str(img), "text": "round trip caption"})
    r = c.get("/api/caption", query_string={"path": str(img)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["text"] == "round trip caption"
    assert body["data"]["path"] == str(img.with_suffix(".txt"))


def test_caption_get_empty_when_no_txt(client):
    c, dash, _ = client
    folder = dash.PIPELINE_SORTED / "Keep" / "civitai"
    folder.mkdir(parents=True, exist_ok=True)
    img = folder / "no_caption.png"
    img.write_bytes(b"x")   # image present, but no .txt sidecar
    r = c.get("/api/caption", query_string={"path": str(img)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    assert body["data"]["text"] == ""


def test_caption_get_guards_traversal(client):
    c, dash, _ = client
    evil = str(dash.PIPELINE_SORTED / ".." / ".." / "secret.png")
    r = c.get("/api/caption", query_string={"path": evil})
    assert r.status_code == 400
    body = r.get_json()
    assert body["success"] is False and body["data"] is None


def test_caption_get_guards_outside_roots(client):
    c, dash, _ = client
    r = c.get("/api/caption", query_string={"path": "/etc/passwd"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


# ── POST /api/gallery/recategorize ──────────────────────────────────────────

def test_recategorize_moves_image_and_sidecars(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep", stem="r1")
    r = c.post("/api/gallery/recategorize", json={"path": str(img), "target": "OffTopic"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    assert body["error"] is None
    assert body["data"]["moved"] is True
    assert body["data"]["from_category"] == "Keep"
    assert body["data"]["to_category"] == "OffTopic"
    # Image + sidecars moved to the new category, same source sub-folder.
    new_img = dash.PIPELINE_SORTED / "OffTopic" / "civitai" / "r1.png"
    assert new_img.exists()
    assert new_img.with_suffix(".txt").exists()
    assert new_img.with_name("r1.vision.json").exists()
    assert not img.exists()
    # The returned path points at the new location (for the undo stack).
    assert body["data"]["path"] == str(new_img)


def test_recategorize_undo_round_trip(client):
    """Undo reverses a move by calling the same endpoint with from/to swapped."""
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep", stem="r2")
    moved = c.post("/api/gallery/recategorize",
                   json={"path": str(img), "target": "DISCARD"}).get_json()
    new_path = moved["data"]["path"]
    assert (dash.PIPELINE_SORTED / "DISCARD" / "civitai" / "r2.png").exists()
    # Reverse: move it back to the original category.
    back = c.post("/api/gallery/recategorize",
                  json={"path": new_path, "target": "Keep"}).get_json()
    assert back["success"] is True and back["data"]["moved"] is True
    assert (dash.PIPELINE_SORTED / "Keep" / "civitai" / "r2.png").exists()
    assert not (dash.PIPELINE_SORTED / "DISCARD" / "civitai" / "r2.png").exists()


def test_recategorize_same_category_is_noop(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep", stem="r3")
    r = c.post("/api/gallery/recategorize", json={"path": str(img), "target": "Keep"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    assert body["data"]["moved"] is False
    assert img.exists()   # untouched


def test_recategorize_rejects_unknown_category(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep", stem="r4")
    r = c.post("/api/gallery/recategorize",
               json={"path": str(img), "target": "TotallyBogusBucket"})
    assert r.status_code == 400
    body = r.get_json()
    assert body["success"] is False
    assert "TotallyBogusBucket" in body["error"]
    # No stray folder was created and the image stayed put.
    assert not (dash.PIPELINE_SORTED / "TotallyBogusBucket").exists()
    assert img.exists()


def test_recategorize_requires_target(client):
    c, dash, _ = client
    img = _make_sorted_image(dash, "Keep", stem="r5")
    r = c.post("/api/gallery/recategorize", json={"path": str(img)})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_recategorize_guards_traversal_source(client):
    c, dash, _ = client
    evil = str(dash.PIPELINE_SORTED / ".." / ".." / "x" / "y.png")
    r = c.post("/api/gallery/recategorize", json={"path": evil, "target": "Keep"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_recategorize_guards_path_outside_roots(client, tmp_path):
    c, dash, _ = client
    outside = tmp_path / "outside" / "z.png"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"x")
    r = c.post("/api/gallery/recategorize", json={"path": str(outside), "target": "Keep"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_recategorize_missing_file_is_400(client):
    c, dash, _ = client
    ghost = dash.PIPELINE_SORTED / "Keep" / "civitai" / "ghost.png"
    r = c.post("/api/gallery/recategorize", json={"path": str(ghost), "target": "OffTopic"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


# ── GET / still renders with the new markup ─────────────────────────────────

def test_index_renders_200_with_culling_markup(client):
    c, dash, _ = client
    r = c.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    # Keyboard-culling wiring is present.
    assert "onCullKeydown" in html
    assert "cullRecategorize" in html
    assert "data-cull-idx" in html
    assert "Keyboard culling" in html
    # Inline caption editor wiring is present.
    assert "saveCaption" in html
    assert "captionTokenCount" in html
    assert "+ trigger word" in html


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
