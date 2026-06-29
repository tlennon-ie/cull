"""Last-mile dashboard-wiring tests (feat/roadmap-wave-3).

Covers the four surfaces wired in this slice on top of the existing dashboard:

  1. yt-dlp per-job editor — a preset carrying ``scrapers.yt_dlp`` round-trips
     through the GET→PUT preset endpoints (the backend validator accepts it).
  2. active_learning.record_correction on a queue re-label — the move branch of
     /api/queue/action feeds a correction into the learning log (mocked).
  3. HF upload off the request thread — POST returns a terminal result inline for
     a fast (mocked) push, and the status endpoint reports the job state; a slow
     push hands back a 202 + id.
  4. Embeddings "find similar" + status — the endpoints degrade gracefully when
     open_clip is unavailable and return neighbours (mocked present) otherwise.

All network / heavy work is mocked. The fixture mirrors
test_dashboard_integration.py: it sets PIPELINE_BASE_DIR first, then reloads the
storage modules + the dashboard so every bit of state lands under a temp dir.
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


# ── 1. yt-dlp per-job editor: preset GET→PUT round-trip ───────────────────────

def _put_preset_with_yt_dlp(c, name, yt):
    """GET the full preset cfg, inject scrapers.yt_dlp, PUT the whole thing.

    Mirrors the dashboard's preset editor, which auto-saves the complete cfg (a
    full, non-partial PUT) — so the validator's "categories is required" gate is
    satisfied.
    """
    cfg = c.get(f"/api/presets/{name}").get_json()["cfg"]
    cfg.setdefault("scrapers", {})["yt_dlp"] = yt
    return c.put(f"/api/presets/{name}", json={"cfg": cfg})


def test_preset_put_round_trips_yt_dlp_block(client):
    """A preset PUT carrying scrapers.yt_dlp validates and is echoed back."""
    c, dash, _ = client
    name = dash.job_config.default_preset_name()
    yt = {"enabled": True, "urls": ["https://www.youtube.com/watch?v=abc"],
          "limit": 50, "cookies": ""}
    r = _put_preset_with_yt_dlp(c, name, yt)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["success"] is True
    got = body["cfg"]["scrapers"]["yt_dlp"]
    assert got["enabled"] is True
    assert got["urls"] == ["https://www.youtube.com/watch?v=abc"]
    assert got["limit"] == 50

    # And a fresh GET still shows the persisted block.
    again = c.get(f"/api/presets/{name}").get_json()
    assert again["cfg"]["scrapers"]["yt_dlp"]["urls"] == [
        "https://www.youtube.com/watch?v=abc"]


def test_preset_put_rejects_unknown_yt_dlp_key(client):
    """An unknown yt_dlp key is a clean 400, not a 500."""
    c, dash, _ = client
    name = dash.job_config.default_preset_name()
    r = _put_preset_with_yt_dlp(c, name, {"bogus": 1})
    assert r.status_code == 400
    assert "yt_dlp" in (r.get_json().get("error") or "")


def test_preset_put_clamps_yt_dlp_limit(client):
    """yt_dlp.limit is clamped into the 1..5000 range by the validator."""
    c, dash, _ = client
    name = dash.job_config.default_preset_name()
    r = _put_preset_with_yt_dlp(c, name, {"limit": 999999})
    assert r.status_code == 200
    assert r.get_json()["cfg"]["scrapers"]["yt_dlp"]["limit"] == 5000


# ── 2. active_learning.record_correction on queue re-label ────────────────────

def _make_queue_image(dash, *, category: str, stem: str = "img1",
                      vision: dict | None = None):
    """Create a queue image (+ optional .vision.json) under PIPELINE_QUEUE/<category>."""
    img = dash.PIPELINE_QUEUE / category / f"{stem}.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n")
    if vision is not None:
        import json
        img.with_name(f"{stem}.vision.json").write_text(
            json.dumps(vision), encoding="utf-8")
    return img


def test_queue_move_records_correction(client, monkeypatch):
    """A successful re-label move calls active_learning.record_correction with the
    predicted category + scores mined from the sibling .vision.json."""
    c, dash, _ = client
    dash.job_config.create_job("Relabel", subject="x")
    dash.job_config.activate("relabel")

    img = _make_queue_image(
        dash, category="OffTopic", stem="cat42",
        vision={"category": "OffTopic", "OVR_Quality_Score": 71,
                "REL_Quality_Score": 40})

    captured = {}
    import active_learning

    def fake_record(slug, key, predicted, corrected, scores, **kw):
        captured.update(slug=slug, key=key, predicted=predicted,
                        corrected=corrected, scores=scores)

    monkeypatch.setattr(active_learning, "record_correction", fake_record)

    r = c.post("/api/queue/action",
               json={"path": str(img), "action": "move", "target": "Keep"})
    assert r.status_code == 200 and r.get_json()["success"] is True
    # The image actually moved.
    assert (dash.PIPELINE_QUEUE / "Keep" / "cat42.png").exists()
    # And the correction was recorded with the right fields.
    assert captured["slug"] == "relabel"
    assert captured["key"] == "cat42"
    assert captured["predicted"] == "OffTopic"
    assert captured["corrected"] == "Keep"
    assert captured["scores"]["OVR_Quality_Score"] == 71


def test_queue_move_correction_failure_never_breaks_move(client, monkeypatch):
    """If record_correction raises, the move still succeeds (best-effort)."""
    c, dash, _ = client
    dash.job_config.create_job("Relabel2", subject="x")
    dash.job_config.activate("relabel2")
    img = _make_queue_image(dash, category="OffTopic", stem="boom",
                            vision={"category": "OffTopic"})

    import active_learning

    def boom(*a, **k):
        raise RuntimeError("learning store on fire")

    monkeypatch.setattr(active_learning, "record_correction", boom)
    r = c.post("/api/queue/action",
               json={"path": str(img), "action": "move", "target": "Keep"})
    assert r.status_code == 200 and r.get_json()["success"] is True
    assert (dash.PIPELINE_QUEUE / "Keep" / "boom.png").exists()


def test_queue_move_without_vision_json_uses_folder_name(client, monkeypatch):
    """No sibling .vision.json → predicted falls back to the parent folder name."""
    c, dash, _ = client
    dash.job_config.create_job("Relabel3", subject="x")
    dash.job_config.activate("relabel3")
    img = _make_queue_image(dash, category="Borderline", stem="nojson")

    captured = {}
    import active_learning
    monkeypatch.setattr(active_learning, "record_correction",
                        lambda slug, key, predicted, corrected, scores, **k:
                        captured.update(predicted=predicted, corrected=corrected))
    r = c.post("/api/queue/action",
               json={"path": str(img), "action": "move", "target": "Keep"})
    assert r.status_code == 200
    assert captured["predicted"] == "Borderline"
    assert captured["corrected"] == "Keep"


def test_queue_delete_does_not_record_correction(client, monkeypatch):
    """A delete action must not touch the learning log."""
    c, dash, _ = client
    dash.job_config.create_job("Del", subject="x")
    dash.job_config.activate("del")
    img = _make_queue_image(dash, category="OffTopic", stem="gone",
                            vision={"category": "OffTopic"})

    import active_learning
    monkeypatch.setattr(active_learning, "record_correction",
                        lambda *a, **k: pytest.fail("delete must not record a correction"))
    r = c.post("/api/queue/action", json={"path": str(img), "action": "delete"})
    assert r.status_code == 200 and r.get_json()["success"] is True


# ── 3. HF upload off the request thread + status endpoint ─────────────────────

def test_hf_push_fast_returns_result_inline_and_status_done(client, monkeypatch):
    """A fast (mocked) push finishes inside the sync window: the POST returns the
    terminal result and the status endpoint then reports state=done."""
    c, dash, _ = client
    dash.job_config.create_job("Hf", subject="x")
    seen = {}

    def fake_push(slug, repo_id, private=True, include_video=False, categories=None):
        seen.update(slug=slug, repo_id=repo_id)
        return {"repo_id": repo_id, "uploaded": 7, "private": private}

    monkeypatch.setattr(dash.hf_export, "push_to_hf", fake_push)
    r = c.post("/api/jobs/hf/export/huggingface",
               json={"repo_id": "me/ds", "private": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True and body["data"]["uploaded"] == 7
    assert seen["slug"] == "hf" and seen["repo_id"] == "me/ds"


def test_hf_push_async_returns_202_and_status_tracks(client, monkeypatch):
    """A slow push hands back a 202 + id; the status endpoint reports running then
    done once the worker thread finishes."""
    import threading
    c, dash, _ = client
    dash.job_config.create_job("Hf", subject="x")

    # Make the sync wait tiny and the push slow so the POST returns 202.
    monkeypatch.setattr(dash, "_HF_SYNC_WAIT_SECONDS", 0.05)
    release = threading.Event()

    def slow_push(slug, repo_id, private=True, include_video=False, categories=None):
        release.wait(5)
        return {"repo_id": repo_id, "uploaded": 3}

    monkeypatch.setattr(dash.hf_export, "push_to_hf", slow_push)
    r = c.post("/api/jobs/hf/export/huggingface", json={"repo_id": "me/ds"})
    assert r.status_code == 202
    body = r.get_json()
    assert body["success"] is True
    job_id = body["data"]["id"]
    assert body["data"]["state"] == "running"

    # Status reports running while the worker is blocked.
    st = c.get(f"/api/jobs/hf/export/huggingface/status?id={job_id}").get_json()
    assert st["data"]["state"] == "running"

    # Let the worker finish, join it, then the status flips to done.
    release.set()
    for t in threading.enumerate():
        if t.name.startswith("hf-push-"):
            t.join(5)
    st2 = c.get(f"/api/jobs/hf/export/huggingface/status?id={job_id}").get_json()
    assert st2["data"]["state"] == "done"
    assert st2["data"]["result"]["uploaded"] == 3


def test_hf_status_unknown_id_is_404(client):
    c, dash, _ = client
    dash.job_config.create_job("Hf", subject="x")
    r = c.get("/api/jobs/hf/export/huggingface/status?id=deadbeef")
    assert r.status_code == 404
    assert r.get_json()["success"] is False


def test_hf_push_missing_token_message_still_maps(client, monkeypatch):
    """A MissingCredentialError raised by a fast push still maps to a 400 + the
    HF_TOKEN hint (surfaced inline because the worker finished synchronously)."""
    c, dash, _ = client
    dash.job_config.create_job("Hf", subject="x")

    def boom(*a, **k):
        raise dash.credentials.MissingCredentialError("HF_TOKEN", scraper="hf_export")

    monkeypatch.setattr(dash.hf_export, "push_to_hf", boom)
    r = c.post("/api/jobs/hf/export/huggingface", json={"repo_id": "me/ds"})
    assert r.status_code == 400
    assert "HF_TOKEN" in r.get_json()["error"]


def test_hf_push_rejects_bad_repo_id_without_threading(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Hf", subject="x")
    monkeypatch.setattr(dash.hf_export, "push_to_hf",
                        lambda *a, **k: pytest.fail("must not push a bad repo_id"))
    r = c.post("/api/jobs/hf/export/huggingface", json={"repo_id": "no-slash"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


# ── 4. Embeddings status + find-similar ───────────────────────────────────────

def test_embeddings_status_unavailable_is_friendly(client, monkeypatch):
    """When open_clip is unavailable the status endpoint reports available:false
    with a hint, never a 500."""
    c, dash, _ = client
    dash.job_config.create_job("Emb", subject="x")
    import embeddings
    monkeypatch.setattr(embeddings, "open_clip_available", lambda: False)
    body = c.get("/api/embeddings/status?job=emb").get_json()
    assert body["success"] is True
    assert body["data"]["available"] is False
    assert "ml" in body["data"]["hint"].lower()


def test_embeddings_status_available_reports_indexed_count(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Emb", subject="x")
    import embeddings
    import embeddings_index
    monkeypatch.setattr(embeddings, "open_clip_available", lambda: True)
    idx = embeddings_index.EmbeddingIndex(slug="emb")
    idx.add("/some/key.png", [0.1, 0.2, 0.3])
    idx.flush()
    body = c.get("/api/embeddings/status?job=emb").get_json()
    assert body["success"] is True
    assert body["data"]["available"] is True
    assert body["data"]["indexed"] == 1


def test_embeddings_similar_returns_neighbours(client, monkeypatch):
    """With open_clip mocked present and a populated index, /similar returns
    safe_inside-guarded neighbours with thumb_url + score."""
    c, dash, _ = client
    dash.job_config.create_job("Emb", subject="x")
    import embeddings
    import embeddings_index
    monkeypatch.setattr(embeddings, "open_clip_available", lambda: True)

    # Two real paths inside the sorted root + one outside (must be dropped).
    inside = dash.PIPELINE_SORTED / "Keep" / "civitai" / "a.png"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_bytes(b"x")
    other = dash.PIPELINE_SORTED / "Keep" / "civitai" / "b.png"
    other.write_bytes(b"x")
    outside = "/etc/passwd"

    idx = embeddings_index.EmbeddingIndex(slug="emb")
    idx.add(str(inside), [1.0, 0.0, 0.0])
    idx.add(str(other), [0.9, 0.1, 0.0])
    idx.add(outside, [0.0, 1.0, 0.0])
    idx.flush()

    body = c.get(
        f"/api/embeddings/similar?job=emb&key={str(inside)}&top_k=12").get_json()
    assert body["success"] is True
    results = body["data"]["results"]
    paths = {m["path"] for m in results}
    assert str(other) in paths            # a real neighbour
    assert str(inside) not in paths       # the query image itself is excluded
    assert outside not in paths           # traversal-guarded out
    assert all(m["thumb_url"].startswith("/api/thumbnail?path=") for m in results)
    assert all("score" in m for m in results)


def test_embeddings_similar_unavailable_is_friendly(client, monkeypatch):
    """No ML extra → /similar returns an empty result set with a message, not 500."""
    c, dash, _ = client
    dash.job_config.create_job("Emb", subject="x")
    import embeddings
    monkeypatch.setattr(embeddings, "open_clip_available", lambda: False)
    body = c.get("/api/embeddings/similar?job=emb&key=/whatever.png").get_json()
    assert body["success"] is True
    assert body["data"]["results"] == []
    assert body["data"]["message"]


def test_embeddings_similar_requires_key(client):
    c, dash, _ = client
    dash.job_config.create_job("Emb", subject="x")
    r = c.get("/api/embeddings/similar?job=emb")
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_embeddings_similar_unknown_key_is_friendly(client, monkeypatch):
    """A key that isn't in the index returns an empty, friendly envelope."""
    c, dash, _ = client
    dash.job_config.create_job("Emb", subject="x")
    import embeddings
    import embeddings_index
    monkeypatch.setattr(embeddings, "open_clip_available", lambda: True)
    idx = embeddings_index.EmbeddingIndex(slug="emb")
    idx.add("/indexed/a.png", [1.0, 0.0])
    idx.flush()
    body = c.get(
        "/api/embeddings/similar?job=emb&key=/not/indexed.png").get_json()
    assert body["success"] is True
    assert body["data"]["results"] == []
    assert body["data"]["message"]
