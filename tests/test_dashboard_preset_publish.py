"""Tests for the preset thumbnail-upload and git-based publish endpoints.

Covers the six invariants the PR promised:

* Thumbnail POST is loopback-only.
* Thumbnail POST rejects non-image bodies (magic-byte check, not just mime).
* Thumbnail POST rejects >2 MB uploads.
* Thumbnail POST routes built-in preset keys to ``presets/thumbnails/`` and
  community preset keys to ``presets/community/thumbnails/`` — the location
  the git-publish flow ships from.
* Thumbnail DELETE removes the user upload but never touches shipped defaults.
* ``/api/presets/<key>/publish`` refuses to sweep the operator's unrelated
  working-tree edits into the publish commit (``git commit -o`` path-only) —
  the single hardest invariant of the whole flow.
"""
from __future__ import annotations

import importlib
import io
import os
import shutil
import subprocess
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


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A dashboard test client whose WORKSPACE_ROOT is a real git repo.

    The publish endpoint runs real ``git`` subprocesses against the workspace,
    so the fixture initialises a repo with a fake ``origin`` pointing at a
    bare repo under ``tmp_path/origin.git`` — that way ``git push`` succeeds
    end-to-end without touching the network.
    """
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")

    # Set up a bare origin.
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True,
                   capture_output=True)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True,
                   capture_output=True)
    # Configure the workspace with a committer identity (git commit fails
    # without one) and point origin at the bare repo.
    for args in (
        ["git", "-C", str(workspace), "config", "user.email", "test@example.com"],
        ["git", "-C", str(workspace), "config", "user.name", "Test User"],
        ["git", "-C", str(workspace), "config", "commit.gpgsign", "false"],
        ["git", "-C", str(workspace), "remote", "add", "origin", str(origin)],
    ):
        subprocess.run(args, check=True, capture_output=True)

    # Seed a first commit so HEAD exists and push can succeed on the first
    # publish. Also creates the layout the dashboard expects.
    (workspace / "presets").mkdir()
    (workspace / "presets" / "community").mkdir()
    (workspace / "presets" / "thumbnails").mkdir()
    (workspace / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(workspace), "add", "README.md"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(workspace), "commit", "-m", "seed"],
                   check=True, capture_output=True)
    # Make sure HEAD is called 'main' so the branch label in the response is stable.
    subprocess.run(["git", "-C", str(workspace), "branch", "-M", "main"],
                   check=True, capture_output=True)

    monkeypatch.setenv("PIPELINE_BASE_DIR", str(workspace / "data"))
    monkeypatch.setenv("PIPELINE_QUEUE", str(workspace / "data" / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(workspace / "data" / "sorted"))
    monkeypatch.setenv("LOG_DIR", str(workspace / "data" / "logs"))
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace))
    (workspace / ".env").write_text("BLUR_NSFW_THUMBS=true\n", encoding="utf-8")

    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", workspace / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)

    import paths as _paths
    importlib.reload(_paths)
    import job_config
    importlib.reload(job_config)
    import index_store
    importlib.reload(index_store)
    import thumb_cache
    importlib.reload(thumb_cache)
    import dashboard_enhanced
    dashboard = importlib.reload(dashboard_enhanced)
    dashboard.app.config.update(TESTING=True)
    return dashboard.app.test_client(), workspace, origin


# ── minimal image factories ─────────────────────────────────────────────────

def _png_bytes(size: int = 128) -> bytes:
    """A tiny valid PNG. Uses PIL to keep the fixture honest — matches the
    server's magic-byte check exactly.
    """
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, "PNG")
    body = buf.getvalue()
    if len(body) < size:
        body = body + b"\x00" * (size - len(body))
    return body


# ── thumbnail upload / delete ───────────────────────────────────────────────

def test_thumbnail_upload_rejects_non_loopback(client):
    c, _ws, _o = client
    r = c.post(
        "/api/presets/default/thumbnail",
        data={"file": (io.BytesIO(_png_bytes()), "t.png")},
        content_type="multipart/form-data",
        environ_base={"REMOTE_ADDR": "10.0.0.5"},
    )
    assert r.status_code == 403
    assert r.get_json()["ok"] is False


def test_thumbnail_upload_rejects_bad_key(client):
    c, _ws, _o = client
    # A key with a slash would let a hostile caller escape the thumbnail dir;
    # even after .lower() the regex rejects any non ``[a-z0-9_]`` character.
    # Flask's routing itself blocks / in a single path segment, but a dot
    # or a dash still reaches the regex — verify one of each is refused.
    for bad in ("with-dash", "with.dot", "bad!name"):
        r = c.post(
            f"/api/presets/{bad}/thumbnail",
            data={"file": (io.BytesIO(_png_bytes()), "t.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400, (bad, r.status_code, r.get_data(as_text=True))


def test_thumbnail_upload_rejects_non_image(client):
    c, _ws, _o = client
    r = c.post(
        "/api/presets/default/thumbnail",
        data={"file": (io.BytesIO(b"this is definitely not an image"), "t.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert "unsupported" in r.get_json()["error"].lower()


def test_thumbnail_upload_rejects_oversized_body(client):
    c, ws, _o = client
    # Compose a valid PNG then pad with a huge tail — the magic bytes pass,
    # size check fires. Simulate a 3 MB body by putting a real PNG head + 3 MB
    # of null tail. We keep this under the 2 MB cap on the server.
    head = _png_bytes()
    payload = head + b"\x00" * (3 * 1024 * 1024)
    r = c.post(
        "/api/presets/default/thumbnail",
        data={"file": (io.BytesIO(payload), "big.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert "exceeds" in r.get_json()["error"].lower()


def test_thumbnail_upload_builtin_routes_to_user_slot(client):
    c, ws, _o = client
    r = c.post(
        "/api/presets/default/thumbnail",
        data={"file": (io.BytesIO(_png_bytes()), "t.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"] is True
    # Built-in preset name → user override slot.
    assert (ws / "presets" / "thumbnails" / "default.png").is_file()
    assert not (ws / "presets" / "community" / "thumbnails" / "default.png").is_file()


def test_thumbnail_upload_community_routes_to_community_slot(client):
    c, ws, _o = client
    # Create a community preset file so the router picks the community slot.
    community_dir = ws / "presets" / "community"
    community_dir.mkdir(parents=True, exist_ok=True)
    (community_dir / "my_community.preset.json").write_text('{"cull_preset_version":1}', encoding="utf-8")

    r = c.post(
        "/api/presets/my_community/thumbnail",
        data={"file": (io.BytesIO(_png_bytes()), "t.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    assert (ws / "presets" / "community" / "thumbnails" / "my_community.png").is_file()


def test_thumbnail_delete_removes_user_upload(client):
    c, ws, _o = client
    c.post(
        "/api/presets/default/thumbnail",
        data={"file": (io.BytesIO(_png_bytes()), "t.png")},
        content_type="multipart/form-data",
    )
    assert (ws / "presets" / "thumbnails" / "default.png").is_file()

    r = c.delete("/api/presets/default/thumbnail")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert not (ws / "presets" / "thumbnails" / "default.png").is_file()


def test_thumbnail_status_reports_custom(client):
    c, _ws, _o = client
    before = c.get("/api/presets/default/thumbnail/status").get_json()
    assert before["ok"] and before["has_custom"] is False

    c.post(
        "/api/presets/default/thumbnail",
        data={"file": (io.BytesIO(_png_bytes()), "t.png")},
        content_type="multipart/form-data",
    )
    after = c.get("/api/presets/default/thumbnail/status").get_json()
    assert after["has_custom"] is True


# ── git-based publish ───────────────────────────────────────────────────────

def _read_head_sha(workspace: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def test_publish_rejects_non_loopback(client):
    c, _ws, _o = client
    r = c.post("/api/presets/default/publish", json={},
               environ_base={"REMOTE_ADDR": "10.0.0.5"})
    assert r.status_code == 403


def test_publish_rejects_missing_preset(client):
    c, _ws, _o = client
    r = c.post("/api/presets/does_not_exist/publish", json={})
    assert r.status_code == 404


def test_publish_commits_and_pushes_preset(client):
    c, ws, origin = client

    r = c.post("/api/presets/default/publish", json={})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"] is True
    assert j["commit"], "commit sha should be non-empty"
    assert "presets/community/default.preset.json" in j["files"]
    assert (ws / "presets" / "community" / "default.preset.json").is_file()

    # The commit must be visible on origin/main (the bare repo we pointed at).
    out = subprocess.run(
        ["git", "-C", str(origin), "log", "--format=%s", "main"],
        capture_output=True, text=True, check=True,
    )
    assert "community preset: default" in out.stdout


def test_publish_never_sweeps_operator_wip_into_commit(client):
    """The core safety invariant: git commit -o is path-scoped.

    Simulate the operator having other unstaged work in the tree before
    hitting Publish. That work MUST stay uncommitted — otherwise the button
    would silently ship a user's private edits with the shared preset.
    """
    c, ws, _o = client
    # Simulate operator WIP: an existing tracked file with a dirty edit
    # and a brand-new untracked file. Neither should land in the publish commit.
    tracked = ws / "README.md"
    tracked.write_text("seed\nDIRTY WIP\n", encoding="utf-8")
    untracked = ws / "notes.txt"
    untracked.write_text("private notes\n", encoding="utf-8")

    r = c.post("/api/presets/default/publish", json={})
    assert r.status_code == 200, r.get_data(as_text=True)
    sha = r.get_json()["commit"]

    # Inspect the commit's file list — should be JUST the preset file.
    out = subprocess.run(
        ["git", "-C", str(ws), "show", "--name-only", "--format=", sha],
        capture_output=True, text=True, check=True,
    )
    committed_files = {p.strip() for p in out.stdout.splitlines() if p.strip()}
    assert committed_files == {"presets/community/default.preset.json"}

    # And the operator's WIP is still visible in git status as unstaged.
    status = subprocess.run(
        ["git", "-C", str(ws), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    # ' M README.md' and '?? notes.txt' should both still be present.
    lines = set(status.stdout.splitlines())
    assert any("README.md" in ln for ln in lines), lines
    assert any("notes.txt" in ln for ln in lines), lines


def test_publish_ships_thumbnail_when_present(client):
    c, ws, origin = client
    # Upload a thumbnail first — publish should copy it into the community
    # slot and include it in the commit.
    c.post(
        "/api/presets/default/thumbnail",
        data={"file": (io.BytesIO(_png_bytes()), "t.png")},
        content_type="multipart/form-data",
    )
    r = c.post("/api/presets/default/publish", json={})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert "presets/community/thumbnails/default.png" in j["files"]
    assert (ws / "presets" / "community" / "thumbnails" / "default.png").is_file()


def test_publish_reports_missing_origin(client, tmp_path, monkeypatch):
    """If ``origin`` is unset, the endpoint returns a friendly error hint.

    The fixture always sets up an origin; we tear it down here to hit the
    branch that would matter for a fresh clone with no remote configured.
    """
    c, ws, _o = client
    subprocess.run(["git", "-C", str(ws), "remote", "remove", "origin"],
                   check=True, capture_output=True)

    r = c.post("/api/presets/default/publish", json={})
    assert r.status_code == 400
    j = r.get_json()
    assert j["ok"] is False
    assert "origin" in j["error"].lower()
    assert "hint" in j
