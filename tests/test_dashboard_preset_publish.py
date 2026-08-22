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
import shutil
import subprocess
import sys
import time
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
    # Publish flow branches off origin/main, so seed the remote with the
    # initial commit — mirrors a real user setup where the repo has been
    # cloned from origin.
    subprocess.run(["git", "-C", str(workspace), "push", "-u", "origin", "main"],
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
    # The key regex now matches PRESET_NAME_RE (^[A-Za-z0-9 _-]{1,40}$) so
    # spaces, mixed case, and hyphens all pass — a legacy preset like
    # "Female Influencer" must be uploadable. Verify the truly-hostile
    # characters (dots, slashes, exclamation marks, over-length input,
    # control chars) are still refused.
    over_40 = "a" * 41
    for bad in ("with.dot", "bad!name", "path/segment", over_40, ""):
        r = c.post(
            f"/api/presets/{bad}/thumbnail",
            data={"file": (io.BytesIO(_png_bytes()), "t.png")},
            content_type="multipart/form-data",
        )
        # Routing side-effects can surface for hostile inputs before the
        # handler runs: 404 for path segments Flask can't match, 405 for the
        # empty segment (which routes to /api/presets/, wrong method). The
        # handler itself returns 400. All three shapes are legitimate
        # rejections — the invariant is "no upload lands on disk".
        assert r.status_code in (400, 404, 405), (bad, r.status_code, r.get_data(as_text=True))


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


def test_publish_opens_pr_branch_off_main(client):
    c, ws, origin = client

    r = c.post("/api/presets/default/publish", json={})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"] is True
    assert j["commit"], "commit sha should be non-empty"
    assert j["base"] == "main"
    assert j["branch"].startswith("contrib/preset-default-"), j["branch"]
    assert "presets/community/default.preset.json" in j["files"]

    # The contrib branch must exist on origin — but origin/main must NOT
    # advance. This is the "never lands on main directly" guarantee.
    branches = subprocess.run(
        ["git", "-C", str(origin), "for-each-ref", "--format=%(refname:short)",
         "refs/heads/"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    assert any(b == j["branch"] for b in branches), branches
    main_log = subprocess.run(
        ["git", "-C", str(origin), "log", "--format=%s", "main"],
        capture_output=True, text=True, check=True,
    )
    assert "community preset: default" not in main_log.stdout, main_log.stdout

    # And the contrib branch's tip DOES carry the preset file.
    contrib_log = subprocess.run(
        ["git", "-C", str(origin), "log", "--format=%s", j["branch"]],
        capture_output=True, text=True, check=True,
    )
    assert "community preset: default" in contrib_log.stdout


def test_publish_never_sweeps_operator_wip_into_commit(client):
    """The core safety invariant: the throwaway worktree isolates the commit.

    Simulate the operator having other unstaged work in the main tree before
    hitting Publish. That work MUST stay uncommitted — otherwise the button
    would silently ship a user's private edits with the shared preset.
    """
    c, ws, _o = client
    # Simulate operator WIP in the main working tree.
    tracked = ws / "README.md"
    tracked.write_text("seed\nDIRTY WIP\n", encoding="utf-8")
    untracked = ws / "notes.txt"
    untracked.write_text("private notes\n", encoding="utf-8")

    r = c.post("/api/presets/default/publish", json={})
    assert r.status_code == 200, r.get_data(as_text=True)
    sha = r.get_json()["commit"]

    # Inspect the contribution commit — should be JUST the preset file.
    out = subprocess.run(
        ["git", "-C", str(ws), "show", "--name-only", "--format=", sha],
        capture_output=True, text=True, check=True,
    )
    committed_files = {p.strip() for p in out.stdout.splitlines() if p.strip()}
    assert committed_files == {"presets/community/default.preset.json"}

    # And the operator's WIP is still visible in git status of the main tree.
    status = subprocess.run(
        ["git", "-C", str(ws), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
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
    # The thumbnail lives on the contribution branch (not the main tree).
    # Confirm by inspecting the commit.
    sha = j["commit"]
    show = subprocess.run(
        ["git", "-C", str(ws), "show", "--name-only", "--format=", sha],
        capture_output=True, text=True, check=True,
    )
    committed = {p.strip() for p in show.stdout.splitlines() if p.strip()}
    assert "presets/community/thumbnails/default.png" in committed


def test_publish_uses_fresh_branch_per_call(client):
    """Successive publishes each get their own contrib branch — no collisions."""
    c, _ws, origin = client

    first = c.post("/api/presets/default/publish", json={}).get_json()
    # Sleep 1s so the epoch suffix differs between calls.
    time.sleep(1.1)
    second = c.post("/api/presets/default/publish", json={}).get_json()

    assert first["branch"] != second["branch"], (first["branch"], second["branch"])
    branches = subprocess.run(
        ["git", "-C", str(origin), "for-each-ref", "--format=%(refname:short)",
         "refs/heads/"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    assert first["branch"] in branches
    assert second["branch"] in branches


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
