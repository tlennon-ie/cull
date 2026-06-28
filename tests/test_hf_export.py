"""tests/test_hf_export.py - pytest suite for pipeline_code/hf_export.py.

TDD: written before the implementation. ``huggingface_hub`` is mocked in every
test that touches the network path (a fake module is injected into
``sys.modules`` / ``HfApi`` is monkeypatched), so nothing here ever creates a
real repo or uploads a real byte. A tiny fake ``data/sorted/<slug>`` tree is
built under a temp ``PIPELINE_BASE_DIR``.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_hf_export.py -q

What we assert about hf_export:
  * a ``metadata.jsonl`` is generated with one row per kept sample, carrying the
    correct ``file_name`` + ``caption`` + ``category`` (+ ``score`` / ``source``
    when present);
  * terminal categories (DISCARD / CORRUPT) are NEVER exported;
  * ``create_repo`` + ``upload_folder`` are called with ``private=True`` and
    ``repo_type="dataset"``;
  * a missing token raises a clear ``MissingCredentialError`` (a ``SystemExit``);
  * the original source files in ``data/sorted`` are left untouched (the export
    copies into a temp staging dir, it never moves the originals);
  * videos are only staged when ``include_video=True``.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Make pipeline_code importable as a flat namespace (mirrors the other suites).
PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ---------------------------------------------------------------------------
# Fake huggingface_hub SDK — captures the kwargs handed to create_repo /
# upload_folder so we can assert on them without any network I/O.
# ---------------------------------------------------------------------------

class _FakeHfCapture:
    """Records every call made against the fake ``HfApi``."""

    def __init__(self) -> None:
        self.init_kwargs: dict[str, Any] | None = None
        self.create_repo_calls: list[dict[str, Any]] = []
        self.upload_folder_calls: list[dict[str, Any]] = []


def _install_fake_hf(monkeypatch: pytest.MonkeyPatch) -> _FakeHfCapture:
    """Inject a fake ``huggingface_hub`` module exposing a stub ``HfApi``."""
    capture = _FakeHfCapture()
    mod = types.ModuleType("huggingface_hub")

    class HfApi:  # noqa: N801 - mirror the SDK's class name
        def __init__(self, **kwargs: Any) -> None:
            capture.init_kwargs = kwargs

        def create_repo(self, repo_id: str, **kwargs: Any) -> Any:
            record = {"repo_id": repo_id, **kwargs}
            capture.create_repo_calls.append(record)
            return types.SimpleNamespace(repo_id=repo_id)

        def upload_folder(self, **kwargs: Any) -> Any:
            # Snapshot the staging dir AT CALL TIME — push_to_hf cleans the temp
            # dir up in a finally block, so the only correct place to observe
            # what was uploaded is here, inside the call.
            folder = Path(kwargs.get("folder_path", ""))
            meta = folder / "metadata.jsonl"
            kwargs = dict(kwargs)
            kwargs["_metadata_existed"] = meta.exists()
            kwargs["_metadata_text"] = (
                meta.read_text(encoding="utf-8") if meta.exists() else None
            )
            capture.upload_folder_calls.append(kwargs)
            return types.SimpleNamespace(commit_url="https://hf/fake/commit")

    mod.HfApi = HfApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return capture


def _import_fresh():
    """Import hf_export fresh so its gated SDK import re-binds to whatever is in
    ``sys.modules`` right now."""
    sys.modules.pop("hf_export", None)
    return importlib.import_module("hf_export")


# ---------------------------------------------------------------------------
# Fixtures: an isolated base dir + a fake sorted tree.
# ---------------------------------------------------------------------------

@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the whole pipeline at a temp base dir and a clean token env."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_SLUG", "default")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    # Neutralise dotenv so a developer's real .env can't leak a token.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    return tmp_path


def _write_sample(
    base: Path,
    slug: str,
    category: str,
    source: str,
    name: str,
    *,
    caption: str | None,
    vision: dict[str, Any] | None,
    ext: str = ".png",
    image_bytes: bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
) -> Path:
    """Drop one <image> (+ optional .txt / .vision.json) into a fake sorted
    tree at data/sorted/<slug>/<category>/<source>/ and return the image path."""
    folder = base / "sorted" / slug / category / source
    folder.mkdir(parents=True, exist_ok=True)
    img = folder / f"{name}{ext}"
    img.write_bytes(image_bytes)
    if caption is not None:
        (folder / f"{name}.txt").write_text(caption, encoding="utf-8")
    if vision is not None:
        (folder / f"{name}.vision.json").write_text(
            json.dumps(vision), encoding="utf-8"
        )
    return img


def _read_metadata(staging_dir: Path) -> list[dict[str, Any]]:
    """Parse the metadata.jsonl produced inside the staging dir."""
    meta_path = staging_dir / "metadata.jsonl"
    assert meta_path.exists(), "metadata.jsonl was not written"
    rows: list[dict[str, Any]] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# module import: must import even without huggingface_hub installed.
# ---------------------------------------------------------------------------

def test_module_imports_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """hf_export must import cleanly when huggingface_hub is absent (gated import)."""
    # Simulate the package being uninstalled.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    mod = _import_fresh()
    assert hasattr(mod, "push_to_hf")


def test_kept_categories_excludes_terminal(env) -> None:
    """The default category set never includes DISCARD / CORRUPT."""
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    assert "DISCARD" not in kept
    assert "CORRUPT" not in kept
    assert len(kept) >= 1  # at least one keep bucket exists in the default taxonomy


# ---------------------------------------------------------------------------
# Staging: metadata.jsonl content + terminal exclusion + file copying.
# ---------------------------------------------------------------------------

def test_metadata_rows_have_filename_caption_category(env) -> None:
    """Each kept sample produces a row with file_name + caption + category."""
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    cat = kept[0]

    _write_sample(
        env, "default", cat, "civitai", "keep_001",
        caption="a serene mountain lake at dawn",
        vision={"category": cat, "OVR_Quality_Score": 82, "REL_Quality_Score": 71},
    )

    staging = env / "stage"
    rows = mod._stage_dataset("default", staging, kept, include_video=False)

    assert len(rows) == 1
    row = rows[0]
    assert row["caption"] == "a serene mountain lake at dawn"
    assert row["category"] == cat
    # file_name is RELATIVE and points at a file that actually exists in staging.
    assert not Path(row["file_name"]).is_absolute()
    assert (staging / row["file_name"]).exists()
    assert row["file_name"].endswith(".png")
    # score is mined from the .vision.json (OVR is the topic-independent craft score).
    assert row["score"] == 82
    assert row["source"] == "civitai"

    # And the metadata.jsonl on disk matches the returned rows.
    disk_rows = _read_metadata(staging)
    assert disk_rows == rows


def test_terminal_categories_are_never_exported(env) -> None:
    """DISCARD / CORRUPT samples are skipped even though they exist on disk."""
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    cat = kept[0]

    _write_sample(
        env, "default", cat, "x", "good_1",
        caption="kept image", vision={"category": cat, "OVR_Quality_Score": 50},
    )
    _write_sample(
        env, "default", "DISCARD", "x", "bad_1",
        caption="discarded image", vision={"category": "DISCARD"},
    )
    _write_sample(
        env, "default", "CORRUPT", "x", "bad_2",
        caption="corrupt image", vision=None,
    )

    staging = env / "stage"
    rows = mod._stage_dataset("default", staging, kept, include_video=False)

    cats = {r["category"] for r in rows}
    assert cat in cats
    assert "DISCARD" not in cats
    assert "CORRUPT" not in cats
    assert len(rows) == 1


def test_missing_caption_yields_empty_string(env) -> None:
    """A sample with no sibling .txt still exports with an empty caption."""
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    cat = kept[0]

    _write_sample(
        env, "default", cat, "reddit", "nocap_1",
        caption=None, vision={"category": cat},
    )

    staging = env / "stage"
    rows = mod._stage_dataset("default", staging, kept, include_video=False)
    assert len(rows) == 1
    assert rows[0]["caption"] == ""


def test_videos_excluded_by_default_included_on_flag(env) -> None:
    """Video files only stage when include_video=True."""
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    cat = kept[0]

    _write_sample(
        env, "default", cat, "yt", "clip_1",
        caption="a neon city", vision={"category": cat},
        ext=".mp4", image_bytes=b"\x00" * 64,
    )

    staging_off = env / "stage_off"
    rows_off = mod._stage_dataset("default", staging_off, kept, include_video=False)
    assert rows_off == []  # no images, video excluded

    staging_on = env / "stage_on"
    rows_on = mod._stage_dataset("default", staging_on, kept, include_video=True)
    assert len(rows_on) == 1
    assert rows_on[0]["file_name"].endswith(".mp4")
    assert (staging_on / rows_on[0]["file_name"]).exists()


def test_source_files_are_untouched(env) -> None:
    """Staging COPIES — the originals under data/sorted must remain in place."""
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    cat = kept[0]

    img = _write_sample(
        env, "default", cat, "civitai", "orig_1",
        caption="original caption",
        vision={"category": cat, "OVR_Quality_Score": 90},
    )
    txt = img.parent / "orig_1.txt"
    vj = img.parent / "orig_1.vision.json"

    staging = env / "stage"
    mod._stage_dataset("default", staging, kept, include_video=False)

    # Every original sidecar still exists and is unchanged.
    assert img.exists()
    assert txt.exists() and txt.read_text(encoding="utf-8") == "original caption"
    assert vj.exists()


# ---------------------------------------------------------------------------
# Token resolution.
# ---------------------------------------------------------------------------

def test_missing_token_raises_clear_error(env) -> None:
    """No token anywhere => MissingCredentialError (a SystemExit subclass)."""
    from credentials import MissingCredentialError

    _install_fake_hf  # not installed here on purpose; token check happens first
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    _write_sample(
        env, "default", kept[0], "x", "k1",
        caption="c", vision={"category": kept[0]},
    )

    with pytest.raises(MissingCredentialError):
        mod.push_to_hf("default", "user/dataset", token=None)


def test_explicit_token_argument_is_used(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A token passed directly is preferred and handed to HfApi."""
    capture = _install_fake_hf(monkeypatch)
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    _write_sample(
        env, "default", kept[0], "x", "k1",
        caption="c", vision={"category": kept[0]},
    )

    mod.push_to_hf("default", "user/dataset", token="hf_explicit_token")

    # The token reached HfApi (either via constructor or the call kwargs).
    used = (capture.init_kwargs or {}).get("token")
    if used is None and capture.create_repo_calls:
        used = capture.create_repo_calls[0].get("token")
    if used is None and capture.upload_folder_calls:
        used = capture.upload_folder_calls[0].get("token")
    assert used == "hf_explicit_token"


def test_token_read_from_env(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """HF_TOKEN in the environment is picked up when no token arg is given."""
    capture = _install_fake_hf(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    _write_sample(
        env, "default", kept[0], "x", "k1",
        caption="c", vision={"category": kept[0]},
    )

    result = mod.push_to_hf("default", "user/dataset")
    assert result["uploaded"] >= 1


# ---------------------------------------------------------------------------
# Upload wiring: create_repo + upload_folder kwargs.
# ---------------------------------------------------------------------------

def test_push_creates_private_dataset_repo(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """create_repo is called with repo_type='dataset', private=True, exist_ok=True."""
    capture = _install_fake_hf(monkeypatch)
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    _write_sample(
        env, "default", kept[0], "civitai", "k1",
        caption="a caption", vision={"category": kept[0], "OVR_Quality_Score": 60},
    )

    mod.push_to_hf("default", "me/my-dataset", private=True, token="hf_x")

    assert len(capture.create_repo_calls) == 1
    cr = capture.create_repo_calls[0]
    assert cr["repo_id"] == "me/my-dataset"
    assert cr["repo_type"] == "dataset"
    assert cr["private"] is True
    assert cr["exist_ok"] is True


def test_push_uploads_folder_to_dataset(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """upload_folder targets the dataset repo and points at the staging dir."""
    capture = _install_fake_hf(monkeypatch)
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    _write_sample(
        env, "default", kept[0], "civitai", "k1",
        caption="a caption", vision={"category": kept[0]},
    )

    result = mod.push_to_hf("default", "me/my-dataset", token="hf_x")

    assert len(capture.upload_folder_calls) == 1
    uf = capture.upload_folder_calls[0]
    assert uf["repo_id"] == "me/my-dataset"
    assert uf["repo_type"] == "dataset"
    # metadata.jsonl existed in the staging dir at the moment upload_folder ran.
    assert uf["_metadata_existed"] is True
    assert uf["_metadata_text"] is not None
    first_row = json.loads(uf["_metadata_text"].splitlines()[0])
    assert first_row["caption"] == "a caption"
    # The result dict reports back the essentials.
    assert result["repo_id"] == "me/my-dataset"
    assert result["repo_type"] == "dataset"
    assert result["private"] is True
    assert result["uploaded"] == 1


def test_push_with_no_samples_raises(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pushing an empty curation set is an explicit error, not a silent no-op."""
    _install_fake_hf(monkeypatch)
    mod = _import_fresh()
    # No samples written at all.
    with pytest.raises(ValueError):
        mod.push_to_hf("default", "me/empty", token="hf_x")


def test_push_raises_when_sdk_absent(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """When huggingface_hub is not importable, push_to_hf fails with a clear
    error (after the token check) rather than an opaque NameError."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    _write_sample(
        env, "default", kept[0], "x", "k1",
        caption="c", vision={"category": kept[0]},
    )
    with pytest.raises(RuntimeError):
        mod.push_to_hf("default", "me/x", token="hf_x")


def test_category_filter_subset(env, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit categories= list restricts the export to that subset (still
    never terminal)."""
    mod = _import_fresh()
    kept = mod._kept_categories(None)
    if len(kept) < 2:
        pytest.skip("default taxonomy has <2 keep buckets")
    cat_a, cat_b = kept[0], kept[1]

    _write_sample(env, "default", cat_a, "x", "a1",
                  caption="a", vision={"category": cat_a})
    _write_sample(env, "default", cat_b, "x", "b1",
                  caption="b", vision={"category": cat_b})

    staging = env / "stage"
    rows = mod._stage_dataset(
        "default", staging, mod._kept_categories([cat_a]), include_video=False
    )
    cats = {r["category"] for r in rows}
    assert cats == {cat_a}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
