"""Tests for ``export_profiles`` — dataset export profiles for cull.

``export_profiles.export_dataset(slug, profile, out_dir, **opts)`` reads the
sorted library at ``data/sorted/<slug>/<category>/<source>/`` and packs the
KEPT categories into a training-ready layout. Terminal buckets (DISCARD /
CORRUPT) must NEVER be exported, and the source tree must be left untouched
(export = copy/pack only).

These tests build a tiny fake sorted tree under a temp ``PIPELINE_BASE_DIR``
so they are fully hermetic — no network, no real ``.env``.

Runnable:
    pytest tests/test_export_profiles.py -q
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise ``dotenv.load_dotenv`` so a developer's real ``.env`` cannot
    leak into these hermetic tests."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


def _write_png(path: Path, size: tuple[int, int] = (64, 64), colour: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, format="PNG")


def _vision_record(category: str, caption: str = "") -> dict[str, Any]:
    return {
        "description": "a deterministic test image",
        "primary_subject": "a test subject",
        "category": category,
        "OVR_Quality_Score": 70,
        "REL_Quality_Score": 60,
        "quality_score": 7,
        "caption": caption,
    }


def _make_sample(
    sorted_root: Path,
    category: str,
    source: str,
    stem: str,
    *,
    caption: str,
    size: tuple[int, int] = (64, 64),
    colour: str = "red",
    with_vision: bool = True,
) -> Path:
    """Create one (image + .txt [+ .vision.json]) sample. Returns the image path."""
    folder = sorted_root / category / source
    img = folder / f"{stem}.png"
    _write_png(img, size=size, colour=colour)
    (folder / f"{stem}.txt").write_text(caption, encoding="utf-8")
    if with_vision:
        (folder / f"{stem}.vision.json").write_text(
            json.dumps(_vision_record(category, caption), ensure_ascii=False),
            encoding="utf-8",
        )
    return img


@pytest.fixture()
def fake_sorted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build a fake sorted tree under a temp PIPELINE_BASE_DIR.

    Two KEPT categories (Keep, Borderline) with 1-2 samples each, plus a
    terminal DISCARD bucket that must never be exported.
    """
    base = tmp_path / "data"
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(base))
    monkeypatch.setenv("PIPELINE_SLUG", "demo")

    slug = "demo"
    sorted_root = base / "sorted" / slug

    # KEPT: Keep (two samples, different aspect ratios for bucketing tests)
    _make_sample(sorted_root, "Keep", "civitai", "keep_a_1700000000_123",
                 caption="a red square", size=(64, 64), colour="red")
    _make_sample(sorted_root, "Keep", "x", "keep_b_1700000001_456",
                 caption="a tall blue rectangle", size=(64, 128), colour="blue")
    # KEPT: Borderline (one sample, no vision sidecar to test optional meta)
    _make_sample(sorted_root, "Borderline", "reddit", "border_c_1700000002_789",
                 caption="a green square", size=(64, 64), colour="green",
                 with_vision=False)
    # TERMINAL: must never be exported
    _make_sample(sorted_root, "DISCARD", "civitai", "discard_d_1700000003_111",
                 caption="garbage", size=(64, 64), colour="black")
    _make_sample(sorted_root, "CORRUPT", "x", "corrupt_e_1700000004_222",
                 caption="broken", size=(64, 64), colour="white")

    return {
        "slug": slug,
        "base": base,
        "sorted_root": sorted_root,
        "kept_categories": ("Borderline", "Keep"),
        "out_dir": tmp_path / "export",
    }


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    """Map every file under ``root`` to its bytes — used to assert immutability."""
    snap: dict[str, bytes] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            snap[str(p.relative_to(root))] = p.read_bytes()
    return snap


# ── Import smoke ─────────────────────────────────────────────────────────────

def test_module_imports() -> None:
    import export_profiles  # noqa: F401

    assert hasattr(export_profiles, "export_dataset")


# ── Sample iteration ─────────────────────────────────────────────────────────

def test_iter_samples_excludes_terminal(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    samples = list(export_profiles.iter_samples(fake_sorted["slug"]))
    # 3 KEPT samples (2 Keep + 1 Borderline); 0 terminal.
    assert len(samples) == 3
    cats = {s.category for s in samples}
    assert cats == {"Keep", "Borderline"}
    assert "DISCARD" not in cats
    assert "CORRUPT" not in cats


def test_iter_samples_is_deterministic(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    first = [str(s.image_path) for s in export_profiles.iter_samples(fake_sorted["slug"])]
    second = [str(s.image_path) for s in export_profiles.iter_samples(fake_sorted["slug"])]
    assert first == second
    assert first == sorted(first)


def test_iter_samples_reads_caption_and_meta(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    by_cat = {s.category: s for s in export_profiles.iter_samples(fake_sorted["slug"])}
    # Borderline sample had no vision sidecar -> meta is None but caption present.
    assert by_cat["Borderline"].caption == "a green square"
    assert by_cat["Borderline"].meta is None
    # Keep samples have a vision record.
    assert by_cat["Keep"].meta is not None
    assert by_cat["Keep"].meta.get("category") == "Keep"


def test_category_filter(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    only_keep = list(
        export_profiles.iter_samples(fake_sorted["slug"], categories=["Keep"])
    )
    assert len(only_keep) == 2
    assert all(s.category == "Keep" for s in only_keep)


def test_category_filter_rejects_terminal(fake_sorted: dict[str, Any]) -> None:
    """Even if a caller asks for DISCARD, terminal buckets are never exported."""
    import export_profiles

    samples = list(
        export_profiles.iter_samples(
            fake_sorted["slug"], categories=["Keep", "DISCARD", "CORRUPT"]
        )
    )
    assert {s.category for s in samples} == {"Keep"}


# ── kohya profile ────────────────────────────────────────────────────────────

def test_kohya_export_produces_image_txt_pairs(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    out = fake_sorted["out_dir"]
    summary = export_profiles.export_dataset(
        fake_sorted["slug"], "kohya", out, concept="subject",
    )
    assert summary["profile"] == "kohya"
    assert summary["sample_count"] == 3

    pngs = sorted(out.rglob("*.png"))
    txts = sorted(out.rglob("*.txt"))
    assert len(pngs) == 3
    assert len(txts) == 3
    # Every image has a sibling .txt of the same stem.
    for img in pngs:
        assert (img.parent / f"{img.stem}.txt").exists()


def test_kohya_respects_repeats(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    out = fake_sorted["out_dir"]
    export_profiles.export_dataset(
        fake_sorted["slug"], "kohya", out, repeats=10, concept="hero",
    )
    # kohya/diffusers repeat-count convention: "<repeats>_<concept>".
    repeat_dirs = [p for p in out.rglob("10_hero") if p.is_dir()]
    assert repeat_dirs, f"expected a 10_hero folder under {out}"
    pngs = sorted(out.rglob("*.png"))
    assert len(pngs) == 3
    for img in pngs:
        assert "10_hero" in img.parts


def test_kohya_trigger_word_injected(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    out = fake_sorted["out_dir"]
    export_profiles.export_dataset(
        fake_sorted["slug"], "kohya", out, trigger_word="mytoken",
    )
    txts = sorted(out.rglob("*.txt"))
    assert txts
    for txt in txts:
        body = txt.read_text(encoding="utf-8")
        assert "mytoken" in body
        # Trigger word is prepended.
        assert body.startswith("mytoken")


# ── webdataset profile ───────────────────────────────────────────────────────

def test_webdataset_produces_tar_with_expected_members(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    out = fake_sorted["out_dir"]
    summary = export_profiles.export_dataset(fake_sorted["slug"], "webdataset", out)
    assert summary["profile"] == "webdataset"

    tars = sorted(out.rglob("*.tar"))
    assert tars, "expected at least one .tar shard"
    # Default shard size is large -> one shard for 3 samples.
    assert (out / "shard-000000.tar").exists()

    members: list[str] = []
    for tar_path in tars:
        with tarfile.open(tar_path, mode="r") as tf:
            members.extend(tf.getnames())

    pngs = [m for m in members if m.endswith(".png")]
    txts = [m for m in members if m.endswith(".txt")]
    jsons = [m for m in members if m.endswith(".json")]
    assert len(pngs) == 3
    assert len(txts) == 3
    # Each sample shares a basename across its three members (webdataset rule).
    bases = {m.rsplit(".", 1)[0] for m in pngs}
    for base in bases:
        assert f"{base}.txt" in members
        assert f"{base}.json" in members
    assert len(jsons) == 3


def test_webdataset_shards_by_max_samples(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    out = fake_sorted["out_dir"]
    export_profiles.export_dataset(
        fake_sorted["slug"], "webdataset", out, max_samples_per_shard=1,
    )
    tars = sorted(out.rglob("shard-*.tar"))
    # 3 samples, 1 per shard -> 3 shards.
    assert len(tars) == 3
    assert (out / "shard-000000.tar").exists()
    assert (out / "shard-000001.tar").exists()
    assert (out / "shard-000002.tar").exists()


# ── folders profile ──────────────────────────────────────────────────────────

def test_folders_export_per_category(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    out = fake_sorted["out_dir"]
    export_profiles.export_dataset(fake_sorted["slug"], "folders", out)
    assert (out / "Keep").is_dir()
    assert (out / "Borderline").is_dir()
    # Terminal buckets never appear.
    assert not (out / "DISCARD").exists()
    assert not (out / "CORRUPT").exists()
    assert len(list((out / "Keep").rglob("*.png"))) == 2
    assert len(list((out / "Borderline").rglob("*.png"))) == 1


# ── Cross-cutting: train/val split ───────────────────────────────────────────

def test_train_val_split_partitions_deterministically(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    out_a = fake_sorted["out_dir"] / "a"
    out_b = fake_sorted["out_dir"] / "b"
    summary_a = export_profiles.export_dataset(
        fake_sorted["slug"], "folders", out_a, train_val_split=0.34,
    )
    summary_b = export_profiles.export_dataset(
        fake_sorted["slug"], "folders", out_b, train_val_split=0.34,
    )
    # train/ and val/ subdirs exist.
    assert (out_a / "train").is_dir()
    assert (out_a / "val").is_dir()
    # 3 samples, 0.34 val -> 1 val, 2 train.
    train_pngs = list((out_a / "train").rglob("*.png"))
    val_pngs = list((out_a / "val").rglob("*.png"))
    assert len(train_pngs) + len(val_pngs) == 3
    assert len(val_pngs) == 1
    assert len(train_pngs) == 2
    # Deterministic: the same image lands in val both times.
    val_a = sorted(p.name for p in (out_a / "val").rglob("*.png"))
    val_b = sorted(p.name for p in (out_b / "val").rglob("*.png"))
    assert val_a == val_b
    assert summary_a["splits"] == summary_b["splits"]


# ── Cross-cutting: resolution bucketing ──────────────────────────────────────

def test_resolution_bucketing_groups_by_dimensions(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    out = fake_sorted["out_dir"]
    summary = export_profiles.export_dataset(
        fake_sorted["slug"], "folders", out, resolution_bucketing=True,
    )
    # Two 64x64 squares + one 64x128 portrait -> two buckets.
    bucket_dirs = {p.name for p in out.rglob("*") if p.is_dir()
                   and "x" in p.name and p.name.split("x")[0].isdigit()}
    assert "64x64" in bucket_dirs
    assert "64x128" in bucket_dirs
    assert "buckets" in summary
    assert summary["buckets"].get("64x64") == 2
    assert summary["buckets"].get("64x128") == 1


# ── Cross-cutting: trigger word across profiles ──────────────────────────────

def test_trigger_word_appears_in_folders_export(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    out = fake_sorted["out_dir"]
    export_profiles.export_dataset(
        fake_sorted["slug"], "folders", out, trigger_word="zzk",
    )
    txts = sorted(out.rglob("*.txt"))
    assert txts
    assert all("zzk" in t.read_text(encoding="utf-8") for t in txts)


# ── Safety: terminal categories & source immutability ────────────────────────

def test_terminal_categories_never_exported_any_profile(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    for profile in ("kohya", "webdataset", "folders"):
        out = fake_sorted["out_dir"] / profile
        export_profiles.export_dataset(fake_sorted["slug"], profile, out)
        for tar_path in out.rglob("*.tar"):
            with tarfile.open(tar_path, mode="r") as tf:
                names = " ".join(tf.getnames()).lower()
                assert "discard" not in names
                assert "corrupt" not in names
        for txt in out.rglob("*.txt"):
            assert "garbage" not in txt.read_text(encoding="utf-8")
            assert "broken" not in txt.read_text(encoding="utf-8")


def test_source_files_untouched_after_export(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    sorted_root = fake_sorted["sorted_root"]
    before = _snapshot_tree(sorted_root)
    for profile in ("kohya", "webdataset", "folders"):
        export_profiles.export_dataset(
            fake_sorted["slug"], profile, fake_sorted["out_dir"] / profile,
            repeats=5, trigger_word="t", train_val_split=0.34,
        )
    after = _snapshot_tree(sorted_root)
    assert before == after, "export must not move, delete, or modify source files"


def test_unknown_profile_raises(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    with pytest.raises(ValueError):
        export_profiles.export_dataset(
            fake_sorted["slug"], "not_a_profile", fake_sorted["out_dir"],
        )


def test_summary_shape(fake_sorted: dict[str, Any]) -> None:
    import export_profiles

    summary = export_profiles.export_dataset(
        fake_sorted["slug"], "folders", fake_sorted["out_dir"],
    )
    assert summary["slug"] == "demo"
    assert summary["profile"] == "folders"
    assert summary["sample_count"] == 3
    assert "out_dir" in summary
    assert set(summary["categories"]) == {"Keep", "Borderline"}
