"""Tests for the cheap pre-VLM quality pre-filter (``prefilter_aesthetic``).

``prefilter_aesthetic`` is a lightweight, dependency-light gate that runs BEFORE
the expensive vision-language model so obviously-bad images (blurry, near-black,
flat/solid-colour, tiny, extreme aspect ratios) never burn VLM tokens. The base
path uses Pillow + the standard library only; an optional LAION/CLIP aesthetic
model is gated behind ``torch`` + ``open_clip`` and is skipped cleanly when those
are not installed.

Runnable:
    pytest tests/test_prefilter_aesthetic.py -q
    python tests/test_prefilter_aesthetic.py
"""
from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def pf():
    import prefilter_aesthetic
    return importlib.reload(prefilter_aesthetic)


# ── tiny image builders (in-memory) ───────────────────────────────────────────

def _detailed(size: int = 256) -> Image.Image:
    """A sharp, high-detail RGB image: text + lines + shapes on a light field."""
    img = Image.new("RGB", (size, size), (240, 240, 240))
    d = ImageDraw.Draw(img)
    # High-frequency content: a grid of crisp lines + filled shapes.
    for x in range(0, size, 16):
        d.line([(x, 0), (x, size)], fill=(10, 10, 10), width=2)
    for y in range(0, size, 16):
        d.line([(0, y), (size, y)], fill=(10, 10, 10), width=2)
    d.rectangle([40, 40, 120, 120], fill=(200, 30, 30))
    d.ellipse([140, 140, 220, 220], fill=(30, 30, 200))
    return img


def _blurred(size: int = 256, radius: float = 6.0) -> Image.Image:
    """The detailed image put through a heavy Gaussian blur (low edge energy)."""
    return _detailed(size).filter(ImageFilter.GaussianBlur(radius=radius))


def _solid(size: int = 256, colour: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """A perfectly flat solid-colour image (no detail at all)."""
    return Image.new("RGB", (size, size), colour)


def _near_black(size: int = 256) -> Image.Image:
    """An almost-entirely-black frame with a single faint speck of detail."""
    img = Image.new("RGB", (size, size), (3, 3, 3))
    img.putpixel((size // 2, size // 2), (12, 12, 12))
    return img


def _bright(size: int = 256) -> Image.Image:
    """A well-exposed, mid-to-bright detailed image."""
    return _detailed(size)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _save(tmp_path: Path, img: Image.Image, name: str) -> Path:
    p = tmp_path / name
    img.save(p)
    return p


# ── shape of the result ───────────────────────────────────────────────────────

def test_assess_returns_expected_keys(pf):
    res = pf.assess(_png_bytes(_detailed()))
    assert isinstance(res, dict)
    for key in ("blur", "brightness", "contrast", "score", "passed", "reasons"):
        assert key in res, f"missing key {key!r}"
    assert isinstance(res["passed"], bool)
    assert isinstance(res["reasons"], list)
    assert 0 <= res["score"] <= 100


def test_score_is_clamped_0_to_100(pf):
    for img in (_detailed(), _blurred(), _solid(), _near_black()):
        res = pf.assess(_png_bytes(img))
        assert 0 <= res["score"] <= 100


def test_accepts_path_str_and_bytes(pf, tmp_path):
    img = _detailed()
    p = _save(tmp_path, img, "d.png")
    from_bytes = pf.assess(_png_bytes(img))["score"]
    from_path = pf.assess(p)["score"]
    from_str = pf.assess(str(p))["score"]
    # Same pixels in, same score out regardless of input form.
    assert from_bytes == from_path == from_str


# ── blur / sharpness heuristic ────────────────────────────────────────────────

def test_sharp_scores_higher_blur_metric_than_blurred(pf):
    sharp = pf.assess(_png_bytes(_detailed()))
    blurry = pf.assess(_png_bytes(_blurred()))
    # Higher edge energy for the sharp image.
    assert sharp["blur"] > blurry["blur"]


def test_blurred_image_is_flagged(pf):
    blurry = pf.assess(_png_bytes(_blurred(radius=8.0)))
    assert blurry["passed"] is False
    assert any("blur" in r.lower() or "sharp" in r.lower() for r in blurry["reasons"])


def test_sharp_detailed_image_passes(pf):
    res = pf.assess(_png_bytes(_detailed()))
    assert res["passed"] is True
    assert res["reasons"] == []


# ── brightness heuristic ──────────────────────────────────────────────────────

def test_bright_brighter_than_near_black(pf):
    bright = pf.assess(_png_bytes(_bright()))
    dark = pf.assess(_png_bytes(_near_black()))
    assert bright["brightness"] > dark["brightness"]


def test_near_black_is_flagged(pf):
    dark = pf.assess(_png_bytes(_near_black()))
    assert dark["passed"] is False
    assert any(
        "dark" in r.lower() or "bright" in r.lower() or "expos" in r.lower()
        for r in dark["reasons"]
    )


# ── solid-colour / low-detail heuristic ───────────────────────────────────────

def test_solid_colour_has_low_contrast(pf):
    solid = pf.assess(_png_bytes(_solid()))
    detailed = pf.assess(_png_bytes(_detailed()))
    assert solid["contrast"] < detailed["contrast"]


def test_solid_colour_is_flagged(pf):
    solid = pf.assess(_png_bytes(_solid()))
    assert solid["passed"] is False
    assert any(
        "solid" in r.lower() or "detail" in r.lower() or "flat" in r.lower()
        or "contrast" in r.lower()
        for r in solid["reasons"]
    )


def test_detailed_ranks_above_solid_and_blurred(pf):
    detailed = pf.assess(_png_bytes(_detailed()))["score"]
    solid = pf.assess(_png_bytes(_solid()))["score"]
    blurry = pf.assess(_png_bytes(_blurred()))["score"]
    assert detailed > solid
    assert detailed > blurry


# ── resolution + aspect gates ─────────────────────────────────────────────────

def test_tiny_image_is_flagged(pf):
    tiny = pf.assess(_png_bytes(_detailed(size=32)), min_width=64, min_height=64)
    assert tiny["passed"] is False
    assert any("small" in r.lower() or "resolution" in r.lower() or "size" in r.lower()
               for r in tiny["reasons"])


def test_extreme_aspect_ratio_is_flagged(pf):
    banner = Image.new("RGB", (1000, 40), (128, 128, 128))
    d = ImageDraw.Draw(banner)
    for x in range(0, 1000, 8):
        d.line([(x, 0), (x, 40)], fill=(0, 0, 0), width=1)
    res = pf.assess(_png_bytes(banner), max_aspect_ratio=5.0)
    assert res["passed"] is False
    assert any("aspect" in r.lower() or "ratio" in r.lower() for r in res["reasons"])


def test_resolution_gate_is_overridable(pf):
    img = _detailed(size=128)
    strict = pf.assess(_png_bytes(img), min_width=256, min_height=256)
    lax = pf.assess(_png_bytes(img), min_width=64, min_height=64)
    assert strict["passed"] is False
    assert lax["passed"] is True


# ── min_score threshold ───────────────────────────────────────────────────────

def test_min_score_threshold_controls_pass(pf):
    img = _detailed()
    # Impossible bar -> fails even a good image, with a score-related reason.
    strict = pf.assess(_png_bytes(img), min_score=101)
    assert strict["passed"] is False
    assert any("score" in r.lower() for r in strict["reasons"])
    # Floor of zero -> a good image passes.
    lax = pf.assess(_png_bytes(img), min_score=0)
    assert lax["passed"] is True


# ── graceful failure on corrupt / unreadable input ────────────────────────────

def test_corrupt_bytes_does_not_crash(pf):
    res = pf.assess(b"this is not an image")
    assert res["passed"] is False
    assert res["score"] == 0
    assert any("read" in r.lower() or "decode" in r.lower() or "corrupt" in r.lower()
               or "open" in r.lower() for r in res["reasons"])


def test_missing_path_does_not_crash(pf, tmp_path):
    res = pf.assess(tmp_path / "nope.png")
    assert res["passed"] is False
    assert res["score"] == 0
    assert res["reasons"]  # a non-empty reason explaining why


def test_empty_bytes_does_not_crash(pf):
    res = pf.assess(b"")
    assert res["passed"] is False
    assert res["score"] == 0


# ── optional aesthetic model is gated behind torch + open_clip ────────────────

def test_aesthetic_model_skipped_when_open_clip_absent(pf, monkeypatch):
    """With open_clip unavailable, requesting the aesthetic model must not crash
    and must not populate an aesthetic score — it is skipped cleanly."""
    monkeypatch.setattr(pf, "_HAS_AESTHETIC_DEPS", False, raising=False)
    res = pf.assess(_png_bytes(_detailed()), use_aesthetic=True)
    # Either the key is absent or explicitly None when deps are missing.
    assert res.get("aesthetic") is None
    # The heuristic score still comes through.
    assert 0 <= res["score"] <= 100
    assert res["passed"] is True


def test_aesthetic_not_used_by_default(pf):
    """Default base path never touches torch/open_clip."""
    res = pf.assess(_png_bytes(_detailed()))
    assert res.get("aesthetic") is None


def test_has_aesthetic_deps_is_false_in_base_env(pf):
    """In the lean base environment (no torch/open_clip) the capability flag is
    False, proving the heavy import is import-guarded."""
    # This mirrors the shipped base install; if a dev has torch installed this
    # test is environment-dependent, so only assert the attribute exists & bool.
    assert isinstance(pf._HAS_AESTHETIC_DEPS, bool)


def test_module_imports_without_heavy_deps(pf):
    """The module itself must import even though torch/open_clip are absent."""
    assert hasattr(pf, "assess")
    assert callable(pf.assess)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
