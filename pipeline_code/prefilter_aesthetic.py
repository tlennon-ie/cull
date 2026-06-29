"""Cheap, dependency-light quality pre-filter that runs *before* the VLM.

The expensive part of cull is the vision-language model call. Most rejected
images are rejected for cheap, obvious reasons — they're blurry, near-black,
flat solid-colour, tiny, or an extreme banner aspect ratio. Spending VLM tokens
to discover that is wasteful. This module is a fast gate that scores each image
0-100 on a handful of classic image-quality heuristics computed with **Pillow +
the standard library only** (no torch / numpy in the base path), so only the
survivors hit the model.

Public surface
--------------
    assess(image, *, min_score=..., **opts) -> dict

The returned dict always has::

    {
      "blur":       float,   # edge-energy metric (variance of a Laplacian); higher = sharper
      "brightness": float,   # mean luminance, 0-255
      "contrast":   float,   # luminance std-dev, 0-255
      "aesthetic":  float | None,   # optional LAION/CLIP score, or None if unused/unavailable
      "score":      int,     # combined quality score, 0-100
      "passed":     bool,    # score >= min_score AND no hard-gate failure
      "reasons":    [str],   # human-readable failure reasons (empty when passed)
    }

Design notes
------------
* **Graceful failure.** Unreadable / corrupt / missing inputs return
  ``passed=False, score=0`` with an explanatory reason — never an exception. A
  single bad file must never crash a pre-worker stage or a dashboard request.
* **Blur metric = variance of a Laplacian.** The classic "is it sharp?" measure.
  We convolve the grayscale image with a 3x3 Laplacian kernel via
  ``ImageFilter.Kernel`` and take the variance of the response: a sharp image
  has lots of high-frequency edge energy (high variance); a blurred or flat one
  has little. This is the Pillow-only analogue of OpenCV's
  ``cv2.Laplacian(...).var()``.
* **Hard gates vs soft score.** Resolution, aspect ratio, blur, brightness and
  detail each contribute a soft sub-score AND can trip a hard gate (with a
  reason). ``passed`` requires both: no hard gate tripped *and* the combined
  score clears ``min_score``. This lets callers tune behaviour purely through
  ``min_score`` while still surfacing *why* something failed.
* **Optional aesthetic model.** A LAION-aesthetic / CLIP-aesthetic predictor is
  far better at "is this a *good* image" than any heuristic, but it pulls in
  ``torch`` + ``open_clip`` (hundreds of MB). It is import-guarded behind
  ``_HAS_AESTHETIC_DEPS`` and only consulted when the deps are installed AND the
  caller passes ``use_aesthetic=True``. Absent deps → the field is ``None`` and
  the heuristic score stands; nothing crashes.
"""
from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

try:  # library context — log when available, no-op otherwise.
    from pipeline_logging import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover - logging is best-effort here
    import logging

    logger = logging.getLogger(__name__)

ImageInput = "Path | str | bytes | bytearray"

# ── default thresholds (all overridable per-call via **opts) ──────────────────
DEFAULT_MIN_SCORE: int = 35
DEFAULT_MIN_WIDTH: int = 256
DEFAULT_MIN_HEIGHT: int = 256
DEFAULT_MAX_ASPECT_RATIO: float = 4.0
# Variance-of-Laplacian below this reads as "blurry / out of focus". Calibrated
# for the unnormalised 8-neighbour Laplacian kernel below: a sharp, detailed
# image lands in the thousands; a heavily Gaussian-blurred one in the low
# hundreds. 1500 cleanly separates the two without catching busy-but-soft art.
DEFAULT_BLUR_MIN: float = 1500.0
# Mean luminance gates (0-255): too dark or blown-out is unusable.
DEFAULT_BRIGHTNESS_MIN: float = 12.0
DEFAULT_BRIGHTNESS_MAX: float = 248.0
# Luminance std-dev below this reads as "near solid colour / no detail".
DEFAULT_CONTRAST_MIN: float = 8.0

# Reference points used to normalise raw metrics into 0-100 sub-scores. The
# blur reference matches the unnormalised Laplacian kernel's scale (sharp images
# reach the thousands); contrast saturates at a healthy luminance spread.
_BLUR_REF: float = 6000.0
_CONTRAST_REF: float = 70.0
# Sub-score blend weights (sum to 1.0).
_W_BLUR: float = 0.45
_W_CONTRAST: float = 0.30
_W_BRIGHTNESS: float = 0.15
_W_RESOLUTION: float = 0.10

# 3x3 Laplacian (edge) kernel: centre 8, neighbours -1. Sums to zero, so flat
# regions respond ~0 and only edges light up.
_LAPLACIAN_KERNEL: tuple[int, ...] = (
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1,
)


# ── optional aesthetic model dependency probe (heavy; import-guarded) ─────────
def _probe_aesthetic_deps() -> bool:
    """True only if both torch and open_clip import. Never raises."""
    try:
        import importlib.util

        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("open_clip") is not None
        )
    except Exception:  # pragma: no cover - defensive
        return False


_HAS_AESTHETIC_DEPS: bool = _probe_aesthetic_deps()


# ── input helpers ─────────────────────────────────────────────────────────────
def _describe(image: "Path | str | bytes | bytearray") -> str:
    if isinstance(image, (bytes, bytearray)):
        return f"<{len(image)} bytes>"
    return str(image)


def _flat_pixels(img: Image.Image) -> list[int]:
    """Flat list of band-0 pixel values, future-proof across Pillow versions.

    Pillow 12 deprecates ``Image.getdata`` in favour of ``get_flattened_data``;
    older pins (the repo floor is Pillow>=11) only have ``getdata``. Prefer the
    new API when present and fall back otherwise.
    """
    getter = getattr(img, "get_flattened_data", None)
    if callable(getter):
        return list(getter())
    return list(img.getdata())


def _open_rgb(image: "Path | str | bytes | bytearray") -> Image.Image | None:
    """Open ``image`` as a loaded RGB PIL image, or ``None`` if unreadable.

    Accepts a filesystem path (``Path``/``str``) or raw image ``bytes``. Returns
    ``None`` for anything missing / corrupt / truncated / empty so callers never
    wrap this in their own try/except.
    """
    try:
        if isinstance(image, (bytes, bytearray)):
            if not image:
                return None
            img = Image.open(io.BytesIO(bytes(image)))
        else:
            img = Image.open(Path(image))
        img.load()
        return img.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        logger.debug("prefilter: could not open image (%s): %s", _describe(image), exc)
        return None


def _variance(values: list[int]) -> float:
    """Population variance of a list of ints (0.0 for empty / single value)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def _mean_std(values: list[int]) -> tuple[float, float]:
    """Mean and population std-dev of a list of ints."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


# ── metric computation ────────────────────────────────────────────────────────
def _laplacian_variance(gray: Image.Image) -> float:
    """Variance of the Laplacian response — the classic sharpness metric.

    Higher = more high-frequency edge energy = sharper. A blurred or flat image
    yields a low value. Downscaled to a bounded working size first so cost is
    independent of input resolution.
    """
    work = gray
    longest = max(work.size)
    cap = 512
    if longest > cap:
        scale = cap / longest
        work = work.resize(
            (max(1, int(work.width * scale)), max(1, int(work.height * scale))),
            Image.BILINEAR,
        )
    edges = work.filter(ImageFilter.Kernel((3, 3), _LAPLACIAN_KERNEL, scale=1))
    return _variance(_flat_pixels(edges))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _brightness_subscore(mean: float, lo: float, hi: float) -> float:
    """1.0 in the comfortable mid-range, tapering to 0 at the gate edges."""
    if mean <= lo or mean >= hi:
        return 0.0
    centre = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    # Triangular falloff from the centre; sqrt softens it so only extremes hurt.
    return _clamp01(math.sqrt(1.0 - abs(mean - centre) / half))


def _metrics(img: Image.Image) -> dict[str, float]:
    """Compute the raw heuristic metrics for an RGB image."""
    gray = img.convert("L")
    pixels = _flat_pixels(gray)
    mean, std = _mean_std(pixels)
    blur = _laplacian_variance(gray)
    return {"blur": blur, "brightness": mean, "contrast": std}


# ── optional aesthetic scorer (only reached when deps present + opted in) ─────
def _aesthetic_score(img: Image.Image) -> float | None:  # pragma: no cover
    """LAION/CLIP-style aesthetic score in [0, 100], or ``None`` on any failure.

    Import-guarded: only invoked when ``_HAS_AESTHETIC_DEPS`` is True. Kept tiny
    and defensive — if the model can't be built or run, we return ``None`` and
    the heuristic score stands. The heavy imports live *inside* the function so
    importing this module never drags in torch.
    """
    if not _HAS_AESTHETIC_DEPS:
        return None
    try:
        import open_clip  # type: ignore
        import torch  # type: ignore

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        with torch.no_grad():
            tensor = preprocess(img).unsqueeze(0)
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        # Without a trained aesthetic head we can't produce a calibrated MOS, so
        # map the feature-norm proxy into 0-100. Real deployments swap in a
        # LAION aesthetic predictor head here.
        proxy = float(feats.abs().mean().item())
        return _clamp01(proxy) * 100.0
    except Exception as exc:
        logger.debug("prefilter: aesthetic scoring skipped: %s", exc)
        return None


# ── scoring + gating ──────────────────────────────────────────────────────────
def _opt(opts: dict[str, Any], key: str, default: Any) -> Any:
    val = opts.get(key, default)
    return default if val is None else val


def _resolution_subscore(width: int, height: int, min_w: int, min_h: int) -> float:
    """Ramp from 0 at the minimum to 1.0 at ~2x the minimum on the short edge."""
    if min_w <= 0 or min_h <= 0:
        return 1.0
    wr = width / min_w
    hr = height / min_h
    return _clamp01((min(wr, hr) - 1.0))


def _build_score(metrics: dict[str, float], width: int, height: int,
                 opts: dict[str, Any]) -> tuple[int, dict[str, float]]:
    """Blend metrics into a 0-100 score; return (score, subscores)."""
    blur_sub = _clamp01(metrics["blur"] / _BLUR_REF)
    contrast_sub = _clamp01(metrics["contrast"] / _CONTRAST_REF)
    bright_sub = _brightness_subscore(
        metrics["brightness"],
        float(_opt(opts, "brightness_min", DEFAULT_BRIGHTNESS_MIN)),
        float(_opt(opts, "brightness_max", DEFAULT_BRIGHTNESS_MAX)),
    )
    res_sub = _resolution_subscore(
        width, height,
        int(_opt(opts, "min_width", DEFAULT_MIN_WIDTH)),
        int(_opt(opts, "min_height", DEFAULT_MIN_HEIGHT)),
    )
    blended = (
        _W_BLUR * blur_sub
        + _W_CONTRAST * contrast_sub
        + _W_BRIGHTNESS * bright_sub
        + _W_RESOLUTION * res_sub
    )
    score = int(round(_clamp01(blended) * 100))
    subs = {
        "blur": blur_sub, "contrast": contrast_sub,
        "brightness": bright_sub, "resolution": res_sub,
    }
    return score, subs


def _hard_gate_reasons(metrics: dict[str, float], width: int, height: int,
                       opts: dict[str, Any]) -> list[str]:
    """Collect human-readable reasons for any hard-gate failure."""
    reasons: list[str] = []

    min_w = int(_opt(opts, "min_width", DEFAULT_MIN_WIDTH))
    min_h = int(_opt(opts, "min_height", DEFAULT_MIN_HEIGHT))
    if width < min_w or height < min_h:
        reasons.append(
            f"resolution too small: {width}x{height} < {min_w}x{min_h}"
        )

    max_ar = float(_opt(opts, "max_aspect_ratio", DEFAULT_MAX_ASPECT_RATIO))
    if width > 0 and height > 0 and max_ar > 0:
        ar = max(width / height, height / width)
        if ar > max_ar:
            reasons.append(
                f"extreme aspect ratio: {ar:.2f} > {max_ar:.2f}"
            )

    blur_min = float(_opt(opts, "blur_min", DEFAULT_BLUR_MIN))
    if metrics["blur"] < blur_min:
        reasons.append(
            f"too blurry / low sharpness: laplacian-var {metrics['blur']:.1f} "
            f"< {blur_min:.1f}"
        )

    b_min = float(_opt(opts, "brightness_min", DEFAULT_BRIGHTNESS_MIN))
    b_max = float(_opt(opts, "brightness_max", DEFAULT_BRIGHTNESS_MAX))
    if metrics["brightness"] < b_min:
        reasons.append(
            f"too dark / underexposed: mean luminance {metrics['brightness']:.1f} "
            f"< {b_min:.1f}"
        )
    elif metrics["brightness"] > b_max:
        reasons.append(
            f"too bright / overexposed: mean luminance {metrics['brightness']:.1f} "
            f"> {b_max:.1f}"
        )

    c_min = float(_opt(opts, "contrast_min", DEFAULT_CONTRAST_MIN))
    if metrics["contrast"] < c_min:
        reasons.append(
            f"near solid-colour / low detail: contrast {metrics['contrast']:.1f} "
            f"< {c_min:.1f}"
        )

    return reasons


def assess(
    image: "Path | str | bytes | bytearray",
    *,
    min_score: int | None = None,
    use_aesthetic: bool = False,
    **opts: Any,
) -> dict[str, Any]:
    """Score an image's technical quality and decide whether it should reach the VLM.

    Parameters
    ----------
    image:
        Filesystem path (``Path``/``str``) or raw image ``bytes``.
    min_score:
        Pass threshold on the combined 0-100 score. Defaults to
        ``DEFAULT_MIN_SCORE``; override per call (e.g. from ``PREFILTER_MIN_SCORE``).
    use_aesthetic:
        When True *and* torch+open_clip are installed, also compute an optional
        LAION/CLIP aesthetic score. Absent deps → silently skipped (field None).
    **opts:
        Threshold overrides — ``min_width``, ``min_height``, ``max_aspect_ratio``,
        ``blur_min``, ``brightness_min``, ``brightness_max``, ``contrast_min``.

    Returns
    -------
    dict with keys ``blur, brightness, contrast, aesthetic, score, passed,
    reasons`` (see module docstring). Never raises: corrupt/unreadable input
    yields ``passed=False, score=0`` and an explanatory reason.
    """
    threshold = DEFAULT_MIN_SCORE if min_score is None else int(min_score)

    img = _open_rgb(image)
    if img is None:
        return {
            "blur": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "aesthetic": None,
            "score": 0,
            "passed": False,
            "reasons": ["could not open / decode image (corrupt, empty or missing)"],
        }

    try:
        width, height = img.size
        metrics = _metrics(img)
        score, _subs = _build_score(metrics, width, height, opts)
        reasons = _hard_gate_reasons(metrics, width, height, opts)

        aesthetic: float | None = None
        if use_aesthetic and _HAS_AESTHETIC_DEPS:
            aesthetic = _aesthetic_score(img)

        if score < threshold:
            reasons.append(
                f"quality score {score} below minimum {threshold}"
            )

        passed = not reasons
        return {
            "blur": round(metrics["blur"], 3),
            "brightness": round(metrics["brightness"], 3),
            "contrast": round(metrics["contrast"], 3),
            "aesthetic": aesthetic,
            "score": score,
            "passed": passed,
            "reasons": reasons,
        }
    finally:
        img.close()


def is_enabled() -> bool:
    """Whether the pre-filter stage is enabled via ``PREFILTER_ENABLED`` env.

    Off by default — adopting a new gate must be an explicit opt-in so existing
    pipelines don't silently start dropping images on upgrade.
    """
    raw = os.environ.get("PREFILTER_ENABLED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def env_min_score() -> int:
    """Read ``PREFILTER_MIN_SCORE`` (0-100), falling back to the default."""
    raw = os.environ.get("PREFILTER_MIN_SCORE", "").strip()
    if not raw:
        return DEFAULT_MIN_SCORE
    try:
        return max(0, min(100, int(float(raw))))
    except ValueError:
        logger.debug("prefilter: bad PREFILTER_MIN_SCORE %r; using default", raw)
        return DEFAULT_MIN_SCORE


__all__ = [
    "assess",
    "is_enabled",
    "env_min_score",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_MIN_WIDTH",
    "DEFAULT_MIN_HEIGHT",
    "DEFAULT_MAX_ASPECT_RATIO",
    "DEFAULT_BLUR_MIN",
    "DEFAULT_BRIGHTNESS_MIN",
    "DEFAULT_BRIGHTNESS_MAX",
    "DEFAULT_CONTRAST_MIN",
]
