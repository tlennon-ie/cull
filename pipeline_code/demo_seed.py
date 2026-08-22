"""Dashboard-callable demo seeder — creates a self-contained demo job.

This module wraps ``tools/seed_demo_data.py`` into an importable API the
dashboard can invoke to seed / un-seed a synthetic dataset without any network
access. Everything it produces lives under a single, well-known slug
(:data:`DEMO_SLUG`) and is gated by a marker file so ``unseed_demo`` cannot
delete anything it did not create.

Contract:
  - ``seed_demo`` is idempotent. Re-calling it without ``force`` returns the
    existing counts. ``force=True`` un-seeds first, then re-seeds.
  - ``unseed_demo`` refuses to delete anything unless the marker file is
    present. It only touches paths under :data:`DEMO_SLUG`.
  - Everything routes through :mod:`paths`, :mod:`job_config`, and
    :mod:`categories` — no absolute paths, no direct jobs-JSON writes, no
    inlined category names.

Public API:
    seed_demo(force=False) -> dict
    unseed_demo() -> dict
    demo_status() -> dict
"""
from __future__ import annotations

import io
import json
import pathlib
import random
import shutil
import time
from dataclasses import dataclass
from typing import Any

# Sibling modules use absolute imports (see job_config, dashboard_enhanced,
# hf_export). This file lives in the same package and follows the same rule.
import categories
import job_config
import paths
import vision_prompt
from pipeline_logging import get_logger

logger = get_logger(__name__)

# ── Constants (public contract) ──────────────────────────────────────────────

DEMO_SLUG: str = "demo_cull_dataset"
DEMO_MARKER_FILENAME: str = ".demo_marker"

_DEMO_JOB_NAME: str = "Demo cull dataset"
_DEMO_SUBJECT: str = "sample subject demo"
_DEMO_PRESET: str = "default"
_DEMO_SOURCE: str = "local"

_QUEUE_COUNT: int = 12
_SORTED_LAYOUT: tuple[tuple[str, int], ...] = (
    ("Keep", 4),
    ("Borderline", 2),
    ("OffTopic", 2),
)

_IMAGE_SIZE: tuple[int, int] = (512, 768)


# ── Marker file (safety gate for unseed) ─────────────────────────────────────

def _marker_path() -> pathlib.Path:
    """Return the marker file path under the jobs directory.

    Placed alongside the job JSON, not inside the queue/sorted trees, so it
    survives any queue-directory rewrite and lives next to the artefact it
    guards.
    """
    return paths.jobs_dir() / f"{DEMO_SLUG}_demo_marker"


def _write_marker(payload: dict[str, Any]) -> None:
    p = _marker_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)


def _marker_exists() -> bool:
    return _marker_path().is_file()


# ── Synthetic image generation ───────────────────────────────────────────────

@dataclass(frozen=True)
class _Palette:
    """A pair of RGB colours forming a linear vertical gradient."""

    top: tuple[int, int, int]
    bottom: tuple[int, int, int]


def _palette_for(index: int) -> _Palette:
    """Deterministic palette from an index — same index always yields same colours."""
    rng = random.Random(f"cull-demo-{index}")

    def _rgb() -> tuple[int, int, int]:
        return (rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220))

    top = _rgb()
    # Push the bottom colour away from top for visible contrast.
    bottom = tuple(
        max(0, min(255, c + rng.choice([-1, 1]) * rng.randint(60, 150)))
        for c in top
    )
    return _Palette(top=top, bottom=(bottom[0], bottom[1], bottom[2]))


def _render_gradient_png(index: int, label: str) -> bytes:
    """Return PNG bytes for a 512x768 gradient tile stamped with ``label``.

    Pillow is a hard dependency (already listed in requirements.txt via
    ``Pillow>=11.0``); we import inside the function so this module still
    imports cleanly during the CLAUDE.md smoke check even on a partial install.
    """
    from PIL import Image, ImageDraw, ImageFont  # local import — see docstring

    width, height = _IMAGE_SIZE
    palette = _palette_for(index)
    img = Image.new("RGB", (width, height), palette.top)

    # Vertical gradient — one horizontal band per row.
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(palette.top[0] * (1 - t) + palette.bottom[0] * t)
        g = int(palette.top[1] * (1 - t) + palette.bottom[1] * t)
        b = int(palette.top[2] * (1 - t) + palette.bottom[2] * t)
        for x in range(width):
            img.putpixel((x, y), (r, g, b))

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Compute label position via textbbox (Pillow 10+ removed textsize).
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:  # pragma: no cover — defensive for older Pillow
        text_w, text_h = (len(label) * 6, 12)

    # Drop-shadow so the label reads on either half of the gradient.
    tx = (width - text_w) // 2
    ty = (height - text_h) // 2
    draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
    draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _synthetic_prompt(index: int) -> str:
    return (
        f"a demo scene number {index}, {_DEMO_SUBJECT}, gradient background"
    )


def _synthetic_vision_meta(index: int, category: str) -> dict[str, Any]:
    """Build a valid ``.vision.json`` payload matching ``build_response_format``.

    Only fields required by the strict JSON schema are populated; ``caption``
    is intentionally an empty string (auto-caption disabled in the demo).
    """
    # Deterministic scores so screenshots look the same on every seed.
    rng = random.Random(f"cull-demo-scores-{category}-{index}")
    ovr = rng.randint(55, 92)
    rel = rng.randint(50, 90)
    return {
        "description": (
            f"Demo gradient tile #{index} rendered by demo_seed.py — "
            "no real classification was performed."
        ),
        "primary_subject": f"{_DEMO_SUBJECT} tile {index}",
        "is_screenshot": False,
        "is_composite_grid": False,
        "contains_text_overlay": True,
        "is_human_photograph": False,
        "art_medium": "illustration",
        "photorealistic_style": False,
        "has_ai_flaws": False,
        "woman_present": False,
        "nsfw": False,
        "OVR_Quality_Score": ovr,
        "REL_Quality_Score": rel,
        "quality_score": max(1, min(10, ovr // 10)),
        "category": category,
        "reason": f"Synthetic demo item routed to {category} bucket.",
        "caption": "",
    }


# ── Filesystem writers ───────────────────────────────────────────────────────

def _write_triple(base_dir: pathlib.Path, index: int, category: str, kind: str) -> None:
    """Write one ``<stem>.png`` + ``.txt`` + ``.vision.json`` triple.

    ``base_dir`` is the directory the triple lands in. ``kind`` is a short tag
    that disambiguates queue vs sorted names in logs.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{category.lower()}_{kind}_demo_{index:03d}_{int(time.time())}"
    img_path = base_dir / f"{stem}.png"
    txt_path = base_dir / f"{stem}.txt"
    json_path = base_dir / f"{stem}.vision.json"

    label = f"cull demo #{index}"
    img_path.write_bytes(_render_gradient_png(index, label))
    txt_path.write_text(_synthetic_prompt(index), encoding="utf-8")
    json_path.write_text(
        json.dumps(_synthetic_vision_meta(index, category), indent=2),
        encoding="utf-8",
    )


def _seed_queue(slug: str) -> int:
    """Populate ``data/queue/<slug>/<source>/`` with :data:`_QUEUE_COUNT` items."""
    queue_source_dir = paths.queue_dir(slug) / _DEMO_SOURCE
    if queue_source_dir.exists():
        shutil.rmtree(queue_source_dir)
    for i in range(1, _QUEUE_COUNT + 1):
        _write_triple(queue_source_dir, i, "Keep", kind="queue")
    return _QUEUE_COUNT


def _seed_sorted(slug: str) -> int:
    """Populate ``data/sorted/<slug>/<category>/<source>/`` per :data:`_SORTED_LAYOUT`.

    Categories are taken from ``categories.CATEGORIES`` when they match the
    layout labels; otherwise we fall back to the layout label itself so the
    demo still works on a taxonomy the labels aren't in.
    """
    sorted_root = paths.sorted_dir(slug)
    if sorted_root.exists():
        shutil.rmtree(sorted_root)

    active = set(categories.get_categories())
    total = 0
    idx = 1
    for label, count in _SORTED_LAYOUT:
        # Prefer an active-taxonomy category with a matching name; otherwise
        # fall back to the label. Never inline a category name outside
        # categories.py — this bridges the demo to whatever taxonomy is live.
        target_cat = label if label in active else next(iter(active), label)
        cat_dir = sorted_root / target_cat / _DEMO_SOURCE
        for _ in range(count):
            _write_triple(cat_dir, idx, target_cat, kind="sorted")
            idx += 1
            total += 1
    return total


def _count_queue(slug: str) -> int:
    root = paths.queue_dir(slug)
    if not root.is_dir():
        return 0
    # Only count image files, not sidecars.
    return sum(
        1 for p in root.rglob("*.png") if p.is_file()
    )


def _count_sorted(slug: str) -> int:
    root = paths.sorted_dir(slug)
    if not root.is_dir():
        return 0
    return sum(
        1 for p in root.rglob("*.png") if p.is_file()
    )


# ── Public API ──────────────────────────────────────────────────────────────

def demo_status() -> dict[str, Any]:
    """Return the demo job's on-disk state — read-only, safe to call anytime."""
    job = job_config.get_job(DEMO_SLUG)
    return {
        "exists": job is not None,
        "active": job_config.get_active_slug() == DEMO_SLUG,
        "queue_count": _count_queue(DEMO_SLUG),
        "sorted_count": _count_sorted(DEMO_SLUG),
    }


def seed_demo(force: bool = False) -> dict[str, Any]:
    """Idempotently seed the demo job.

    Args:
        force: When True, unseed first, then re-create everything.

    Returns:
        ``{slug, created, queue_count, sorted_count, job_active}``.
    """
    if force:
        unseed_demo()

    existing = job_config.get_job(DEMO_SLUG)
    if existing is not None and _marker_exists() and not force:
        status = demo_status()
        logger.info("demo_seed: already seeded (%d queue, %d sorted); no-op",
                    status["queue_count"], status["sorted_count"])
        return {
            "slug": DEMO_SLUG,
            "created": False,
            "queue_count": status["queue_count"],
            "sorted_count": status["sorted_count"],
            "job_active": status["active"],
        }

    if existing is None:
        # Route job creation exclusively through job_config primitives — never
        # write the jobs JSON by hand.
        job = job_config.create_job(
            _DEMO_JOB_NAME,
            subject=_DEMO_SUBJECT,
            preset=_DEMO_PRESET,
        )
        # ``create_job`` slugifies the name; confirm it matches our contract.
        if job.slug != DEMO_SLUG:
            # Unlikely (slugify("Demo cull dataset") == "demo_cull_dataset"),
            # but if a caller ever tweaks _DEMO_JOB_NAME we fail loud.
            raise RuntimeError(
                f"demo job slug drift: expected {DEMO_SLUG!r}, got {job.slug!r}"
            )
        created = True
    else:
        created = False

    # Activate so the dashboard's "current job" chip lands on the demo and
    # projected categories match what the sidecars claim.
    job_config.activate(DEMO_SLUG)

    queue_count = _seed_queue(DEMO_SLUG)
    sorted_count = _seed_sorted(DEMO_SLUG)

    _write_marker({
        "slug": DEMO_SLUG,
        "created_at": time.time(),
        "queue_count": queue_count,
        "sorted_count": sorted_count,
        "note": (
            "Managed by pipeline_code/demo_seed.py — delete via unseed_demo() "
            "only. This file is the safety gate that authorises removal."
        ),
    })

    logger.info(
        "demo_seed: seeded job %r (queue=%d, sorted=%d, created=%s)",
        DEMO_SLUG, queue_count, sorted_count, created,
    )
    return {
        "slug": DEMO_SLUG,
        "created": created,
        "queue_count": queue_count,
        "sorted_count": sorted_count,
        "job_active": job_config.get_active_slug() == DEMO_SLUG,
    }


def unseed_demo() -> dict[str, Any]:
    """Delete the demo dataset, but only if the marker file is present.

    Refuses to touch anything under a slug other than :data:`DEMO_SLUG`. Every
    filesystem path is derived from ``paths.*`` and validated by
    ``paths.validate_slug`` implicitly (all paths.* per-slug helpers reject
    unsafe slugs).
    """
    if not _marker_exists():
        return {"slug": DEMO_SLUG, "removed": False, "reason": "no marker; refusing"}

    # Sanity gate: paths.queue_dir / paths.sorted_dir will raise a ValueError
    # on a slug outside the safe charset, and DEMO_SLUG is a compile-time
    # constant of the safe form. Belt-and-braces: revalidate.
    paths.validate_slug(DEMO_SLUG)

    removed_any = False

    queue_root = paths.queue_dir(DEMO_SLUG)
    if queue_root.is_dir():
        shutil.rmtree(queue_root, ignore_errors=True)
        removed_any = True

    sorted_root = paths.sorted_dir(DEMO_SLUG)
    if sorted_root.is_dir():
        shutil.rmtree(sorted_root, ignore_errors=True)
        removed_any = True

    # If the demo happens to be active, clearing it prevents delete_job from
    # refusing. This is safe — we're only clearing the pointer for the demo,
    # and only when it points at the demo.
    if job_config.get_active_slug() == DEMO_SLUG:
        job_config.set_active(None)
        removed_any = True

    # Remove from queue index if present.
    idx = job_config.get_index()
    if DEMO_SLUG in idx.get("queue", []):
        try:
            job_config.dequeue(DEMO_SLUG)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("demo_seed: dequeue failed for %r: %s", DEMO_SLUG, exc)

    if job_config.get_job(DEMO_SLUG) is not None:
        try:
            job_config.delete_job(DEMO_SLUG)
            removed_any = True
        except ValueError as exc:
            logger.warning("demo_seed: delete_job refused for %r: %s", DEMO_SLUG, exc)

    marker = _marker_path()
    if marker.is_file():
        try:
            marker.unlink()
            removed_any = True
        except OSError as exc:  # pragma: no cover
            logger.warning("demo_seed: marker removal failed: %s", exc)

    return {
        "slug": DEMO_SLUG,
        "removed": removed_any,
        "reason": None if removed_any else "nothing to remove",
    }


# Touch vision_prompt at import time to make the schema dependency explicit
# (the .vision.json shape we write must satisfy build_response_format). This is
# a defensive assertion, not a runtime cost — a single dict build.
def _assert_schema_alignment() -> None:  # pragma: no cover — sanity only
    rf = vision_prompt.build_response_format()
    required = set(rf["json_schema"]["schema"]["required"])
    sample = _synthetic_vision_meta(1, next(iter(categories.get_categories()), "Keep"))
    missing = required - set(sample)
    if missing:
        raise RuntimeError(f"demo vision meta missing required schema keys: {sorted(missing)}")


_assert_schema_alignment()


__all__ = [
    "DEMO_SLUG",
    "DEMO_MARKER_FILENAME",
    "seed_demo",
    "unseed_demo",
    "demo_status",
    "_marker_path",
]
