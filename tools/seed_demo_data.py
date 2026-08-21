"""seed_demo_data.py - Synthetic multi-job dataset for README screenshots.

Produces a self-contained, network-free demo dataset the dashboard can render
against so a fresh clone shows populated Jobs / Global Gallery / Global Stats
tabs without touching a real scraper. Ships two demo jobs so multi-job
surfaces (job grid, filter chips, source bars) look plausible:

  * ``car_ads``                       — general dataset (Keep/Borderline/OffTopic)
  * ``realistic_female_influencer``   — photoreal-portrait taxonomy
    (InstagramInfluencer/Professional/Amateur/Watermarked/Unknown/NSFW)

Everything routes through ``pipeline_code.job_config`` for job creation +
activation and ``pipeline_code.paths`` for filesystem layout, so the seeded
tree matches what the pipeline would produce on its own. Images are Pillow
gradients rendered from a deterministic seed — the same call always writes
the same bytes, no external URL fetched, no NSFW-looking content produced.

Usage:

    python tools/seed_demo_data.py             # seed both jobs (idempotent)
    python tools/seed_demo_data.py --reset     # wipe queue/sorted for demo
                                               # slugs, then re-seed
    python tools/seed_demo_data.py --dry-run   # print the plan, write nothing

A ``<slug>_demo_marker`` file lives under ``data/jobs/`` for each seeded job
so re-running the script without ``--reset`` is a no-op and ``--reset`` only
removes what the marker authorised.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CODE = REPO_ROOT / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

# Sibling imports are intentionally lazy — see the note in the file header.
# We only touch pipeline_code once we know Pillow is available.


# ── Demo dataset shape ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CategoryPlan:
    """How many items land in a sorted category, distributed across sources."""

    name: str
    count: int


@dataclass(frozen=True)
class _JobPlan:
    slug: str
    name: str
    subject: str
    preset: str
    palette_seed: str
    queue_sources: tuple[tuple[str, int], ...]
    sorted_layout: tuple[tuple[_CategoryPlan, tuple[str, ...]], ...]


# Source folder names below match the ones real scrapers write into.
# See ``feed_local_folder.SOURCE_NAME``, ``scraper_x`` (writes ``twitter_x``),
# ``scraper_gallery_dl.SOURCE_NAME`` (``gallery_dl``), and the Civitai +
# Discord scrapers for the origin of every literal here.

_CAR_SOURCES: tuple[str, ...] = (
    "civitai_civitai_com",
    "twitter_x",
    "gallery_dl",
    "discord_1",
    "local",
)

_INFLUENCER_SOURCES: tuple[str, ...] = (
    "civitai_civitai_com",
    "twitter_x",
    "gallery_dl",
    "discord_1",
    "local",
)


_CAR_JOB = _JobPlan(
    slug="car_ads",
    name="Car ads",
    subject="commercial automotive photography",
    preset="default",
    palette_seed="cull-demo-car-ads",
    queue_sources=(
        ("civitai_civitai_com", 4),
        ("twitter_x", 3),
        ("gallery_dl", 3),
        ("discord_1", 2),
        ("local", 2),
    ),
    sorted_layout=(
        (_CategoryPlan("Keep", 14), _CAR_SOURCES),
        (_CategoryPlan("Borderline", 8), _CAR_SOURCES),
        (_CategoryPlan("OffTopic", 5), _CAR_SOURCES),
        (_CategoryPlan("DISCARD", 6), _CAR_SOURCES),
    ),
)


_INFLUENCER_JOB = _JobPlan(
    slug="realistic_female_influencer",
    name="Realistic female influencer",
    subject="photorealistic female influencer portrait",
    preset="photoreal_portrait",
    palette_seed="cull-demo-influencer",
    queue_sources=(
        ("civitai_civitai_com", 5),
        ("twitter_x", 4),
        ("gallery_dl", 2),
        ("discord_1", 2),
        ("local", 2),
    ),
    sorted_layout=(
        (_CategoryPlan("InstagramInfluencer", 9), _INFLUENCER_SOURCES),
        (_CategoryPlan("Professional", 7), _INFLUENCER_SOURCES),
        (_CategoryPlan("Amateur", 5), _INFLUENCER_SOURCES),
        (_CategoryPlan("Watermarked", 4), _INFLUENCER_SOURCES),
        (_CategoryPlan("Unknown", 3), _INFLUENCER_SOURCES),
        # Keep the NSFW bucket small — the gradient tiles are abstract, but the
        # bucket still populates the source bar and the NSFW filter chip.
        (_CategoryPlan("NSFW", 2), _INFLUENCER_SOURCES),
        (_CategoryPlan("DISCARD", 4), _INFLUENCER_SOURCES),
    ),
)


_DEMO_JOBS: tuple[_JobPlan, ...] = (_CAR_JOB, _INFLUENCER_JOB)


# Small tiles keep the on-disk footprint tiny (< 50 KB / image) while still
# reading as populated in the gallery grid.
_IMAGE_SIZE: tuple[int, int] = (384, 384)

_MARKER_TEMPLATE: str = "{slug}_demo_marker"


# ── Prompts + category hints (surface as .txt sidecars) ──────────────────────


_CAR_PROMPTS: tuple[str, ...] = (
    "commercial photograph of a matte-black electric sedan on a wet street at night, "
    "cinematic rim light, shallow depth of field, 85mm",
    "top-down aerial of a silver coupe cornering on a mountain switchback, "
    "sunrise, long shadow, hero shot for automotive campaign",
    "studio product shot of a chrome-wheeled roadster on seamless white, "
    "soft key light from camera left, subtle floor reflection",
    "editorial magazine spread of a vintage roadster parked outside a Mid-Century "
    "modernist house, warm golden hour, kodachrome film simulation",
    "action shot of a compact hatchback drifting through mist, tail lights "
    "streaking, motion blur, low angle three-quarter view",
    "concept-car reveal on a rotating platform, hard rim lights, "
    "black backdrop, ISO 100 studio look",
    "documentary-style shot of a delivery van at dawn in an urban loading bay, "
    "practical lighting, natural colour",
)


_INFLUENCER_PROMPTS: tuple[str, ...] = (
    "editorial portrait of a woman on a Parisian balcony at golden hour, "
    "soft rim light, 85mm f/1.4, subtle grain, photorealistic",
    "candid coffee-shop lifestyle scene, natural window light, warm palette, "
    "shallow depth of field, minimal makeup, cinematic colour",
    "high-key studio beauty close-up on grey paper backdrop, "
    "clamshell lighting, photoreal skin texture, 100mm macro",
    "polished commercial fashion look, seamless white cyc, editorial styling, "
    "large softbox key, subtle vignette, magazine cover crop",
    "documentary street portrait, overcast sky, muted palette, "
    "35mm lens, subject centered, single-source natural light",
    "beach lifestyle shot, backlit golden hour, sand and ocean bokeh, "
    "photoreal skin, warm skin-tone grade",
    "urban rooftop portrait after sunset, ambient purple sky, "
    "practical string lights, natural film simulation",
)


# ── Gradient generator (self-contained; Pillow only) ─────────────────────────


def _rgb_pair(seed: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return ``(top, bottom)`` RGB tuples with visible contrast."""

    rng = random.Random(seed)

    def _clr() -> tuple[int, int, int]:
        return (rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220))

    top = _clr()
    push = rng.choice([-1, 1]) * rng.randint(70, 150)
    bottom = tuple(max(0, min(255, c + push)) for c in top)
    return top, (bottom[0], bottom[1], bottom[2])


def _render_gradient_jpeg(seed: str, label: str) -> bytes:
    """Vertical-gradient JPG with ``label`` painted centre-frame.

    Pillow is a hard dep of cull (``Pillow>=11.0`` in requirements.txt); we
    import lazily so the top-level import of this module is stdlib-only and
    the module clears any lightweight import-sanity check even before a
    partial install.
    """
    from PIL import Image, ImageDraw, ImageFont

    width, height = _IMAGE_SIZE
    top, bottom = _rgb_pair(seed)

    # Row-by-row lerp: cheap enough for 384x384 and yields a smooth gradient
    # without pulling numpy just for the seeder.
    img = Image.new("RGB", (width, height), top)
    for y in range(height):
        t = y / max(1, height - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        for x in range(width):
            img.putpixel((x, y), (r, g, b))

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:  # pragma: no cover — Pillow < 10 defensive branch
        text_w, text_h = (len(label) * 6, 12)
    tx = (width - text_w) // 2
    ty = (height - text_h) // 2
    draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
    draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    # quality 82 + optimize = readable in the gallery yet tiny on disk
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()


# ── Vision-json shape (aligns with build_response_format's required set) ─────


def _synthetic_vision_meta(
    idx: int,
    category: str,
    source: str,
    subject: str,
    is_terminal: bool = False,
) -> dict[str, Any]:
    """Return a payload that satisfies ``vision_prompt.build_response_format``.

    Score ranges are set so DISCARD reads low, Borderline / Watermarked land
    middling, and Keep-equivalent buckets score high — the resulting Global
    Stats donut has a plausible spread rather than a single flat colour.
    """
    rng = random.Random(f"cull-demo-scores-{category}-{source}-{idx}")

    if is_terminal:
        ovr = rng.randint(15, 42)
        rel = rng.randint(10, 35)
    elif category in {"Keep", "Professional", "InstagramInfluencer"}:
        ovr = rng.randint(72, 94)
        rel = rng.randint(70, 92)
    elif category in {"Borderline", "Watermarked", "Amateur"}:
        ovr = rng.randint(55, 74)
        rel = rng.randint(50, 72)
    else:  # OffTopic, Unknown, NSFW
        ovr = rng.randint(45, 68)
        rel = rng.randint(20, 58)

    quality = max(1, min(10, ovr // 10))
    contains_overlay = category == "Watermarked"
    is_nsfw = category == "NSFW"

    return {
        "description": (
            f"Demo gradient tile #{idx} routed to {category} — seed_demo_data.py "
            f"placeholder for the {subject} preview. No real classification."
        ),
        "primary_subject": f"{subject} demo tile {idx}",
        "is_screenshot": False,
        "is_composite_grid": False,
        "contains_text_overlay": contains_overlay,
        "is_human_photograph": False,
        "art_medium": "illustration",
        "photorealistic_style": False,
        "has_ai_flaws": is_terminal,
        "woman_present": False,
        "nsfw": is_nsfw,
        "OVR_Quality_Score": ovr,
        "REL_Quality_Score": rel,
        "quality_score": quality,
        "category": category,
        "reason": (
            f"Synthetic demo item generated by seed_demo_data.py; routed to "
            f"{category} bucket for a plausible dashboard preview."
        ),
        "caption": "",
    }


# ── Filesystem writers ───────────────────────────────────────────────────────


def _write_triple(
    base_dir: Path,
    idx: int,
    category: str,
    source: str,
    job: _JobPlan,
    kind: str,
    prompts: tuple[str, ...],
) -> None:
    """Write one ``<stem>.jpg`` + ``.txt`` + ``.vision.json`` triple."""
    import os as _os
    import time as _time

    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{category.lower()}_{source}_{kind}_{idx:04d}"
    img_path = base_dir / f"{stem}.jpg"
    txt_path = base_dir / f"{stem}.txt"
    json_path = base_dir / f"{stem}.vision.json"

    seed = f"{job.palette_seed}-{category}-{source}-{idx}"
    label = f"{job.slug}\n{category}\n#{idx:04d}"
    img_path.write_bytes(_render_gradient_jpeg(seed, label))
    txt_path.write_text(prompts[idx % len(prompts)], encoding="utf-8")
    payload = _synthetic_vision_meta(
        idx,
        category,
        source,
        job.subject,
        is_terminal=category in {"DISCARD", "CORRUPT"},
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Nudge the mtime so newest-first lists on the dashboard have some order
    # variety instead of clumping every file at now().
    stagger = idx * 5
    ts = _time.time() - stagger
    for p in (img_path, txt_path, json_path):
        try:
            _os.utime(p, (ts, ts))
        except OSError:
            pass


def _seed_queue(job: _JobPlan) -> int:
    """Populate ``data/queue/<slug>/<source>/`` per ``job.queue_sources``."""
    import paths

    prompts = _INFLUENCER_PROMPTS if job.slug == _INFLUENCER_JOB.slug else _CAR_PROMPTS
    queue_root = paths.queue_dir(job.slug)
    if queue_root.exists():
        shutil.rmtree(queue_root)

    total = 0
    idx = 1
    for source, count in job.queue_sources:
        target = queue_root / source
        for _ in range(count):
            _write_triple(target, idx, "Keep", source, job, "queue", prompts)
            idx += 1
            total += 1
    return total


def _seed_sorted(job: _JobPlan) -> int:
    """Populate ``data/sorted/<slug>/<category>/<source>/``."""
    import paths

    prompts = _INFLUENCER_PROMPTS if job.slug == _INFLUENCER_JOB.slug else _CAR_PROMPTS
    sorted_root = paths.sorted_dir(job.slug)
    if sorted_root.exists():
        shutil.rmtree(sorted_root)

    total = 0
    idx = 1
    for cat_plan, sources in job.sorted_layout:
        for i in range(cat_plan.count):
            # Fan the count out across sources so no bucket lands on a single
            # source (the "per-source DISCARD" bar needs at least two sources
            # per category to look real).
            source = sources[i % len(sources)]
            target = sorted_root / cat_plan.name / source
            _write_triple(target, idx, cat_plan.name, source, job, "sorted", prompts)
            idx += 1
            total += 1
    return total


def _seed_logs(job: _JobPlan, queue_ct: int, sorted_ct: int) -> None:
    """Write a small JSONL activity log for tail-log surfaces.

    The dashboard's Activity feed reads the SQLite index (not this file), but
    a plain jsonl next to the other per-slug artefacts gives forensic tools
    something to grep for the demo state and matches the task's spec.
    """
    import paths

    log_root = paths.log_dir()
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"pipeline_{job.slug}_demo.jsonl"
    now = time.time()
    lines: list[str] = []
    lines.append(json.dumps({
        "ts": now, "level": "info", "component": "demo",
        "msg": "seeded demo dataset",
        "slug": job.slug, "queue": queue_ct, "sorted": sorted_ct,
    }))
    for i in range(min(sorted_ct, 25)):
        lines.append(json.dumps({
            "ts": now - i * 4, "level": "info", "component": "vision",
            "slug": job.slug, "event": "classified",
            "category": "Keep" if i % 3 else "Borderline",
            "ovr": 82 - (i % 20), "rel": 78 - (i % 15),
        }))
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Marker file (safety gate for --reset) ────────────────────────────────────


def _marker_path(slug: str) -> Path:
    import paths

    return paths.jobs_dir() / _MARKER_TEMPLATE.format(slug=slug)


def _write_marker(slug: str, payload: dict[str, Any]) -> None:
    p = _marker_path(slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)


def _marker_exists(slug: str) -> bool:
    return _marker_path(slug).is_file()


# ── Public seeding + reset ───────────────────────────────────────────────────


def _ensure_job(job: _JobPlan) -> None:
    """Idempotently create the demo job through the real job_config API."""
    import job_config

    existing = job_config.get_job(job.slug)
    if existing is not None:
        # Bring subject / preset back to canonical values if a previous run
        # left them drifted; harmless when they already match.
        updates: dict[str, Any] = {}
        if existing.subject != job.subject:
            updates["subject"] = job.subject
        if existing.preset != job.preset:
            updates["preset"] = job.preset
        if updates:
            job_config.save_job(existing.with_updates(**updates))
        return
    job_config.create_job(job.name, subject=job.subject, preset=job.preset)


def _reset_job(job: _JobPlan) -> None:
    """Delete queue/sorted trees + drop the job entirely if the marker is present."""
    import job_config
    import paths

    if not _marker_exists(job.slug):
        return

    for tree in (paths.queue_dir(job.slug), paths.sorted_dir(job.slug)):
        if tree.is_dir():
            shutil.rmtree(tree, ignore_errors=True)

    # If the job is active, clear the active pointer first so delete_job()
    # doesn't refuse.
    active = job_config.get_active_slugs()
    if job.slug in active:
        remaining = [s for s in active if s != job.slug]
        job_config.set_active_slugs(remaining)

    if job_config.get_job(job.slug) is not None:
        try:
            job_config.delete_job(job.slug)
        except ValueError:
            # Job is queued / still referenced — dequeue and retry once.
            try:
                job_config.dequeue(job.slug)
                job_config.delete_job(job.slug)
            except Exception:
                pass

    marker = _marker_path(job.slug)
    if marker.is_file():
        marker.unlink()


def _seed_one(job: _JobPlan) -> dict[str, Any]:
    _ensure_job(job)
    queue_ct = _seed_queue(job)
    sorted_ct = _seed_sorted(job)
    _seed_logs(job, queue_ct, sorted_ct)
    _write_marker(job.slug, {
        "slug": job.slug,
        "seeded_at": time.time(),
        "queue": queue_ct,
        "sorted": sorted_ct,
        "note": "Managed by tools/seed_demo_data.py — remove via --reset only.",
    })
    return {"slug": job.slug, "queue": queue_ct, "sorted": sorted_ct}


def seed_all(force: bool = False) -> list[dict[str, Any]]:
    """Idempotently seed every demo job.

    When ``force`` is True, ``--reset``-equivalent cleanup runs first.
    """
    if force:
        for job in _DEMO_JOBS:
            _reset_job(job)

    import job_config

    out: list[dict[str, Any]] = []
    for job in _DEMO_JOBS:
        if _marker_exists(job.slug) and not force:
            out.append({"slug": job.slug, "skipped": True})
            continue
        out.append(_seed_one(job))

    # Multi-active the demo jobs so the sidebar's "running (N):" chip
    # showcases the parallel-jobs feature. Everyone still lands on the first
    # one via get_active_slug().
    try:
        job_config.set_active_slugs([job.slug for job in _DEMO_JOBS])
    except Exception as exc:  # pragma: no cover — never block the seed on this
        print(f"[warn] could not multi-activate demo jobs: {exc}")
    return out


def reset_all() -> None:
    for job in _DEMO_JOBS:
        _reset_job(job)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _print_plan() -> None:
    for job in _DEMO_JOBS:
        queue = sum(n for _, n in job.queue_sources)
        sorted_ct = sum(cp.count for cp, _ in job.sorted_layout)
        print(f"  {job.slug}: preset={job.preset}, queue={queue}, sorted={sorted_ct}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed a self-contained multi-job demo dataset for cull screenshots.",
    )
    parser.add_argument("--reset", action="store_true",
                        help="Wipe queue/sorted + delete demo jobs before seeding.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be seeded and exit.")
    args = parser.parse_args(argv)

    if args.dry_run:
        print("dry-run — would seed:")
        _print_plan()
        return 0

    # Confirm Pillow up-front so we fail with a helpful message rather than a
    # deep traceback from inside _render_gradient_jpeg.
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("[error] Pillow is required. Install with: pip install Pillow>=11", file=sys.stderr)
        return 2

    if args.reset:
        print("--reset: dropping existing demo jobs + trees...")
        reset_all()

    print("Seeding demo dataset (deterministic, no network)...")
    results = seed_all(force=False)
    for r in results:
        if r.get("skipped"):
            print(f"  {r['slug']}: already seeded (use --reset to rebuild)")
        else:
            print(f"  {r['slug']}: queue={r['queue']}, sorted={r['sorted']}")

    print()
    print("Next: open the dashboard against the seeded data:")
    print("    python pipeline_code/dashboard_enhanced.py")
    print("    # then browse http://localhost:5000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
