"""seed_demo_data.py - Synthetic multi-job dataset for README screenshots.

Produces a self-contained demo dataset the dashboard can render against so a
fresh clone shows populated Jobs / Global Gallery / Global Stats tabs without
running a real scraper OR leaking the user's real production data.

Design:

* **Real photos**, not gradients. Sample photos ship with the repo under
  ``docs/mock-data/samples/*.jpg`` (fetched once from picsum.photos — public
  domain / CC0). The seeder copies + renames them into per-job queue and
  sorted trees. Re-run ``python tools/fetch_demo_samples.py`` to refresh the
  cached photos.
* **Generic demo slugs** so nothing in the demo dataset collides with a real
  user job. Every slug starts with ``demo_``.
* **Idempotent** via per-slug marker files under ``data/jobs/``. ``--reset``
  removes only what a marker authorised (never your real jobs' data).

The five demo jobs cover a visually distinct spread so screenshots look like
"cull is running across many topics":

  * ``tiktoker``      — TikTok-style portrait photography
  * ``car_spotting``   — Automotive photography and car spotting
  * ``landscapes``    — Landscape and nature photography
  * ``street_photography``   — Street photography, candid urban
  * ``pet_gallery``    — Pet and animal photography

Usage:

    python tools/seed_demo_data.py             # seed all jobs (idempotent)
    python tools/seed_demo_data.py --reset     # wipe + reseed
    python tools/seed_demo_data.py --dry-run   # print the plan, write nothing

Requires Pillow (for the vision-json metadata, not for image generation) and
the cached samples under ``docs/mock-data/samples/`` (populated by
``tools/fetch_demo_samples.py``).
"""
from __future__ import annotations

import argparse
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
SAMPLES_DIR = REPO_ROOT / "docs" / "mock-data" / "samples"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ── Demo dataset shape ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CategoryPlan:
    """How many items land in a sorted category."""

    name: str
    count: int


@dataclass(frozen=True)
class _JobPlan:
    slug: str
    name: str
    subject: str
    preset: str
    prompts: tuple[str, ...]
    queue_sources: tuple[tuple[str, int], ...]
    sorted_layout: tuple[tuple[_CategoryPlan, tuple[str, ...]], ...]


# Source folder names match what real scrapers write. See
# feed_local_folder.SOURCE_NAME, scraper_x, scraper_gallery_dl.SOURCE_NAME,
# scraper_discord, scraper_civitai for the origin of each literal.
_STANDARD_SOURCES: tuple[str, ...] = (
    "civitai_civitai_com",
    "twitter_x",
    "gallery_dl",
    "discord_1",
    "local",
    "reddit",
)


_TIKTOKER_PROMPTS: tuple[str, ...] = (
    "TikTok-style vertical portrait, ring light, warm skin tones, natural background",
    "candid content-creator lifestyle shot, phone-camera aesthetic, soft window light",
    "creator sitting at a desk with LED strip lighting behind, mid-conversation on camera",
    "outdoor coffee-shop portrait, golden hour, natural DSLR bokeh",
    "clean studio headshot, single softbox, editorial styling",
    "creator posing outdoors in casual streetwear, blurred storefronts behind",
    "close-up product-hold beauty shot, natural window light, neutral wall backdrop",
)

_CAR_PROMPTS: tuple[str, ...] = (
    "commercial photograph of a matte-black electric sedan on a wet street at night, cinematic rim light",
    "silver coupe cornering on a mountain switchback at sunrise, hero shot for automotive campaign",
    "studio product shot of a chrome-wheeled roadster on seamless white, soft key light",
    "vintage roadster parked outside a Mid-Century modernist house, kodachrome film simulation",
    "compact hatchback drifting through mist, tail-light streaks, motion blur, low angle",
    "concept-car reveal on a rotating platform, hard rim lights, black backdrop, ISO 100 studio look",
    "delivery van at dawn in an urban loading bay, practical lighting, natural colour",
)

_LANDSCAPE_PROMPTS: tuple[str, ...] = (
    "wide golden-hour vista over rolling foothills, cool shadow tones, long lens compression",
    "still alpine lake reflecting jagged peaks at first light, low ISO, deep DOF",
    "sea-cliff coastline at blue hour, long-exposure surf, moody clouds",
    "forest interior with god rays through morning mist, warm earthy palette",
    "arid desert canyon, harsh midday sun, textured sandstone walls",
    "farmland at dusk, wide horizon, subtle atmospheric haze, warm sky gradient",
    "black sand beach with basalt columns, overcast lighting, minimalist composition",
)

_STREET_PROMPTS: tuple[str, ...] = (
    "candid street portrait in a rainy alleyway, single practical light, 35mm lens",
    "commuters crossing at a busy intersection, motion blur, telephoto compression",
    "market stall vendor at dawn, soft ambient light, documentary framing",
    "narrow European lane with a lone cyclist, film-grain look, muted palette",
    "neon-lit downtown corner after midnight, reflections in wet asphalt",
    "cafe patrons framed through a rainy window, high-contrast monochrome feel",
    "child chasing pigeons in a public square, wide-angle, natural sunlight",
)

_PET_PROMPTS: tuple[str, ...] = (
    "portrait of a golden retriever in a sunlit backyard, shallow DOF, joyful expression",
    "grey tabby cat perched on a windowsill, backlit, soft rim light",
    "action shot of a border collie mid-jump catching a frisbee, freeze-frame lighting",
    "close-up macro of a rabbit's whiskers, natural warm tones",
    "puppy asleep on a linen blanket, top-down view, minimal composition",
    "black lab swimming in a lake, water splash, telephoto compression",
    "two kittens playing on a hardwood floor, wide-angle low perspective",
)


_TIKTOKER_JOB = _JobPlan(
    slug="tiktoker",
    name="TikToker",
    subject="TikTok-style portrait photography",
    preset="default",
    prompts=_TIKTOKER_PROMPTS,
    queue_sources=(
        ("twitter_x", 6),
        ("reddit", 5),
        ("civitai_civitai_com", 4),
        ("gallery_dl", 3),
        ("local", 2),
    ),
    sorted_layout=(
        (_CategoryPlan("Keep", 22), _STANDARD_SOURCES),
        (_CategoryPlan("Borderline", 11), _STANDARD_SOURCES),
        (_CategoryPlan("OffTopic", 6), _STANDARD_SOURCES),
        (_CategoryPlan("DISCARD", 9), _STANDARD_SOURCES),
    ),
)

_CARSPOTTING_JOB = _JobPlan(
    slug="car_spotting",
    name="Car spotting",
    subject="Automotive photography and car spotting",
    preset="default",
    prompts=_CAR_PROMPTS,
    queue_sources=(
        ("reddit", 5),
        ("gallery_dl", 4),
        ("civitai_civitai_com", 3),
        ("twitter_x", 3),
        ("discord_1", 2),
        ("local", 2),
    ),
    sorted_layout=(
        (_CategoryPlan("Keep", 26), _STANDARD_SOURCES),
        (_CategoryPlan("Borderline", 10), _STANDARD_SOURCES),
        (_CategoryPlan("OffTopic", 5), _STANDARD_SOURCES),
        (_CategoryPlan("DISCARD", 7), _STANDARD_SOURCES),
    ),
)

_LANDSCAPE_JOB = _JobPlan(
    slug="landscapes",
    name="Landscapes",
    subject="Landscape and nature photography",
    preset="default",
    prompts=_LANDSCAPE_PROMPTS,
    queue_sources=(
        ("reddit", 6),
        ("gallery_dl", 4),
        ("local", 3),
        ("civitai_civitai_com", 2),
        ("twitter_x", 2),
    ),
    sorted_layout=(
        (_CategoryPlan("Keep", 30), _STANDARD_SOURCES),
        (_CategoryPlan("Borderline", 8), _STANDARD_SOURCES),
        (_CategoryPlan("OffTopic", 4), _STANDARD_SOURCES),
        (_CategoryPlan("DISCARD", 5), _STANDARD_SOURCES),
    ),
)

_STREETPHOTO_JOB = _JobPlan(
    slug="street_photography",
    name="Street photography",
    subject="Street photography, candid urban",
    preset="default",
    prompts=_STREET_PROMPTS,
    queue_sources=(
        ("reddit", 4),
        ("twitter_x", 4),
        ("gallery_dl", 3),
        ("local", 2),
    ),
    sorted_layout=(
        (_CategoryPlan("Keep", 18), _STANDARD_SOURCES),
        (_CategoryPlan("Borderline", 9), _STANDARD_SOURCES),
        (_CategoryPlan("OffTopic", 5), _STANDARD_SOURCES),
        (_CategoryPlan("DISCARD", 6), _STANDARD_SOURCES),
    ),
)

_PETGALLERY_JOB = _JobPlan(
    slug="pet_gallery",
    name="Pet gallery",
    subject="Pet and animal photography",
    preset="default",
    prompts=_PET_PROMPTS,
    queue_sources=(
        ("reddit", 5),
        ("gallery_dl", 3),
        ("civitai_civitai_com", 3),
        ("twitter_x", 2),
        ("local", 2),
    ),
    sorted_layout=(
        (_CategoryPlan("Keep", 24), _STANDARD_SOURCES),
        (_CategoryPlan("Borderline", 8), _STANDARD_SOURCES),
        (_CategoryPlan("OffTopic", 4), _STANDARD_SOURCES),
        (_CategoryPlan("DISCARD", 6), _STANDARD_SOURCES),
    ),
)


_DEMO_JOBS: tuple[_JobPlan, ...] = (
    _TIKTOKER_JOB,
    _CARSPOTTING_JOB,
    _LANDSCAPE_JOB,
    _STREETPHOTO_JOB,
    _PETGALLERY_JOB,
)

# Slugs from prior seeder versions that we still purge on --reset so upgrading
# an existing demo install leaves no stale jobs behind.
_LEGACY_DEMO_SLUGS: tuple[str, ...] = ("car_ads", "realistic_female_influencer")

_MARKER_TEMPLATE: str = "{slug}_demo_marker"


# ── Sample loader ────────────────────────────────────────────────────────────


def _list_samples() -> list[Path]:
    """Return every cached sample photo. Raises if the cache is empty."""
    if not SAMPLES_DIR.is_dir():
        raise FileNotFoundError(
            f"sample cache missing: {SAMPLES_DIR}\n"
            "Run: python tools/fetch_demo_samples.py"
        )
    photos = sorted(p for p in SAMPLES_DIR.glob("*.jpg") if p.is_file())
    if not photos:
        raise FileNotFoundError(
            f"sample cache is empty: {SAMPLES_DIR}\n"
            "Run: python tools/fetch_demo_samples.py"
        )
    return photos


def _pick_sample(seed: str, samples: list[Path]) -> Path:
    """Deterministically pick one cached photo for a given seed string."""
    rng = random.Random(seed)
    return samples[rng.randrange(len(samples))]


# ── Vision-json shape (aligns with build_response_format's required set) ─────


def _synthetic_vision_meta(
    idx: int,
    category: str,
    source: str,
    subject: str,
    is_terminal: bool = False,
) -> dict[str, Any]:
    """Return a payload that satisfies ``vision_prompt.build_response_format``."""
    rng = random.Random(f"demo-scores-{category}-{source}-{idx}")

    if is_terminal:
        ovr = rng.randint(15, 42)
        rel = rng.randint(10, 35)
    elif category == "Keep":
        ovr = rng.randint(72, 94)
        rel = rng.randint(70, 92)
    elif category == "Borderline":
        ovr = rng.randint(55, 74)
        rel = rng.randint(50, 72)
    else:  # OffTopic and everything else
        ovr = rng.randint(45, 68)
        rel = rng.randint(20, 58)

    quality = max(1, min(10, ovr // 10))
    # Alternate obviously-fake private-IP worker instance labels so the
    # Activity feed + Vision tab surface a plausible fleet without ever
    # referencing a real endpoint.
    workers = (
        "mock-lm-studio-a (10.0.0.42:1234)",
        "mock-lm-studio-b (192.168.99.5:1235)",
        "mock-ollama-desktop (127.0.0.1:11434)",
        "mock-groq-cloud (api.mock.local)",
    )
    worker = workers[idx % len(workers)]

    return {
        "description": (
            f"Mock demo item #{idx} routed to {category} — seed_demo_data.py "
            f"placeholder for the {subject} preview. No real classification."
        ),
        "primary_subject": subject,
        "is_screenshot": False,
        "is_composite_grid": False,
        "contains_text_overlay": False,
        "is_human_photograph": True,
        "art_medium": "photograph",
        "photorealistic_style": True,
        "has_ai_flaws": is_terminal,
        "woman_present": False,
        "nsfw": False,
        "OVR_Quality_Score": ovr,
        "REL_Quality_Score": rel,
        "quality_score": quality,
        "category": category,
        "reason": f"Mock demo item routed to {category} by seed_demo_data.py.",
        "caption": "",
        "worker_instance": worker,
    }


# ── Filesystem writers ───────────────────────────────────────────────────────


def _write_triple(
    base_dir: Path,
    idx: int,
    category: str,
    source: str,
    job: _JobPlan,
    kind: str,
    samples: list[Path],
) -> None:
    """Write one ``<stem>.jpg`` + ``.txt`` + ``.vision.json`` triple."""
    import os as _os
    import time as _time

    base_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{category.lower()}_{source}_{kind}_{idx:04d}"
    img_path = base_dir / f"{stem}.jpg"
    txt_path = base_dir / f"{stem}.txt"
    json_path = base_dir / f"{stem}.vision.json"

    sample = _pick_sample(f"{job.slug}-{category}-{source}-{idx}", samples)
    shutil.copyfile(sample, img_path)
    txt_path.write_text(job.prompts[idx % len(job.prompts)], encoding="utf-8")
    payload = _synthetic_vision_meta(
        idx,
        category,
        source,
        job.subject,
        is_terminal=category in {"DISCARD", "CORRUPT"},
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stagger = idx * 5
    ts = _time.time() - stagger
    for p in (img_path, txt_path, json_path):
        try:
            _os.utime(p, (ts, ts))
        except OSError:
            pass


def _seed_queue(job: _JobPlan, samples: list[Path]) -> int:
    import paths

    queue_root = paths.queue_dir(job.slug)
    if queue_root.exists():
        shutil.rmtree(queue_root)

    total = 0
    idx = 1
    for source, count in job.queue_sources:
        target = queue_root / source
        for _ in range(count):
            _write_triple(target, idx, "Keep", source, job, "queue", samples)
            idx += 1
            total += 1
    return total


def _seed_sorted(job: _JobPlan, samples: list[Path]) -> int:
    import paths

    sorted_root = paths.sorted_dir(job.slug)
    if sorted_root.exists():
        shutil.rmtree(sorted_root)

    total = 0
    idx = 1
    for cat_plan, sources in job.sorted_layout:
        for i in range(cat_plan.count):
            source = sources[i % len(sources)]
            target = sorted_root / cat_plan.name / source
            _write_triple(target, idx, cat_plan.name, source, job, "sorted", samples)
            idx += 1
            total += 1
    return total


def _seed_logs(job: _JobPlan, queue_ct: int, sorted_ct: int) -> None:
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


# ── Marker file ──────────────────────────────────────────────────────────────


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
        updates: dict[str, Any] = {}
        if existing.subject != job.subject:
            updates["subject"] = job.subject
        if existing.preset != job.preset:
            updates["preset"] = job.preset
        if updates:
            job_config.save_job(existing.with_updates(**updates))
        return
    job_config.create_job(job.name, subject=job.subject, preset=job.preset)


def _drop_job(slug: str) -> None:
    """Best-effort delete of ``slug`` — clears active pointer + dequeues first."""
    import job_config
    import paths

    for tree in (paths.queue_dir(slug), paths.sorted_dir(slug)):
        if tree.is_dir():
            shutil.rmtree(tree, ignore_errors=True)

    active = job_config.get_active_slugs()
    if slug in active:
        job_config.set_active_slugs([s for s in active if s != slug])

    if job_config.get_job(slug) is not None:
        try:
            job_config.delete_job(slug)
        except ValueError:
            try:
                job_config.dequeue(slug)
                job_config.delete_job(slug)
            except Exception:
                pass


def _reset_job(job: _JobPlan) -> None:
    if not _marker_exists(job.slug):
        return
    _drop_job(job.slug)
    marker = _marker_path(job.slug)
    if marker.is_file():
        marker.unlink()


def _purge_legacy_slugs() -> list[str]:
    """Drop any leftover demo jobs from prior seeder versions.

    Only touches slugs that carry the demo marker OR match the well-known
    legacy names. Never touches a slug the user created themselves.
    """
    import job_config

    purged: list[str] = []
    for slug in _LEGACY_DEMO_SLUGS:
        if _marker_exists(slug) or job_config.get_job(slug) is not None:
            _drop_job(slug)
            marker = _marker_path(slug)
            if marker.is_file():
                marker.unlink()
            purged.append(slug)
    return purged


def _seed_one(job: _JobPlan, samples: list[Path]) -> dict[str, Any]:
    _ensure_job(job)
    queue_ct = _seed_queue(job, samples)
    sorted_ct = _seed_sorted(job, samples)
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
    """Idempotently seed every demo job. ``force=True`` wipes + reseeds."""
    if force:
        for job in _DEMO_JOBS:
            _reset_job(job)

    import job_config

    samples = _list_samples()

    out: list[dict[str, Any]] = []
    for job in _DEMO_JOBS:
        if _marker_exists(job.slug) and not force:
            out.append({"slug": job.slug, "skipped": True})
            continue
        out.append(_seed_one(job, samples))

    # Multi-active all demo jobs so the sidebar chip reads "running (5)" —
    # showcases the parallel-jobs feature front-and-centre.
    try:
        job_config.set_active_slugs([job.slug for job in _DEMO_JOBS])
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not multi-activate demo jobs: {exc}")
    return out


def reset_all() -> None:
    for job in _DEMO_JOBS:
        _reset_job(job)
    _purge_legacy_slugs()


# ── CLI ─────────────────────────────────────────────────────────────────────


def _print_plan() -> None:
    for job in _DEMO_JOBS:
        queue = sum(n for _, n in job.queue_sources)
        sorted_ct = sum(cp.count for cp, _ in job.sorted_layout)
        print(f"  {job.slug}: preset={job.preset}, queue={queue}, sorted={sorted_ct}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed a multi-job demo dataset using real cached SFW photos.",
    )
    parser.add_argument("--reset", action="store_true",
                        help="Wipe queue/sorted + delete demo jobs (incl. legacy) before seeding.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be seeded and exit.")
    args = parser.parse_args(argv)

    if args.dry_run:
        print("dry-run — would seed:")
        _print_plan()
        return 0

    if not SAMPLES_DIR.is_dir() or not any(SAMPLES_DIR.glob("*.jpg")):
        print(
            "[error] sample cache missing.\n"
            f"        Populate: python tools/fetch_demo_samples.py\n"
            f"        Expected: {SAMPLES_DIR}",
            file=sys.stderr,
        )
        return 2

    if args.reset:
        print("--reset: dropping existing demo + legacy demo jobs...")
        reset_all()
        legacy = _purge_legacy_slugs()
        if legacy:
            print(f"  purged legacy demo slugs: {', '.join(legacy)}")

    print("Seeding demo dataset (real cached photos, no network)...")
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
