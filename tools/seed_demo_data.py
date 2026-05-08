"""seed_demo_data.py - Synthetic dataset for README screenshots.

Builds a self-contained "Artistic Showcase" mock dataset under
``data/sorted/artistic_showcase/`` so a reviewer can boot the dashboard
on a fresh clone and immediately see populated tabs, leaderboards, and
gallery cards - without exposing any real pipeline output.

Images come from picsum.photos (Lorem Picsum, free placeholder photos
released under the Unsplash license). Categories follow the live
taxonomy (InstagramInfluencer / NSFW / Professional / Amateur / Unknown
/ Watermarked); we just remap them to artistic equivalents in the prompt
text so the screenshots read as a generic curation use case.

Run from the repo root:

    python tools/seed_demo_data.py            # build mock data
    python tools/seed_demo_data.py --clean    # delete + rebuild

Then point the dashboard at the demo slug:

    PIPELINE_TOPIC="Artistic Showcase" PIPELINE_SLUG=artistic_showcase \\
        FLASK_PORT=5050 \\
        python pipeline_code/dashboard_enhanced.py
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "sorted" / "artistic_showcase"

random.seed(0xC0FFEE)  # reproducible mock - same images every run


# ── Mock taxonomy ────────────────────────────────────────────────────────────

# We keep the live category names so the dashboard's chips/filters match
# what a real run would show. The prompts/descriptions paint them in
# artistic-curation terms.
CATEGORY_FLAVORS: dict[str, dict[str, Any]] = {
    "InstagramInfluencer": {
        "n": 8,
        "ovr_range": (78, 94),
        "rel_range": (80, 92),
        "subjects": [
            "candid coastal shot at golden hour",
            "rooftop city skyline silhouette",
            "studio editorial portrait of seated subject",
            "lifestyle living-room composition with natural light",
            "minimal product flat-lay on linen backdrop",
            "neon-lit night street with reflections",
            "festival crowd shot with bokeh background",
            "café window scene with steam rising from cup",
        ],
    },
    "Professional": {
        "n": 7,
        "ovr_range": (82, 96),
        "rel_range": (78, 90),
        "subjects": [
            "architectural detail of a modernist staircase",
            "studio still-life with copper kettle and citrus",
            "commercial automotive front-three-quarter shot",
            "interior magazine spread of a Scandinavian kitchen",
            "long-exposure waterfall with silky motion blur",
            "high-key beauty close-up on grey paper backdrop",
            "documentary street portrait, Leica-style framing",
        ],
    },
    "Amateur": {
        "n": 5,
        "ovr_range": (45, 70),
        "rel_range": (50, 72),
        "subjects": [
            "phone snapshot of a golden retriever in a park",
            "handheld concert shot with motion blur",
            "selfie reflection in a metro carriage window",
            "kitchen counter shot of homemade pasta dish",
            "garden flower close-up taken on a cloudy morning",
        ],
    },
    "Unknown": {
        "n": 3,
        "ovr_range": (55, 75),
        "rel_range": (35, 60),
        "subjects": [
            "ambiguous mixed-media composition",
            "abstract geometric pattern on textured paper",
            "unidentified macro detail of weathered metal",
        ],
    },
    "Watermarked": {
        "n": 4,
        "ovr_range": (72, 88),
        "rel_range": (70, 84),
        "subjects": [
            "photographer-watermarked landscape, mountain range at dusk",
            "branded studio shot, large logo overlay top-right",
            "tourism-board promotional cityscape with banner text",
            "stock-agency placeholder, faint diagonal watermark",
        ],
    },
    "NSFW": {
        "n": 2,
        "ovr_range": (40, 55),
        "rel_range": (5, 25),
        "subjects": [
            "[mock] redacted - placeholder shows landscape image",
            "[mock] redacted - placeholder shows architectural detail",
        ],
    },
    "DISCARD": {
        "n": 6,
        "ovr_range": (10, 40),
        "rel_range": (5, 30),
        "subjects": [
            "screenshot of a social media feed with UI chrome",
            "low-resolution thumbnail, severe compression artefacts",
            "comparison grid of three model outputs labelled A/B/C",
            "blurry handheld shot, subject completely out of focus",
            "infographic with heavy text overlay, chart annotations",
            "duplicate frame already classified earlier in the run",
        ],
    },
}

SOURCES = ["nanobanana", "civitai", "civitai_red", "twitter_x", "reddit", "discord_ud", "my_dataset"]
ART_MEDIUMS = ["photograph", "photograph", "photograph", "digital_painting", "illustration", "mixed"]


def _picsum_url(width: int, height: int, seed: int) -> str:
    """Deterministic Lorem Picsum URL so the same seed always returns the same image."""
    return f"https://picsum.photos/seed/{seed}/{width}/{height}.jpg"


def _download(url: str, dest: Path, attempts: int = 3) -> bool:
    for attempt in range(attempts):
        try:
            r = requests.get(url, timeout=20, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 5000:
                dest.write_bytes(r.content)
                return True
        except requests.RequestException as exc:
            print(f"  [retry {attempt + 1}/{attempts}] {url}: {exc}")
        time.sleep(1.5 * (attempt + 1))
    return False


def _synthetic_prompt(category: str, subject: str, source: str) -> str:
    """Build a believable prompt-style caption for the demo. Not used by the
    pipeline at runtime - just shown in the dashboard preview."""
    style_extras = [
        "shot on Sony A7R IV, 85mm f/1.4 prime",
        "natural light, soft fill from a north-facing window",
        "shot on Hasselblad medium format, leaf shutter",
        "Kodak Portra 400 film simulation, slight grain",
        "cinematic color grade, teal-and-orange palette",
        "high dynamic range, minimal shadow clipping",
        "rim light from rear left, key light at 45 degrees",
    ]
    moods = ["serene", "candid", "polished", "spontaneous", "moody", "ethereal"]
    return (
        f"A {random.choice(moods)} {subject}. "
        f"Source: {source}. {random.choice(style_extras)}. "
        f"Demo image curated for the {category} bucket of the artistic showcase preview."
    )


def _synthetic_meta(category: str, subject: str) -> dict[str, Any]:
    flavor = CATEGORY_FLAVORS[category]
    ovr = random.randint(*flavor["ovr_range"])
    rel = random.randint(*flavor["rel_range"])
    quality = max(1, min(10, ovr // 10))
    art_medium = random.choice(ART_MEDIUMS)
    nsfw = category == "NSFW"
    contains_overlay = category == "Watermarked"
    is_grid = category == "DISCARD" and "comparison grid" in subject
    is_screenshot = category == "DISCARD" and "screenshot" in subject

    return {
        "description": (
            f"A {subject} - rendered as a {art_medium}; mock data for the "
            f"dashboard preview. No real classification was performed."
        ),
        "primary_subject": subject,
        "is_screenshot": is_screenshot,
        "is_composite_grid": is_grid,
        "contains_text_overlay": contains_overlay,
        "is_human_photograph": art_medium == "photograph" and category != "Unknown",
        "art_medium": art_medium,
        "photorealistic_style": art_medium == "photograph",
        "has_ai_flaws": category in ("Amateur", "DISCARD") and random.random() < 0.4,
        "woman_present": False,
        "nsfw": nsfw,
        "OVR_Quality_Score": ovr,
        "REL_Quality_Score": rel,
        "quality_score": quality,
        "category": category,
        "reason": f"Mock {category} entry generated by seed_demo_data.py for screenshot purposes.",
    }


def _make_one_item(idx: int, category: str, subject: str, source: str) -> bool:
    seed_val = (hash((category, source, idx)) & 0xFFFF) + 1
    target_dir = DATA_ROOT / category / source
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int((datetime.utcnow() - timedelta(days=random.randint(0, 30))).timestamp())
    safe_name = f"{category.lower()}_demo_{seed_val:05d}_{timestamp}_{random.randint(100, 999)}"

    img_path = target_dir / f"{safe_name}.jpg"
    txt_path = target_dir / f"{safe_name}.txt"
    json_path = target_dir / f"{safe_name}.vision.json"

    width, height = random.choice([(800, 1200), (1200, 800), (1024, 1024)])
    if not _download(_picsum_url(width, height, seed_val), img_path):
        print(f"  [skip] failed to download placeholder for {safe_name}")
        return False

    txt_path.write_text(_synthetic_prompt(category, subject, source), encoding="utf-8")
    payload = _synthetic_meta(category, subject)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed mock dashboard data.")
    parser.add_argument("--clean", action="store_true", help="delete data/sorted/artistic_showcase/ before seeding")
    parser.add_argument("--dry-run", action="store_true", help="print what would be created, don't fetch images")
    args = parser.parse_args()

    if args.clean and DATA_ROOT.exists():
        print(f"removing {DATA_ROOT}")
        shutil.rmtree(DATA_ROOT)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    total_planned = sum(flavor["n"] for flavor in CATEGORY_FLAVORS.values())
    print(f"seeding {total_planned} mock items into {DATA_ROOT}")
    print(f"sources: {SOURCES}")

    success = 0
    skipped = 0
    counter = 0
    for category, flavor in CATEGORY_FLAVORS.items():
        subjects = list(flavor["subjects"])
        random.shuffle(subjects)
        for i in range(flavor["n"]):
            counter += 1
            subject = subjects[i % len(subjects)]
            source = random.choice(SOURCES)
            print(f"  [{counter:>2}/{total_planned}] {category}/{source}: {subject[:60]}")
            if args.dry_run:
                continue
            if _make_one_item(i, category, subject, source):
                success += 1
            else:
                skipped += 1

    print()
    print(f"done. created={success} skipped={skipped}")
    print()
    print("Next: boot the dashboard against the demo slug")
    print("    PIPELINE_TOPIC=\"Artistic Showcase\" PIPELINE_SLUG=artistic_showcase \\")
    print("        FLASK_PORT=5050 \\")
    print("        python pipeline_code/dashboard_enhanced.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
