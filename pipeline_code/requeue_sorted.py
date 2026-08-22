"""
requeue_sorted.py - re-process every classified image in <sorted>/<slug>/.

Three operations, gated by flags so the tool is safe to run in stages:

  --audit              (default) print per-(category,source) totals and orphans,
                       no filesystem changes.
  --clean-orphans      move images that have no matching .txt sidecar into a
                       quarantine folder under <sorted>/.quarantine/. Reversible.
  --requeue            move every valid triple (image + .txt + .vision.json)
                       back into <queue>/<slug>/<source>/ and synthesise a fresh
                       .meta.json, so the vision worker re-classifies them on
                       the next pass. Old .vision.json files are archived under
                       <sorted>/.requeue_archive/.

Filters:
  --source CIVITAI     only operate on this scraper source.
  --category NSFW      only operate on this category.
  --limit 100          stop after N files (per pass) - good for piloting.
  --dry-run            print what would happen, change nothing.

Filename convention assumed: vision worker writes
  <category>_<msg_id>_<unix_ts>_<3digit_rand>.<ext>
so we strip the leading <category>_ and trailing _<ts>_<rand> to recover
<msg_id>, then write `<msg_id>.<ext>` into the queue. The vision worker
will re-classify and emit a fresh sorted filename.

Run from pipeline_code/. All paths come from .env via paths.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from paths import queue_dir, sorted_dir  # noqa: E402

SLUG: str = os.environ.get("PIPELINE_SLUG", "default")
SORTED_ROOT: Path = sorted_dir(SLUG)
QUEUE_ROOT: Path = queue_dir(SLUG)
QUARANTINE_DIR: Path = SORTED_ROOT / ".quarantine"
ARCHIVE_DIR: Path = SORTED_ROOT / ".requeue_archive"

IMAGE_EXT: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

# Vision worker writes: <lowercategory>_<msg_id>_<ts>_<rand>.<ext>
# msg_id can itself contain underscores (e.g. "civitai_113099841", "x_<tid>_<hash>").
# We capture msg_id by anchoring on the trailing _<10digit_ts>_<3digit_rand>.
_FILENAME_PARTS = re.compile(r"^([a-z]+)_(.+)_(\d{9,11})_(\d{2,4})$")


# ── Audit ─────────────────────────────────────────────────────────────────────


def collect_buckets(source: str | None = None, category: str | None = None):
    """Return {(cat, src): {stem: {"img": <ext or None>, "txt": bool, "vision": bool, "files": [paths]}}}."""
    buckets: dict[tuple[str, str], dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"img": None, "txt": False, "vision": False, "files": []}
    ))
    if not SORTED_ROOT.exists():
        return buckets
    for path in SORTED_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(SORTED_ROOT)
        parts = rel.parts
        if len(parts) < 3 or parts[0].startswith("."):
            continue  # ignore hidden dirs (.quarantine etc.) + top-level junk
        cat, src = parts[0], parts[1]
        if category and cat.lower() != category.lower():
            continue
        if source and src.lower() != source.lower():
            continue
        name = path.name
        if name.endswith(".vision.json"):
            stem = name[:-len(".vision.json")]
            buckets[(cat, src)][stem]["vision"] = True
        elif path.suffix.lower() in IMAGE_EXT:
            stem = path.stem
            buckets[(cat, src)][stem]["img"] = path.suffix.lower()
        elif path.suffix.lower() == ".txt":
            stem = path.stem
            buckets[(cat, src)][stem]["txt"] = True
        else:
            continue
        buckets[(cat, src)][stem]["files"].append(path)
    return buckets


def audit(buckets) -> tuple[int, int, int, int]:
    print(f"{'Category':<22} {'Source':<18} {'Triples':>8} {'OrphanImg':>10} {'OrphanTxt':>10} {'OrphanJson':>11}")
    print("-" * 84)
    tot_t = tot_oi = tot_ot = tot_oj = 0
    for (cat, src), items in sorted(buckets.items()):
        t = oi = ot = oj = 0
        for stem, flags in items.items():
            if flags["img"] and flags["txt"]:
                t += 1
            elif flags["img"]:
                oi += 1
            elif flags["txt"]:
                ot += 1
            elif flags["vision"]:
                oj += 1
        tot_t += t
        tot_oi += oi
        tot_ot += ot
        tot_oj += oj
        print(f"{cat:<22} {src:<18} {t:>8} {oi:>10} {ot:>10} {oj:>11}")
    print("-" * 84)
    print(f"{'TOTAL':<41} {tot_t:>8} {tot_oi:>10} {tot_ot:>10} {tot_oj:>11}")
    return tot_t, tot_oi, tot_ot, tot_oj


# ── Cleanup ──────────────────────────────────────────────────────────────────


def quarantine_orphans(buckets, dry_run: bool, limit: int | None) -> int:
    """Move images with no matching .txt into <sorted>/.quarantine/<cat>/<src>/."""
    moved = 0
    for (cat, src), items in buckets.items():
        for stem, flags in items.items():
            if flags["img"] and not flags["txt"]:
                src_paths = [p for p in flags["files"] if p.suffix.lower() in IMAGE_EXT]
                for p in src_paths:
                    dest = QUARANTINE_DIR / cat / src / p.name
                    print(f"  QUARANTINE {p.relative_to(SORTED_ROOT)} -> .quarantine/{dest.relative_to(QUARANTINE_DIR)}")
                    if not dry_run:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(p), str(dest))
                    moved += 1
                    if limit and moved >= limit:
                        return moved
    return moved


# ── Requeue ──────────────────────────────────────────────────────────────────


def _strip_filename(stem: str) -> str | None:
    """Recover the original `<msg_id>` from a sorted-style filename.

    Returns None for files that don't match the convention (caller can fall
    back to using the full stem).
    """
    m = _FILENAME_PARTS.match(stem)
    if m:
        return m.group(2)
    return None


def requeue_triples(buckets, dry_run: bool, limit: int | None) -> int:
    """Move every (image + .txt) triple back into the queue and archive .vision.json."""
    moved = 0
    for (cat, src), items in buckets.items():
        for stem, flags in items.items():
            if not (flags["img"] and flags["txt"]):
                continue
            ext = flags["img"]
            old_image = next((p for p in flags["files"] if p.suffix.lower() == ext), None)
            old_txt = next((p for p in flags["files"] if p.suffix.lower() == ".txt"), None)
            old_vision = next((p for p in flags["files"] if p.name.endswith(".vision.json")), None)
            if old_image is None or old_txt is None:
                continue

            msg_id = _strip_filename(stem) or stem
            queue_target = QUEUE_ROOT / src
            new_image = queue_target / f"{msg_id}{ext}"
            new_txt = queue_target / f"{msg_id}.txt"
            new_meta = queue_target / f"{msg_id}.meta.json"

            if new_image.exists():
                # Already queued under this msg_id - skip to avoid clobbering.
                continue

            print(f"  REQUEUE  [{cat}/{src}] {stem} -> queue/{src}/{msg_id}{ext}")
            if not dry_run:
                queue_target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_image), str(new_image))
                shutil.move(str(old_txt), str(new_txt))
                # Synthesise a minimal .meta.json so the vision worker has the
                # source label - it falls back to {} if absent, but giving it
                # an explicit message_id keeps audit trails consistent.
                new_meta.write_text(json.dumps({
                    "message_id":     msg_id,
                    "source_channel": src,
                    "requeued_from":  f"{cat}/{src}/{stem}{ext}",
                    "requeued_at":    datetime.utcnow().isoformat(),
                }, indent=2), encoding="utf-8")
                if old_vision is not None and old_vision.exists():
                    archive_dest = ARCHIVE_DIR / cat / src / old_vision.name
                    archive_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_vision), str(archive_dest))
            moved += 1
            if limit and moved >= limit:
                return moved
    return moved


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit / clean / re-queue the sorted tree.")
    parser.add_argument("--audit", action="store_true", help="(default) print summary only")
    parser.add_argument("--clean-orphans", action="store_true",
                        help="move images-with-no-text into <sorted>/.quarantine/")
    parser.add_argument("--requeue", action="store_true",
                        help="move every valid triple back into the queue for reprocessing")
    parser.add_argument("--dry-run", action="store_true", help="don't change anything, just print")
    parser.add_argument("--source", help="filter to one scraper source (e.g. civitai)")
    parser.add_argument("--category", help="filter to one sorted category (e.g. NSFW)")
    parser.add_argument("--limit", type=int, help="stop after N files in any single phase")
    args = parser.parse_args()

    print(f"sorted_root = {SORTED_ROOT}")
    print(f"queue_root  = {QUEUE_ROOT}")
    print(f"slug        = {SLUG}")
    if args.source:
        print(f"filter src  = {args.source}")
    if args.category:
        print(f"filter cat  = {args.category}")
    if args.dry_run:
        print("(dry-run: no files will be changed)")
    print()

    buckets = collect_buckets(source=args.source, category=args.category)
    if not buckets:
        print("Nothing matched the filters.")
        return 0

    audit(buckets)

    if args.clean_orphans:
        print("\n--- quarantining orphan images ---")
        moved = quarantine_orphans(buckets, dry_run=args.dry_run, limit=args.limit)
        print(f"\nQuarantined {moved} orphan image(s).")
        if not args.dry_run:
            buckets = collect_buckets(source=args.source, category=args.category)

    if args.requeue:
        print("\n--- requeuing valid triples ---")
        moved = requeue_triples(buckets, dry_run=args.dry_run, limit=args.limit)
        print(f"\nRequeued {moved} triple(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
