"""fetch_demo_samples.py - Populate the cached SFW photo pool used by
``tools/seed_demo_data.py``.

Fetches ~40 real photos from picsum.photos (public-domain / CC0 stock,
sourced from Unsplash) into ``docs/mock-data/samples/*.jpg``. Idempotent —
existing files are kept unless ``--refresh`` is passed. Deterministic —
the same picsum IDs are re-requested every run so the demo dataset is
reproducible across machines.

Usage:
    python tools/fetch_demo_samples.py             # fill any missing files
    python tools/fetch_demo_samples.py --refresh   # re-download everything

Requires network on first run. The seeder itself never hits the network.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "docs" / "mock-data" / "samples"

# Curated set of picsum.photos IDs. These are stable — picsum.photos
# guarantees the same ID always returns the same source photo.
_PICSUM_IDS: tuple[int, ...] = (
    1, 10, 20, 25, 33, 45, 60, 65, 80, 91,
    100, 111, 120, 129, 152, 164, 180, 200, 220, 237,
    250, 260, 290, 305, 316, 342, 350, 370, 401, 428,
    445, 460, 501, 520, 555, 600, 634, 659, 700, 750,
)

_IMG_SIZE: tuple[int, int] = (1024, 1024)
_URL_TEMPLATE: str = "https://picsum.photos/id/{pid}/{w}/{h}"


def _fetch_one(pid: int, dest: Path, force: bool) -> tuple[str, int]:
    """Download one photo. Returns ``(status, size_bytes)`` for reporting."""
    if dest.exists() and not force:
        return ("skipped", dest.stat().st_size)
    url = _URL_TEMPLATE.format(pid=pid, w=_IMG_SIZE[0], h=_IMG_SIZE[1])
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return ("fetched", dest.stat().st_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download every file, even ones already cached.")
    args = parser.parse_args(argv)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    fetched = 0
    skipped = 0
    total_bytes = 0
    for i, pid in enumerate(_PICSUM_IDS):
        dest = SAMPLES_DIR / f"photo_{i:03d}.jpg"
        try:
            status, size = _fetch_one(pid, dest, force=args.refresh)
        except Exception as exc:  # noqa: BLE001
            print(f"  photo_{i:03d}.jpg (picsum id={pid}): FAILED — {exc}", file=sys.stderr)
            continue
        total_bytes += size
        if status == "fetched":
            fetched += 1
            print(f"  photo_{i:03d}.jpg: {size // 1024} KB (id={pid})")
        else:
            skipped += 1

    print()
    print(f"Cache: {SAMPLES_DIR}")
    print(f"Fetched: {fetched} · Skipped (already cached): {skipped}")
    print(f"Total: {total_bytes // 1024} KB across {fetched + skipped} files")
    if fetched + skipped == 0:
        print("[error] no samples cached — cannot seed a demo dataset without them.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
