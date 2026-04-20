#!/usr/bin/env python3
"""
queue_manager.py - Shared queue management with source-based organization

All scrapers use this to:
1. Save images to source-specific subdirectories
2. Vision worker reads from source directories in round-robin
3. Ensures fair processing across all sources
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from paths import base_dir

load_dotenv()

BASE_DIR = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
TOPIC = os.environ.get("PIPELINE_TOPIC", "Realistic Female Influencer")
SLUG = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
# PIPELINE_QUEUE may be either ".../queue" (root) or ".../queue/<slug>"; normalise to include slug exactly once.
BASE_QUEUE_DIR = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG

# Source directory names
SOURCES = {
    "civitai": "civitai",
    "civitai_red": "civitai_red",
    "zforfree": "zforfree",
    "zforfree_web": "zforfree_web",
    "discord_ud": "discord_ud",
    "discord_mj": "discord_mj",
    "reddit": "reddit",
    "twitter_x": "twitter_x",
    "nanobanana": "nanobanana",
    "unknown": "unknown",
}

import re as _re

_SAFE_SOURCE = _re.compile(r"^[a-z0-9_]+$")


def get_queue_dir(source: str):
    """Get or create the queue directory for a source.

    Known sources (see SOURCES) keep their canonical folder name. Unknown but
    safe identifiers (lowercase alphanumerics + underscore, e.g. an admin-
    configured LOCAL_IMPORT_NAME) are allowed through as-is so the local-folder
    feeder and any future sources don't collapse into `unknown/`.
    """
    source_key = source.lower().replace(" ", "_").replace("-", "_")
    if source_key in SOURCES:
        dir_name = SOURCES[source_key]
    elif _SAFE_SOURCE.match(source_key):
        dir_name = source_key
    else:
        dir_name = "unknown"

    queue_dir = BASE_QUEUE_DIR / dir_name
    queue_dir.mkdir(parents=True, exist_ok=True)
    return queue_dir

def save_to_queue(source: str, image_path: Path, prompt_text: str = "", metadata: dict = None):
    """Save image and metadata to source-specific queue"""
    import json
    import shutil

    queue_dir = get_queue_dir(source)
    
    try:
        # Validate temp file exists and has content
        if not image_path.exists():
            print(f"[QUEUE-ERROR] Temp file does not exist: {image_path}")
            return None
        
        size_before = image_path.stat().st_size
        if size_before < 5000:
            print(f"[QUEUE-ERROR] Temp file too small ({size_before} bytes): {image_path}")
            image_path.unlink(missing_ok=True)
            return None
        
        # Move image (handles cross-drive better than copy2 on Windows)
        dest_img = queue_dir / image_path.name
        try:
            # Try move first (atomic on same drive, fast on different drives)
            shutil.move(str(image_path), str(dest_img))
            print(f"[QUEUE] Moved {image_path.name} to {source}/ ({size_before} bytes)")
        except Exception as move_err:
            # Fallback to copy if move fails
            print(f"[QUEUE] Move failed, falling back to copy: {move_err}")
            shutil.copy2(image_path, dest_img)
            image_path.unlink(missing_ok=True)
            print(f"[QUEUE] Copied {image_path.name} to {source}/ ({size_before} bytes)")
        
        # Verify destination file
        if not dest_img.exists():
            print(f"[QUEUE-ERROR] File failed to reach destination: {dest_img}")
            return None
        
        dest_size = dest_img.stat().st_size
        if dest_size < size_before * 0.95:  # Allow small compression
            print(f"[QUEUE-ERROR] File corrupted in transfer ({dest_size} != {size_before}): {dest_img}")
            dest_img.unlink(missing_ok=True)
            return None
        
        # Save prompt
        dest_txt = queue_dir / f"{dest_img.stem}.txt"
        if prompt_text:
            dest_txt.write_text(prompt_text, encoding='utf-8')
            print(f"[QUEUE] Saved prompt: {dest_txt.name} ({len(prompt_text)} chars)")
        
        # Save metadata
        if metadata:
            dest_meta = queue_dir / f"{dest_img.stem}.meta.json"
            dest_meta.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print(f"[QUEUE] Saved metadata: {dest_meta.name}")
        
        print(f"[QUEUE-SUCCESS] {dest_img.name} ({dest_size} bytes) -> {source}/")
        return dest_img
    except Exception as e:
        print(f"[QUEUE-ERROR] Unexpected error saving to queue ({source}): {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp file if it still exists
        image_path.unlink(missing_ok=True)
        return None

_QUEUE_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def _discover_source_dirs() -> list[Path]:
    """Every immediate subdirectory of BASE_QUEUE_DIR is a potential source.

    Using the filesystem as the source of truth means admin-configured labels
    (e.g. LOCAL_IMPORT_NAME=my_dataset) are picked up automatically, without
    needing to be registered in SOURCES. Names that don't match our safe-id
    pattern are ignored so stray files can't confuse the worker.
    """
    if not BASE_QUEUE_DIR.exists():
        return []
    dirs: list[Path] = []
    for child in BASE_QUEUE_DIR.iterdir():
        if child.is_dir() and _SAFE_SOURCE.match(child.name):
            dirs.append(child)
    return dirs


def _images_in(source_dir: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in _QUEUE_IMAGE_GLOBS:
        images.extend(source_dir.glob(pattern))
    return sorted(images)


def list_queue_sources():
    """List every source directory that currently contains images."""
    counts: dict[str, int] = {}
    for source_dir in _discover_source_dirs():
        imgs = _images_in(source_dir)
        if imgs:
            counts[source_dir.name] = len(imgs)
    return counts


def get_next_image_round_robin():
    """Return (source_name, image_path) cycling fairly across every non-empty source.

    Discovers source folders from the filesystem so LOCAL_IMPORT_NAME and any
    future sources participate automatically.
    """
    source_images: dict[str, list[Path]] = {}
    for source_dir in _discover_source_dirs():
        imgs = _images_in(source_dir)
        if imgs:
            source_images[source_dir.name] = imgs

    if not source_images:
        return None, None

    # Round-robin across sources, oldest-first within each source.
    all_images: list[tuple[str, Path]] = []
    max_len = max(len(v) for v in source_images.values())
    for i in range(max_len):
        for source in sorted(source_images.keys()):
            if i < len(source_images[source]):
                all_images.append((source, source_images[source][i]))

    source, img_path = all_images[0]
    return source, img_path
