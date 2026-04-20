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

def list_queue_sources():
    """List all source directories and their image counts"""
    counts = {}
    for source, dir_name in SOURCES.items():
        source_dir = BASE_QUEUE_DIR / dir_name
        if source_dir.exists():
            images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
            if images:
                counts[source] = len(images)
    return counts

def get_next_image_round_robin():
    """
    Get next image in round-robin fashion across all sources
    This ensures fair processing: one from civitai, one from zforfree, etc.
    """
    all_images = []
    
    # Collect images from all sources
    source_images = {}
    for source, dir_name in SOURCES.items():
        source_dir = BASE_QUEUE_DIR / dir_name
        if source_dir.exists():
            images = sorted(list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")))
            if images:
                source_images[source] = images
    
    if not source_images:
        return None, None
    
    # Build round-robin list
    max_len = max(len(v) for v in source_images.values())
    for i in range(max_len):
        for source in sorted(source_images.keys()):
            if i < len(source_images[source]):
                all_images.append((source, source_images[source][i]))
    
    if all_images:
        source, img_path = all_images[0]
        return source, img_path
    
    return None, None
