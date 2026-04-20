#!/usr/bin/env python3
"""
LOCAL FALLBACK VISION WORKER - WINDOWS-SAFE LOCKING
- Uses simple .processing flag files to prevent race conditions
- Only ONE worker processes each file at a time
"""
import os
import sys
import json
import io
import time
from pathlib import Path
from PIL import Image
import numpy as np

BASE_DIR = Path("I:/AI/openclaw/workspace/prompt-library")
SLUG = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
QUEUE_DIR = BASE_DIR / "queue" / SLUG
SORTED_DIR = BASE_DIR / "sorted" / SLUG

CATEGORIES = ["InstagramInfluencer", "Professional", "Cinematic", "Vintage", "Fantasy", "Sports", "Amateur", "Unknown", "DISCARD"]
for cat in CATEGORIES:
    (SORTED_DIR / cat).mkdir(parents=True, exist_ok=True)

def log_op(msg, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)

def detect_source(filename):
    filename_lower = filename.lower()
    if "civitai_" in filename_lower:
        return "civitai"
    elif "zff_" in filename_lower:
        return "zforfree"
    elif "reddit_" in filename_lower:
        return "reddit"
    elif "twitter" in filename_lower or "x_" in filename_lower:
        return "twitter_x"
    elif "nanobanana" in filename_lower:
        return "nanobanana"
    elif "discord" in filename_lower or "ud_" in filename_lower:
        return "discord_ud"
    return "unknown"

def classify_image_local(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        
        quality = 5
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        if brightness > 50 and brightness < 200:
            quality += 2
        if contrast > 30:
            quality += 1
        quality = min(10, max(1, quality))
        
        category = "Professional" if quality >= 7 else "Unknown" if quality >= 5 else "DISCARD"
        
        return {
            "photorealistic": True,
            "ai_flaws": False,
            "woman_present": True,
            "nsfw": False,
            "quality": quality,
            "style": "portrait",
            "category": category,
            "source": detect_source(""),
            "api_used": "local_fallback",
            "reason": "Local analysis"
        }, category
    except Exception as e:
        log_op(f"Analysis failed: {str(e)}", "WARN")
        return None, "DISCARD"

def try_acquire_lock(filepath):
    """Create .processing flag to claim ownership"""
    lock_path = filepath.parent / f"{filepath.stem}.processing"
    if lock_path.exists():
        return False  # Another worker has it
    
    try:
        lock_path.write_text(str(os.getpid()))
        return True
    except:
        return False

def release_lock(filepath):
    lock_path = filepath.parent / f"{filepath.stem}.processing"
    try:
        lock_path.unlink(missing_ok=True)
    except:
        pass

def get_next_image():
    sources = [d for d in QUEUE_DIR.iterdir() if d.is_dir()]
    if not sources:
        return None
    
    for source_dir in sorted(sources):
        all_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        real_images = [f for f in all_files if ".meta" not in f.name and not (f.parent / f"{f.stem}.processing").exists()]
        if real_images:
            return real_images[0]
    
    return None

def process_queue():
    print("=" * 70)
    print("LOCAL FALLBACK VISION WORKER - WINDOWS SAFE")
    print("=" * 70)
    print(f"Watching: {QUEUE_DIR}")
    print("")
    
    processed_count = 0
    
    while True:
        image_path = get_next_image()
        
        if image_path is None:
            time.sleep(5)
            continue
        
        # Try to acquire lock
        if not try_acquire_lock(image_path):
            continue
        
        try:
            stem = image_path.stem
            queue_parent = image_path.parent
            
            log_op(f"Processing: {image_path.name}")
            
            img_bytes = image_path.read_bytes()
            if not img_bytes:
                log_op(f"Empty file", "WARN")
                image_path.unlink(missing_ok=True)
                continue
            
            result_data, category = classify_image_local(img_bytes)
            if result_data is None:
                log_op(f"Classification failed", "WARN")
                image_path.unlink(missing_ok=True)
                continue
            
            source = detect_source(image_path.name)
            dest_dir = SORTED_DIR / category / source
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Move image
            dest_img = dest_dir / image_path.name
            image_path.rename(dest_img)
            log_op(f"Moved: {image_path.name} -> {category}/{source}")
            
            # Move metadata
            for ext in [".txt", ".meta.json"]:
                src = queue_parent / f"{stem}{ext}"
                if src.exists():
                    try:
                        src.rename(dest_dir / src.name)
                    except:
                        pass
            
            # Create vision.json
            (dest_dir / f"{stem}.vision.json").write_text(json.dumps(result_data, indent=2))
            
            processed_count += 1
            if processed_count % 20 == 0:
                log_op(f"PROGRESS: {processed_count} processed")
        
        except Exception as e:
            log_op(f"Error: {str(e)}", "ERROR")
        
        finally:
            release_lock(image_path)

if __name__ == "__main__":
    try:
        process_queue()
    except KeyboardInterrupt:
        print("\nShutdown.")
