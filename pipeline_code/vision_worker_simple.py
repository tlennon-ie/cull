#!/usr/bin/env python3
"""
VISION WORKER - SIMPLIFIED
"""
import os
import sys
import json
import io
import time
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
from paths import base_dir
    print("[OK] Dependencies loaded")
except Exception as e:
    print(f"[ERROR] Dependencies failed: {e}")
    sys.exit(1)

BASE_DIR = Path(str(base_dir()))
SLUG = "realistic_female_influencer"
QUEUE_DIR = BASE_DIR / "queue" / SLUG
SORTED_DIR = BASE_DIR / "sorted" / SLUG

CATEGORIES = ["InstagramInfluencer", "Professional", "Unknown", "DISCARD"]
for cat in CATEGORIES:
    (SORTED_DIR / cat).mkdir(parents=True, exist_ok=True)

def log_op(msg, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)

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

def classify_image(img_bytes):
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
        
        return {"quality": quality, "category": category}, category
    except:
        return None, "DISCARD"

def process_queue():
    log_op("=== STARTED ===")
    processed = 0
    
    while True:
        image_path = get_next_image()
        if image_path is None:
            time.sleep(5)
            continue
        
        lock_path = image_path.parent / f"{image_path.stem}.processing"
        if lock_path.exists():
            continue
        
        try:
            lock_path.write_text(str(os.getpid()))
            
            stem = image_path.stem
            queue_parent = image_path.parent
            
            img_bytes = image_path.read_bytes()
            if not img_bytes:
                image_path.unlink(missing_ok=True)
                continue
            
            result_data, category = classify_image(img_bytes)
            if result_data is None:
                image_path.unlink(missing_ok=True)
                continue
            
            source = "unknown"
            if "reddit" in image_path.parent.name:
                source = "reddit"
            elif "twitter" in image_path.parent.name or "x_" in image_path.name:
                source = "twitter_x"
            
            dest_dir = SORTED_DIR / category / source
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            dest_img = dest_dir / image_path.name
            image_path.rename(dest_img)
            
            for ext in [".txt", ".meta.json"]:
                src = queue_parent / f"{stem}{ext}"
                if src.exists():
                    try:
                        src.rename(dest_dir / src.name)
                    except:
                        pass
            
            vision_path = dest_dir / f"{stem}.vision.json"
            vision_path.write_text(json.dumps(result_data, indent=2))
            
            processed += 1
            if processed % 20 == 0:
                log_op(f"Progress: {processed}")
        
        except Exception as e:
            log_op(f"Error: {str(e)}", "WARN")
        
        finally:
            lock_path.unlink(missing_ok=True)

if __name__ == "__main__":
    try:
        process_queue()
    except KeyboardInterrupt:
        log_op("Shutdown")
