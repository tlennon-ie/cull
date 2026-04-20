#!/usr/bin/env python3
"""
VISION WORKER - PRODUCTION GRADE
Bulletproof with extensive error handling
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
except Exception as e:
    print(f"[FATAL] Dependencies failed: {e}")
    sys.exit(1)

BASE_DIR = Path(str(base_dir()))
SLUG = "realistic_female_influencer"
QUEUE_DIR = BASE_DIR / "queue" / SLUG
SORTED_DIR = BASE_DIR / "sorted" / SLUG

# Ensure directories exist
QUEUE_DIR.mkdir(parents=True, exist_ok=True)
SORTED_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["InstagramInfluencer", "Professional", "Unknown", "DISCARD"]
for cat in CATEGORIES:
    (SORTED_DIR / cat).mkdir(parents=True, exist_ok=True)

def log_op(msg, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)

def get_next_image():
    """Find next unprocessed image"""
    try:
        if not QUEUE_DIR.exists():
            return None
        
        sources = [d for d in QUEUE_DIR.iterdir() if d.is_dir()]
        if not sources:
            return None
        
        for source_dir in sorted(sources):
            try:
                all_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
                real_images = [
                    f for f in all_files 
                    if ".meta" not in f.name and ".processing" not in f.name
                ]
                if real_images:
                    return real_images[0]
            except Exception as e:
                log_op(f"Error scanning {source_dir.name}: {str(e)}", "WARN")
                continue
        
        return None
    except Exception as e:
        log_op(f"get_next_image error: {str(e)}", "ERROR")
        return None

def classify_image(img_bytes):
    """Classify image locally"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        
        quality = 5
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        if 50 < brightness < 200:
            quality += 2
        if contrast > 30:
            quality += 1
        quality = min(10, max(1, quality))
        
        category = "Professional" if quality >= 7 else "Unknown" if quality >= 5 else "DISCARD"
        
        return {"quality": quality, "category": category}, category
    except Exception as e:
        log_op(f"Classification error: {str(e)}", "WARN")
        return None, "DISCARD"

def process_queue():
    """Main processing loop"""
    log_op("=== VISION WORKER STARTED ===")
    processed = 0
    errors = 0
    
    while True:
        try:
            image_path = get_next_image()
            if image_path is None:
                time.sleep(3)
                continue
            
            # Create lock
            lock_path = image_path.parent / f"{image_path.stem}.processing"
            if lock_path.exists():
                continue
            
            try:
                lock_path.write_text(str(os.getpid()))
            except:
                continue
            
            try:
                stem = image_path.stem
                queue_parent = image_path.parent
                
                log_op(f"Processing: {image_path.name}")
                
                # Read image
                img_bytes = image_path.read_bytes()
                if not img_bytes:
                    image_path.unlink(missing_ok=True)
                    lock_path.unlink(missing_ok=True)
                    continue
                
                # Classify
                result_data, category = classify_image(img_bytes)
                if result_data is None:
                    image_path.unlink(missing_ok=True)
                    lock_path.unlink(missing_ok=True)
                    errors += 1
                    continue
                
                # Detect source
                source = "unknown"
                if "reddit" in queue_parent.name:
                    source = "reddit"
                elif "twitter" in queue_parent.name or "x_" in image_path.name.lower():
                    source = "twitter_x"
                elif "civitai" in queue_parent.name:
                    source = "civitai"
                
                # Create destination
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
                vision_path = dest_dir / f"{stem}.vision.json"
                vision_path.write_text(json.dumps(result_data, indent=2))
                
                processed += 1
                if processed % 50 == 0:
                    log_op(f"Progress: {processed} processed, {errors} errors")
            
            finally:
                lock_path.unlink(missing_ok=True)
        
        except KeyboardInterrupt:
            log_op("Shutdown requested")
            break
        except Exception as e:
            errors += 1
            log_op(f"Main loop error: {str(e)}", "ERROR")
            time.sleep(1)

if __name__ == "__main__":
    try:
        process_queue()
    except Exception as e:
        log_op(f"Fatal: {str(e)}", "FATAL")
        sys.exit(1)
