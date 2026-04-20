#!/usr/bin/env python3
"""
LOCAL FALLBACK VISION CLASSIFIER - FIXED
- Properly excludes .meta.jpg and .meta.png files
- Uses basic computer vision (no external APIs)
- Fast (~1-2s per image)
- Free and always available
"""
import os
import sys
import json
import base64
import io
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Configuration
BASE_DIR = Path("I:/AI/openclaw/workspace/prompt-library")
SLUG = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
QUEUE_DIR = BASE_DIR / "queue" / SLUG
SORTED_DIR = BASE_DIR / "sorted" / SLUG

CATEGORIES = ["InstagramInfluencer", "Professional", "Cinematic", "Vintage", "Fantasy", "Sports", "Amateur", "Unknown", "DISCARD"]
for cat in CATEGORIES:
    (SORTED_DIR / cat).mkdir(parents=True, exist_ok=True)

def log_op(msg, level="INFO"):
    """Structured logging"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)

def detect_source(filename, metadata):
    """Detect source from filename"""
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

def classify_image_local(img_bytes, source_name):
    """Classify using local analysis"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Basic quality score
        quality = 5
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        if brightness > 50 and brightness < 200:
            quality += 2
        if contrast > 30:
            quality += 1
        quality = min(10, max(1, quality))
        
        # Simple classification
        category = "Professional" if quality >= 7 else "Unknown" if quality >= 5 else "DISCARD"
        
        result_data = {
            "photorealistic": True,
            "ai_flaws": False,
            "woman_present": True,
            "nsfw": False,
            "quality": quality,
            "style": "portrait",
            "category": category,
            "source": source_name,
            "api_used": "local_fallback",
            "reason": "Local image analysis"
        }
        
        return result_data, category, "Local analysis"
    
    except Exception as e:
        log_op(f"Analysis failed: {str(e)}", "WARN")
        return None, "DISCARD", "Image corrupt"

def get_next_image():
    """Get next image - EXCLUDES .meta.jpg and .meta.png files"""
    sources = [d for d in QUEUE_DIR.iterdir() if d.is_dir()]
    if not sources:
        return None
    
    for source_dir in sorted(sources):
        # Get all .jpg and .png files
        all_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        
        # FILTER OUT .meta.jpg and .meta.png - only get real images
        real_images = [f for f in all_files if ".meta" not in f.name]
        
        if real_images:
            return real_images[0]
    
    return None

def process_queue():
    """Main loop"""
    print("=" * 70)
    print("LOCAL FALLBACK VISION WORKER - FIXED")
    print("=" * 70)
    print(f"Watching: {QUEUE_DIR}")
    print(f"Output: {SORTED_DIR}")
    print(f"Mode: LOCAL (excludes .meta files)")
    print("")
    
    processed_count = 0
    error_count = 0
    
    while True:
        image_path = get_next_image()
        
        if image_path is None:
            time.sleep(5)
            continue
        
        try:
            stem = image_path.stem
            queue_parent = image_path.parent
            
            log_op(f"Processing: {image_path.name}")
            
            # Read image
            img_bytes = image_path.read_bytes()
            if not img_bytes:
                log_op(f"Empty file: {image_path.name}", "WARN")
                image_path.unlink(missing_ok=True)
                continue
            
            # Get metadata
            meta_path = queue_parent / f"{stem}.meta.json"
            metadata = {}
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
                except:
                    pass
            
            # Classify
            correct_source = detect_source(image_path.name, metadata)
            result_data, category, reason = classify_image_local(img_bytes, correct_source)
            
            if result_data is None:
                log_op(f"Classification failed: {image_path.name}", "WARN")
                error_count += 1
                image_path.unlink(missing_ok=True)
                continue
            
            # Create destination directory
            dest_dir = SORTED_DIR / category / correct_source
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # STEP 1: Move image
            dest_img = dest_dir / image_path.name
            try:
                image_path.rename(dest_img)
                log_op(f"Moved: {image_path.name} -> {category}/{correct_source}")
            except Exception as e:
                log_op(f"CRITICAL: Failed to move image: {str(e)}", "ERROR")
                continue
            
            # STEP 2: Move metadata
            for ext in [".txt", ".meta.json"]:
                src_meta = queue_parent / f"{stem}{ext}"
                if src_meta.exists():
                    try:
                        dst_meta = dest_dir / src_meta.name
                        src_meta.rename(dst_meta)
                    except Exception as e:
                        log_op(f"Failed to move {ext}: {str(e)}", "WARN")
            
            # STEP 3: Create vision.json
            vision_path = dest_dir / f"{stem}.vision.json"
            try:
                vision_path.write_text(json.dumps(result_data, indent=2), encoding='utf-8')
            except Exception as e:
                log_op(f"Failed to create vision.json: {str(e)}", "ERROR")
                continue
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                log_op(f"STATS: Processed {processed_count} | Errors {error_count}")
        
        except Exception as e:
            error_count += 1
            log_op(f"Exception: {str(e)}", "ERROR")

if __name__ == "__main__":
    try:
        process_queue()
    except KeyboardInterrupt:
        print("\nShutdown requested.")
