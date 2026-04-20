#!/usr/bin/env python3
"""
VISION WORKER - REMOTE LMSTUDIO (100.75.25.43:1234)
"""
import os
import sys
import json
import io
import time
import requests
import base64
from pathlib import Path

try:
    from PIL import Image
except Exception as e:
    print(f"[FATAL] PIL failed: {e}")
    sys.exit(1)

BASE_DIR = Path("I:/AI/openclaw/workspace/prompt-library")
SLUG = "realistic_female_influencer"
QUEUE_DIR = BASE_DIR / "queue" / SLUG
SORTED_DIR = BASE_DIR / "sorted" / SLUG

# REMOTE LMStudio (the correct one!)
LMSTUDIO_URL = "http://100.75.25.43:1234/v1"

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

def classify_image_lmstudio(img_bytes):
    """Classify image using REMOTE LMStudio"""
    try:
        # Convert to base64
        b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt = """Analyze this image of a person. Respond with ONLY valid JSON:
{"photorealistic": true/false, "ai_flaws": true/false, "woman_present": true/false, "nsfw": true/false, "quality": 1-10, "style": "portrait|selfie|fashion|editorial|cinematic|vintage|fantasy|sports|other", "reason": "One sentence"}"""
        
        response = requests.post(
            f"{LMSTUDIO_URL}/chat/completions",
            json={
                "model": "local-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 300
            },
            timeout=60
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            
            # Determine category
            quality = result.get("quality", 5)
            woman_present = result.get("woman_present", False)
            ai_flaws = result.get("ai_flaws", False)
            style = result.get("style", "other").lower()
            
            if not woman_present or ai_flaws or quality <= 2:
                category = "DISCARD"
            elif style in ["selfie", "portrait"] and quality >= 6:
                category = "InstagramInfluencer"
            elif style == "editorial" and quality >= 6:
                category = "Professional"
            else:
                category = "Unknown"
            
            return result, category
        else:
            log_op(f"LMStudio error {response.status_code}: {response.text[:100]}", "WARN")
            return None, "DISCARD"
    
    except Exception as e:
        log_op(f"LMStudio classification error: {str(e)}", "WARN")
        return None, "DISCARD"

def process_queue():
    """Main processing loop"""
    log_op(f"=== VISION WORKER STARTED (Remote LMStudio: {LMSTUDIO_URL}) ===")
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
                
                # Classify with REMOTE LMStudio
                result_data, category = classify_image_lmstudio(img_bytes)
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
                log_op(f"Moved: {image_path.name} -> {category}/{source} (quality: {result_data.get('quality', '?')})")
                
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
                if processed % 20 == 0:
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
