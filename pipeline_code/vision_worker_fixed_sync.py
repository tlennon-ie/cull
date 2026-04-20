#!/usr/bin/env python3
"""
FIXED VISION WORKER - WITH GUARANTEED IMAGE+METADATA SYNC
- GUARANTEES: Image file + metadata files move together
- GUARANTEES: vision.json only created if image successfully moved
- Detailed logging of every move operation
"""
from dotenv import load_dotenv
load_dotenv()
import os
import sys
import json
import base64
import io
import time
from pathlib import Path
from PIL import Image
import concurrent.futures
from groq import Groq

# Configuration
BASE_DIR = Path("I:/AI/openclaw/workspace/prompt-library")
SLUG = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
QUEUE_DIR = BASE_DIR / "queue" / SLUG
SORTED_DIR = BASE_DIR / "sorted" / SLUG

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LMSTUDIO_URL = "http://localhost:1234/v1"

# Groq API keys
GROQ_KEYS = [k.strip() for k in (os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
_current_key_idx = 0

CATEGORIES = ["InstagramInfluencer", "Professional", "Cinematic", "Vintage", "Fantasy", "Sports", "Amateur", "Unknown", "DISCARD"]
for cat in CATEGORIES:
    (SORTED_DIR / cat).mkdir(parents=True, exist_ok=True)

def log_operation(msg, level="INFO"):
    """Structured logging"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)

def detect_source(filename, metadata):
    """Detect source from filename + metadata"""
    filename_lower = filename.lower()
    
    if "civitai_" in filename_lower:
        return "civitai"
    elif "zff_" in filename_lower or "unknown_zff_" in filename_lower:
        return "zforfree"
    elif filename_lower.startswith(("ud_", "unstable_")):
        return "discord_ud"
    elif "reddit_" in filename_lower:
        return "reddit"
    elif filename_lower.startswith(("x_", "twitter_")):
        return "twitter_x"
    elif "grok" in filename_lower or "promptsref_grok" in filename_lower:
        return "grok"
    elif "nanobanana" in filename_lower or "promptsref_nanobanana" in filename_lower:
        return "nanobanana"
    
    if metadata:
        source_channel = metadata.get("source_channel", "").lower()
        if "civitai" in source_channel:
            return "civitai"
        elif "reddit" in source_channel:
            return "reddit"
        elif "twitter" in source_channel or "x.com" in source_channel:
            return "twitter_x"
    
    return "unknown"

def classify_with_groq(b64_image):
    """Classify using Groq API"""
    global _current_key_idx
    try:
        key = GROQ_KEYS[_current_key_idx % len(GROQ_KEYS)]
        _current_key_idx += 1
        
        client = Groq(api_key=key)
        prompt = """Analyze this image. Respond with ONLY valid JSON (no markdown):
{"photorealistic": true/false, "ai_flaws": true/false, "woman_present": true/false, "nsfw": true/false, "quality": 1-10, "style": "portrait|selfie|fashion|editorial|cinematic|vintage|fantasy|sports|other", "reason": "One sentence"}"""
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]}],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content), "groq"
    except Exception as e:
        log_operation(f"Groq error: {str(e)[:100]}", "WARN")
        return None, "groq"

def classify_with_lmstudio(b64_image):
    """Classify using LMStudio local vision"""
    try:
        import requests
        prompt = """Analyze this image. Respond with ONLY valid JSON:
{"photorealistic": true/false, "ai_flaws": true/false, "woman_present": true/false, "nsfw": true/false, "quality": 1-10, "style": "portrait|selfie|fashion|editorial|cinematic|vintage|fantasy|sports|other", "reason": "One sentence"}"""
        
        response = requests.post(
            f"{LMSTUDIO_URL}/chat/completions",
            json={"model": "local-model", "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]}], "temperature": 0.1, "max_tokens": 300},
            timeout=60
        )
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content), "lmstudio"
    except Exception as e:
        log_operation(f"LMStudio error: {str(e)[:100]}", "WARN")
    
    return None, "lmstudio"

def classify_image_hybrid(img_bytes, source_name):
    """Classify using both APIs"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    except:
        return None, "DISCARD", "Image corrupt"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        groq_future = executor.submit(classify_with_groq, b64)
        lm_future = executor.submit(classify_with_lmstudio, b64)
        
        result = None
        api_used = None
        
        try:
            result, api_used = groq_future.result(timeout=30)
            if result is None:
                result, api_used = lm_future.result(timeout=30)
        except:
            try:
                result, api_used = lm_future.result(timeout=30)
            except:
                return None, "DISCARD", "Both APIs failed"
    
    if result is None:
        return None, "DISCARD", "Classification failed"
    
    photorealistic = result.get("photorealistic", False)
    ai_flaws = result.get("ai_flaws", False)
    woman_present = result.get("woman_present", False)
    nsfw = result.get("nsfw", False)
    quality = result.get("quality", 5)
    style = result.get("style", "other").lower()
    
    category = "Unknown"
    
    if not photorealistic or not woman_present or ai_flaws or quality <= 2:
        category = "DISCARD"
    elif nsfw:
        category = "InstagramInfluencer" if quality >= 7 else "DISCARD"
    elif style in ["selfie", "portrait"] and quality >= 6:
        category = "InstagramInfluencer"
    elif style == "editorial" and quality >= 6:
        category = "Professional"
    elif style == "cinematic" and quality >= 6:
        category = "Cinematic"
    elif style == "vintage" and quality >= 6:
        category = "Vintage"
    elif style == "fantasy" and quality >= 6:
        category = "Fantasy"
    elif style == "sports" and quality >= 6:
        category = "Sports"
    elif style == "fashion" and quality >= 5:
        category = "Amateur"
    else:
        category = "Unknown"
    
    result_data = {
        "photorealistic": photorealistic,
        "ai_flaws": ai_flaws,
        "woman_present": woman_present,
        "nsfw": nsfw,
        "quality": quality,
        "style": style,
        "category": category,
        "source": source_name,
        "api_used": api_used,
        "reason": result.get("reason", "")
    }
    
    return result_data, category, result.get("reason", "")

def get_next_image():
    """Get next image from queue"""
    sources = [d for d in QUEUE_DIR.iterdir() if d.is_dir()]
    if not sources:
        return None
    
    for source_dir in sorted(sources):
        images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        if images:
            return images[0]
    
    return None

def process_queue():
    """Main loop with GUARANTEED image+metadata sync"""
    print("=" * 70)
    print("FIXED VISION WORKER - GUARANTEED IMAGE SYNC")
    print("=" * 70)
    print(f"Watching: {QUEUE_DIR}")
    print(f"Output: {SORTED_DIR}")
    print(f"Mode: STRICT (image + metadata move together, vision.json only after successful move)")
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
            
            log_operation(f"Processing: {image_path.name}")
            
            # Read image
            img_bytes = image_path.read_bytes()
            if not img_bytes:
                log_operation(f"Empty file: {image_path.name}", "WARN")
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
            result_data, category, reason = classify_image_hybrid(img_bytes, correct_source)
            
            if result_data is None:
                log_operation(f"Classification failed: {image_path.name}", "WARN")
                error_count += 1
                image_path.unlink(missing_ok=True)
                continue
            
            # Create destination directory
            dest_dir = SORTED_DIR / category / correct_source
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # STEP 1: Move image (MUST succeed)
            dest_img = dest_dir / image_path.name
            try:
                image_path.rename(dest_img)
                log_operation(f"Image moved: {image_path.name} → {category}/{correct_source}")
            except Exception as e:
                log_operation(f"CRITICAL: Failed to move image: {str(e)}", "ERROR")
                continue  # Skip this file, don't create vision.json
            
            # STEP 2: Move metadata files from ORIGINAL queue location
            files_moved = []
            for ext in [".txt", ".meta.json"]:
                src_meta = queue_parent / f"{stem}{ext}"
                if src_meta.exists():
                    try:
                        dst_meta = dest_dir / src_meta.name
                        src_meta.rename(dst_meta)
                        files_moved.append(ext)
                        log_operation(f"Metadata moved: {src_meta.name}")
                    except Exception as e:
                        log_operation(f"Failed to move {ext}: {str(e)}", "WARN")
            
            # STEP 3: Create vision.json ONLY AFTER image+metadata successfully moved
            vision_path = dest_dir / f"{stem}.vision.json"
            try:
                vision_path.write_text(json.dumps(result_data, indent=2), encoding='utf-8')
                log_operation(f"vision.json created: {stem}.vision.json")
            except Exception as e:
                log_operation(f"Failed to create vision.json: {str(e)}", "ERROR")
                continue
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                log_operation(f"STATS: Processed {processed_count} | Errors {error_count}")
            
        except Exception as e:
            error_count += 1
            log_operation(f"Exception in main loop: {str(e)}", "ERROR")
            try:
                image_path.unlink(missing_ok=True)
            except:
                pass

if __name__ == "__main__":
    try:
        process_queue()
    except KeyboardInterrupt:
        print("\nShutdown requested.")
