"""
vision_worker_gemini.py - DISABLED
Google Cloud Project suspended due to NSFW content policy.
Use vision_worker_groq.py instead.
"""
import os, re, json, time, base64, io, shutil, threading, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import google.generativeai as genai
from PIL import Image

# Config
from dotenv import load_dotenv
from paths import base_dir
load_dotenv()

BASE_DIR   = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
SLUG       = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
_RAW_SORTED = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted")))
QUEUE_DIR  = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG
SORTED_DIR = _RAW_SORTED if _RAW_SORTED.name == SLUG else _RAW_SORTED / SLUG

API_KEY    = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env")
MODEL_ID   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
PARALLEL_WORKERS = 50   # Paid tier: 2000 RPM allowed
POLL_INTERVAL    = 0.5

_fs_lock = threading.Lock()

CATEGORIES = ["InstagramInfluencer", "NSFW", "Professional", "Amateur", "Unknown"]
for c in CATEGORIES:
    (SORTED_DIR / c).mkdir(parents=True, exist_ok=True)

def resize_bytes(data, max_size=4*1024*1024): 
    try:
        if not data: return None
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
        
        buffer_before = io.BytesIO()
        img.save(buffer_before, format="JPEG", quality=80) 
        current_size = buffer_before.tell()

        if current_size < max_size:
            return data 

        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
        
        buffer_after = io.BytesIO()
        img.save(buffer_after, format="JPEG", quality=70) 
        final_size = buffer_after.tell()
        
        if final_size < max_size:
            return buffer_after.getvalue()
        else:
            w, h = img.size
            scale = max_size / final_size * 0.8 
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            buffer_final = io.BytesIO()
            img.save(buffer_final, format="JPEG", quality=60) 
            return buffer_final.getvalue()

    except Exception as e:
        return None

def vision_classify(image_path):
    processing_path = Path(str(image_path) + ".processing")
    try:
        image_path.rename(processing_path)
    except FileNotFoundError:
        return "SKIPPED" 
    except Exception as e:
        print(f"  [SHUTIL ERR] Failed to rename {image_path} to {processing_path}: {e}", flush=True)
        return "SKIPPED"

    stem      = processing_path.stem
    txt_path  = processing_path.parent / f"{stem}.txt"
    meta_path = processing_path.parent / f"{stem}.meta.json"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        prompt_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
    except FileNotFoundError:
        return "SKIPPED" 

    msg_id_key = meta.get("message_id", stem)

    try:
        img_bytes = processing_path.read_bytes()
        if not img_bytes: return {"category": "SKIPPED"}
        
        resized_bytes = resize_bytes(img_bytes)
        if not resized_bytes:
            return {"category": "DISCARD", "reason": "Failed to resize image"}

        b64_image = base64.b64encode(resized_bytes).decode('utf-8')

        prompt_text = (
            "Analyze this image. Respond ONLY with valid JSON.\n"
            "{\n"
            '  "photorealistic_style": true/false,\n'
            '  "has_ai_flaws": true/false,\n'
            '  "woman_present": true/false,\n'
            '  "nsfw": true/false,\n'
            '  "quality_score": 1-10,\n'
            '  "category": "InstagramInfluencer|NSFW|Professional|Amateur|Unknown|DISCARD",\n'
            '  "reason": "short reason"\n'
            "}\n"
            "CATEGORY RULES:\n"
            "- DISCARD: Not photorealistic OR no woman present OR severe AI flaws (score <= 3).\n"
            "- NSFW: Explicit nudity.\n"
            "- InstagramInfluencer: Photorealistic woman, social media style.\n"
            "- Professional: Studio/editorial.\n"
            "- Amateur: Casual phone style.\n"
            "- Unknown: Other."
        )

        # Use the correct model ID
        response = model.generate_content([
            {"text": prompt_text},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}
        ])
        
        if not response.text:
             return {"category": "DISCARD", "reason": "Empty response (safety block?)"}

        result = json.loads(response.text)
        
        photorealistic = result.get("photorealistic_style", False)
        woman          = result.get("woman_present", False)
        nsfw           = result.get("nsfw", False)
        cat            = result.get("category", "Unknown")
        qual           = result.get("quality_score", 0)

        if not photorealistic or not woman:
            result["category"] = "DISCARD"
            return result

        if qual <= 3:
            result["category"] = "DISCARD"
            return result

        if nsfw:
            result["category"] = "NSFW"
        elif cat not in CATEGORIES and cat != "DISCARD":
            result["category"] = "Unknown"
        
        return result

    except Exception as e:
        err_str = str(e)
        if "429" in err_str:
            key_manager.mark_rate_limited(key)
            return {"category": "RETRY"}
        if "413 Request Entity Too Large" in err_str: 
             print(f"  [FILE SIZE ERROR] {image_path.name} too large, will retry with lower quality next time or skip.", flush=True)
             return {"category": "RETRY"}
        if "400" in err_str: 
             return {"category": "DISCARD", "reason": f"API 400: {err_str[:100]}"}
             
        return {"category": "DISCARD", "reason": f"Gemini Error: {err_str[:100]}"}

def process_queue_item(img_path):
    # Use the claimed path (which will be the .processing file)
    result   = vision_classify(img_path)
    category = result.get("category", "Unknown")
    
    if category == "RETRY":
        time.sleep(2) 
        return "RETRY"
    
    if category == "SKIPPED":
        return "SKIPPED"

    qual = result.get("quality_score", 0)
    print(f"[{category}] Q:{qual} | {img_path.name}", flush=True)

    if category == "DISCARD":
        debug_dir = SORTED_DIR / "DISCARD"
        debug_dir.mkdir(exist_ok=True)
        try:
            final_discard_path = debug_dir / img_path.name.replace(".processing", "") 
            shutil.move(str(img_path), str(final_discard_path))
            
            stem = img_path.stem 
            txt_path = img_path.parent / f"{stem}.txt"
            meta_path = img_path.parent / f"{stem}.meta.json"

            if txt_path.exists(): shutil.move(str(txt_path), str(debug_dir / txt_path.name.replace(".processing", "")))
            if meta_path.exists(): shutil.move(str(meta_path), str(debug_dir / meta_path.name.replace(".processing", "")))
        except FileNotFoundError: return "SKIPPED"
        return "DISCARD"

    ext = img_path.suffix
    with _fs_lock:
        cat_dir    = SORTED_DIR / category
        msg_id_key = img_path.stem 
        safe_name  = f"{category.lower()}_{msg_id_key}_{int(time.time())}_{random.randint(100,999)}"
        final_img  = cat_dir / f"{safe_name}{ext}"
        final_txt  = cat_dir / f"{safe_name}.txt"
        final_vis  = cat_dir / f"{safe_name}.vision.json"

        try:
            shutil.move(str(img_path), str(final_img))
            if txt_path.exists():
                shutil.move(str(txt_path), str(final_txt))
            elif prompt_text:
                final_txt.write_text(prompt_text, encoding="utf-8")
        except FileNotFoundError: return "SKIPPED"

    vision_result = {
        "image_file": str(final_img),
        "prompt": prompt_text,
        "category": category,
        "model": MODEL_ID,
        **result,
        "meta": meta
    }
    final_vis.write_text(json.dumps(vision_result, indent=2, ensure_ascii=False), encoding="utf-8")
    
    if meta_path.exists(): 
        try: meta_path.unlink()
        except: pass
        
    return category

def main():
    print(f"=== Gemini Vision Worker (x{PARALLEL_WORKERS}) ===", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)

    processed = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        while True:
            all_imgs = []
            try:
                all_imgs = [p for p in QUEUE_DIR.iterdir()
                            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                            and not p.name.endswith(".processing") 
                           ]
                if not all_imgs:
                    time.sleep(POLL_INTERVAL)
                    continue
            except FileNotFoundError:
                time.sleep(POLL_INTERVAL)
                continue

            random.shuffle(all_imgs)
            batch = all_imgs[:PARALLEL_WORKERS * 2]
            
            futures = []
            for img in batch:
                futures.append(pool.submit(process_queue_item, img))
            
            for f in futures:
                res = f.result()
                if res != "SKIPPED" and res != "RETRY":
                    processed += 1
            
            print(f"Batch done. Total processed: {processed}", flush=True)

if __name__ == "__main__":
    main()
