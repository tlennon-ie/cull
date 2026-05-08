"""
vision_worker.py - Processes images from queue using LM Studio.
Fixes applied: proper file path handling, imports, stem logic.
"""
import os, re, json, time, base64, io, shutil, threading, random, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from datetime import datetime
from paths import base_dir

# Config
BASE_DIR   = Path(str(base_dir()))
SLUG       = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
QUEUE_DIR  = Path(os.environ.get("PIPELINE_QUEUE",  str(BASE_DIR / "queue" / SLUG)))
SORTED_DIR = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted" / SLUG)))

LMS_URL          = os.environ.get("LMSTUDIO_PRIMARY_URL", "http://127.0.0.1:1234")
LMS_MODEL        = os.environ.get("LMSTUDIO_PRIMARY_MODEL", "qwen3-vl-8b-thinking-abliterated")
TIMEOUT          = 600
MAX_SIZE         = 512
POLL_INTERVAL    = 2
PARALLEL_WORKERS = 1

_fs_lock = threading.Lock()

CATEGORIES = ["InstagramInfluencer", "NSFW", "Professional", "Amateur", "Unknown"]
for c in CATEGORIES:
    (SORTED_DIR / c).mkdir(parents=True, exist_ok=True)

def resize_bytes(data, max_size=MAX_SIZE):
    try:
        if not data: return None
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
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

    # Use image_path.stem for sidecar files (standard convention: foo.txt for foo.png)
    stem      = image_path.stem
    txt_path  = image_path.parent / f"{stem}.txt"
    if not txt_path.exists():
        txt_path = image_path.parent / f"{image_path.name}.txt"
    meta_path = image_path.parent / f"{stem}.meta.json"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        prompt_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
    except FileNotFoundError:
        return "SKIPPED" 

    msg_id_key = meta.get("message_id", stem)

    img_bytes = processing_path.read_bytes()
    if not img_bytes:
        processing_path.unlink(missing_ok=True)
        return {"category": "SKIPPED"}
         
    small = resize_bytes(img_bytes)
    if small is None:
         return {"category": "DISCARD", "reason": "Image corrupt/invalid format/truncated"}
         
    b64 = base64.standard_b64encode(small).decode()

    user = (
        "You are an image classifier for an AI image dataset. Your job is to sort images "
        "of women into categories. Respond ONLY with valid JSON.\n"
        "{\n"
        '  "photorealistic_style": true/false,\n'
        '  "has_ai_flaws": true/false,\n'
        '  "woman_present": true/false,\n'
        '  "nsfw": true/false,\n'
        '  "quality_score": 1-10,\n'
        '  "category": "InstagramInfluencer|NSFW|Professional|Amateur|Unknown|DISCARD",\n'
        '  "reason": "short reason"\n'
        "}\n\n"
        "RULES:\n"
        "- photorealistic_style: TRUE if the image is rendered in a photorealistic or hyperrealistic "
        "style. FALSE if anime, cartoon, painting, or 3D render.\n"
        "- has_ai_flaws: TRUE only if there are SEVERE and obvious AI artifacts. Minor skin smoothness is NOT "
        "a flaw.\n"
        "- woman_present: TRUE if a human female is clearly the primary subject.\n"
        "- nsfw: TRUE if there is explicit nudity or sexual content.\n"
        "- quality_score: 1-10 overall photorealistic quality.\n\n"
        "CATEGORY RULES:\n"
        "- DISCARD: Not photorealistic OR no woman present OR severe AI flaws (score <= 3).\n"
        "- NSFW: Photorealistic woman, explicit nudity or sexual content.\n"
        "- InstagramInfluencer: Photorealistic woman, social media style.\n"
        "- Professional: Photorealistic woman, studio/editorial/fashion photography.\n"
        "- Amateur: Photorealistic woman, casual phone-camera style.\n"
        "- Unknown: Photorealistic woman that doesn't fit the above."
    )

    for attempt in range(3):
        try:
            resp = requests.post(f"{LMS_URL}/v1/chat/completions", json={
                "model": LMS_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": user}
                ]}],
                "temperature": 0.1,
                "max_tokens": 8000,
            }, timeout=TIMEOUT)

            if not resp.ok:
                err = resp.text[:200]
                if resp.status_code in [500, 503, 429]:
                    time.sleep(2 * (attempt + 1))
                    continue
                if resp.status_code == 400:
                     return {"category": "DISCARD", "reason": f"API 400 Bad Request: {err}"}
                raise ValueError(f"HTTP {resp.status_code}: {err}")

            raw = resp.json()["choices"][0]["message"]["content"] or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(f"```(?:json)?", "", raw).strip().rstrip("`").strip()
            result = json.loads(raw) 
            
            photorealistic = result.get("photorealistic_style", False)
            ai_flaws       = result.get("has_ai_flaws", True)
            woman          = result.get("woman_present", False)
            nsfw           = result.get("nsfw", False)
            cat            = result.get("category", "Unknown")
            qual           = result.get("quality_score", 0) or 0

            if not photorealistic or not woman:
                result["category"] = "DISCARD"
                result["realistic"] = False
                return result

            if qual <= 3:
                result["category"] = "DISCARD"
                result["realistic"] = False
                return result

            if nsfw:
                result["category"] = "NSFW"
            elif cat not in ["InstagramInfluencer", "NSFW", "Professional", "Amateur", "Unknown"]:
                result["category"] = "Unknown"
            else:
                result["category"] = cat

            result["realistic"]    = True
            result["ai_generated"] = False  
            return result

        except Exception as e:
            if attempt == 2:
                return {"category": "DISCARD", "reason": f"API Error after 3 attempts: {str(e)[:100]}"}
            time.sleep(2)
            
    return {"category": "DISCARD", "reason": "API Retry exhausted"}

def process_queue_item(img_path):
    processing_path = Path(str(img_path) + ".processing")
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
            final_discard_path = debug_dir / img_path.name
            shutil.move(str(processing_path), str(final_discard_path))
            
            stem = img_path.stem 
            txt_path = img_path.parent / f"{stem}.txt"
            if not txt_path.exists():
                txt_path = img_path.parent / f"{img_path.name}.txt"
            meta_path = img_path.parent / f"{stem}.meta.json"

            if txt_path.exists(): shutil.move(str(txt_path), str(debug_dir / txt_path.name))
            if meta_path.exists(): shutil.move(str(meta_path), str(debug_dir / meta_path.name))
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
            shutil.move(str(processing_path), str(final_img))
            
            stem = img_path.stem
            txt_path = img_path.parent / f"{stem}.txt"
            if not txt_path.exists():
                txt_path = img_path.parent / f"{img_path.name}.txt"
            prompt_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            
            if txt_path.exists():
                shutil.move(str(txt_path), str(final_txt))
            elif prompt_text:
                final_txt.write_text(prompt_text, encoding="utf-8")
        except FileNotFoundError: return "SKIPPED"

    meta_path = img_path.parent / f"{img_path.stem}.meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except:
        meta = {}

    vision_result = {
        "image_file": str(final_img),
        "prompt": prompt_text,
        "category": category,
        "model": LMS_MODEL,
        **result,
        "meta": meta
    }
    final_vis.write_text(json.dumps(vision_result, indent=2, ensure_ascii=False), encoding="utf-8")
    
    if meta_path.exists(): 
        try: meta_path.unlink()
        except: pass
        
    return category

def main():
    print(f"=== Vision Worker (x{PARALLEL_WORKERS}) ===", flush=True)
    print(f"Model: {LMS_MODEL}", flush=True)

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
            
            if batch:
                print(f"Batch processed: {len([r for r in [f.result() for f in futures] if r not in ['SKIPPED', 'RETRY']])}", flush=True)

if __name__ == "__main__":
    main()
