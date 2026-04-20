"""
vision_worker_antigravity.py - High-speed cloud vision worker using Google Antigravity.
Co-exists with local vision_worker.py (handles race conditions via file existence checks).
"""
import os, re, json, time, base64, io, shutil, requests, threading, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from datetime import datetime
from paths import base_dir

BASE_DIR   = Path(str(base_dir()))
SLUG       = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
QUEUE_DIR  = Path(os.environ.get("PIPELINE_QUEUE",  str(BASE_DIR / "queue" / SLUG)))
SORTED_DIR = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted" / SLUG)))

# Antigravity Auth Proxy Configuration
API_URL          = "http://127.0.0.1:8069"
API_MODEL        = "gemini-3-pro-image"
TIMEOUT          = 60
MAX_SIZE         = 1024  # Larger size for cloud models
POLL_INTERVAL    = 1
PARALLEL_WORKERS = 12    # Higher concurrency for cloud

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
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as e:
        return None

def vision_classify(image_bytes):
    small = resize_bytes(image_bytes)
    if small is None:
         return {"category": "DISCARD", "reason": "Image corrupt/invalid format"}
         
    b64 = base64.standard_b64encode(small).decode()

    user = (
        "You are an image classifier for an AI image dataset. Your job is to sort images "
        "of women into categories. These are AI-generated images — that is expected and fine.\n\n"
        "Respond ONLY with valid JSON, no markdown:\n"
        "{\n"
        '  "photorealistic_style": true/false,\n'
        '  "has_ai_flaws": true/false,\n'
        '  "woman_present": true/false,\n'
        '  "nsfw": true/false,\n'
        '  "quality_score": 1-10,\n'
        '  "category": "InstagramInfluencer|NSFW|Professional|Amateur|Unknown|DISCARD",\n'
        '  "reason": "One short sentence describing the image style and why you chose this category."\n'
        "}\n\n"
        "RULES:\n"
        "- photorealistic_style: TRUE if the image is rendered in a photorealistic or hyperrealistic "
        "style. FALSE if anime, cartoon, painting, or 3D render.\n"
        "- has_ai_flaws: TRUE only if there are SEVERE and obvious AI artifacts (melted faces, extra fingers). "
        "Minor skin smoothness is NOT a flaw.\n"
        "- woman_present: TRUE if a human female is clearly the primary subject.\n"
        "- nsfw: TRUE if there is explicit nudity or sexual content.\n"
        "- quality_score: 1-10 overall photorealistic quality.\n\n"
        "CATEGORY RULES:\n"
        "- DISCARD: Not photorealistic OR no woman present OR severe AI flaws (quality_score <= 3).\n"
        "- NSFW: Photorealistic woman, explicit nudity or sexual content.\n"
        "- InstagramInfluencer: Photorealistic woman, high-quality social media style (lifestyle, fashion, casual). No nudity.\n"
        "- Professional: Photorealistic woman, studio/editorial/fashion photography. No nudity.\n"
        "- Amateur: Photorealistic woman, casual phone-camera style, selfies.\n"
        "- Unknown: Photorealistic woman that doesn't fit the above."
    )

    for attempt in range(3):
        try:
            # Antigravity/OpenAI compatible request
            resp = requests.post(f"{API_URL}/v1/chat/completions", json={
                "model": API_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": user}
                ]}],
                "temperature": 0.1,
                "max_tokens": 1000,
            }, timeout=TIMEOUT)

            if not resp.ok:
                err = resp.text[:200]
                if resp.status_code in [500, 503, 429]:
                    time.sleep(1 * (attempt + 1))
                    continue
                if resp.status_code == 400:
                     return {"category": "DISCARD", "reason": f"API 400: {err}"}
                raise ValueError(f"HTTP {resp.status_code}: {err}")

            raw = resp.json()["choices"][0]["message"]["content"] or ""
            # Cleanup markdown if present
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # Fallback if model chats instead of JSON
                if "InstagramInfluencer" in raw: result = {"category": "InstagramInfluencer"}
                elif "NSFW" in raw: result = {"category": "NSFW"}
                else: return {"category": "Unknown", "reason": "JSON decode failed"}

            # Validate fields
            photorealistic = result.get("photorealistic_style", False)
            woman          = result.get("woman_present", False)
            nsfw           = result.get("nsfw", False)
            cat            = result.get("category", "Unknown")
            qual           = result.get("quality_score", 0)

            # Hard discard logic
            if not photorealistic or not woman:
                result["category"] = "DISCARD"
                result["realistic"] = False
                return result

            if qual <= 3:
                result["category"] = "DISCARD"
                result["realistic"] = False
                return result

            # Category routing
            if nsfw:
                result["category"] = "NSFW"
            elif cat not in ["InstagramInfluencer", "Professional", "Amateur", "Unknown", "DISCARD"]:
                result["category"] = "Unknown"
            
            result["realistic"] = True
            return result

        except Exception as e:
            if attempt == 2:
                return {"category": "DISCARD", "reason": f"API Error: {str(e)[:100]}"}
            time.sleep(1)
            
    return {"category": "DISCARD", "reason": "Retry exhausted"}

def process_queue_item(img_path):
    # Verify file still exists (race condition check)
    if not img_path.exists():
        return "SKIPPED"

    stem      = img_path.stem
    txt_path  = img_path.parent / f"{stem}.txt"
    meta_path = img_path.parent / f"{stem}.meta.json"

    # Read content
    try:
        img_bytes = img_path.read_bytes()
        prompt_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
        meta        = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except FileNotFoundError:
        return "SKIPPED" # Someone else took it

    msg_id_key = meta.get("message_id", stem)

    # Classify
    result   = vision_classify(img_bytes)
    category = result["category"]
    
    # Console output
    qual = result.get("quality_score", 0)
    print(f"[{category}] Q:{qual} | {img_path.name}", flush=True)

    # Move logic
    if category == "DISCARD":
        debug_dir = SORTED_DIR / "DISCARD"
        debug_dir.mkdir(exist_ok=True)
        try:
            (debug_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
            shutil.move(str(img_path), str(debug_dir / img_path.name))
            if txt_path.exists(): shutil.move(str(txt_path), str(debug_dir / txt_path.name))
            if meta_path.exists(): shutil.move(str(meta_path), str(debug_dir / meta_path.name))
        except FileNotFoundError: return "SKIPPED"
        return "DISCARD"

    # Move to sorted
    ext = img_path.suffix
    with _fs_lock:
        cat_dir    = SORTED_DIR / category
        # Use simple timestamp-random naming to avoid globbing in high concurrency
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
        except FileNotFoundError:
            return "SKIPPED"

    # Write metadata
    vision_result = {
        "image_file": str(final_img),
        "prompt": prompt_text,
        "category": category,
        "antigravity_model": API_MODEL,
        **result,
        "meta": meta
    }
    final_vis.write_text(json.dumps(vision_result, indent=2, ensure_ascii=False), encoding="utf-8")
    
    if meta_path.exists(): 
        try: meta_path.unlink()
        except: pass
        
    return category

def main():
    print(f"=== Antigravity Vision Worker (x{PARALLEL_WORKERS}) ===", flush=True)
    print(f"Target: {API_URL} ({API_MODEL})", flush=True)

    processed = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        while True:
            # Quick glob
            all_imgs = list(QUEUE_DIR.glob("*.png")) + list(QUEUE_DIR.glob("*.jpg"))
            if not all_imgs:
                time.sleep(POLL_INTERVAL)
                continue
                
            # Shuffle to reduce collision chance with other workers
            random.shuffle(all_imgs)
            
            # Take a batch
            batch = all_imgs[:PARALLEL_WORKERS * 2]
            
            futures = []
            for img in batch:
                futures.append(pool.submit(process_queue_item, img))
            
            for f in futures:
                res = f.result()
                if res != "SKIPPED":
                    processed += 1
            
            print(f"Batch done. Total processed: {processed}", flush=True)

if __name__ == "__main__":
    main()
