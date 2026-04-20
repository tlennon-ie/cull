"""
vision_worker_groq.py - High-speed cloud vision worker using Groq API (Llama 4 Scout).
Handles rate limits per-key with automatic cool-down.
"""
from dotenv import load_dotenv
load_dotenv()
import os, re, json, time, base64, io, shutil, threading, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from groq import Groq
from PIL import Image
from paths import base_dir

# Config
BASE_DIR   = Path(str(base_dir()))
SLUG       = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
QUEUE_DIR  = Path(os.environ.get("PIPELINE_QUEUE",  str(BASE_DIR / "queue" / SLUG)))
SORTED_DIR = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted" / SLUG)))

API_KEYS = [k.strip() for k in (os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
MODEL_ID   = "meta-llama/llama-4-scout-17b-16e-instruct"
PARALLEL_WORKERS = 6   # Reduced to prevent DDOSing the keys (2 per key)
POLL_INTERVAL    = 1

_fs_lock = threading.Lock()

class KeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.clients = {k: Groq(api_key=k) for k in keys}
        self.cooldowns = {k: 0 for k in keys}
        self.lock = threading.Lock()

    def get_client(self):
        with self.lock:
            now = time.time()
            available = [k for k in self.keys if self.cooldowns[k] < now]
            
            if not available:
                soonest = min(self.cooldowns.values())
                wait = soonest - now
                if wait > 0:
                    time.sleep(min(wait, 5)) 
                return None, None
            
            key = random.choice(available)
            return self.clients[key], key

    def mark_rate_limited(self, key):
        with self.lock:
            self.cooldowns[key] = time.time() + 60 # 60s cooldown
            print(f"  [RATE LIMIT] Key ...{key[-4:]} on cooldown for 60s", flush=True)

key_manager = KeyManager(API_KEYS)

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
    client, key = key_manager.get_client()
    if not client:
        return {"category": "RETRY"}

    try:
        img_bytes = image_path.read_bytes()
        if not img_bytes: return {"category": "SKIPPED"}
        
        resized_bytes = resize_bytes(img_bytes)
        if not resized_bytes:
            return {"category": "DISCARD", "reason": "Failed to resize image"}

        b64_image = base64.b64encode(resized_bytes).decode('utf-8')

        prompt_text = (
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

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                    ],
                }
            ],
            model=MODEL_ID,
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        raw = chat_completion.choices[0].message.content
        result = json.loads(raw)
        
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
             # A simple retry here, but a better approach might be to permanently lower quality for this image or skip
             return {"category": "RETRY"}
        if "400" in err_str: 
             return {"category": "DISCARD", "reason": f"API 400: {err_str[:100]}"}
             
        return {"category": "DISCARD", "reason": f"Groq Error: {err_str[:100]}"}

def process_queue_item(img_path):
    if not img_path.exists():
        return "SKIPPED"

    stem      = img_path.stem
    txt_path  = img_path.parent / f"{stem}.txt"
    meta_path = img_path.parent / f"{stem}.meta.json"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        prompt_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
    except FileNotFoundError:
        return "SKIPPED" 

    msg_id_key = meta.get("message_id", stem)

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
            (debug_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
            shutil.move(str(img_path), str(debug_dir / img_path.name))
            if txt_path.exists(): shutil.move(str(txt_path), str(debug_dir / txt_path.name))
            if meta_path.exists(): shutil.move(str(meta_path), str(debug_dir / meta_path.name))
        except FileNotFoundError: return "SKIPPED"
        return "DISCARD"

    ext = img_path.suffix
    with _fs_lock:
        cat_dir    = SORTED_DIR / category
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
        "groq_model": MODEL_ID,
        **result,
        "meta": meta
    }
    final_vis.write_text(json.dumps(vision_result, indent=2, ensure_ascii=False), encoding="utf-8")
    
    if meta_path.exists(): 
        try: meta_path.unlink()
        except: pass
        
    return category

def main():
    print(f"=== Groq Vision Worker (x{PARALLEL_WORKERS}) ===", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"API Keys: 3 active", flush=True)

    processed = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        while True:
            all_imgs = list(QUEUE_DIR.glob("*.png")) + list(QUEUE_DIR.glob("*.jpg"))
            if not all_imgs:
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
