#!/usr/bin/env python3
"""
vision_worker_balanced_groq.py - Round-robin processing with Groq API

Ensures all sources get processed fairly:
- One from civitai, one from zforfree, one from discord, etc.
- Uses Groq llama-4-scout (fast, handles NSFW)
"""
import os, re, json, time, base64, io, threading, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))
from queue_manager import get_next_image_round_robin

# Config
from dotenv import load_dotenv
load_dotenv()
from paths import base_dir

BASE_DIR   = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
SLUG       = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
_RAW_SORTED = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted")))
SORTED_DIR = _RAW_SORTED if _RAW_SORTED.name == SLUG else _RAW_SORTED / SLUG

# Groq Config (GROQ_API_KEYS is a comma-separated rotation list)
_RAW_KEYS = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY", "")
API_KEYS = [k.strip() for k in _RAW_KEYS.split(",") if k.strip()]
if not API_KEYS:
    raise ValueError("No GROQ_API_KEYS / GROQ_API_KEY set in .env")
MODEL_ID   = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
PARALLEL_WORKERS = 6
POLL_INTERVAL    = 1

_fs_lock = threading.Lock()

CATEGORIES = ["InstagramInfluencer", "NSFW", "Professional", "Amateur", "Unknown", "Watermarked"]
for c in CATEGORIES:
    (SORTED_DIR / c).mkdir(parents=True, exist_ok=True)

try:
    from groq import Groq
except ImportError:
    print("Installing groq...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "groq"])
    from groq import Groq

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
            self.cooldowns[key] = time.time() + 60
            print(f"  [Rate Limited] Key ...{key[-4:]} cooling down 60s", flush=True)

key_manager = KeyManager(API_KEYS)

def resize_bytes(data, max_size=4*1024*1024):
    try:
        if not data: return None
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80) 
        if buffer.tell() < max_size:
            return data 
        
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70) 
        if buffer.tell() < max_size:
            return buffer.getvalue()
        
        w, h = img.size
        scale = max_size / buffer.tell() * 0.8
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        buffer_final = io.BytesIO()
        img.save(buffer_final, format="JPEG", quality=60) 
        return buffer_final.getvalue()
    except Exception as e:
        return None

def vision_classify(image_path, source_name):
    """Classify image using Groq API"""
    client, key = key_manager.get_client()
    if not client:
        print(f"  [Wait] Rate limited, retrying...", flush=True)
        return {"category": "RETRY"}

    processing_path = Path(str(image_path) + ".processing")
    try:
        image_path.rename(processing_path)
    except FileNotFoundError:
        return "SKIPPED" 
    except Exception as e:
        print(f"  [ERR] Rename failed: {e}", flush=True)
        return "SKIPPED"

    stem = image_path.stem
    txt_path = image_path.parent / f"{stem}.txt"
    meta_path = image_path.parent / f"{stem}.meta.json"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        prompt_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
    except:
        meta = {}
        prompt_text = ""

    img_bytes = processing_path.read_bytes()
    if not img_bytes:
        processing_path.unlink(missing_ok=True)
        return {"category": "SKIPPED"}
         
    small = resize_bytes(img_bytes)
    if small is None:
         return {"category": "DISCARD", "reason": "Image corrupt/invalid"}
         
    b64 = base64.b64encode(small).decode('utf-8')

    from vision_prompt import build_classification_prompt, apply_scores, _safe_parse_vision_json
    prompt_instruction = build_classification_prompt()

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
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

        raw = response.choices[0].message.content
        parsed = _safe_parse_vision_json(raw)
        if parsed is None:
            preview = (raw or "")[:300].replace("\n", " ")
            print(f"  [Error] empty/invalid JSON from Groq; raw={preview!r}", flush=True)
            try:
                processing_path.rename(image_path)
            except OSError:
                pass
            return {"category": "RETRY"}
        result = apply_scores(parsed)
        final_category = result.get("category", "Unknown")
        
        # Save to sorted folder with source-based subfolder
        # Structure: sorted/realistic_female_influencer/{category}/{source}/image.jpg
        dest_dir = SORTED_DIR / final_category / source_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_img = dest_dir / processing_path.name.replace(".processing", "")
        
        with _fs_lock:
            processing_path.rename(dest_img)
            
            # Save vision.json with source tracking
            vision_path = dest_img.with_stem(dest_img.stem).with_suffix(".vision.json")
            result["source"] = source_name
            vision_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            
            # Move metadata
            for ext in [".txt", ".meta.json"]:
                meta_src = image_path.parent / f"{stem}{ext}"
                if meta_src.exists():
                    try:
                        meta_src.rename(dest_dir / f"{dest_img.stem}{ext}")
                    except:
                        pass
        
        return final_category

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg.lower():
            key_manager.mark_rate_limited(key)
            processing_path.rename(image_path)
            return {"category": "RETRY"}
        
        print(f"  [Error] {error_msg[:100]}", flush=True)
        processing_path.rename(image_path)
        return {"category": "RETRY", "error": error_msg}

def run_vision_worker():
    print("=== Vision Worker Groq (Balanced Round-Robin) ===")
    print(f"Processing images in balanced order across sources")
    print(f"Using {len(API_KEYS)} Groq API keys\n")
    
    processed = {cat: 0 for cat in CATEGORIES}
    
    try:
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = []
            
            while True:
                # Submit tasks
                while len(futures) < PARALLEL_WORKERS:
                    source, image_path = get_next_image_round_robin()
                    if not image_path:
                        break
                    
                    future = executor.submit(vision_classify, image_path, source)
                    futures.append((future, source))
                
                if not futures:
                    print(f"  [Wait] No images (checking in {POLL_INTERVAL}s)", flush=True)
                    time.sleep(POLL_INTERVAL)
                    continue
                
                # Wait for at least one to complete
                done, _ = [(f, s) for f, s in futures if f.done()], []
                if done:
                    for future, source in done:
                        try:
                            result = future.result()
                            if isinstance(result, dict):
                                cat = result.get("category", "Unknown")
                            else:
                                cat = result
                            processed[cat] = processed.get(cat, 0) + 1
                        except Exception as e:
                            print(f"  [Error] {e}", flush=True)
                        futures.remove((future, source))
                    
                    # Log progress every 10
                    if sum(processed.values()) % 10 == 0:
                        summary = " | ".join(f"{k}:{v}" for k,v in sorted(processed.items()) if v > 0)
                        print(f"  [Progress] {summary}", flush=True)
                else:
                    time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n[Stopped] Vision worker stopped", flush=True)

if __name__ == "__main__":
    run_vision_worker()
