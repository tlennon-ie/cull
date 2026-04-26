#!/usr/bin/env python3
"""
vision_worker_lm_keepalive.py - Keep model warm while processing

Prevents LMStudio from auto-unloading by making continuous requests.
Uses a background thread to keep the model alive.
"""
import os, re, json, time, base64, io, threading, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import sys
import requests

# Import queue manager
sys.path.insert(0, str(Path(__file__).parent))
from queue_manager import get_next_image_round_robin

# Config
from dotenv import load_dotenv
from paths import base_dir
load_dotenv()

BASE_DIR   = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
SLUG       = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
_RAW_SORTED = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted")))
SORTED_DIR = _RAW_SORTED if _RAW_SORTED.name == SLUG else _RAW_SORTED / SLUG

LMS_URL          = os.environ.get("LMSTUDIO_PRIMARY_URL", "http://100.75.25.43:1234")
LMS_MODEL        = os.environ.get("LMSTUDIO_PRIMARY_MODEL", "qwen3.5-9b-uncensored-hauhaucs-aggressive")
TIMEOUT          = 600
MAX_SIZE         = 512
POLL_INTERVAL    = 2
PARALLEL_WORKERS = 4
KEEPALIVE_INTERVAL = 15  # Send keepalive every 15 seconds

_fs_lock = threading.Lock()

CATEGORIES = ["InstagramInfluencer", "NSFW", "Professional", "Amateur", "Unknown"]
for c in CATEGORIES:
    (SORTED_DIR / c).mkdir(parents=True, exist_ok=True)

def keepalive_task():
    """Make a tiny request to keep model from unloading"""
    try:
        requests.post(
            f"{LMS_URL}/v1/chat/completions",
            json={
                "model": LMS_MODEL,
                "messages": [{"role": "user", "content": " "}],
                "max_tokens": 1,
                "temperature": 0.1
            },
            timeout=30
        )
    except:
        pass  # Silent fail - just a keepalive

def start_keepalive_thread():
    """Start background thread that sends keepalive requests"""
    def run():
        while True:
            time.sleep(KEEPALIVE_INTERVAL)
            keepalive_task()
    
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

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

def vision_classify(image_path, source_name):
    """Classify image using LM Studio with keepalive"""
    processing_path = Path(str(image_path) + ".processing")
    try:
        image_path.rename(processing_path)
    except FileNotFoundError:
        return "SKIPPED"
    except Exception as e:
        print(f"  [ERR] Failed to rename {image_path}: {e}", flush=True)
        return "SKIPPED"

    stem = image_path.stem
    txt_path = image_path.parent / f"{stem}.txt"
    if not txt_path.exists():
        txt_path = image_path.parent / f"{image_path.name}.txt"
    meta_path = image_path.parent / f"{stem}.meta.json"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        prompt_text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
    except FileNotFoundError:
        processing_path.unlink(missing_ok=True)
        return "SKIPPED"

    msg_id_key = meta.get("message_id", stem)

    img_bytes = processing_path.read_bytes()
    if not img_bytes:
        processing_path.unlink(missing_ok=True)
        return {"category": "SKIPPED"}

    small = resize_bytes(img_bytes)
    if small is None:
        processing_path.unlink(missing_ok=True)
        return {"category": "DISCARD", "reason": "Image corrupt/invalid format/truncated"}

    b64 = base64.standard_b64encode(small).decode()

    from vision_prompt import build_classification_prompt, apply_scores, _safe_parse_vision_json, extract_message_text, build_response_format
    prompt_instruction = build_classification_prompt()

    try:
        response = requests.post(
            f"{LMS_URL}/v1/chat/completions",
            json={
                "model": LMS_MODEL,
                "messages": [
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
                "temperature": 0.1,
                # Thinking-style models (qwen3-vl-*-thinking, deepseek-r1, etc.)
                # spend a chunk of tokens inside <think>...</think> before
                # emitting the JSON answer. 2000 leaves headroom for both the
                # reasoning and the JSON tail.
                "max_tokens": 2000,
                # Force structured output - LM Studio constrains the decoder to
                # only emit tokens that match our JSON schema. Eliminates
                # `raw=''` / "<think>... no JSON" failure modes.
                "response_format": build_response_format(),
            },
            timeout=TIMEOUT,
        )

        if response.status_code != 200:
            print(
                f"  [LMS Error {response.status_code}] model={LMS_MODEL} url={LMS_URL} "
                f"body={response.text[:600].replace(chr(10), ' ')}",
                flush=True,
            )
            processing_path.rename(image_path)
            return {"category": "RETRY"}

        message = response.json()["choices"][0]["message"]
        raw = extract_message_text(message)
        parsed = _safe_parse_vision_json(raw)
        if parsed is None:
            preview = (raw or "")[:300].replace("\n", " ")
            print(
                f"  [Error] empty/invalid JSON from {LMS_MODEL}@{LMS_URL}; raw={preview!r}",
                flush=True,
            )
            try:
                processing_path.rename(image_path)
            except OSError:
                pass
            return {"category": "RETRY"}
        result = apply_scores(parsed)

        final_category = result.get("category", "Unknown")

        # Save to sorted folder with source-based subfolder
        dest_dir = SORTED_DIR / final_category / source_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        safe_name = f"{final_category.lower()}_{msg_id_key}_{int(time.time())}_{random.randint(100,999)}"
        ext = image_path.suffix
        final_img = dest_dir / f"{safe_name}{ext}"
        final_txt = dest_dir / f"{safe_name}.txt"
        final_meta = dest_dir / f"{safe_name}.vision.json"

        try:
            import shutil
            shutil.move(str(processing_path), str(final_img))
            if txt_path.exists():
                shutil.move(str(txt_path), str(final_txt))
        except FileNotFoundError:
            return "SKIPPED"

        final_meta.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        ovr = result.get("OVR_Quality_Score", 0)
        rel = result.get("REL_Quality_Score", 0)
        print(f"[{final_category}] OVR:{ovr} REL:{rel} | {image_path.name}", flush=True)
        return final_category

    except Exception as e:
        err_str = str(e)
        print(f"  [Error] {err_str[:100]}", flush=True)
        try:
            processing_path.rename(image_path)
        except:
            pass
        return {"category": "RETRY"}

_VISION_HINTS = ("vl", "vision", "visual", "llava", "gemma-3", "qwen3-vl",
                 "qwen2.5-vl", "qwen2-vl", "moondream", "minicpm-v", "pixtral")


def _looks_vision_capable(model_id: str) -> bool:
    return any(h in (model_id or "").lower() for h in _VISION_HINTS)


def run_vision_worker():
    print(f"=== Vision Worker (LMStudio with Keepalive) ===")
    print(f"  URL:   {LMS_URL}")
    print(f"  Model: {LMS_MODEL}")
    print(f"  Workers: {PARALLEL_WORKERS}")
    print(f"  Keepalive: Every {KEEPALIVE_INTERVAL}s")
    if LMS_MODEL and not _looks_vision_capable(LMS_MODEL):
        print(
            f"  [WARNING] Model '{LMS_MODEL}' does not look vision-capable. "
            "Expected a VL model (e.g. qwen2.5-vl-7b, qwen3-vl-8b, gemma-3-*, "
            "llava-*). LMStudio will likely return 400 on every image. "
            "Update LMSTUDIO_PRIMARY_MODEL / LMSTUDIO_SECONDARY_MODEL in Settings.",
            flush=True,
        )
    print()

    # Start keepalive thread FIRST - keep model warm
    start_keepalive_thread()
    
    # Wait for first keepalive to ensure model is loaded
    time.sleep(2)

    processed = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        try:
            while True:
                # Get next image in round-robin by source
                source_name, img_path = get_next_image_round_robin()
                if img_path is None:
                    time.sleep(POLL_INTERVAL)
                    continue

                img_path = Path(img_path)
                if not img_path.exists():
                    continue

                future = pool.submit(vision_classify, img_path, source_name)
                result = future.result(timeout=300)

                if result not in ["SKIPPED", "RETRY"] and isinstance(result, str):
                    processed += 1
                    if processed % 10 == 0:
                        print(f"  [Progress] Processed: {processed}", flush=True)

        except KeyboardInterrupt:
            print("\n[Stopped] Vision worker stopped", flush=True)

if __name__ == "__main__":
    run_vision_worker()
