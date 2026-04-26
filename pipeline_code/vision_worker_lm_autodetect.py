#!/usr/bin/env python3
"""
vision_worker_balanced_lm_autodetect.py - Auto-detect and use whatever model is in LMStudio

Uses first available loaded model, falls back gracefully if none loaded
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

LMS_URL          = os.environ.get("LMSTUDIO_PRIMARY_URL", "http://127.0.0.1:1234")
LMS_URL_SECONDARY = os.environ.get("LMSTUDIO_SECONDARY_URL", "")
TIMEOUT          = 600
MAX_SIZE         = 512
POLL_INTERVAL    = 2
PARALLEL_WORKERS = 4

# Vision-capable model name hints (case-insensitive). Used to pick a VL model
# when an LMStudio instance has multiple models loaded.
VISION_HINTS = ("vl", "vision", "visual", "llava", "gemma-3", "qwen3-vl", "qwen2.5-vl",
                "qwen2-vl", "moondream", "minicpm-v", "pixtral")
EMBED_HINTS = ("embed", "nomic", "bge")

_fs_lock = threading.Lock()

CATEGORIES = ["InstagramInfluencer", "NSFW", "Professional", "Amateur", "Unknown"]
for c in CATEGORIES:
    (SORTED_DIR / c).mkdir(parents=True, exist_ok=True)

def _pick_vision_model(models: list[str]) -> str | None:
    """Return the best vision candidate from a list of model IDs, or None."""
    if not models:
        return None
    non_embed = [m for m in models if not any(h in m.lower() for h in EMBED_HINTS)]
    vision = [m for m in non_embed if any(h in m.lower() for h in VISION_HINTS)]
    return vision[0] if vision else None


def _models_on(base_url: str) -> list[str]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=5)
        if resp.status_code == 200:
            return [item.get("id", "") for item in resp.json().get("data", []) if item.get("id")]
    except requests.RequestException:
        pass
    return []


def discover_endpoint() -> tuple[str, str] | tuple[None, None]:
    """Return (base_url, model_id) for whichever LMStudio instance has a vision model.

    Strategy: probe both endpoints, prefer an instance with a vision-capable model
    (qwen-vl, gemma-3, llava, etc.). Never picks a text-only model - the worker
    needs multimodal to classify images.
    """
    endpoints: list[tuple[str, str, list[str]]] = []
    for label, url in (("primary", LMS_URL), ("secondary", LMS_URL_SECONDARY)):
        if not url:
            continue
        models = _models_on(url)
        if not models:
            print(f"  [autodetect] {label} ({url}) unreachable or empty", flush=True)
            continue
        endpoints.append((label, url, models))

    # Prefer any instance that actually has a vision-capable model.
    for label, url, models in endpoints:
        picked = _pick_vision_model(models)
        if picked:
            print(f"  [autodetect] {label} -> {picked} (all: {models})", flush=True)
            return url, picked

    for label, _url, models in endpoints:
        print(f"  [autodetect] {label} has no vision-capable model loaded: {models}", flush=True)
    return None, None

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

def vision_classify(image_path, source_name, model_name, lms_url=None):
    """Classify image using LM Studio with autodetected model"""
    lms_url = lms_url or LMS_URL
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

    from vision_prompt import build_classification_prompt, apply_scores, _safe_parse_vision_json, extract_message_text  # local import keeps module load-order safe
    prompt_instruction = build_classification_prompt()

    try:
        response = requests.post(
            f"{lms_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model_name,
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
                # Thinking-style models (qwen3-vl-*-thinking, deepseek-r1)
                # need room for <think>...</think> + the JSON answer.
                "max_tokens": 2000,
            },
            timeout=TIMEOUT,
        )

        if response.status_code != 200:
            # Dump the full response body once so VL-vs-text model mismatches etc. are diagnosable.
            print(f"  [LMS Error {response.status_code}] {response.text[:600].replace(chr(10),' ')}", flush=True)
            processing_path.rename(image_path)
            return {"category": "RETRY", "error": f"LMS API error {response.status_code}"}

        message = response.json()["choices"][0]["message"]
        raw = extract_message_text(message)
        parsed = _safe_parse_vision_json(raw)
        if parsed is None:
            preview = (raw or "")[:300].replace("\n", " ")
            print(f"  [Error] empty/invalid JSON from {model_name}; raw={preview!r}", flush=True)
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

def run_vision_worker() -> None:
    print("=== Vision Worker (LMStudio Auto-Detect) ===", flush=True)

    lms_url, model_name = discover_endpoint()
    if not model_name:
        print("  [ERROR] No LMStudio instance with a vision-capable model found.", flush=True)
        print(f"  [ERROR] Primary URL:   {LMS_URL}", flush=True)
        print(f"  [ERROR] Secondary URL: {LMS_URL_SECONDARY or '(not set)'}", flush=True)
        print("  [ERROR] Load a VL model (e.g. qwen3-vl-8b, qwen2.5-vl-7b, gemma-3-*)", flush=True)
        return

    print(f"  Endpoint: {lms_url}", flush=True)
    print(f"  Model:    {model_name}", flush=True)
    print(f"  Workers:  {PARALLEL_WORKERS}", flush=True)

    processed = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        try:
            while True:
                source_name, img_path = get_next_image_round_robin()
                if img_path is None:
                    time.sleep(POLL_INTERVAL)
                    continue

                if not img_path.exists():
                    continue

                future = pool.submit(vision_classify, img_path, source_name, model_name, lms_url)
                result = future.result(timeout=300)

                if result not in ("SKIPPED", "RETRY"):
                    processed += 1
                    if processed % 10 == 0:
                        print(f"  [Progress] Processed: {processed}", flush=True)

        except KeyboardInterrupt:
            print("\n[Stopped] Vision worker stopped", flush=True)

if __name__ == "__main__":
    run_vision_worker()
