#!/usr/bin/env python3
"""
HYBRID VISION WORKER - GROQ + LMSTUDIO (FIXED)
- Groq tries first (fast, rates limited)
- LMStudio fallback if Groq times out
- Both working correctly with proper models
"""
from dotenv import load_dotenv
load_dotenv()
import os
import json
import base64
import io
import time
from pathlib import Path
from PIL import Image
import concurrent.futures
from groq import Groq
import requests

BASE_DIR = Path("I:/AI/openclaw/workspace/prompt-library")
SLUG = "realistic_female_influencer"
SORTED_DIR = BASE_DIR / "sorted" / SLUG

GROQ_KEYS = [k.strip() for k in (os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

LMSTUDIO_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "qwen/qwen2.5-vl-7b"

CATEGORIES = ["InstagramInfluencer", "Professional", "Cinematic", "Vintage", "Fantasy", "Sports", "Amateur", "Unknown", "DISCARD"]
for cat in CATEGORIES:
    (SORTED_DIR / cat).mkdir(parents=True, exist_ok=True)

_key_idx = 0

def detect_source(filename, metadata):
    """Detect source from filename + metadata"""
    filename_lower = filename.lower()
    
    if "civitai_" in filename_lower:
        return "civitai"
    elif "zff_" in filename_lower or "unknown_zff_" in filename_lower:
        return "zforfree"
    elif filename_lower.startswith(("ud_", "unstable_")):
        return "discord_ud"
    elif filename_lower.startswith(("mj_", "midjourney_")):
        return "discord_mj"
    elif "reddit_" in filename_lower:
        return "reddit"
    elif filename_lower.startswith(("x_", "twitter_")):
        return "twitter_x"
    elif "grok" in filename_lower:
        return "grok"
    elif "nanobanana" in filename_lower:
        return "nanobanana"
    
    if metadata:
        src_ch = metadata.get("source_channel", "").lower()
        src_gl = metadata.get("source_guild", "").lower()
        
        if "civitai" in src_ch:
            return "civitai"
        elif "reddit" in src_ch:
            return "reddit"
        elif "zforfree" in src_ch or "zff" in src_ch:
            return "zforfree"
        elif "unstable" in src_gl or "ud" in src_gl:
            return "discord_ud"
        elif "midjourney" in src_gl or "mj" in src_gl:
            return "discord_mj"
        elif "grok" in src_ch:
            return "grok"
        elif "nanobanana" in src_ch:
            return "nanobanana"
        elif "twitter" in src_ch or "x.com" in src_ch:
            return "twitter_x"
    
    return "unknown"

def resize_image(img_bytes, max_size=1024*1024):
    """Resize image to fit within limit"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        if buffer.tell() < max_size:
            return buffer.getvalue()
        
        w, h = img.size
        scale = (max_size / buffer.tell() * 0.8) ** 0.5
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        
        buf_final = io.BytesIO()
        img.save(buf_final, format="JPEG", quality=70)
        return buf_final.getvalue()
    except:
        return None

def classify_with_groq(b64_image):
    """Classify using Groq (primary)"""
    global _key_idx
    
    try:
        key = GROQ_KEYS[_key_idx % len(GROQ_KEYS)]
        _key_idx += 1
        
        client = Groq(api_key=key)
        
        prompt = """Analyze this image. Respond ONLY with valid JSON (no markdown):
{
  "photorealistic": true/false,
  "ai_flaws": true/false,
  "woman_present": true/false,
  "nsfw": true/false,
  "quality": 1-10,
  "style": "portrait|selfie|fashion|editorial|cinematic|vintage|fantasy|sports|other",
  "reason": "Brief reason"
}

RULES:
- photorealistic: True if photo-realistic/hyperrealistic. False if cartoon/painting/3D/anime.
- ai_flaws: True ONLY if OBVIOUS severe defects (melted faces, extra limbs, distorted anatomy).
- woman_present: True if adult human female is PRIMARY subject.
- nsfw: True ONLY if explicit nudity/sexual content.
- quality: 1-3 poor, 4-6 okay, 7-10 excellent.
- style: What you see in the image composition."""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
            timeout=30
        )
        
        return json.loads(response.choices[0].message.content), "groq"
    except Exception as e:
        return None, "groq"

def classify_with_lmstudio(b64_image):
    """Classify using LMStudio (fallback)"""
    try:
        prompt = """Analyze this image. Respond ONLY with valid JSON (no markdown):
{
  "photorealistic": true/false,
  "ai_flaws": true/false,
  "woman_present": true/false,
  "nsfw": true/false,
  "quality": 1-10,
  "style": "portrait|selfie|fashion|editorial|cinematic|vintage|fantasy|sports|other",
  "reason": "Brief reason"
}

RULES:
- photorealistic: True if photo-realistic/hyperrealistic. False if cartoon/painting/3D/anime.
- ai_flaws: True ONLY if OBVIOUS severe defects (melted faces, extra limbs, distorted anatomy).
- woman_present: True if adult human female is PRIMARY subject.
- nsfw: True ONLY if explicit nudity/sexual content.
- quality: 1-3 poor, 4-6 okay, 7-10 excellent.
- style: What you see in the image composition."""
        
        response = requests.post(
            f"{LMSTUDIO_URL}/chat/completions",
            json={
                "model": LMSTUDIO_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 300,
                "stream": False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content), "lmstudio"
        else:
            return None, "lmstudio"
    except:
        return None, "lmstudio"

def classify_image_hybrid(img_bytes, source_name):
    """Classify using both APIs: Groq first, LMStudio fallback"""
    small_bytes = resize_image(img_bytes)
    if not small_bytes:
        return None, "DISCARD", "Image corrupt"
    
    b64 = base64.b64encode(small_bytes).decode('utf-8')
    
    # Try Groq first (faster)
    result, api_used = classify_with_groq(b64)
    
    # If Groq failed, try LMStudio
    if result is None:
        result, api_used = classify_with_lmstudio(b64)
    
    if result is None:
        return None, "DISCARD", "Both APIs failed"
    
    # Apply hierarchical logic
    photorealistic = result.get("photorealistic", False)
    ai_flaws = result.get("ai_flaws", False)
    woman_present = result.get("woman_present", False)
    nsfw = result.get("nsfw", False)
    quality = result.get("quality", 5)
    style = result.get("style", "other").lower()
    
    category = "Unknown"
    
    # DISCARD checks
    if not photorealistic or not woman_present or ai_flaws or quality <= 2:
        category = "DISCARD"
    # NSFW (only if high quality)
    elif nsfw:
        category = "NSFW" if quality >= 7 else "DISCARD"
    # InstagramInfluencer (selfie/portrait, good quality) - PRIORITY
    elif style in ["selfie", "portrait"] and quality >= 6:
        category = "InstagramInfluencer"
    # Professional (editorial style)
    elif style == "editorial" and quality >= 6:
        category = "Professional"
    # Cinematic
    elif style == "cinematic" and quality >= 6:
        category = "Cinematic"
    # Vintage
    elif style == "vintage" and quality >= 6:
        category = "Vintage"
    # Fantasy
    elif style == "fantasy" and quality >= 6:
        category = "Fantasy"
    # Sports
    elif style == "sports" and quality >= 6:
        category = "Sports"
    # Amateur (fashion/casual)
    elif style == "fashion" and quality >= 5:
        category = "Amateur"
    # Everything else is Unknown or DISCARD
    elif quality <= 3:
        category = "DISCARD"
    else:
        category = "Unknown"
    
    result["category"] = category
    result["source"] = source_name
    result["api"] = api_used
    
    return result, category, result.get("reason", "")

def reclassify_all():
    """Reclassify all sorted files"""
    print("=" * 70)
    print("RECLASSIFYING ALL FILES - HYBRID GROQ + LMSTUDIO")
    print("=" * 70)
    print("")
    
    all_images = list(SORTED_DIR.rglob("*.jpg")) + list(SORTED_DIR.rglob("*.png"))
    print(f"Found {len(all_images)} images")
    print("")
    
    stats = {"processed": 0, "errors": 0, "reclassified": 0, "unchanged": 0, "groq": 0, "lmstudio": 0}
    start = time.time()
    
    for i, img_path in enumerate(all_images):
        if i % 50 == 0:
            elapsed = time.time() - start
            rate = i / max(elapsed, 0.1)
            remaining = (len(all_images) - i) / max(rate, 0.1)
            print(f"[{i}/{len(all_images)}] ETA: {remaining:.0f}s | Groq: {stats['groq']} | LM: {stats['lmstudio']}", flush=True)
        
        try:
            img_bytes = img_path.read_bytes()
            if not img_bytes:
                continue
            
            stem = img_path.stem
            meta_path = img_path.parent / f"{stem}.meta.json"
            metadata = {}
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
                except:
                    pass
            
            correct_source = detect_source(img_path.name, metadata)
            result, new_cat, reason = classify_image_hybrid(img_bytes, correct_source)
            
            if result is None:
                stats["errors"] += 1
                continue
            
            # Track which API was used
            if result.get("api") == "groq":
                stats["groq"] += 1
            else:
                stats["lmstudio"] += 1
            
            old_cat = [p.name for p in img_path.parents if p.parent == SORTED_DIR][0]
            
            # Move if category or source changed
            if new_cat != old_cat or correct_source != img_path.parent.name:
                new_path = SORTED_DIR / new_cat / correct_source
                new_path.mkdir(parents=True, exist_ok=True)
                
                dest_img = new_path / img_path.name
                img_path.rename(dest_img)
                
                for ext in [".txt", ".meta.json"]:
                    src = img_path.with_stem(stem).with_suffix(ext)
                    if src.exists():
                        dst = new_path / src.name
                        src.rename(dst)
                
                vision_path = new_path / f"{stem}.vision.json"
                vision_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
                
                stats["reclassified"] += 1
            else:
                vision_path = img_path.with_stem(stem).with_suffix(".vision.json")
                vision_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
                stats["unchanged"] += 1
            
            stats["processed"] += 1
        
        except Exception as e:
            stats["errors"] += 1
    
    elapsed = time.time() - start
    
    print("")
    print("=" * 70)
    print("COMPLETE")
    print(f"  Processed: {stats['processed']}")
    print(f"  Reclassified: {stats['reclassified']}")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  API Usage:")
    print(f"    Groq: {stats['groq']}")
    print(f"    LMStudio: {stats['lmstudio']}")
    print(f"  Time: {elapsed/60:.1f} min")
    print("=" * 70)

if __name__ == "__main__":
    reclassify_all()
