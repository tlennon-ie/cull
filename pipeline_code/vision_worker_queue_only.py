#!/usr/bin/env python3
"""
QUEUE-ONLY VISION WORKER
Process ONLY the files in queue/ folder.
Do NOT touch sorted folder.
Simple, focused, single-purpose.
"""
from dotenv import load_dotenv
load_dotenv()
import os
import json
import base64
import io
from pathlib import Path
from PIL import Image
import time
from groq import Groq
from paths import base_dir

BASE_DIR = Path(str(base_dir()))
SLUG = "realistic_female_influencer"
QUEUE_DIR = BASE_DIR / "queue" / SLUG
SORTED_DIR = BASE_DIR / "sorted" / SLUG

GROQ_KEYS = [k.strip() for k in (os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

CATEGORIES = ["InstagramInfluencer", "Professional", "Cinematic", "Vintage", "Fantasy", "Sports", "Amateur", "Unknown", "DISCARD"]
for cat in CATEGORIES:
    (SORTED_DIR / cat).mkdir(parents=True, exist_ok=True)

_key_idx = 0

def detect_source(filename):
    """Detect source from filename"""
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
    """Classify using Groq"""
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
  "style": "portrait|selfie|fashion|editorial|cinematic|vintage|fantasy|sports|other"
}

RULES:
- photorealistic: True if photo-realistic/hyperrealistic. False if cartoon/painting/3D/anime.
- ai_flaws: True ONLY if OBVIOUS severe defects (melted faces, extra limbs).
- woman_present: True if adult human female is PRIMARY subject.
- nsfw: True ONLY if explicit nudity/sexual content.
- quality: 1-3 poor, 4-6 okay, 7-10 excellent.
- style: What you see in composition."""

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
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

def process_queue():
    """Process ALL files in queue folder"""
    print("=" * 70)
    print("QUEUE-ONLY VISION WORKER")
    print("=" * 70)
    print("")
    
    # Get all images from queue
    all_images = list(QUEUE_DIR.rglob("*.jpg")) + list(QUEUE_DIR.rglob("*.png"))
    all_images = [p for p in all_images if p.parent.name != "realistic_female_influencer"]  # Skip root
    
    print(f"Found {len(all_images)} images in queue")
    print("")
    
    stats = {"processed": 0, "errors": 0, "moved": 0, "start": time.time()}
    
    for i, img_path in enumerate(all_images):
        if i % 25 == 0:
            elapsed = time.time() - stats["start"]
            rate = i / max(elapsed, 0.1)
            remaining = (len(all_images) - i) / max(rate, 0.1)
            print(f"[{i}/{len(all_images)}] {remaining:.0f}s remaining", flush=True)
        
        try:
            img_bytes = img_path.read_bytes()
            if not img_bytes:
                continue
            
            # Classify
            small_bytes = resize_image(img_bytes)
            if not small_bytes:
                img_path.unlink()
                continue
            
            b64 = base64.b64encode(small_bytes).decode('utf-8')
            result = classify_with_groq(b64)
            
            if result is None:
                continue
            
            # Apply logic
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
                category = "NSFW" if quality >= 7 else "DISCARD"
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
            elif quality <= 3:
                category = "DISCARD"
            else:
                category = "Unknown"
            
            # Detect source
            source = detect_source(img_path.name)
            
            # Create destination
            dest_dir = SORTED_DIR / category / source
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_img = dest_dir / img_path.name
            
            # Move image
            img_path.rename(dest_img)
            
            # Move related files (.txt, .meta.json)
            stem = img_path.stem
            for ext in [".txt", ".meta.json"]:
                src = img_path.with_stem(stem).with_suffix(ext)
                if src.exists():
                    dst = dest_dir / src.name
                    src.rename(dst)
            
            # Save vision.json
            vision_path = dest_img.with_stem(stem).with_suffix(".vision.json")
            result["category"] = category
            result["source"] = source
            vision_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            
            stats["moved"] += 1
            stats["processed"] += 1
        
        except Exception as e:
            stats["errors"] += 1
    
    elapsed = time.time() - stats["start"]
    
    print("")
    print("=" * 70)
    print("QUEUE PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Processed: {stats['processed']}")
    print(f"Moved: {stats['moved']}")
    print(f"Errors: {stats['errors']}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    process_queue()
