#!/usr/bin/env python3
"""
HYBRID VISION WORKER - Groq + LMStudio Parallel Processing
- Uses both APIs simultaneously for maximum throughput
- Falls back if one is unavailable
- Hierarchical classification logic
- Better source detection
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
from paths import base_dir

# Configuration
BASE_DIR = Path(str(base_dir()))
QUEUE_DIR = BASE_DIR / "queue" / "realistic_female_influencer"
SORTED_DIR = BASE_DIR / "sorted" / "realistic_female_influencer"

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LMSTUDIO_URL = "http://localhost:1234/v1"

# Groq API keys (from existing setup)
GROQ_KEYS = [k.strip() for k in (os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
_current_key_idx = 0

# Ensure sorted dirs exist
CATEGORIES = ["InstagramInfluencer", "Professional", "Cinematic", "Vintage", "Fantasy", "Sports", "Amateur", "Unknown", "DISCARD"]
for cat in CATEGORIES:
    (SORTED_DIR / cat).mkdir(parents=True, exist_ok=True)

def detect_source(filename, metadata):
    """Detect source from filename + metadata"""
    filename_lower = filename.lower()
    
    # Check filename patterns
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
    elif "grok" in filename_lower or "promptsref_grok" in filename_lower:
        return "grok"
    elif "nanobanana" in filename_lower or "promptsref_nanobanana" in filename_lower:
        return "nanobanana"
    
    # Check metadata
    if metadata:
        source_channel = metadata.get("source_channel", "").lower()
        source_guild = metadata.get("source_guild", "").lower()
        
        if "civitai" in source_channel:
            return "civitai"
        elif "reddit" in source_channel:
            return "reddit"
        elif "zforfree" in source_channel or "zff" in source_channel:
            return "zforfree"
        elif "unstable" in source_guild or "ud" in source_guild:
            return "discord_ud"
        elif "midjourney" in source_guild or "mj" in source_guild:
            return "discord_mj"
        elif "grok" in source_channel or "grok" in source_guild:
            return "grok"
        elif "nanobanana" in source_channel or "promptsref" in source_channel:
            return "nanobanana"
        elif "twitter" in source_channel or "x.com" in source_channel:
            return "twitter_x"
    
    return "unknown"

def resize_image(img_bytes, max_size=1024*1024):
    """Resize image to fit within size limit"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        
        # If already small enough, return as JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        if buffer.tell() < max_size:
            return buffer.getvalue()
        
        # Resize until it fits
        w, h = img.size
        scale = (max_size / buffer.tell() * 0.8) ** 0.5
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        
        buffer_final = io.BytesIO()
        img.save(buffer_final, format="JPEG", quality=70)
        return buffer_final.getvalue()
    except:
        return None

def classify_with_groq(b64_image):
    """Classify using Groq API with key rotation"""
    global _current_key_idx
    
    try:
        # Rotate through keys for load balancing
        key = GROQ_KEYS[_current_key_idx % len(GROQ_KEYS)]
        _current_key_idx += 1
        
        client = Groq(api_key=key)
        
        prompt = """Analyze this image and respond with ONLY valid JSON (no markdown):
{
  "photorealistic": true/false,
  "ai_flaws": true/false,
  "woman_present": true/false,
  "nsfw": true/false,
  "quality": 1-10,
  "style": "portrait|selfie|fashion|editorial|cinematic|vintage|fantasy|sports|other",
  "reason": "One sentence"
}

RULES:
- photorealistic: True if photo-realistic or hyperrealistic. False if painting/cartoon/3D/anime.
- ai_flaws: True ONLY if obvious severe artifacts (melted faces, wrong limbs).
- woman_present: True if adult human female is main subject.
- nsfw: True ONLY if explicit nudity or sexual content.
- quality: 1-3 (poor), 4-6 (okay), 7-10 (excellent).
- style: Infer from composition."""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                        }
                    ]
                }
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content), "groq"
    except Exception as e:
        print(f"  [Groq Error] {str(e)[:60]}", flush=True)
        return None, "groq"

def classify_with_lmstudio(b64_image):
    """Classify using LMStudio local vision"""
    try:
        import requests
        
        prompt = """Analyze this image and respond with ONLY valid JSON (no markdown):
{
  "photorealistic": true/false,
  "ai_flaws": true/false,
  "woman_present": true/false,
  "nsfw": true/false,
  "quality": 1-10,
  "style": "portrait|selfie|fashion|editorial|cinematic|vintage|fantasy|sports|other",
  "reason": "One sentence"
}

RULES:
- photorealistic: True if photo-realistic or hyperrealistic. False if painting/cartoon/3D/anime.
- ai_flaws: True ONLY if obvious severe artifacts (melted faces, wrong limbs).
- woman_present: True if adult human female is main subject.
- nsfw: True ONLY if explicit nudity or sexual content.
- quality: 1-3 (poor), 4-6 (okay), 7-10 (excellent).
- style: Infer from composition."""
        
        response = requests.post(
            f"{LMSTUDIO_URL}/chat/completions",
            json={
                "model": "local-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 300
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content), "lmstudio"
        else:
            return None, "lmstudio"
    except Exception as e:
        print(f"  [LMStudio Error] {str(e)[:60]}", flush=True)
        return None, "lmstudio"

def classify_image_hybrid(img_bytes, source_name):
    """Classify using both APIs in parallel, use first result"""
    small_bytes = resize_image(img_bytes)
    if not small_bytes:
        return None, "DISCARD", "Image corrupt"
    
    b64 = base64.b64encode(small_bytes).decode('utf-8')
    
    # Try both APIs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        groq_future = executor.submit(classify_with_groq, b64)
        lm_future = executor.submit(classify_with_lmstudio, b64)
        
        # Get first successful result
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
    # NSFW check
    elif nsfw:
        category = "InstagramInfluencer" if quality >= 7 else "DISCARD"
    # InstagramInfluencer (priority 1)
    elif style in ["selfie", "portrait"] and quality >= 6:
        category = "InstagramInfluencer"
    # Then try other categories
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
    
    # Build result with source tracking
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
        "reason": result.get("reason", ""),
        "classified_at": __import__("datetime").datetime.utcnow().isoformat()
    }
    
    return result_data, category, result.get("reason", "")

def reclassify_all():
    """Reclassify all sorted files with new hierarchical logic"""
    print("=" * 70)
    print("RECLASSIFYING ALL FILES - HYBRID GROQ + LMSTUDIO")
    print("=" * 70)
    print("")
    
    all_images = list(SORTED_DIR.rglob("*.jpg")) + list(SORTED_DIR.rglob("*.png"))
    print(f"Found {len(all_images)} images to reclassify")
    print("")
    
    stats = {
        "processed": 0,
        "errors": 0,
        "reclassified": 0,
        "groq_used": 0,
        "lmstudio_used": 0,
        "unchanged": 0
    }
    
    start_time = time.time()
    
    for i, img_path in enumerate(all_images):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = i / max(elapsed, 0.1)
            remaining = (len(all_images) - i) / max(rate, 0.1)
            print(f"[{i}/{len(all_images)}] ETA: {remaining:.0f}s | Rate: {rate:.1f}/s", flush=True)
        
        try:
            # Read image
            img_bytes = img_path.read_bytes()
            if not img_bytes:
                continue
            
            # Get old source from current path
            old_source = img_path.parent.name
            
            # Read metadata for better source detection
            stem = img_path.stem
            meta_path = img_path.parent / f"{stem}.meta.json"
            metadata = {}
            if meta_path.exists():
                try:
                    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
                except:
                    pass
            
            # Detect correct source
            correct_source = detect_source(img_path.name, metadata)
            
            # Classify with hybrid logic
            result_data, new_category, reason = classify_image_hybrid(img_bytes, correct_source)
            
            if result_data is None:
                stats["errors"] += 1
                continue
            
            # Track which API was used
            if result_data.get("api_used") == "groq":
                stats["groq_used"] += 1
            else:
                stats["lmstudio_used"] += 1
            
            # Get old category
            old_category = [p.name for p in img_path.parents if p.parent == SORTED_DIR][0]
            
            # If category or source changed, move file
            if new_category != old_category or correct_source != old_source:
                new_path = SORTED_DIR / new_category / correct_source
                new_path.mkdir(parents=True, exist_ok=True)
                
                # Move image
                dest_img = new_path / img_path.name
                img_path.rename(dest_img)
                
                # Move related files
                for ext in [".txt", ".meta.json"]:
                    src_meta = img_path.with_stem(stem).with_suffix(ext)
                    if src_meta.exists():
                        dst_meta = new_path / src_meta.name
                        src_meta.rename(dst_meta)
                
                # Save new vision.json
                vision_path = new_path / f"{stem}.vision.json"
                vision_path.write_text(json.dumps(result_data, indent=2), encoding='utf-8')
                
                stats["reclassified"] += 1
            else:
                # Update vision.json even if not moving
                vision_path = img_path.with_stem(stem).with_suffix(".vision.json")
                vision_path.write_text(json.dumps(result_data, indent=2), encoding='utf-8')
                stats["unchanged"] += 1
            
            stats["processed"] += 1
        
        except Exception as e:
            stats["errors"] += 1
    
    elapsed = time.time() - start_time
    
    print("")
    print("=" * 70)
    print("RECLASSIFICATION COMPLETE")
    print(f"  Processed: {stats['processed']}")
    print(f"  Reclassified: {stats['reclassified']}")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  API Usage:")
    print(f"    Groq: {stats['groq_used']}")
    print(f"    LMStudio: {stats['lmstudio_used']}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    reclassify_all()
