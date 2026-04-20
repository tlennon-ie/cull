#!/usr/bin/env python3
"""
IMPROVED VISION WORKER - Hierarchical Classification
- No prompt bias (image-only analysis)
- Better source detection from filename + metadata
- Hierarchical categories:
  * InstagramInfluencer (priority 1)
  * Then if not IG: Professional, Cinematic, Vintage, Fantasy, Sports, Amateur
  * Then if none fit: Unknown
  * DISCARD if: not photorealistic, no woman, severe AI flaws
- Better classification rules
"""
import os
import sys
import json
import base64
import io
from pathlib import Path
from PIL import Image
from groq import Groq

# Configuration
BASE_DIR = Path("I:/AI/openclaw/workspace/prompt-library")
QUEUE_DIR = BASE_DIR / "queue" / "realistic_female_influencer"
SORTED_DIR = BASE_DIR / "sorted" / "realistic_female_influencer"
MODEL_ID = "llama-2-vision-90b"

# Ensure sorted dirs exist
CATEGORIES = ["InstagramInfluencer", "Professional", "Cinematic", "Vintage", "Fantasy", "Sports", "Amateur", "Unknown", "DISCARD"]
for cat in CATEGORIES:
    (SORTED_DIR / cat).mkdir(parents=True, exist_ok=True)

SOURCES = {
    "civitai": "civitai",
    "zforfree": "zforfree", 
    "discord_ud": "discord_ud",
    "discord_mj": "discord_mj",
    "reddit": "reddit",
    "twitter_x": "twitter_x",
    "grok": "grok",
    "nanobanana": "nanobanana",
}

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

def classify_image(image_path, source_name):
    """Classify image using improved hierarchical logic"""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # Read image and metadata
    img_bytes = image_path.read_bytes()
    small_bytes = resize_image(img_bytes)
    if not small_bytes:
        return None, "DISCARD", "Image corrupt"
    
    # Read metadata
    stem = image_path.stem
    meta_path = image_path.parent / f"{stem}.meta.json"
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        except:
            pass
    
    b64 = base64.b64encode(small_bytes).decode('utf-8')
    
    # Classification prompt - NO prompt text, only image analysis
    prompt = """Analyze this image and respond with ONLY valid JSON (no markdown):
{
  "photorealistic": true/false,
  "ai_flaws": true/false,
  "woman_present": true/false,
  "nsfw": true/false,
  "quality": 1-10,
  "style": "portrait|selfie|fashion|editorial|cinematic|vintage|fantasy|sports|other",
  "reason": "One sentence explanation"
}

RULES:
- photorealistic: True if photo-realistic or hyperrealistic. False if painting/cartoon/3D/anime.
- ai_flaws: True ONLY if obvious severe artifacts (melted faces, wrong limbs, duplicate body parts).
- woman_present: True if adult human female is the main subject.
- nsfw: True ONLY if explicit nudity or sexual content.
- quality: 1-3 (poor), 4-6 (okay), 7-10 (excellent).
- style: Infer from visual composition (portrait for headshots, selfie for mirror/phone POV, fashion for clothing-focused, editorial for studio, cinematic for dramatic lighting/scene, vintage for retro aesthetic, fantasy for stylized/otherworldly, sports for athletic, other for ambiguous).

DISCARD if: NOT photorealistic OR NOT woman OR severe AI flaws."""

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        }
                    ]
                }
            ],
            model=MODEL_ID,
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        photorealistic = result.get("photorealistic", False)
        ai_flaws = result.get("ai_flaws", False)
        woman_present = result.get("woman_present", False)
        nsfw = result.get("nsfw", False)
        quality = result.get("quality", 5)
        style = result.get("style", "other").lower()
        
        # HIERARCHICAL LOGIC
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
            "reason": result.get("reason", ""),
            "classified_at": __import__("datetime").datetime.utcnow().isoformat()
        }
        
        return result_data, category, result.get("reason", "")
    
    except Exception as e:
        print(f"Classification error: {e}")
        return None, "DISCARD", f"Error: {str(e)[:50]}"

def reclassify_all():
    """Reclassify all sorted files with new logic"""
    print("=" * 70)
    print("RECLASSIFYING ALL FILES WITH NEW HIERARCHICAL LOGIC")
    print("=" * 70)
    print("")
    
    all_images = list(SORTED_DIR.rglob("*.jpg")) + list(SORTED_DIR.rglob("*.png"))
    print(f"Found {len(all_images)} images to reclassify")
    print("")
    
    stats = {"processed": 0, "errors": 0}
    moved = 0
    
    for i, img_path in enumerate(all_images):
        if i % 100 == 0:
            print(f"Processed {i}/{len(all_images)}...", flush=True)
        
        try:
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
            
            # Classify with new logic
            result_data, new_category, reason = classify_image(img_path, correct_source)
            
            if result_data is None:
                continue
            
            # If category or source changed, move file
            old_category = [p.name for p in img_path.parents if p.parent == SORTED_DIR][0]
            
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
                
                moved += 1
            else:
                # Update vision.json even if not moving
                vision_path = img_path.with_stem(stem).with_suffix(".vision.json")
                vision_path.write_text(json.dumps(result_data, indent=2), encoding='utf-8')
            
            stats["processed"] += 1
        
        except Exception as e:
            stats["errors"] += 1
    
    print("")
    print("=" * 70)
    print("RECLASSIFICATION COMPLETE")
    print(f"  Processed: {stats['processed']}")
    print(f"  Reclassified: {moved}")
    print(f"  Errors: {stats['errors']}")
    print("=" * 70)

if __name__ == "__main__":
    reclassify_all()
