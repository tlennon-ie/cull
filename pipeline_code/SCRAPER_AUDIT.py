#!/usr/bin/env python3
"""
COMPREHENSIVE SCRAPER AUDIT
- Verify each scraper's integration with queue_manager
- Check for broken image downloads
- Report missing images vs metadata
- Identify orphaned metadata files
"""
import json, os, requests, tempfile
from pathlib import Path
from collections import defaultdict
from paths import base_dir

BASE_DIR = Path(str(base_dir()))
QUEUE_DIR = BASE_DIR / "queue" / "realistic_female_influencer"

print("=" * 80)
print("SCRAPER AUDIT REPORT")
print("=" * 80)
print()

# ── SECTION 1: Queue Structure Analysis ────────────────────────────────────
print("1. QUEUE STRUCTURE")
print("-" * 80)

sources = {}
for source_dir in QUEUE_DIR.iterdir():
    if source_dir.is_dir():
        images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        txts = list(source_dir.glob("*.txt"))
        metas = list(source_dir.glob("*.meta.json"))
        jsons = list(source_dir.glob("*.json"))
        
        sources[source_dir.name] = {
            "images": len(images),
            "txt": len(txts),
            "meta": len(metas),
            "json": len(jsons),
            "total_files": len(list(source_dir.iterdir()))
        }
        
        print(f"\n{source_dir.name}/")
        print(f"  Images:       {len(images):3d}")
        print(f"  Text files:   {len(txts):3d}")
        print(f"  Meta JSON:    {len(metas):3d}")
        print(f"  Total files:  {len(list(source_dir.iterdir())):3d}")
        
        # Check for missing images
        orphaned_txt = [f.name for f in txts if f.stem not in [img.stem for img in images]]
        orphaned_meta = [f.name for f in metas if f.stem.replace('.meta', '') not in [img.stem for img in images]]
        
        if orphaned_txt:
            print(f"  ⚠️  Orphaned TXT: {len(orphaned_txt)}")
        if orphaned_meta:
            print(f"  ⚠️  Orphaned META: {len(orphaned_meta)}")

print("\n" + "=" * 80)
print("2. CRITICAL FINDINGS")
print("-" * 80)

total_images = sum(s["images"] for s in sources.values())
total_metadata = sum(s["meta"] + s["txt"] for s in sources.values())

print(f"\nTotal images in queue:   {total_images}")
print(f"Total metadata files:    {total_metadata}")

if total_images == 0:
    print("\n🔴 CRITICAL: NO IMAGES FOUND IN ANY QUEUE FOLDER!")
    print("   Metadata exists but images are missing.")
    print("   This indicates image downloads are failing silently.")
else:
    print(f"\n✅ Images found: {total_images}")

# ── SECTION 3: Test Download Function ──────────────────────────────────────
print("\n" + "=" * 80)
print("3. TESTING IMAGE DOWNLOADS")
print("-" * 80)

# Get a sample image URL from metadata
sample_urls = []
for source_dir in QUEUE_DIR.iterdir():
    if source_dir.is_dir():
        metas = list(source_dir.glob("*.meta.json"))
        if metas:
            try:
                meta_data = json.loads(metas[0].read_text())
                url = meta_data.get("image_url")
                if url:
                    sample_urls.append((source_dir.name, url, metas[0].name))
                    if len(sample_urls) >= 3:
                        break
            except:
                pass

print(f"\nTesting {len(sample_urls)} sample URLs:\n")
for source, url, meta_file in sample_urls:
    print(f"Source: {source}")
    print(f"  Meta file: {meta_file}")
    print(f"  URL: {url[:80]}...")
    
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            size_kb = len(r.content) / 1024
            print(f"  ✅ Download: SUCCESS ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ Download: FAILED (status {r.status_code})")
    except Exception as e:
        print(f"  ❌ Download: ERROR ({e})")
    print()

# ── SECTION 4: Scraper Status ──────────────────────────────────────────────────
print("=" * 80)
print("4. SCRAPER SOURCE STATUS")
print("-" * 80)

scrapers = {
    "civitai": "scraper_civitai.py (tRPC API)",
    "civitai_search": "scraper_civitai_search.py (Meilisearch)",
    "discord_ud": "scraper_discord.py (Unstable Diffusion)",
    "discord_mj": "scraper_discord.py (Midjourney)",
    "reddit": "scraper_web.py (Reddit)",
    "twitter_x": "scraper_x.py (X.com)",
    "nanobanana": "scraper_web.py (PromptSref)",
    "zforfree": "scraper_web.py (ZForFree)"
}

print("\nScraper integration check:\n")
for source, desc in scrapers.items():
    found = source in sources
    images = sources.get(source, {}).get("images", 0) if found else 0
    metadata = sources.get(source, {}).get("meta", 0) if found else 0
    
    status = "✅ ACTIVE" if found and (images + metadata > 0) else "⚠️  EMPTY" if found else "❌ MISSING"
    print(f"{source:15} {status:12} Images: {images:3d} | Metadata: {metadata:3d}")
    print(f"                 └─ {desc}")

# ── SECTION 5: Recommendations ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("5. RECOMMENDED FIXES")
print("-" * 80)

recommendations = []

if total_images == 0:
    recommendations.append(
        "🔴 CRITICAL: Fix image download failures in queue_manager.save_to_queue()\n"
        "   - Add error logging to see why shutil.copy2() is failing\n"
        "   - Check temp file is being created with content before copy\n"
        "   - Verify download_image() functions are returning True"
    )

# Check for orphaned metadata
for source, stats in sources.items():
    if stats["meta"] > 0 and stats["images"] == 0:
        recommendations.append(
            f"🟡 {source}: {stats['meta']} metadata files but 0 images\n"
            f"   - Images were downloaded but not copied to queue\n"
            f"   - Check scraper_*.py download_image() function"
        )
    elif stats["txt"] > stats["meta"]:
        recommendations.append(
            f"🟡 {source}: More TXT files ({stats['txt']}) than META ({stats['meta']})\n"
            f"   - Some prompts missing metadata\n"
            f"   - May indicate incomplete saves"
        )

if not recommendations:
    recommendations.append(
        "✅ No critical issues found (but verify image downloads are actually working)"
    )

for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec}")

# ── SECTION 6: Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("6. SUMMARY")
print("-" * 80)

summary = {
    "Total sources active": len([s for s in sources if sources[s]["images"] > 0 or sources[s]["meta"] > 0]),
    "Total queue items": total_metadata,
    "Total images": total_images,
    "Images/Metadata ratio": f"{total_images}/{total_metadata}" if total_metadata > 0 else "0/0",
    "Status": "🔴 CRITICAL - No images found" if total_images == 0 else "⚠️  WARNING - Mismatch detected" if total_images < total_metadata * 0.5 else "✅ HEALTHY"
}

for key, value in summary.items():
    print(f"{key:.<40} {value}")

print("\n" + "=" * 80)
