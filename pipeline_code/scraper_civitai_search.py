"""
scraper_civitai_search.py - Scrape Civitai via Meilisearch API (search-new.civitai.com)
Uses specific search queries (e.g. "female influencer") + filters for high relevance.
Saves to source-based queue using queue_manager for balanced processing.
"""
import json, os, time, requests, re, tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from paths import base_dir

BASE_DIR  = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
TOPIC     = os.environ.get("PIPELINE_TOPIC", "Realistic Female Influencer")
SLUG      = os.environ.get("PIPELINE_SLUG",  "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
QUEUE_DIR = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG

# Import queue manager for source-based saving
from queue_manager import save_to_queue

# Support both civitai.com (clean) and civitai.red (NSFW). Per-domain seen file so parallel runs don't collide.
CIVITAI_DOMAIN = os.getenv("CIVITAI_DOMAIN", "civitai.com")
if CIVITAI_DOMAIN not in {"civitai.com", "civitai.red"}:
    CIVITAI_DOMAIN = "civitai.com"
SOURCE_KEY = "civitai_red" if CIVITAI_DOMAIN == "civitai.red" else "civitai"
SEEN_FILE = BASE_DIR / f"seen_civitai_search_{CIVITAI_DOMAIN.replace('.', '_')}_{SLUG}.json"

# civitai.red is a UI rebrand - there is no dedicated `search-new.civitai.red`
# Meilisearch host. Both domains share civitai.com's search backend. Admins can
# still override via CIVITAI_SEARCH_URL if they have a custom mirror.
_SEARCH_HOST = os.environ.get("CIVITAI_SEARCH_HOST", "search-new.civitai.com").strip()
SEARCH_URL = os.environ.get(
    "CIVITAI_SEARCH_URL",
    f"https://{_SEARCH_HOST}/multi-search?x-meilisearch-client=Meilisearch%20instant-meilisearch%20(v0.13.5)",
)
CDN_BASE   = os.environ.get("CIVITAI_CDN_BASE", "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA")

# Civitai.red runs on a separate auth backend, so the admin can supply a dedicated
# key via CIVITAI_API_RED_KEY. Fall back to the .com key if only one is set.
if CIVITAI_DOMAIN == "civitai.red":
    TOKEN = os.getenv("CIVITAI_API_RED_KEY", "").strip() or os.getenv("CIVITAI_API_KEY", "").strip()
else:
    TOKEN = os.getenv("CIVITAI_API_KEY", "").strip()

if not TOKEN:
    raise ValueError(
        f"No API key available for Civitai domain '{CIVITAI_DOMAIN}'. "
        f"Set CIVITAI_API_KEY (and optionally CIVITAI_API_RED_KEY) in .env."
    )

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

print(
    f"[Civitai Scraper] Domain={CIVITAI_DOMAIN} | Key=...{TOKEN[-4:]} | "
    f"Search host={_SEARCH_HOST} | URL={SEARCH_URL[:90]}...",
    flush=True,
)

# Supported Base Models (from user list)
BASE_MODELS = [
    "Flux.2 D", "Flux.2 Klein 4B", "Flux.2 Klein 4B-base", 
    "Flux.2 Klein 9B", "Flux.2 Klein 9B-base", "Qwen", 
    "ZImageBase", "ZImageTurbo", "Kling", "Nano Banana", 
    "PixArt E", "Sora 2", "Veo 3", "Imagen4", "OpenAI"
]

# Filters
# Only images, only Portrait, created in last ~14 days (1759273200000 approx), NSFW allowed
FILTER_EXPRESSION = [
    'aspectRatio = "Portrait"',
    [f'baseModel = "{m}"' for m in BASE_MODELS],
    'type = "image"',
    # "createdAtUnix >= 1759273200000", # Can adjust or remove for broader history
    # "createdAtUnix <= 1772236800000",
    "(poi != true OR user.username = TL_) AND (minor != true) AND (nsfwLevel IN [1, 2, 4, 8, 16])"
]

MAX_PAGES = 40  # 50 * 40 = 2000 images
LIMIT     = 50

LEGACY_SEEN_FILE = BASE_DIR / f"seen_civitai_search_{SLUG}.json"
SORTED_ROOT = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted")))
_SORTED_SLUG_DIR = SORTED_ROOT if SORTED_ROOT.name == SLUG else SORTED_ROOT / SLUG


def sorted_id_index() -> set[str]:
    """Scan <sorted>/<slug>/**/civitai*/*.jpg|*.png and extract civitai IDs.

    Sorted filenames look like `<cat>_civitai_<id>_<ts>_<rand>.<ext>`, so we
    pull the number that follows 'civitai_'. This is a safety net for when the
    seen file was lost or never written.
    """
    ids: set[str] = set()
    if not _SORTED_SLUG_DIR.exists():
        return ids
    pat = re.compile(r"civitai_(\d+)")
    for category in _SORTED_SLUG_DIR.iterdir():
        if not category.is_dir():
            continue
        for source_dir in category.iterdir():
            if not source_dir.is_dir() or "civitai" not in source_dir.name:
                continue
            for f in source_dir.iterdir():
                m = pat.search(f.name)
                if m:
                    ids.add(m.group(1))
    return ids


def load_seen():
    """Merge per-domain seen + legacy (domain-less) seen so older dedup lists aren't lost.

    Before the civitai.com / civitai.red split we stored one file per slug. That
    legacy file still holds thousands of IDs that were already downloaded and
    sorted; ignoring it would cause the scraper to re-queue them.
    """
    seen: set[str] = set()
    for path in (LEGACY_SEEN_FILE, SEEN_FILE):
        if path.exists():
            try:
                seen |= set(json.loads(path.read_text(encoding='utf-8')))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  [seen] skipping {path.name}: {exc}", flush=True)
    sorted_ids = sorted_id_index()
    if sorted_ids:
        print(f"  [seen] +{len(sorted_ids - seen)} ids recovered from sorted/ on disk", flush=True)
        seen |= sorted_ids
    return seen

def save_seen(seen):
    SEEN_FILE.write_text(json.dumps(sorted(seen)), encoding='utf-8')

def build_search_payload(query: str, offset: int = 0):
    return {
        "queries": [
            {
                "q": query,
                "indexUid": "images_v6",
                "facets": ["aspectRatio", "baseModel", "createdAtUnix", "tagNames", "type", "user.username"],
                "limit": LIMIT,
                "sort": ["createdAt:desc"],
                "offset": offset,
                "filter": FILTER_EXPRESSION
            }
        ]
    }

def download_image(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=30, stream=True)
        if r.ok:
            data = r.content
            if len(data) > 5000:
                dest.write_bytes(data)
                return True
    except Exception as e:
        print(f"  Download err: {e}")
    return False

def clean_prompt(text: str) -> str:
    if not text: return ""
    # Remove ComfyUI node noise if present
    text = re.sub(r'Positive Prompt:|Negative Prompt:', '', text, flags=re.I)
    return text.strip()

def scrape_civitai_search(seen: set):
    print("Starting scrape_civitai_search...")
    # Derive search query from topic
    # e.g. "Realistic Female Influencer" -> "female influencer"
    query_terms = TOPIC.lower().replace("realistic", "").replace("prompt", "").split()
    query = " ".join(query_terms).strip() or "female influencer"
    
    print(f"=== Civitai Meilisearch Scraper ===")
    print(f"Query: '{query}'")
    print(f"Models: {len(BASE_MODELS)} allowed")
    
    saved_count = 0
    
    for page in range(MAX_PAGES):
        offset = page * LIMIT
        payload = build_search_payload(query, offset)
        
        try:
            r = requests.post(SEARCH_URL, headers=HEADERS, json=payload, timeout=20)
            if r.status_code == 429:
                print(f"  [429] Rate limited, sleeping 20s...")
                time.sleep(20)
                continue
            if not r.ok:
                print(f"  [Error] {r.status_code}: {r.text[:200]}")
                break
                
            data = r.json()
            # print(json.dumps(data, indent=2)[:500]) # Debug print
            
            # The structure is usually {"results": [{"hits": [...]}]}
            results = data.get("results", [])
            if not results: 
                print("  [Warn] No 'results' key in response")
                break
            
            hits = results[0].get("hits", [])
            print(f"  [Page {page+1}] Offset {offset}: Found {len(hits)} hits")
            
            if not hits:
                print("  No more hits.")
                break
                
            for hit in hits:
                iid = str(hit.get("id"))
                if iid in seen:
                    continue
                
                # Extract image URL
                # Logic: CDN_BASE / {url} / original=true,quality=90 / {name}
                url_part = hit.get("url")
                name_part = hit.get("name")
                
                if not url_part or not name_part:
                    continue
                    
                img_url = f"{CDN_BASE}/{url_part}/original=true,quality=90/{name_part}"
                
                # Extract prompt
                meta = hit.get("meta") or {}
                # Meilisearch hit often has 'prompt' at top level too?
                # User's visualization script used `h.prompt` or `h.meta.prompt`?
                # The script says: `const promptFull = (h?.prompt ?? '');`
                prompt = hit.get("prompt") or meta.get("prompt") or ""
                prompt = clean_prompt(prompt)
                
                # Fallback for empty prompts (check ComfyUI data inside meta if available)
                if not prompt or len(prompt) < 15:
                    # Search hits might not have full Comfy JSON, but check just in case
                    # Often search index prompts are truncated or missing if workflow-only
                    continue
                    
                seen.add(iid)
                
                stem = f"civitai_{iid}"
                
                # Download to temp file first
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                
                if download_image(img_url, tmp_path):
                    # Prepare metadata
                    meta_data = {
                        "message_id": stem,
                        "image_url": img_url,
                        "civitai_id": iid,
                        "source_channel": f"civitai_search_{CIVITAI_DOMAIN}",
                        "source_guild": CIVITAI_DOMAIN,
                        "base_model": hit.get("baseModel"),
                        "nsfw_level": hit.get("nsfwLevel"),
                        "user": hit.get("user", {}).get("username"),
                        "stats": hit.get("stats", {}),
                        "pipeline_topic": TOPIC,
                    }

                    # Save to source-based queue using queue_manager (civitai / civitai_red sub-folders)
                    result = save_to_queue(SOURCE_KEY, tmp_path, prompt, meta_data)
                    
                    if result:
                        print(f"    SAVED {stem} | {prompt[:60]}...")
                        saved_count += 1
                    else:
                        print(f"    FAILED {stem} - save_to_queue returned None")
                        seen.discard(iid)
                else:
                    tmp_path.unlink(missing_ok=True)
                
                time.sleep(0.1) # Polite delay
                
            save_seen(seen)
            time.sleep(1) # Delay between pages
            
        except Exception as e:
            print(f"  [Exception] {e}")
            break
            
    print(f"=== Done. Saved {saved_count} images. ===")

if __name__ == "__main__":
    seen = load_seen()
    scrape_civitai_search(seen)
