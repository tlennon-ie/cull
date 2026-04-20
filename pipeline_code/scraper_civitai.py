"""
scraper_civitai.py - Scrape Civitai images + prompts via tRPC API
Uses the same filters as the Civitai web UI:
  - baseModels: ZImageBase, ZImageTurbo, Flux.2 Klein 9B, Flux.2 D, Qwen
  - browsingLevel: 31 (full NSFW)
  - excludedTagIds: anime/cartoon/minor exclusions
  - withMeta: true (only images with prompt metadata)
  - sort: Most Collected, period: Month

Prompt is fetched per-image via image.getGenerationData tRPC endpoint.
Images are downloaded from Civitai CDN.
"""
import json, os, time, requests
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

BASE_DIR  = Path(os.environ.get("PIPELINE_BASE_DIR", "I:/AI/openclaw/workspace/prompt-library"))
TOPIC     = os.environ.get("PIPELINE_TOPIC", "Realistic Female Influencer")
SLUG      = os.environ.get("PIPELINE_SLUG",  "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
QUEUE_DIR = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG
_DOMAIN = os.environ.get("CIVITAI_DOMAIN", "civitai.com")
SEEN_FILE = BASE_DIR / f"seen_civitai_{_DOMAIN.replace('.', '_')}_{SLUG}.json"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)

CDN_BASE  = os.environ.get("CIVITAI_CDN_BASE", "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA")
TRPC_BASE = os.environ.get("CIVITAI_TRPC_BASE", f"https://{_DOMAIN}/api/trpc")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# Exact filter params from the Civitai web UI
BROWSE_PARAMS = {
    "period":         "Month",
    "periodMode":     "published",
    "sort":           "Most Collected",
    "types":          ["image"],
    "withMeta":       True,
    "fromPlatform":   False,
    "baseModels":     ["ZImageBase", "ZImageTurbo", "Flux.2 Klein 9B", "Flux.2 D", "Flux.2 Klein 9B-base", "Qwen"],
    "useIndex":       True,
    "browsingLevel":  31,
    "include":        ["cosmetics"],
    "excludedTagIds": [415792, 426772, 5188, 5249, 130818, 130820, 133182,
                       5351, 306619, 154326, 161829, 163032],
    "disablePoi":     True,
    "disableMinor":   True,
    "cursor":         None,
    "authed":         True,
}

MAX_PAGES = 40   # 50 items/page → up to 2000 images
MIN_PROMPT = 15


LEGACY_SEEN_FILE = BASE_DIR / f"seen_civitai_{SLUG}.json"

def load_seen():
    """Merge per-domain seen with the pre-split legacy file so we don't re-queue already-sorted IDs."""
    seen: set[str] = set()
    for path in (LEGACY_SEEN_FILE, SEEN_FILE):
        if path.exists():
            try:
                seen |= set(json.loads(path.read_text(encoding='utf-8')))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"  [seen] skipping {path.name}: {exc}", flush=True)
    return seen

def save_seen(seen):
    SEEN_FILE.write_text(json.dumps(sorted(seen)), encoding='utf-8')


def trpc_get(endpoint: str, input_obj: dict) -> dict:
    """Call a Civitai tRPC GET endpoint."""
    payload = json.dumps({"json": input_obj, "meta": {"values": {"cursor": ["undefined"]}}})
    url = f"{TRPC_BASE}/{endpoint}?input={requests.utils.quote(payload)}"
    
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 429:
                print(f"  [429] Rate limited on {endpoint}, sleeping 20s...")
                time.sleep(20)
                continue # Retry
            if not r.ok:
                print(f"  tRPC {endpoint} error {r.status_code}")
                return {}
            return r.json().get("result", {}).get("data", {}).get("json", {})
        except Exception as e:
            print(f"  tRPC error: {e}")
            return {}
    return {}


def get_prompt(image_id: int) -> str | None:
    """Fetch prompt for a single image via image.getGenerationData."""
    data = trpc_get("image.getGenerationData", {"id": image_id})
    if not data:
        # print(f"    [{image_id}] No data from tRPC")
        return None
    meta = data.get("meta") or {}
    prompt = (meta.get("prompt") or meta.get("Prompt") or "").strip()
    
    if not prompt:
        # Try to find ComfyUI prompt
        comfy = meta.get("comfy") or {}
        workflow = comfy.get("prompt") or {}
        candidates = []
        for node_id, node_data in workflow.items():
            inputs = node_data.get("inputs") or {}
            # Check common prompt fields in Comfy nodes
            for field in ["text", "text_g", "text_l", "populated_text", "body_text", "multiline_text", "wildcard_text"]:
                val = inputs.get(field)
                if val and isinstance(val, str) and len(val) > 20:
                    candidates.append(val)
        
        if candidates:
            # Join all unique candidates or pick the longest? 
            # Comfy workflows often have multiple parts (pos, neg, wildcards). 
            # Let's join them to be safe, filtering duplicates.
            unique_candidates = sorted(list(set(candidates)), key=len, reverse=True)
            # Take the longest one as the primary prompt, or join top 3 if fragments
            prompt = unique_candidates[0] 

    if not prompt:
        # print(f"    [{image_id}] Empty prompt field (keys: {list(meta.keys())})")
        return None
        
    if len(prompt) < MIN_PROMPT:
        print(f"    [{image_id}] Prompt too short: {len(prompt)}")
        return None
        
    return prompt


def build_image_url(uuid: str, mime: str = None) -> str:
    ext = "original=true"
    return f"{CDN_BASE}/{uuid}/{ext}"


def download_image(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        if r.ok:
            data = r.content
            if len(data) > 5000:
                dest.write_bytes(data)
                return True
    except Exception as e:
        print(f"  Download err: {e}")
    return False


def scrape_pages(seen: set) -> int:
    saved = 0
    cursor = None
    params = dict(BROWSE_PARAMS)

    print(f"=== Civitai tRPC scraper ===")
    print(f"Base models: {params['baseModels']}")
    print(f"browsingLevel: {params['browsingLevel']} | excludedTags: {len(params['excludedTagIds'])}")

    for page in range(MAX_PAGES):
        params["cursor"] = cursor

        # Build input for getInfinite
        input_obj = dict(params)
        meta_values = {}
        if cursor is None:
            meta_values["cursor"] = ["undefined"]

        payload = json.dumps({"json": input_obj, "meta": {"values": meta_values} if meta_values else {}})
        url = f"{TRPC_BASE}/image.getInfinite?input={requests.utils.quote(payload)}"

        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 429:
                print(f"  [page {page+1}] Rate limited, sleeping 15s...")
                time.sleep(15)
                continue
            if not r.ok:
                print(f"  [page {page+1}] Error {r.status_code}")
                break
            data = r.json().get("result", {}).get("data", {}).get("json", {})
        except Exception as e:
            print(f"  [page {page+1}] Fetch error: {e}")
            break

        items = data.get("items", [])
        next_cursor = data.get("nextCursor")

        print(f"  [page {page+1}] {len(items)} items | cursor={cursor}")

        if not items:
            break

        for item in items:
            image_id  = item.get("id")
            uuid      = item.get("url", "")
            base_model = item.get("baseModel", "")
            nsfw_level = item.get("nsfwLevel", 0)
            collected  = item.get("collectedCount", 0)
            mime       = item.get("mimeType", "image/jpeg")

            if not image_id or not uuid or str(image_id) in seen:
                continue

            # Fetch prompt (one API call per image)
            prompt = get_prompt(image_id)
            if not prompt:
                print(f"    [{image_id}] no prompt — skip")
                continue

            seen.add(str(image_id))

            # Determine extension
            ext = ".jpg"
            if mime == "image/png" or "png" in uuid.lower(): ext = ".png"
            elif mime == "image/webp": ext = ".webp"

            img_url   = build_image_url(uuid, mime)
            stem      = f"civitai_{image_id}"
            img_path  = QUEUE_DIR / f"{stem}{ext}"
            txt_path  = QUEUE_DIR / f"{stem}.txt"
            meta_path = QUEUE_DIR / f"{stem}.meta.json"

            if not download_image(img_url, img_path):
                # Try width fallback
                img_url2 = f"{CDN_BASE}/{uuid}/width=1024"
                if not download_image(img_url2, img_path):
                    seen.discard(str(image_id))
                    print(f"    [{image_id}] download failed — skip")
                    continue

            txt_path.write_text(prompt, encoding="utf-8")
            meta_path.write_text(json.dumps({
                "message_id":     stem,
                "image_url":      img_url,
                "source_channel": f"civitai_trpc_{_DOMAIN}",
                "source_guild":   _DOMAIN,
                "civitai_id":     str(image_id),
                "base_model":     base_model,
                "nsfw_level":     nsfw_level,
                "collected":      collected,
                "pipeline_topic": TOPIC,
                "prompt":         prompt,
            }, indent=2), encoding="utf-8")

            size_kb = img_path.stat().st_size // 1024
            print(f"    SAVED {stem}{ext} ({size_kb}KB) [{base_model}] nsfw={nsfw_level} | {prompt[:70]}".encode('ascii','replace').decode())
            saved += 1

            time.sleep(0.3)  # polite rate limit per prompt fetch

        save_seen(seen)

        if not next_cursor:
            print("  No more pages.")
            break

        cursor = next_cursor
        time.sleep(0.5)

    print(f"\n=== Civitai done. Total saved: {saved} ===")
    return saved


def main():
    seen = load_seen()
    print(f"Loaded {len(seen)} already-seen IDs")
    scrape_pages(seen)


if __name__ == "__main__":
    main()
