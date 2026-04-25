"""
scraper_web.py - Reddit + ZforFree + promptsref.com scraper
ALL sources download both image AND prompt to queue/, never just text.
Saves to source-based queue using queue_manager for balanced processing.

Sources:
  1. Reddit search (topic-aware queries, downloads linked images)
  2. ZforFree API (paginated, downloads image + prompt pairs)
  3. promptsref.com/library/nano-banana-pro + /library/grok (structured JSON prompts)
"""
import re, json, os, html, time, hashlib, requests, tempfile
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from topic_filter import load_config as _load_topic_config, passes as _topic_passes
from paths import base_dir

BASE_DIR  = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
TOPIC     = os.environ.get("PIPELINE_TOPIC", "Realistic Female Influencer")
SLUG      = os.environ.get("PIPELINE_SLUG",  "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
QUEUE_DIR = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG
ZFORFREE_WEB_ENABLED = os.environ.get("ZFORFREE_WEB_ENABLED", "false").lower() == "true"

# Import queue manager
from queue_manager import save_to_queue as queue_save

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

IMG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]

def download_image(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=IMG_HEADERS, timeout=25, stream=True)
        if r.ok and len(r.content) > 5000:
            dest.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"    Download err: {e}")
    return False

def save_to_queue(stem: str, img_url: str, img_bytes: bytes, prompt: str, meta: dict, source: str = "unknown") -> bool:
    """Save image + txt + meta.json to source-based queue. Returns True on success."""
    if not img_bytes or len(img_bytes) < 5000:
        return False
    if not prompt or len(prompt.strip()) < 30:
        return False

    ext = "jpg"
    url_lower = img_url.lower().split("?")[0]
    if url_lower.endswith(".png"):   ext = "png"
    elif url_lower.endswith(".webp"): ext = "webp"
    elif url_lower.endswith(".gif"):  return False  # skip gifs

    # Create temp file
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    tmp_path.write_bytes(img_bytes)

    # Prepare metadata
    meta_data = {
        "message_id": stem,
        "image_url": img_url,
        "pipeline_topic": TOPIC,
        **meta,
    }

    # Save to source-based queue
    result = queue_save(source, tmp_path, prompt.strip(), meta_data)

    if result:
        size_kb = len(img_bytes) // 1024
        print(f"  SAVED {result.name} ({size_kb}KB) | {prompt[:70]}", flush=True)
        return True
    else:
        tmp_path.unlink(missing_ok=True)
        return False

def extract_prompt(text: str) -> str:
    """Extract best prompt from text - JSON block > XML > 'Prompt:' keyword > full text."""
    if not text:
        return ""
    text = clean_html(text)

    # Try JSON brace-match (largest block wins)
    matches = re.findall(r'\{[^{}]{50,}\}', text, re.DOTALL)
    if matches:
        biggest = max(matches, key=len)
        try:
            obj = json.loads(biggest)
            return json.dumps(obj, indent=2)
        except:
            return biggest.strip()

    # Try XML/role blocks
    xml = re.search(r'<(?:prompt|system|image)[^>]*>([\s\S]+?)</(?:prompt|system|image)>', text, re.I)
    if xml:
        return xml.group(1).strip()

    # Try "Prompt:" keyword
    kw = re.search(r'(?:prompt|image prompt)[:\s]+(.{40,})', text, re.I | re.DOTALL)
    if kw:
        return kw.group(1).strip()[:2000]

    return text.strip()


# ── 1. Reddit ─────────────────────────────────────────────────────────────────

def build_reddit_queries(topic: str) -> list:
    t = topic.lower()
    skip = {"realistic","real","ultra","ai","generated","style","a","an","the"}
    words = [w for w in t.split() if w not in skip]
    core  = " ".join(words[:3])

    queries = [
        f"AI {topic} prompt",
        f"{core} prompt",
        f"realistic {core} AI image prompt",
        f"nano banana {core} prompt",
        f"flux {core} prompt",
        f"midjourney {core} prompt",
        f"grok {core} prompt",
    ]
    if "influencer" in t or "instagram" in t:
        queries += [
            "AI girl prompt photorealistic",
            "instagram influencer AI prompt",
            "realistic female influencer prompt tutorial",
        ]
    return queries

def reddit_image_urls(post_data: dict) -> list:
    """Extract all image URLs from a Reddit post (direct + gallery)."""
    urls = []
    url = post_data.get("url","")

    # Direct image link
    if re.search(r'\.(jpg|jpeg|png|webp)(\?|$)', url, re.I):
        urls.append(url)

    # Reddit preview images
    preview = post_data.get("preview", {})
    for img in preview.get("images", []):
        src = img.get("source", {}).get("url","")
        if src:
            urls.append(clean_html(src))  # Reddit HTML-encodes & as &amp;

    # Reddit gallery
    gallery = post_data.get("gallery_data", {})
    media_meta = post_data.get("media_metadata", {})
    for item in gallery.get("items", []):
        mid = item.get("media_id","")
        if mid and mid in media_meta:
            m = media_meta[mid]
            # Try to get largest image
            s = m.get("s", {})
            img_url = s.get("u") or s.get("gif","")
            if img_url:
                urls.append(clean_html(img_url))

    # i.redd.it or i.imgur direct
    if "i.redd.it" in url or "i.imgur.com" in url:
        urls.append(url)

    return list(dict.fromkeys(urls))  # dedup preserving order

_REDDIT_KEY_FROM_SORTED = re.compile(r"_reddit_([A-Za-z0-9]+)_(\d+)_\d+_\d+\.")
_REDDIT_KEY_FROM_QUEUE = re.compile(r"^reddit_([A-Za-z0-9]+)_(\d+)\.")


def _scan_disk_for_reddit_post_ids() -> set[str]:
    """Walk sorted/ + queue/ for already-handled Reddit posts.

    Vision worker writes `<cat>_reddit_<post_id>_<i>_<ts>_<rand>.<ext>` and
    queue items are `reddit_<post_id>_<i>.<ext>`. We just need post_id.
    """
    found: set[str] = set()
    sorted_root = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted")))
    sorted_slug = sorted_root if sorted_root.name == SLUG else sorted_root / SLUG
    if sorted_slug.exists():
        for path in sorted_slug.glob("**/reddit/*"):
            if path.is_file():
                m = _REDDIT_KEY_FROM_SORTED.search(path.name)
                if m:
                    found.add(m.group(1))
    queue_target = QUEUE_DIR / "reddit"
    if queue_target.exists():
        for path in queue_target.iterdir():
            if path.is_file():
                m = _REDDIT_KEY_FROM_QUEUE.match(path.name)
                if m:
                    found.add(m.group(1))
    return found


def scrape_reddit(seen: set) -> int:
    seen_file = BASE_DIR / f"seen_reddit_{SLUG}.json"
    if seen_file.exists():
        try:
            seen.update(json.loads(seen_file.read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    disk_keys = _scan_disk_for_reddit_post_ids()
    new_from_disk = disk_keys - seen
    if new_from_disk:
        print(f"  [seen-reddit] +{len(new_from_disk)} post_ids recovered from disk", flush=True)
    seen |= disk_keys

    topic_cfg = _load_topic_config()
    queries = build_reddit_queries(TOPIC)
    subreddit_allowlist = {s.strip().lower() for s in
                           os.environ.get("REDDIT_SUBREDDITS", "").split(",") if s.strip()}
    if subreddit_allowlist:
        print(f"  Reddit allowlist: {sorted(subreddit_allowlist)}", flush=True)

    saved = 0
    dropped_offtopic = 0
    dropped_subreddit = 0

    for query in queries:
        # If an allowlist is set, restrict each query to those subreddits via OR syntax.
        effective_query = query
        if subreddit_allowlist:
            sub_filter = " OR ".join(f"subreddit:{s}" for s in sorted(subreddit_allowlist))
            effective_query = f"({query}) ({sub_filter})"
        print(f"  Reddit query: {effective_query}", flush=True)
        for sort in ["relevance", "new", "top"]:
            try:
                url = f"https://www.reddit.com/search.json?q={requests.utils.quote(effective_query)}&sort={sort}&t=month&limit=50&type=link"
                r = requests.get(url, headers={**HEADERS, "Accept": "application/json"}, timeout=15)
                if r.status_code == 429:
                    print("  Reddit rate limit, sleeping 15s...")
                    time.sleep(15)
                    continue
                if not r.ok:
                    continue

                posts = r.json().get("data", {}).get("children", [])
                for post in posts:
                    d = post["data"]
                    post_id   = d.get("id","")
                    title     = d.get("title","")
                    body      = d.get("selftext","")
                    permalink = d.get("permalink","")
                    upvotes   = d.get("ups", 0)
                    subreddit = d.get("subreddit","")

                    if post_id in seen:
                        continue
                    seen.add(post_id)

                    # Subreddit allowlist (belt + braces: the query already restricts)
                    if subreddit_allowlist and subreddit.lower() not in subreddit_allowlist:
                        dropped_subreddit += 1
                        continue

                    # Extract prompt from title + body
                    combined  = f"{title}\n{body}"
                    prompt    = extract_prompt(combined)
                    if len(prompt) < 30:
                        prompt = title  # fallback to title as minimal prompt

                    # Topic filter: drop spam / off-topic / weak prompts.
                    ok, reason = _topic_passes(combined, prompt, cfg=topic_cfg)
                    if not ok:
                        dropped_offtopic += 1
                        continue

                    # Get image URLs from post
                    img_urls  = reddit_image_urls(d)
                    if not img_urls:
                        continue  # skip text-only posts with no images

                    for i, img_url in enumerate(img_urls[:4]):  # max 4 images per post
                        stem = f"reddit_{post_id}_{i}"
                        if (QUEUE_DIR / f"{stem}.jpg").exists() or (QUEUE_DIR / f"{stem}.png").exists():
                            continue
                        try:
                            resp = requests.get(img_url, headers=IMG_HEADERS, timeout=20)
                            if resp.ok:
                                ok = save_to_queue(stem, img_url, resp.content, prompt, {
                                    "source_channel": f"reddit_search",
                                    "source_guild":   f"reddit.com/r/{subreddit}",
                                    "author":         d.get("author",""),
                                    "timestamp":      str(d.get("created_utc","")),
                                    "reddit_post_id": post_id,
                                    "reddit_url":     f"https://reddit.com{permalink}",
                                    "upvotes":        upvotes,
                                    "query":          query,
                                }, source="reddit")
                                if ok:
                                    saved += 1
                                    # Persist seen IDs every successful save so a kill
                                    # mid-run can't make the next pass re-grab the same posts.
                                    seen_file.write_text(json.dumps(sorted(seen)), encoding="utf-8")
                        except Exception as e:
                            pass

                time.sleep(1)  # polite between sorts

            except Exception as e:
                print(f"  [Reddit ERR] {query}/{sort}: {e}")

        seen_file.write_text(json.dumps(sorted(seen)), encoding="utf-8")

    print(
        f"  Reddit done: saved={saved} dropped_offtopic={dropped_offtopic} "
        f"dropped_subreddit={dropped_subreddit}",
        flush=True,
    )
    return saved


# ── 2. ZforFree API ───────────────────────────────────────────────────────────

def build_zff_queries(topic: str) -> list:
    t = topic.lower()
    skip = {"realistic","real","ultra","ai","generated","a","an","the"}
    words = [w for w in t.split() if w not in skip]

    queries = list(dict.fromkeys([
        " ".join(words[:2]) if len(words) >= 2 else words[0] if words else topic,
        topic,
        "selfie", "portrait", "influencer", "instagram",
        "mirror selfie", "candid", "lifestyle", "beach",
    ]))
    if "influencer" in t or "instagram" in t:
        queries += ["realistic woman", "girl influencer", "social media"]
    if "male" in t or "man" in t:
        queries += ["realistic man", "guy influencer", "male portrait"]
    return queries

def scrape_zforfree(seen: set) -> int:
    seen_file = BASE_DIR / f"seen_zff_{SLUG}.json"
    if seen_file.exists():
        seen.update(json.loads(seen_file.read_text()))

    queries = build_zff_queries(TOPIC)
    saved   = 0

    # Possible ZFF API patterns - we'll try each
    API_BASES = [
        "https://zforfree.com/api/feed",
        "https://zforfree.com/api/images",
        "https://zforfree.com/api/v1/images",
    ]

    def try_zff_api(base: str, params: dict) -> list:
        try:
            r = requests.get(base, params=params, headers=HEADERS, timeout=15)
            if not r.ok:
                return None  # None = this base doesn't work
            data = r.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("items") or data.get("data") or data.get("results") or []
        except:
            return None
        return []

    # Discover which API base works
    working_base = None
    for base in API_BASES:
        result = try_zff_api(base, {"limit": 5, "skip": 0})
        if result is not None:
            working_base = base
            print(f"  ZFF API working at: {base}", flush=True)
            break

    if not working_base:
        print("  ZFF API not reachable - trying browser-style fetch...", flush=True)
        # Try without search param - just paginate top
        for base in API_BASES:
            result = try_zff_api(base, {"limit": 5})
            if result is not None:
                working_base = base
                break

    if not working_base:
        print("  ZFF API unavailable - skipping", flush=True)
        return 0

    for query in queries:
        print(f"  ZFF query: {query}", flush=True)
        skip  = 0
        query_saved = 0

        while query_saved < 300:  # max 300 per query term
            params = {"skip": skip, "limit": 50, "sort": "top", "search": query}
            items  = try_zff_api(working_base, params)
            if items is None or len(items) == 0:
                break

            for item in items:
                # Handle various field name conventions
                iid     = str(item.get("id") or item.get("_id") or "")
                prompt  = (item.get("prompt") or item.get("text") or item.get("description") or "").strip()
                img_url = (item.get("image_url") or item.get("imageUrl") or
                           item.get("image") or item.get("url") or "")

                # Try to construct image URL if missing
                if not img_url and iid:
                    img_url = f"https://zforfree.com/images/{iid}.jpg"

                dedup_key = f"zff_{iid}" if iid else f"zff_{url_hash(img_url)}"
                if dedup_key in seen or not img_url:
                    continue
                if len(prompt) < 30:
                    continue

                seen.add(dedup_key)

                try:
                    resp = requests.get(img_url, headers=IMG_HEADERS, timeout=20)
                    if resp.ok:
                        stem = f"zff_api_{iid or url_hash(img_url)}"
                        ok = save_to_queue(stem, img_url, resp.content, prompt, {
                            "source_channel": f"zforfree_api",
                            "source_guild":   "zforfree.com",
                            "author":         str(item.get("author") or item.get("username") or ""),
                            "timestamp":      str(item.get("created_at") or item.get("createdAt") or ""),
                            "zff_id":         iid,
                            "query":          query,
                        }, source="zforfree")
                        if ok:
                            saved += 1
                            query_saved += 1
                except Exception as e:
                    pass

                time.sleep(0.05)

            seen_file.write_text(json.dumps(list(seen)))
            skip += 50
            if len(items) < 50:
                break
            time.sleep(0.5)

    print(f"  ZFF done: {saved} saved", flush=True)
    return saved


# ── 3. promptsref.com ─────────────────────────────────────────────────────────

PROMPTSREF_PAGES = [
    ("https://promptsref.com/library/nano-banana-pro", "promptsref_nano_banana"),
    ("https://promptsref.com/library/grok",            "promptsref_grok"),
]

def scrape_promptsref(seen: set) -> int:
    saved = 0
    seen_file = BASE_DIR / f"seen_promptsref_{SLUG}.json"
    if seen_file.exists():
        seen.update(json.loads(seen_file.read_text()))

    for page_url, source_label in PROMPTSREF_PAGES:
        print(f"  promptsref: {page_url}", flush=True)
        try:
            r = requests.get(page_url, headers=HEADERS, timeout=20)
            if not r.ok:
                print(f"    Failed: {r.status_code}")
                continue

            text = r.text

            # Extract JSON prompt blocks - promptsref renders them as JSON in script or pre tags
            # Pattern 1: JSON objects in <pre> or <code> blocks
            code_blocks = re.findall(r'<(?:pre|code)[^>]*>([\s\S]*?)</(?:pre|code)>', text, re.I)
            # Pattern 2: JSON in script data / __NEXT_DATA__
            next_data = re.search(r'__NEXT_DATA__[^>]*>([\s\S]+?)</script>', text)
            if next_data:
                try:
                    nd = json.loads(next_data.group(1))
                    # Walk the structure looking for prompts
                    nd_str = json.dumps(nd)
                    code_blocks += re.findall(r'"prompt"\s*:\s*"((?:[^"\\]|\\.){50,})"', nd_str)
                except:
                    pass

            # Also try extracting from visible text - promptsref shows prompts in cards
            # Find all JSON-like blocks in raw HTML
            json_blocks = re.findall(r'(\{(?:[^{}]|\{[^{}]*\}){100,}\})', text)
            for block in json_blocks:
                block = clean_html(block).strip()
                try:
                    obj = json.loads(block)
                    prompt_text = json.dumps(obj, indent=2)
                except:
                    prompt_text = block

                if len(prompt_text) < 50:
                    continue

                dedup_key = f"promptsref_{url_hash(prompt_text)}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                # promptsref doesn't always have an associated image per prompt
                # Save as a text-only queue item - vision worker will handle it
                stem = dedup_key
                
                # Create dummy image (will be saved as metadata only)
                # For text-only prompts, we still need to save them to the queue
                meta_data = {
                    "message_id":      stem,
                    "image_url":       "",
                    "source_channel":  source_label,
                    "source_guild":    "promptsref.com",
                    "author":          "promptsref",
                    "timestamp":       datetime.utcnow().isoformat(),
                    "page_url":        page_url,
                    "pipeline_topic":  TOPIC,
                    "prompt_only":     True,  # Flag that this is text-only
                }
                
                # Save prompt-only to queue (using queue_manager API)
                # Since we don't have an image, create a minimal text marker
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as tmp:
                    tmp.write(prompt_text)
                    tmp_path = Path(tmp.name)
                
                result = queue_save("nanobanana", tmp_path, prompt_text, meta_data)
                if result:
                    print(f"  SAVED {stem} (prompt only) | {prompt_text[:70]}", flush=True)
                    saved += 1
                    # Persist after every save so a kill mid-run doesn't lose progress.
                    seen_file.write_text(json.dumps(sorted(seen)), encoding="utf-8")
                else:
                    tmp_path.unlink(missing_ok=True)

            seen_file.write_text(json.dumps(sorted(seen)), encoding="utf-8")

        except Exception as e:
            print(f"  [promptsref ERR] {page_url}: {e}")

    print(f"  promptsref done: {saved} saved", flush=True)
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Web Scraper (Reddit + ZFF + promptsref) ===", flush=True)
    print(f"Topic: {TOPIC}", flush=True)
    print(f"Queue: {QUEUE_DIR}", flush=True)

    seen_reddit      = set()
    seen_zff         = set()
    seen_promptsref  = set()

    total = 0

    print("\n[Reddit]", flush=True)
    total += scrape_reddit(seen_reddit)

    print("\n[ZforFree API]", flush=True)
    if ZFORFREE_WEB_ENABLED:
        total += scrape_zforfree(seen_zff)
    else:
        print("  Skipping ZforFree API (ZFORFREE_WEB_ENABLED=false)", flush=True)

    print("\n[promptsref.com]", flush=True)
    total += scrape_promptsref(seen_promptsref)

    print(f"\n=== Web scraper done. Total saved: {total} ===", flush=True)
