"""
scraper_x.py - Scrape X.com for AI image prompts + promptsref.com
- Searches X with multiple queries
- Scrapes specific accounts known for good prompts
- Opens each tweet page to extract full prompt (body + first comments)
- Handles JSON prompts, XML/role prompts, plain text prompts
- Also scrapes promptsref.com/library/grok and /nano-banana-pro
- Saves to source-based queue using queue_manager for balanced processing
"""
import asyncio, json, re, hashlib, os, requests, tempfile
from pathlib import Path
from playwright.async_api import async_playwright

from dotenv import load_dotenv
load_dotenv()

from topic_filter import load_config as _load_topic_config, passes as _topic_passes
_TOPIC_CFG = _load_topic_config()
from paths import base_dir
from rate_limit import RateLimitConfig, RateLimiter

# Per-source pacing + 429 backoff + optional proxy for the requests-based image
# downloads (page scraping is Playwright). Opt-in: unthrottled unless
# RATE_LIMIT_X_* env is set, so existing behaviour is unchanged.
_LIMITER = RateLimiter(RateLimitConfig.from_mapping("x", os.environ))


def _retry_after_seconds(resp) -> float | None:
    """Parse a Retry-After header (delta-seconds form) into a float, or None."""
    raw = resp.headers.get("Retry-After") if resp is not None else None
    if not raw:
        return None
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None

BASE_DIR  = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
TOPIC     = os.environ.get("PIPELINE_TOPIC", "Realistic Female Influencer")
SLUG      = os.environ.get("PIPELINE_SLUG",  "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
QUEUE_DIR = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG
# Import queue manager + dedup
from queue_manager import save_to_queue as queue_save  # noqa: E402
from seen_store import SeenStore  # noqa: E402

_SEEN_NAME = "x"


def _parse_twitter_cookies(raw: str) -> list[dict]:
    """Parse `name=value; name=value;` into Playwright cookie dicts."""
    cookies: list[dict] = []
    for chunk in (c.strip() for c in raw.split(";") if c.strip()):
        if "=" not in chunk:
            continue
        name, value = chunk.split("=", 1)
        cookies.append({"name": name.strip(), "value": value.strip(), "domain": ".x.com", "path": "/"})
    return cookies


X_COOKIES: list[dict] = _parse_twitter_cookies(os.environ.get("TWITTER_COOKIES", ""))
if not X_COOKIES:
    print("[scraper_x] WARNING: TWITTER_COOKIES empty in .env — X.com scraping will likely fail", flush=True)


# ── Topic-aware search query generator ──────────────────────────────────────
def build_searches(topic: str) -> list:
    """
    Auto-generate X.com search queries from a topic string.
    Always includes AI-tool queries + topic-specific keywords.
    """
    t = topic.lower()
    ai_tools = "(zimage OR qwen OR flux OR midjourney OR grok OR nanobanana OR sdxl)"

    # Extract key noun from topic (last meaningful word usually)
    words = [w for w in t.split() if w not in ("realistic","real","ultra","ai","generated","style","the","a","an")]
    subject = " OR ".join(words[:3]) if words else topic

    queries = [
        f"{ai_tools} {topic} prompt filter:images min_faves:5",
        f"{ai_tools} ({subject}) prompt filter:images min_faves:5",
        f"realistic {topic} prompt filter:images min_faves:10",
        f"(photorealistic OR realistic) {topic} AI prompt filter:images min_faves:10",
        f"{topic} prompt filter:images min_faves:20",
        f"(flux OR midjourney OR grok) {topic} prompt filter:images min_faves:5",
        f"{topic} \"prompt\" filter:images min_faves:5",
        f"nanobanana2 {topic} prompt filter:images min_faves:5",
    ]
    return queries

SEARCHES = build_searches(TOPIC)


# ── Accounts to scrape (admin-configurable) ─────────────────────────────────
# Read from X_ACCOUNTS env var (comma-separated). Empty string = search-only
# mode (no per-account scraping). Admins edit from the dashboard Settings tab.
_DEFAULT_ACCOUNTS: list[str] = [
    "AFantagirl85190", "PrometheanAIX", "brindleyai", "KeorUnreal",
    "EXM7777", "daaaaanc", "Artist04048661", "rowundd",
    "NameIsSudee", "underwoodxie96", "johnnprofits", "aimodelabuser",
]


def _load_accounts() -> list[str]:
    raw = os.environ.get("X_ACCOUNTS")
    if raw is None:
        return list(_DEFAULT_ACCOUNTS)
    return [a.strip().lstrip("@") for a in raw.split(",") if a.strip()]


ACCOUNTS: list[str] = _load_accounts()
if ACCOUNTS:
    print(f"[scraper_x] Targeting {len(ACCOUNTS)} accounts + search results", flush=True)
else:
    print("[scraper_x] X_ACCOUNTS empty - search results only, no per-account scraping", flush=True)

SCROLL_ROUNDS = 6
SCROLL_PAUSE  = 2.0
MAX_PER_QUERY = 40

# ── Prompt extraction helpers ───────────────────────────────────────────────
def extract_prompt_from_text(text: str) -> str:
    """
    Try to find the best prompt content from raw tweet/comment text.
    Handles: JSON objects, XML/role blocks, plain text prompts after 'Prompt:' keyword.
    Returns the extracted prompt or the full text if no pattern matched.
    """
    if not text or len(text) < 20:
        return text or ""

    # 1. JSON block — find the outermost { ... }
    brace_start = text.find("{")
    if brace_start != -1:
        # Walk to find matching closing brace
        depth = 0
        for i, ch in enumerate(text[brace_start:], brace_start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start:i+1]
                    if len(candidate) > 50:
                        return candidate
                    break

    # 2. XML/role block — grab everything between first < and last >
    if "<role>" in text.lower() or "<cognitive_framework>" in text.lower() or "<instructions>" in text.lower():
        tag_start = text.find("<")
        tag_end   = text.rfind(">")
        if tag_start != -1 and tag_end > tag_start:
            return text[tag_start:tag_end+1]

    # 3. "Prompt:" keyword
    m = re.search(r'(?i)prompt\s*[:\-]\s*(.+)', text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 4. Return full text if long enough to be a prompt
    return text if len(text) > 40 else ""


# ── Image download ──────────────────────────────────────────────────────────
def download_image(url: str, dest: Path) -> bool:
    try:
        h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"}
        _LIMITER.acquire()
        r = requests.get(url, headers=h, timeout=20, **_LIMITER.requests_kwargs())
        if r.status_code == 429:
            _LIMITER.note_429(retry_after=_retry_after_seconds(r))
            return False
        if r.ok and len(r.content) > 5000:
            _LIMITER.note_success()
            dest.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"    Download err: {e}")
    return False


def save_item(tweet_id: str, img_src: str, prompt: str, author: str, source: str, seen: set) -> bool:
    """Download image + write txt + meta to source-based queue. Returns True if saved."""
    img_hash  = hashlib.md5(img_src.encode()).hexdigest()[:8]
    dedup_key = f"x_{tweet_id}_{img_hash}"
    if dedup_key in seen:
        return False
    seen.add(dedup_key)

    # Topic filter: reject off-topic / spammy tweets before downloading.
    context = f"{author} {source}"
    ok, reason = _topic_passes(context, prompt, cfg=_TOPIC_CFG)
    if not ok:
        print(f"    SKIP {dedup_key} ({reason})", flush=True)
        return False

    stem = dedup_key
    
    # Upgrade to orig quality
    img_url = re.sub(r'[?&]name=\w+', '', img_src)
    img_url = img_url + ("&name=orig" if "?" in img_url else "?name=orig")

    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    if not download_image(img_url, tmp_path):
        seen.discard(dedup_key)
        tmp_path.unlink(missing_ok=True)
        return False

    # Prepare metadata
    meta_data = {
        "message_id":     dedup_key,
        "image_url":      img_url,
        "source_channel": source,
        "source_guild":   "x.com",
        "author":         author,
        "timestamp":      "",
        "tweet_id":       tweet_id,
    }
    
    # Save to source-based queue
    result = queue_save("twitter_x", tmp_path, prompt, meta_data)

    if result:
        size_kb = result.stat().st_size // 1024
        print(f"    SAVED {result.name} ({size_kb}KB) | {prompt[:80] or '(no prompt)'}")
        # Persist dedup IDs immediately so a kill / restart can't lose them.
        seen.flush()
        return True
    else:
        seen.discard(dedup_key)
        return False


# ── Extract tweet images from a search/timeline page ───────────────────────
async def collect_tweet_ids_from_page(page) -> list:
    """Return list of {tweet_id, author, img_srcs} from currently visible page."""
    return await page.evaluate("""
        () => {
            const results = [];
            const imgs = Array.from(document.querySelectorAll('img[src*="pbs.twimg.com/media"]'));
            const seen = new Set();
            for (const img of imgs) {
                let el = img, tweetId = '', author = '';
                for (let i = 0; i < 20; i++) {
                    el = el.parentElement;
                    if (!el) break;
                    if (!tweetId) {
                        const a = el.querySelector('a[href*="/status/"]');
                        if (a) { const m = a.href.match(/\\/status\\/(\\d+)/); if (m) tweetId = m[1]; }
                    }
                    if (!author) {
                        const a = el.querySelector('a[href^="/"][href*="status"]');
                        if (a) { const m = a.href.match(/\\/([^/]+)\\/status/); if (m) author = m[1]; }
                    }
                    if (tweetId && author) break;
                }
                if (!tweetId || seen.has(tweetId)) continue;
                seen.add(tweetId);
                results.push({ tweetId, author });
            }
            return results;
        }
    """)


# ── Visit individual tweet page and extract images + prompt ────────────────
async def scrape_tweet_page(ctx, tweet_id: str, author: str, seen: set, saved_count: list, source: str):
    url = f"https://x.com/{author}/status/{tweet_id}"
    page = await ctx.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        await asyncio.sleep(2.5)

        # ── Grab all text in the tweet (body + expand "Show more" if present) ──
        show_more = await page.query_selector('[data-testid="tweet-text-show-more-link"]')
        if show_more:
            await show_more.click()
            await asyncio.sleep(0.5)

        # Main tweet text
        tweet_texts = await page.evaluate("""
            () => {
                // Primary tweet is the first article on a status page
                const arts = Array.from(document.querySelectorAll('article'));
                if (!arts.length) return [];
                const art = arts[0];
                const spans = Array.from(art.querySelectorAll('[data-testid="tweetText"]'));
                return spans.map(s => s.innerText.trim()).filter(t => t.length > 5);
            }
        """)

        # Also grab first 5 replies/comments — prompt often lives there
        reply_texts = await page.evaluate("""
            () => {
                const arts = Array.from(document.querySelectorAll('article'));
                const replies = arts.slice(1, 6);
                const out = [];
                for (const a of replies) {
                    const spans = a.querySelectorAll('[data-testid="tweetText"]');
                    for (const s of spans) {
                        const t = s.innerText.trim();
                        if (t.length > 30) out.push(t);
                    }
                }
                return out;
            }
        """)

        # Best prompt = longest extracted content across tweet body + comments
        all_texts = tweet_texts + reply_texts
        best_prompt = ""
        for t in all_texts:
            candidate = extract_prompt_from_text(t)
            if len(candidate) > len(best_prompt):
                best_prompt = candidate

        # ── Grab images ──
        img_srcs = await page.evaluate("""
            () => {
                const arts = Array.from(document.querySelectorAll('article'));
                if (!arts.length) return [];
                const art = arts[0];
                return Array.from(art.querySelectorAll('img[src*="pbs.twimg.com/media"]'))
                       .map(i => i.src);
            }
        """)

        for img_src in img_srcs:
            ok = save_item(tweet_id, img_src, best_prompt, author, source, seen)
            if ok:
                saved_count[0] += 1

    except Exception as e:
        print(f"    Tweet page error {tweet_id}: {e}")
    finally:
        await page.close()


# ── Scrape a search query ───────────────────────────────────────────────────
async def scrape_search(ctx, query: str, seen: set, saved_count: list):
    print(f"\n  Search: {query[:80]}")
    encoded = requests.utils.quote(query)
    url = f"https://x.com/search?q={encoded}&src=typed_query&f=media"

    page = await ctx.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)
    except Exception as e:
        print(f"  Nav error: {e}")
        await page.close()
        return

    collected = {}  # tweet_id -> author
    for _ in range(SCROLL_ROUNDS):
        items = await collect_tweet_ids_from_page(page)
        for item in items:
            tid = item["tweetId"]
            if tid not in collected:
                collected[tid] = item["author"]
        if len(collected) >= MAX_PER_QUERY:
            break
        await page.keyboard.press("End")
        await asyncio.sleep(SCROLL_PAUSE)

    await page.close()
    print(f"  Found {len(collected)} unique tweets, visiting each...")

    for tweet_id, author in list(collected.items())[:MAX_PER_QUERY]:
        dedup_check = f"x_{tweet_id}_"
        if any(k.startswith(dedup_check) for k in seen):
            continue
        await scrape_tweet_page(ctx, tweet_id, author, seen, saved_count, f"x_search")
        await asyncio.sleep(0.5)

    seen.flush()
    print(f"  Search done. Total saved so far: {saved_count[0]}")


# ── Scrape a specific account timeline ─────────────────────────────────────
async def scrape_account(ctx, handle: str, seen: set, saved_count: list):
    print(f"\n  Account: @{handle}")
    page = await ctx.new_page()
    try:
        await page.goto(f"https://x.com/{handle}/media", wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(5)  # media tab needs extra time to hydrate
    except Exception as e:
        print(f"  Nav error @{handle}: {e}")
        await page.close()
        return

    collected = {}
    for _ in range(4):
        items = await collect_tweet_ids_from_page(page)
        for item in items:
            tid = item["tweetId"]
            if tid not in collected:
                collected[tid] = handle  # use handle as author fallback
        await page.keyboard.press("End")
        await asyncio.sleep(2)

    await page.close()
    print(f"  @{handle}: {len(collected)} tweets found")

    for tweet_id, author in collected.items():
        dedup_check = f"x_{tweet_id}_"
        if any(k.startswith(dedup_check) for k in seen):
            continue
        await scrape_tweet_page(ctx, tweet_id, author, seen, saved_count, f"x_account_{handle}")
        await asyncio.sleep(0.5)

    seen.flush()


# ── Scrape promptsref.com ───────────────────────────────────────────────────
async def scrape_promptsref(ctx, seen: set, saved_count: list):
    """Scrape promptsref.com Grok and NanoBanana libraries — prompts + share images."""
    for lib_path, label in [("/library/grok", "grok"), ("/library/nano-banana-pro", "nanobanana")]:
        url = f"https://promptsref.com{lib_path}"
        print(f"\n  promptsref.com {label}...")
        page = await ctx.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(3)

            # Scroll to load all items
            for _ in range(8):
                await page.keyboard.press("End")
                await asyncio.sleep(1.5)

            # Extract prompt cards: each has a prompt text block + image
            cards = await page.evaluate("""
                () => {
                    const out = [];
                    // Look for elements containing long JSON-like text
                    const candidates = Array.from(document.querySelectorAll('div, pre, code, p'));
                    for (const el of candidates) {
                        const txt = el.innerText ? el.innerText.trim() : '';
                        if (txt.length > 100 && (txt.includes('"subject"') || txt.includes('"prompt"') || txt.includes('"description"'))) {
                            // Find nearest image
                            let imgSrc = '';
                            let parent = el;
                            for (let i = 0; i < 10; i++) {
                                parent = parent.parentElement;
                                if (!parent) break;
                                const img = parent.querySelector('img[src]');
                                if (img && img.src && !img.src.includes('logo') && !img.src.includes('icon')) {
                                    imgSrc = img.src;
                                    break;
                                }
                            }
                            out.push({ prompt: txt.slice(0, 3000), imgSrc });
                        }
                    }
                    // Deduplicate by first 50 chars of prompt
                    const seen = new Set();
                    return out.filter(c => {
                        const k = c.prompt.slice(0, 50);
                        if (seen.has(k)) return false;
                        seen.add(k);
                        return true;
                    });
                }
            """)

            print(f"  Found {len(cards)} prompt cards")
            for i, card in enumerate(cards):
                prompt  = extract_prompt_from_text(card.get("prompt", ""))
                img_src = card.get("imgSrc", "")
                if not prompt or not img_src:
                    continue

                dedup_key = f"promptsref_{label}_{hashlib.md5(prompt[:80].encode()).hexdigest()[:10]}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                stem = dedup_key
                
                # Download to temp file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                if not download_image(img_src, tmp_path):
                    seen.discard(dedup_key)
                    tmp_path.unlink(missing_ok=True)
                    continue

                # Prepare metadata
                meta_data = {
                    "message_id":     dedup_key,
                    "image_url":      img_src,
                    "source_channel": f"promptsref_{label}",
                    "source_guild":   "promptsref.com",
                    "author":         "",
                    "timestamp":      "",
                }
                
                # Save to source-based queue
                result = queue_save("nanobanana", tmp_path, prompt, meta_data)
                
                if result:
                    size_kb = result.stat().st_size // 1024
                    print(f"    SAVED {result.name} ({size_kb}KB) | {prompt[:70]}")
                    saved_count[0] += 1
                else:
                    seen.discard(dedup_key)

        except Exception as e:
            print(f"  promptsref error: {e}")
        finally:
            await page.close()

    seen.flush()


# ── Seen helpers ─────────────────────────────────────────────────────────────

# Pull dedup keys back out of sorted filenames. Vision worker writes:
#   <category>_x_<tweet_id>_<img_hash>_<ts>_<rand>.<ext>
# So the dedup key matches `x_<tweet>_<hash>` - same shape we add to `seen`.
_X_KEY_FROM_SORTED = re.compile(r"_(x_[A-Za-z0-9]+_[A-Za-z0-9]+)_\d+_\d+\.")

# Same shape lives in queue/ when an item is pending classification.
_X_KEY_FROM_QUEUE = re.compile(r"^(x_[A-Za-z0-9]+_[A-Za-z0-9]+)\.")


def _scan_sorted_for_x_keys() -> set[str]:
    """Find every X dedup key already classified under <sorted>/<slug>/."""
    sorted_root = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted")))
    sorted_slug = sorted_root if sorted_root.name == SLUG else sorted_root / SLUG
    if not sorted_slug.exists():
        return set()
    found: set[str] = set()
    for path in sorted_slug.glob("**/twitter_x/*"):
        if not path.is_file():
            continue
        m = _X_KEY_FROM_SORTED.search(path.name)
        if m:
            found.add(m.group(1))
    return found


def _scan_queue_for_x_keys() -> set[str]:
    """Find every X dedup key still pending in <queue>/<slug>/twitter_x/."""
    target = QUEUE_DIR / "twitter_x"
    if not target.exists():
        return set()
    found: set[str] = set()
    for path in target.iterdir():
        if not path.is_file():
            continue
        m = _X_KEY_FROM_QUEUE.match(path.name)
        if m:
            found.add(m.group(1))
    return found


def _make_seen_store() -> SeenStore:
    """Build the X.com SeenStore + adopt IDs already on disk so an interrupted
    prior run doesn't make us re-pull the same tweets. Three sources fold in:
      1. seen_x_<slug>.json itself (cheapest, native)
      2. <sorted>/<slug>/**/twitter_x/*  (already-classified)
      3. <queue>/<slug>/twitter_x/*     (still pending classification)
    """
    store = SeenStore(_SEEN_NAME, slug=SLUG, autoflush_every=25)
    sorted_keys = _scan_sorted_for_x_keys()
    queue_keys = _scan_queue_for_x_keys()
    new_from_disk = (sorted_keys | queue_keys) - store.snapshot()
    if new_from_disk:
        print(
            f"  [seen] +{len(new_from_disk)} ids recovered from disk "
            f"(sorted={len(sorted_keys)}, queue={len(queue_keys)})",
            flush=True,
        )
    store.update(sorted_keys | queue_keys)
    return store


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    seen = _make_seen_store()
    print(f"=== X.com + promptsref Scraper ===")
    print(f"Loaded {len(seen)} already-seen IDs")
    saved_count = [0]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1440, "height": 900}
        )
        await ctx.add_cookies(X_COOKIES)

        # 1. Scrape specific accounts first (best quality)
        for handle in ACCOUNTS:
            await scrape_account(ctx, handle, seen, saved_count)

        # 2. Search queries
        for query in SEARCHES:
            await scrape_search(ctx, query, seen, saved_count)

        # 3. promptsref.com (no auth needed)
        await scrape_promptsref(ctx, seen, saved_count)

        await browser.close()

    seen.flush()
    print(f"\n=== Done. Total saved: {saved_count[0]} ===")


if __name__ == "__main__":
    asyncio.run(main())
