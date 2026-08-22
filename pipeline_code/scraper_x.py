"""
scraper_x.py - Scrape X.com for image/media posts
- Searches X with multiple queries
- Scrapes specific accounts known for good content
- Opens each tweet page to extract full prompt (body + first comments)
- Handles JSON prompts, XML/role prompts, plain text prompts
- Saves to source-based queue using queue_manager for balanced processing
"""
import asyncio, re, hashlib, os, requests, tempfile, time
from pathlib import Path
from playwright.async_api import async_playwright

from dotenv import load_dotenv
load_dotenv()

from topic_filter import (
    load_config as _load_topic_config,
    passes as _topic_passes,
    prompt_optional as _prompt_optional,
)
import media_policy
_TOPIC_CFG = _load_topic_config()
from paths import base_dir
from rate_limit import RateLimitConfig, RateLimiter
from credentials import get_optional as _cred_optional

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


# X.com rebranded from twitter.com in 2023 and continues to serve pages under
# both hostnames — depending on how a user copied their cookies (from an
# x.com session, a twitter.com session, or a Netscape cookies.txt export)
# the ``Domain`` attribute could legitimately be either ``.x.com`` or
# ``.twitter.com``. Playwright ONLY sends a cookie whose ``domain`` matches the
# request's host, so setting a single domain silently loses cookies for the
# other. Fix: emit each parsed cookie for BOTH domains. Costs nothing (the
# browser deduplicates by (name, domain)); makes the scraper robust to
# whichever domain the user's session came from.
_X_COOKIE_DOMAINS: tuple[str, ...] = (".x.com", ".twitter.com")


def _cookie_entries(name: str, value: str) -> list[dict]:
    """Emit one Playwright cookie entry per X-family domain."""
    return [
        {"name": name, "value": value, "domain": domain, "path": "/"}
        for domain in _X_COOKIE_DOMAINS
    ]


def _parse_twitter_cookies(raw: str) -> list[dict]:
    """Parse a raw ``name=value; name=value;`` header into Playwright cookies.

    Tolerates a leading ``Cookie:`` prefix (users often paste the whole request
    header line, header name and all) and skips malformed pairs quietly instead
    of crashing the scraper on a single stray semicolon.
    """
    raw = (raw or "").strip()
    if raw[:7].lower() == "cookie:":
        raw = raw[7:].strip()
    cookies: list[dict] = []
    for chunk in (c.strip() for c in raw.split(";") if c.strip()):
        if "=" not in chunk:
            continue
        name, value = chunk.split("=", 1)
        name = name.strip()
        if not name or " " in name:
            continue
        cookies.extend(_cookie_entries(name, value.strip()))
    return cookies


def _load_netscape_cookies_x(path: str) -> list[dict]:
    """Load Playwright cookies from a Netscape cookies.txt file.

    Used when ``X_COOKIES_TXT`` is set — preferred over the raw
    ``TWITTER_COOKIES`` string because a cookies.txt export carries the
    per-cookie ``Domain`` and ``expires`` attributes (so the scraper stops
    guessing whether a session cookie belongs to ``.x.com`` or ``.twitter.com``,
    and can surface an early ``expired`` warning). Malformed lines are logged
    and skipped; a completely unreadable file drops back to an empty list so
    the scraper degrades to search-only mode rather than crashing.
    """
    from pathlib import Path as _Path

    p = _Path(path).expanduser()
    if not p.is_file():
        print(f"[scraper_x] X_COOKIES_TXT path is not a file: {p}", flush=True)
        return []
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[scraper_x] cannot read X_COOKIES_TXT ({p}): {exc}", flush=True)
        return []

    now = int(time.time())
    cookies: list[dict] = []
    expired_seen = 0
    parsed = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        parsed += 1
        try:
            expires = int(parts[4])
        except ValueError:
            expires = 0
        name = parts[5].strip()
        value = parts[6].strip()
        if not name:
            continue
        # Skip clearly-expired cookies; if they're relevant to auth_token/ct0
        # the summary print will flag it.
        if expires > 0 and expires < now:
            expired_seen += 1
            continue
        # Ignore the source domain — apply to both x.com and twitter.com for
        # the same reason as _cookie_entries. Users who exported from one
        # domain still get cookies attached to the other.
        cookies.extend(_cookie_entries(name, value))
    if expired_seen:
        print(
            f"[scraper_x] {expired_seen} cookies in {p.name} are already expired "
            "— re-export from your browser if X.com scraping fails",
            flush=True,
        )
    if parsed == 0:
        print(f"[scraper_x] {p.name} parsed no Netscape cookie rows — check the file format", flush=True)
    return cookies


def _load_x_cookies() -> tuple[list[dict], str]:
    """Load X.com cookies from either ``X_COOKIES_TXT`` (preferred) or the
    inline ``TWITTER_COOKIES`` header. Returns (playwright_cookies, source_label).
    """
    txt_path = _cred_optional("X_COOKIES_TXT")
    if txt_path:
        cookies = _load_netscape_cookies_x(txt_path)
        return cookies, f"X_COOKIES_TXT ({txt_path})"
    raw = _cred_optional("TWITTER_COOKIES") or ""
    return _parse_twitter_cookies(raw), "TWITTER_COOKIES env"


X_COOKIES, _X_COOKIE_SOURCE = _load_x_cookies()
if X_COOKIES:
    # Two entries per cookie name (one per domain) — collapse for the count.
    _unique_names = {c["name"] for c in X_COOKIES}
    _has_auth = "auth_token" in _unique_names
    _has_ct0 = "ct0" in _unique_names
    _missing = [n for n, ok in (("auth_token", _has_auth), ("ct0", _has_ct0)) if not ok]
    if _missing:
        print(
            f"[scraper_x] cookies from {_X_COOKIE_SOURCE} missing required "
            f"name(s): {', '.join(_missing)} — X.com will refuse the session",
            flush=True,
        )
    else:
        print(
            f"[scraper_x] loaded {len(_unique_names)} cookies from {_X_COOKIE_SOURCE} "
            f"(applied to {', '.join(_X_COOKIE_DOMAINS)})",
            flush=True,
        )
else:
    print(
        "[scraper_x] WARNING: no X cookies loaded — set TWITTER_COOKIES (paste "
        "the full Cookie header from a logged-in x.com tab) or X_COOKIES_TXT "
        "(path to a Netscape cookies.txt export). X.com scraping will fail.",
        flush=True,
    )


# ── Topic-aware search query generator ──────────────────────────────────────
def build_searches(topic: str) -> list:
    """
    Auto-generate X.com search queries from a topic string.

    Two modes, gated on whether a prompt is required (``REQUIRE_PROMPT``):

    * **Prompt required (default):** hunt AI-art posts — wrap the topic in the
      AI-tool / "prompt" framing so results are likely to carry a usable
      generation prompt.
    * **Prompt NOT required:** search plainly by the topic terms and pull any
      matching media (``filter:media`` so video tweets match too). No AI-tool or
      "prompt" keywords are added — the user asked for raw topic media, and the
      vision worker does the relevance culling downstream.
    """
    t = topic.lower()

    # Extract key noun from topic (last meaningful word usually)
    words = [w for w in t.split() if w not in ("realistic","real","ultra","ai","generated","style","the","a","an")]
    subject = " OR ".join(words[:3]) if words else topic

    if _prompt_optional():
        # Plain topic media search — no AI-tool / "prompt" framing.
        queries = [
            f"{topic} filter:media",
            f"({subject}) filter:media",
        ]
        return list(dict.fromkeys(queries))  # de-dup when subject == topic

    ai_tools = "(zimage OR qwen OR flux OR midjourney OR grok OR nanobanana OR sdxl)"
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

    # Topic filter: reject off-topic / spammy tweets before downloading. Only
    # applied when a prompt is REQUIRED — the filter judges the tweet's *text*,
    # which an image/video-only tweet (already matched by X's own search) won't
    # have. When prompts are optional the user wants the raw media pulled and the
    # vision worker does the relevance culling, so we skip the text gate.
    if not _prompt_optional():
        context = f"{author} {source}"
        ok, reason = _topic_passes(context, prompt, cfg=_TOPIC_CFG)
        if not ok:
            print(f"    SKIP {dedup_key} ({reason})", flush=True)
            return False

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


_AUTH_FAIL_ANNOUNCED = False


def _looks_like_login_page(url: str) -> bool:
    """True if Playwright navigated to X's login/flow page instead of content.

    X redirects unauthenticated requests to ``/i/flow/login`` or ``/login``.
    Catching this early lets us print a single, actionable AUTH FAILURE
    message rather than letting every subsequent scrape return zero results.
    """
    if not url:
        return False
    low = url.lower()
    return (
        "/i/flow/login" in low
        or low.rstrip("/").endswith("/login")
        or "/i/flow/signup" in low
    )


def _announce_auth_failure(context: str) -> None:
    """Print the standard AUTH FAILURE line once per process.

    The supervisor's stdout capture surfaces this in the dashboard's log tab
    (see pipeline_logging.py comment) — the phrasing ("AUTH FAILURE for X.com")
    is deliberately fixed so we can grep for it across scrapers.
    """
    global _AUTH_FAIL_ANNOUNCED
    if _AUTH_FAIL_ANNOUNCED:
        return
    _AUTH_FAIL_ANNOUNCED = True
    print(
        f"AUTH FAILURE for X.com: {context}. "
        "Cookies are missing/invalid/expired — re-copy the full Cookie header "
        "from a logged-in x.com tab into Global Settings > TWITTER_COOKIES (or "
        "point X_COOKIES_TXT at a fresh Netscape cookies.txt export) and "
        "re-test in Global Settings.",
        flush=True,
    )


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

        # If cookies didn't authenticate us, X redirects to /i/flow/login.
        # Print a single AUTH FAILURE line and bail — every downstream selector
        # will otherwise miss and log a torrent of noise.
        if _looks_like_login_page(page.url):
            _announce_auth_failure(f"redirected to login while opening tweet {tweet_id}")
            return

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

    # Search results silently redirect to /login when cookies are stale.
    if _looks_like_login_page(page.url):
        _announce_auth_failure(f"redirected to login while opening search '{query[:60]}'")
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
        await scrape_tweet_page(ctx, tweet_id, author, seen, saved_count, "x_search")
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

    if _looks_like_login_page(page.url):
        _announce_auth_failure(f"redirected to login while opening @{handle}/media")
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
    # This scraper harvests still images (pbs.twimg.com/media). For a video-only
    # job it has nothing to contribute — and its stills would sit unpoppable in a
    # video queue — so skip. Use gallery-dl or yt-dlp for X video.
    if not media_policy.wants_image():
        print("[scraper_x] media policy excludes images — skipping (use gallery-dl "
              "or yt-dlp for X video)", flush=True)
        return
    seen = _make_seen_store()
    print("=== X.com Scraper ===")
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

        await browser.close()

    seen.flush()
    print(f"\n=== Done. Total saved: {saved_count[0]} ===")


if __name__ == "__main__":
    asyncio.run(main())
