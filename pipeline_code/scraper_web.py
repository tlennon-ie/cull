"""
scraper_web.py - Reddit scraper (Playwright browser transport)

Fetches Reddit listings/search through a real Chromium browser (Playwright) so
Reddit's anti-bot 403s on the public ``.json`` endpoints are bypassed — a real
browser carries a genuine session / TLS fingerprint that raw ``requests`` calls
do not, which is what Reddit blocks. JSON feeds and images are both fetched with
in-page ``fetch()`` (same-origin, cookies included), then routed through the
existing parse → topic-filter → queue path unchanged.

gallery-dl's own Reddit support (scraper_gallery_dl.py) is separate and left
intact — this is cull's first-party Reddit scraper.

Sources:
  1. Reddit (topic-aware search when a prompt is required; direct subreddit
     listing when it isn't), downloads linked images.
"""
import re, json, os, html, time, tempfile
from pathlib import Path
from urllib.parse import urlparse, quote

from dotenv import load_dotenv
load_dotenv()

from playwright.sync_api import sync_playwright

from topic_filter import (
    load_config as _load_topic_config,
    passes as _topic_passes,
    prompt_optional as _prompt_optional,
)
import media_policy
from paths import base_dir
from rate_limit import RateLimitConfig, RateLimiter

# Per-source pacing + 429 backoff + optional proxy. Opt-in: unthrottled unless
# RATE_LIMIT_WEB_* env is set, so existing behaviour is unchanged.
_LIMITER = RateLimiter(RateLimitConfig.from_mapping("web", os.environ))

BASE_DIR  = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
TOPIC     = os.environ.get("PIPELINE_TOPIC", "Realistic Female Influencer")
SLUG      = os.environ.get("PIPELINE_SLUG",  "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
QUEUE_DIR = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG

# Import queue manager
from queue_manager import save_to_queue as queue_save
from seen_store import SeenStore
from credentials import get_optional as _cred_optional

# A real browser is what gets past Reddit's 403s, so we drive Chromium. Use a
# normal Chrome UA (REDDIT_USER_AGENT can override). The old descriptive
# API-style UA only mattered for the retired requests/OAuth transport.
BROWSER_UA = (_cred_optional("REDDIT_USER_AGENT", "") or "").strip() or (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _parse_reddit_cookies(raw: str) -> list[dict]:
    """Parse an optional ``name=value; name2=value2`` cookie string. Paste your
    logged-in reddit.com cookies (REDDIT_COOKIES) to reach NSFW / gated
    subreddits. Mirrors the X scraper's TWITTER_COOKIES handling. Tolerates a
    pasted ``Cookie:`` header prefix and skips malformed pairs."""
    raw = (raw or "").strip()
    if raw[:7].lower() == "cookie:":         # user pasted the whole request header
        raw = raw[7:].strip()
    cookies: list[dict] = []
    for part in raw.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if not name or " " in name:          # a space in the name = malformed pair
            continue
        cookies.append({"name": name, "value": value.strip(),
                        "domain": ".reddit.com", "path": "/"})
    return cookies


REDDIT_COOKIES = _parse_reddit_cookies(os.environ.get("REDDIT_COOKIES", ""))

_NAV_TIMEOUT_MS = 30000


def _browser_fetch_json(page, url: str):
    """Navigate to a Reddit ``.json`` endpoint and return (data, error).

    ``page.goto`` is a real browser navigation — it carries the genuine session /
    TLS fingerprint that gets past Reddit's 403 — and reading ``Response.json()``
    avoids the "execution context destroyed" races that ``page.evaluate`` hits
    against Reddit's live SPA. ``error`` is an HTTP status int / message, or None.
    """
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001 - navigation/timeout/download
        return None, str(exc)
    if resp is None:
        return None, "no response"
    if not resp.ok:
        snippet = ""
        try:
            snippet = (resp.text() or "")[:160].replace("\n", " ").strip()
        except Exception:  # noqa: BLE001
            pass
        return None, f"HTTP {resp.status}" + (f": {snippet}" if snippet else "")
    try:
        return resp.json(), None          # parses the raw response body, not the DOM
    except Exception as exc:  # noqa: BLE001 - non-JSON (e.g. an HTML redirect/gate)
        return None, f"non-JSON response ({exc})"


def _looks_like_image(b: bytes) -> bool:
    """Magic-byte sniff so we never queue an HTML placeholder as a .jpg. Reddit
    serves a consent/blocked HTML page for some preview.redd.it requests, and
    that must not land in the queue (PIL can't open it → broken thumbnails)."""
    if not b or len(b) < 12:
        return False
    return (
        b[:3] == b"\xff\xd8\xff"                       # JPEG
        or b[:8] == b"\x89PNG\r\n\x1a\n"               # PNG
        or b[:6] in (b"GIF87a", b"GIF89a")             # GIF
        or (b[:4] == b"RIFF" and b[8:12] == b"WEBP")   # WEBP
    )


def _looks_like_video(b: bytes) -> bool:
    """Magic-byte sniff for common web video containers, so an HTML gate is never
    queued as a clip. v.redd.it fallback_url is a plain MP4 (video-only)."""
    if not b or len(b) < 12:
        return False
    return (
        b[4:8] == b"ftyp"                              # MP4 / MOV (ISO-BMFF)
        or b[:4] == b"\x1a\x45\xdf\xa3"                # WebM / Matroska (EBML)
        or (b[:4] == b"RIFF" and b[8:12] == b"AVI ")   # AVI
    )


def _browser_fetch_bytes(page, url: str, want: str = "image"):
    """Pull real image bytes, trying strategies in order because Reddit's image
    hosts disagree: preview.redd.it serves a consent HTML page WITHOUT a reddit
    ``Referer``; i.redd.it redirects to an HTML media gate WITH one. So we try
    browser navigation with a Referer, then without, then the context HTTP client
    — returning the first response whose bytes pass a magic-byte image check (so
    an HTML page can never be queued as an image). Returns (bytes, error)."""
    last = "no attempt"
    _ok = _looks_like_video if want == "video" else _looks_like_image

    def _valid(resp):
        nonlocal last
        if resp is None:
            last = "no response"; return None
        if not resp.ok:
            last = f"HTTP {resp.status}"; return None
        try:
            body = resp.body()
        except Exception as exc:  # noqa: BLE001
            last = f"body error: {exc}"; return None
        if body and _ok(body):
            return body
        ctype = (resp.headers.get("content-type") or "").lower()
        last = f"not {want} (ctype={ctype!r}, {len(body) if body else 0}B)"
        return None

    for referer in ("https://www.reddit.com/", None):
        try:
            resp = page.goto(url, referer=referer, wait_until="commit",
                             timeout=_NAV_TIMEOUT_MS)
        except Exception as exc:  # noqa: BLE001 - nav/timeout/download-as-attachment
            last = f"nav error: {exc}"; continue
        body = _valid(resp)
        if body is not None:
            return body, None

    # HTTP-client fallback (shares the context's cookies).
    try:
        resp = page.context.request.get(
            url, headers={"Referer": "https://www.reddit.com/", "User-Agent": BROWSER_UA},
            timeout=_NAV_TIMEOUT_MS)
        body = _valid(resp)
        if body is not None:
            return body, None
    except Exception as exc:  # noqa: BLE001
        last = f"request error: {exc}"

    return None, last

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

def save_to_queue(stem: str, img_url: str, img_bytes: bytes, prompt: str, meta: dict, source: str = "unknown") -> bool:
    """Save image + txt + meta.json to source-based queue. Returns True on success."""
    if not img_bytes or len(img_bytes) < 5000:
        return False
    if not prompt or len(prompt.strip()) < 30:
        return False

    url_lower = img_url.lower().split("?")[0]
    # Prefer the URL's real extension when the media policy accepts it (keeps a
    # v.redd.it .mp4 a clip); else fall back to the historical image handling.
    m = re.search(r"\.([a-z0-9]{2,5})$", url_lower)
    cand = f".{m.group(1)}" if m else ""
    if cand and media_policy.accepts(cand):
        ext = cand.lstrip(".")
    elif url_lower.endswith(".png"):
        ext = "png"
    elif url_lower.endswith(".webp"):
        ext = "webp"
    elif url_lower.endswith(".gif"):
        return False  # skip gifs
    else:
        ext = "jpg"

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

    # Prompt NOT required: search plainly by the topic — no AI / "prompt" framing.
    # (When subreddits are configured, scrape_reddit lists them directly instead;
    # this plain search is the fallback for the no-subreddit case.)
    if _prompt_optional():
        return list(dict.fromkeys([topic, core] if core else [topic]))

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
    host = (urlparse(url).hostname or "").lower()
    if host in ("i.redd.it", "i.imgur.com") or host.endswith((".i.redd.it", ".i.imgur.com")):
        urls.append(url)

    return list(dict.fromkeys(urls))  # dedup preserving order


def reddit_video_urls(post_data: dict) -> list:
    """Extract playable video URLs from a Reddit post. v.redd.it clips expose a
    ``fallback_url`` — a plain MP4 (video-only, no audio, but perfectly fine for
    frame-based classification). Covers native posts and crosspost previews."""
    urls: list[str] = []
    for block in (
        (post_data.get("media") or {}).get("reddit_video") or {},
        (post_data.get("preview") or {}).get("reddit_video_preview") or {},
        (post_data.get("secure_media") or {}).get("reddit_video") or {},
    ):
        fb = block.get("fallback_url") or ""
        if fb:
            urls.append(clean_html(fb).split("?")[0])
    # Direct video link in the post URL (e.g. an .mp4 crosspost).
    url = post_data.get("url", "")
    if re.search(r'\.(mp4|webm|mov|mkv)(\?|$)', url, re.I):
        urls.append(url)
    return list(dict.fromkeys(urls))

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


def scrape_reddit(seen: "SeenStore") -> int:
    """Reddit branch. ``seen`` is a SeenStore created in main(). Adopts any
    post_ids found on disk (queue or sorted) before the first request so an
    interrupted prior run doesn't re-pull the same posts."""
    disk_keys = _scan_disk_for_reddit_post_ids()
    new_from_disk = disk_keys - seen.snapshot()
    if new_from_disk:
        print(f"  [seen-reddit] +{len(new_from_disk)} post_ids recovered from disk", flush=True)
    seen.update(disk_keys)

    topic_cfg = _load_topic_config()
    queries = build_reddit_queries(TOPIC)
    subreddit_allowlist = {s.strip().lower() for s in
                           os.environ.get("REDDIT_SUBREDDITS", "").split(",") if s.strip()}
    if subreddit_allowlist:
        print(f"  Reddit allowlist: {sorted(subreddit_allowlist)}", flush=True)

    saved = 0
    dropped_offtopic = 0
    dropped_subreddit = 0

    # Two modes, gated on whether a prompt is required:
    #   * listing_mode (prompt NOT required + subreddits set): pull EVERY post's
    #     media from the configured subreddits directly via /r/<sub>/<sort>.json.
    #     No keyword search, no topic-text filter — the subreddit IS the curation
    #     and the vision worker culls. Real-photo subs whose titles never contain
    #     the topic word now work.
    #   * search mode (prompt required, or no subreddits): keyword-search for
    #     prompt-bearing AI-art posts (the original behaviour).
    listing_mode = _prompt_optional() and bool(subreddit_allowlist)
    apply_topic_filter = not _prompt_optional()

    # old.reddit.com's legacy .json is far less aggressively blocked than
    # new-reddit's (which 403s direct .json navigations even in a real browser),
    # and it's server-rendered HTML — no SPA to destroy the JS context.
    _BASE = "https://old.reddit.com"
    feeds: list[tuple[str, str]] = []   # (url, label)
    if listing_mode:
        print("  Reddit: prompt not required -> listing subreddits directly "
              "(no AI/prompt search)", flush=True)
        for sub in sorted(subreddit_allowlist):
            for sort in ("hot", "new", "top"):
                t_param = "&t=month" if sort == "top" else ""
                feeds.append((
                    f"{_BASE}/r/{sub}/{sort}.json?limit=50&raw_json=1{t_param}",
                    f"r/{sub}/{sort}",
                ))
    else:
        for query in queries:
            # If an allowlist is set, restrict each query to those subreddits.
            effective_query = query
            if subreddit_allowlist:
                sub_filter = " OR ".join(f"subreddit:{s}" for s in sorted(subreddit_allowlist))
                effective_query = f"({query}) ({sub_filter})"
            print(f"  Reddit query: {effective_query}", flush=True)
            for sort in ("relevance", "new", "top"):
                feeds.append((
                    f"{_BASE}/search.json?q={quote(effective_query)}"
                    f"&sort={sort}&t=month&limit=50&type=link&raw_json=1",
                    effective_query,
                ))

    src_channel = "reddit_listing" if listing_mode else "reddit_search"
    blocked_warned = False
    img_warned = False
    img_skipped = 0
    # Transparency counters so a pass that skips everything as already-seen is
    # distinguishable from one that never fetched anything.
    feeds_ok = 0
    feeds_failed = 0
    candidates = 0
    already_seen = 0

    # Drive a real Chromium browser so Reddit sees a genuine session/fingerprint
    # rather than a raw-requests bot. gallery-dl's Reddit path is untouched.
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                headless=True,
                # Reduce headless-automation fingerprinting Reddit screens for.
                args=["--disable-blink-features=AutomationControlled"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Reddit: could not launch Chromium ({exc}). Install the "
                  "Playwright browser: `python -m playwright install chromium`.", flush=True)
            return saved

        ctx = browser.new_context(user_agent=BROWSER_UA, locale="en-US",
                                  viewport={"width": 1440, "height": 900})
        # Hide the biggest automation tell (navigator.webdriver === true).
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        # over18 + redesign_optout keep us on old.reddit with NSFW visible;
        # REDDIT_COOKIES (if set) unlock gated / quarantined subs behind a login.
        ctx.add_cookies([
            {"name": "over18", "value": "1", "domain": ".reddit.com", "path": "/"},
            {"name": "_options", "value": "%7B%22pref_quarantine_optin%22%3A%20true%7D",
             "domain": ".reddit.com", "path": "/"},
        ] + REDDIT_COOKIES)
        print(f"  Reddit: {len(REDDIT_COOKIES)} cookie(s) loaded; fetching via "
              f"{_BASE} (browser)", flush=True)
        page = ctx.new_page()
        # Each feed/image is its own page.goto, so we never sit on a live SPA whose
        # hydration navigations would destroy the JS execution context.

        for feed_url, feed_label in feeds:
            try:
                _LIMITER.acquire()
                data, err = _browser_fetch_json(page, feed_url)
                if err is not None:
                    feeds_failed += 1
                    # Surface the actual response (status + body snippet) once, so a
                    # block vs. a redirect vs. an empty result is diagnosable rather
                    # than a silent 0.
                    if not blocked_warned:
                        blocked_warned = True
                        print(f"  Reddit fetch failed [{feed_label}]: {err}", flush=True)
                        print("  → If HTTP 403/429: paste your logged-in reddit.com "
                              "cookies into REDDIT_COOKIES (Settings). Fetching uses "
                              "old.reddit.com via a real browser.", flush=True)
                    continue

                posts = (data or {}).get("data", {}).get("children", [])
                feeds_ok += 1
                candidates += len(posts)
                for post in posts:
                    d = post.get("data", {})
                    post_id   = d.get("id","")
                    title     = d.get("title","")
                    body      = d.get("selftext","")
                    permalink = d.get("permalink","")
                    upvotes   = d.get("ups", 0)
                    subreddit = d.get("subreddit","")

                    if not post_id:
                        continue
                    if post_id in seen:
                        already_seen += 1
                        continue

                    # Subreddit allowlist (belt + braces: the feed already restricts)
                    if subreddit_allowlist and subreddit.lower() not in subreddit_allowlist:
                        dropped_subreddit += 1
                        seen.add(post_id)
                        continue

                    # Extract prompt from title + body
                    combined  = f"{title}\n{body}"
                    prompt    = extract_prompt(combined)
                    if len(prompt) < 30:
                        prompt = title  # fallback to title as minimal prompt

                    # Topic filter: only when a prompt is REQUIRED. When prompts are
                    # optional the user wants every image pulled (a real-photo
                    # subreddit's titles rarely contain the topic word); the vision
                    # worker does the relevance culling instead.
                    if apply_topic_filter:
                        ok, reason = _topic_passes(combined, prompt, cfg=topic_cfg)
                        if not ok:
                            dropped_offtopic += 1
                            seen.add(post_id)
                            continue

                    # Collect media per the job's policy: images and/or video.
                    media: list[tuple[str, str]] = []   # (url, kind)
                    if media_policy.wants_image():
                        media += [(u, "image") for u in reddit_image_urls(d)]
                    if media_policy.wants_video():
                        media += [(u, "video") for u in reddit_video_urls(d)]
                    if not media:
                        seen.add(post_id)  # nothing of the wanted kind(s) here
                        continue

                    saved_this = 0
                    for i, (murl, kind) in enumerate(media[:4]):  # cap per post
                        stem = f"reddit_{post_id}_{i}"
                        _LIMITER.acquire()
                        mbytes, merr = _browser_fetch_bytes(page, murl, want=kind)
                        if mbytes is None:
                            img_skipped += 1
                            # Surface the first failure so an HTML gate / 403 is
                            # visible instead of a silent skip.
                            if not img_warned:
                                img_warned = True
                                print(f"  Reddit {kind} skipped [{murl[:70]}]: {merr}", flush=True)
                            continue
                        ok = save_to_queue(stem, murl, mbytes, prompt, {
                            "source_channel": src_channel,
                            "source_guild":   f"reddit.com/r/{subreddit}",
                            "author":         d.get("author",""),
                            "timestamp":      str(d.get("created_utc","")),
                            "reddit_post_id": post_id,
                            "reddit_url":     f"https://reddit.com{permalink}",
                            "upvotes":        upvotes,
                            "query":          feed_label,
                        }, source="reddit")
                        if ok:
                            saved += 1
                            saved_this += 1

                    # Only mark the post seen once a file actually saved, so a
                    # transient failure (403/timeout/HTML) retries next run
                    # instead of being permanently skipped.
                    if saved_this:
                        seen.add(post_id)

                time.sleep(1)  # polite between feeds

            except Exception as exc:  # noqa: BLE001
                print(f"  [Reddit ERR] {feed_label}: {exc}")

            seen.flush()

        browser.close()

    print(
        f"  Reddit done: saved={saved} new | already_seen={already_seen} | "
        f"candidates={candidates} across feeds_ok={feeds_ok}/{len(feeds)} "
        f"(failed={feeds_failed}) | img_skipped={img_skipped} "
        f"dropped_offtopic={dropped_offtopic} dropped_subreddit={dropped_subreddit}",
        flush=True,
    )
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Web Scraper (Reddit) ===", flush=True)
    print(f"Topic: {TOPIC}", flush=True)
    print(f"Queue: {QUEUE_DIR}", flush=True)

    seen_reddit = SeenStore("reddit", slug=SLUG, autoflush_every=10)

    total = 0

    print("\n[Reddit]", flush=True)
    total += scrape_reddit(seen_reddit)

    seen_reddit.flush()

    print(f"\n=== Web scraper done. Total saved: {total} ===", flush=True)
