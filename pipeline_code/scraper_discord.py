"""
scraper_discord.py — Discord channel scraper (runs as parallel agents)
Fetches images + prompts from Discord channels, drops to queue/ folder.
Does NOT do vision — just download + prompt extract + basic prompt validation.
Saves to source-based queue using queue_manager.
"""
import os, re, json, time, struct, requests, sys, tempfile
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# DISCORD_BOT_TOKEN holds either a real bot token OR a personal user-account
# token, depending on DISCORD_AUTH_MODE:
#   "bot"   - prepend "Bot " to the Authorization header (real Discord bot).
#   "user"  - send the token raw (your own account; required when the bot
#             can't be added to private servers).
#   "auto"  - try `Bot <token>` first, fall back to raw on 401. Default.
# DISCORD_AUTH_PREFIX (legacy) still works as a manual override.
from credentials import get_optional, warn_if_missing  # noqa: E402

USER_TOKEN = get_optional("DISCORD_BOT_TOKEN") or ""
if not USER_TOKEN:
    print("[scraper_discord] WARNING: DISCORD_BOT_TOKEN not set in .env", flush=True)

_AUTH_MODE = os.environ.get("DISCORD_AUTH_MODE", "auto").strip().lower()
_LEGACY_PREFIX = os.environ.get("DISCORD_AUTH_PREFIX")
if _LEGACY_PREFIX is not None:
    # Honour legacy override: empty string -> user mode, "Bot " -> bot mode.
    _AUTH_MODE = "user" if _LEGACY_PREFIX.strip() == "" else "bot"

if _AUTH_MODE == "bot":
    _AUTH_HEADER = f"Bot {USER_TOKEN}"
elif _AUTH_MODE == "user":
    _AUTH_HEADER = USER_TOKEN
else:  # auto - start as bot, may flip to user on 401
    _AUTH_MODE = "auto"
    _AUTH_HEADER = f"Bot {USER_TOKEN}"

print(
    f"[scraper_discord] Auth mode={_AUTH_MODE!r}, header prefix="
    f"{'Bot ' if _AUTH_HEADER.startswith('Bot ') else '(none/user-token)'}",
    flush=True,
)
from paths import base_dir

BASE_DIR   = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
SLUG       = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
QUEUE_DIR  = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG

# Import queue manager + dedup
from queue_manager import save_to_queue  # noqa: E402
from seen_store import SeenStore  # noqa: E402

from topic_filter import (
    load_config as _load_topic_config,
    passes as _topic_passes,
    prompt_optional as _prompt_optional,
)
import media_policy
from rate_limit import RateLimitConfig, RateLimiter  # noqa: E402

# Per-source pacing + 429 backoff + optional proxy. Opt-in: unthrottled unless
# RATE_LIMIT_DISCORD_* env is set, so existing behaviour is unchanged.
_LIMITER = RateLimiter(RateLimitConfig.from_mapping("discord", os.environ))


def _retry_after_seconds(resp) -> float | None:
    """Parse Discord's Retry-After (header or JSON ``retry_after`` body) → float."""
    if resp is None:
        return None
    raw = resp.headers.get("Retry-After")
    if not raw:
        try:
            raw = resp.json().get("retry_after")
        except Exception:
            raw = None
    if raw is None or raw == "":
        return None
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None

MESSAGES_PER_CHANNEL = 400
_TOPIC_CFG = _load_topic_config()
MIN_PROMPT_LENGTH = _TOPIC_CFG.min_prompt_length
DL_HEADERS = {"Authorization": _AUTH_HEADER}

def is_valid_prompt(text, context: str = "") -> bool:
    """Shared topic-aware filter. `context` is any surrounding text (channel,
    message author); `text` is the prompt itself."""
    ok, _reason = _topic_passes(context, text, cfg=_TOPIC_CFG)
    return ok

_AUTH_HINT_SHOWN = False


def _flip_auth_to_user() -> bool:
    """In auto-mode, flip the global Authorization header to user-token style.

    Returns True if the flip happened (so the caller can retry once).
    """
    global _AUTH_HEADER, _AUTH_MODE
    if _AUTH_MODE != "auto":
        return False
    if not _AUTH_HEADER.startswith("Bot "):
        return False
    _AUTH_HEADER = USER_TOKEN
    DL_HEADERS["Authorization"] = _AUTH_HEADER
    print(
        "  [auth-fallback] 401 with 'Bot <token>'; retrying as USER token "
        "(your personal account). Set DISCORD_AUTH_MODE=user in .env to "
        "skip this dance on next run.",
        flush=True,
    )
    return True


def fetch_messages(channel_id, limit=400):
    global _AUTH_HINT_SHOWN
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    results, params = [], {"limit": 100}
    while len(results) < limit:
        _LIMITER.acquire()
        resp = requests.get(url, headers=DL_HEADERS, params=params, **_LIMITER.requests_kwargs())
        if resp.status_code == 429:
            _LIMITER.note_429(retry_after=_retry_after_seconds(resp))
            print(f"  [429] Rate limited on {channel_id}, backing off...")
            continue
        if resp.status_code == 403:
            print(f"  [SKIP] No access: {channel_id} (token not a member of server / missing intent)")
            return []
        if resp.status_code == 401:
            # Auto mode: flip from Bot -> user once and retry the same request.
            if _flip_auth_to_user():
                _LIMITER.acquire()
                resp = requests.get(url, headers=DL_HEADERS, params=params, **_LIMITER.requests_kwargs())
                if resp.status_code == 200:
                    pass  # fall through to normal handling
                else:
                    print(f"  [ERR] 401 even after user-token fallback ({channel_id})", flush=True)
                    return []
            else:
                if not _AUTH_HINT_SHOWN:
                    tail = USER_TOKEN[-6:] if USER_TOKEN else "(empty)"
                    print(
                        f"  [AUTH-FAIL 401] token ending ...{tail} rejected in mode={_AUTH_MODE!r}. "
                        "Fixes: (a) regenerate the token in the Discord Developer Portal, "
                        "or (b) flip DISCORD_AUTH_MODE between 'bot' / 'user' / 'auto' in Settings. "
                        "User mode = use your personal Discord account token (against TOS but works "
                        "for private servers your account is in).",
                        flush=True,
                    )
                    _AUTH_HINT_SHOWN = True
                else:
                    print(f"  [ERR] 401 on {channel_id}", flush=True)
                return []
        if resp.status_code != 200:
            print(f"  [ERR] {resp.status_code}")
            break
        _LIMITER.note_success()
        batch = resp.json()
        if not batch: break
        results.extend(batch)
        params["before"] = batch[-1]["id"]
        if len(batch) < 100: break
        time.sleep(0.4)
    return results[:limit]

def download_bytes(url):
    try:
        _LIMITER.acquire()
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}, **_LIMITER.requests_kwargs())
        if r.status_code == 429:
            _LIMITER.note_429(retry_after=_retry_after_seconds(r))
            return None
        if r.status_code == 200:
            _LIMITER.note_success()
            return r.content
        return None
    except: return None

def extract_prompt_from_png(data):
    if not data or len(data) < 8 or data[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    pos, texts = 8, {}
    while pos < len(data):
        if pos + 12 > len(data): break
        length = struct.unpack(">I", data[pos:pos+4])[0]
        ctype  = data[pos+4:pos+8].decode("ascii", errors="ignore")
        cdata  = data[pos+8:pos+8+length]
        pos   += 12 + length
        if ctype == "tEXt":
            try:
                k, v = cdata.split(b'\x00', 1)
                texts[k.decode()] = v.decode("latin-1")
            except: pass
        elif ctype == "iTXt":
            try:
                parts = cdata.split(b'\x00', 4)
                texts[parts[0].decode()] = parts[4].decode("utf-8", errors="replace")
            except: pass
        elif ctype == "IEND": break

    for key in ("workflow", "prompt", "parameters", "Parameters"):
        if key not in texts: continue
        val = texts[key]
        try:
            obj = json.loads(val)
            if isinstance(obj, dict):
                pos_texts = []
                for node in obj.values():
                    if not isinstance(node, dict): continue
                    title = node.get("_meta", {}).get("title", "").lower()
                    if node.get("class_type") == "CLIPTextEncode":
                        txt = node.get("inputs", {}).get("text", "")
                        if isinstance(txt, str) and txt.strip():
                            if "positive" in title or ("negative" not in title and "neg" not in title):
                                pos_texts.append(txt.strip())
                if pos_texts: return pos_texts[0]
        except:
            clean = val.split("Negative prompt:")[0].split("Steps:")[0].strip()
            if len(clean) > MIN_PROMPT_LENGTH: return clean
    return None

def extract_prompt_from_yaml(yaml_bytes):
    try:
        text = yaml_bytes.decode("utf-8", errors="replace")
        idx = text.find("'prompt':") if "'prompt':" in text else text.find('"prompt":')
        if idx == -1: return None
        start = text.index("{", idx)
        depth, end = 0, start
        for i, c in enumerate(text[start:], start):
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0: end = i; break
        workflow = json.loads(text[start:end+1].replace("\\'", "'"))
        pos_texts = []
        for node in workflow.values():
            if not isinstance(node, dict): continue
            title = node.get("_meta", {}).get("title", "").lower()
            if node.get("class_type") == "CLIPTextEncode":
                txt = node.get("inputs", {}).get("text", "")
                if isinstance(txt, str) and txt.strip():
                    if "positive" in title: pos_texts.insert(0, txt.strip())
                    elif "negative" not in title and "neg" not in title: pos_texts.append(txt.strip())
        return pos_texts[0] if pos_texts else None
    except: return None

def process_channel(ch):
    print(f"\n-> {ch['name']} ({ch['guild']})", flush=True)
    seen = SeenStore("discord", slug=SLUG, autoflush_every=25)
    messages = fetch_messages(ch["id"])
    print(f"   {len(messages)} messages fetched", flush=True)
    queued = 0

    for msg in messages:
        atts = msg.get("attachments", [])

        def _att_ok(a) -> bool:
            ct = a.get("content_type", "") or ""
            url = a["url"].split("?")[0].lower()
            if media_policy.wants_image() and ("image" in ct or url.endswith(media_policy.image_exts())):
                return True
            if media_policy.wants_video() and ("video" in ct or url.endswith(media_policy.video_exts())):
                return True
            return False

        images = [a for a in atts if _att_ok(a)]
        yamls  = [a for a in atts if a.get("filename","").endswith((".yaml",".yml"))]

        embed_imgs = []
        for emb in msg.get("embeds", []):
            if emb.get("image"): embed_imgs.append(emb["image"]["url"])
            if emb.get("thumbnail"): embed_imgs.append(emb["thumbnail"]["url"])

        # Get prompt from yaml first, then PNG, then message text
        prompt_text = None
        for y in yamls:
            yb = download_bytes(y["url"])
            if yb:
                prompt_text = extract_prompt_from_yaml(yb)
                if prompt_text: break

        all_img_urls = [a["url"] for a in images] + embed_imgs
        if not all_img_urls: continue

        # Detect MJ grid via component buttons — grids have U1/U2/U3/U4 buttons,
        # single upscaled images have "Upscale (Subtle)" / "Vary (Strong)" etc.
        is_mj_grid = False
        if msg.get("author", {}).get("username") == "Midjourney Bot":
            button_labels = set()
            for row in msg.get("components", []):
                for btn in row.get("components", []):
                    lbl = btn.get("label", "") or btn.get("emoji", {}).get("name", "")
                    button_labels.add(str(lbl).strip())
            grid_buttons = {"U1", "U2", "U3", "U4"}
            if grid_buttons & button_labels:
                is_mj_grid = True

        for img_url in all_img_urls:
            # Skip MJ grids detected via buttons or fallback dimension/filename checks
            if is_mj_grid:
                continue
            clean_url = img_url.split("?")[0].lower()
            if "_grid_" in clean_url:
                continue
            att_meta = next((a for a in images if a["url"] == img_url), None)
            if att_meta:
                w = att_meta.get("width", 0)
                h = att_meta.get("height", 0)
                if w >= 2048 and h >= 2048:
                    continue

            uid = msg["id"] + "_" + img_url.split("?")[0]
            if uid in seen: continue

            img_bytes = download_bytes(img_url)
            if not img_bytes: continue

            # Also skip grids detected by actual image size after download
            try:
                from PIL import Image as _PIL
                import io as _io
                _im = _PIL.open(_io.BytesIO(img_bytes))
                _w, _h = _im.size
                if _w >= 2048 and _h >= 2048:
                    continue  # grid
            except: pass
            if not img_bytes: continue

            # Try PNG metadata if no yaml prompt
            pt = prompt_text
            if not pt and img_url.split("?")[0].lower().endswith(".png"):
                pt = extract_prompt_from_png(img_bytes)

            # Fall back to message text
            if not pt:
                parts = [msg.get("content","").strip()]
                for emb in msg.get("embeds",[]):
                    if emb.get("description"): parts.append(emb["description"])
                    if emb.get("footer",{}).get("text"): parts.append(emb["footer"]["text"])
                pt = " | ".join(p for p in parts if p).strip() or None

            # Topic-text filter only when a prompt is REQUIRED. When prompts are
            # optional the user wants every image from the configured channels
            # pulled regardless of caption; the vision worker culls afterwards.
            if not _prompt_optional() and not is_valid_prompt(pt):
                continue

            # Save to queue. Keep the real extension when the policy accepts it
            # (so a .mp4 clip stays a clip); otherwise fall back to jpg.
            ext = img_url.split("?")[0].split(".")[-1].lower()
            if not media_policy.accepts("." + ext): ext = "jpg"
            qid = f"{msg['id']}_{queued}"
            
            # Create temp file for queue_manager
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            tmp_path.write_bytes(img_bytes)
            
            # Determine source based on guild
            source = "discord_ud" if "unstable" in ch["guild"].lower() else "discord_mj"
            
            # Prepare metadata
            meta_data = {
                "message_id": msg["id"],
                "image_url": img_url,
                "source_channel": ch["name"],
                "source_guild": ch["guild"],
                "author": msg["author"]["username"],
                "timestamp": msg["timestamp"],
                "queued_at": datetime.utcnow().isoformat(),
            }
            
            # Save to source-based queue
            save_to_queue(source, tmp_path, pt, meta_data)

            seen.add(uid)
            queued += 1

        if queued % 20 == 0 and queued > 0:
            seen.flush()

    seen.flush()
    print(f"   Queued {queued} items from {ch['name']}", flush=True)
    return queued

# Channels passed as JSON arg
if __name__ == "__main__":
    channels = json.loads(sys.argv[1])
    total = 0
    for ch in channels:
        total += process_channel(ch)
    print(f"\nTotal queued: {total}", flush=True)
