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

USER_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
if not USER_TOKEN:
    print("[scraper_discord] WARNING: DISCORD_BOT_TOKEN not set in .env", flush=True)

BASE_DIR   = Path(os.environ.get("PIPELINE_BASE_DIR", "I:/AI/openclaw/workspace/prompt-library"))
SLUG       = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
QUEUE_DIR  = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG
SEEN_FILE  = BASE_DIR / f"seen_discord_{SLUG}.json"

# Import queue manager
from queue_manager import save_to_queue

# Topic keyword pre-filter — cheap string check before downloading
# Keeps only prompts mentioning women/people; blocks furniture/landscape/random
TOPIC_KEYWORDS_RE = re.compile(
    r"\bwom[ae]n\b|\bgirl\b|\bfemale\b|\blady\b|\bladies\b|\bshe\b|\bher\b|"
    r"\binfluencer\b|\bmodel\b|\bbeauty\b|\bportra(?:it|ying)\b|\bperson\b|\bface\b|"
    r"\bbody\b|\bhair\b|\bskin\b|\blips\b|\beyes\b|\bbreasts?\b|\bnude\b|\bnaked\b|"
    r"\bsexy\b|\blingerie\b|\bbikini\b|\bbabe\b|\bhot\b|\bpinup\b|\bcurvy\b|"
    r"\bblonde\b|\bbrunette\b|\bredhead\b|\b1girl\b|\bsolo\b",
    re.IGNORECASE
)

MESSAGES_PER_CHANNEL = 400
MIN_PROMPT_LENGTH    = 40
FAKE_PROMPT_RE = re.compile(
    r"my workflow|my lora|my model|my pack|my preset|made with|check (?:my|out)|"
    r"link in bio|dm me|for sale|patreon|onlyfans|subscribe|follow me|commission|"
    r"workflow\s*\+\s*lora|best (?:picture|image|photo|gen)|i(?:'ve)? generated|"
    r"i made|i created|these are|here(?:'s| is) my|look at|what do you think|"
    r"rate this|feedback|drop (?:a|your)|let me know", re.IGNORECASE
)
GENERATION_KEYWORDS_RE = re.compile(
    r"photorealistic|hyperrealistic|cinematic|8k|4k|detailed|sharp focus|bokeh|"
    r"depth of field|dof|lighting|portrait|photograph|camera|lens|masterpiece|"
    r"best quality|ultra realistic|skin texture|natural light|studio light|"
    r"golden hour|intricate|detailed face|((long|short|curly|wavy|straight)\s+hair)|"
    r"(wearing|dressed in|outfit|clothing)|(\d+\s*steps?)|cfg|sampler|"
    r"negative prompt|lora:|<lora|model:|checkpoint|aperture|f\/\d|mm lens|"
    r"subject|expression|skin|eyes|teeth|lips|pose", re.IGNORECASE
)
DL_HEADERS = {"Authorization": USER_TOKEN}

def load_seen():
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()

def save_seen(seen):
    SEEN_FILE.write_text(json.dumps(list(seen)))

def is_valid_prompt(text):
    if not text or len(text.strip()) < MIN_PROMPT_LENGTH:
        return False
    if FAKE_PROMPT_RE.search(text.strip()):
        return False
    if not GENERATION_KEYWORDS_RE.search(text):
        return False
    if not TOPIC_KEYWORDS_RE.search(text):
        return False
    return True

def fetch_messages(channel_id, limit=400):
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    results, params = [], {"limit": 100}
    while len(results) < limit:
        resp = requests.get(url, headers=DL_HEADERS, params=params)
        if resp.status_code == 403:
            print(f"  [SKIP] No access: {channel_id}")
            return []
        if resp.status_code != 200:
            print(f"  [ERR] {resp.status_code}")
            break
        batch = resp.json()
        if not batch: break
        results.extend(batch)
        params["before"] = batch[-1]["id"]
        if len(batch) < 100: break
        time.sleep(0.4)
    return results[:limit]

def download_bytes(url):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        return r.content if r.status_code == 200 else None
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
    seen = load_seen()
    messages = fetch_messages(ch["id"])
    print(f"   {len(messages)} messages fetched", flush=True)
    queued = 0

    for msg in messages:
        atts = msg.get("attachments", [])
        images = [a for a in atts if "image" in a.get("content_type","") or
                  a["url"].split("?")[0].lower().endswith((".png",".jpg",".jpeg",".webp"))]
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

            if not is_valid_prompt(pt):
                continue

            # Save to queue
            ext = img_url.split("?")[0].split(".")[-1].lower()
            if ext not in ("jpg","jpeg","png","webp"): ext = "jpg"
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
            save_seen(seen)

    save_seen(seen)
    print(f"   Queued {queued} items from {ch['name']}", flush=True)
    return queued

# Channels passed as JSON arg
if __name__ == "__main__":
    channels = json.loads(sys.argv[1])
    total = 0
    for ch in channels:
        total += process_channel(ch)
    print(f"\nTotal queued: {total}", flush=True)
