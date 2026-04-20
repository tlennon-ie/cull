# CLAUDE.md — Image Pipeline Project

**Project:** Realistic Female Influencer Image Pipeline  
**Purpose:** Multi-source web scraping → vision-based classification → organized asset library  
**Tech Stack:** Python 3.8+, Flask, LMStudio (local AI), Redis/BullMQ (optional), Groq API (cloud), OpenAI-compat APIs  
**Workspace:** `I:\AI\openclaw\workspace\`  
**Main Orchestrator:** `.claude/agents/orchestrator.md`

---

## 🎯 Pipeline Goals

1. **Scrape** images from 7+ sources (Twitter, Reddit, Civitai, Discord, ZforFree, web)
2. **Queue** images with metadata (source, timestamp, original URL, prompt)
3. **Assess** images using local vision models (LMStudio Qwen/Gemma or cloud Groq)
4. **Classify** into categories (Professional, Amateur, NSFW) + subcategories
5. **Sort** to organized folders with complete metadata triple (image + .json + .txt)
6. **Manage** via admin dashboard with error highlighting, worker controls, API switching
7. **Track** historical logs: image → source → folder → classification → prompts generated

---

## 📁 Key Directories

```
I:\AI\openclaw\workspace/
├── prompt-library/           ← Main pipeline code
│  ├── queue/                 ← Images waiting to be processed
│  │  ├── civitai/
│  │  ├── twitter_x/
│  │  ├── reddit/
│  │  ├── discord_mj/
│  │  ├── discord_ud/
│  │  ├── zforfree/
│  │  └── unknown/
│  ├── sorted/                ← Classified & organized images
│  │  └── realistic_female_influencer/
│  │     ├── Professional/
│  │     ├── Amateur/
│  │     ├── NSFW/
│  │     ├── Unknown/
│  │     └── DISCARD/
│  ├── run_pipeline.py        ← Main orchestrator (starts all scrapers + vision worker)
│  ├── queue_manager.py       ← Queue APIs
│  ├── vision_worker*.py      ← Vision assessment workers (Groq, LMStudio)
│  ├── scraper_*.py           ← Individual scrapers (Twitter, Reddit, etc)
│  └── logs_test/             ← Pipeline execution logs
├── dashboard_enhanced.py      ← Admin dashboard (Flask web UI, http://localhost:5000)
├── integrated_launcher.py     ← Starts pipeline + dashboard together
├── claude/                    ← Agent framework (THIS FOLDER)
│  ├── CLAUDE.md              ← This file
│  ├── AGENTS.md              ← Cross-tool mirror
│  ├── agents/                ← Sub-agent definitions
│  └── skills/                ← Knowledge packages
└── I:\AI\Scripts\zforfree\downloads/  ← External ZforFree image source
```

---

## 🔧 Tech Stack Details

### Vision Assessment
- **LMStudio** (local, on-device)
  - Models: Qwen-VL, Gemma-Vision
  - Endpoint: `http://127.0.0.1:8000/v1/` (default) — **configurable via admin panel**
  - Used when local processing preferred
  
- **Groq** (cloud, fast inference)
  - API Key: `GROQ_API_KEY` env var
  - Model: `mixtral-8x7b-32768`
  - Used for faster processing

### Queuing
- **File-system based queue** (no Redis required)
  - Queue depth: monitored per-source
  - Dead letter queue: `queue/unknown/`
  
### Scraping Sources
1. **Twitter/X** — `scraper_x.py` (OAuth token required)
2. **Reddit** — `scraper_reddit.py` (PRAW, credentials in .env)
3. **Civitai** — `scraper_civitai.py` & `scraper_civitai_search.py` (API key)
4. **Discord** — `scraper_discord.py` (bot token, channel IDs)
5. **ZforFree** — Images copied from `I:\AI\Scripts\zforfree\downloads`
6. **Web** — `scraper_web.py` (direct URL)

### Dashboard
- **Framework:** Flask + Flask-CORS
- **Port:** 5000 (configurable)
- **Features:**
  - Real-time queue monitoring
  - Queue file operations (delete, move, requeue)
  - Vision worker control (enable/disable, throttle %, model switching)
  - Scraper control (per-source start/stop)
  - Error highlighting (RED for corrupted files)
  - Historical logs (image → source → folder → classification)
  - Corruption detection
  - API switching (LMStudio IP/port/model, Groq key, etc)

---

## 🚀 How to Run

### Option 1: Integrated (Recommended)
```bash
python integrated_launcher.py
# Starts BOTH pipeline + dashboard
# Dashboard: http://localhost:5000
```

### Option 2: Separate Windows
```bash
# Window 1: Start pipeline
cd prompt-library
python run_pipeline.py --topic "Realistic Female Influencer"

# Window 2: Start dashboard
python dashboard_enhanced.py
# Open http://localhost:5000
```

### Option 3: With Custom Vision Worker
```bash
# Use LMStudio (local) instead of Groq (cloud)
cd prompt-library
python run_pipeline.py --topic "Realistic Female Influencer" --vision-worker balanced-lm

# With custom LMStudio endpoint
python run_pipeline.py --topic "Realistic Female Influencer" --lmstudio-url http://192.168.1.100:8000
```

---

## 📊 Admin Dashboard Features

### Real-Time Monitoring
- Queue depth by source
- Vision worker status
- Scraper logs
- Error log with timestamps
- Throughput metrics

### Admin Controls
- **Pipeline:** Pause/Resume all processing
- **Scrapers:** Enable/Disable per-source (toggle on/off)
- **Vision Worker:** 
  - Enable/Disable
  - Throttle: 0-100%
  - Switch model (Qwen, Gemma, etc)
- **Queue:** 
  - View corrupted files (RED highlighted)
  - Delete/Move/Requeue files
  - Bulk operations
- **API Switching:**
  - Change LMStudio endpoint (IP:port)
  - Change LMStudio model
  - Switch between Groq and LMStudio
  - Add/remove sources

### Historical Logs
- Image file name → Source (civitai/twitter/etc) → Category folder (Professional/Amateur/NSFW) → Classification (via vision worker)
- Prompt history: All prompts generated during assessment
- Traceability: Where did this image come from? What classification was assigned? Why?

---

## ⚙️ Configuration

### Environment Variables
Create `.env` in workspace root:

```bash
# Vision Workers
GROQ_API_KEY=your_groq_key_here
LM_STUDIO_URL=http://127.0.0.1:8000

# Scrapers
TWITTER_API_KEY=your_twitter_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
CIVITAI_API_KEY=your_civitai_key
DISCORD_BOT_TOKEN=your_discord_token

# Pipeline
PIPELINE_TOPIC="Realistic Female Influencer"
BATCH_SIZE=10
WORKER_THREADS=4
```

### Dashboard Config
Edit in `dashboard_enhanced.py` (top of file):

```python
LMSTUDIO_URL = "http://127.0.0.1:8000"  # Change IP/port here
GROQ_MODEL = "mixtral-8x7b-32768"
FLASK_PORT = 5000
```

---

## 🔌 Admin Panel: Switching LMStudio

**Use Case:** Change from local LMStudio to cloud Groq, or switch LMStudio IP/port, or change model

**Via Dashboard:**
1. Open http://localhost:5000
2. Go to "Admin" → "Vision Worker Config"
3. Options:
   - **Provider:** "LMStudio" or "Groq"
   - **LMStudio Endpoint:** 127.0.0.1:8000 (change IP/port)
   - **LMStudio Model:** Qwen-VL or Gemma-Vision (dropdown)
   - **Groq API Key:** (for cloud option)
4. Click "Apply & Restart Worker"

**Via CLI:**
```bash
# Switch to local LMStudio at different IP
python dashboard_enhanced.py --lmstudio-url http://192.168.1.50:8000 --lmstudio-model qwen-vl

# Switch to Groq
python dashboard_enhanced.py --vision-worker groq
```

**Via Code:**
```python
# In queue_manager.py or vision_worker.py
config = {
    "vision_provider": "lmstudio",  # or "groq"
    "lmstudio_url": "http://192.168.1.50:8000",
    "lmstudio_model": "qwen-vl",
    "groq_api_key": os.getenv("GROQ_API_KEY")
}
```

---

## 📋 Rules & Conventions

### File Organization
- **Images:** Always have 3 files per item: `.jpg` (or `.png`/`.webp`) + `.json` metadata + `.txt` text file
- **Metadata triple:** Never separate them; when an image moves from queue → sorted, all 3 move together
- **Naming:** `source_uniqueid_timestamp.jpg` (e.g., `civitai_abc123_1705000000.jpg`)

### Queue Management
- **Depth limit:** None (filesystem limited only)
- **Retry policy:** Failed items go to `queue/unknown/` (dead letter queue)
- **Cleanup:** Auto-remove orphaned metadata files weekly

### Category Taxonomy
```
realistic_female_influencer/
├── Professional/        ← High-quality, branded content
├── Amateur/             ← User-generated, less polished
├── NSFW/                ← Adult content (flagged for review)
├── Unknown/             ← Classification uncertain
└── DISCARD/             ← Corrupted, deleted, or rejected
```

### Vision Assessment Workflow
1. **Input:** Image file + `.json` metadata (source, URL, timestamp)
2. **Processing:** LMStudio/Groq vision model
3. **Output:** 
   - Classification (Professional/Amateur/NSFW)
   - Confidence score
   - Generated prompt/description
   - Quality flags
4. **Metadata update:** Append to `.json` file with timestamp

### Error Handling
- **Corrupted image:** RED highlight in dashboard, moved to DISCARD/
- **Missing metadata:** Flag for reindex
- **Failed assessment:** Retry up to 3x, then dead letter
- **Scraper error:** Log, skip item, continue

---

## 🛠️ Development Workflow

1. **Add new scraper:**
   - Create `scraper_newsource.py` in `prompt-library/`
   - Implement `scrape()` and `save_to_queue()` functions
   - Reference in `run_pipeline.py` launch sequence
   - Test with `python scraper_newsource.py --test`

2. **Modify vision assessment:**
   - Edit prompt in `vision_worker.py` or `vision_worker_*.py`
   - Test with single image: `python vision_worker.py --test-image queue/civitai/test.jpg`
   - Deploy via dashboard restart

3. **Change classification taxonomy:**
   - Update folder structure in `sorted/realistic_female_influencer/`
   - Update routing logic in `sorter.py`
   - Run `python sorter.py --resort-all` to re-sort existing images

4. **Debug pipeline issues:**
   - Check logs: `prompt-library/logs_test/`
   - View dashboard error log (red items)
   - Run health check: `python pipeline_health_check.py`

---

## 📞 Support & Debugging

### Check Pipeline Status
```bash
# See what's running
ps aux | grep python | grep pipeline

# View recent errors
tail -f prompt-library/logs_test/pipeline.log
```

### Dashboard Troubleshooting
- **Dashboard won't start:** Check if port 5000 is in use: `netstat -ano | findstr :5000`
- **Pipeline won't start:** Verify `run_pipeline.py` has execution permissions
- **Vision worker fails:** Check LMStudio is running (http://127.0.0.1:8000/health)

### Common Issues
| Issue | Solution |
|-------|----------|
| "Cannot connect to LMStudio" | Ensure LMStudio running, check IP/port in config |
| "Queue filling up, vision worker slow" | Reduce batch size, enable throttling in dashboard |
| "Metadata files corrupted" | Run `cleanup_orphaned_metadata.py` |
| "Images in wrong folders" | Run `sorter.py --resort-all` |

---

## 📚 Related Files

- **AGENTS.md** — Cross-tool agent definitions
- **.claude/agents/orchestrator.md** — Main coordinator agent
- **.claude/agents/vision-captioner.md** — Vision assessment agent
- **.claude/agents/sorter.md** — Category routing agent
- **.claude/skills/lmstudio-vision/** — LMStudio integration guide
- **.claude/skills/metadata-schema/** — Metadata structure
- **prompt-library/README.md** — Pipeline code documentation

---

Last Updated: 2026-04-19  
Maintained by: [You]
