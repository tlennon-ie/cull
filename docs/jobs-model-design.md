# cull — Jobs Model Design

> Status: APPROVED FOR BUILD · Author: design pass 2026-06-28
> This doc is the shared contract for the parallel build agents. Read it before touching code.

## 1. Goal

Today cull holds one curation target's config in a single flat `.env`. Running a
new target means overwriting the previous one. We are promoting the latent
**slug** concept into a first-class **Job**: a named curation target with its own
self-contained config bundle. The pipeline does not run until a job is **active**.

Global concerns (credentials, model endpoints) stay in `.env`. Everything that
defines *what to scrape and how to judge it* moves into the job.

## 2. Decisions (locked)

| Decision | Choice |
|---|---|
| Execution | **Sequential job queue.** One supervisor runs the active job; on stop/done it advances to the next queued job. No concurrency. |
| Config store | **Per-job JSON** at `data/jobs/<slug>.json`, plus `data/jobs/_index.json` for queue order + active pointer. |
| Vision boundary | **Endpoints global** (provider, URLs, keys, timeouts, throttle, idle-unload). **Per-job:** auto-captioning (enable/style/overwrite) + score gates (ovr/rel/notes). |
| Rollout | Parallel multi-agent build, frequent test scripts, commits, code reviews. Migrate current `.env` into a `default` job so nothing is lost. |

## 3. The keystone: projection, not rewiring

A Job is the **source of truth**. Activating a job **projects** its config down
into the contracts the runtime *already* consumes:

1. **Env vars** — `job_config.resolve_env(job)` flattens the JSON into the existing
   env-var names (`PIPELINE_TOPIC`, `TOPIC_KEYWORDS_EXTRA`, `X_ACCOUNTS`,
   `REDDIT_SUBREDDITS`, `SCRAPER_DISABLED`, `VISION_OVR_MIN_SCORE`,
   `AUTO_CAPTION_*`, `GALLERY_DL_URLS`, `LOCAL_IMPORT_*`, …). The supervisor merges
   this over the global `.env` base env when spawning children. **Scrapers and
   vision workers are unchanged** — they still read `os.environ`.
2. **Taxonomy** — `job_config.project_categories(job)` writes the job's category
   list into `data/cull_categories.json` via `categories.set_active(...)`. The
   supervisor already watches that file's mtime and soft-restarts workers, and
   `vision_prompt` rebuilds the schema enum from it. **No worker change.**

Because activation is just "write env + write `cull_categories.json`", switching
the active job reuses the supervisor's **existing hot-reload machinery**
(`.env` mtime watch + `cull_categories.json` mtime watch + structural restart).

```
data/jobs/<slug>.json  ──(activate)──►  resolve_env()  ──►  merged spawn env  ──►  scrapers + vision workers
                        └──────────────►  project_categories()  ──►  data/cull_categories.json  ──►  schema enum
```

## 4. Filesystem layout (unchanged where it already works)

```
data/
  jobs/
    _index.json                 # { "active": "<slug>|null", "queue": ["slug", ...], "updated_at": "..." }
    female_influencer.json      # one job config bundle
    car_ads.json
  queue/<slug>/<source>/...     # already per-slug — unchanged
  sorted/<slug>/<cat>/<src>/... # already per-slug — unchanged
  seen_<name>_<slug>.json       # already per-slug — unchanged
  cull_categories.json          # active taxonomy — now PROJECTED from the active job
  cull_index.sqlite3            # already has topic_slug column — unchanged
  logs/pipeline_<slug>.log      # already per-slug — unchanged
```

`paths.py` gains `jobs_dir()` → `<base>/jobs`. Nothing else in `paths.py` changes.

## 5. Job config schema (`data/jobs/<slug>.json`)

`slug` is the identity (folder/namespace key); never changes after create.

```jsonc
{
  "slug": "female_influencer",            // identity, immutable, [a-z0-9_]+
  "name": "Female Influencer",            // display name (editable)
  "status": "idle",                        // idle | queued | running | paused | done
  "created_at": "2026-06-28T00:00:00Z",
  "updated_at": "2026-06-28T00:00:00Z",

  "topic": {
    "topic": "Realistic Female Influencer",
    "keywords_extra": ["influencer", "model"],
    "banned_keywords": ["anime"],
    "generation_hints": ["photorealistic"],
    "min_prompt_length": 30,
    "require_prompt": true
  },

  "scrapers": {
    // per-job enable/disable, keyed by the dashboard scraper NAME (see _STATIC_SCRAPERS)
    "enabled": {
      "X.com": false, "Discord-1": false, "Civitai-Com": true, "Civitai-Red": true,
      "Web": true, "ZFF-Local": false, "Gallery-DL": false
    },
    "x_accounts": ["someacct"],
    "reddit_subreddits": ["EarthPorn"],
    "discord_channels_json": "[]",        // stored as JSON string (matches DISCORD_CHANNELS_JSON)
    "civitai_domains": ["civitai.com"],
    "gallery_dl": { "enabled": false, "urls": [], "limit_per_url": 200,
                     "cookies_file": "", "config_path": "" },
    "local_import": { "enabled": false, "dir": "", "name": "local", "migrate_from": "" },
    "zforfree": { "local_enabled": false, "web_enabled": false, "local_src": "" }
  },

  "categories": [ { "name": "Photorealistic", "hint": "..." }, ... ],  // same shape as cull_categories.json "categories"
  "category_rules": "STRICT JUDGEMENT RULES ...",                      // per-job global_rules text

  "scoring": { "ovr_min": 60, "rel_min": 55, "notes": "" },

  "captioning": { "enabled": false, "style": "sd_prompt", "overwrite": false }
}
```

### Global (stay in `.env`, NOT in job)
Credentials & endpoints only: `GROQ_API_KEY(S)`, `GROQ_MODEL`, `LMSTUDIO_*`,
`OPENAI_COMPAT_*`, `OLLAMA_*`, `CIVITAI_API_KEY(_RED)`, `TWITTER_COOKIES`,
`DISCORD_BOT_TOKEN`, `DISCORD_AUTH_MODE`, `REDDIT_CLIENT_ID/SECRET/USER_AGENT`,
`PIPELINE_BASE_DIR`, `LOG_DIR`, vision worker selection
(`PIPELINE_VISION_WORKERS`), throttle, `BLUR_NSFW_THUMBS`,
`PIPELINE_RECONCILE_SECONDS`, `LMSTUDIO_UNLOAD_ON_STOP/IDLE_UNLOAD_MINUTES`.

> Note: scraper *credentials* are global; scraper *enable/disable* and *targets*
> (accounts, subreddits, channels JSON, gallery-dl urls, local dirs) are per-job.

## 6. `job_config.py` — public API (Phase A, the keystone module)

```python
# Types
@dataclass(frozen=True) class Job: slug, name, status, created_at, updated_at, data: dict
# data holds topic/scrapers/categories/category_rules/scoring/captioning sub-dicts

# Constants
JOB_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
JOB_STATUSES = ("idle", "queued", "running", "paused", "done")

# CRUD / store
def jobs_dir() -> Path                          # delegates to paths.jobs_dir()
def list_jobs() -> list[Job]                    # sorted by queue order then name
def get_job(slug) -> Job | None
def save_job(job: Job) -> Job                    # atomic write <slug>.json, bumps updated_at
def create_job(name, *, base_on: str|None=None, **overrides) -> Job   # slugify name; clone if base_on
def delete_job(slug) -> None                     # refuses if slug == active; caller confirms data loss
def slugify(name: str) -> str                    # reuse run_pipeline.topic_slug rules

# Queue / active pointer (data/jobs/_index.json)
def get_index() -> dict                          # {"active": str|None, "queue": [slug,...]}
def set_active(slug: str|None) -> None           # validates job exists; updates _index.json (mtime → supervisor reload)
def get_active_slug() -> str | None
def set_queue(order: list[str]) -> None          # validates all exist
def enqueue(slug) / dequeue(slug) -> None
def advance() -> str | None                      # pop current active, promote head of queue, return new active

# Projection (the keystone)
def resolve_env(job: Job) -> dict[str, str]      # job → existing env-var names
def project_categories(job: Job) -> None         # write job.categories+rules into cull_categories.json
def activate(slug: str) -> None                  # set_active + project_categories (atomic-ish)

# Migration
def migrate_env_to_default_job() -> Job | None   # if jobs dir empty, build "default" job from current os.environ + cull_categories.json; set active. Idempotent.
```

### `resolve_env` mapping (authoritative — both supervisor & dashboard rely on it)

| Job field | Env var |
|---|---|
| `slug` | `PIPELINE_SLUG` |
| `topic.topic` | `PIPELINE_TOPIC` |
| `topic.keywords_extra` (csv) | `TOPIC_KEYWORDS_EXTRA` |
| `topic.banned_keywords` (csv) | `TOPIC_BANNED_KEYWORDS` |
| `topic.generation_hints` (csv) | `TOPIC_GENERATION_HINTS` |
| `topic.min_prompt_length` | `MIN_PROMPT_LENGTH` |
| `topic.require_prompt` | `REQUIRE_PROMPT` |
| `scrapers.enabled` (the False ones, by name, csv) | `SCRAPER_DISABLED` |
| `scrapers.x_accounts` (csv) | `X_ACCOUNTS` |
| `scrapers.reddit_subreddits` (csv) | `REDDIT_SUBREDDITS` |
| `scrapers.discord_channels_json` | `DISCORD_CHANNELS_JSON` |
| `scrapers.civitai_domains` (csv) | `CIVITAI_DOMAINS` |
| `scrapers.gallery_dl.enabled` | `GALLERY_DL_ENABLED` |
| `scrapers.gallery_dl.urls` (newline) | `GALLERY_DL_URLS` |
| `scrapers.gallery_dl.limit_per_url` | `GALLERY_DL_LIMIT_PER_URL` |
| `scrapers.gallery_dl.cookies_file` | `GALLERY_DL_COOKIES_FILE` |
| `scrapers.gallery_dl.config_path` | `GALLERY_DL_CONFIG_PATH` |
| `scrapers.local_import.*` | `LOCAL_IMPORT_ENABLED/DIR/NAME/MIGRATE_FROM` |
| `scrapers.zforfree.*` | `ZFORFREE_LOCAL_ENABLED/WEB_ENABLED/ZFORFREE_LOCAL_SRC` |
| `scoring.ovr_min` | `VISION_OVR_MIN_SCORE` |
| `scoring.rel_min` | `VISION_REL_MIN_SCORE` |
| `scoring.notes` | `VISION_SCORE_NOTES` |
| `captioning.enabled` | `AUTO_CAPTION_ENABLED` |
| `captioning.style` | `AUTO_CAPTION_STYLE` |
| `captioning.overwrite` | `AUTO_CAPTION_OVERWRITE` |

Booleans → `"true"`/`"false"`. Lists → comma-joined except gallery-dl urls
(newline-joined, matching today). Empty/missing → empty string.

## 7. Supervisor integration (`run_pipeline.py`)

- On startup: `job_config.migrate_env_to_default_job()`; read active slug.
- Replace the `for topic in topics` loop with: run the **active job**. Build the
  spawn base env as `{**os.environ, **resolve_env(active_job)}`; call
  `project_categories(active_job)` before spawning.
- Watch `data/jobs/_index.json` mtime alongside `.env` and `cull_categories.json`.
  When `active` changes → structural restart with the new job's env+categories
  (reuse existing `STRUCTURAL_ENV_KEYS` path; the resolved env vars are already in
  that set).
- **Advance:** when the active job is stopped/marked `done`, call
  `job_config.advance()` to promote the next queued job, then restart into it.
  (Manual "stop" from dashboard sets active→none and does not auto-advance; an
  explicit "mark done / next" advances. Keep it simple: dashboard drives advance.)

## 8. Dashboard — API additions (`dashboard_enhanced.py`)

New routes:

| Route | Method | Purpose |
|---|---|---|
| `/api/jobs` | GET | list jobs + status + counts (queue/sorted from index_store per slug) + queue order + active |
| `/api/jobs` | POST | create `{name, base_on?}` → new job (slugified) |
| `/api/jobs/<slug>` | GET | full job config |
| `/api/jobs/<slug>` | PUT | update job config (validated) |
| `/api/jobs/<slug>` | DELETE | delete (refuse if active/running; `?force=1` to also drop data dirs — guarded by `safe_inside`) |
| `/api/jobs/<slug>/clone` | POST | clone to new name |
| `/api/jobs/<slug>/activate` | POST | `job_config.activate(slug)` (projects + sets active; supervisor restarts) |
| `/api/jobs/queue` | POST | set `{order:[...]}` |
| `/api/jobs/<slug>/enqueue` / `/dequeue` | POST | add/remove from queue |

Existing endpoints gain an optional `?job=<slug>` (default = active slug) and
scope by it. Most already filter by `topic_slug` in `index_store`, so this is
mostly threading the slug through:
`/api/status`, `/api/activity`, `/api/queue/files`, `/api/logs/history`,
`/api/stats`, `/api/gallery`, `/api/gallery/insights`, `/api/gallery/download.zip`.

**Scraper toggles** (`/api/scrapers`, `/toggle`, `/bulk`) now read/write the job's
`scrapers.enabled` for `?job=<slug>` (default active) instead of global
`SCRAPER_DISABLED`. If `slug == active`, also re-project env so the running
supervisor picks it up (write via `activate`/`set_active` mtime bump).

**Categories** (`/api/categories` GET/POST) now read/write the job's `categories`
+ `category_rules`. POST to the active job also re-projects into
`cull_categories.json`.

**Settings** (`/api/settings`) shrinks to GLOBAL keys only (credentials, endpoints,
vision selection, throttle, base paths, blur, reconcile). Per-job keys move to the
job editor.

## 9. Dashboard — UI restructure (Alpine.js in `HTML_TEMPLATE`)

New top-level state: `view: 'jobs' | 'job'`, `currentJob: slug|null`.

- **Jobs view (landing):** grid of job cards (name, status pill, queue position,
  queued/sorted counts, actions: Open · Activate · Pause · Edit · Clone · Delete)
  + "New Job" card. This is the first thing the user sees.
- **Enter a job → job view**, default tab **Historical** (job-scoped). Job-scoped
  tabs reuse today's sections, all filtered by `currentJob`:
  - Historical (default), Queue, Scrapers (per-job toggles), Vision
    (global endpoint readout + this job's captioning/scoring editor), Stats
    (this job), Overview (this job), Job Settings (the job config editor:
    topic/keywords/subreddits/x-accounts/discord/gallery-dl/local/categories/
    scoring/captioning). A "← Jobs" back control.
- **Queue tab** shows the **job queue** as cards (active job + next queued) per the
  user's description; clicking a job card drills into that job's **image** queue
  (today's queue table), scoped to the job slug.
- **Global Settings** (reachable from Jobs view): credentials, model endpoints,
  vision provider selection, throttle, base paths, blur, reconcile.
- **Global Stats/Overview** (from Jobs view): top-level aggregate with a per-job
  filter dropdown.

Keep FAQ/About. Vision endpoint config stays "as today" (global); only
captioning+scoring become job-scoped within the job's Vision tab.

## 10. Migration & back-compat

- `migrate_env_to_default_job()` runs at dashboard + supervisor startup. If
  `data/jobs/` has no job files, it builds `default` from current `os.environ`
  (topic block, scraper targets, scrapers.enabled from `SCRAPER_DISABLED`, scoring,
  captioning) + current `cull_categories.json` (categories + rules), writes it,
  sets it active. Idempotent — never overwrites existing jobs.
- Old `.env` per-job keys may remain; they're simply ignored once a job is active
  (job env overrides them at spawn). `.env.example` updated to mark them legacy.

## 11. Build plan (parallel agents, with review gates)

**Phase A (BLOCKING, single focused stream):** `paths.jobs_dir()` +
`job_config.py` + migration + `tests/test_job_config.py` (round-trip, slugify,
resolve_env mapping, migration idempotency, queue advance). Code review → commit.
Everything below imports this; it must be green first.

**Phase B (parallel — disjoint files):**
- **B1 Supervisor:** `run_pipeline.py` integration + `tests/test_supervisor_jobs.py`.
- **B2 Dashboard (owns `dashboard_enhanced.py` entirely):** API routes first, then
  HTML/Alpine UI. One agent owns this whole file to avoid intra-file conflicts.
  Flask test-client tests for jobs CRUD + scoping.
- **B3 Docs/config:** `.env.example`, `CLAUDE.md`, `README.md`,
  `.claude/skills/cull-helper/SKILL.md` updates describing the jobs model.

**Phase C:** integration — run the 25-module import smoke test + new pytest
suites + boot the dashboard and hit `/api/jobs`. Cross-cutting code review
(factual + security: path traversal on delete, secret handling). Merge.

## 12. Invariants to honor (from cull-helper)
- One source of truth per concern — `job_config.py` is THE jobs source of truth;
  do not scatter job state.
- Structured-output schema stays mandatory; categories still flow through
  `cull_categories.json` → `vision_prompt.build_response_format()`.
- `safe_inside()` guards every user-supplied path (esp. job delete with data).
- Subprocess workers print; library code (`job_config.py`) uses
  `pipeline_logging.get_logger`.
- Atomic writes for `<slug>.json` and `_index.json` (temp + `os.replace`).
