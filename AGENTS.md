# AGENTS.md — Operator guide for driving cull

This file is for an AI agent (or automation) **using** cull to accomplish a
curation goal — building a LoRA training set, deduping an archive, triaging
scraped content. Prefer this over reading [`CLAUDE.md`](CLAUDE.md) unless you
are modifying cull's own code.

For editing cull's internals, read [`CLAUDE.md`](CLAUDE.md) and the skills
under [`.claude/skills/`](.claude/skills/).

---

## What cull is (for an agent)

cull is a single-machine image-curation engine. Given a **subject** and a
**preset**, it scrapes candidate images from configured sources, classifies
each one with a vision-language model against a strict JSON schema, and sorts
keepers into category folders next to a caption (`.txt`) and audit record
(`.vision.json`).

Two things you need to internalise:

1. **Jobs are the unit of work.** A *job* wraps `{slug, subject, preset,
   overrides}`. Everything you do — scoring gates, scraper URLs, category
   rules, exports — is attached to a job.
2. **Presets are inherited-by-default.** A job carries only *sparse overrides*
   on top of a preset. Editing a job's `--min-ovr` writes an override; leaving
   it alone means the preset default wins. Change the shipped preset library
   by editing files under [`presets/`](presets/), not per-job.

## Your interface

The `cull` CLI is the sanctioned agent-facing surface. Every subcommand takes
`--json` for machine-readable output. See [`pipeline_code/cull_cli.py`](pipeline_code/cull_cli.py)
for the source of truth.

### Exit codes (contract)

| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | Bad arguments / invalid predicate |
| 3 | `jobs watch` timed out before the condition matched |
| 4 | Missing job or preset |
| 5 | Subprocess or export failure |

### Subcommands

Every command emits JSON on stdout when passed `--json`, prefixed with
`{"ok": true, …}` on success or `{"ok": false, "error": …, "exit_code": N}`
on failure. Do NOT parse the human-readable output.

#### `cull jobs list [--json]`

Enumerates every job on disk. JSON shape:
```json
{"ok": true, "active": "slug-or-null", "active_slugs": ["slug", ...],
 "jobs": [{"slug": "...", "name": "...", "status": "...", "subject": "...",
           "preset": "...", "active": bool}, ...]}
```

#### `cull jobs activate <slug> [--json]`

Marks a job active and projects its config into env vars +
`cull_categories.json`. JSON: `{"ok": true, "activated": "slug",
"active_slugs": [...]}`.

#### `cull jobs watch --slug SLUG [--until PREDICATE] [--interval N] [--timeout N] [--json]`

Polls a job's sorted / queue counts until a stop predicate matches. Predicate
grammar:

- `sorted-count>=N` / `sorted-count<=N` / `sorted-count=N`
- `queue-count>=N` / `queue-count<=N` / `queue-count=N`
- `active-slug=X`
- `elapsed>=Ns`

JSON on success: `{"ok": true, "condition_met": true, "ticks": N,
"elapsed": seconds, "state": {"slug":"...", "sorted-count":N,
"queue-count":N, "active-slug":"..."}}`. Timeouts exit `3` with
`"condition_met": false`.

With neither `--until` nor `--timeout` set, the command emits one snapshot
and exits 0 (useful for polling from a scripted parent).

#### `cull job create <slug> [--preset NAME] [--subject "text"] [--json]`

Creates a new job. JSON: `{"ok": true, "job": {"slug": "...", "name": "...",
"preset": "...", "subject": "...", "status": "..."}}`.

#### `cull presets list [--json]`

Lists the available preset library. JSON: `{"ok": true, "default": "name",
"presets": [{"name": "...", "builtin": bool, "is_default": bool}, ...]}`.

#### `cull status [--json]`

Global snapshot. JSON: `{"ok": true, "active_slug": "...",
"active_slugs": [...], "queue": [...], "data_dir": "...",
"counts": {"queue": N, "sorted": N}}`.

#### `cull stats [--job SLUG] [--json]`

Per-job category counts + a 10-point-bucketed OVR/REL score distribution.
JSON: `{"ok": true, "slug": "...", "queue_count": N, "sorted_count": N,
"counts_by_category": {"Keep": N, ...},
"score_distribution": {"ovr": {"80-89": N, ...}, "rel": {"70-79": N, ...}},
"nsfw_count": N, "watermark_count": N, "with_prompt": N}`.

#### `cull gallery sample --job SLUG [--category NAME] [--n N] [--json]`

Returns up to `N` random records from `sorted/<slug>[/<category>]`. Each
sample carries the raw `vision_json` payload so you can inspect the model's
verdict without re-parsing `.vision.json`. JSON:
`{"ok": true, "slug": "...", "category": "...", "requested": N,
"returned": N, "samples": [{"path", "relative_path", "category", "ovr",
"rel", "quality_score", "nsfw", "watermark", "prompt", "vision_json"},
...]}`.

#### `cull scoring set --job SLUG [--min-ovr N] [--min-rel N] [--require-prompt BOOL]`

Writes score-gate overrides. Ranges are 0-100 for scores; booleans accept
`true/false/yes/no/1/0/on/off`. JSON:
`{"ok": true, "slug": "...", "updates": {"scoring.ovr_min": N, ...},
"overrides": {...}}`.

#### `cull scrapers list [--job SLUG] [--json]`

Per-job scraper on/off map + gallery-dl / yt-dlp URL counts + local imports.
JSON: `{"ok": true, "slug": "...", "scrapers": [{"name": "X.com",
"enabled": true}, ...], "gallery_dl": {"enabled": bool, "url_count": N,
"limit_per_url": N}, "yt_dlp": {"enabled": bool, "url_count": N},
"local_imports": [...], "x_accounts": [...], "reddit_subreddits": [...]}`.

#### `cull scrapers add-url --job SLUG --source {gallery_dl,yt_dlp} --url URL [--json]`

Appends a URL to a URL-driven scraper. Passes through the SSRF guard
(rejects localhost / private / link-local hosts, exit `2`). Auto-enables the
source when it is the first URL added. JSON:
`{"ok": true, "slug": "...", "source": "gallery_dl", "url": "...",
"changed": bool, "urls": [...]}`.

#### `cull scrapers toggle --job SLUG --name NAME --enabled BOOL [--json]`

Enables or disables a scraper. `NAME` must be one of `SCRAPER_NAMES`
(`X.com`, `Discord-1`, `Civitai-Com`, `Civitai-Red`, `Web`, `Gallery-DL`).

#### `cull config show [--job SLUG] [--json]`

Prints the fully-merged effective config. `api_key`, `token`, `cookies`,
`password`, `secret`, `authorization` values are masked to `***`. Never
log the raw JSON — the masking is defensive, but adjacent fields (URLs
carrying auth params, custom `config_json` blobs) can still carry secrets.

#### `cull run`

Starts the supervisor for the currently active job. Blocking. Prefer
launching this in a detached process and driving with `cull jobs watch`.

#### `cull export kohya --job SLUG --out PATH [--json]`

Convenience wrapper on `export_profiles.export_dataset(slug, "kohya", out)`.
Writes a flat image + caption pair tree under `PATH`.

#### `cull export hf --job SLUG --repo user/name [--public] [--include-video] [--json]`

Pushes the curated set to a HuggingFace dataset repo (PRIVATE by default).
Requires `HF_TOKEN` in the environment. Never surfaces the token.

#### `cull export <slug> --profile P --out PATH [--json]` (legacy)

The pre-existing profile export contract. Prefer the two convenience wrappers
above; only reach for this when you need `webdataset`, `folders`, or
`clip_caption`.

---

## Golden-path workflows

Concrete step-by-step recipes for the four common goals live under
[`recipes/`](recipes/):

- [`recipes/lora-training-dataset.md`](recipes/lora-training-dataset.md) —
  build a 500-image LoRA training set for a named subject.
- [`recipes/personal-archive-triage.md`](recipes/personal-archive-triage.md) —
  dedupe + score-rank a local folder you already own.
- [`recipes/brand-ad-curation.md`](recipes/brand-ad-curation.md) —
  commercial-quality images from mixed sources.
- [`recipes/dedupe-existing-folder.md`](recipes/dedupe-existing-folder.md) —
  a pure dedup pass; no new scraping.

Follow the numbered `cull` commands. Every recipe ends with a success check.

---

## Interpreting `.vision.json`

Every classified image lands next to a JSON audit record. The schema is
enforced by structured output — the fields are always present.

| Field | Type | Meaning |
|------|------|---------|
| `category` | string | The kept-bucket / DISCARD / CORRUPT label. |
| `OVR_Quality_Score` | int 0-100 | Overall image quality (composition, sharpness, lighting). |
| `REL_Quality_Score` | int 0-100 | Relevance to the subject / topic. |
| `quality_score` | int 1-10 | Legacy coarse quality band. |
| `nsfw` | bool | The classifier flagged adult content. |
| `watermark` | bool | Visible watermark / studio logo detected. |
| `caption` | string | Auto-generated caption (empty when captioning off). |

For deeper schema notes, load the `metadata-schema` skill.

Do NOT edit `.vision.json` by hand. If you need to re-classify an image,
move it back into `data/queue/<slug>/<source>/` and let the vision workers
pick it up on the next tick.

---

## Deciding when to stop

Curation is open-ended. Use one of these signals as a hard stop:

- **Sorted count** — target N images in a specific keep bucket:
  `cull jobs watch --slug S --until "sorted-count>=500"`.
- **Queue drain** — every scraped image has been classified:
  `cull jobs watch --slug S --until "queue-count<=0"`.
- **Time budget** — bounded run:
  `cull jobs watch --slug S --until "elapsed>=3600s"`.
- **Kept ratio** — poll `cull stats --json` and compute
  `counts_by_category["Keep"] / sorted_count`. Below 0.15 usually means the
  score gate or scraper targets need tightening.

Combine them: run to `sorted-count>=500` under a `--timeout 7200` guardrail
and treat timeout (exit 3) as "loosen a gate or add sources, then retry".

---

## Failure modes and recovery

### Scraper auth failure

Symptom: `cull scrapers list` shows a source on but its counters stay at 0
across two ticks. The supervisor prints `MissingCredentialError` in
`logs/pipeline_<slug>.log`.

Fix: set the missing env var (see below) and either restart the supervisor
or wait for its per-scraper cool-down retry.

### Score gate too strict

Symptom: `sorted_count` grows slowly, `counts_by_category["Keep"]` is nearly
empty, `OFF_TOPIC` and `DISCARD` folders bloat. `stats --json` shows OVR
distribution concentrated below your gate.

Fix: `cull scoring set --job S --min-ovr 55` (dropping ~10 points at a time
is usually right). Prefer never dropping REL below 40 — a model that
matches on relevance despite being off-subject is often a captioning bug.

### Disk full

Symptom: exit code `5` from any command that writes, plus
`OSError: [Errno 28]` in logs.

Fix: `cull export …` the current job to an external target and delete the
`data/queue/<slug>/` tree while the supervisor is stopped. Never delete
under `data/queue/` while `cull run` is executing — the atomic
`.processing` rename that guards the cross-worker lock lives in that tree
and mid-flight deletes race the vision workers.

### Vision worker unresponsive

Symptom: `queue_count` climbs, `sorted_count` flat, worker log silent for
minutes.

Fix: check `LMSTUDIO_PRIMARY_URL` (or the appropriate cloud provider) is
reachable. Restart the supervisor (`cull run` again — it is idempotent and
picks up wherever the queue was).

---

## Never do

- **Never edit `.vision.json` by hand.** The vision workers own the schema
  and re-classify by re-queuing.
- **Never delete `data/queue/<slug>/` while `cull run` is executing.** The
  atomic `.processing` rename is the cross-worker lock; mid-flight deletes
  race the workers.
- **Never override `PIPELINE_SLUG` in the environment.** The active slug is
  managed by `cull jobs activate` (which also projects categories); a stray
  env var routes writes to a slug that has no job config.
- **Never call vision workers or scrapers directly.** Everything routes
  through the supervisor so it can throttle, retry, and log.
- **Never commit `data/`.** It's gitignored. Rely on `cull export …` for
  data you want to move off-machine.

---

## Environment variables

Credentials are global (they live in `.env`, never per-job):

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` (or `GROQ_API_KEYS`) | Cloud Groq vision worker. |
| `LMSTUDIO_PRIMARY_URL` | Local LM Studio vision worker (defaults to `http://127.0.0.1:1234`). |
| `CIVITAI_API_KEY` | Civitai scrapers. |
| `TWITTER_COOKIES` | X.com scraper session cookies. |
| `DISCORD_BOT_TOKEN` | Discord scraper (bot token, NOT a user token). |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | `cull export hf`. |
| `PIPELINE_BASE_DIR` | Override the data root (defaults to `<repo>/data`). |
| `FLASK_HOST` | Bind host for the dashboard (defaults to `127.0.0.1`). |

Per-job targets (X account list, subreddits, gallery-dl URLs, category
rules, score gates) are stored in the job's `overrides` — set them through
the CLI, not the environment.

---

For code-level changes read [`CLAUDE.md`](CLAUDE.md) and the skills under
[`.claude/skills/`](.claude/skills/).
