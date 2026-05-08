# CLAUDE.md — Agent guide for cull

This file orients an AI coding agent (Claude Code, Cursor, Aider, Codex, etc.) to
the repo. It complements [`README.md`](README.md), which is for human users.

If you're a human reading this: nothing here is required to use the project. Skip to the README.

---

## What cull is

cull is a single-machine curation engine for AI-generated images:

1. **Scrapes** images + their generation prompts from 7+ sources (Civitai, X/Twitter, Reddit, Discord, ZForFree, generic local folders).
2. **Queues** them on the local filesystem under `data/queue/<slug>/<source>/`.
3. **Classifies** each image with a vision-language model (LM Studio or Groq) using a strict JSON schema for structured output.
4. **Sorts** results into category folders alongside the original `.txt` prompt and a `.vision.json` audit record.
5. **Surfaces** everything through a Flask + Alpine.js admin dashboard
   (`http://localhost:5000`) — pipeline control, scraper toggles, gallery
   browsing, prompt editing, ZIP export, per-source analytics.

The product positioning is *automating taste, not running a model*. The architecture is small and the conventions are load-bearing.

## Conventions you must follow

These are load-bearing — breaking them will silently misroute images.

- **Categories** live in [`pipeline_code/categories.py`](pipeline_code/categories.py). Three tuples: `CATEGORIES` (kept buckets), `TERMINAL_CATEGORIES` (DISCARD/CORRUPT), `ALL_CATEGORIES` (everything). Never inline a category list anywhere else.
- **Vision worker registry** lives in [`pipeline_code/vision_workers.py`](pipeline_code/vision_workers.py). Adding a new provider: register a `WorkerSpec` here AND add it to `dashboard_enhanced.ALLOWED_VISION_WORKERS`. The supervisor maps worker name → script via this registry; mismatched names silently no-op.
- **Vision worker base class:** [`pipeline_code/vision_worker_base.py`](pipeline_code/vision_worker_base.py). Subclass `BaseVisionWorker` and implement `classify_image_bytes`. Don't reimplement the resize / rename / save dance — the base owns it.
- **Queue:** use `queue_manager.save_to_queue` / `get_next_image_round_robin` (or the underlying `Queue` Protocol + `FSQueue`). Do NOT iterate the filesystem yourself; the cache layer in `FSQueue` exists for a reason.
- **Dedup:** every scraper uses `seen_store.SeenStore("name", slug=SLUG)`. Adding a new scraper = one `SeenStore(...)` instance + `seen.add(id)` calls and `seen.flush()` between batches. Don't roll your own JSON file.
- **Credentials:** every scraper uses `credentials.get_required("KEY", scraper="name")` for hard requirements and `get_optional` / `get_keylist` for soft ones. `MissingCredentialError` is a `SystemExit` subclass so the supervisor's cooldown applies on misconfigured scrapers.
- **Logging:** library code uses `pipeline_logging.get_logger(__name__)`. Subprocess workers (scrapers, vision workers) keep their `print(..., flush=True)` calls because the supervisor captures and labels stdout cleanly — see [`pipeline_logging.py`](pipeline_code/pipeline_logging.py) for the reasoning.
- **Paths:** [`pipeline_code/paths.py`](pipeline_code/paths.py) is the single source of truth. Default is `<repo>/data/`. Never hardcode an absolute path.
- **Vision prompt + JSON schema:** [`pipeline_code/vision_prompt.py`](pipeline_code/vision_prompt.py) exposes `build_classification_prompt` + `build_response_format` + `apply_scores`. Every worker MUST send the schema in `response_format` (or its provider equivalent) — empty/invalid JSON failure modes were fixed by structured output, don't regress.

## Repository map (where to look first)

| You want to… | Look at |
|---|---|
| Add a new scraper source | [`scraper_civitai.py`](pipeline_code/scraper_civitai.py) as a template, register in [`run_pipeline.py`](pipeline_code/run_pipeline.py) `compute_desired_agents` |
| Add a new vision provider | Subclass [`BaseVisionWorker`](pipeline_code/vision_worker_base.py), register in [`vision_workers.py`](pipeline_code/vision_workers.py), update `ALLOWED_VISION_WORKERS` in [`dashboard_enhanced.py`](pipeline_code/dashboard_enhanced.py) |
| Change classification taxonomy | [`categories.py`](pipeline_code/categories.py) — affects JSON schema + worker mkdir + dashboard automatically |
| Tune classification quality | [`vision_prompt.py`](pipeline_code/vision_prompt.py) — `build_classification_prompt` for the model-side instruction, `apply_scores` for post-hoc validation gates |
| Add a dashboard endpoint | [`dashboard_enhanced.py`](pipeline_code/dashboard_enhanced.py) — single-file Flask + giant `HTML_TEMPLATE` Alpine.js string |
| Swap the queue backend | Implement the `Queue` Protocol from [`queue_manager.py`](pipeline_code/queue_manager.py); change `_default_queue` factory |
| Configure a setting via UI | Add the env var name to `SETTINGS_KEYS` in [`dashboard_enhanced.py`](pipeline_code/dashboard_enhanced.py) and add inputs in the Settings tab template |

## Run / test commands

```bash
# Bootstrap on first run (creates .venv, installs deps, copies .env)
./launch.sh                      # macOS/Linux
launch.bat                       # Windows

# Run individual modules directly (after activating .venv)
python pipeline_code/run_pipeline.py                  # supervisor only
python pipeline_code/dashboard_enhanced.py            # dashboard only
python pipeline_code/integrated_launcher.py           # both together
python pipeline_code/scraper_civitai.py               # ad-hoc scraper run
python tools/seed_demo_data.py                        # synthetic demo data for screenshots

# Fast import sanity check across every active module
python -c "import sys; sys.path.insert(0, 'pipeline_code'); import importlib; [importlib.import_module(m) for m in (
  'paths','pipeline_logging','categories','vision_workers','vision_prompt',
  'queue_manager','topic_filter','seen_store','credentials',
  'feed_local_folder','feed_zforfree_local',
  'scraper_civitai','scraper_civitai_search','scraper_x','scraper_discord','scraper_web',
  'vision_worker_base','vision_worker_balanced_lm','vision_worker_balanced_groq',
  'vision_worker_lm_autodetect','vision_worker_lm_keepalive','vision_worker',
  'run_pipeline','integrated_launcher','dashboard_enhanced')]; print('OK')"
```

## Things you should NOT do

- Do not add a Gemini worker without porting to `google.genai` (the new SDK). The previous attempt was deleted because it used the deprecated `google.generativeai` package and never adopted structured output. See the commit that removed `vision_worker_gemini.py`.
- Do not write secrets into committed files. `.env` is gitignored; `.env.example` is safe but every value there should be a placeholder.
- Do not bypass `safe_inside()` in dashboard endpoints that accept user-supplied paths. It's the only thing preventing path traversal.
- Do not touch the atomic `.processing` rename in vision workers. It's the cross-worker lock; replacing it with anything fancier reintroduces races.
- Do not auto-pip-install dependencies at runtime. The Groq worker used to do this and it broke CI; declare deps in `requirements.txt` instead.

## Where the audit / refactor history lives

The architecture you see today landed in a series of commits whose bodies explain *why* each module looks the way it does. `git log --oneline pipeline_code/` shows them all; the most consequential are:

- `c196f57` — release prep (dead code purged, README/requirements/LICENSE)
- `1848f09` — vision-worker registry + Gemini removal
- `9e14117` — Queue Protocol + FSQueue with mtime cache (R4)
- `61fe5cd` — VisionWorker base + thin subclasses (R1)
- `a7e0d88` — `seen_store.py` + `credentials.py` + scraper migration (R2)

Read those commit bodies before refactoring near them.

## Pointers for AI agents

- **Skill bundle:** [`.claude/skills/cull-helper/`](.claude/skills/cull-helper/SKILL.md) — load this when working on cull.
- **Project guidelines:** the conventions section above is the load-bearing list. Re-read it before opening a PR.
- **Task scope:** small, focused changes. The architect's audit recommended five refactors (R1-R5); they're all done. New work should follow the established seams (registry, Protocol, base class) rather than introducing new top-level abstractions.
- **Testing:** there is no formal test suite yet. Use the import sanity check above as your CI smoke test. If you add a new module, make sure it imports cleanly with the others.
