---
name: pipeline-helper
description: Use when working on the image-classification pipeline repo. Provides architecture invariants, the right module to touch for each kind of change, run/test commands, and guardrails (path traversal, atomic .processing rename, structured-output schema enforcement).
---

# Pipeline Helper

Load this skill any time you're contributing to or operating the
image-classification pipeline. It collapses the load-bearing parts of
[`CLAUDE.md`](../../../CLAUDE.md) into instinctive rules so you don't have to
re-read the whole file each session.

## Invariants — never violate

1. **One source of truth per concern.** If you're tempted to inline a
   category list, build a JSON-schema enum, hardcode a path, or invent a new
   seen-set / credentials helper, stop. The right module already exists:

   | Concern | Module |
   |---|---|
   | Categories | `pipeline_code/categories.py` |
   | Vision worker registration | `pipeline_code/vision_workers.py` |
   | Vision worker scaffolding | `pipeline_code/vision_worker_base.py` |
   | Filesystem paths | `pipeline_code/paths.py` |
   | Queue (FS by default) | `pipeline_code/queue_manager.py` |
   | Per-source dedup | `pipeline_code/seen_store.py` |
   | API credentials | `pipeline_code/credentials.py` |
   | Logging | `pipeline_code/pipeline_logging.py` |
   | Classification prompt + schema | `pipeline_code/vision_prompt.py` |

2. **Structured output is mandatory.** Every vision worker sends
   `build_response_format()` (or its provider equivalent) on every request.
   Never rely on the model returning unconstrained text and a regex parser to
   recover JSON — that failure mode was fixed in commit `e1a3849` and several
   bugs since.

3. **Atomic `.processing` rename is the cross-worker lock.** In
   `BaseVisionWorker._process_image`, `image_path.rename(processing_path)` is
   what prevents two workers grabbing the same image. Don't replace it with
   any "better" lock; the loser's `FileNotFoundError` short-circuit is the
   correct behaviour.

4. **Path-traversal guards.** Any dashboard endpoint that accepts a
   user-supplied path goes through `safe_inside(raw, [PIPELINE_QUEUE,
   PIPELINE_SORTED])`. Never read or write a path that didn't come back from
   that helper.

5. **Subprocess workers print, library code logs.** Scrapers and vision
   workers run as subprocess children of the supervisor; their `print(...,
   flush=True)` lines end up in `pipeline_<slug>.log` with `[label]` prefixes
   and that's the format humans tail. Library code (queue_manager, seen_store,
   etc.) uses `pipeline_logging.get_logger(__name__)`.

## Common tasks

### Add a new scraper source

1. Copy `pipeline_code/scraper_civitai.py` as a template.
2. Replace API specifics (URL, headers, auth, response shape).
3. Use `credentials.get_required("YOUR_KEY", scraper="your-name")` for any
   required key; let `MissingCredentialError` propagate (it's `SystemExit`).
4. Use `seen_store.SeenStore("your-name", slug=SLUG)` for dedup. Call
   `seen.add(id)` after each successful download and `seen.flush()` between
   batches.
5. Save items via `queue_manager.save_to_queue(source, tmp_path, prompt, meta)`.
6. Register in `run_pipeline.py` `compute_desired_agents` (search for
   `add(AgentSpec(label="...", script="scraper_..."`).
7. Add a row to `_STATIC_SCRAPERS` in `dashboard_enhanced.py` so it shows
   up in the Scrapers tab toggle list.
8. Add the credential key to `SETTINGS_KEYS` in `dashboard_enhanced.py` so
   admins can enter it via UI.

### Add a new vision provider

1. Subclass `BaseVisionWorker` in a new file `pipeline_code/vision_worker_<name>.py`.
2. Implement `classify_image_bytes(b64_jpeg, prompt_instruction) -> dict | None`.
   Returning `None` triggers a uniform RETRY.
3. Override `setup()` if you need model discovery or a keepalive thread.
4. Override `banner()` if your config needs more than the default startup log.
5. Register in `vision_workers.py` `WORKERS` dict with a `WorkerSpec`.
6. Add the name to `ALLOWED_VISION_WORKERS` and `_VISION_WORKER_DESCRIPTIONS`
   in `dashboard_enhanced.py`.
7. If the provider needs new env vars, add them to `SETTINGS_KEYS` and the
   credentials card in the Settings tab.

### Change classification quality / categories

- **Adjust scoring thresholds:** `VISION_OVR_MIN_SCORE` / `VISION_REL_MIN_SCORE`
  in `.env`. Surface in dashboard via Settings tab (already wired).
- **Add a category:** edit `pipeline_code/categories.py` `CATEGORIES` tuple.
  Workers will mkdir the new folder on next start; the JSON schema enum and
  prompt instruction text rebuild from this tuple automatically.
- **Tune the prompt:** `build_classification_prompt()` in `vision_prompt.py`.
  Watch for confusing the model — every change here costs you re-running
  the full backlog through `tools/requeue_sorted.py`.
- **Tune post-hoc validation:** `apply_scores()` in `vision_prompt.py`.
  Negation-aware keyword matching is in `_contains` + `_is_negated_at`.

### Operate / debug a running pipeline

- **Boot:** `./launch.sh` (Linux/Mac) or `launch.bat` (Windows).
- **See what's running:** dashboard sidebar pill says "running" / "stopped".
- **Per-process logs:** `data/logs/pipeline_<slug>.log`. Each line prefixed
  with the agent label.
- **Stuck `.processing` files:** the supervisor's `_sweep_stale_processing`
  recovers them on restart. Don't delete them by hand mid-run.
- **Dashboard endpoint sanity check:**
  ```bash
  curl -s http://localhost:5000/api/status | python -m json.tool
  ```

### Smoke-test before committing

```bash
python -c "import sys; sys.path.insert(0, 'pipeline_code'); import importlib; [importlib.import_module(m) for m in (
  'paths','pipeline_logging','categories','vision_workers','vision_prompt',
  'queue_manager','topic_filter','seen_store','credentials',
  'feed_local_folder','feed_zforfree_local',
  'scraper_civitai','scraper_civitai_search','scraper_x','scraper_discord','scraper_web',
  'vision_worker_base','vision_worker_balanced_lm','vision_worker_balanced_groq',
  'vision_worker_lm_autodetect','vision_worker_lm_keepalive','vision_worker',
  'run_pipeline','integrated_launcher','dashboard_enhanced')]; print('all 25 modules import')"
```

If that prints `all 25 modules import`, your changes haven't broken any
import-time invariants.

## Anti-patterns to call out in code review

- Iterating the queue filesystem directly instead of `Queue.pop_next()`.
- Building a category list inline instead of importing from `categories`.
- `os.environ[KEY]` (raises) where `credentials.get_required` would give a
  cleaner error path.
- A scraper that creates its own `seen_*.json` instead of using `SeenStore`.
- A vision worker that re-implements `.processing` rename / resize /
  save-to-sorted instead of subclassing `BaseVisionWorker`.
- Auto-pip-install in module-import paths (was removed from `balanced_groq`).
- `print(...)` in library code that's not a subprocess worker.

## Reference: where the architecture decisions are documented

The seven release commits each have a long body explaining the *why*. Read
the relevant one before refactoring near it:

- `c196f57` — release cleanup, defaults, README/requirements/LICENSE
- `0568f78` — dashboard release blockers (modal warns, requeue removed, stats spinner)
- `6f3f311` — categories module, a11y, mobile sidebar, expanded settings
- `1848f09` — vision-worker registry, Gemini removal, logging hook
- `9e14117` — Queue Protocol + FSQueue with mtime cache (R4)
- `61fe5cd` — VisionWorker base + thin subclasses (R1)
- `a7e0d88` — seen_store + credentials + scraper migration (R2)
