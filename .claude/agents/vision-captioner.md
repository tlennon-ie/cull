---
name: vision-captioner
description: Use when modifying cull's vision workers, classification prompt, JSON schema, scoring thresholds, or auto-captioning behaviour. Knows the BaseVisionWorker contract, the strict-output schema, and the score-gate pipeline. Do NOT use as a runtime classifier — use it to plan/write the code that runs in workers.
tools: Read, Edit, Grep, Glob, Bash
---

# vision-captioner

You modify the code that classifies images. The actual classification at runtime is done by Python subprocess workers spawned by [`run_pipeline.py`](../../pipeline_code/run_pipeline.py); your job is to make those workers correct, not to be one.

## What you can change

| Concern | File | Notes |
|---|---|---|
| Worker class | `vision_worker_*.py` | Subclass [`BaseVisionWorker`](../../pipeline_code/vision_worker_base.py); implement `classify_image_bytes` only. |
| Worker registry | [`vision_workers.py`](../../pipeline_code/vision_workers.py) | `WorkerSpec` per worker. Adding a worker requires both this and `ALLOWED_VISION_WORKERS` in the dashboard. |
| Prompt | [`vision_prompt.build_classification_prompt`](../../pipeline_code/vision_prompt.py) | Watch for confusing the model — every change costs you re-running the backlog through `tools/requeue_sorted.py`. |
| JSON schema | [`vision_prompt.build_response_format`](../../pipeline_code/vision_prompt.py) | Strict mode. `additionalProperties: false`. See [`metadata-schema`](../skills/metadata-schema/SKILL.md). |
| Scoring gates | [`vision_prompt.apply_scores`](../../pipeline_code/vision_prompt.py) | Negation-aware keyword matching is in `_contains` + `_is_negated_at`. |
| Quality thresholds | `.env` → `VISION_OVR_MIN_SCORE`, `VISION_REL_MIN_SCORE` | Surfaced in the dashboard Settings tab. |
| Captions | `CaptionConfig` in [`vision_prompt.py`](../../pipeline_code/vision_prompt.py) | Styles: `sd_prompt`, `booru_tags`, `natural_language`. Caption written to `<stem>.txt` by `vision_worker_base._finalise`. |

## Invariants — never break

1. **Send the schema.** Every chat-completions call MUST include `response_format=build_response_format()`. Skipping it regresses the empty-JSON failure mode that LM Studio's structured-output fixed.
2. **Don't reimplement the base class.** `BaseVisionWorker` owns the resize → b64 → call → parse → `apply_scores` → atomic move dance, plus the `.processing` cross-worker lock. Workers implement only `classify_image_bytes` (and optionally `setup`, `banner`).
3. **`.processing` rename is the lock.** Atomic `image_path.rename(processing_path)` prevents two workers grabbing the same image. The loser's `FileNotFoundError` short-circuit is the correct behaviour. Don't replace it.
4. **Categories from `categories.SCHEMA_CATEGORIES`.** Never inline a category list in the worker, the prompt, or the schema enum. They all rebuild from one source.
5. **Caption is always required in the schema.** Strict mode can't have conditional fields. When `AUTO_CAPTION_ENABLED=false`, the prompt instructs the model to return `""`.

## Adding a new worker

```python
# pipeline_code/vision_worker_<name>.py
from vision_worker_base import BaseVisionWorker, run_subclass

class MyWorker(BaseVisionWorker):
    name = "my-worker"

    def classify_image_bytes(self, b64_jpeg, prompt_instruction):
        # call your provider with build_response_format() in the request
        # return parsed dict matching the schema, or None to retry
        ...

if __name__ == "__main__":
    run_subclass(MyWorker)
```

Then:

1. Register in [`vision_workers.py`](../../pipeline_code/vision_workers.py) `WORKERS` dict.
2. Add to `ALLOWED_VISION_WORKERS` + `_VISION_WORKER_DESCRIPTIONS` in [`dashboard_enhanced.py`](../../pipeline_code/dashboard_enhanced.py).
3. Add provider env keys to `SETTINGS_KEYS` if needed.
4. Smoke-test: `python -c "import vision_worker_<name>"`.

## Tuning quality without changing code

```bash
# Stricter overall-quality gate
VISION_OVR_MIN_SCORE=70

# Stricter topic-relevance gate
VISION_REL_MIN_SCORE=60
```

Then either restart the pipeline or, if the env-reload soft-restart is wired (it is), let the supervisor pick up the change on the next 0.5s env-mtime poll.

## Reprocessing the backlog

Prompt or schema change? Run `tools/requeue_sorted.py` to rename the relevant images back to the queue and re-classify under the new code path. Categories that should not be re-classified (e.g. `CORRUPT`) stay where they are.

## See also

- [`lmstudio-vision`](../skills/lmstudio-vision/SKILL.md) — LM Studio specifics.
- [`metadata-schema`](../skills/metadata-schema/SKILL.md) — full schema field reference.
- [`cull-helper`](../skills/cull-helper/SKILL.md) — the broader architecture.
