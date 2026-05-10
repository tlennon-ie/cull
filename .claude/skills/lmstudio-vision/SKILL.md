---
name: lmstudio-vision
description: Use when adding, modifying, or debugging LM Studio vision workers in cull. Covers the OpenAI-compatible chat-completions call, structured-output schema requirement, model autodetect, keepalive vs idle-unload, and the unloader helper. Pairs with cull-helper for the broader architecture.
---

# lmstudio-vision

cull talks to LM Studio over the OpenAI-compatible `POST /v1/chat/completions` endpoint. Three workers wrap that call; pick the right one and stay inside the conventions.

## Worker registry

| Worker | Class | When to use |
|---|---|---|
| `balanced-lm` (primary) | [`BalancedLMWorker`](../../../pipeline_code/vision_worker_balanced_lm.py) | Standard local LM Studio. Reads `LMSTUDIO_PRIMARY_URL` / `LMSTUDIO_PRIMARY_MODEL`. |
| `balanced-lm-secondary` | same script, env-rebound to `LMSTUDIO_SECONDARY_*` | Run two LM Studio hosts in parallel. |
| `lm-autodetect` | [`LMAutodetectWorker`](../../../pipeline_code/vision_worker_lm_autodetect.py) | Picks whichever vision-capable model is currently loaded. |
| `lm-keepalive` | [`LMKeepaliveWorker`](../../../pipeline_code/vision_worker_lm_keepalive.py) | Pings every 15s so LM Studio's idle-unload TTL doesn't fire mid-run. Mutually exclusive with cull's idle-unloader. |

All four subclass [`BaseVisionWorker`](../../../pipeline_code/vision_worker_base.py) — never re-implement the `.processing` rename, resize, or save dance.

## Mandatory: structured output

Every request MUST send `response_format` from [`vision_prompt.build_response_format()`](../../../pipeline_code/vision_prompt.py). LM Studio enforces the JSON schema server-side — any worker that drops it regresses the bug fixed in commit `e1a3849` (empty / malformed JSON returns).

```python
from vision_prompt import build_classification_prompt, build_response_format

payload = {
    "model": self.lms_model,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": build_classification_prompt()},
    ]}],
    "response_format": build_response_format(),
    "temperature": 0.2,
    "max_tokens": 800,
}
```

Categories and the JSON schema enum rebuild from [`categories.py`](../../../pipeline_code/categories.py) — never inline a category list.

## Adding a new LM-Studio-flavoured worker

1. Subclass `BaseVisionWorker`. Implement only `classify_image_bytes(b64_jpeg, prompt_instruction) -> dict | None`.
2. Override `setup()` if you need model discovery (see autodetect) or a keepalive thread.
3. Register in [`vision_workers.py`](../../../pipeline_code/vision_workers.py) `WORKERS` dict with a `WorkerSpec`.
4. Add the name to `ALLOWED_VISION_WORKERS` and `_VISION_WORKER_DESCRIPTIONS` in [`dashboard_enhanced.py`](../../../pipeline_code/dashboard_enhanced.py).
5. Smoke-test: `python -c "import vision_worker_<your_name>"`.

## Unloading models

cull ships [`lmstudio_admin.unload_all()`](../../../pipeline_code/lmstudio_admin.py). It shells out to `lms unload --all` (with per-model fallback for older versions). The dashboard wires three triggers:

- **Manual:** `POST /api/lmstudio/unload` (button next to "Test LM Studio").
- **On stop:** `LMSTUDIO_UNLOAD_ON_STOP=true` (default) — runs after the pipeline subprocess terminates.
- **On idle:** `LMSTUDIO_IDLE_UNLOAD_MINUTES > 0` — runs while the pipeline is up but the queue has been empty that long. Skipped automatically when `lm-keepalive` is in `PIPELINE_VISION_WORKERS`.

LM Studio's REST API does NOT expose a stable unload endpoint; do not regress to `POST /api/v0/models/unload` (returns "Unexpected endpoint or method" on most versions).

## Health check

```bash
curl -s "$LMSTUDIO_PRIMARY_URL/v1/models" | python -m json.tool
```

If `data` is empty, no model is loaded. JIT load fires on the next chat-completions request.
