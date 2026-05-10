---
name: metadata-schema
description: Use when modifying cull's classification JSON schema, adding/removing fields the vision worker emits, changing scoring keys, or working on the .vision.json audit record. Authoritative pointer to vision_prompt.build_response_format() — the single source of truth.
---

# metadata-schema

cull writes two artefacts per image after classification:

1. **`<stem>.txt`** — the prompt / caption (existing or auto-generated).
2. **`<stem>.vision.json`** — audit record of the vision call.

The schema for the JSON the vision model returns is defined in **one place**:
[`vision_prompt.build_response_format()`](../../../pipeline_code/vision_prompt.py). The
same shape lives at [`pipeline_code/vision_response_schema.json`](../../../pipeline_code/vision_response_schema.json)
so you can paste it into LM Studio's chat panel for ad-hoc testing.

## Required fields (from `build_response_format`)

| Field | Type | Notes |
|---|---|---|
| `description` | string (10–1000) | One-paragraph summary. |
| `primary_subject` | string (3–200) | What the image is centrally about. |
| `is_screenshot` | boolean | Filters obvious non-images. |
| `is_composite_grid` | boolean | Multi-panel / contact-sheet detection. |
| `contains_text_overlay` | boolean | Watermarks, captions, memes. |
| `is_human_photograph` | boolean | Real photo vs synthetic. |
| `art_medium` | enum | `photograph`, `digital_painting`, `anime`, `3d_render`, `illustration`, `mixed`, `unclear`. |
| `photorealistic_style` | boolean | |
| `has_ai_flaws` | boolean | Hands, eyes, anatomy artefacts. |
| `woman_present` | boolean | Topic-relevance gate. |
| `nsfw` | boolean | |
| `OVR_Quality_Score` | int 0–100 | Overall composition / craft. |
| `REL_Quality_Score` | int 0–100 | Topic-relevance score. |
| `quality_score` | int 1–10 | Coarse triage score. |
| `category` | enum from [`categories.SCHEMA_CATEGORIES`](../../../pipeline_code/categories.py) | NEVER inline this list. |
| `reason` | string (5–300) | Why the category was chosen. |
| `caption` | string (0–2000) | Auto-caption or empty. Required-but-empty when `AUTO_CAPTION_ENABLED=false`. |

`additionalProperties: false` and `strict: true` — adding a field server-side without updating the schema causes LM Studio to reject the response.

## Modifying the schema

1. Edit `build_response_format()` in [`vision_prompt.py`](../../../pipeline_code/vision_prompt.py).
2. If the new field affects routing, update [`apply_scores()`](../../../pipeline_code/vision_prompt.py) or the relevant gate in [`vision_worker_base.py`](../../../pipeline_code/vision_worker_base.py).
3. Re-run the import smoke check.
4. Optionally regenerate `vision_response_schema.json` from the function output for the LM Studio chat panel.

## Categories

Categories live in [`categories.py`](../../../pipeline_code/categories.py):

- `CATEGORIES` — kept buckets (per-topic).
- `TERMINAL_CATEGORIES` — `DISCARD`, `CORRUPT`.
- `ALL_CATEGORIES` — union; do not redeclare anywhere else.
- `SCHEMA_CATEGORIES` — the enum the JSON schema actually exposes.

Adding a category is a single edit to `CATEGORIES`. Workers mkdir new folders on next start; the schema enum and prompt instruction text rebuild automatically.

## Caption field

The schema's `caption` is **always required** because strict-mode JSON schemas can't have conditional fields. When `AUTO_CAPTION_ENABLED=false`, the prompt instructs the model to return `""`. When `true`, the prompt swaps in style-specific instructions (`sd_prompt`, `booru_tags`, `natural_language` — see `CaptionConfig`) and `vision_worker_base._finalise` writes the caption to `<stem>.txt`. Existing source-side prompts are preserved unless `AUTO_CAPTION_OVERWRITE=true`.

## What's stored alongside the image

After a successful classification, the vision worker base class atomically writes:

```
sorted/<slug>/<category>/<source>/<stem>.<ext>          # the image
sorted/<slug>/<category>/<source>/<stem>.txt            # caption (existing or auto)
sorted/<slug>/<category>/<source>/<stem>.vision.json    # full audit record
```

The `.vision.json` includes the model's strict-schema response plus host metadata (worker name, model id, latency, retry count). Treat it as immutable history — don't rewrite it from a downstream tool; instead, use `tools/requeue_sorted.py` to re-run classification and produce a new audit record.
