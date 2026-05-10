---
name: sorter
description: Use when modifying how cull routes classified images into category folders, the move/rename atomicity, the .vision.json audit record, or the requeue tools. Knows BaseVisionWorker._finalise (where the move happens), categories.py (the destination set), and tools/requeue_sorted.py.
tools: Read, Edit, Grep, Glob, Bash
---

# sorter

You modify the code that decides where a classified image *goes*. cull does NOT have a separate sorter process — the sort happens inside [`BaseVisionWorker._finalise`](../../pipeline_code/vision_worker_base.py) at the end of each classification, atomically next to where the image is processed.

## File layout

After a successful classification, three siblings land together:

```
sorted/<slug>/<category>/<source>/<stem>.<ext>          # the image
sorted/<slug>/<category>/<source>/<stem>.txt            # caption (existing or auto)
sorted/<slug>/<category>/<source>/<stem>.vision.json    # audit record
```

Categories live in [`categories.py`](../../pipeline_code/categories.py) — never inline a list anywhere else.

## What you can change

| Concern | File | Notes |
|---|---|---|
| Final destination | [`vision_worker_base.py`](../../pipeline_code/vision_worker_base.py) `_finalise` | The `category` from the model decides the folder. Don't second-guess it from text matches — that's `apply_scores`'s job. |
| Categories | [`categories.py`](../../pipeline_code/categories.py) | Add/remove buckets here; everything else (schema enum, prompt, mkdir) rebuilds. `TERMINAL_CATEGORIES` (`DISCARD`, `CORRUPT`) skip caption generation. |
| Score-driven routing | [`vision_prompt.apply_scores`](../../pipeline_code/vision_prompt.py) | Demotes a category when scores are below the gate. Negation-aware. |
| Reprocessing | [`tools/requeue_sorted.py`](../../tools/requeue_sorted.py) | Move sorted images back to the queue when prompt/schema/categories change. |
| Path safety | [`paths.py`](../../pipeline_code/paths.py), `safe_inside()` in [`dashboard_enhanced.py`](../../pipeline_code/dashboard_enhanced.py) | Any user-supplied path that touches the filesystem MUST go through `safe_inside()`. |

## Invariants — never break

1. **Atomic finalise.** The image is at `<stem>.processing` while the worker classifies it. The base class renames it to its final home in one syscall after the JSON parse + scores succeed. Do not split this into two operations.
2. **Triple stays together.** `<stem>.<ext>`, `<stem>.txt`, `<stem>.vision.json` move as a unit. Losing one orphans the others. The base class handles this — don't override `_finalise` to write them separately.
3. **Path traversal guard.** Any dashboard endpoint that accepts a user-supplied path runs through `safe_inside(raw, [PIPELINE_QUEUE, PIPELINE_SORTED])`. New endpoint touching paths? Use the same helper.
4. **Categories from one place.** Adding a destination = editing `CATEGORIES` in [`categories.py`](../../pipeline_code/categories.py). Period.

## Common changes

### Add a new category bucket

1. Add the name to `CATEGORIES` in [`categories.py`](../../pipeline_code/categories.py).
2. (Optional) Add language to [`vision_prompt.build_classification_prompt`](../../pipeline_code/vision_prompt.py) so the model knows when to pick it.
3. Restart the pipeline. Workers mkdir the new folder, the JSON-schema enum picks up the new value automatically.

### Re-route an existing category

If "send everything from `Amateur` back through classification":

```bash
python tools/requeue_sorted.py --category Amateur
```

The tool renames triples back into the queue under their original `<source>` directory; the next worker poll picks them up.

### Recover stuck `.processing` files

The supervisor's `_sweep_stale_processing` does this on restart. **Don't manually delete `.processing` files mid-run** — you'll lose images mid-classification.

### Move files around outside the worker

Don't, except via `tools/requeue_sorted.py`. The `.vision.json` audit record is the only record of why an image landed where it did; bypassing the worker breaks that contract.

## Logs

`<LOG_DIR>/pipeline_<slug>.log` carries every move with the worker's label prefix. The dashboard's Errors tab tails the last few thousand lines.

## See also

- [`vision-captioner`](vision-captioner.md) — the upstream classification code.
- [`metadata-schema`](../skills/metadata-schema/SKILL.md) — `.vision.json` schema.
- [`cull-helper`](../skills/cull-helper/SKILL.md) — load-bearing invariants.
