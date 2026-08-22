---
recipe: lora-training-dataset
goal: Build a 500-image LoRA training set on a named subject
inputs:
  subject: "A short subject phrase, e.g. 'goldendoodle puppy at the beach'"
  slug: "URL-safe job name: a-z 0-9 _"
  target_count: 500
outputs:
  - "data/sorted/<slug>/Keep/**/*.png (with .txt captions)"
  - "kohya-shaped export folder at <out>/"
prereqs:
  - "GROQ_API_KEY or LMSTUDIO_PRIMARY_URL configured"
  - "Optional: TWITTER_COOKIES for X.com, CIVITAI_API_KEY for Civitai"
---

# Recipe: build a 500-image LoRA training set

Curate a Kohya-ready training set of ~500 in-topic images with captions, from
scraped sources and quality-gated by the vision model.

## 1. Pick a preset

```bash
cull presets list --json
```

For a photo-realistic subject, the general-purpose `default` preset is fine.
For anime / illustration, use `anime_illustration`. For portrait / person work
use `photoreal_portrait` (has Keep / OffTopic / etc. defined for people).

## 2. Create the job

```bash
cull job create <slug> --preset default --subject "<subject phrase>" --json
```

## 3. Configure sources

Set at least one URL-driven source or scraper. Examples:

```bash
# Add a gallery-dl URL (Reddit thread, Pixiv gallery, DeviantArt page, ...)
cull scrapers add-url --job <slug> --source gallery_dl \
    --url https://www.reddit.com/r/<subreddit>/

# Or enable X.com if TWITTER_COOKIES is set:
cull scrapers toggle --job <slug> --name X.com --enabled true
```

Disable sources you don't want to spend on:

```bash
cull scrapers toggle --job <slug> --name Discord-1 --enabled false
```

## 4. Tune scoring gates

Start moderate. Loosen only if the queue drains without hitting the target.

```bash
cull scoring set --job <slug> --min-ovr 65 --min-rel 55 --require-prompt false
```

## 5. Activate + run

```bash
cull jobs activate <slug>
cull run &                  # detach; the supervisor is blocking
```

## 6. Wait for the target

```bash
cull jobs watch --slug <slug> \
    --until "sorted-count>=500" --timeout 7200 --interval 30 --json
```

Exit codes:

- `0` — condition met (proceed to step 7).
- `3` — timed out. Loosen the score gate (`cull scoring set --min-ovr 55`),
  add more sources (step 3), or extend the timeout, then re-run watch.

## 7. Inspect the distribution before exporting

```bash
cull stats --job <slug> --json
```

Compute the kept ratio from `counts_by_category["Keep"] / sorted_count`.
Above 0.30 is healthy; below 0.15 usually means the gates or sources need
another pass.

Sanity-check a few random keepers:

```bash
cull gallery sample --job <slug> --category Keep --n 5 --json
```

## 8. Export

```bash
cull export kohya --job <slug> --out /path/to/lora_dataset --json
```

## Success criteria

- Exit 0 from `cull export kohya`.
- Summary shows `sample_count >= 500`.
- The output directory contains `.png` + matching `.txt` pairs, no
  `.vision.json` sidecars, and no `DISCARD` / `CORRUPT` folders.
- Kept ratio (`Keep / sorted_count`) is above 0.30.
