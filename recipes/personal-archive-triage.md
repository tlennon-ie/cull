---
recipe: personal-archive-triage
goal: Triage and score-rank a local folder you already own; no new scraping
inputs:
  slug: "URL-safe job name for this triage pass"
  archive_dir: "Absolute path to the local folder to import"
  target_kept: 0  # or a target if you want a hard stop
outputs:
  - "data/sorted/<slug>/<category>/**"
  - "OVR / REL scores in each .vision.json"
prereqs:
  - "A working vision worker (LMSTUDIO_PRIMARY_URL or GROQ_API_KEY)"
---

# Recipe: personal archive triage

Take a folder you already own (a camera roll, a scraped archive, a
downloads bucket) and let cull score-rank + bucket every image. No new
scraping — only the local-import feeder runs.

## 1. Create a triage-shaped job

The `quality_only` preset ships without subject / person gates and is
tuned for pure quality triage.

```bash
cull job create <slug> --preset quality_only \
    --subject "personal archive triage" --json
```

## 2. Disable every scraper

You are importing local content only; leaving scrapers on would add noise
and burn credentials for no reason.

```bash
for s in X.com Discord-1 Civitai-Com Civitai-Red Web Gallery-DL; do
    cull scrapers toggle --job <slug> --name "$s" --enabled false
done
```

## 3. Wire up the local import (edit the job's overrides)

`local_imports` is a list of `{name, dir, enabled, migrate_from}`. Set it
via the dashboard OR by adding an override programmatically. From the CLI
you can inspect the current shape:

```bash
cull config show --job <slug> --json
```

Then edit `data/jobs/<slug>.json` directly and set:

```json
{
  "overrides": {
    "scrapers": {
      "local_imports": [
        {"name": "archive", "dir": "/absolute/path/to/your/folder",
         "enabled": true, "migrate_from": ""}
      ]
    }
  }
}
```

(An `add-local-import` CLI verb is on the roadmap; today this is the
one-line manual edit.)

## 4. Set loose gates so nothing is dropped for score alone

Triage is about ranking, not gating.

```bash
cull scoring set --job <slug> --min-ovr 0 --min-rel 0 --require-prompt false
```

## 5. Activate + run

```bash
cull jobs activate <slug>
cull run &
```

## 6. Wait for the queue to drain

Every imported image lands in `data/queue/<slug>/local/` first. The vision
workers pop them into category folders.

```bash
cull jobs watch --slug <slug> \
    --until "queue-count<=0" --timeout 14400 --interval 60 --json
```

## 7. Sort by score

The `.vision.json` next to each image carries `OVR_Quality_Score` and
`REL_Quality_Score` (0-100). Rank however you like:

```bash
cull stats --job <slug> --json
```

Random-sample the top bucket:

```bash
cull gallery sample --job <slug> --category Keep --n 20 --json \
    | jq '.samples | sort_by(-.ovr) | .[:5]'
```

## Success criteria

- `queue-count` reaches 0 before the timeout (exit 0 from watch).
- `cull stats --json` shows a full OVR/REL score distribution — no
  `unknown` bucket dominating.
- `data/sorted/<slug>/` contains one folder per category plus the
  terminal `DISCARD` / `CORRUPT` folders.
