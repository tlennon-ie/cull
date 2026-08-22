---
recipe: dedupe-existing-folder
goal: Pure dedup pass over a local folder; no new scraping, no re-scoring
inputs:
  slug: "URL-safe job name"
  archive_dir: "Absolute path to the local folder to dedup"
outputs:
  - "data/sorted/<slug>/**  (duplicates converge into a single kept sample)"
  - "phash / SQLite index that reveals near-duplicates"
prereqs:
  - "A working vision worker (LMSTUDIO_PRIMARY_URL or GROQ_API_KEY)"
---

# Recipe: dedupe an existing folder

Feed a local folder through cull's queue purely so the dedup layers
(scraper-level `seen_store` + `phash_dedup` on the SQLite index) catch and
collapse duplicates. No new scraping happens — every other source is
disabled.

## 1. Create a triage-shaped job

The `quality_only` preset is deliberately loose so nothing is dropped for
scoring reasons — dedup is about identity, not quality.

```bash
cull job create <slug> --preset quality_only \
    --subject "dedup pass" --json
```

## 2. Disable every scraper

```bash
for s in X.com Discord-1 Civitai-Com Civitai-Red Web Gallery-DL; do
    cull scrapers toggle --job <slug> --name "$s" --enabled false
done
```

## 3. Point the local-import feeder at the folder

`local_imports` is a list — see the personal archive recipe for the exact
override shape. Set `migrate_from = archive_dir` (cull will move dupes
into a `.dedup` sidecar rather than deleting them):

```json
{
  "overrides": {
    "scrapers": {
      "local_imports": [
        {"name": "dedup", "dir": "/absolute/path/to/folder",
         "enabled": true, "migrate_from": "/absolute/path/to/folder"}
      ]
    }
  }
}
```

## 4. Open the gates fully

Dedup must not lose an image because of a score threshold.

```bash
cull scoring set --job <slug> --min-ovr 0 --min-rel 0 --require-prompt false
```

## 5. Activate + run

```bash
cull jobs activate <slug>
cull run &
```

## 6. Wait for the queue to drain

Every image enters the queue exactly once. The scraper-level `seen_store`
skips exact-hash duplicates as they arrive; near-duplicates get caught by
the perceptual-hash pass after they land in `sorted/`.

```bash
cull jobs watch --slug <slug> \
    --until "queue-count<=0" --timeout 14400 --interval 60 --json
```

## 7. Confirm dedup ratio

```bash
cull stats --job <slug> --json | jq '{sorted_count, counts_by_category}'
```

The near-duplicate scan is exposed programmatically by
`pipeline_code/phash_dedup.py`; consult its docstring for a one-shot report
of how many collisions were found — a healthy dedup pass reports a
non-trivial number.

## Success criteria

- `queue-count` reaches 0 within the timeout (exit 0).
- `sorted_count < len(source_folder)` — the delta is the dedup gain.
- No `CORRUPT` bucket growth: an unreadable image was there before cull
  arrived, not caused by the pass.

## Follow-up

Once satisfied, either point downstream tooling at `data/sorted/<slug>/` or
`cull export <slug> --profile folders --out /path/to/deduped` to migrate the
deduplicated set elsewhere.
