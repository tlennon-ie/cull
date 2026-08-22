---
name: cull-operator
description: Use when driving cull (the AI dataset curation engine) as an end-user to accomplish a curation goal — building a LoRA dataset, deduping an archive, or triaging scraped content. Distinct from cull-helper (for editing cull's code). Prefer the `cull` CLI over REST endpoints when both cover the operation.
---

# cull-operator

Load this skill when using cull to accomplish a curation goal. This is the
*operator* skill — you are USING cull, not editing its code. For code-level
changes load [`cull-helper`](../cull-helper/SKILL.md) instead.

The complete operator reference lives at [`AGENTS.md`](../../../AGENTS.md).

## Mental model in one paragraph

A **job** is `{slug, subject, preset, overrides}`. A **preset** ships the
default topic filters, scrapers, categories, and scoring gates; a job carries
only *sparse overrides* on top. Activating a job projects its config into env
vars + `data/cull_categories.json`; the supervisor spawns scrapers → the
queue → vision workers, which sort each image into a category folder next to
a `.txt` caption and `.vision.json` audit record.

## Golden path

1. `cull presets list --json` — pick the closest preset.
2. `cull job create <slug> --preset NAME --subject "..." --json` — create the job.
3. `cull jobs activate <slug>` — mark it active.
4. Tighten before running: `cull scoring set --job SLUG --min-ovr 65
   --require-prompt false`, `cull scrapers add-url --job SLUG --source
   gallery_dl --url https://…`, `cull scrapers toggle --job SLUG --name Web
   --enabled false`.
5. `cull run` in a detached shell.
6. `cull jobs watch --slug SLUG --until "sorted-count>=500" --timeout 7200`.
7. `cull stats --job SLUG --json` — inspect distribution.
8. `cull export kohya --job SLUG --out /path/to/dataset` (or `export hf`).

For four full walk-throughs, load a recipe:

- [`recipes/lora-training-dataset.md`](../../../recipes/lora-training-dataset.md)
- [`recipes/personal-archive-triage.md`](../../../recipes/personal-archive-triage.md)
- [`recipes/brand-ad-curation.md`](../../../recipes/brand-ad-curation.md)
- [`recipes/dedupe-existing-folder.md`](../../../recipes/dedupe-existing-folder.md)

## Interface rules

- Prefer the `cull` CLI over the REST endpoints when both cover the operation.
  The CLI wraps the SAME public APIs the dashboard uses, so state stays
  consistent, and it emits stable machine-parseable JSON via `--json`.
- Every subcommand supports `--json`. **Parse only the JSON payload.** The
  human-readable output is for humans and may change without notice.
- Exit codes are a contract: `0` success, `2` bad args, `3` watch timed out,
  `4` missing job/preset, `5` subprocess failure. Branch on the code, not on
  stderr text.
- `cull config show` masks `api_key` / `token` / `cookies` / `password` /
  `secret` / `authorization` to `***`. That masking is defensive but not
  sufficient on its own — do not log the raw JSON payload of any command that
  might carry a credential.

## Interpreting `.vision.json`

The strict-mode schema guarantees these fields are always present. Load the
`metadata-schema` skill for the full definition, but the load-bearing ones are:

- `category` — the assigned bucket (a Keep bucket, `DISCARD`, or `CORRUPT`).
- `OVR_Quality_Score` (0-100) — overall quality (composition, sharpness,
  lighting).
- `REL_Quality_Score` (0-100) — relevance to the subject.
- `nsfw` (bool) — the model flagged adult content.
- `watermark` (bool) — visible watermark / studio logo detected.

Score gates in `cull scoring set` filter on OVR/REL. `--min-ovr 70` drops any
image the model rated below 70 overall; `--min-rel 60` drops anything
insufficiently on-topic.

## Deciding when to stop

Curation is open-ended — pick a signal in advance:

- **Target count** — `cull jobs watch --slug S --until "sorted-count>=500"`.
- **Queue drain** — `cull jobs watch --slug S --until "queue-count<=0"`.
- **Time budget** — `cull jobs watch --slug S --until "elapsed>=3600s"`.
- **Kept ratio** — periodically compute
  `counts_by_category["Keep"] / sorted_count` from `cull stats --json`. Below
  0.15 usually means the gate or targets need tightening.

Always combine a target with `--timeout`. Treat exit 3 (timeout) as "loosen a
gate or add sources, then retry", not as a failure of cull.

## Failure modes

| Symptom | Recovery |
|---------|---------|
| Scraper counters flat, log shows `MissingCredentialError` | Set the missing env var and let the supervisor's per-scraper cooldown retry. |
| `sorted_count` grows slowly, Keep near empty | `cull scoring set --job S --min-ovr <-10>`; never drop REL below 40. |
| Disk full (exit 5, `OSError [Errno 28]`) | Export the job then delete `data/queue/<slug>/` **only while the supervisor is stopped**. |
| Vision worker silent for minutes | Check `LMSTUDIO_PRIMARY_URL` (or the cloud provider) is reachable; restart `cull run` — it is idempotent. |

## Never do

- Never edit `.vision.json` by hand. Re-classify by moving the image back
  into `data/queue/<slug>/<source>/`.
- Never delete `data/queue/<slug>/` while `cull run` is executing — the
  atomic `.processing` rename lives there and mid-flight deletes race the
  workers.
- Never override `PIPELINE_SLUG` in the environment. The active slug is
  managed by `cull jobs activate`.
- Never call scrapers or vision workers directly. Everything routes through
  the supervisor.
- Never commit anything under `data/` — it is gitignored and holds
  potentially private curation output.

## Environment variables

Credentials are global (they live in `.env`, never per-job): `GROQ_API_KEY`
(or `GROQ_API_KEYS`), `LMSTUDIO_PRIMARY_URL`, `CIVITAI_API_KEY`,
`TWITTER_COOKIES`, `DISCORD_BOT_TOKEN`, `HF_TOKEN` / `HUGGINGFACE_TOKEN`,
`PIPELINE_BASE_DIR`, `FLASK_HOST`.

Per-job targets (X accounts, subreddits, gallery-dl URLs, category rules,
score gates) live in the job's `overrides` — set them through the CLI, not
the environment.
