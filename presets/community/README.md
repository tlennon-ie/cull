# Community preset gallery

This folder is cull's **community preset gallery** — a curated set of clean,
shareable preset files you can import into your own install, and the place you
open a pull request to share one of your own.

A *preset* is the inheritable config bundle a job builds on
(`effective_config(job) = preset ⊕ overrides`). Each file here is a portable,
versioned envelope produced by the same export path the dashboard uses
(`config_io.export_preset`), so importing one is loss-free.

## What's shipped

| File | What it curates |
|---|---|
| `film_photography.preset.json` | Analog / film-emulation LoRA sets — rewards organic grain, halation and period colour; routes sterile digital captures off-theme. |
| `clean_anime.preset.json` | Clean anime / illustration datasets — drops the real-human gates, penalises jpeg blocking and visible signatures, routes photoreal images off-theme. |
| `product_ecommerce.preset.json` | Product / e-commerce packshots — rewards seamless backgrounds and even lighting, splits lifestyle shots into their own bucket. |

## Importing a preset

- **Dashboard:** Presets tab → *Import* → choose a `.json` file from this folder.
- **Programmatically:**

  ```python
  import config_io, job_config
  name, cfg = config_io.import_preset(open("film_photography.preset.json", "rb").read())
  merged = job_config._deep_merge(job_config._default_preset_cfg(), cfg)
  job_config.save_preset(name, merged)
  ```

Import is strict at the boundary: unknown keys, oversized files, out-of-range
values, or an envelope from a *newer* cull are rejected with a clear error.
Older envelope shapes (the legacy `{"version", "name", "cfg"}` form) are migrated
forward automatically.

## The envelope format

```json
{
  "cull_preset_version": 1,
  "kind": "cull.preset",
  "name": "Human Readable Name",
  "preset": {
    "topic_filters":  { "keywords_extra": [], "banned_keywords": [], "generation_hints": [], "min_prompt_length": 0, "require_prompt": true },
    "scrapers":       { "reddit_subreddits": [], "civitai_domains": ["civitai.com", "civitai.red"] },
    "categories":     [ { "name": "Keep", "hint": "…" }, { "name": "Borderline", "hint": "…" }, { "name": "OffTopic", "hint": "…" } ],
    "category_rules": "STRICT JUDGEMENT RULES …",
    "scoring":        { "ovr_min": 50, "rel_min": 25, "notes": "…" },
    "captioning":     { "enabled": false, "style": "sd_prompt", "overwrite": false }
  }
}
```

Shape rules (enforced by `config_io.import_preset`):

- **`name`** — letters, digits, space, `_`, `-`; max 40 chars.
- **`categories`** — non-empty; each `name` starts with a letter, is `[A-Za-z0-9_]`
  (max 32), and is **not** `DISCARD` or `CORRUPT` (those are system-owned). Hints
  are capped at 2000 chars.
- **`category_rules`** — free text, max 8000 chars; injected verbatim into the
  vision prompt.
- **`scoring.ovr_min` / `rel_min`** — integers 0–100.
- **`captioning.style`** — one of `sd_prompt`, `booru_tags`, `natural_language`.
- Any block you omit inherits the default shape on import, so a preset only needs
  to carry the taste-bearing fields.
- **No secrets.** Never include API keys, tokens, cookies, private URLs, or
  personal handles in a `vision.workers` block — leave credentials and endpoints
  out so the file is safe to share publicly.

## Contributing a preset (PR workflow)

1. **Build it in cull.** Create a preset in the dashboard, tune its categories,
   rules and scoring against a real batch of images.
2. **Export it.** Presets tab → *Export* (or `config_io.export_preset_bytes(name)`),
   which writes a versioned envelope.
3. **Scrub it.** Open the file and remove anything personal: drop or blank any
   `vision.workers`, `gallery_dl.cookies_file`/`config_path`, `local_imports`,
   and private `x_accounts`/URLs. Presets here should carry only public,
   verifiable scraper targets.
4. **Name the file** `your_theme.preset.json` (lowercase, `_`-separated) and drop
   it in this folder. Give the envelope a clear human-readable `name`.
5. **Open a pull request** against the repo with:
   - the new `.json` file,
   - a one-line entry added to the *What's shipped* table above,
   - a short PR description: what it curates and what kind of dataset it's for.

CI imports every file in this folder through `config_io.import_preset` (see
`tests/test_config_io.py`), so a malformed or unsafe preset fails the build. Run
it locally before opening the PR:

```bash
pytest tests/test_config_io.py
```

## A note on a separate templates repo

For now the gallery lives **in-repo** so presets version alongside the schema and
CI can validate them on every change. If the collection grows large enough that
PR churn becomes noisy, it can graduate to a dedicated `cull-presets` repository
fetched on demand — the envelope format and `config_io` validators are
deliberately repo-independent, so nothing about a preset file would need to
change in that move.
