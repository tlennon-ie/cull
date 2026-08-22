---
recipe: brand-ad-curation
goal: Curate commercial-quality product images from mixed sources; quality-first
inputs:
  brand: "Product / brand name and short descriptor"
  slug: "URL-safe job name"
  target_count: 300
outputs:
  - "data/sorted/<slug>/Keep/**"
  - "Folders exported via 'cull export <slug> --profile folders'"
prereqs:
  - "GROQ_API_KEY (or LMSTUDIO_PRIMARY_URL) for the vision worker"
  - "CIVITAI_API_KEY / TWITTER_COOKIES optional but recommended"
---

# Recipe: brand / ad curation

Curate a bank of ~300 commercial-quality images for a product or brand,
biased hard toward quality (composition, lighting, no watermarks) and against
low-effort user-generated content.

## 1. Create a product-shaped job

`product_ecommerce` ships with a taxonomy that separates clean packshots
from lifestyle context shots.

```bash
cull job create <slug> \
    --preset product_ecommerce \
    --subject "<brand or product descriptor>" \
    --json
```

## 2. Wire up sources — commercial-friendly only

Enable structured sources that carry good metadata; disable ones that
mostly return user snapshots.

```bash
# Enable Civitai (has strong prompts) and web scraping.
cull scrapers toggle --job <slug> --name Civitai-Com --enabled true
cull scrapers toggle --job <slug> --name Web --enabled true
# Discord + X.com are noisy for commercial work; leave them off.
cull scrapers toggle --job <slug> --name Discord-1 --enabled false
cull scrapers toggle --job <slug> --name X.com --enabled false

# Add gallery-dl URLs for the specific brand galleries you care about.
cull scrapers add-url --job <slug> --source gallery_dl \
    --url https://example.com/brand-portfolio
```

## 3. Tighten scoring — quality-first

Aim high on OVR (image quality), keep REL moderate so lifestyle shots that
tangentially reference the brand still land. Require prompts — commercial
imagery usually carries one, and it's the cheapest watermark filter.

```bash
cull scoring set --job <slug> --min-ovr 80 --min-rel 55 --require-prompt true
```

## 4. Activate + run

```bash
cull jobs activate <slug>
cull run &
```

## 5. Wait for the target — with a shorter timeout

Commercial curation should run tight. If the gate is right, the queue
either fills or you learn quickly that your sources are wrong.

```bash
cull jobs watch --slug <slug> \
    --until "sorted-count>=300" --timeout 3600 --interval 30 --json
```

Exit `3` (timeout) → check `cull stats --json`:

- If `watermark_count / sorted_count > 0.2`, the source is a low-quality
  archive; swap it for a curated brand URL.
- If `Keep` is small but `Borderline` is big, drop OVR to 75 and retry.
- If everything is `OFF_TOPIC`, your subject phrase is too narrow.

## 6. Reject watermarked or NSFW keepers manually

Random-sample and eyeball a batch:

```bash
cull gallery sample --job <slug> --category Keep --n 20 --json \
    | jq '.samples[] | select(.watermark == true or .nsfw == true) | .path'
```

Move any listed images out of `data/sorted/<slug>/Keep/` before exporting
(and let the indexer re-scan on next tick).

## 7. Export

Use the `folders` profile so category structure is preserved for downstream
review; skip `kohya` unless you're training on the set.

```bash
cull export <slug> --profile folders --out /path/to/brand_curated --json
```

## Success criteria

- Exit 0 from `cull export`.
- Kept ratio `Keep / sorted_count >= 0.4` (a tighter quality gate should
  land a smaller, cleaner set).
- `watermark_count / sorted_count < 0.05` after the manual sweep.
- The export folder contains at least 300 samples under `Keep/`, cleanly
  bucketed by category subfolder.
