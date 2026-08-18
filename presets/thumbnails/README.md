# Preset thumbnails

The Preset picker + comparison grid in the dashboard renders one thumbnail per preset. Two sources, checked in order:

1. **Your own drop-in** — `presets/thumbnails/<key>.{gif,png,jpg,jpeg,webp,svg}`
2. **Shipped default SVG** — `presets/thumbnails/_builtin/<key>.svg`

The shipped SVGs are minimal stylised icons (a play triangle for `video_default`, a shopping tag + box for `product_ecommerce`, etc.) so every preset has *something* to show out of the box. If you want real photos or animated GIFs, drop your own file at the top-level path and it wins — no rebuild required.

## Overriding a preset thumbnail

```bash
# Example: use an actual product-photo GIF for the product_ecommerce preset
cp ~/Pictures/my-packshot-example.gif presets/thumbnails/product_ecommerce.gif
```

Then reload the dashboard — the endpoint (`GET /api/presets/thumbnail?key=product_ecommerce`) will now serve your GIF.

### Recommended dimensions

- 16:9 aspect ratio (e.g. 640×360 or 1280×720)
- < 500 KB per file for fast card rendering
- WebP or animated GIF is ideal; PNG/JPG work too
- SVG works and is theme-friendly if you want vector

### Where to find rights-clean example imagery

Public-domain / CC0 sources (verify licence per image before shipping):

- [Unsplash](https://unsplash.com) — search e.g. "product photography", "aerial drone", "underwater". Free for commercial use; attribution appreciated.
- [Pexels](https://pexels.com) — free stock photo library, similar terms.
- [Pixabay](https://pixabay.com) — includes both images and looping GIFs / short MP4s.
- [Wikimedia Commons](https://commons.wikimedia.org) — filter by CC0.

If you settle on a set that would benefit other cull users, open a PR against the repo and drop them under `presets/thumbnails/` — the endpoint's precedence order automatically picks them over the shipped SVGs.

## The shipped SVG defaults (offline)

Fully self-contained (no external CDN references, no fonts) so they render anywhere. Small (~1–4 KB each). Distinctive per preset:

| Preset key           | Visual concept                              |
|----------------------|---------------------------------------------|
| `default`            | Generic 3-thumbnail grid                    |
| `photoreal_portrait` | Stylised portrait silhouette                |
| `aerial_drone`       | Mountain + sun horizon                      |
| `underwater_marine`  | Waves + fish silhouette                     |
| `wildlife_macro`     | Leaf + butterfly                            |
| `product_ecommerce`  | Product box + shopping tag                  |
| `anime_illustration` | Speech bubble + sparkles                    |
| `quality_only`       | Checkmark on gradient                       |
| `video_default`      | Play triangle                               |
| `video_cinematic`    | Clapperboard                                |
| `video_anime`        | Stylised motion lines                       |
| `video_product`      | Rotating box outline                        |
| `video_nature`       | Sun over mountains + reel                   |

Delete or replace any file under `_builtin/` at your own risk — the endpoint will fall through to a hashed-gradient placeholder for unknown keys, which is generic. Prefer overriding via the top-level drop-in.
