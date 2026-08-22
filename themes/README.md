# Themes

Colour schemes for the cull dashboard. Structure mirrors `presets/`.

```
themes/
  builtin/       # shipped with cull — read-only in the dashboard
    *.theme.json
  community/     # contributed via the dashboard "Publish" button
    *.theme.json
  thumbnails/    # optional 256x256 previews
    <slug>.jpg
```

User customs live in `data/themes/<slug>.json` (gitignored, per install).

## Envelope

```json
{
  "name": "beige",
  "font_family": "Inter, ui-sans-serif, system-ui, sans-serif",
  "vars": {
    "--color-bg": "#f5efe4",
    "--color-fg": "#1a1613",
    "--color-accent": "#b8543a",
    ...
  }
}
```

The complete variable list is enforced by `pipeline_code/theme_config.py`
(`THEME_VAR_KEYS`) and consumed by the `:root` block in
`pipeline_code/dashboard_enhanced.py`. Unknown keys are dropped on write;
values containing `<`, `>`, `;`, backticks, `url(`, `javascript:` or
`expression(` are rejected.

## Adding a theme

1. Open the dashboard → Settings → Themes → **New theme** (start from a clone).
2. Tweak colours + font in the editor. Live preview mirrors the real
   dashboard's tokens.
3. Save writes to `data/themes/<slug>.json`.
4. Publish commits `themes/community/<slug>.theme.json` and pushes to origin.
