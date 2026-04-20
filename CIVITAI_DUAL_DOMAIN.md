# Civitai Dual-Domain Support

**Added:** 2026-04-19 23:00 GMT+1

---

## Why Two Domains?

Civitai hosts content on two separate domains:

| Domain | Content | Use Case |
|--------|---------|----------|
| **civitai.com** | Clean, SFW models + images | Family-friendly pipelines |
| **civitai.red** | NSFW, uncensored models + images | Adult content pipelines |

Both use the same API structure, just different URLs.

---

## Configuration

### In `.env` file:

```env
# Choose one:
CIVITAI_DOMAIN=civitai.com      # SFW content
# OR
CIVITAI_DOMAIN=civitai.red      # NSFW content

# Your API key (same for both domains)
CIVITAI_API_KEY=your_key_here
```

### Default

- **Default domain:** `civitai.com` (clean content)
- **To use NSFW:** Change to `civitai.red`

---

## How It Works

**In the scraper (`scraper_civitai_search.py`):**

```python
# Load from .env
CIVITAI_DOMAIN = os.getenv("CIVITAI_DOMAIN", "civitai.com")

# Build search URL dynamically
SEARCH_URL = f"https://search-new.{CIVITAI_DOMAIN}/multi-search?..."

# Rest of the code works the same for both domains
```

### Search URLs Generated

```
civitai.com   → https://search-new.civitai.com/multi-search
civitai.red   → https://search-new.civitai.red/multi-search
```

### CDN URLs Generated

Both domains use the same CDN:
```
https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA
```

---

## For Your Use Case

Since you're scraping "Realistic Female Influencer" with NSFW acceptance:

```env
# Recommended configuration:
CIVITAI_DOMAIN=civitai.red      # Get all models including NSFW
CIVITAI_API_KEY=your_api_key
```

This will:
- Access all models on civitai.red
- Include uncensored content
- Same filtering as SFW version (portrait orientation, quality, etc.)

---

## Switching Between Domains

You can change at any time:

```bash
# Edit .env
CIVITAI_DOMAIN=civitai.red

# Restart pipeline
python integrated_launcher.py
```

Scraper will automatically use the new domain next time it runs.

---

## Implementation Details

**File Updated:**
- `scraper_civitai_search.py` — Now reads `CIVITAI_DOMAIN` from .env

**Removed:**
- Hardcoded API token (was: `8c46eb2508...`)
- Hardcoded domain (was: `civitai.com`)

**Configuration Files:**
- `.env` — Added `CIVITAI_DOMAIN` field
- `.env.template` — Added documentation

---

## Verification

To verify Civitai scraper works:

```bash
cd I:\AI\openclaw\workspace\prompt-library

# Test with SFW domain
CIVITAI_DOMAIN=civitai.com python scraper_civitai_search.py

# Or test with NSFW domain
CIVITAI_DOMAIN=civitai.red python scraper_civitai_search.py
```

Look for logs:
```
[Civitai Scraper] Using domain: civitai.com | Search: https://search-new.civitai.com/...
```

---

## Current Configuration

In your `.env`:

```env
CIVITAI_DOMAIN=civitai.com
CIVITAI_API_KEY=your_civitai_api_key_here
```

**To enable NSFW:** Change first line to `CIVITAI_DOMAIN=civitai.red`

---

## Summary

✅ **What changed:**
- Civitai scraper now supports both civitai.com and civitai.red
- Domain controlled via `.env` variable
- No hardcoded URLs or tokens

✅ **How to use:**
- Set `CIVITAI_DOMAIN=civitai.com` for SFW
- Set `CIVITAI_DOMAIN=civitai.red` for NSFW
- Restart pipeline if you change it

✅ **Status:**
- Production-ready
- Backward compatible
- No pipeline changes needed
