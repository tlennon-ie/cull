# Civitai Configuration Guide

**Updated:** 2026-04-19 23:00 GMT+1

---

## Quick Summary

Civitai has two separate websites hosting different content:

```
┌─────────────────────────────────────────────────────────┐
│ civitai.com       │ civitai.red                         │
├─────────────────────────────────────────────────────────┤
│ Clean / SFW       │ NSFW / Uncensored                   │
│ Family-friendly   │ Adult content                       │
│ Models & images   │ All models + images                 │
│ Safe for work     │ Restricted access                   │
└─────────────────────────────────────────────────────────┘
```

**Your configuration** can now choose which one to use.

---

## Setup

### Option 1: SFW Content (Default)

```env
# In .env file:
CIVITAI_DOMAIN=civitai.com
CIVITAI_API_KEY=your_api_key_here
```

Scraper will download from `https://search-new.civitai.com/`

### Option 2: NSFW Content (Uncensored)

```env
# In .env file:
CIVITAI_DOMAIN=civitai.red
CIVITAI_API_KEY=your_api_key_here
```

Scraper will download from `https://search-new.civitai.red/`

### Switch Anytime

Just edit `.env` and restart:
```bash
# Change domain in .env
CIVITAI_DOMAIN=civitai.red

# Restart pipeline
python integrated_launcher.py
```

---

## Getting Your API Key

1. Go to civitai.com or civitai.red
2. Log in to your account
3. Account Settings → API Keys
4. Create new key
5. Copy to `.env`:
   ```env
   CIVITAI_API_KEY=your_copied_key
   ```

---

## How It Works

**In the scraper (`scraper_civitai_search.py`):**

```python
# Load from .env
CIVITAI_DOMAIN = os.getenv("CIVITAI_DOMAIN", "civitai.com")

# Build URL dynamically
SEARCH_URL = f"https://search-new.{CIVITAI_DOMAIN}/multi-search?..."

# Rest works the same for both domains
```

**Search URLs:**
```
civitai.com   → https://search-new.civitai.com/multi-search
civitai.red   → https://search-new.civitai.red/multi-search
```

---

## Your Recommendation

For scraping "Realistic Female Influencer" content:

```env
# Use NSFW domain for all models including uncensored
CIVITAI_DOMAIN=civitai.red
CIVITAI_API_KEY=your_key
```

This gives you:
- ✅ All models (including NSFW)
- ✅ Better selection for your use case
- ✅ No filtering by content type

---

## Verification

To test Civitai scraper:

```bash
cd prompt-library

# Test SFW domain
CIVITAI_DOMAIN=civitai.com python scraper_civitai_search.py

# Or test NSFW domain
CIVITAI_DOMAIN=civitai.red python scraper_civitai_search.py
```

Look for log output:
```
[Civitai Scraper] Using domain: civitai.red | Search: https://search-new.civitai.red/...
```

---

## Current Configuration

In your `.env`:

```
CIVITAI_DOMAIN=civitai.com
CIVITAI_API_KEY=your_civitai_api_key_here
```

**What to do:**
1. Add your API key from civitai.com
2. Change domain to `civitai.red` if you want NSFW
3. Save and restart pipeline

---

## Both Domains Use Same API Key

If you have an account, your API key works on both:
- civitai.com login → use `CIVITAI_DOMAIN=civitai.com`
- civitai.red login → use `CIVITAI_DOMAIN=civitai.red`

You might need separate keys if accounts are different, but typically same key works.

---

## Summary

| Feature | civitai.com | civitai.red |
|---------|-------------|------------|
| Content Type | SFW/Clean | NSFW/Adult |
| API Endpoint | search-new.civitai.com | search-new.civitai.red |
| Same scraper? | ✅ Yes | ✅ Yes |
| Switch easily? | ✅ One .env change | ✅ One .env change |
| API Key | Same | Same |

---

## For Production Use

**Recommendation:** Use `civitai.red` if your use case is adult content scraping, otherwise use `civitai.com`.

Change anytime by editing:
```env
CIVITAI_DOMAIN=civitai.red     # Change this
```

Done! Scraper uses new domain next run.
