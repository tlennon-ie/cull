# Metadata Schema Skill

**Purpose:** Canonical schema for `.json` sidecar files that pair with each image.

Every image in the pipeline has a matching `.json` metadata file with standardized structure.

---

## File Triple

Each image requires:
1. **Image file** — `.jpg`, `.png`, or `.webp` (actual image)
2. **Metadata file** — `.json` (sidecar with image info & assessment results)
3. **Text file** — `.txt` (human-readable summary)

Example:
```
civitai_abc123_001.jpg       ← Image
civitai_abc123_001.json      ← Metadata (THIS SKILL DEFINES STRUCTURE)
civitai_abc123_001.txt       ← Human-readable text
```

---

## Canonical JSON Schema

```json
{
  "version": "1.0",
  "source": "civitai",
  "source_url": "https://civitai.com/api/download/...",
  "source_id": "abc123",
  "timestamp_scraped": "2026-04-19T18:00:00Z",
  
  "image": {
    "filename": "civitai_abc123_001.jpg",
    "filesize_bytes": 1234567,
    "dimensions": {
      "width": 1920,
      "height": 1080
    },
    "format": "jpg",
    "hash_sha256": "abc123def456..."
  },
  
  "classification": {
    "category": "Professional",
    "confidence": 0.92,
    "model": "qwen-vl",
    "timestamp_assessed": "2026-04-19T18:05:00Z"
  },
  
  "description": {
    "summary": "Portrait photo of woman in studio setting",
    "tags": ["portrait", "studio", "professional", "influencer"],
    "quality_score": 8.5,
    "prompt_generated": "A high-quality professional portrait of a woman in studio lighting with soft bokeh background"
  },
  
  "metadata": {
    "original_title": "Influencer Portfolio Shot",
    "original_description": "New portfolio photo from photoshoot",
    "likes": 1234,
    "comments": 56,
    "platform_rating": 4.8
  },
  
  "history": {
    "queue_entry_date": "2026-04-19T17:55:00Z",
    "queue_exit_date": "2026-04-19T18:05:00Z",
    "sorted_folder": "sorted/realistic_female_influencer/Professional/civitai/",
    "processing_time_ms": 600000
  },
  
  "flags": {
    "corrupted": false,
    "duplicate": false,
    "reviewed": false,
    "needs_reprocessing": false
  }
}
```

---

## Minimal Schema (For Scrapers)

When first scraped, metadata can be minimal:

```json
{
  "version": "1.0",
  "source": "twitter_x",
  "source_url": "https://twitter.com/...",
  "timestamp_scraped": "2026-04-19T18:00:00Z",
  "image": {
    "filename": "twitter_x_def789_001.jpg"
  }
}
```

Vision worker **fills in** classification/description later.

---

## Fields Reference

| Field | Type | Scraped | Vision Worker | Optional |
|-------|------|---------|----------------|----------|
| `version` | string | ✅ | ✅ | No |
| `source` | string | ✅ | - | No |
| `source_url` | string | ✅ | - | No |
| `image.filename` | string | ✅ | - | No |
| `image.filesize_bytes` | number | ✅ | - | No |
| `image.dimensions` | object | ✅ | - | Yes |
| `classification.category` | string | - | ✅ | No (after assessment) |
| `classification.confidence` | number | - | ✅ | No |
| `description.summary` | string | - | ✅ | No |
| `description.prompt_generated` | string | - | ✅ | No |
| `flags.corrupted` | boolean | ✅ | ✅ | Yes |

---

## Update Workflow

```
1. SCRAPED (Scraper creates minimal .json)
   {
     "source": "civitai",
     "source_url": "...",
     "timestamp_scraped": "2026-04-19T18:00:00Z",
     "image": {"filename": "..."}
   }

2. QUEUED (File sits in queue/ waiting)
   [no changes to .json]

3. ASSESSED (Vision worker updates .json)
   [adds classification, description, model, timestamp_assessed]

4. SORTED (Sorter finalizes, adds history)
   [adds sorted_folder, processing_time_ms, history details]
   
5. FINAL (In sorted/, ready for use)
   {
     [complete schema with all fields]
   }
```

---

## Python Helper Functions

```python
import json
from pathlib import Path
from datetime import datetime

def create_minimal_metadata(source: str, source_url: str, filename: str) -> dict:
    """Create minimal metadata for newly scraped image"""
    return {
        "version": "1.0",
        "source": source,
        "source_url": source_url,
        "timestamp_scraped": datetime.utcnow().isoformat() + "Z",
        "image": {
            "filename": filename
        }
    }

def update_classification(metadata: dict, category: str, confidence: float, model: str, prompt: str):
    """Update with vision assessment results"""
    metadata["classification"] = {
        "category": category,
        "confidence": confidence,
        "model": model,
        "timestamp_assessed": datetime.utcnow().isoformat() + "Z"
    }
    metadata["description"] = {
        "summary": f"{category} image",
        "prompt_generated": prompt
    }
    return metadata

def save_metadata(metadata: dict, json_path: str):
    """Save metadata to .json file"""
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_metadata(json_path: str) -> dict:
    """Load metadata from .json file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def validate_metadata(metadata: dict) -> tuple[bool, str]:
    """Check if metadata is valid"""
    required = ["version", "source", "image"]
    for field in required:
        if field not in metadata:
            return False, f"Missing required field: {field}"
    return True, "Valid"
```

---

## Historical Log Format

Separate from per-image .json, maintain **historical_log.jsonl** in sorted/:

```
sorted/realistic_female_influencer/historical_log.jsonl

{"timestamp": "2026-04-19T18:05:00Z", "image": "civitai_abc123_001.jpg", "source": "civitai", "classification": "Professional", "folder": "Professional/civitai/", "action": "moved"}
{"timestamp": "2026-04-19T18:10:00Z", "image": "twitter_x_def789_001.jpg", "source": "twitter_x", "classification": "Amateur", "folder": "Amateur/twitter_x/", "action": "moved"}
...
```

This feeds the admin dashboard's "Historical Logs" viewer.

---

**See also:**
- `assets/schema.json` — Full JSON schema for validation
- Dashboard historical logs viewer
- Sorter agent (updates metadata during moves)
