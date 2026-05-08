"""Single source of truth for the classification taxonomy.

Every vision worker, the supervisor, the JSON schema, and the dashboard
import from here. Adding/renaming a category is a one-line change.

Two distinctions:
  CATEGORIES        - keep-bucket folders that vision workers create at startup.
  ALL_CATEGORIES    - includes terminal buckets (DISCARD, CORRUPT) that the
                      supervisor + storage layer also need to know about.
"""
from __future__ import annotations

# Categories where classified images land. Vision workers pre-create these
# folders under <sorted>/<slug>/ so writes don't race against mkdir.
CATEGORIES: tuple[str, ...] = (
    "InstagramInfluencer",
    "NSFW",
    "Professional",
    "Amateur",
    "Unknown",
    "Watermarked",
)

# Terminal buckets - images that won't be in the curated output but still
# need a folder to land in. CORRUPT is written by the supervisor's safety
# net for images that fail to decode; DISCARD by the post-hoc validator.
TERMINAL_CATEGORIES: tuple[str, ...] = ("DISCARD", "CORRUPT")

# Full set the storage layer must support.
ALL_CATEGORIES: tuple[str, ...] = CATEGORIES + TERMINAL_CATEGORIES

# Schema enum value used in vision_prompt.build_response_format() and the
# standalone vision_response_schema.json. Excludes CORRUPT because the
# vision model never returns it (the supervisor writes that bucket).
SCHEMA_CATEGORIES: tuple[str, ...] = CATEGORIES + ("DISCARD",)


def category_pipe_string() -> str:
    """Pipe-separated category list for embedding in the model prompt text.

    Example: "InstagramInfluencer|NSFW|Professional|Amateur|Unknown|Watermarked|DISCARD".
    """
    return "|".join(SCHEMA_CATEGORIES)


__all__ = [
    "CATEGORIES",
    "TERMINAL_CATEGORIES",
    "ALL_CATEGORIES",
    "SCHEMA_CATEGORIES",
    "category_pipe_string",
]
