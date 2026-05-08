"""Topic-aware content filter shared by every scraper.

Drives what counts as an in-topic post based on `.env`:

    PIPELINE_TOPIC            e.g. "Realistic Female Influencer"
    TOPIC_KEYWORDS_EXTRA      comma-list of MUST-MATCH keywords (at least one
                              must appear in title+body OR prompt). Default:
                              auto-derived from PIPELINE_TOPIC.
    TOPIC_BANNED_KEYWORDS     comma-list of BAN phrases (case-insensitive).
                              Post rejected if any appears.
    TOPIC_GENERATION_HINTS    comma-list of prompt-craft words (e.g.
                              "photorealistic,cfg,lora"). At least one must
                              appear for a prompt to be considered "real".
                              Empty -> this gate is disabled.
    MIN_PROMPT_LENGTH         int, default 40.

Everything loads at import time from the current environment. Scrapers call
`passes(title_body, prompt)` once per candidate post before queueing.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger("topic_filter")

# Words that show up in every topic but carry no signal.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "and", "or", "for", "with", "ai", "realistic",
    "real", "ultra", "generated", "style", "image", "images", "prompt",
    "prompts",
})


def _split_csv(raw: str) -> list[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _auto_keywords(topic: str) -> list[str]:
    """Derive default required keywords from the topic string."""
    tokens = re.findall(r"[A-Za-z][A-Za-z-]+", topic.lower())
    kept = [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]
    # Add a few common synonyms for the known "influencer" family so existing
    # setups keep behaving, but do not impose anything else.
    out = list(dict.fromkeys(kept))
    if any(t in kept for t in ("influencer", "instagram", "instagrammer")):
        out += ["woman", "girl", "female", "model", "portrait", "selfie"]
    if "male" in kept or "man" in kept:
        out += ["man", "male", "guy", "portrait"]
    return out


@dataclass(frozen=True)
class TopicConfig:
    topic: str
    required_keywords: tuple[str, ...]
    banned_keywords: tuple[str, ...]
    generation_hints: tuple[str, ...]
    min_prompt_length: int

    @property
    def required_re(self) -> re.Pattern[str] | None:
        if not self.required_keywords:
            return None
        joined = "|".join(re.escape(k) for k in self.required_keywords)
        return re.compile(rf"\b({joined})\b", re.IGNORECASE)

    @property
    def banned_re(self) -> re.Pattern[str] | None:
        if not self.banned_keywords:
            return None
        joined = "|".join(re.escape(k) for k in self.banned_keywords)
        return re.compile(joined, re.IGNORECASE)

    @property
    def generation_re(self) -> re.Pattern[str] | None:
        if not self.generation_hints:
            return None
        joined = "|".join(re.escape(k) for k in self.generation_hints)
        return re.compile(joined, re.IGNORECASE)


_DEFAULT_BANNED = (
    # Self-promotion spam
    "link in bio", "dm me", "for sale", "patreon", "onlyfans", "only fans",
    "subscribe", "follow me", "commission", "workflow+lora", "workflow + lora",
    "check my", "check out my", "my pack", "my preset", "rate this",
    # Meta posts that never contain real prompts
    "what do you think", "feedback please", "drop your",
)

_DEFAULT_GENERATION_HINTS = (
    "photorealistic", "hyperrealistic", "cinematic", "8k", "4k", "bokeh",
    "depth of field", "dof", "masterpiece", "best quality", "ultra realistic",
    "skin texture", "natural light", "studio light", "golden hour",
    "portrait", "photograph", "camera", "lens", "aperture", "f/",
    "lora", "checkpoint", "negative prompt", "cfg", "sampler", "steps",
)


def load_config() -> TopicConfig:
    topic = os.environ.get("PIPELINE_TOPIC", "").strip()

    extra = _split_csv(os.environ.get("TOPIC_KEYWORDS_EXTRA", ""))
    required = extra or _auto_keywords(topic)

    banned = _split_csv(os.environ.get("TOPIC_BANNED_KEYWORDS", ""))
    if not banned:
        banned = list(_DEFAULT_BANNED)

    hints_raw = os.environ.get("TOPIC_GENERATION_HINTS")
    if hints_raw is None:
        hints = list(_DEFAULT_GENERATION_HINTS)
    else:
        hints = _split_csv(hints_raw)

    try:
        min_len = int(os.environ.get("MIN_PROMPT_LENGTH", 40))
    except ValueError:
        min_len = 40

    return TopicConfig(
        topic=topic,
        required_keywords=tuple(required),
        banned_keywords=tuple(banned),
        generation_hints=tuple(hints),
        min_prompt_length=min_len,
    )


def prompt_optional() -> bool:
    """Whether scrapers may queue images without a usable prompt.

    Driven by `REQUIRE_PROMPT` (default ``"true"``). When set to ``"false"``,
    scrapers ingest images regardless of prompt length / generation hints,
    and the vision worker's auto-caption step (when enabled) fills in a
    `.txt` afterwards.
    """
    return os.environ.get("REQUIRE_PROMPT", "true").strip().lower() == "false"


def passes(
    context: str,
    prompt: str,
    *,
    cfg: TopicConfig | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason). `context` is title+body/source text, `prompt` is the image prompt.

    A post passes when:
      1. `prompt` is at least MIN_PROMPT_LENGTH chars (skipped if REQUIRE_PROMPT=false).
      2. No banned phrase appears in either context or prompt.
      3. At least one required keyword appears in (context + prompt).
      4. At least one generation hint appears in prompt (skipped if REQUIRE_PROMPT=false).
    """
    cfg = cfg or load_config()
    prompt = (prompt or "").strip()
    haystack = f"{context or ''}\n{prompt}"
    require_prompt = not prompt_optional()

    if require_prompt and len(prompt) < cfg.min_prompt_length:
        return False, f"prompt too short ({len(prompt)} < {cfg.min_prompt_length})"

    if cfg.banned_re and cfg.banned_re.search(haystack):
        return False, "banned phrase"

    if cfg.required_re and not cfg.required_re.search(haystack):
        return False, "missing required keyword"

    if require_prompt and cfg.generation_re and not cfg.generation_re.search(prompt):
        return False, "no generation-style keywords in prompt"

    return True, "ok"


def log_rejection(source: str, reason: str, preview: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    clipped = preview[:80].replace("\n", " ")
    logger.debug("[%s] drop (%s): %s", source, reason, clipped)


__all__ = ["TopicConfig", "load_config", "passes", "log_rejection", "prompt_optional"]
