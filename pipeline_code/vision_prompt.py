"""
Shared classification prompt + scoring for every vision worker.

Field schema:
  description           short literal pixel description (chain-of-thought anchor)
  primary_subject       what is actually in the frame (free text)
  is_human_photograph   strict YES/NO: real photo of a real human?
  art_medium            photograph | digital_painting | anime | 3d_render | illustration | mixed | unclear
  photorealistic_style  true/false (strict; cartoon/anime/CG/painting MUST be false)
  has_ai_flaws          true/false (severe artefacts only)
  woman_present         true/false  (a real human female visible in pixels)
  nsfw                  true/false
  OVR_Quality_Score     0-100  craft quality, topic-independent
  REL_Quality_Score     0-100  closeness to admin's PIPELINE_TOPIC ideal
  quality_score         1-10   legacy field, kept for compatibility
  category              InstagramInfluencer|NSFW|Professional|Amateur|Unknown|DISCARD
  reason                one short sentence

The prompt explicitly forbids using the topic to *infer* subject presence; the
topic only informs REL_Quality_Score. apply_scores() then re-validates the
output and forces DISCARD whenever the model's own description contradicts its
labels (the wineglass / cartoon / CG fantasy bugs).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from categories import (
    category_pipe_string,
    get_category_hints,
    get_global_rules,
    get_schema_categories,
)

# Ten-criteria rubric used by competition judges / art directors.
_OVR_CRITERIA = (
    "1. Composition - rule of thirds, leading lines, balance, negative space.",
    "2. Lighting - direction, quality, contrast, dimensionality.",
    "3. Colour - harmony, palette discipline, saturation control, mood.",
    "4. Focus & sharpness - subject clarity, intentional depth of field, micro-detail.",
    "5. Exposure & tonal range - highlight/shadow control, tonal separation.",
    "6. Subject + pose - presence, expression, body language, eye contact.",
    "7. Emotion & storytelling - does the image make the viewer feel something?",
    "8. Technical execution - lack of artefacts, clean rendering, believable anatomy.",
    "9. Styling & production value - wardrobe, hair, set, props, post-production.",
    "10. Originality & impact - memorable, non-generic, hard to reproduce.",
)

_RUBRIC_ANCHOR = (
    "Calibration for BOTH scores (OVR and REL):\n"
    "  0-19    broken, unusable.\n"
    "  20-39   amateur, significant flaws.\n"
    "  40-59   competent but forgettable.\n"
    "  60-74   above average, portfolio-worthy.\n"
    "  75-89   standout work, would win a local competition.\n"
    "  90-95   top 1% - magazine cover / gallery grade.\n"
    "  96-100  once-in-a-generation. Use sparingly."
)

# Substrings whose presence in the model's `art_medium` field forces the
# image off the photorealistic track even if the model also said
# `photorealistic_style: true`.
_NON_PHOTO_MEDIUMS = (
    "anime", "cartoon", "illustration", "drawing", "painting", "sketch",
    "digital_painting", "digital painting", "concept_art", "concept art",
    "3d_render", "3d render", "render", "cgi", "cg", "stylised", "stylized",
    "watercolor", "watercolour", "vector", "comic", "manga", "pixel_art",
    "pixel art", "polaroid_drawing",
)

# If any of these appears in the model's `description` or `primary_subject`,
# it is NOT a photo of a real human even when the model claims otherwise -
# catches statues, busts, mannequins, dolls, etc.
_NON_HUMAN_OBJECT_TERMS = (
    "sculpture", "sculpted", "statue", "bust", "mannequin", "doll",
    "figurine", "puppet", "carved", "monument", "wax figure", "android",
    "robot", "mascot",
)

# Tokens that mark an obviously non-human primary subject.
_NON_HUMAN_SUBJECT_HINTS = (
    "boulder", "asteroid", "rock", "vehicle", " car ", " car,", "sedan",
    "truck", "motorbike", "motorcycle", "building", "skyscraper",
    "landscape", "mountain", "forest", "beach scene", "cityscape",
    "monster", "creature", "demon", "alien", "animal", "wolf", "dog",
    "cat ", "horse", "bird", "fish", "insect", "object", "wineglass",
    "wine glass", "cup", "bottle", "food", "pasta", "burger", "salad",
    "flower (alone)", "fruit", "machine", "engine",
)

# Substrings that betray a screenshot / phone UI / browser chrome instead
# of a clean photograph of a person. Tokens here must indicate the WHOLE
# image is a screen capture - not just that a screen/phone/monitor is
# visible IN the frame (the phone-in-hand and monitor-in-background false
# positives killed legitimate selfies).
_SCREENSHOT_HINTS = (
    "screenshot", "screen capture", "screen-capture", "screencap",
    "screen recording", "screen-recording",
    "browser window", "browser chrome", "address bar", "url bar",
    "tab bar", "navigation bar",
    "reddit ui", "twitter ui", "instagram ui",
    "post title", "upvote button", "downvote button", "comment thread",
    "messaging app interface", "chat window interface",
    "system tray", "taskbar",
)

# Substrings that say "this is a multi-panel comparison / collage", not a
# single photograph.
_COMPOSITE_HINTS = (
    "side-by-side", "side by side", "split image", "split-image",
    "comparison grid", "comparison image", "comparison shot",
    "two panels", "three panels", "four panels", "multi-panel",
    "multiple panels", "panel a", "panel b", "left panel",
    "right panel", "top panel", "bottom panel", "first panel",
    "second panel", "third panel", "fourth panel",
    "grid of images", "image grid", "photo grid", "collage",
    "diptych", "triptych", "before and after", "before/after",
    "labelled panels", "labeled panels", "annotated grid",
    "two versions", "three versions", "four versions",
    "model a vs", "model b vs", "version a vs", "version b vs",
)

# Watermark / text-overlay keywords - branded captions, generation-tool
# annotations, etc.
_TEXT_OVERLAY_HINTS = (
    "watermark", "branded text", "logo overlay", "title text",
    "caption text", "text overlay", "text label", "subtitle",
    "annotation", "graphic overlay", "promotional text",
    "headline text", "infographic", "meme",
)

# Words whose presence in description / primary_subject signals an actual
# female human is the (or a) subject. If NONE of these appear we cannot
# trust woman_present=True even if the model insists.
_FEMININE_TERMS = (
    "woman", "women", "girl", "girls", "female", "lady", "ladies",
    "she ", " she,", " her ", "actress", "model ", "model,", "model.",
    "selfie", "blonde", "brunette", "redhead", "auburn", "instagrammer",
    "influencer", "daughter", "mother", "sister", "wife", "bride",
    "queen", "princess", "her hair", "feminine",
)


# Words that turn a positive mention into a negation. We look for any of
# these within ~60 chars before the matched term, not crossing a sentence
# boundary, so phrases like "no UI elements, side-by-side panels, or text
# overlays visible" don't count as positive matches for "side-by-side" /
# "text overlay".
_NEGATION_TOKENS = (
    "no", "not", "without", "absent", "absence", "none", "no signs of",
    "no visible", "without any", "free of", "lacks", "lacking", "missing",
    "doesn't", "doesnt", "doesnt have", "does not", "do not", "don't",
)


def _term_indices(haystack: str, term: str) -> list[int]:
    """Find every occurrence of `term` in `haystack` (lowercased)."""
    out: list[int] = []
    if not term:
        return out
    if any(ch in term for ch in (" ", "-", "/", ".", ",")):
        # Multi-word / punctuated term -> substring search.
        start = 0
        while True:
            idx = haystack.find(term, start)
            if idx == -1:
                return out
            out.append(idx)
            start = idx + len(term)
    # Word-boundary single-word match.
    for m in re.finditer(rf"\b{re.escape(term)}\b", haystack):
        out.append(m.start())
    return out


def _is_negated_at(haystack: str, idx: int, term_len: int, window: int = 60) -> bool:
    """True if a negation token appears in the ~60 chars immediately before
    `haystack[idx:idx+term_len]`, in the same sentence (no period/semicolon
    separator between negation and term).
    """
    start = max(0, idx - window)
    prefix = haystack[start:idx]
    # Trim any prior sentence so negation in an earlier sentence doesn't bleed.
    for sep in (".", ";"):
        last = prefix.rfind(sep)
        if last != -1:
            prefix = prefix[last + 1:]
    for token in _NEGATION_TOKENS:
        if " " in token or "'" in token:
            if token in prefix:
                return True
        else:
            if re.search(rf"\b{re.escape(token)}\b", prefix):
                return True
    return False


def _contains(text: str, terms: tuple[str, ...]) -> str | None:
    """Return the first POSITIVE (non-negated) match or None.

    Word-boundary matching for single-word terms, substring for multi-word
    terms (their punctuation already provides boundaries). Skips matches
    whose preceding context is a negation phrase ("no X", "without Y",
    "no signs of Z") - that fixes self-contradicting model outputs that
    say things like 'no side-by-side panels visible' yet still got
    flagged because the substring 'side-by-side' was present.
    """
    haystack = (text or "").lower()
    for term in terms:
        t = term.lower()
        if not t:
            continue
        for idx in _term_indices(haystack, t):
            if not _is_negated_at(haystack, idx, len(t)):
                return t
    return None


def _description_negates(text: str, terms: tuple[str, ...]) -> bool:
    """True if any term appears, AND every appearance is in a negation
    context. Used to override a model boolean that contradicts the model's
    own description (e.g. is_composite_grid=true while description says
    'no side-by-side panels visible')."""
    haystack = (text or "").lower()
    saw_any = False
    for term in terms:
        t = term.lower()
        for idx in _term_indices(haystack, t):
            saw_any = True
            if not _is_negated_at(haystack, idx, len(t)):
                return False  # there is a positive mention; not negation-only
    return saw_any


@dataclass(frozen=True)
class ScoreConfig:
    topic: str
    ovr_min: int
    rel_min: int
    notes: str

    @classmethod
    def from_env(cls) -> "ScoreConfig":
        try:
            ovr = max(0, min(100, int(os.environ.get("VISION_OVR_MIN_SCORE", 0))))
        except ValueError:
            ovr = 0
        try:
            rel = max(0, min(100, int(os.environ.get("VISION_REL_MIN_SCORE", 0))))
        except ValueError:
            rel = 0
        return cls(
            topic=os.environ.get("PIPELINE_TOPIC", "").strip() or "(topic not set)",
            ovr_min=ovr,
            rel_min=rel,
            notes=os.environ.get("VISION_SCORE_NOTES", "").strip(),
        )


# ── Auto-captioning ─────────────────────────────────────────────────────────
# When AUTO_CAPTION_ENABLED=true the vision call is asked to ALSO emit a
# `caption` string suitable for downstream LoRA / fine-tune training. Style
# choice is user-configurable so the same pipeline can produce SD prompts
# for Flux LoRAs, Booru tags for anime LoRAs, natural-language captions for
# general image-text models, or motion-oriented captions for video-model
# trainers (LTX-Video, Wan, Hunyuan-Video, Mochi, CogVideoX).
CAPTION_STYLES: tuple[str, ...] = (
    "sd_prompt", "booru_tags", "natural_language", "motion",
)

_CAPTION_STYLE_INSTRUCTIONS: dict[str, str] = {
    "sd_prompt": (
        "Stable Diffusion / Flux style prompt. Comma-separated descriptive "
        "phrases. Lead with the subject, then composition, lighting, lens / "
        "camera cues, style modifiers, and quality tags. Example: "
        "'photo of a young woman with brown hair, soft window light, "
        "shallow depth of field, professional headshot, 35mm lens, "
        "photorealistic, high detail'."
    ),
    "booru_tags": (
        "Booru-style tags. Lowercase, underscored, comma-separated. Cover "
        "subject count, hair, clothing, pose, expression, setting, medium. "
        "Example: '1girl, solo, brown_hair, long_hair, smile, "
        "looking_at_viewer, indoors, soft_lighting, photo_(medium)'."
    ),
    "natural_language": (
        "Natural-language description. 1-3 plain sentences describing what "
        "the image contains, the composition, and the mood. Example: "
        "'A young woman with brown hair smiles softly at the camera in a "
        "warmly-lit indoor space. The shot is a tight headshot with shallow "
        "depth of field, suggesting a 35mm portrait lens.'"
    ),
    "motion": (
        "Motion / video-trainer caption (LTX-Video, Wan, Hunyuan-Video, "
        "Mochi, CogVideoX). One flowing sentence that reads as a directable "
        "shot description. Cover, in this order: (1) CAMERA MOVEMENT - the "
        "camera move (static, slow pan left/right, tilt up/down, dolly in/out, "
        "tracking, crane, handheld, orbit, zoom); (2) SUBJECT ACTION - what "
        "the subject is doing and how it moves through the frame; (3) SHOT "
        "TYPE - the framing / angle (wide establishing shot, medium shot, "
        "close-up, low-angle, aerial); (4) TEMPORAL description - how the "
        "scene evolves over time / throughout the clip (what changes from "
        "start to end), plus pacing, lighting and mood. Even for a still "
        "frame, infer the most plausible motion. Example: 'A slow dolly-in on "
        "a lone runner as she sprints toward camera across a misty bridge at "
        "dawn; the wide establishing shot tightens to a medium shot while the "
        "sun rises over the duration, warm light building throughout.'"
    ),
}


@dataclass(frozen=True)
class CaptionConfig:
    enabled: bool
    style: str  # one of CAPTION_STYLES
    overwrite: bool

    @classmethod
    def from_env(cls) -> "CaptionConfig":
        enabled = os.environ.get("AUTO_CAPTION_ENABLED", "false").strip().lower() == "true"
        style = os.environ.get("AUTO_CAPTION_STYLE", "sd_prompt").strip().lower()
        if style not in CAPTION_STYLES:
            style = "sd_prompt"
        overwrite = os.environ.get("AUTO_CAPTION_OVERWRITE", "false").strip().lower() == "true"
        return cls(enabled=enabled, style=style, overwrite=overwrite)


def caption_style_instruction(style: str) -> str:
    return _CAPTION_STYLE_INSTRUCTIONS.get(style, _CAPTION_STYLE_INSTRUCTIONS["sd_prompt"])


# ── Active-learning few-shot exemplars ───────────────────────────────────────
# job_config.resolve_env projects the active job's accumulated user corrections
# into ACTIVE_LEARNING_EXEMPLARS_JSON ({category: [{image_key, caption?, note},
# ...]}). When present we fold a short few-shot block into the prompt so the
# model is steered by THIS curator's past calls. The var is an internal
# projection (never user-set); when absent/empty the prompt is byte-identical.
_MAX_EXEMPLAR_CATEGORIES = 12
_MAX_EXEMPLARS_PER_CATEGORY = 3


def _active_learning_exemplars_block(raw: str | None) -> str:
    """Render the few-shot block from ACTIVE_LEARNING_EXEMPLARS_JSON.

    Returns "" (so the prompt is unchanged) when the var is absent, blank,
    unparseable, or carries no usable exemplars. Tolerant by design — a malformed
    projection must never corrupt the prompt."""
    if not raw or not raw.strip():
        return ""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    if not isinstance(data, dict) or not data:
        return ""

    lines: list[str] = []
    for category in list(data)[:_MAX_EXEMPLAR_CATEGORIES]:
        items = data.get(category)
        if not isinstance(items, list) or not items:
            continue
        rendered: list[str] = []
        for ex in items[:_MAX_EXEMPLARS_PER_CATEGORY]:
            if not isinstance(ex, dict):
                continue
            note = str(ex.get("note") or "").strip()
            caption = str(ex.get("caption") or "").strip()
            detail = note or str(ex.get("image_key") or "").strip()
            if caption:
                detail = f"{detail} — {caption}" if detail else caption
            if detail:
                rendered.append(f"    - {detail}")
        if rendered:
            lines.append(f"  {category}:")
            lines.extend(rendered)
    if not lines:
        return ""
    return (
        "\nExamples from this curator's past corrections (treat as strong "
        "guidance for borderline calls, NOT as overrides of what the pixels "
        "show):\n" + "\n".join(lines) + "\n"
    )


# ── Prompt / schema build cache ─────────────────────────────────────────────
# ``build_classification_prompt`` and ``build_response_format`` are called on
# EVERY image classification (once per API call) and each rebuild does a bunch
# of string interpolation + a categories module read + env parsing. On a busy
# fleet that's thousands of redundant kilobytes per second. Since the supervisor
# soft-restarts children whenever any input to these functions changes (env
# mtime, categories mtime, jobs index mtime — see run_pipeline STRUCTURAL_ENV_KEYS
# and _current_categories_mtime), the result is process-lifetime-stable. We
# memoise by a lightweight cache key so a mid-process env mutation (from a test
# harness or a future in-process reload) still refreshes without a restart.
_prompt_cache: dict[tuple, str] = {}
_response_format_cache: dict[tuple, dict[str, Any]] = {}


def _prompt_cache_key(cfg: "ScoreConfig", caption_cfg: "CaptionConfig") -> tuple:
    """Bundle every input that changes the built prompt into a hashable tuple."""
    return (
        cfg.topic, cfg.ovr_min, cfg.rel_min, cfg.notes,
        caption_cfg.enabled, caption_cfg.style, caption_cfg.overwrite,
        # categories module state (schema enum + hints + rules)
        tuple(get_schema_categories()),
        get_global_rules(),
        tuple(get_category_hints()),
        # active-learning few-shot exemplars are keyed by their raw env json
        os.environ.get("ACTIVE_LEARNING_EXEMPLARS_JSON", ""),
    )


def invalidate_prompt_cache() -> None:
    """Clear the memoised prompt + response-format builds.

    Called by tests and any in-process reload paths that mutate env / taxonomy
    inputs. Not needed on the supervisor's soft-restart path because a new
    subprocess starts with an empty cache anyway.
    """
    _prompt_cache.clear()
    _response_format_cache.clear()


def build_classification_prompt(
    cfg: ScoreConfig | None = None,
    caption_cfg: CaptionConfig | None = None,
) -> str:
    """User-message text every vision worker sends alongside the image.

    Structured to fight three failure modes seen in production:
      1. Cartoon/CG/illustration scoring as 'photorealistic'.
      2. Topic context bleeding into woman_present (e.g. wineglass + topic
         'Female Influencer' -> model invents a woman).
      3. Vague "could be a woman" hallucinations.

    When ``caption_cfg.enabled`` is true the model is also asked to emit a
    `caption` string in the configured style (SD-prompt / Booru / natural).
    The schema always reserves the field for strict-mode consistency; when
    captioning is off the model is told to return an empty string.

    Memoised: the built string is process-cached by
    ``_prompt_cache_key(cfg, caption_cfg)``; hot loops pay one dict lookup
    per call instead of a full rebuild. See ``invalidate_prompt_cache``.
    """
    cfg = cfg or ScoreConfig.from_env()
    caption_cfg = caption_cfg or CaptionConfig.from_env()

    cache_key = _prompt_cache_key(cfg, caption_cfg)
    cached = _prompt_cache.get(cache_key)
    if cached is not None:
        return cached
    notes = f"\nAdmin scoring notes (apply to REL only): {cfg.notes}\n" if cfg.notes else ""
    if caption_cfg.enabled:
        caption_field_doc = (
            f'  "caption": "auto-generated training caption per the AUTO-CAPTION block ({caption_cfg.style})"'
        )
        caption_block = (
            "\nAUTO-CAPTION REQUIREMENTS (always fill `caption` when this "
            "block is present):\n"
            f"  Style: {caption_cfg.style}\n"
            f"  {caption_style_instruction(caption_cfg.style)}\n"
            "  Describe THIS image objectively for downstream training. Do "
            "NOT mention the curation topic. Do NOT invent details that are "
            "not visible in the pixels. If the image is unsuitable / "
            "DISCARDed, still emit a caption describing what is visible.\n"
        )
    else:
        caption_field_doc = '  "caption": ""'
        caption_block = (
            "\nAUTO-CAPTION: disabled. Return `caption` as an empty string.\n"
        )
    # Per-taxonomy bits: rules + bullet list rebuilt every call so the user can
    # edit them in the dashboard without restarting workers.
    global_rules = get_global_rules().rstrip()
    if global_rules and not global_rules.endswith("\n"):
        global_rules = global_rules + "\n"
    category_bullets = "\n".join(
        f"- {name}: {hint}" for name, hint in get_category_hints()
    )
    if category_bullets:
        category_bullets += "\n"
    # DISCARD is always last-resort; spell it out so the model knows it's an
    # explicit option even though it's not in the user-edited list.
    category_bullets += (
        "- DISCARD: image fails the gates above OR doesn't fit any keep bucket."
    )

    exemplars_block = _active_learning_exemplars_block(
        os.environ.get("ACTIVE_LEARNING_EXEMPLARS_JSON")
    )

    built = (
        "You are a strict, literal image auditor. You will be told the topic "
        "an admin is curating, but the topic ONLY informs REL_Quality_Score. "
        "It MUST NOT influence what you say is in the image. Judge what the "
        "pixels actually show, not what would fit the topic.\n\n"
        f"Curation topic (for REL_Quality_Score only): {cfg.topic!r}\n\n"
        "Return ONLY valid JSON (no markdown, no commentary), in this exact shape:\n"
        "{\n"
        '  "description": "1-2 sentence literal description of WHAT THE WHOLE IMAGE CONTAINS, including any UI / labels / panels / watermarks / borders.",\n'
        '  "primary_subject": "the main subject in plain words (e.g. wine glass on table, woman seated, group of monsters)",\n'
        '  "is_screenshot": true/false,\n'
        '  "is_composite_grid": true/false,\n'
        '  "contains_text_overlay": true/false,\n'
        '  "is_human_photograph": true/false,\n'
        '  "art_medium": "photograph | digital_painting | anime | 3d_render | illustration | mixed | unclear",\n'
        '  "photorealistic_style": true/false,\n'
        '  "has_ai_flaws": true/false,\n'
        '  "woman_present": true/false,\n'
        '  "nsfw": true/false,\n'
        '  "OVR_Quality_Score": 0-100,\n'
        '  "REL_Quality_Score": 0-100,\n'
        '  "quality_score": 1-10,\n'
        f'  "category": "{category_pipe_string()}",\n'
        '  "reason": "One short sentence explaining the call.",\n'
        f'{caption_field_doc}\n'
        "}\n\n"
        f"{global_rules}\n"
        "OVR_Quality_Score rubric - ten criteria, weighted equally, average them:\n"
        + "\n".join(f"  {c}" for c in _OVR_CRITERIA) + "\n\n"
        "REL_Quality_Score: how close the image sits to the *platonic ideal* "
        "of the curation topic. Reserve 90+ for images indistinguishable "
        "from a top-tier real creator's portfolio in that genre.\n\n"
        f"{_RUBRIC_ANCHOR}\n"
        f"{notes}\n"
        "CATEGORY ASSIGNMENT (apply in order, first match wins):\n"
        f"{category_bullets}"
        + caption_block
        + exemplars_block
    )
    _prompt_cache[cache_key] = built
    return built


def apply_scores(result: dict[str, Any], cfg: ScoreConfig | None = None) -> dict[str, Any]:
    """Post-process the model's JSON.

    Beyond clamping scores and applying admin thresholds, this re-validates
    the result against the model's own description-first answers:

      * If `art_medium` is anything other than 'photograph' -> photorealistic_style
        is forced FALSE, even if the model claimed otherwise. This catches the
        cartoon/CG cases where the model self-contradicts.
      * If `is_human_photograph` is False -> woman_present is forced FALSE.
      * Either of those forces category = DISCARD with a `score_reason`.
    """
    cfg = cfg or ScoreConfig.from_env()

    def _clamp(value: Any) -> int:
        try:
            return max(0, min(100, int(value)))
        except (TypeError, ValueError):
            return 0

    ovr = _clamp(result.get("OVR_Quality_Score"))
    rel = _clamp(result.get("REL_Quality_Score"))
    result["OVR_Quality_Score"] = ovr
    result["REL_Quality_Score"] = rel

    # ─── Self-contradiction guards ────────────────────────────────────────
    description = result.get("description") or ""
    primary_subject = result.get("primary_subject") or ""
    combined = f"{description}\n{primary_subject}"

    art_medium = (result.get("art_medium") or "").strip().lower()
    reasons: list[str] = []

    if any(token in art_medium for token in _NON_PHOTO_MEDIUMS):
        result["photorealistic_style"] = False
        reasons.append(f"art_medium={art_medium}")

    # ─── Description -> framing booleans (description is source of truth) ──
    # Models in practice flip is_screenshot / is_composite_grid /
    # contains_text_overlay carelessly while filling out the JSON. We treat
    # the prose `description` as the source of truth because chain-of-thought
    # forces the model to commit to the visible content there before the
    # booleans. Each flag is rebuilt from the description:
    #
    #   POSITIVE mention (negation-aware)  -> flag = True
    #   anything else                       -> flag = False
    #
    # If a model genuinely sees something the description omitted, that's a
    # rare false negative we accept; the false positives we were paying for
    # (lazy true flags discarded clean bikini photos) were the bigger cost.
    screenshot_token = _contains(combined, _SCREENSHOT_HINTS)
    grid_token = _contains(combined, _COMPOSITE_HINTS)
    overlay_token = _contains(combined, _TEXT_OVERLAY_HINTS)

    result["is_screenshot"] = bool(screenshot_token)
    result["is_composite_grid"] = bool(grid_token)
    result["contains_text_overlay"] = bool(overlay_token)

    if screenshot_token:
        reasons.append(f"description mentions screenshot/UI ({screenshot_token.strip()!r})")
    if grid_token:
        reasons.append(f"description mentions composite/grid ({grid_token.strip()!r})")
    if overlay_token:
        reasons.append(f"description mentions overlay ({overlay_token.strip()!r})")

    # Screenshots and composite grids are unrecoverable - the image isn't
    # a clean photograph. Force the gates closed so DISCARD fires below.
    # Text overlays / watermarks are different: the underlying photo may
    # still be a usable shot whose only flaw is a removable watermark, so
    # we route those to "Watermarked" further down rather than destroying
    # them here.
    if result.get("is_screenshot") or result.get("is_composite_grid"):
        result["photorealistic_style"] = False
        result["woman_present"] = False
    else:
        # ─── Symmetric photoreal override ───────────────────────────────
        # If art_medium is explicitly "photograph" and the model flagged
        # is_human_photograph=true, photoreal MUST be true. Models sometimes
        # flip this to false when they see e.g. heavy LoRA artefacts and
        # mistake them for stylisation - we trust their own art_medium call.
        if (art_medium == "photograph"
                and result.get("is_human_photograph")
                and not result.get("photorealistic_style")):
            result["photorealistic_style"] = True

    # The model often calls a sculpture / mannequin / doll a 'photograph of a
    # real human'. The word it uses gives it away.
    # Two flavours of "non-human" check, each with different precedence:
    #
    # 1) _NON_HUMAN_OBJECT_TERMS (statue / sculpted / bust / mannequin / doll
    #    / carved). If ANY of these appear in primary_subject, the image is
    #    a photograph OF an inanimate object, not a real human - even if the
    #    object happens to be female-shaped ("female torso (bust)").
    #
    # 2) _NON_HUMAN_SUBJECT_HINTS (boulder / car / landscape / rock / animal
    #    etc). These often appear in scene descriptions where the woman is
    #    still the actual subject ("woman in bikini against rocky terrain").
    #    So we only treat as non-human when primary_subject DOESN'T also
    #    name a female - i.e. these are background props, not the subject.
    feminine_in_description = _contains(combined, _FEMININE_TERMS)
    feminine_in_primary = _contains(primary_subject, _FEMININE_TERMS)

    statue_in_primary = _contains(primary_subject, _NON_HUMAN_OBJECT_TERMS)
    statue_in_combined = _contains(combined, _NON_HUMAN_OBJECT_TERMS)
    if statue_in_primary or (statue_in_combined and not feminine_in_primary):
        result["is_human_photograph"] = False
        result["woman_present"] = False
        token = statue_in_primary or statue_in_combined
        reasons.append(f"non-human object detected ({token!r})")

    object_token = _contains(combined, _NON_HUMAN_SUBJECT_HINTS)
    if object_token and not feminine_in_primary:
        result["woman_present"] = False
        reasons.append(f"primary subject is non-human ({object_token.strip()!r})")

    # Bidirectional cross-check on woman_present:
    # - If the description has NO feminine cue, force False (model gamed topic).
    # - If primary_subject IS a woman, override the model's lying False so
    #   NSFW images with woman_present mistakenly cleared still route correctly.
    if result.get("woman_present") and not feminine_in_description:
        result["woman_present"] = False
        reasons.append("woman_present=True but no feminine term in description")
    elif (not result.get("woman_present")) and feminine_in_primary:
        result["woman_present"] = True

    if not result.get("is_human_photograph", False):
        # Don't drop woman_present here if the primary_subject explicitly says
        # so - keep the bidirectional override above intact.
        if not feminine_in_primary:
            result["woman_present"] = False

    photoreal = bool(result.get("photorealistic_style"))
    has_woman = bool(result.get("woman_present"))
    is_human_photo = bool(result.get("is_human_photograph"))

    # The portrait-curation default taxonomy demands all three gates; other
    # taxonomies (art-style sorting, quality-only, generic LoRA triage) don't
    # care about woman_present or photoreal. We only enforce the photoreal +
    # woman + real-human gates when the user's active taxonomy includes the
    # portrait-style category names that depend on them.
    active_categories = set(get_schema_categories())
    portrait_mode = (
        "InstagramInfluencer" in active_categories
        or "Professional" in active_categories
        or "Amateur" in active_categories
    )
    nsfw_routing = "NSFW" in active_categories
    watermark_routing = "Watermarked" in active_categories
    unknown_routing = "Unknown" in active_categories

    if portrait_mode and (not photoreal or not has_woman or not is_human_photo):
        result["category"] = "DISCARD"
        if not photoreal:
            reasons.append("not photorealistic")
        if not has_woman:
            reasons.append("no woman in frame")
        if not is_human_photo:
            reasons.append("not a real human photograph")
        # De-duplicate while keeping order so the audit trail is readable.
        seen: set[str] = set()
        ordered: list[str] = []
        for r in reasons:
            if r not in seen:
                ordered.append(r)
                seen.add(r)
        result["score_reason"] = "; ".join(ordered) or "self-contradiction"
    elif result.get("has_ai_flaws") and int(result.get("quality_score", 0) or 0) <= 4:
        result["category"] = "DISCARD"
        result["score_reason"] = "severe AI flaws + low quality_score"
    elif nsfw_routing and result.get("nsfw"):
        result["category"] = "NSFW"
    elif watermark_routing and (result.get("contains_text_overlay") or overlay_token):
        # Photo passes every quality gate but has a watermark / branded text
        # overlay. The model might be wrong (real-world signage misread as a
        # watermark), but in either case the underlying shot is salvageable -
        # park it in Watermarked so the admin can review and potentially
        # remove the overlay later instead of destroying the asset.
        result["category"] = "Watermarked"
        token = overlay_token or "model flagged overlay"
        result["score_reason"] = f"watermark/overlay detected ({token!r})"
    elif result.get("category") == "DISCARD" and unknown_routing and portrait_mode:
        # Rescue: the model said DISCARD off the back of its own (now-corrected)
        # contradictory booleans. All gates pass and it isn't NSFW, but we
        # don't have enough signal to pick between InstagramInfluencer /
        # Professional / Amateur, so park it in Unknown for admin review.
        result["category"] = "Unknown"
        result["score_reason"] = "rescued from DISCARD after boolean self-contradiction"

    # If the model returned a category that isn't in the active taxonomy
    # (e.g. user removed it after the worker spawned), force DISCARD so the
    # finalise step doesn't try to mkdir an unknown bucket.
    final_cat = result.get("category")
    if final_cat not in active_categories and final_cat != "DISCARD":
        result["category"] = "DISCARD"
        result["score_reason"] = (result.get("score_reason") or "") + (
            f"; model picked {final_cat!r} which is not in the active taxonomy"
        )

    # ─── Admin score thresholds (SFW only) ────────────────────────────────
    # Once the basic gates pass (photoreal + woman + real-human-photo), an
    # NSFW image should ALWAYS land in the NSFW folder regardless of OVR/REL.
    # Models tend to penalise NSFW on the REL axis because it scores it as
    # off-topic vs an "Influencer" prompt - that's a model bias, not a
    # quality signal. Threshold-gating SFW only fixes that.
    if result.get("category") != "NSFW":
        if cfg.ovr_min > 0 and ovr < cfg.ovr_min:
            result["category"] = "DISCARD"
            result["score_reason"] = f"OVR {ovr} below min {cfg.ovr_min}"
        if cfg.rel_min > 0 and rel < cfg.rel_min:
            result["category"] = "DISCARD"
            result["score_reason"] = f"REL {rel} below min {cfg.rel_min}"

    return result


_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_DANGLING_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _safe_parse_vision_json(raw: str | None) -> dict[str, Any] | None:
    """Extract a JSON object from a vision worker response.

    Tolerates:
      * empty / whitespace strings (LMStudio overload),
      * markdown-fenced JSON,
      * trailing prose around a JSON blob,
      * <think>...</think> reasoning blocks emitted by Qwen3-thinking-style
        models. We strip both balanced and dangling-open think tags before
        parsing - some thinkers run out of tokens before they emit the
        closing </think>, leaving an unterminated reasoning block.

    Returns the parsed dict, or None if the response can't be salvaged - in
    which case the worker should log the raw content and re-queue the image.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # Drop balanced <think>...</think> blocks first.
    text = _THINK_BLOCK.sub("", text).strip()
    # If a model ran out of tokens mid-think, kill the dangling block too.
    if "<think>" in text and "</think>" not in text:
        text = _DANGLING_THINK_OPEN.sub("", text).strip()
    # Strip markdown fences if any wrapping survived.
    fenced = _FENCED_JSON.search(text)
    if fenced:
        text = fenced.group(1)
    if not text:
        return None
    # Direct parse.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Last-resort: grab the largest {...} substring.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def extract_message_text(message: dict[str, Any] | None) -> str:
    """Pull a usable string out of an OpenAI-compatible message envelope.

    Handles three thinking-model conventions:
      1. content carries the full output (legacy behaviour),
      2. reasoning is in `reasoning_content` and content has the final answer,
      3. content is empty / null and the only thing populated is `reasoning_content`
         (LM Studio's split-mode for Qwen3-VL-thinking; we still try to mine
         a JSON object out of it).
    """
    if not message:
        return ""
    content = message.get("content") or ""
    if content.strip():
        return content
    return message.get("reasoning_content") or ""


def build_response_format() -> dict[str, Any]:
    """OpenAI-compatible `response_format` for LM Studio's structured output.

    Sent alongside the chat-completions request so the model is forced to emit
    a valid JSON object matching our pipeline's schema. Eliminates the
    "<think>...</think> only, no JSON" failure mode when paired with
    high-reasoning models. See:
        https://lmstudio.ai/docs/developer/openai-compat/structured-output

    The same schema lives at pipeline_code/vision_response_schema.json so
    you can paste it into LM Studio's chat panel directly.

    Memoised by the current taxonomy's schema-category tuple so a live edit
    (which the supervisor picks up via categories mtime) is reflected on the
    next worker respawn. Callers must NOT mutate the returned dict — the same
    reference is handed out repeatedly.
    """
    schema_cats = tuple(get_schema_categories())
    cached = _response_format_cache.get(schema_cats)
    if cached is not None:
        return cached
    built = {
        "type": "json_schema",
        "json_schema": {
            "name": "VisionClassification",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "description", "primary_subject",
                    "is_screenshot", "is_composite_grid", "contains_text_overlay",
                    "is_human_photograph", "art_medium",
                    "photorealistic_style", "has_ai_flaws",
                    "woman_present", "nsfw",
                    "OVR_Quality_Score", "REL_Quality_Score", "quality_score",
                    "category", "reason", "caption",
                ],
                # NOTE: OpenAI structured-output strict mode (which LM Studio
                # enforces via a grammar sampler) does NOT support minLength,
                # maxLength, minimum, maximum, pattern, or format keywords.
                # See https://platform.openai.com/docs/guides/structured-outputs
                # Bounds are re-enforced in Python by ``apply_scores`` (clamps
                # 0-100 / 1-10) and by the prompt's rubric, so removing them
                # from the schema is behaviourally neutral for the pipeline
                # and unblocks strict-mode servers.
                "properties": {
                    "description":          {"type": "string"},
                    "primary_subject":      {"type": "string"},
                    "is_screenshot":        {"type": "boolean"},
                    "is_composite_grid":    {"type": "boolean"},
                    "contains_text_overlay": {"type": "boolean"},
                    "is_human_photograph":  {"type": "boolean"},
                    "art_medium": {
                        "type": "string",
                        "enum": ["photograph", "digital_painting", "anime", "3d_render",
                                 "illustration", "mixed", "unclear"],
                    },
                    "photorealistic_style": {"type": "boolean"},
                    "has_ai_flaws":         {"type": "boolean"},
                    "woman_present":        {"type": "boolean"},
                    "nsfw":                 {"type": "boolean"},
                    "OVR_Quality_Score":    {"type": "integer"},
                    "REL_Quality_Score":    {"type": "integer"},
                    "quality_score":        {"type": "integer"},
                    "category": {
                        "type": "string",
                        "enum": list(schema_cats),
                    },
                    "reason":  {"type": "string"},
                    "caption": {"type": "string"},
                },
            },
        },
    }
    _response_format_cache[schema_cats] = built
    return built


__all__ = [
    "ScoreConfig",
    "CaptionConfig",
    "CAPTION_STYLES",
    "caption_style_instruction",
    "build_classification_prompt",
    "apply_scores",
    "build_response_format",
    "invalidate_prompt_cache",
    "_safe_parse_vision_json",
    "extract_message_text",
]
