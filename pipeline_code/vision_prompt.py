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
# of a clean photograph of a person.
_SCREENSHOT_HINTS = (
    "screenshot", "screen capture", "screen-capture", "phone screen",
    "smartphone", "iphone", "android screen", " ios ", "status bar",
    "battery icon", "wifi icon", "browser window", "browser chrome",
    "address bar", "url bar", "tab bar", "navigation bar",
    "reddit ui", "twitter ui", "instagram ui", " app interface ",
    "u/", "r/", "post title", "upvote", "downvote", "comment thread",
    "messaging app", "chat window", "menu bar", "system tray",
    "screen ", "screen,", "screen.",
)

# Substrings that say "this is a multi-panel comparison / collage", not a
# single photograph.
_COMPOSITE_HINTS = (
    "side-by-side", "side by side", "split image", "comparison",
    "comparison grid", "two panels", "three panels", "four panels",
    "panel ", "multi-panel", "grid of images", "image grid", "collage",
    "diptych", "triptych", "before and after", "before/after",
    "labelled panels", "labeled panels", "annotated image",
    "two versions", "three versions", "four versions",
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


def _contains(text: str, terms: tuple[str, ...]) -> str | None:
    """Return the first matched term (lowercased) or None.

    Each term is matched with WORD BOUNDARIES so "rock" doesn't fire on
    "rocky terrain" and "her" doesn't fire on "weather". Multi-word terms
    that already contain spaces or symbols (e.g. "side-by-side", "u/") are
    matched as substrings since their non-letter characters give them an
    implicit boundary.
    """
    haystack = (text or "").lower()
    for term in terms:
        t = term.lower()
        if not t:
            continue
        # Multi-word / punctuation-bearing terms - keep substring match.
        if any(ch in t for ch in (" ", "-", "/", ".", ",")):
            if t in haystack:
                return t
            continue
        # Single-word terms - require word boundaries on both sides.
        if re.search(rf"\b{re.escape(t)}\b", haystack):
            return t
    return None


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


def build_classification_prompt(cfg: ScoreConfig | None = None) -> str:
    """User-message text every vision worker sends alongside the image.

    Structured to fight three failure modes seen in production:
      1. Cartoon/CG/illustration scoring as 'photorealistic'.
      2. Topic context bleeding into woman_present (e.g. wineglass + topic
         'Female Influencer' -> model invents a woman).
      3. Vague "could be a woman" hallucinations.
    """
    cfg = cfg or ScoreConfig.from_env()
    notes = f"\nAdmin scoring notes (apply to REL only): {cfg.notes}\n" if cfg.notes else ""
    return (
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
        '  "category": "InstagramInfluencer|NSFW|Professional|Amateur|Unknown|DISCARD",\n'
        '  "reason": "One short sentence explaining the call."\n'
        "}\n\n"
        "STRICT JUDGEMENT RULES (no exceptions):\n"
        "- Fill `description` and `primary_subject` FIRST, from the pixels only. "
        "Then make every binary judgement consistent with that description. "
        "Your description and labels MUST agree - if your description says "
        "'boulder smashing a car' you cannot then say woman_present=true.\n"
        "- BEFORE describing the subject, scan the WHOLE frame for: phone or "
        "browser UI elements (status bar, battery icon, app chrome, reddit/"
        "twitter/instagram interface, comment threads, address bars), "
        "side-by-side panels or comparison grids (multiple distinct images "
        "separated by borders or labels), and large text overlays / "
        "watermarks (e.g. 'Flux 9B', '1GIRL GARDENS', subtitle text, brand "
        "logos). If you see any of these, set the matching is_screenshot / "
        "is_composite_grid / contains_text_overlay field to true and "
        "MENTION it in `description`. These images are NEVER valid "
        "influencer photographs and category MUST be DISCARD.\n"
        "- An image is a SCREENSHOT if you can see system chrome around the "
        "actual photo (status bar, battery, time, OS icons, app UI, post "
        "metadata). Do not strip the chrome out of your description.\n"
        "- An image is a COMPOSITE/GRID if it contains 2+ separate images, "
        "even if they all show similar subjects. Hint: visible vertical or "
        "horizontal seams, repeated near-identical figures, distinct titles "
        "above each panel, or labels like 'Model A vs Model B'.\n"
        "- art_medium = `photograph` ONLY for an actual camera-captured (or "
        "AI-generated *photoreal*) image of a real-looking subject. Anime, "
        "cel-shaded art, hand-drawn illustration, oil/watercolour painting, "
        "comic, manga, stylised 3D, concept art -> the matching label, "
        "NEVER `photograph`. SIGNALS that mean NOT a photograph: cel-shaded "
        "skin, oversized eyes, hand-drawn outlines, flat colour fills, "
        "exaggerated saturated hair, painterly brush textures, 2D-render "
        "look, anime/manga aesthetic. If you see ANY of these, art_medium "
        "is one of {anime, illustration, digital_painting} and "
        "photorealistic_style MUST be false.\n"
        "- A photograph of a STATUE, BUST, MANNEQUIN, DOLL, or SCULPTURE is "
        "NOT a photograph of a real human. is_human_photograph is FALSE in "
        "those cases, and woman_present is FALSE even if the sculpture "
        "depicts a female form.\n"
        "- photorealistic_style = TRUE ONLY when art_medium = `photograph` "
        "AND the subject reads as a real-world scene/person. Visible brush "
        "strokes, line art, cel shading, painterly textures, exaggerated "
        "proportions, or stylised lighting all force FALSE.\n"
        "- woman_present = TRUE ONLY if a real human female face or body is "
        "clearly visible as the primary OR co-primary subject of the image. "
        "Background pedestrians do NOT count. Sculpted/illustrated/animated "
        "figures do NOT count. DO NOT INFER a woman from the topic, caption, "
        "prompt, or context.\n"
        "- has_ai_flaws = TRUE only for SEVERE artefacts (malformed face, "
        "wrong finger count, melted limbs, warped text).\n"
        "- nsfw = TRUE only for explicit nudity or sexual content.\n\n"
        "OVR_Quality_Score rubric - ten criteria, weighted equally, average them:\n"
        + "\n".join(f"  {c}" for c in _OVR_CRITERIA) + "\n\n"
        "REL_Quality_Score: how close the image sits to the *platonic ideal* "
        "of the curation topic. If photorealistic_style or woman_present is "
        "false, REL must be <= 30. Reserve 90+ for images indistinguishable "
        "from a top-tier real creator's portfolio in that genre.\n\n"
        f"{_RUBRIC_ANCHOR}\n"
        f"{notes}\n"
        "CATEGORY ASSIGNMENT (apply in order, first match wins):\n"
        "- DISCARD: photorealistic_style=false OR woman_present=false OR "
        "art_medium != 'photograph' OR has_ai_flaws=true with quality<=4.\n"
        "- NSFW: photorealistic woman AND explicit nudity/sexual content.\n"
        "- InstagramInfluencer: photorealistic woman, social-media aesthetic, no nudity.\n"
        "- Professional: photorealistic woman, studio/editorial polish, no nudity.\n"
        "- Amateur: photorealistic woman, casual/selfie style.\n"
        "- Unknown: photorealistic woman but doesn't fit categories above."
    )


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

    # ─── Container guards: screenshots, composite grids, text overlays ─
    # The model can describe an embedded face inside a screenshot or one panel
    # of a side-by-side and miss the surrounding chrome. We catch both via:
    #   1) the model's own boolean flags (is_screenshot / is_composite_grid /
    #      contains_text_overlay), and
    #   2) keyword detection in the description, in case the flags weren't set
    #      but the words leaked through anyway.
    if result.get("is_screenshot"):
        reasons.append("is_screenshot=true")
    if result.get("is_composite_grid"):
        reasons.append("is_composite_grid=true")
    if result.get("contains_text_overlay"):
        reasons.append("contains_text_overlay=true")

    screenshot_token = _contains(combined, _SCREENSHOT_HINTS)
    if screenshot_token:
        result["is_screenshot"] = True
        reasons.append(f"description mentions screenshot/UI ({screenshot_token.strip()!r})")
    grid_token = _contains(combined, _COMPOSITE_HINTS)
    if grid_token:
        result["is_composite_grid"] = True
        reasons.append(f"description mentions composite/grid ({grid_token.strip()!r})")
    overlay_token = _contains(combined, _TEXT_OVERLAY_HINTS)
    if overlay_token:
        result["contains_text_overlay"] = True
        reasons.append(f"description mentions overlay ({overlay_token.strip()!r})")

    if (result.get("is_screenshot") or result.get("is_composite_grid")
            or result.get("contains_text_overlay")):
        # Force the photoreal/woman gates closed so the DISCARD path below
        # fires consistently regardless of what the model returned.
        result["photorealistic_style"] = False
        result["woman_present"] = False

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

    if not photoreal or not has_woman or not is_human_photo:
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
    elif result.get("nsfw"):
        result["category"] = "NSFW"

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


__all__ = [
    "ScoreConfig",
    "build_classification_prompt",
    "apply_scores",
    "_safe_parse_vision_json",
    "extract_message_text",
]
