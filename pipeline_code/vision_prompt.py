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

import os
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
# `photorealistic_style: true`. These are the cases admins reported being
# misclassified (anime/CG/illustration coming through as InstagramInfluencer).
_NON_PHOTO_MEDIUMS = (
    "anime", "cartoon", "illustration", "drawing", "painting", "sketch",
    "digital_painting", "digital painting", "concept_art", "concept art",
    "3d_render", "3d render", "render", "cgi", "cg", "stylised", "stylized",
    "watercolor", "watercolour", "vector", "comic", "manga", "pixel_art",
    "pixel art", "polaroid_drawing",
)


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
        '  "description": "1-2 sentence literal description of what the pixels show.",\n'
        '  "primary_subject": "the main subject in plain words (e.g. wine glass on table, woman seated, group of monsters)",\n'
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
        "If the primary subject is not a human, all of (is_human_photograph, "
        "photorealistic_style, woman_present) MUST be false. No exceptions.\n"
        "- art_medium = `photograph` ONLY for an actual camera-captured (or "
        "AI-generated *photoreal*) image. Anime, cel-shaded art, hand-drawn "
        "illustration, oil/watercolour painting, comic art, polaroid-drawing, "
        "stylised 3D, concept art -> the matching label, NEVER `photograph`.\n"
        "- photorealistic_style = TRUE ONLY when art_medium = `photograph`. "
        "Visible brush strokes, line art, cel shading, painterly textures, "
        "exaggerated proportions, or stylised lighting all force FALSE.\n"
        "- woman_present = TRUE ONLY if you can clearly see a real human "
        "female face or body in the frame as a primary or co-primary subject. "
        "DO NOT INFER a woman from the topic, caption, prompt, or context. "
        "If the subject is an object, animal, monster, landscape, or a "
        "non-photoreal figure, woman_present is FALSE.\n"
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
    art_medium = (result.get("art_medium") or "").strip().lower()
    if any(token in art_medium for token in _NON_PHOTO_MEDIUMS):
        result["photorealistic_style"] = False
        if "score_reason" not in result:
            result["score_reason"] = f"art_medium={art_medium} is not photograph"
    if not result.get("is_human_photograph", False):
        result["woman_present"] = False

    photoreal = bool(result.get("photorealistic_style"))
    has_woman = bool(result.get("woman_present"))
    is_human_photo = bool(result.get("is_human_photograph"))

    if not photoreal or not has_woman or not is_human_photo:
        result["category"] = "DISCARD"
        if "score_reason" not in result:
            reasons: list[str] = []
            if not photoreal:
                reasons.append("not photorealistic")
            if not has_woman:
                reasons.append("no woman in frame")
            if not is_human_photo:
                reasons.append("not a real human photograph")
            result["score_reason"] = "; ".join(reasons)
    elif result.get("has_ai_flaws") and int(result.get("quality_score", 0) or 0) <= 4:
        result["category"] = "DISCARD"
        result["score_reason"] = "severe AI flaws + low quality_score"
    elif result.get("nsfw"):
        result["category"] = "NSFW"

    # ─── Admin score thresholds ──────────────────────────────────────────
    if cfg.ovr_min > 0 and ovr < cfg.ovr_min:
        result["category"] = "DISCARD"
        result["score_reason"] = f"OVR {ovr} below min {cfg.ovr_min}"
    if cfg.rel_min > 0 and rel < cfg.rel_min:
        result["category"] = "DISCARD"
        result["score_reason"] = f"REL {rel} below min {cfg.rel_min}"

    return result


__all__ = ["ScoreConfig", "build_classification_prompt", "apply_scores"]
