"""
Shared classification prompt + scoring for every vision worker.

Emits a JSON schema with both:
  OVR_Quality_Score (0-100): absolute craft quality, independent of topic.
  REL_Quality_Score (0-100): how closely the image matches the admin's topic
                             (PIPELINE_TOPIC) at its platonic best.

Admins can raise thresholds via:
  VISION_OVR_MIN_SCORE   images below this are routed to DISCARD.
  VISION_REL_MIN_SCORE   images below this are routed to DISCARD.
  VISION_SCORE_NOTES     free-text appended to the scoring rubric. Use this to
                         nudge the model towards your personal taste.

All workers call `build_messages(...)` and `apply_scores(result)` so the prompt,
field names, and thresholds stay in lock-step.
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

_REL_NOTE = (
    "REL_Quality_Score rates how close this image sits to the *platonic ideal* "
    "of the admin's topic at the highest end. A topic of 'Realistic Female "
    "Influencer' expects the image to read as a photorealistic woman shot in "
    "a social-media-ready aesthetic (professional but believable, not obvious "
    "AI). Reserve 90+ for images indistinguishable from a top-tier real "
    "creator's portfolio. 30 or lower means the subject or genre is off."
)

# Both scores use this calibration so 100 is genuinely rare.
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
    """The user-role instruction text every vision worker sends alongside the image."""
    cfg = cfg or ScoreConfig.from_env()
    notes = f"\nAdmin notes (weight heavily): {cfg.notes}\n" if cfg.notes else ""
    return (
        "You are an uncensored image classifier and quality judge. Analyse the "
        "image and return ONLY valid JSON (no markdown, no commentary) in this "
        "exact shape:\n"
        "{\n"
        '  "photorealistic_style": true/false,\n'
        '  "has_ai_flaws": true/false,\n'
        '  "woman_present": true/false,\n'
        '  "nsfw": true/false,\n'
        '  "OVR_Quality_Score": 0-100,\n'
        '  "REL_Quality_Score": 0-100,\n'
        '  "quality_score": 1-10,\n'   # legacy field, kept for compatibility
        '  "category": "InstagramInfluencer|NSFW|Professional|Amateur|Unknown|DISCARD",\n'
        '  "reason": "One short sentence explaining the call."\n'
        "}\n\n"
        f"Topic the admin is curating: {cfg.topic!r}. "
        f"REL_Quality_Score must reflect fit to this topic.\n\n"
        f"OVR_Quality_Score rubric - ten criteria, weighted equally, average them:\n"
        + "\n".join(f"  {c}" for c in _OVR_CRITERIA) + "\n\n"
        f"{_REL_NOTE}\n\n"
        f"{_RUBRIC_ANCHOR}\n"
        f"{notes}\n"
        "CLASSIFICATION RULES:\n"
        "- photorealistic_style: TRUE only if the image reads as a real photo. "
        "FALSE for anime, cartoon, illustration, painting, stylised 3D render.\n"
        "- has_ai_flaws: TRUE only if SEVERE artefacts are obvious (malformed "
        "face, wrong finger count, melted body parts, warped text).\n"
        "- woman_present: TRUE if a human female is the primary subject.\n"
        "- nsfw: TRUE if explicit nudity or sexual content.\n\n"
        "CATEGORY ASSIGNMENT:\n"
        "- DISCARD: not photorealistic OR no woman OR severe AI flaws.\n"
        "- NSFW: photorealistic woman AND explicit nudity/sexual content.\n"
        "- InstagramInfluencer: photorealistic woman, social-media aesthetic, no nudity.\n"
        "- Professional: photorealistic woman, studio/editorial polish, no nudity.\n"
        "- Amateur: photorealistic woman, casual/selfie style.\n"
        "- Unknown: photorealistic woman but doesn't fit the categories above."
    )


def apply_scores(result: dict[str, Any], cfg: ScoreConfig | None = None) -> dict[str, Any]:
    """Post-process the model's JSON: normalise scores, enforce thresholds, set final category.

    Returns the (possibly mutated) `result` dict. Mutations:
      * `OVR_Quality_Score`/`REL_Quality_Score` clamped to 0-100 ints.
      * `category` forced to DISCARD when either score is below the admin's min.
      * `score_reason` appended when a threshold forces a DISCARD.
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

    # Category overrides from existing fields (these match the pre-scoring behaviour).
    if not result.get("photorealistic_style", False) or not result.get("woman_present", False):
        result["category"] = "DISCARD"
    elif result.get("has_ai_flaws") and int(result.get("quality_score", 0) or 0) <= 3:
        result["category"] = "DISCARD"
    elif result.get("nsfw"):
        result["category"] = "NSFW"

    # Then apply admin-defined thresholds on top.
    if cfg.ovr_min > 0 and ovr < cfg.ovr_min:
        result["category"] = "DISCARD"
        result["score_reason"] = f"OVR {ovr} below min {cfg.ovr_min}"
    if cfg.rel_min > 0 and rel < cfg.rel_min:
        result["category"] = "DISCARD"
        result["score_reason"] = f"REL {rel} below min {cfg.rel_min}"

    return result


__all__ = ["ScoreConfig", "build_classification_prompt", "apply_scores"]
