"""Shipped starter preset library for the Jobs v2 model.

A *preset* is the inheritable config bundle a Job pulls from before customising
(`effective_config(job) = preset ⊕ overrides`). This module is the single source
of truth for the presets cull ships with, so a fresh install lands on sensible,
general-purpose curation defaults instead of the old influencer-only taxonomy.

Each preset here is intentionally *sparse*: it carries only the taste-bearing
blocks — `topic_filters`, `categories`, `category_rules` (the strict judgement
rules injected into the vision prompt), `scoring`, and `captioning`. Scraper
targets/toggles are left out on purpose so they inherit the default shape
(all scrapers enabled, no hardcoded accounts/URLs); `job_config.get_preset`
deep-merges each preset over `_default_preset_cfg()` to fill those in.

Design notes:
- The default preset (`"default"`) is a general dataset-prep triage
  (Keep / Borderline / OffTopic) with NO subject/woman gates.
- Real-photo themes (aerial, underwater, wildlife, product, quality-only)
  curate image-first, so `require_prompt=False` — they're usually scraped or
  imported photos with no generation prompt. AI-art themes keep `require_prompt`.
- `captioning.enabled` is always False — we never auto-spend caption tokens; the
  per-theme `style` is just the sensible choice for when a user turns it on.
- This module imports nothing from `job_config` (it is imported BY it), so it
  stays free of the SCRAPER_NAMES dependency by omitting the scrapers block.
"""
from __future__ import annotations

import copy

__all__ = ["DEFAULT_PRESET", "PRESET_NAMES", "builtin_library"]

DEFAULT_PRESET = "default"


# ── shared judgement-rule preamble ───────────────────────────────────────────
# Topic-agnostic gates every non-portrait preset starts from. The portrait
# preset keeps its own stricter, woman-specific block (retained verbatim below).
_COMMON_RULES = (
    "STRICT JUDGEMENT RULES (no exceptions):\n"
    "- Fill `description` and `primary_subject` FIRST, from the pixels only. "
    "Then make every label agree with that description.\n"
    "- art_medium = `photograph` ONLY for a camera-captured (or AI photoreal) "
    "image. Anime, illustration, 3D render, digital painting, or a photo of a "
    "statue/figurine -> the matching label, NEVER `photograph`.\n"
    "- has_ai_flaws = TRUE only for SEVERE artefacts (malformed face/hands, "
    "wrong finger count, melted limbs, warped text, impossible geometry).\n"
    "- A SCREENSHOT is the WHOLE frame being a screen capture (UI chrome / "
    "browser bars). A photo that merely contains a phone or monitor as a scene "
    "object is NOT a screenshot.\n"
    "- A COMPOSITE/GRID is 2+ separate images stitched together -> "
    "is_composite_grid=true and DISCARD.\n"
    "- contains_text_overlay = TRUE for burned-in watermarks, captions, logos, "
    "or infographic text.\n"
    "- nsfw = TRUE only for explicit nudity or sexual content.\n"
    "- The curation topic only informs REL_Quality_Score; do NOT invent details "
    "that aren't in the pixels."
)


def _preset(
    *,
    require_prompt: bool,
    categories: list[tuple[str, str]],
    theme_rules: str,
    scoring_notes: str,
    caption_style: str = "sd_prompt",
    keywords_extra: tuple[str, ...] = (),
    banned_keywords: tuple[str, ...] = (),
    generation_hints: tuple[str, ...] = (),
    min_prompt_length: int = 0,
    ovr_min: int = 0,
    rel_min: int = 0,
    rules_preamble: str = _COMMON_RULES,
) -> dict:
    """Build one sparse preset bundle. `categories` is [(name, hint), ...]."""
    rules = f"{rules_preamble}\n{theme_rules}".strip() if theme_rules else rules_preamble
    return {
        "topic_filters": {
            "keywords_extra": list(keywords_extra),
            "banned_keywords": list(banned_keywords),
            "generation_hints": list(generation_hints),
            "min_prompt_length": int(min_prompt_length),
            "require_prompt": bool(require_prompt),
        },
        "categories": [{"name": n, "hint": h} for n, h in categories],
        "category_rules": rules,
        "scoring": {"ovr_min": int(ovr_min), "rel_min": int(rel_min),
                    "notes": scoring_notes},
        "captioning": {"enabled": False, "style": caption_style, "overwrite": False},
    }


# Retained verbatim from the original portrait_curation taxonomy so users who
# DO want influencer/model curation keep an exact, battle-tested preset.
_PORTRAIT_RULES = (
    "STRICT JUDGEMENT RULES (no exceptions):\n"
    "- Fill `description` and `primary_subject` FIRST, from the pixels only. Then "
    "make every binary judgement consistent with that description.\n"
    "- BEFORE describing the subject, scan the WHOLE frame for phone/browser UI, "
    "side-by-side panels or comparison grids, and large text overlays/watermarks. "
    "If present, set the matching is_screenshot / is_composite_grid / "
    "contains_text_overlay field and category MUST be DISCARD (overlays may route "
    "to Watermarked instead).\n"
    "- art_medium = `photograph` ONLY for an actual camera-captured (or AI "
    "photoreal) image of a real-looking subject. Anime, cel-shaded art, "
    "illustration, painting, comic, stylised 3D -> the matching label.\n"
    "- A photograph of a STATUE, MANNEQUIN, DOLL or SCULPTURE is NOT a photo of a "
    "real human: is_human_photograph and woman_present are FALSE.\n"
    "- woman_present = TRUE ONLY if a real human female face or body is clearly the "
    "primary or co-primary subject. Background pedestrians, sculpted/illustrated "
    "figures do NOT count. Do NOT infer a woman from the topic or caption.\n"
    "- has_ai_flaws = TRUE only for SEVERE artefacts. nsfw = TRUE only for explicit "
    "nudity or sexual content."
)


def _build_presets() -> dict[str, dict]:
    return {
        # ── general dataset-prep triage — the new shipped DEFAULT ───────────
        "default": _preset(
            require_prompt=True,
            categories=[
                ("Keep", "Strong on-topic example, no severe AI flaws, no "
                 "watermark/overlay, OVR_Quality_Score >= 60 AND "
                 "REL_Quality_Score >= 60"),
                ("Borderline", "On-topic and usable but average quality, or one "
                 "minor issue (mild AI flaws, soft focus, light compression). "
                 "Bucket for human review"),
                ("OffTopic", "Unrelated to the curation topic regardless of "
                 "quality (REL_Quality_Score < 30)"),
            ],
            theme_rules=(
                "This is a GENERAL dataset-prep triage. Keep the clear wins, send "
                "the maybes to Borderline for human review, and drop the "
                "unrelated. Judge quality on sharpness, exposure, composition and "
                "absence of artefacts — not on subject matter."),
            scoring_notes=(
                "Reward sharp focus, correct exposure, balanced composition and a "
                "clear primary subject. Penalise heavy compression, motion blur "
                "and cluttered/distracting backgrounds unless clearly "
                "intentional."),
            caption_style="sd_prompt",
        ),

        # ── aerial / drone / satellite imagery ──────────────────────────────
        "aerial_drone": _preset(
            require_prompt=False,
            categories=[
                ("Keep", "Clear aerial / drone / satellite shot of the subject, "
                 "sharp, minimal haze or cloud obstruction, OVR>=60 and REL>=60"),
                ("Borderline", "Aerial but soft, hazy, low-detail or only average "
                 "quality -> human review"),
                ("GroundLevel", "A ground-level or eye-level photograph, NOT an "
                 "elevated/overhead/oblique-from-altitude viewpoint -> off-theme "
                 "for an aerial dataset"),
                ("OffTopic", "Not relevant to the aerial subject at all"),
            ],
            theme_rules=(
                "This dataset is AERIAL imagery. An image qualifies as aerial ONLY "
                "if the camera viewpoint is clearly elevated: top-down (nadir), "
                "high-oblique, drone, helicopter, aircraft or satellite. A normal "
                "ground-level or eye-level photo is GroundLevel even if it shows "
                "landscape, fields or buildings. Penalise heavy cloud cover, "
                "atmospheric haze and motion smear; reward crisp ground detail and "
                "even lighting."),
            scoring_notes=(
                "Reward crisp ground detail, even/diffuse lighting, minimal haze "
                "and a clearly elevated viewpoint. Penalise cloud occlusion, "
                "atmospheric haze, heavy compression and motion blur."),
            caption_style="natural_language",
        ),

        # ── underwater / marine ─────────────────────────────────────────────
        "underwater_marine": _preset(
            require_prompt=False,
            categories=[
                ("Keep", "Clearly underwater/marine subject, good visibility, "
                 "sharp, on-topic, natural or well-corrected colour"),
                ("Borderline", "Underwater but turbid, low-visibility, heavy "
                 "backscatter or strong colour cast, or average quality -> "
                 "review"),
                ("AboveWater", "Surface or above-water shot (boat deck, beach, "
                 "splash seen from above) that is NOT beneath the surface -> "
                 "off-theme"),
                ("OffTopic", "Unrelated to the marine subject"),
            ],
            theme_rules=(
                "Underwater imagery only. An image qualifies when the scene is "
                "clearly BENEATH the water surface (diffuse blue/green light, "
                "suspended particles or backscatter, reefs, marine life, divers). "
                "A boat, beach or above-surface splash is AboveWater. Penalise "
                "heavy backscatter, green murk and colour casts that obscure the "
                "subject; reward good visibility, a clear primary subject and "
                "natural or corrected white balance."),
            scoring_notes=(
                "Reward visibility, a clear primary subject and natural/corrected "
                "colour. Penalise backscatter, murk, strong colour cast and "
                "blur."),
            caption_style="natural_language",
        ),

        # ── wildlife & macro nature ─────────────────────────────────────────
        "wildlife_macro": _preset(
            require_prompt=False,
            categories=[
                ("Keep", "Sharp wildlife or macro-nature subject with the eye/key "
                 "plane in focus, natural habitat, clean subject separation, "
                 "on-topic"),
                ("Borderline", "Soft focus, subject small/distant in frame, busy "
                 "background, or average quality -> review"),
                ("CaptiveStaged", "Obvious zoo bars/enclosure, taxidermy mount, "
                 "plush toy or sculpture standing in for a real wild animal -> "
                 "off-theme"),
                ("OffTopic", "Unrelated to the wildlife/nature subject"),
            ],
            theme_rules=(
                "Wildlife & macro-nature dataset. Reward critical sharpness on the "
                "animal's eye (or the macro subject's key plane), natural habitat "
                "and clean separation from the background. Penalise visible "
                "cages/bars, heavy crops that destroy detail and motion blur on "
                "the subject. A taxidermy mount, plush toy or sculpture is NOT a "
                "real animal — real-subject = false and route CaptiveStaged."),
            scoring_notes=(
                "Reward eye-level sharpness on the subject, natural habitat and "
                "clean background separation. Penalise soft focus, "
                "tiny-in-frame subjects, cage bars and motion blur."),
            caption_style="natural_language",
        ),

        # ── product / e-commerce ────────────────────────────────────────────
        "product_ecommerce": _preset(
            require_prompt=False,
            categories=[
                ("Keep", "Single clearly-presented product, clean/seamless "
                 "background, sharp, well-lit, accurate colour, no distracting "
                 "watermark"),
                ("Lifestyle", "Product shown in-context / lifestyle scene — "
                 "useful but a different bucket from clean packshots"),
                ("Borderline", "Product but cluttered background, soft focus, poor "
                 "lighting, or a minor watermark -> review"),
                ("OffTopic", "No clear product, or unrelated to the catalogue "
                 "subject"),
            ],
            theme_rules=(
                "Product / e-commerce dataset. Reward a single clearly-presented "
                "product, even studio lighting, a seamless or clean background and "
                "accurate colour. A product shown within a real-world/lifestyle "
                "scene goes to Lifestyle. Penalise busy/cluttered backgrounds, "
                "distracting reflections, and burned-in promotional text or "
                "watermarks (contains_text_overlay=true -> Borderline at best; "
                "DISCARD if the overlay dominates)."),
            scoring_notes=(
                "Reward clean/seamless backgrounds, even lighting, sharpness and "
                "accurate colour. Penalise clutter, distracting reflections and "
                "burned-in promo text/watermarks."),
            caption_style="natural_language",
        ),

        # ── anime / illustration (drops the photoreal gates) ────────────────
        "anime_illustration": _preset(
            require_prompt=True,
            categories=[
                ("Keep", "Clean anime/illustration artwork, on-topic, no severe AI "
                 "flaws, no watermark/signature overlay"),
                ("Borderline", "Usable but average, minor flaws, jpeg blocking, or "
                 "a visible signature -> review"),
                ("Photoreal", "An actual photograph or photoreal render — NOT "
                 "drawn/painted illustration -> off-theme for an illustration "
                 "dataset"),
                ("OffTopic", "Unrelated to the curation subject"),
            ],
            theme_rules=(
                "Anime / illustration dataset. art_medium MUST be a drawn or "
                "painted style (anime, manga, cel-shaded, digital_painting, "
                "illustration, line art). A camera photograph or photoreal-AI "
                "image is Photoreal (off-theme). Do NOT apply real-human gates "
                "here. has_ai_flaws = TRUE only for severe artefacts (broken "
                "anatomy, extra fingers, melted detail, warped text). Penalise "
                "heavy jpeg blocking and visible artist signatures/watermarks "
                "(route Borderline)."),
            scoring_notes=(
                "Reward clean line work / rendering, coherent anatomy and absence "
                "of compression artefacts. Penalise jpeg blocking, severe AI "
                "anatomy flaws and watermark/signature overlays."),
            caption_style="booru_tags",
        ),

        # ── retained: photoreal portrait / influencer taxonomy (NOT default) ─
        "photoreal_portrait": _preset(
            require_prompt=True,
            rules_preamble=_PORTRAIT_RULES,
            theme_rules="",
            categories=[
                ("InstagramInfluencer", "photorealistic person, social-media "
                 "aesthetic, no nudity, no overlay"),
                ("NSFW", "photorealistic person AND explicit nudity/sexual "
                 "content"),
                ("Professional", "photorealistic person, studio/editorial polish, "
                 "no nudity, no overlay"),
                ("Amateur", "photorealistic person, casual/selfie style, no "
                 "overlay"),
                ("Unknown", "photorealistic person but doesn't fit the categories "
                 "above"),
                ("Watermarked", "all other gates pass BUT "
                 "contains_text_overlay=true (visible watermark, branded caption, "
                 "logo). Salvageable if the overlay is removed -> route here "
                 "instead of DISCARD"),
            ],
            scoring_notes=(
                "Reward photoreal skin/lighting and a clear single subject. "
                "Penalise AI flaws, overlays and composite grids."),
            caption_style="sd_prompt",
        ),

        # ── retained: topic-agnostic quality triage ─────────────────────────
        "quality_only": _preset(
            require_prompt=False,
            categories=[
                ("Top", "OVR_Quality_Score >= 80, no severe AI flaws, no overlay"),
                ("Mid", "OVR_Quality_Score 60-79"),
                ("Low", "OVR_Quality_Score 40-59 OR mild AI flaws"),
            ],
            theme_rules=(
                "Topic-agnostic quality triage. Route purely on OVR_Quality_Score; "
                "the curation topic informs REL_Quality_Score but does not change "
                "the bucket here."),
            scoring_notes=(
                "Score on technical quality only: sharpness, exposure, dynamic "
                "range and absence of artefacts."),
            caption_style="sd_prompt",
        ),
    }


PRESET_NAMES: tuple[str, ...] = tuple(_build_presets().keys())


def builtin_library() -> dict:
    """Return a fresh {default, presets} library (safe for the caller to mutate)."""
    return {"default": DEFAULT_PRESET, "presets": copy.deepcopy(_build_presets())}
