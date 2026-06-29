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
- Scraper targets seeded here are PUBLIC, verifiable handles (subreddits, X
  accounts) and both Civitai hosts. Discord is intentionally NOT pre-seeded:
  the Discord scraper needs numeric server+channel IDs reachable by *your* bot
  token, so a shipped guess would be wrong for everyone — leave it per-job.
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


# Both Civitai hosts are seeded by default — the dashboard exposes them as a
# two-option multiselect (civitai.com / civitai.red), not free text.
_CIVITAI_BOTH: tuple[str, ...] = ("civitai.com", "civitai.red")

# Promo/spam phrases that are never part of a real generation prompt. Mirrors
# (in spirit) topic_filter._DEFAULT_BANNED — when a preset seeds its OWN banned
# list the runtime stops falling back to that default, so every seeded list
# carries these forward and then adds its theme-specific exclusions on top.
_SPAM_BANNED: tuple[str, ...] = (
    "link in bio", "dm me", "onlyfans", "patreon", "subscribe", "follow me",
    "promo code", "for sale", "commission", "check my",
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
    x_accounts: tuple[str, ...] = (),
    reddit_subreddits: tuple[str, ...] = (),
    civitai_domains: tuple[str, ...] = _CIVITAI_BOTH,
    rules_preamble: str = _COMMON_RULES,
) -> dict:
    """Build one sparse preset bundle. `categories` is [(name, hint), ...].

    `keywords_extra` is a generous synonym list: the required-keyword gate is
    always on (when empty it auto-derives from the subject), so a broad list
    only *widens* what a prompt-based scraper accepts. `generation_hints` only
    gate when require_prompt=True. `x_accounts` / `reddit_subreddits` are
    starter scraper targets (verify/extend per job); `civitai_domains` defaults
    to both hosts. Scraper enable/disable + gallery-dl + local imports inherit
    the default shape via get_preset's deep-merge.
    """
    rules = f"{rules_preamble}\n{theme_rules}".strip() if theme_rules else rules_preamble
    return {
        "topic_filters": {
            "keywords_extra": list(keywords_extra),
            "banned_keywords": list(banned_keywords),
            "generation_hints": list(generation_hints),
            "min_prompt_length": int(min_prompt_length),
            "require_prompt": bool(require_prompt),
        },
        "scrapers": {
            "x_accounts": list(x_accounts),
            "reddit_subreddits": list(reddit_subreddits),
            "civitai_domains": list(civitai_domains),
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


# ── shared judgement-rule preamble for VIDEO presets ─────────────────────────
# Video-model trainers (LTX-Video, Wan, Hunyuan-Video, Mochi, CogVideoX) curate
# CLIPS, judged on a sampled frame plus the prompt's motion description. This
# reuses the SAME category vocabulary spine as the image presets (no new global
# categories) but reframes the gates in temporal/motion terms. The cull worker
# still classifies a representative frame, so the per-frame gates (medium,
# composite, overlay, nsfw) carry over verbatim; the additions are motion-aware.
_VIDEO_RULES = (
    "STRICT JUDGEMENT RULES (no exceptions):\n"
    "- This is a VIDEO-CLIP dataset judged from a representative frame plus the "
    "prompt's motion description. Fill `description` and `primary_subject` FIRST "
    "from the pixels, then make every label agree.\n"
    "- art_medium = `photograph` ONLY for live-action (camera-captured or AI "
    "photoreal) footage. Animation, 2D/cel motion, 3D render or motion graphics -> "
    "the matching label, NEVER `photograph`.\n"
    "- has_ai_flaws = TRUE for SEVERE per-frame artefacts (malformed face/hands, "
    "wrong finger count, melted limbs, warped text, impossible geometry). Temporal "
    "breakdown (flicker, morphing, identity drift) also counts as a severe flaw.\n"
    "- A SCREEN-RECORDING is the WHOLE frame being a screen capture (UI chrome, "
    "browser bars, cursor). A clip that merely shows a phone or monitor as a scene "
    "object is NOT a screen-recording.\n"
    "- A COMPOSITE/SPLIT-SCREEN is 2+ separate shots stitched into one frame -> "
    "is_composite_grid=true and DISCARD.\n"
    "- contains_text_overlay = TRUE for burned-in watermarks, captions, lower "
    "thirds, subtitles, logos or end-card text.\n"
    "- nsfw = TRUE only for explicit nudity or sexual content.\n"
    "- The curation topic only informs REL_Quality_Score; do NOT invent motion or "
    "details that aren't visible in the frame or stated in the prompt."
)


def _build_video_presets() -> dict[str, dict]:
    """Video-dataset curation presets (a parallel set to the image presets).

    Each mirrors an image preset's dict shape EXACTLY via the same `_preset`
    factory, so `job_config` projection + the content-signature baseline
    reconcile keep working unchanged. They reuse the shared `_SPAM_BANNED`
    tuple and the SAME category vocabulary (Keep / Borderline / OffTopic plus a
    per-theme off-theme bucket); the terminal DISCARD/CORRUPT stay system-owned.

    `require_prompt=True` everywhere: a video prompt's MOTION description is the
    load-bearing training signal, so a clip with no prompt is dropped. The
    motion `generation_hints` then gate prompts to video-shaped descriptions.
    """
    return {
        # ── general motion-clip triage (parallel to image "default") ─────────
        "video_default": _preset(
            require_prompt=True,
            ovr_min=45, rel_min=20,
            min_prompt_length=20,
            reddit_subreddits=("aivideo", "StableDiffusion", "MediaSynthesis"),
            x_accounts=("runwayml", "LumaLabsAI"),
            keywords_extra=("video", "clip", "motion", "footage", "b-roll",
                            "camera pan", "tracking shot", "slow motion",
                            "moving", "cinematic"),
            banned_keywords=_SPAM_BANNED + ("still image", "photo only", "slideshow"),
            generation_hints=("camera pan", "tracking shot", "slow motion",
                              "b-roll", "establishing shot", "moving camera",
                              "smooth motion"),
            rules_preamble=_VIDEO_RULES,
            categories=[
                ("Keep", "Strong on-topic clip with clear, coherent motion, no "
                 "severe per-frame or temporal flaws, no watermark/overlay, "
                 "OVR_Quality_Score >= 60 AND REL_Quality_Score >= 60"),
                ("Borderline", "On-topic and usable but average — minor motion "
                 "blur, soft focus, light compression, brief flicker, or jerky "
                 "motion. Bucket for human review"),
                ("StaticShot", "A frozen/locked frame with no meaningful motion "
                 "(a still exported as video, a held freeze) -> off-theme for a "
                 "motion dataset"),
                ("OffTopic", "Unrelated to the curation topic regardless of "
                 "quality (REL_Quality_Score < 30)"),
            ],
            theme_rules=(
                "GENERAL video-clip triage. Keep clips with clean, intentional "
                "motion; send borderline motion/quality to review; route a "
                "motionless held frame to StaticShot; drop the unrelated. Judge on "
                "sharpness, exposure, composition, motion smoothness and absence "
                "of temporal artefacts (flicker, warping, identity drift)."),
            scoring_notes=(
                "Reward sharp frames, stable exposure, smooth coherent motion and "
                "temporal consistency across the clip. Penalise heavy compression, "
                "rolling-shutter wobble, stutter, flicker and morphing."),
            caption_style="natural_language",
        ),

        # ── cinematic / film-like camera work (parallel to "photoreal") ──────
        "video_cinematic": _preset(
            require_prompt=True,
            ovr_min=55, rel_min=25,
            min_prompt_length=24,
            reddit_subreddits=("cinematography", "filmmaking", "videography"),
            keywords_extra=("cinematic", "film", "dolly", "tracking shot",
                            "establishing shot", "anamorphic", "depth of field",
                            "color grade", "crane shot", "steadicam"),
            banned_keywords=_SPAM_BANNED + ("webcam", "screen recording", "meme",
                                            "selfie"),
            generation_hints=("cinematic", "dolly shot", "tracking shot",
                              "establishing shot", "crane shot", "anamorphic",
                              "shallow depth of field", "film grain"),
            rules_preamble=_VIDEO_RULES,
            categories=[
                ("Keep", "Film-like clip — deliberate camera move (dolly, crane, "
                 "tracking), cinematic lighting/grade, shallow depth of field, "
                 "stable and sharp, on-topic, no overlay"),
                ("Borderline", "Cinematic intent but average execution — soft "
                 "focus, uneven exposure, slightly unstable rig, or a minor "
                 "lower-third/subtitle -> review"),
                ("Amateur", "Casual handheld/webcam/phone-vertical look with no "
                 "cinematic camera language -> off-theme for a film-look dataset"),
                ("OffTopic", "Unrelated to the cinematic subject"),
            ],
            theme_rules=(
                "Cinematic / film-look dataset. Reward deliberate camera language "
                "(dolly, crane, tracking, push-in), motivated lighting, colour "
                "grading and shallow depth of field with stable, sharp frames. A "
                "casual webcam/selfie/screen-grab look is Amateur. Penalise shaky "
                "handheld, flat phone-grade footage and burned-in lower thirds or "
                "subtitles (contains_text_overlay=true -> Borderline at best; "
                "DISCARD if the overlay dominates)."),
            scoring_notes=(
                "Reward cinematic composition, motivated lighting, smooth motivated "
                "camera moves and grade. Penalise shaky handheld, flat exposure, "
                "rolling shutter and overlay text."),
            caption_style="natural_language",
        ),

        # ── animated / 2D motion (parallel to "anime_illustration") ──────────
        "video_anime": _preset(
            require_prompt=True,
            ovr_min=45, rel_min=25,
            min_prompt_length=20,
            reddit_subreddits=("anime", "animation", "Sakuga"),
            keywords_extra=("anime", "animation", "animated", "2d motion", "amv",
                            "cel animation", "motion", "sakuga", "key animation"),
            banned_keywords=_SPAM_BANNED + ("live action", "photoreal", "webcam"),
            generation_hints=("anime", "animated", "smooth animation",
                              "camera pan", "cel animation", "moving",
                              "key animation", "dynamic motion"),
            rules_preamble=_VIDEO_RULES,
            categories=[
                ("Keep", "Clean animated / 2D-motion clip, on-topic, smooth "
                 "in-betweens, no severe per-frame or temporal flaws, no "
                 "watermark/signature overlay"),
                ("Borderline", "Usable but average — choppy in-betweens, jpeg/AI "
                 "blocking, mild flicker, or a visible signature -> review"),
                ("LiveAction", "Live-action or photoreal footage — NOT drawn/"
                 "animated motion -> off-theme for an animation dataset"),
                ("OffTopic", "Unrelated to the curation subject"),
            ],
            theme_rules=(
                "Animation / 2D-motion dataset. art_medium MUST be a drawn or "
                "rendered animated style (anime, cel, digital animation). "
                "Live-action or photoreal footage is LiveAction (off-theme). Do "
                "NOT apply real-human gates here. has_ai_flaws/temporal flaws = "
                "TRUE only for severe issues (broken anatomy, melted detail, "
                "warped text, heavy flicker or morphing between frames). Penalise "
                "choppy motion, jpeg blocking and visible artist "
                "signatures/watermarks (route Borderline)."),
            scoring_notes=(
                "Reward clean line work, smooth in-betweens, coherent anatomy and "
                "temporal stability. Penalise choppy motion, jpeg blocking, severe "
                "AI anatomy flaws, flicker and signature/watermark overlays."),
            caption_style="booru_tags",
        ),

        # ── product / turntable / demo (parallel to "product_ecommerce") ─────
        "video_product": _preset(
            require_prompt=True,
            ovr_min=50, rel_min=25,
            min_prompt_length=18,
            reddit_subreddits=("videography", "commercials", "productphotography"),
            keywords_extra=("product video", "turntable", "360 spin", "demo",
                            "unboxing", "rotating", "orbit", "studio",
                            "product showcase", "motion"),
            banned_keywords=_SPAM_BANNED + ("nsfw", "selfie", "meme",
                                            "screen recording"),
            generation_hints=("product turntable", "360 rotation", "orbit shot",
                              "slow rotation", "studio lighting", "smooth motion",
                              "product demo", "tracking shot"),
            rules_preamble=_VIDEO_RULES,
            categories=[
                ("Keep", "Single clearly-presented product with clean motion "
                 "(turntable, slow orbit, controlled push-in), seamless/clean "
                 "background, sharp, well-lit, accurate colour, no distracting "
                 "watermark"),
                ("Lifestyle", "Product shown in-use / lifestyle motion — useful "
                 "but a different bucket from clean studio turntables"),
                ("Borderline", "Product motion but cluttered background, soft "
                 "focus, poor lighting, shaky orbit, or a minor watermark -> "
                 "review"),
                ("OffTopic", "No clear product, or unrelated to the catalogue "
                 "subject"),
            ],
            theme_rules=(
                "Product / e-commerce video dataset. Reward a single clearly-"
                "presented product with smooth controlled motion (turntable, slow "
                "360, motivated push-in), even studio lighting, a seamless/clean "
                "background and accurate colour. A product shown in a real-world / "
                "lifestyle scene goes to Lifestyle. Penalise busy backgrounds, "
                "shaky or stuttering motion, distracting reflections and burned-in "
                "promo text or watermarks (contains_text_overlay=true -> "
                "Borderline at best; DISCARD if the overlay dominates)."),
            scoring_notes=(
                "Reward clean/seamless backgrounds, even lighting, sharpness, "
                "accurate colour and smooth controlled motion. Penalise clutter, "
                "shaky/stuttering motion, distracting reflections and burned-in "
                "promo text/watermarks."),
            caption_style="natural_language",
        ),

        # ── nature / wildlife / landscape motion (parallel to wildlife) ──────
        "video_nature": _preset(
            require_prompt=True,
            ovr_min=50, rel_min=25,
            min_prompt_length=18,
            reddit_subreddits=("NatureGifs", "NatureIsFuckingLit", "BBCEarth"),
            x_accounts=("BBCEarth",),
            keywords_extra=("nature", "wildlife", "landscape", "drone", "aerial",
                            "timelapse", "slow motion", "b-roll", "tracking shot",
                            "moving"),
            banned_keywords=_SPAM_BANNED + ("zoo", "captive", "aquarium",
                                            "anime", "cartoon"),
            generation_hints=("drone shot", "aerial footage", "tracking shot",
                              "slow motion", "timelapse", "establishing shot",
                              "b-roll", "smooth motion"),
            rules_preamble=_VIDEO_RULES,
            categories=[
                ("Keep", "Sharp natural-world clip (wildlife, landscape, aerial) "
                 "with clean, coherent motion, natural setting, on-topic, OVR>=60 "
                 "and REL>=60"),
                ("Borderline", "Nature motion but soft focus, distant/tiny "
                 "subject, busy frame, heavy haze, or jerky motion -> review"),
                ("CaptiveStaged", "Obvious zoo bars/enclosure, aquarium glass, or "
                 "a captive/staged setting standing in for the wild -> off-theme"),
                ("OffTopic", "Unrelated to the nature/wildlife subject"),
            ],
            theme_rules=(
                "Nature / wildlife / landscape video dataset. Reward critical "
                "sharpness on the subject (the animal's eye, the landscape's key "
                "plane), a natural setting and smooth, coherent motion (drone, "
                "tracking, gentle slow-motion or timelapse). A zoo enclosure, "
                "aquarium glass or other captive/staged setting is CaptiveStaged. "
                "Penalise visible cages/bars, heavy atmospheric haze, motion smear "
                "and stutter on the subject."),
            scoring_notes=(
                "Reward subject sharpness, natural setting, clean separation and "
                "smooth coherent motion. Penalise soft focus, tiny-in-frame "
                "subjects, cage bars, haze and stutter/motion smear."),
            caption_style="natural_language",
        ),
    }


def _build_presets() -> dict[str, dict]:
    presets: dict[str, dict] = {
        # ── general dataset-prep triage — the new shipped DEFAULT ───────────
        "default": _preset(
            require_prompt=True,
            ovr_min=40, rel_min=20,
            banned_keywords=_SPAM_BANNED,
            # generation_hints intentionally EMPTY: this is the lenient general
            # triage. Seeding hints turns on a require-a-generation-marker gate
            # that would drop any prompt lacking those exact words.
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
            keywords_extra=("aerial", "drone", "overhead", "birds eye", "top-down",
                            "satellite", "from above", "altitude"),
            banned_keywords=_SPAM_BANNED + ("indoor", "selfie", "anime", "cartoon"),
            generation_hints=("aerial photograph", "drone shot", "bird's eye view",
                              "from above", "satellite imagery", "high altitude"),
            reddit_subreddits=("drones", "aerialphotography", "dronephotography",
                               "Multicopter", "SatelliteImages", "fpv", "AerialPorn"),
            x_accounts=("DroneDJ",),
            ovr_min=50, rel_min=25,
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
            keywords_extra=("underwater", "scuba", "diving", "reef", "ocean",
                            "marine", "coral", "sea"),
            banned_keywords=_SPAM_BANNED + ("aquarium", "fish tank", "swimming pool",
                                            "anime"),
            generation_hints=("underwater photograph", "scuba diving", "deep sea",
                              "coral reef", "marine life", "underwater scene"),
            reddit_subreddits=("underwaterphotography", "scuba", "ScubaDiving",
                               "reef", "WaterPorn", "Ocean", "marinebiology"),
            ovr_min=50, rel_min=25,
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
            keywords_extra=("wildlife", "macro", "insect", "bird", "animal",
                            "nature", "butterfly", "close-up"),
            banned_keywords=_SPAM_BANNED + ("zoo", "captive", "taxidermy",
                                            "plush toy", "anime"),
            generation_hints=("wildlife photography", "macro shot", "telephoto",
                              "natural habitat", "nature photography", "in the wild"),
            reddit_subreddits=("macro", "wildlifephotography", "macrophotography",
                               "naturephotography", "birdphotography", "insects"),
            x_accounts=("NatGeoPhotos",),
            ovr_min=50, rel_min=25,
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
            keywords_extra=("product", "packshot", "studio", "catalog",
                            "e-commerce", "white background", "product photography"),
            banned_keywords=_SPAM_BANNED + ("nsfw", "selfie", "meme", "screenshot"),
            generation_hints=("product photography", "studio lighting",
                              "white background", "commercial shot", "packshot",
                              "catalog photo"),
            reddit_subreddits=("productphotography", "commercialphotography"),
            ovr_min=50, rel_min=25,
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
            keywords_extra=("anime", "illustration", "manga", "digital art",
                            "character"),
            banned_keywords=_SPAM_BANNED + ("photograph", "photorealistic",
                                            "3d render"),
            generation_hints=("anime", "illustration", "masterpiece",
                              "best quality", "detailed"),
            reddit_subreddits=("anime", "awwnime", "Animewallpaper",
                               "ImaginaryCharacters", "DigitalArt", "Illustration",
                               "characterdrawing"),
            ovr_min=45, rel_min=25,
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
            keywords_extra=("woman", "model", "portrait", "fashion", "editorial",
                            "photoshoot"),
            banned_keywords=_SPAM_BANNED + ("linktr.ee", "telegram", "snapchat",
                                            "cashapp"),
            generation_hints=("photorealistic", "skin texture", "portrait",
                              "natural light", "studio light", "bokeh", "85mm"),
            reddit_subreddits=("portraits",),
            ovr_min=55, rel_min=0,
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
            ovr_min=40, rel_min=0,
            banned_keywords=_SPAM_BANNED,
            # generation_hints intentionally EMPTY: topic-agnostic quality triage
            # — it routes on craft, not on whether the prompt looks AI-generated.
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
    # Video presets are a parallel themed set built the same way; merge them in
    # so PRESET_NAMES exposes both families from one source of truth.
    presets.update(_build_video_presets())
    return presets


# Build the library ONCE at import; every public accessor deep-copies from this
# single source so PRESET_NAMES and builtin_library() can never diverge.
_PRESETS: dict[str, dict] = _build_presets()
PRESET_NAMES: tuple[str, ...] = tuple(_PRESETS.keys())


def builtin_library() -> dict:
    """Return a fresh {default, presets} library (safe for the caller to mutate)."""
    return {"default": DEFAULT_PRESET, "presets": copy.deepcopy(_PRESETS)}
