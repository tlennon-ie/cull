"""Template-method base class for vision workers.

Every concrete worker before this refactor reimplemented the same ~150 LOC
of scaffolding:

  * resolve env (paths, slug, sorted dir)
  * mkdir each category folder
  * loop: pop next image -> rename to .processing -> resize -> b64 -> call API
  * parse model output -> apply_scores -> move to sorted/<cat>/<source>/
  * write .vision.json next to the moved image
  * handle SKIPPED / RETRY / DISCARD outcomes uniformly

Only the call-API step varied. ``BaseVisionWorker`` owns everything else;
subclasses implement one method:

    def classify_image_bytes(self, b64_jpeg: str, prompt_instruction: str) -> dict | None

Returning ``None`` triggers a RETRY (re-renames .processing back to the queue
location for the next worker / next sweep). Returning a dict is treated as
the model's raw JSON; the base class runs ``apply_scores`` on it before
choosing the destination folder.

Optional hooks:

    def setup(self) -> None
        Called once before ``run()`` enters its main loop. Use this for
        model discovery, warm-up calls, background threads, etc.

    def banner(self) -> None
        Override to print a startup banner identifying the worker.

Class-level attributes that subclasses commonly override:

    name: str               # used in log messages
    parallel_workers: int   # ThreadPoolExecutor size, default 4
    max_image_size: int     # longest-edge resize target, default 512
    poll_interval: float    # sleep when queue empty, default 2.0s
    request_timeout: int    # per-API-call timeout, default 600s
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import shutil
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from categories import CATEGORIES
from paths import sorted_dir
from pipeline_logging import get_logger
from queue_manager import get_next_image_round_robin
from vision_prompt import (
    CaptionConfig,
    _safe_parse_vision_json,
    apply_scores,
    build_classification_prompt,
    build_response_format,
    extract_message_text,
)

logger = get_logger(__name__)


def _safe_component(value: str, fallback: str) -> str:
    """Reduce a user/model-derived string to a single safe path component.

    ``os.path.basename`` strips any directory prefix and ``..``/separator
    sequences, so the result can never escape the directory it's joined to —
    a path-injection barrier for the category / source folder names used when
    routing a classified image. Empty / separator-only inputs fall back to
    ``fallback`` so a malformed model answer still lands somewhere valid.
    """
    base = os.path.basename(str(value or "").strip().replace("\\", "/").rstrip("/"))
    base = base.strip()
    if not base or base in (".", ".."):
        return fallback
    return base


# Single sentinel used for non-final outcomes so callers can branch cleanly.
class _Outcome:
    SKIPPED = "SKIPPED"
    RETRY = "RETRY"


@dataclass(frozen=True)
class ClassifyContext:
    """Inputs assembled before the API call. Subclasses get this + the b64 image."""
    image_path: Path
    processing_path: Path
    source_name: str
    prompt_text: str
    metadata: dict[str, Any]


class BaseVisionWorker(ABC):
    name: str = "base"
    parallel_workers: int = 4
    max_image_size: int = 512
    poll_interval: float = 2.0
    request_timeout: int = 600

    def __init__(self) -> None:
        self.sorted_dir: Path = sorted_dir()
        # Pre-create the active taxonomy's folders so the first write doesn't
        # race against mkdir. _finalise also mkdirs lazily so a category added
        # while the worker is mid-run still routes correctly.
        from categories import get_categories  # lazy: pick up edits made between imports
        for category in get_categories():
            # Constrain each taxonomy-derived folder name to a single safe
            # component before it becomes a path (path-injection barrier).
            (self.sorted_dir / _safe_component(category, "Unknown")).mkdir(
                parents=True, exist_ok=True
            )
        # Subclasses can stash backend-specific state here in setup().
        self._processed_count: int = 0

    # ── Subclass contract ─────────────────────────────────────────────────

    @abstractmethod
    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        """Call the backend with the base64-encoded JPEG + classification
        instruction. Return the parsed JSON dict (the model's raw answer
        before ``apply_scores``) or ``None`` to trigger a RETRY."""

    def setup(self) -> None:
        """Hook for one-time initialisation. Override if you need to discover
        a model, start a keepalive thread, etc. Default: no-op."""

    def banner(self) -> None:
        """Override to print a startup banner. Default prints the class name."""
        logger.info("=== Vision Worker (%s) ===", self.name)
        logger.info("  Workers: %d", self.parallel_workers)

    # ── Main loop (final - subclasses don't override) ─────────────────────

    def run(self) -> None:
        self.banner()
        self.setup()

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            try:
                while True:
                    source_name, img_path = get_next_image_round_robin()
                    if img_path is None:
                        time.sleep(self.poll_interval)
                        continue
                    img_path = Path(img_path)
                    if not img_path.exists():
                        continue
                    future = pool.submit(self._process_image, img_path, source_name)
                    result = future.result(timeout=300)
                    if isinstance(result, str) and result not in (_Outcome.SKIPPED, _Outcome.RETRY):
                        self._processed_count += 1
                        if self._processed_count % 10 == 0:
                            logger.info("processed %d images", self._processed_count)
            except KeyboardInterrupt:
                logger.info("vision worker stopped")

    # ── Per-image pipeline (final) ────────────────────────────────────────

    def _process_image(self, image_path: Path, source_name: str) -> str:
        """Atomic claim → resize → classify → write outcome. Returns the
        final category name (used to count throughput) or a SKIPPED/RETRY
        sentinel string."""
        processing_path = Path(str(image_path) + ".processing")
        try:
            image_path.rename(processing_path)
        except FileNotFoundError:
            return _Outcome.SKIPPED  # another worker grabbed it
        except OSError as exc:
            logger.warning("failed to claim %s: %s", image_path, exc)
            return _Outcome.SKIPPED

        ctx = self._gather_context(image_path, processing_path, source_name)
        if ctx is None:
            return _Outcome.SKIPPED

        # Video with VIDEO_CLASSIFY_ENABLED=true but NO frame-extraction backend
        # installed: HOLD the clip in Unclassified_Video (preserving the file
        # and its .txt / .meta.json siblings) instead of silently DISCARDing it.
        # This is the common "user turned on video mode without installing
        # ffmpeg" foot-gun — see run_pipeline supervisor warnings and the
        # /api/video/backend-status endpoint feeding the dashboard banner.
        if self._is_video_without_backend(ctx):
            return self._finalise_hold_video(ctx)

        # Acquire the bytes to CLASSIFY. For a still that's the file's own bytes.
        # For a video clip (only when VIDEO_CLASSIFY_ENABLED), we extract a
        # representative frame and classify the FRAME's bytes — the CLIP itself is
        # what _finalise then sorts. ``None`` means "video frame extraction failed"
        # (no extraction backend / no decodable frame): route the clip to terminal
        # DISCARD rather than re-queue it (re-queuing would hot-loop on an
        # undecodable clip) or orphan it forever as a stray ``.processing`` file.
        img_bytes = self._acquire_classify_bytes(ctx, processing_path)
        if img_bytes is None:
            return self._finalise_discard(
                ctx, reason="video frame extraction failed (install cull[video])")
        if not img_bytes:
            processing_path.unlink(missing_ok=True)
            return _Outcome.SKIPPED

        small = self._resize_jpeg(img_bytes)
        if small is None:
            return self._finalise_discard(ctx, reason="Image corrupt/invalid format/truncated")

        # Optional cheap pre-stage (gated PREFILTER_ENABLED, default OFF): score
        # the image's technical quality BEFORE the expensive VLM call and route
        # obvious rejects (blurry / near-black / tiny / banner-ratio) straight to
        # DISCARD, saving tokens. Runs on the ORIGINAL bytes (correct resolution /
        # aspect gates) and is fully fail-open — any prefilter error falls through
        # to normal classification. NOT inside the .processing rename.
        prefilter_reason = self._prefilter_reject(img_bytes)
        if prefilter_reason is not None:
            return self._finalise_discard(ctx, reason=prefilter_reason)

        b64 = base64.standard_b64encode(small).decode()
        prompt_instruction = build_classification_prompt()

        try:
            raw_result = self.classify_image_bytes(b64, prompt_instruction)
        except Exception:
            logger.exception("backend call failed for %s", image_path.name)
            self._reset_to_queue(processing_path, image_path)
            return _Outcome.RETRY

        if raw_result is None:
            self._reset_to_queue(processing_path, image_path)
            return _Outcome.RETRY

        result = apply_scores(raw_result)
        return self._finalise(ctx, result)

    # ── Internals ─────────────────────────────────────────────────────────

    def _gather_context(
        self,
        image_path: Path,
        processing_path: Path,
        source_name: str,
    ) -> ClassifyContext | None:
        stem = image_path.stem
        txt_path = image_path.parent / f"{stem}.txt"
        if not txt_path.exists():
            txt_path = image_path.parent / f"{image_path.name}.txt"
        meta_path = image_path.parent / f"{stem}.meta.json"
        try:
            metadata = (
                json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists() else {}
            )
            prompt_text = (
                txt_path.read_text(encoding="utf-8").strip()
                if txt_path.exists() else ""
            )
        except FileNotFoundError:
            processing_path.unlink(missing_ok=True)
            return None
        return ClassifyContext(
            image_path=image_path,
            processing_path=processing_path,
            source_name=source_name,
            prompt_text=prompt_text,
            metadata=metadata,
        )

    def _acquire_classify_bytes(
        self, ctx: ClassifyContext, processing_path: Path,
    ) -> bytes | None:
        """Return the bytes to classify, or ``None`` to skip the item gracefully.

        * A still image → its own bytes (read from the claimed ``.processing``
          file). This is the unchanged default path.
        * A video clip, ONLY when ``VIDEO_CLASSIFY_ENABLED`` → bytes of a single
          extracted representative frame (the CLIP is sorted later by
          ``_finalise``; we only classify the frame). ``video_frames`` is imported
          lazily + defensively so a missing optional dep never crashes the worker.

        Returns ``None`` when a video yields no frame (no backend / extraction
        failure); the caller then routes the clip to terminal DISCARD (it can't
        be classified, and re-queuing would hot-loop on an undecodable clip).
        Never touches the atomic ``.processing`` rename — operates on the
        already-claimed path.
        """
        try:
            import video_frames
            is_video = video_frames.is_video(ctx.image_path) and video_frames.is_enabled()
        except Exception as exc:  # noqa: BLE001 - optional module; fall back to still path
            logger.debug("video_frames unavailable (%s); treating %s as a still",
                         exc, ctx.image_path.name)
            is_video = False

        if not is_video:
            return processing_path.read_bytes()

        # Video lane: extract one representative frame and classify its bytes.
        frames: list[Path] = []
        try:
            frames = video_frames.extract_keyframes(processing_path, max_frames=1)
            if not frames:
                logger.info("no frame extracted from clip %s; skipping",
                            ctx.image_path.name)
                return None
            return frames[0].read_bytes()
        except Exception as exc:  # noqa: BLE001 - extraction is best-effort; skip the clip
            logger.warning("frame extraction failed for %s (%s); skipping",
                           ctx.image_path.name, exc)
            return None
        finally:
            if frames:
                video_frames.cleanup_frames(frames)

    def _resize_jpeg(self, data: bytes) -> bytes | None:
        if not data:
            return None
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > self.max_image_size:
                scale = self.max_image_size / max(w, h)
                img = img.resize(
                    (int(w * scale), int(h * scale)),
                    Image.LANCZOS,
                )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except (OSError, ValueError) as exc:
            logger.debug("resize failed: %s", exc)
            return None

    @staticmethod
    def _prefilter_reject(img_bytes: bytes) -> str | None:
        """Pre-VLM quality gate. Returns a DISCARD reason when the image should
        be rejected before the model call, or ``None`` to proceed.

        Gated behind ``PREFILTER_ENABLED`` (default OFF) — a no-op returning
        ``None`` unless explicitly enabled, so default behaviour is unchanged.
        Lazily imports ``prefilter_aesthetic`` so the worker still runs if the
        optional module/deps are absent, and is fully FAIL-OPEN: any error (or a
        passing score) returns ``None`` so the image continues to normal
        classification rather than being wrongly dropped."""
        try:
            import prefilter_aesthetic
            if not prefilter_aesthetic.is_enabled():
                return None
            verdict = prefilter_aesthetic.assess(
                img_bytes, min_score=prefilter_aesthetic.env_min_score(),
            )
            if verdict.get("passed"):
                return None
            reasons = verdict.get("reasons") or ["prefilter quality gate failed"]
            return "Prefilter rejected: " + "; ".join(str(r) for r in reasons)
        except Exception as exc:  # noqa: BLE001 - fail-open: never drop on prefilter error
            logger.debug("prefilter skipped (fail-open): %s", exc)
            return None

    def _finalise(self, ctx: ClassifyContext, result: dict[str, Any]) -> str:
        final_category = result.get("category", "Unknown")
        msg_id_key = ctx.metadata.get("message_id", ctx.image_path.stem)

        # The category (model output) and source (queue-derived) become directory
        # names — constrain each to a single safe component so a crafted value
        # can't escape the sorted root (path-injection barrier).
        safe_category = _safe_component(final_category, "Unknown")
        safe_source = _safe_component(ctx.source_name, "unknown")
        dest_dir = self.sorted_dir / safe_category / safe_source
        dest_dir.mkdir(parents=True, exist_ok=True)

        safe_name = (
            f"{safe_category.lower()}_{msg_id_key}"
            f"_{int(time.time())}_{random.randint(100, 999)}"
        )
        ext = ctx.image_path.suffix
        final_img = dest_dir / f"{safe_name}{ext}"
        final_txt = dest_dir / f"{safe_name}.txt"
        final_meta = dest_dir / f"{safe_name}.vision.json"

        try:
            shutil.move(str(ctx.processing_path), str(final_img))
            if (ctx.image_path.parent / f"{ctx.image_path.stem}.txt").exists():
                shutil.move(
                    str(ctx.image_path.parent / f"{ctx.image_path.stem}.txt"),
                    str(final_txt),
                )
        except FileNotFoundError:
            return _Outcome.SKIPPED

        # Auto-caption: write the model's caption to .txt when enabled. The
        # source-side prompt (if any) was just moved to `final_txt`; we only
        # overwrite it when the admin has explicitly opted in. This is the
        # contract surfaced in the dashboard's Vision tab.
        caption_cfg = CaptionConfig.from_env()
        if caption_cfg.enabled:
            raw_caption = (result.get("caption") or "").strip()
            if raw_caption:
                had_source_prompt = bool(ctx.prompt_text.strip())
                if not had_source_prompt or caption_cfg.overwrite:
                    try:
                        final_txt.write_text(raw_caption, encoding="utf-8")
                    except OSError as exc:
                        logger.warning("caption write failed for %s: %s",
                                       final_img.name, exc)

        final_meta.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Perceptual-hash the just-sorted image so the Duplicates view has data.
        # Deferred Wave-1 call-site; purely additive — runs only after the image
        # is safely in its final sorted path and never breaks finalize.
        self._store_phash(final_img)
        ovr = result.get("OVR_Quality_Score", 0)
        rel = result.get("REL_Quality_Score", 0)
        logger.info(
            "[%s] OVR:%s REL:%s | %s", final_category, ovr, rel, ctx.image_path.name,
        )
        return final_category

    @staticmethod
    def _store_phash(final_img: Path) -> None:
        """Compute + persist the perceptual hash of an already-sorted image.

        Lazily imported so the worker still starts if ``phash_dedup`` /
        ``index_store`` (or Pillow features they need) are unavailable, and fully
        swallowed: a perceptual-hash failure must NEVER break finalize. The index
        row is upserted by the background indexer; ``set_phash`` only annotates it
        (a no-op when the row isn't indexed yet — the next scan backfills it)."""
        try:
            import index_store
            import phash_dedup
            phash = phash_dedup.compute_phash(final_img)
            index_store.set_phash(str(final_img), phash)
        except Exception as exc:  # noqa: BLE001 - phash is best-effort, never fatal
            logger.debug("phash store skipped for %s: %s", final_img.name, exc)

    @staticmethod
    def _is_video_without_backend(ctx: ClassifyContext) -> bool:
        """True iff the item is a video, video-mode is enabled, AND no frame-
        extraction backend (ffmpeg / scenedetect) is installed.

        The three-way check is deliberate: with video-mode OFF the still-image
        path already handles (misclassifies, technically) a raw video by reading
        its container bytes — that's the pre-existing behaviour and not our fix
        to make. Only when the operator has ASKED for video classification but
        has no backend do we intercept, so the failure mode surfaces exactly
        where the operator wired it up. Fully defensive: any probe error means
        "don't intercept" so a broken import can never mass-hold real work.
        """
        try:
            import video_frames
        except Exception:  # noqa: BLE001 - optional module: don't intercept on failure
            return False
        try:
            if not video_frames.is_video(ctx.image_path):
                return False
            if not video_frames.is_enabled():
                return False
            return not video_frames.has_any_backend()
        except Exception as exc:  # noqa: BLE001 - probe must never break the worker
            logger.debug("video-backend probe failed for %s: %s",
                         ctx.image_path.name, exc)
            return False

    def _finalise_hold_video(self, ctx: ClassifyContext) -> str:
        """Route a video clip to the ``Unclassified_Video`` bucket WITHOUT
        discarding it, when we lack a frame-extraction backend.

        The clip and its sidecar files land in
        ``data/sorted/<slug>/Unclassified_Video/<source>/`` via the standard
        ``_finalise`` move, and a ``.vision.json`` marker records why the
        classifier bailed. The user can install ffmpeg / scenedetect and
        re-queue with ``tools/requeue_sorted.py Unclassified_Video`` without
        losing anything. Logs at WARNING (not INFO) so it's visible on the
        supervisor stream — this is a configuration gap, not a normal outcome.
        """
        reason = (
            "no video backend installed "
            "(install ffmpeg or scenedetect to classify)"
        )
        logger.warning(
            "HOLD video (no backend): %s → Unclassified_Video/%s",
            ctx.image_path.name, ctx.source_name,
        )
        # apply_scores enforces the response shape, but we set the category
        # ourselves so it survives any category-repair pass. _finalise's
        # _safe_component sanitises the folder name.
        result = apply_scores({
            "description": "video clip held: no frame-extraction backend installed",
            "primary_subject": "video",
            "is_screenshot": False,
            "is_composite_grid": False,
            "contains_text_overlay": False,
            "is_human_photograph": False,
            "art_medium": "unclear",
            "photorealistic_style": False,
            "has_ai_flaws": False,
            "woman_present": False,
            "nsfw": False,
            "OVR_Quality_Score": 0,
            "REL_Quality_Score": 0,
            "quality_score": 0,
            "category": "Unclassified_Video",
            "reason": reason,
        })
        # apply_scores may rewrite the category (e.g. into DISCARD) based on the
        # active taxonomy — force it back so the clip lands in the hold bucket.
        result["category"] = "Unclassified_Video"
        result["reason"] = reason
        return self._finalise(ctx, result)

    def _finalise_discard(self, ctx: ClassifyContext, reason: str) -> str:
        """Special-case: image bytes are unreadable. Skip the API and route
        directly to DISCARD with a synthetic result."""
        result = apply_scores({
            "description": "image bytes unreadable",
            "primary_subject": "unknown",
            "is_screenshot": False,
            "is_composite_grid": False,
            "contains_text_overlay": False,
            "is_human_photograph": False,
            "art_medium": "unclear",
            "photorealistic_style": False,
            "has_ai_flaws": True,
            "woman_present": False,
            "nsfw": False,
            "OVR_Quality_Score": 0,
            "REL_Quality_Score": 0,
            "quality_score": 1,
            "category": "DISCARD",
            "reason": reason,
        })
        return self._finalise(ctx, result)

    @staticmethod
    def _reset_to_queue(processing_path: Path, image_path: Path) -> None:
        """Move ``.processing`` back to its original name so another worker /
        the next sweep can re-attempt classification."""
        try:
            processing_path.rename(image_path)
        except OSError:
            pass


# Convenience for `python vision_worker_x.py` — subclasses just call
# `BaseVisionWorker.run_subclass(MyWorker)` from their __main__.

def run_subclass(cls: type[BaseVisionWorker]) -> None:
    cls().run()


__all__ = [
    "BaseVisionWorker",
    "ClassifyContext",
    "run_subclass",
    "build_classification_prompt",  # re-export for subclass convenience
    "build_response_format",
    "extract_message_text",
    "_safe_parse_vision_json",
]
