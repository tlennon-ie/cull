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
    _safe_parse_vision_json,
    apply_scores,
    build_classification_prompt,
    build_response_format,
    extract_message_text,
)

logger = get_logger(__name__)


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
        for category in CATEGORIES:
            (self.sorted_dir / category).mkdir(parents=True, exist_ok=True)
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

        img_bytes = processing_path.read_bytes()
        if not img_bytes:
            processing_path.unlink(missing_ok=True)
            return _Outcome.SKIPPED

        small = self._resize_jpeg(img_bytes)
        if small is None:
            return self._finalise_discard(ctx, reason="Image corrupt/invalid format/truncated")

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

    def _finalise(self, ctx: ClassifyContext, result: dict[str, Any]) -> str:
        final_category = result.get("category", "Unknown")
        msg_id_key = ctx.metadata.get("message_id", ctx.image_path.stem)

        dest_dir = self.sorted_dir / final_category / ctx.source_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        safe_name = (
            f"{final_category.lower()}_{msg_id_key}"
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

        final_meta.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ovr = result.get("OVR_Quality_Score", 0)
        rel = result.get("REL_Quality_Score", 0)
        logger.info(
            "[%s] OVR:%s REL:%s | %s", final_category, ovr, rel, ctx.image_path.name,
        )
        return final_category

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
