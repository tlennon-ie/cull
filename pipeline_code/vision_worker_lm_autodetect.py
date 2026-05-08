#!/usr/bin/env python3
"""LM Studio vision worker that auto-picks whichever VL model is loaded.

Probes ``LMSTUDIO_PRIMARY_URL`` and ``LMSTUDIO_SECONDARY_URL`` at startup,
selects the first endpoint with a vision-capable model (qwen-vl / gemma-3 /
llava / minicpm-v / etc.), and uses that pair for the rest of its life.
Never picks a text-only model - the worker needs multimodal to classify
images.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from pipeline_logging import get_logger
from vision_worker_base import (
    BaseVisionWorker,
    build_response_format,
    extract_message_text,
    _safe_parse_vision_json,
    run_subclass,
)

load_dotenv()

logger = get_logger(__name__)

VISION_HINTS = (
    "vl", "vision", "visual", "llava", "gemma-3", "qwen3-vl",
    "qwen2.5-vl", "qwen2-vl", "moondream", "minicpm-v", "pixtral",
)
EMBED_HINTS = ("embed", "nomic", "bge")


class LMAutodetectWorker(BaseVisionWorker):
    name = "lm-autodetect"
    parallel_workers = 4
    request_timeout = 600

    def __init__(self) -> None:
        super().__init__()
        self.primary_url: str = os.environ.get("LMSTUDIO_PRIMARY_URL", "").rstrip("/")
        self.secondary_url: str = os.environ.get("LMSTUDIO_SECONDARY_URL", "").rstrip("/")
        # These get set by setup() once an endpoint is discovered.
        self.lms_url: str = ""
        self.model_id: str = ""

    # ── Discovery ─────────────────────────────────────────────────────────

    @staticmethod
    def _pick_vision_model(models: list[str]) -> str | None:
        if not models:
            return None
        non_embed = [m for m in models if not any(h in m.lower() for h in EMBED_HINTS)]
        vision = [m for m in non_embed if any(h in m.lower() for h in VISION_HINTS)]
        return vision[0] if vision else None

    @staticmethod
    def _models_on(base_url: str) -> list[str]:
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=5)
            if resp.status_code == 200:
                return [
                    item.get("id", "")
                    for item in resp.json().get("data", [])
                    if item.get("id")
                ]
        except requests.RequestException:
            pass
        return []

    def _discover(self) -> tuple[str, str] | tuple[str, str]:
        """Pick the first endpoint with a VL model. Returns ('','') if none."""
        candidates: list[tuple[str, str, list[str]]] = []
        for label, url in (("primary", self.primary_url), ("secondary", self.secondary_url)):
            if not url:
                continue
            models = self._models_on(url)
            if not models:
                logger.warning("autodetect: %s (%s) unreachable or empty", label, url)
                continue
            candidates.append((label, url, models))

        for label, url, models in candidates:
            picked = self._pick_vision_model(models)
            if picked:
                logger.info("autodetect: %s -> %s (all: %s)", label, picked, models)
                return url, picked

        for label, _url, models in candidates:
            logger.warning("autodetect: %s has no VL model loaded: %s", label, models)
        return "", ""

    # ── Hooks ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        url, model = self._discover()
        if not model:
            raise SystemExit(
                "no LM Studio instance with a vision-capable model found. Load "
                "qwen2.5-vl-7b / qwen3-vl-8b / gemma-3-* / llava-* on at least "
                "one of LMSTUDIO_PRIMARY_URL / LMSTUDIO_SECONDARY_URL and retry."
            )
        self.lms_url = url
        self.model_id = model

    def banner(self) -> None:
        logger.info("=== Vision Worker (LM Studio %s) ===", self.name)
        logger.info("  Primary URL:   %s", self.primary_url or "(unset)")
        logger.info("  Secondary URL: %s", self.secondary_url or "(unset)")
        logger.info("  Workers:       %d", self.parallel_workers)

    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        try:
            response = requests.post(
                f"{self.lms_url}/v1/chat/completions",
                json={
                    "model": self.model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_instruction},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64_jpeg}",
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "response_format": build_response_format(),
                },
                timeout=self.request_timeout,
            )
        except requests.RequestException as exc:
            logger.warning("LM Studio request failed: %s", exc)
            return None

        if response.status_code != 200:
            preview = response.text[:600].replace("\n", " ")
            logger.warning(
                "LM Studio HTTP %d: %s", response.status_code, preview,
            )
            return None

        message = response.json()["choices"][0]["message"]
        raw = extract_message_text(message)
        parsed = _safe_parse_vision_json(raw)
        if parsed is None:
            preview = (raw or "")[:300].replace("\n", " ")
            logger.warning(
                "empty/invalid JSON from %s@%s; raw=%r",
                self.model_id, self.lms_url, preview,
            )
            return None
        return parsed


def run_vision_worker() -> None:
    run_subclass(LMAutodetectWorker)


if __name__ == "__main__":
    run_vision_worker()
