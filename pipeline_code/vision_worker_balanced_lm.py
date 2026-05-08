#!/usr/bin/env python3
"""LM Studio vision worker - targets the PRIMARY endpoint.

Run two parallel instances by spawning ``balanced-lm`` and
``balanced-lm-secondary`` in run_pipeline; the secondary variant inherits
this class but ``LMSTUDIO_PRIMARY_*`` is rebound to ``LMSTUDIO_SECONDARY_*``
in its env at spawn time (see ``vision_workers.WORKERS``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import requests

# Allow `python pipeline_code/vision_worker_balanced_lm.py` direct invocation.
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

_VISION_HINTS = (
    "vl", "vision", "visual", "llava", "gemma-3", "qwen3-vl",
    "qwen2.5-vl", "qwen2-vl", "moondream", "minicpm-v", "pixtral",
)


def _looks_vision_capable(model_id: str) -> bool:
    return any(h in (model_id or "").lower() for h in _VISION_HINTS)


class BalancedLMWorker(BaseVisionWorker):
    name = "balanced-lm"
    parallel_workers = 4
    request_timeout = 600

    def __init__(self) -> None:
        super().__init__()
        self.lms_url: str = os.environ.get(
            "LMSTUDIO_PRIMARY_URL", "http://127.0.0.1:1234"
        ).rstrip("/")
        self.lms_model: str = os.environ.get(
            "LMSTUDIO_PRIMARY_MODEL", "qwen3-vl-8b-thinking-abliterated"
        )

    def banner(self) -> None:
        logger.info("=== Vision Worker (LM Studio %s) ===", self.name)
        logger.info("  URL:     %s", self.lms_url)
        logger.info("  Model:   %s", self.lms_model)
        logger.info("  Workers: %d", self.parallel_workers)
        if self.lms_model and not _looks_vision_capable(self.lms_model):
            logger.warning(
                "model %r does not look vision-capable; expected qwen2.5-vl-7b / "
                "qwen3-vl-8b / gemma-3-* / llava-*. Update LMSTUDIO_PRIMARY_MODEL.",
                self.lms_model,
            )

    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        try:
            response = requests.post(
                f"{self.lms_url}/v1/chat/completions",
                json={
                    "model": self.lms_model,
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
                    # Thinking-style models spend tokens inside <think>...</think>
                    # before the JSON answer. 2000 leaves headroom for both.
                    "max_tokens": 2000,
                    # Force structured output so empty-string responses can't happen.
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
                "LM Studio HTTP %d (model=%s url=%s): %s",
                response.status_code, self.lms_model, self.lms_url, preview,
            )
            return None

        message = response.json()["choices"][0]["message"]
        raw = extract_message_text(message)
        parsed = _safe_parse_vision_json(raw)
        if parsed is None:
            preview = (raw or "")[:300].replace("\n", " ")
            logger.warning(
                "empty/invalid JSON from %s@%s; raw=%r",
                self.lms_model, self.lms_url, preview,
            )
            return None
        return parsed


def run_vision_worker() -> None:  # back-compat name kept for run_pipeline
    run_subclass(BalancedLMWorker)


if __name__ == "__main__":
    run_vision_worker()
