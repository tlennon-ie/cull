#!/usr/bin/env python3
"""OpenRouter cloud vision worker.

OpenRouter exposes the OpenAI ``/v1/chat/completions`` protocol verbatim, so
this worker reuses ``OpenAICompatibleClient`` from ``vision_worker_openai`` —
only the base URL, API key, and model env vars differ. Structured output still
travels via ``response_format={"type": "json_schema", ...}``.

Env vars:

    OPENROUTER_API_KEY        Required. Your OpenRouter key.
    OPENROUTER_VISION_MODEL   Model id. Default ``openai/gpt-4o`` (env-overridable).
    OPENROUTER_BASE_URL       API base. Default ``https://openrouter.ai/api/v1``.
    OPENROUTER_TIMEOUT        Per-request timeout (seconds). Default 600.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Allow direct invocation as a script.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

import vision_model_catalog as catalog
from pipeline_logging import get_logger
from vision_worker_base import BaseVisionWorker, run_subclass
from vision_worker_openai import OpenAICompatibleClient, _resolve_timeout

load_dotenv()

logger = get_logger(__name__)

# Default to a widely-available vision model on OpenRouter. Override via
# OPENROUTER_VISION_MODEL (e.g. anthropic/claude-3.5-sonnet, google/gemini-flash-1.5).
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterVisionWorker(BaseVisionWorker):
    name = "openrouter"
    parallel_workers = 4
    max_image_size = 1024

    def __init__(self) -> None:
        super().__init__()
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "No OPENROUTER_API_KEY configured. Set it in .env or via the "
                "dashboard Settings tab."
            )
        self.base_url: str = os.environ.get(
            "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
        ).strip() or DEFAULT_OPENROUTER_BASE_URL
        # Autodetect a vision model from OpenRouter's catalogue when none is set;
        # the frozen default is only the last-resort fallback.
        model_env = os.environ.get("OPENROUTER_VISION_MODEL", "").strip()
        self.model_id: str = model_env or catalog.autodetect_model(
            "openrouter", base_url=self.base_url, api_key=api_key,
            fallback=DEFAULT_OPENROUTER_MODEL,
        )
        self._client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=self.base_url,
            model=self.model_id,
            timeout=_resolve_timeout("OPENROUTER_TIMEOUT"),
        )

    def banner(self) -> None:
        logger.info("=== Vision Worker (OpenRouter %s) ===", self.name)
        logger.info("  Model:   %s", self.model_id)
        logger.info("  Base:    %s", self.base_url)
        logger.info("  Workers: %d", self.parallel_workers)

    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        return self._client.classify(b64_jpeg, prompt_instruction)


def run_vision_worker() -> None:
    run_subclass(OpenRouterVisionWorker)


if __name__ == "__main__":
    run_vision_worker()
