#!/usr/bin/env python3
"""Generic OpenAI-compatible vision worker.

Targets any local inference server that exposes the OpenAI ``/v1/chat/completions``
endpoint with multimodal ``image_url`` content blocks and strict JSON-schema
``response_format`` structured output. Confirmed compatible:

  * llama.cpp ``llama-server`` (default since the multimodal merge)
  * vLLM (with ``--enable-auto-tool-choice`` for some models)
  * koboldcpp (OpenAI extension)
  * LocalAI
  * text-generation-webui (OpenAI extension)
  * Any other server that implements the same surface

For LM Studio specifically, use ``balanced-lm`` / ``lm-autodetect`` — they read
the ``LMSTUDIO_*`` env vars users already have configured. For Ollama, use
``balanced-ollama`` — its OpenAI compat layer has historically had weaker JSON-
schema support, so the native ``/api/chat`` path is more reliable.

Env vars:

    OPENAI_COMPAT_URL       Base URL, e.g. ``http://127.0.0.1:8080`` (no trailing /v1).
    OPENAI_COMPAT_MODEL     Model identifier the server expects in the ``model`` field.
    OPENAI_COMPAT_API_KEY   Optional bearer token (LocalAI / hosted gateways).
    OPENAI_COMPAT_TIMEOUT   Per-request timeout in seconds. Default 600.
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

# Substrings that hint a listed model is vision-capable — used to pick sensibly
# when the user leaves the model blank (auto-detect on connect).
_VISION_HINTS = ("vl", "vision", "llava", "moondream", "minicpm-v", "pixtral",
                 "gemma-3", "qwen-vl", "internvl", "smolvlm", "phi-3.5-vision")


class BalancedOpenAICompatWorker(BaseVisionWorker):
    name = "balanced-openai-compat"
    parallel_workers = 4
    request_timeout = 600

    def __init__(self) -> None:
        super().__init__()
        self.base_url: str = os.environ.get(
            "OPENAI_COMPAT_URL", "http://127.0.0.1:8080"
        ).rstrip("/")
        self.model: str = os.environ.get("OPENAI_COMPAT_MODEL", "").strip()
        self.api_key: str = os.environ.get("OPENAI_COMPAT_API_KEY", "").strip()
        try:
            self.request_timeout = int(
                os.environ.get("OPENAI_COMPAT_TIMEOUT", str(self.request_timeout))
            )
        except ValueError:
            pass

    def setup(self) -> None:
        if not self.model:
            self.model = self._autodetect_model()
        if not self.model:
            raise SystemExit(
                "OPENAI_COMPAT_MODEL is unset and no model could be auto-detected "
                f"from {self.base_url}/v1/models. Set the model identifier your "
                "local server expects (run e.g. `curl <url>/v1/models` to list)."
            )

    def _autodetect_model(self) -> str:
        """Pick a model when none is configured: prefer a vision-capable id, else
        the first one the server lists. Empty string if the endpoint is down."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = requests.get(f"{self.base_url}/v1/models", headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json().get("data") or []
        except (requests.RequestException, ValueError, KeyError):
            return ""
        ids = [m.get("id", "") for m in data if isinstance(m, dict) and m.get("id")]
        if not ids:
            return ""
        chosen = next((i for i in ids if any(h in i.lower() for h in _VISION_HINTS)), ids[0])
        logger.info("auto-detected model %r from %s/v1/models", chosen, self.base_url)
        return chosen

    def banner(self) -> None:
        logger.info("=== Vision Worker (OpenAI-compatible %s) ===", self.name)
        logger.info("  URL:     %s", self.base_url)
        logger.info("  Model:   %s", self.model)
        logger.info("  Workers: %d", self.parallel_workers)
        logger.info("  Auth:    %s", "bearer token set" if self.api_key else "(none)")

    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
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
            logger.warning("OpenAI-compat request failed: %s", exc)
            return None

        if response.status_code != 200:
            preview = response.text[:600].replace("\n", " ")
            logger.warning(
                "OpenAI-compat HTTP %d (model=%s url=%s): %s",
                response.status_code, self.model, self.base_url, preview,
            )
            return None

        try:
            message = response.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("OpenAI-compat: unexpected response shape: %s", exc)
            return None

        raw = extract_message_text(message)
        parsed = _safe_parse_vision_json(raw)
        if parsed is None:
            preview = (raw or "")[:300].replace("\n", " ")
            logger.warning(
                "empty/invalid JSON from %s@%s; raw=%r",
                self.model, self.base_url, preview,
            )
            return None
        return parsed


def run_vision_worker() -> None:
    run_subclass(BalancedOpenAICompatWorker)


if __name__ == "__main__":
    run_vision_worker()
