#!/usr/bin/env python3
"""Ollama vision worker — hits ``/api/chat`` with native JSON-schema ``format``.

Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, but its
handling of ``response_format: {"type": "json_schema", ...}`` has historically
been weaker than its native ``/api/chat`` + ``format=<schema>`` path. Empty
strings and lightly-malformed JSON are common via the compat layer. We use the
native API so structured output is enforced server-side the way Ollama
documents it.

If you'd rather use the OpenAI compat path against Ollama, point the generic
``balanced-openai-compat`` worker at ``http://127.0.0.1:11434`` instead.

Env vars:

    OLLAMA_URL       Base URL, default ``http://127.0.0.1:11434``.
    OLLAMA_MODEL     Vision-capable model tag, e.g. ``llama3.2-vision`` or
                     ``minicpm-v``. Blank → auto-detected from ``/api/tags``.
    OLLAMA_API_KEY   Optional bearer token (for an auth-fronted / hosted Ollama).
    OLLAMA_TIMEOUT   Per-request timeout in seconds. Default 600.
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
    _safe_parse_vision_json,
    run_subclass,
)

load_dotenv()

logger = get_logger(__name__)


def _bare_json_schema() -> dict[str, Any]:
    """Strip the OpenAI ``response_format`` envelope, leaving the raw schema.

    Ollama's ``format`` field expects the schema directly (no ``type`` /
    ``json_schema`` wrapper). ``build_response_format()`` returns the wrapped
    OpenAI shape; pull out just the inner ``schema`` so we can pass it to
    Ollama unchanged when the upstream schema evolves.
    """
    wrapped = build_response_format()
    return wrapped["json_schema"]["schema"]


class BalancedOllamaWorker(BaseVisionWorker):
    name = "balanced-ollama"
    parallel_workers = 4
    request_timeout = 600

    def __init__(self) -> None:
        super().__init__()
        self.base_url: str = os.environ.get(
            "OLLAMA_URL", "http://127.0.0.1:11434"
        ).rstrip("/")
        self.model: str = os.environ.get("OLLAMA_MODEL", "").strip()
        self.api_key: str = os.environ.get("OLLAMA_API_KEY", "").strip()
        try:
            self.request_timeout = int(
                os.environ.get("OLLAMA_TIMEOUT", str(self.request_timeout))
            )
        except ValueError:
            pass  # non-numeric override — keep the class default

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def setup(self) -> None:
        if not self.model:
            self.model = self._autodetect_model()
        if not self.model:
            raise SystemExit(
                "OLLAMA_MODEL is unset and no model could be auto-detected from "
                f"{self.base_url}/api/tags. Pull a vision-capable model (e.g. "
                "`ollama pull llama3.2-vision`) and set OLLAMA_MODEL to its tag."
            )

    def _autodetect_model(self) -> str:
        """Use the first model the server lists when none is configured."""
        try:
            r = requests.get(f"{self.base_url}/api/tags",
                             headers=self._headers(), timeout=10)
            r.raise_for_status()
            models = r.json().get("models") or []
        except (requests.RequestException, ValueError, KeyError):
            return ""
        names = [m.get("name", "") for m in models if isinstance(m, dict) and m.get("name")]
        if not names:
            return ""
        logger.info("auto-detected model %r from %s/api/tags", names[0], self.base_url)
        return names[0]

    def banner(self) -> None:
        logger.info("=== Vision Worker (Ollama %s) ===", self.name)
        logger.info("  URL:     %s", self.base_url)
        logger.info("  Model:   %s", self.model)
        logger.info("  Workers: %d", self.parallel_workers)
        logger.info("  Auth:    %s", "bearer token set" if self.api_key else "(none)")

    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        # Ollama's chat API expects images as a sibling field on the message,
        # not as image_url content blocks. The `format` field takes the raw
        # JSON schema (no OpenAI-style envelope).
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt_instruction,
                            "images": [b64_jpeg],
                        }
                    ],
                    "stream": False,
                    "format": _bare_json_schema(),
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2000,
                    },
                },
                timeout=self.request_timeout,
            )
        except requests.RequestException as exc:
            logger.warning("Ollama request failed: %s", exc)
            return None

        if response.status_code != 200:
            preview = response.text[:600].replace("\n", " ")
            logger.warning(
                "Ollama HTTP %d (model=%s url=%s): %s",
                response.status_code, self.model, self.base_url, preview,
            )
            return None

        try:
            raw = response.json()["message"]["content"]
        except (KeyError, ValueError) as exc:
            logger.warning("Ollama: unexpected response shape: %s", exc)
            return None

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
    run_subclass(BalancedOllamaWorker)


if __name__ == "__main__":
    run_vision_worker()
