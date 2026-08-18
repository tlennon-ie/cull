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
    OPENAI_COMPAT_PROVIDER  Optional provider hint. When ``llamacpp`` (or when
                            ``VISION_GRAMMAR_ENABLED`` is truthy) the request also
                            carries a GBNF ``grammar`` derived from the response
                            schema — llama.cpp's grammar-constrained decoding is
                            stricter than its json-schema mode. Off by default;
                            best-effort, falls back to ``response_format`` only.
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

# Substrings that mark a model as a reasoning/"thinking" variant. These emit
# chain-of-thought prose and, in LM Studio's reasoning split-mode, routinely burn
# the whole token budget thinking before they ever emit the JSON object — so the
# strict response_format yields "empty/invalid JSON". Auto-detect deprioritises
# them: a non-thinking vision model is always preferred when one is listed.
_THINKING_HINTS = ("thinking", "reasoning", "-r1", "qwq", "deepseek-r")


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
        # GBNF grammar gate (default OFF). Only llama.cpp's server understands the
        # ``grammar`` field, so we opt in when the provider is explicitly llamacpp
        # OR the operator flips VISION_GRAMMAR_ENABLED. Off → byte-identical request.
        provider = os.environ.get("OPENAI_COMPAT_PROVIDER", "").strip().lower()
        grammar_flag = os.environ.get("VISION_GRAMMAR_ENABLED", "false").strip().lower()
        self.use_grammar: bool = (
            provider == "llamacpp" or grammar_flag in ("1", "true", "yes", "on")
        )
        # Grammar is derived from the (process-stable) response schema, so we
        # build it once at worker start and reuse the string forever. Cache it
        # via a lazy sentinel so the first request pays the build cost, all
        # subsequent requests pay a dict-lookup: converts a 5-20ms grammar
        # generation into a ~50ns lookup on the hot per-image path.
        self._cached_grammar: str | None = None
        self._grammar_built: bool = False
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

        def _is_vision(i: str) -> bool:
            return any(h in i.lower() for h in _VISION_HINTS)

        def _is_thinking(i: str) -> bool:
            return any(h in i.lower() for h in _THINKING_HINTS)

        # Prefer a non-thinking vision model; fall back to any vision model (even a
        # thinking one) and finally the first listed id. Thinking models break
        # strict structured output, so they're the last vision resort, not the first.
        chosen = (
            next((i for i in ids if _is_vision(i) and not _is_thinking(i)), "")
            or next((i for i in ids if _is_vision(i)), "")
            or ids[0]
        )
        if _is_thinking(chosen):
            logger.warning(
                "auto-detected model %r looks like a reasoning/thinking model — it may "
                "emit chain-of-thought instead of JSON. Set an explicit non-thinking "
                "vision model in the fleet if classification returns empty JSON.", chosen,
            )
        logger.info("auto-detected model %r from %s/v1/models", chosen, self.base_url)
        return chosen

    def banner(self) -> None:
        logger.info("=== Vision Worker (OpenAI-compatible %s) ===", self.name)
        logger.info("  URL:     %s", self.base_url)
        logger.info("  Model:   %s", self.model)
        logger.info("  Workers: %d", self.parallel_workers)
        logger.info("  Auth:    %s", "bearer token set" if self.api_key else "(none)")

    def _grammar(self) -> str | None:
        """Best-effort GBNF grammar for the response schema, or None.

        Returns None when the gate is off OR the optional ``gbnf_grammar`` module
        is unavailable OR conversion fails — so the request silently falls back to
        the plain ``response_format`` path and never hard-fails on schema drift.
        Lazily imported so the worker has no extra import cost when the gate is off.

        Memoised per-instance: the grammar depends only on the (process-stable)
        response schema, so we build it once and reuse forever. First call:
        ~5-20ms grammar compile; subsequent calls: attribute lookup.
        """
        if not self.use_grammar:
            return None
        if self._grammar_built:
            return self._cached_grammar
        try:
            import gbnf_grammar
            schema = build_response_format()["json_schema"]["schema"]
            grammar = gbnf_grammar.json_schema_to_gbnf(schema)
            self._cached_grammar = grammar or None
        except Exception as exc:  # noqa: BLE001 - grammar is an optional optimisation
            logger.warning("GBNF grammar build failed; using response_format: %s", exc)
            self._cached_grammar = None
        self._grammar_built = True
        return self._cached_grammar

    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: dict[str, Any] = {
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
        }
        # llama.cpp only: attach the GBNF grammar (kept alongside response_format
        # so non-grammar-aware servers still honour the schema envelope). No-op
        # when the gate is off — the body is then byte-identical to before.
        grammar = self._grammar()
        if grammar is not None:
            body["grammar"] = grammar

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=body,
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
