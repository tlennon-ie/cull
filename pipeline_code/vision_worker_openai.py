#!/usr/bin/env python3
"""OpenAI cloud vision worker — GPT vision via ``/v1/chat/completions``.

Mirrors the Groq worker's shape: it reads its API key from the environment,
builds the shared classification prompt + strict JSON schema, sends the image
as a base64 data-URL, and forces structured output via
``response_format={"type": "json_schema", ...}``.

This module also exposes ``call_openai_compatible`` / ``OpenAICompatibleClient``
so the OpenRouter worker (which speaks the identical wire protocol) can reuse
the exact same call path without duplicating it.

Env vars:

    OPENAI_API_KEY        Required. Your OpenAI secret key.
    OPENAI_VISION_MODEL   Model id. Default ``gpt-4o`` (env-overridable).
    OPENAI_BASE_URL       API base. Default ``https://api.openai.com/v1``.
    OPENAI_TIMEOUT        Per-request timeout (seconds). Default 600.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Allow direct invocation as a script.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

import cost_tracker
import vision_model_catalog as catalog
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

# Current OpenAI vision-capable default. Overridable via OPENAI_VISION_MODEL.
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _import_openai() -> Any:
    """Import the ``openai`` SDK lazily so this module imports cleanly when the
    SDK is not installed. Raises a clear ``SystemExit`` only when a worker
    actually tries to run."""
    try:
        import openai  # noqa: WPS433 - intentional lazy import
    except ImportError as exc:  # pragma: no cover - exercised via SystemExit path
        raise SystemExit(
            "openai SDK is not installed. Install pipeline dependencies first:\n"
            "    pip install -r requirements.txt\n"
            "(or `pip install openai` for just this worker)."
        ) from exc
    return openai


class OpenAICompatibleClient:
    """Thin wrapper over the ``openai`` SDK's chat-completions surface.

    Both OpenAI and OpenRouter expose the same OpenAI ``/v1`` protocol, so the
    OpenRouter worker constructs one of these with a different ``base_url`` /
    key and reuses ``classify`` verbatim.

    Splits the request into a ``system`` message (the long, stable classification
    prompt) and a ``user`` message (the image + a tiny prompt reference) so
    OpenAI's automatic prefix cache — which hashes the message-array prefix —
    can hit on every image while the giant instruction block flows through as
    cached tokens. See https://platform.openai.com/docs/guides/prompt-caching.
    OpenRouter proxies the same caching where the underlying model supports it.
    """

    def __init__(self, *, api_key: str, base_url: str, model: str,
                 timeout: int, provider: str = "openai") -> None:
        openai = _import_openai()
        self.model = model
        self.timeout = timeout
        self.provider = provider
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def classify(self, b64_jpeg: str, prompt_instruction: str) -> dict[str, Any] | None:
        """Send one image + instruction; return the parsed schema dict or None."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    # System = the long, stable rubric → maximises prefix cache
                    # hits across calls (OpenAI caches the prefix automatically
                    # for prompts >1024 tokens within a ~5-10min window).
                    {"role": "system", "content": prompt_instruction},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": "Classify the following image per the system instructions."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_jpeg}",
                                },
                            },
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format=build_response_format(),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - provider raises many error types
            logger.warning("OpenAI-compatible request failed: %s", str(exc)[:200])
            return None

        try:
            message = response.choices[0].message
        except (AttributeError, IndexError) as exc:
            logger.warning("OpenAI-compatible: unexpected response shape: %s", exc)
            return None

        # Accumulate token usage into the shared spend ledger. Best-effort:
        # a missing usage field or a ledger fault silently drops the row rather
        # than blocking the classification result the worker just paid for.
        in_tok, out_tok = cost_tracker.extract_openai_usage(response)
        if in_tok or out_tok:
            cost_tracker.record_usage(
                provider=self.provider, model=self.model,
                input_tokens=in_tok, output_tokens=out_tok,
            )

        msg_dict = (
            message.model_dump() if hasattr(message, "model_dump") else dict(message)
        )
        raw = extract_message_text(msg_dict)
        parsed = _safe_parse_vision_json(raw)
        if parsed is None:
            preview = (raw or "")[:300].replace("\n", " ")
            logger.warning("empty/invalid JSON from %s; raw=%r", self.model, preview)
            return None
        return parsed


def _resolve_timeout(env_key: str, default: int = 600) -> int:
    try:
        return int(os.environ.get(env_key, str(default)))
    except ValueError:
        return default


class OpenAIVisionWorker(BaseVisionWorker):
    name = "openai"
    parallel_workers = 4
    max_image_size = 1024

    def __init__(self) -> None:
        super().__init__()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "No OPENAI_API_KEY configured. Set it in .env or via the "
                "dashboard Settings tab."
            )
        self.base_url: str = os.environ.get(
            "OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL
        ).strip() or DEFAULT_OPENAI_BASE_URL
        # When no model is configured, fetch the provider's catalogue and pick a
        # vision-capable id; only fall back to the frozen default as a last resort.
        model_env = os.environ.get("OPENAI_VISION_MODEL", "").strip()
        self.model_id: str = model_env or catalog.autodetect_model(
            "openai", base_url=self.base_url, api_key=api_key,
            fallback=DEFAULT_OPENAI_MODEL,
        )
        self._client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=self.base_url,
            model=self.model_id,
            timeout=_resolve_timeout("OPENAI_TIMEOUT"),
        )

    def banner(self) -> None:
        logger.info("=== Vision Worker (OpenAI %s) ===", self.name)
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
    run_subclass(OpenAIVisionWorker)


if __name__ == "__main__":
    run_vision_worker()
