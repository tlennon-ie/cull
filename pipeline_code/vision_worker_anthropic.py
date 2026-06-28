#!/usr/bin/env python3
"""Anthropic (Claude) cloud vision worker.

Claude's Messages API doesn't take an OpenAI-style ``response_format``; the
robust way to get strict structured output is forced tool-use. We expose a
single tool whose ``input_schema`` IS the pipeline's vision JSON schema, then
set ``tool_choice`` to force that exact tool. The model's answer arrives as the
``input`` of the ``tool_use`` content block, which is already a parsed dict —
no fragile text-JSON extraction needed.

Env vars:

    ANTHROPIC_API_KEY        Required. Your Anthropic key.
    ANTHROPIC_VISION_MODEL   Model id. Default ``claude-3-5-sonnet-latest``.
    ANTHROPIC_BASE_URL       Optional API base override (proxies / gateways).
    ANTHROPIC_TIMEOUT        Per-request timeout (seconds). Default 600.
    ANTHROPIC_MAX_TOKENS     Max output tokens. Default 2000.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Allow direct invocation as a script.
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

# Current Claude vision-capable default. Overridable via ANTHROPIC_VISION_MODEL.
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
_TOOL_NAME = "classify_image"


def _import_anthropic() -> Any:
    """Lazy import so the module loads without the SDK installed."""
    try:
        import anthropic  # noqa: WPS433 - intentional lazy import
    except ImportError as exc:  # pragma: no cover - exercised via SystemExit path
        raise SystemExit(
            "anthropic SDK is not installed. Install pipeline dependencies first:\n"
            "    pip install -r requirements.txt\n"
            "(or `pip install anthropic` for just this worker)."
        ) from exc
    return anthropic


def _classification_tool() -> dict[str, Any]:
    """The forced tool whose input_schema is the shared vision JSON schema."""
    schema = build_response_format()["json_schema"]["schema"]
    return {
        "name": _TOOL_NAME,
        "description": (
            "Record the structured classification of the supplied image. "
            "You MUST call this tool with every field populated."
        ),
        "input_schema": schema,
    }


def _extract_tool_input(content: Any) -> dict[str, Any] | None:
    """Pull the forced tool's ``input`` dict out of the Messages content blocks.

    Falls back to mining JSON out of a text block if (unexpectedly) the model
    answered in prose instead of via the tool."""
    if not content:
        return None
    text_fallback = ""
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type == "tool_use":
            data = getattr(block, "input", None)
            if data is None and isinstance(block, dict):
                data = block.get("input")
            if isinstance(data, dict):
                return data
        elif block_type == "text":
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            text_fallback += text or ""
    return _safe_parse_vision_json(text_fallback) if text_fallback else None


class AnthropicVisionWorker(BaseVisionWorker):
    name = "anthropic"
    parallel_workers = 4
    max_image_size = 1024

    def __init__(self) -> None:
        super().__init__()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "No ANTHROPIC_API_KEY configured. Set it in .env or via the "
                "dashboard Settings tab."
            )
        self.model_id: str = os.environ.get(
            "ANTHROPIC_VISION_MODEL", DEFAULT_ANTHROPIC_MODEL
        ).strip() or DEFAULT_ANTHROPIC_MODEL
        try:
            self.max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "2000"))
        except ValueError:
            self.max_tokens = 2000
        try:
            self.request_timeout = int(os.environ.get("ANTHROPIC_TIMEOUT", "600"))
        except ValueError:
            self.request_timeout = 600

        anthropic = _import_anthropic()
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": float(self.request_timeout),
        }
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**client_kwargs)
        self._tool = _classification_tool()

    def banner(self) -> None:
        logger.info("=== Vision Worker (Anthropic %s) ===", self.name)
        logger.info("  Model:   %s", self.model_id)
        logger.info("  Workers: %d", self.parallel_workers)

    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        try:
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=0.1,
                tools=[self._tool],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_instruction},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64_jpeg,
                                },
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001 - provider raises many error types
            logger.warning("Anthropic request failed: %s", str(exc)[:200])
            return None

        content = getattr(response, "content", None)
        parsed = _extract_tool_input(content)
        if parsed is None:
            logger.warning("Anthropic: no tool_use result in response for %s", self.model_id)
            return None
        return parsed


def run_vision_worker() -> None:
    run_subclass(AnthropicVisionWorker)


if __name__ == "__main__":
    run_vision_worker()
