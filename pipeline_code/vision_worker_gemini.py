#!/usr/bin/env python3
"""Gemini cloud vision worker — uses the NEW ``google.genai`` SDK.

IMPORTANT (CLAUDE.md hard rule): this worker uses ``google.genai`` (package
``google-genai``), NOT the deprecated ``google.generativeai``. Structured
output is enforced with ``response_mime_type="application/json"`` plus a
``response_schema`` derived from the pipeline's shared vision schema.

Gemini's ``response_schema`` accepts only a subset of JSON Schema, so the
pipeline schema is down-converted (``_gemini_schema``) to the supported
keywords (type / properties / required / enum / items / nullable / min/max
numbers). Validation gates are re-applied afterwards by ``apply_scores`` in the
base class, so dropping the string-length constraints here is harmless.

Env vars:

    GEMINI_API_KEY (or GOOGLE_API_KEY)   Required. Your Gemini / Google AI key.
    GEMINI_VISION_MODEL                  Model id. Default ``gemini-2.0-flash``.
    GEMINI_TIMEOUT                       Per-request timeout (seconds). Default 600.
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
from vision_worker_base import (
    BaseVisionWorker,
    build_response_format,
    _safe_parse_vision_json,
    run_subclass,
)

load_dotenv()

logger = get_logger(__name__)

# Current Gemini vision-capable default. Overridable via GEMINI_VISION_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

# JSON-Schema keys Gemini's response_schema understands. Everything else
# (additionalProperties, minLength/maxLength, etc.) is stripped.
_GEMINI_KEEP_KEYS = frozenset(
    {"type", "properties", "required", "items", "enum", "nullable",
     "minimum", "maximum", "description"}
)


def _import_genai() -> tuple[Any, Any]:
    """Lazy import so the module loads without the SDK installed. Returns
    ``(genai, genai.types)``."""
    try:
        from google import genai  # noqa: WPS433 - intentional lazy import
        from google.genai import types as genai_types  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - exercised via SystemExit path
        raise SystemExit(
            "google-genai SDK is not installed. Install pipeline dependencies "
            "first:\n    pip install -r requirements.txt\n"
            "(or `pip install google-genai` for just this worker). NOTE: this "
            "is the NEW SDK, not the deprecated `google-generativeai`."
        ) from exc
    return genai, genai_types


def _gemini_schema(node: Any) -> Any:
    """Recursively down-convert our JSON schema to Gemini's supported subset."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key not in _GEMINI_KEEP_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                out[key] = {k: _gemini_schema(v) for k, v in value.items()}
            elif key == "items":
                out[key] = _gemini_schema(value)
            else:
                out[key] = value
        return out
    return node


def _vision_response_schema() -> dict[str, Any]:
    """The pipeline's vision schema reduced to Gemini's response_schema dialect."""
    schema = build_response_format()["json_schema"]["schema"]
    return _gemini_schema(schema)


class GeminiVisionWorker(BaseVisionWorker):
    name = "gemini"
    parallel_workers = 4
    max_image_size = 1024

    def __init__(self) -> None:
        super().__init__()
        api_key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        if not api_key:
            raise SystemExit(
                "No GEMINI_API_KEY / GOOGLE_API_KEY configured. Set one in .env "
                "or via the dashboard Settings tab."
            )
        # Autodetect from Gemini's model list when none is configured; the
        # frozen default is only the last-resort fallback.
        model_env = os.environ.get("GEMINI_VISION_MODEL", "").strip()
        self.model_id: str = model_env or catalog.autodetect_model(
            "gemini", api_key=api_key, fallback=DEFAULT_GEMINI_MODEL,
        )

        genai, genai_types = _import_genai()
        self._genai_types = genai_types
        self._client = genai.Client(api_key=api_key)
        self._response_schema = _vision_response_schema()

    def banner(self) -> None:
        logger.info("=== Vision Worker (Gemini %s) ===", self.name)
        logger.info("  Model:   %s", self.model_id)
        logger.info("  Workers: %d", self.parallel_workers)

    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        import base64

        types = self._genai_types
        try:
            image_part = types.Part.from_bytes(
                data=base64.b64decode(b64_jpeg),
                mime_type="image/jpeg",
            )
            config = types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=self._response_schema,
            )
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=[prompt_instruction, image_part],
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - provider raises many error types
            logger.warning("Gemini request failed: %s", str(exc)[:200])
            return None

        raw = getattr(response, "text", None)
        parsed = _safe_parse_vision_json(raw)
        if parsed is None:
            preview = (raw or "")[:300].replace("\n", " ")
            logger.warning("empty/invalid JSON from Gemini %s; raw=%r",
                           self.model_id, preview)
            return None
        return parsed


def run_vision_worker() -> None:
    run_subclass(GeminiVisionWorker)


if __name__ == "__main__":
    run_vision_worker()
