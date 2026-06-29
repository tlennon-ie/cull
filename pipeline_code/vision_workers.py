"""Registry of vision-worker names → script + env-override.

The supervisor (run_pipeline.py) uses this to spawn workers; the dashboard
uses it to render the provider dropdown. Keep additions here in lockstep
with `dashboard_enhanced.ALLOWED_VISION_WORKERS`.

Adding a new worker:
  1. Drop a new ``vision_worker_<name>.py`` in this folder. It must import
     ``from categories import CATEGORIES`` and use ``vision_prompt`` for the
     classification prompt + JSON schema (see vision_worker_balanced_lm.py
     for the canonical pattern).
  2. Append its name + script to ``WORKERS`` below. If it inherits env
     from another worker (like ``balanced-lm-secondary`` does), put the
     overrides in the entry's ``env_override`` factory.
  3. Add its label + description to ``dashboard_enhanced.ALLOWED_VISION_WORKERS``
     and ``_VISION_WORKER_DESCRIPTIONS`` so it appears in the UI dropdown.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class WorkerSpec:
    """Static description of one vision worker."""
    name: str
    script: str
    description: str
    # Returns a dict of env-var overrides applied when spawning this worker.
    # Most workers have none; the secondary LM Studio variant uses it to
    # rebind PRIMARY_* to SECONDARY_*.
    env_override: Callable[[], dict[str, str]] = field(
        default_factory=lambda: (lambda: {})
    )


def _secondary_lm_env() -> dict[str, str]:
    """Map LMSTUDIO_SECONDARY_* onto the worker's PRIMARY_* names so the
    same script can run twice in parallel against two LM Studio hosts."""
    return {
        "LMSTUDIO_PRIMARY_URL":     os.environ.get("LMSTUDIO_SECONDARY_URL", ""),
        "LMSTUDIO_PRIMARY_MODEL":   os.environ.get("LMSTUDIO_SECONDARY_MODEL", ""),
        "LMSTUDIO_PRIMARY_TIMEOUT": os.environ.get("LMSTUDIO_SECONDARY_TIMEOUT", "60"),
    }


WORKERS: dict[str, WorkerSpec] = {
    "balanced-groq": WorkerSpec(
        name="balanced-groq",
        script="vision_worker_balanced_groq.py",
        description="Groq cloud (llama-4-scout) - fast, paid",
    ),
    "balanced-lm": WorkerSpec(
        name="balanced-lm",
        script="vision_worker_balanced_lm.py",
        description="LM Studio primary endpoint (local)",
    ),
    "balanced-lm-secondary": WorkerSpec(
        name="balanced-lm-secondary",
        script="vision_worker_balanced_lm.py",  # same script, different env
        description="LM Studio secondary endpoint - parallel to primary",
        env_override=_secondary_lm_env,
    ),
    "lm-autodetect": WorkerSpec(
        name="lm-autodetect",
        script="vision_worker_lm_autodetect.py",
        description="LM Studio - auto-picks whichever VL model is loaded",
    ),
    "balanced-openai-compat": WorkerSpec(
        name="balanced-openai-compat",
        script="vision_worker_balanced_openai.py",
        description=(
            "Generic OpenAI-compatible local server (llama.cpp / vLLM / "
            "koboldcpp / LocalAI / text-generation-webui). Reads "
            "OPENAI_COMPAT_URL + OPENAI_COMPAT_MODEL."
        ),
    ),
    "balanced-ollama": WorkerSpec(
        name="balanced-ollama",
        script="vision_worker_balanced_ollama.py",
        description=(
            "Ollama via native /api/chat (more reliable JSON-schema output "
            "than its OpenAI compat layer). Reads OLLAMA_URL + OLLAMA_MODEL."
        ),
    ),
    "local": WorkerSpec(
        name="local",
        script="vision_worker.py",
        description="Single-thread LM Studio fallback (debug only)",
    ),
    # ── Cloud workers (registry pattern, like balanced-groq) ──────────────
    "openai": WorkerSpec(
        name="openai",
        script="vision_worker_openai.py",
        description=(
            "OpenAI cloud (GPT vision) - structured output via "
            "response_format json_schema. Reads OPENAI_API_KEY + "
            "OPENAI_VISION_MODEL."
        ),
    ),
    "anthropic": WorkerSpec(
        name="anthropic",
        script="vision_worker_anthropic.py",
        description=(
            "Anthropic Claude vision - structured output via forced tool-use. "
            "Reads ANTHROPIC_API_KEY + ANTHROPIC_VISION_MODEL."
        ),
    ),
    "gemini": WorkerSpec(
        name="gemini",
        script="vision_worker_gemini.py",
        description=(
            "Google Gemini vision (new google-genai SDK) - structured output "
            "via response_schema. Reads GEMINI_API_KEY / GOOGLE_API_KEY + "
            "GEMINI_VISION_MODEL."
        ),
    ),
    "openrouter": WorkerSpec(
        name="openrouter",
        script="vision_worker_openrouter.py",
        description=(
            "OpenRouter (OpenAI-compatible gateway to many vision models). "
            "Reads OPENROUTER_API_KEY + OPENROUTER_VISION_MODEL."
        ),
    ),
}


def known_workers() -> list[str]:
    """Return every worker name registered in this module."""
    return list(WORKERS.keys())


__all__ = ["WorkerSpec", "WORKERS", "known_workers"]
