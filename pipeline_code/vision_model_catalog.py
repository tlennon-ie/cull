#!/usr/bin/env python3
"""vision_model_catalog.py — best-effort model discovery for vision providers.

Cloud vision workers should *fetch* the models a provider currently exposes
rather than ship a frozen ``gpt-4o`` / ``claude-3-5-sonnet`` default that rots.
This module is the single source of truth for "what models can provider X serve?"
and is shared by:

  * the four cloud workers (``vision_worker_openai`` / ``_anthropic`` /
    ``_gemini`` / ``_openrouter``) — to autodetect a model when their
    ``*_VISION_MODEL`` env is unset, and
  * the dashboard ``/api/vision/test`` endpoint — to populate a per-provider
    model dropdown (mirroring the local fleet's model picker).

Contract
--------
``list_models(provider, base_url=None, api_key=None, timeout=10) -> list[str]``

  * Returns a flat list of model-id strings (possibly empty).
  * **Best-effort: NEVER raises.** Any error (network, auth, bad JSON, missing
    key, unknown provider, non-http URL) yields ``[]``.

Per-provider wire shapes
------------------------
  openai / openrouter / lmstudio / llamacpp / openai-compat
        GET ``{base or default}/v1/models`` with ``Authorization: Bearer <key>``
        -> ``data[].id``.
  ollama
        GET ``{base}/api/tags`` -> ``models[].name``.
  anthropic
        GET ``https://api.anthropic.com/v1/models`` with ``x-api-key`` +
        ``anthropic-version`` -> ``data[].id`` (requires a key).
  gemini
        Prefer the ``google-genai`` SDK ``client.models.list()`` (gated import);
        fall back to GET ``…/v1beta/models?key=<key>`` -> ``models[].name``
        filtered to ids that advertise ``generateContent`` (requires a key).

Security
--------
All REST probes use ``allow_redirects=False`` and reject any non-``http(s)``
base URL, so a hostile endpoint cannot 302-bounce the probe at a cloud
metadata address (169.254.169.254) the operator never typed. This mirrors the
guard already used by ``scraper_test`` and the dashboard vision probe.
"""
from __future__ import annotations

import re
from typing import Any

import requests

from pipeline_logging import get_logger

logger = get_logger(__name__)

# Per-request timeout default (seconds). Short — this is an interactive probe.
DEFAULT_TIMEOUT = 10

# Default API bases for the OpenAI-compatible cloud providers (used when the
# caller passes no base_url). Local providers (lmstudio/llamacpp/openai-compat)
# have no universal default endpoint, so a base_url is required for them.
_OPENAI_COMPAT_DEFAULT_BASE = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Providers that speak the OpenAI ``/v1/models`` protocol.
_OPENAI_COMPAT_PROVIDERS = frozenset(
    {"openai", "openrouter", "lmstudio", "llamacpp", "openai-compat", "openai_compat"}
)

_ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
# Anthropic requires a dated version header; this is a stable, public value.
_ANTHROPIC_VERSION = "2023-06-01"

_GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Substrings hinting a listed model is vision-capable — used to choose sensibly
# when a worker leaves its model blank. Mirrors the local fleet's heuristic in
# ``vision_worker_balanced_openai._VISION_HINTS`` plus the common cloud ids.
_VISION_HINTS = (
    "vl", "vision", "llava", "moondream", "minicpm-v", "pixtral",
    "gemma-3", "qwen-vl", "internvl", "smolvlm", "phi-3.5-vision",
    "gpt-4o", "gpt-4.1", "gpt-5", "o3", "o4",
    "claude-3", "claude-4", "claude-opus", "claude-sonnet", "claude-haiku",
    "gemini",
)


def _is_http_url(url: str) -> bool:
    """True only for an ``http(s)://`` URL — blocks ``file://`` and friends."""
    return bool(re.match(r"^https?://", url.strip(), re.IGNORECASE))


def _models_base(base_url: str) -> str:
    """Normalise a base URL so it ends in exactly ``/v1`` (no trailing slash).

    Tolerates ``http://h:1234``, ``http://h:1234/``, ``http://h:1234/v1`` and
    ``http://h:1234/v1/`` — all collapse to ``http://h:1234/v1`` so the caller
    can safely append ``/models``.
    """
    trimmed = base_url.strip().rstrip("/")
    if trimmed.lower().endswith("/v1"):
        return trimmed
    return f"{trimmed}/v1"


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = (api_key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _safe_get(url: str, *, headers: dict[str, str], timeout: int) -> Any | None:
    """GET ``url`` and return the decoded JSON, or ``None`` on any failure.

    Never raises. Enforces ``allow_redirects=False`` and an http(s) scheme.
    """
    if not _is_http_url(url):
        logger.debug("refusing non-http(s) model probe: %r", url)
        return None
    try:
        resp = requests.get(
            url, headers=headers, timeout=timeout, allow_redirects=False
        )
    except requests.RequestException as exc:
        logger.debug("model probe failed for %s: %s", url, exc)
        return None
    if resp.status_code != 200:
        logger.debug("model probe HTTP %d for %s", resp.status_code, url)
        return None
    try:
        return resp.json()
    except ValueError as exc:  # non-JSON 200 body
        logger.debug("model probe non-JSON body from %s: %s", url, exc)
        return None


def _ids_from_data(payload: Any) -> list[str]:
    """Extract ``data[].id`` strings from an OpenAI/Anthropic-style payload."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [
        m["id"]
        for m in data
        if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"].strip()
    ]


# ── per-provider helpers ─────────────────────────────────────────────────────

def _list_openai_compatible(
    provider: str, base_url: str | None, api_key: str | None, timeout: int
) -> list[str]:
    base = (base_url or "").strip() or _OPENAI_COMPAT_DEFAULT_BASE.get(provider, "")
    if not base:
        logger.debug("no base_url for provider %r and no default", provider)
        return []
    url = f"{_models_base(base)}/models"
    payload = _safe_get(url, headers=_auth_headers(api_key), timeout=timeout)
    return _ids_from_data(payload)


def _list_ollama(
    base_url: str | None, api_key: str | None, timeout: int
) -> list[str]:
    base = (base_url or "").strip()
    if not base:
        return []
    url = f"{base.rstrip('/')}/api/tags"
    payload = _safe_get(url, headers=_auth_headers(api_key), timeout=timeout)
    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    return [
        m["name"]
        for m in models
        if isinstance(m, dict) and isinstance(m.get("name"), str) and m["name"].strip()
    ]


def _list_anthropic(api_key: str | None, timeout: int) -> list[str]:
    key = (api_key or "").strip()
    if not key:
        # Anthropic's /v1/models always requires authentication.
        return []
    headers = {"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION}
    payload = _safe_get(_ANTHROPIC_MODELS_URL, headers=headers, timeout=timeout)
    return _ids_from_data(payload)


def _gemini_id(name: str) -> str:
    """Strip the ``models/`` prefix Gemini prepends to every model name."""
    return name[len("models/"):] if name.startswith("models/") else name


def _list_gemini_sdk(api_key: str) -> list[str] | None:
    """Try the ``google-genai`` SDK. Returns ``None`` if it is unavailable so the
    caller falls back to REST; returns a (possibly empty) list on success."""
    try:
        from google import genai  # noqa: WPS433 - gated optional import
    except ImportError:
        return None
    try:
        client = genai.Client(api_key=api_key)
        out: list[str] = []
        for model in client.models.list():
            name = getattr(model, "name", "") or ""
            if not name:
                continue
            methods = (
                getattr(model, "supported_actions", None)
                or getattr(model, "supported_generation_methods", None)
                or []
            )
            # If the SDK reports supported methods, require generateContent;
            # otherwise keep the model (older SDKs omit the field).
            if methods and "generateContent" not in methods:
                continue
            out.append(_gemini_id(name))
        return out
    except Exception as exc:  # noqa: BLE001 - SDK raises many error types
        logger.debug("gemini SDK models.list() failed: %s", exc)
        return None


def _list_gemini_rest(api_key: str, timeout: int) -> list[str]:
    url = f"{_GEMINI_REST_BASE}/models?key={api_key}"
    payload = _safe_get(url, headers={}, timeout=timeout)
    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    out: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        methods = model.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        out.append(_gemini_id(name))
    return out


def _list_gemini(api_key: str | None, timeout: int) -> list[str]:
    key = (api_key or "").strip()
    if not key:
        return []
    via_sdk = _list_gemini_sdk(key)
    if via_sdk is not None:
        return via_sdk
    return _list_gemini_rest(key, timeout)


# ── public dispatch ──────────────────────────────────────────────────────────

def list_models(
    provider: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[str]:
    """Return the model ids ``provider`` currently exposes (best-effort).

    Args:
        provider: one of ``openai``, ``openrouter``, ``lmstudio``, ``llamacpp``,
            ``openai-compat``, ``ollama``, ``anthropic``, ``gemini``
            (case-insensitive).
        base_url: API base. Optional for openai/openrouter (sensible defaults);
            required for local providers (lmstudio/llamacpp/openai-compat) and
            ollama. Ignored for anthropic/gemini (fixed cloud endpoints).
        api_key: bearer / API key. Required for anthropic and gemini; optional
            elsewhere (local servers may not need auth).
        timeout: per-request timeout in seconds.

    Returns:
        A flat ``list[str]`` of model ids — empty on any error. Never raises.
    """
    key = (provider or "").strip().lower()
    try:
        if key in _OPENAI_COMPAT_PROVIDERS:
            return _list_openai_compatible(key, base_url, api_key, timeout)
        if key == "ollama":
            return _list_ollama(base_url, api_key, timeout)
        if key == "anthropic":
            return _list_anthropic(api_key, timeout)
        if key == "gemini":
            return _list_gemini(api_key, timeout)
        logger.debug("list_models: unknown provider %r", provider)
        return []
    except Exception as exc:  # noqa: BLE001 - belt-and-braces: never raise
        logger.debug("list_models(%r) unexpected error: %s", provider, exc)
        return []


def pick_vision_model(model_ids: list[str]) -> str | None:
    """Choose the most vision-capable-looking id from ``model_ids``.

    Prefers an id matching a known vision hint; otherwise returns the first id.
    Returns ``None`` for an empty list.
    """
    if not model_ids:
        return None
    for model_id in model_ids:
        low = model_id.lower()
        if any(hint in low for hint in _VISION_HINTS):
            return model_id
    return model_ids[0]


def autodetect_model(
    provider: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    fallback: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Best-effort: list ``provider`` models and pick a vision-capable id.

    Returns the chosen id, or ``fallback`` when discovery yields nothing. This
    is the helper the cloud workers call when their ``*_VISION_MODEL`` env is
    empty, so a frozen default is only a last resort.
    """
    chosen = pick_vision_model(
        list_models(provider, base_url=base_url, api_key=api_key, timeout=timeout)
    )
    if chosen:
        logger.info("autodetected %s vision model %r", provider, chosen)
        return chosen
    logger.info("no %s models discovered; falling back to %r", provider, fallback)
    return fallback


__all__ = [
    "list_models",
    "pick_vision_model",
    "autodetect_model",
    "DEFAULT_TIMEOUT",
]
