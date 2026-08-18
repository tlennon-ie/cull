"""Cloud vision-worker spend tracking + hourly-threshold alerting.

Cloud vision workers (Groq / OpenAI / OpenRouter / Anthropic / Gemini) burn
real money per image. Historically we logged neither token counts nor a
running dollar estimate, so an admin could only find out about a spend spike
by looking at the provider's own billing dashboard hours later.

This module is a tiny, thread-safe, per-process ledger:

    record_usage(
        provider="openai", model="gpt-4o",
        input_tokens=1500, output_tokens=250,
    )

Each call updates ``data/cost_ledger.json`` atomically (temp file + os.replace),
credits the currently active job (``PIPELINE_SLUG``), and, when the last hour's
cost has crossed ``COST_ALERT_HOURLY_USD`` (default 5.0), emits ONE warning
line per calendar-hour window (so a hot loop doesn't spam the log).

Dollar cost is a *conservative estimate* derived from a small hard-coded
price table (``_PRICES``, USD per 1M tokens). Missing models are priced at
zero so unknown-model calls still land in the token log without crashing.
The estimate is intentionally coarse — it is a smoke alarm, not accounting.

Zero mandatory deps and NEVER raises: every entry point catches its own IO /
validation errors and no-ops on failure, so a ledger fault can never break
classification. Fully backward-compatible: without any call sites wired up
the file simply stays empty.

Concurrency caveat
------------------
The ``threading.Lock`` here only serialises writers WITHIN one process. Cull's
supervisor spawns one subprocess per vision worker, so parallel workers on the
same box are separate processes and CAN interleave read-modify-write cycles
under load — last writer wins, and each process can emit its own hourly-spend
warning. That's acceptable given the module's positioning as a coarse smoke
alarm (see paragraph above); adding an interprocess file lock would trade
that intentional simplicity for a mandatory dep or a per-OS filelock branch.
If exact multi-process accounting ever matters, swap ``_write`` to use a
proper filelock (e.g. the ``filelock`` package) and drop this caveat.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from paths import base_dir
from pipeline_logging import get_logger

logger = get_logger(__name__)

LEDGER_FILENAME = "cost_ledger.json"
DEFAULT_ALERT_HOURLY_USD = 5.0

# USD per 1_000_000 tokens. Deliberately coarse and easy to edit. Keys use the
# lowered model-id prefix (startswith match), so a version bump like
# "gpt-4o-2024-11-20" matches "gpt-4o". Missing → priced at zero.
_PRICES: dict[str, tuple[float, float]] = {
    # (input_per_million, output_per_million)
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    # Anthropic
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-4-sonnet": (3.00, 15.00),
    # Gemini
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.075, 0.30),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    # Groq (llama-4-scout / other multimodal open models)
    "llama-4-scout": (0.11, 0.34),
    "llama-4-maverick": (0.20, 0.60),
    "meta-llama/llama-4": (0.11, 0.34),
}

_lock = threading.Lock()
_last_alert_hour: dict[str, int] = {}  # key = "<slug>" → hour bucket already alerted


def _ledger_path() -> Path:
    override = os.environ.get("COST_LEDGER_PATH", "").strip()
    if override:
        return Path(override)
    return base_dir() / LEDGER_FILENAME


def _alert_threshold_usd() -> float:
    raw = os.environ.get("COST_ALERT_HOURLY_USD", "").strip()
    if not raw:
        return DEFAULT_ALERT_HOURLY_USD
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_ALERT_HOURLY_USD


def _active_slug() -> str:
    return (os.environ.get("PIPELINE_SLUG", "") or "unknown").strip() or "unknown"


def _price_for(model: str) -> tuple[float, float]:
    """Return ``(input_per_million, output_per_million)`` for ``model``.

    Longest-prefix wins so ``gpt-4o-2024-11-20`` matches ``gpt-4o`` (not
    ``gpt-4o-mini``). Unknown models get zero pricing — the token count still
    lands in the ledger.
    """
    key = (model or "").lower().strip()
    if not key:
        return (0.0, 0.0)
    for prefix in sorted(_PRICES.keys(), key=len, reverse=True):
        if key.startswith(prefix):
            return _PRICES[prefix]
    return (0.0, 0.0)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Coarse USD estimate. Returns 0.0 for unknown models."""
    in_price, out_price = _price_for(model)
    return round(
        (max(0, int(input_tokens)) * in_price
         + max(0, int(output_tokens)) * out_price) / 1_000_000.0,
        6,
    )


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _hourly_sum(entry: dict[str, Any], now: float) -> float:
    """Sum recent (last 3600s) ``cost_usd`` samples in ``entry['recent']``."""
    cutoff = now - 3600.0
    recent = entry.get("recent") or []
    return round(sum(float(r.get("cost_usd", 0.0)) for r in recent if r.get("ts", 0.0) >= cutoff), 4)


def _prune_recent(entry: dict[str, Any], now: float) -> None:
    """Drop recent samples older than 2h so the file stays small."""
    cutoff = now - 7200.0
    entry["recent"] = [r for r in (entry.get("recent") or []) if r.get("ts", 0.0) >= cutoff]


def record_usage(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    slug: str | None = None,
) -> float:
    """Credit usage to the current (or given) job. Returns estimated USD.

    Never raises. On any IO / parse failure the update is silently dropped —
    the classification path must not be blocked by a ledger fault.
    """
    try:
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
        job_slug = (slug or _active_slug()).strip() or "unknown"
        now = time.time()
        path = _ledger_path()
        with _lock:
            data = _load(path)
            entry = data.get(job_slug) or {}
            entry["provider"] = provider
            entry["model"] = model
            entry["input_tokens"] = int(entry.get("input_tokens") or 0) + max(0, int(input_tokens))
            entry["output_tokens"] = int(entry.get("output_tokens") or 0) + max(0, int(output_tokens))
            entry["total_cost_usd"] = round(float(entry.get("total_cost_usd") or 0.0) + cost, 6)
            entry["last_call_ts"] = now
            entry["call_count"] = int(entry.get("call_count") or 0) + 1
            recent = entry.get("recent") or []
            recent.append({"ts": now, "cost_usd": cost})
            entry["recent"] = recent
            _prune_recent(entry, now)
            hourly = _hourly_sum(entry, now)
            entry["hourly_cost_usd"] = hourly
            data[job_slug] = entry
            _atomic_write(path, data)
            _maybe_alert(job_slug, hourly, now)
        return cost
    except Exception as exc:  # noqa: BLE001 - ledger failure never breaks classification
        logger.debug("cost_tracker: silent failure recording usage: %s", exc)
        return 0.0


def _maybe_alert(slug: str, hourly_usd: float, now: float) -> None:
    threshold = _alert_threshold_usd()
    if threshold <= 0 or hourly_usd < threshold:
        return
    hour_bucket = int(now // 3600)
    if _last_alert_hour.get(slug) == hour_bucket:
        return
    _last_alert_hour[slug] = hour_bucket
    logger.warning(
        "cost alert: job %r spent ~$%.2f in the last hour (threshold $%.2f). "
        "Adjust COST_ALERT_HOURLY_USD or switch to a cheaper provider.",
        slug, hourly_usd, threshold,
    )


def extract_openai_usage(response: Any) -> tuple[int, int]:
    """Pull ``(input_tokens, output_tokens)`` out of an OpenAI-shaped response.

    Handles the openai / groq SDK object shape (``response.usage.prompt_tokens``)
    AND the raw-dict shape returned by ``response.json()`` / requests. Returns
    ``(0, 0)`` when the field is missing so a provider that omits usage still
    logs a zero-cost row rather than crashing.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return (0, 0)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "prompt_tokens"):
            usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            }
        if not isinstance(usage, dict):
            return (0, 0)
        return (
            int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        )
    except Exception:  # noqa: BLE001 - never let usage extraction break the caller
        return (0, 0)


def extract_anthropic_usage(response: Any) -> tuple[int, int]:
    """Pull ``(input_tokens, output_tokens)`` out of an Anthropic response.

    Anthropic exposes ``response.usage.input_tokens`` /
    ``response.usage.output_tokens``. Cache hits count against the same total
    (``cache_read_input_tokens`` is included in ``input_tokens`` when caching
    is used, per the Anthropic docs).
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return (0, 0)
        return (
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )
    except Exception:  # noqa: BLE001
        return (0, 0)


def summary(slug: str | None = None) -> dict[str, Any]:
    """Read-only view of the ledger. Useful for dashboards / tests."""
    data = _load(_ledger_path())
    if slug is None:
        return data
    return data.get(slug) or {}


__all__ = [
    "record_usage",
    "estimate_cost_usd",
    "extract_openai_usage",
    "extract_anthropic_usage",
    "summary",
    "LEDGER_FILENAME",
    "DEFAULT_ALERT_HOURLY_USD",
]
