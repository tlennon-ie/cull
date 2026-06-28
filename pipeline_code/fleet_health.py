"""fleet_health.py — health probing, failover ordering, and telemetry for the
local vision-worker fleet.

The fleet is the user-defined ``vision.workers`` list (see CLAUDE.md "Vision-
worker fleet" and ``job_config.clean_vision_fleet``), each instance a flat dict
``{id, name, provider, base_url, model, api_key}`` with ``provider`` in
``lmstudio | llamacpp | ollama``. This module gives the supervisor / dashboard
three pure-ish building blocks:

  * ``probe(endpoint, timeout)`` — a single cheap GET against the provider's
    model-list endpoint (``/v1/models`` for lmstudio/llamacpp, ``/api/tags`` for
    ollama). Returns ``{ok, latency_ms, error}``. Never raises. All HTTP is
    routed through the private ``_http_get`` helper so tests monkeypatch it and
    no real network is hit.

  * ``pick_healthy(fleet, probes)`` — orders the fleet healthy-first, then by
    ascending latency, dropping endpoints whose probe failed. Used for failover
    and naive load-balancing (front of the list = best next target).

  * ``Telemetry`` — a pure aggregator. Feed it per-request samples
    (``add(tokens, latency_ms, ok)``) and it reports tokens/sec, latency p50/p95
    (nearest-rank), error rate, and a queue-drain ETA. No IO, no clock — inject
    everything so it's deterministic in tests and cheap to call from a request
    hot-path.

Pure stdlib + ``requests``. No secrets, no runtime pip.
"""
from __future__ import annotations

import math
import time
from typing import Any

import requests

from pipeline_logging import get_logger

logger = get_logger(__name__)

# Provider → the cheap "list models" path used as a liveness probe. Mirrors the
# auto-detect endpoints the balanced workers already hit on connect.
_PROVIDER_PROBE_PATH: dict[str, str] = {
    "lmstudio": "/v1/models",
    "llamacpp": "/v1/models",
    "ollama": "/api/tags",
}

_DEFAULT_TIMEOUT = 5.0


# ── HTTP helper (monkeypatched in tests) ─────────────────────────────────────

def _http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> tuple[int, int]:
    """GET ``url``; return ``(status_code, latency_ms)``.

    Tests replace this so probing never touches the network. Production measures
    latency around the real ``requests.get`` call.
    """
    t0 = time.monotonic()
    resp = requests.get(url, headers=headers or {}, timeout=timeout)
    latency_ms = int((time.monotonic() - t0) * 1000)
    return resp.status_code, latency_ms


# ── probe ────────────────────────────────────────────────────────────────────

def _probe_result(ok: bool, latency_ms: int | None, error: str | None) -> dict[str, Any]:
    return {"ok": ok, "latency_ms": latency_ms, "error": error}


def probe(endpoint: dict[str, Any], timeout: float = _DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Liveness-probe one fleet ``endpoint``.

    Returns ``{ok, latency_ms, error}`` and never raises:

      * ``ok`` is True only for a 2xx response from the provider's model-list
        endpoint (a healthy server answers it quickly without loading a model).
      * ``latency_ms`` is the measured round-trip on a reachable server, else
        ``None`` (timeout / connection error / short-circuit).
      * ``error`` is ``None`` on success, else a short human-readable reason.

    A blank ``base_url`` or unknown ``provider`` short-circuits with no network.
    """
    provider = str(endpoint.get("provider", "") or "").strip().lower()
    base_url = str(endpoint.get("base_url", "") or "").strip().rstrip("/")
    path = _PROVIDER_PROBE_PATH.get(provider)
    if path is None:
        return _probe_result(False, None, f"unknown provider: {provider!r}")
    if not base_url:
        return _probe_result(False, None, "blank base_url")

    api_key = str(endpoint.get("api_key", "") or "").strip()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"{base_url}{path}"

    try:
        status, latency_ms = _http_get(url, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        return _probe_result(False, None, f"timed out after {timeout}s")
    except requests.exceptions.ConnectionError as exc:
        return _probe_result(False, None, f"connection error: {str(exc)[:120]}")
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        logger.debug("probe unexpected error for %s: %s", url, exc)
        return _probe_result(False, None, f"{type(exc).__name__}: {str(exc)[:120]}")

    if 200 <= status < 300:
        return _probe_result(True, max(0, int(latency_ms)), None)
    return _probe_result(False, max(0, int(latency_ms)), f"HTTP {status}")


def probe_fleet(fleet: list[dict[str, Any]], timeout: float = _DEFAULT_TIMEOUT) -> dict[str, dict]:
    """Probe every endpoint in ``fleet``; return ``{id: probe_result}``.

    Convenience wrapper so callers can do ``pick_healthy(fleet, probe_fleet(fleet))``.
    Sequential by design — a fleet is a handful of local endpoints, not a swarm.
    """
    out: dict[str, dict] = {}
    for ep in fleet:
        if isinstance(ep, dict) and ep.get("id"):
            out[str(ep["id"])] = probe(ep, timeout=timeout)
    return out


# ── pick_healthy ─────────────────────────────────────────────────────────────

# Sort key for a healthy endpoint with no measured latency: rank it behind every
# endpoint that DID report one, but ahead of unhealthy (dropped) endpoints.
_NO_LATENCY_RANK = math.inf


def pick_healthy(
    fleet: list[dict[str, Any]],
    probes: dict[str, dict],
) -> list[dict[str, Any]]:
    """Return the healthy subset of ``fleet`` ordered best-first.

    Ordering: healthy endpoints only (probe ``ok`` is True), ascending by
    ``latency_ms`` so the fastest live endpoint is first (good failover + naive
    load-balancing target). An endpoint with no probe entry, or a falsy/failed
    probe, is dropped. A healthy probe missing a latency sorts after those that
    have one. Stable for equal latencies (preserves fleet order).
    """
    healthy: list[tuple[float, int, dict]] = []
    for idx, ep in enumerate(fleet):
        if not isinstance(ep, dict):
            continue
        eid = str(ep.get("id", "") or "")
        result = probes.get(eid)
        if not result or not result.get("ok"):
            continue
        latency = result.get("latency_ms")
        rank = float(latency) if isinstance(latency, (int, float)) else _NO_LATENCY_RANK
        healthy.append((rank, idx, ep))
    healthy.sort(key=lambda t: (t[0], t[1]))  # latency, then original index (stable)
    return [ep for _, _, ep in healthy]


# ── Telemetry ────────────────────────────────────────────────────────────────

def _percentile_nearest_rank(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    rank = ceil(pct * N); index = rank - 1 (clamped to the list bounds). Returns
    0.0 for an empty list. ``pct`` is a fraction in [0, 1].
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    rank = math.ceil(pct * n)
    idx = min(max(rank, 1), n) - 1
    return float(sorted_values[idx])


class Telemetry:
    """Pure per-request sample aggregator for one fleet (or one worker).

    Inject samples with :meth:`add`; read derived metrics with :meth:`report`.
    No clock, no IO — latency and token counts come from the caller, so the same
    aggregator is trivial to unit-test and cheap to update on a hot path.

    Latency percentiles are computed over *successful* samples only (a failed
    request's latency is noise — a fast 500 shouldn't flatter p95). ``tokens/sec``
    is summed tokens over summed successful latency, i.e. effective throughput.
    """

    def __init__(self) -> None:
        self._ok_latencies_ms: list[float] = []
        self._ok_tokens: int = 0
        self._ok_count: int = 0
        self._err_count: int = 0

    def add(self, *, tokens: int, latency_ms: float, ok: bool) -> None:
        """Ingest one request sample.

        ``tokens`` — tokens produced (0 when unknown / on error).
        ``latency_ms`` — wall time of the request in milliseconds.
        ``ok`` — whether the request succeeded (failures feed only the error rate).
        """
        if ok:
            self._ok_count += 1
            self._ok_tokens += max(0, int(tokens))
            self._ok_latencies_ms.append(float(latency_ms))
        else:
            self._err_count += 1

    @property
    def total(self) -> int:
        return self._ok_count + self._err_count

    def error_rate(self) -> float:
        total = self.total
        return (self._err_count / total) if total else 0.0

    def tokens_per_sec(self) -> float:
        total_latency_s = sum(self._ok_latencies_ms) / 1000.0
        if total_latency_s <= 0:
            return 0.0
        return self._ok_tokens / total_latency_s

    def latency_percentile_ms(self, pct: float) -> float:
        return _percentile_nearest_rank(sorted(self._ok_latencies_ms), pct)

    @staticmethod
    def queue_drain_eta(remaining_count: int, throughput_per_sec: float) -> float:
        """Seconds to drain ``remaining_count`` items at ``throughput_per_sec``.

        0.0 when nothing remains; ``math.inf`` when throughput is zero/negative
        (the queue never drains). Throughput is injected by the caller (e.g.
        recent images-sorted/sec) so this stays a pure division.
        """
        if remaining_count <= 0:
            return 0.0
        if throughput_per_sec <= 0:
            return math.inf
        return remaining_count / throughput_per_sec

    def report(self) -> dict[str, Any]:
        """Snapshot all derived metrics as a plain dict (JSON-serialisable)."""
        return {
            "count": self.total,
            "ok_count": self._ok_count,
            "error_count": self._err_count,
            "error_rate": self.error_rate(),
            "tokens_per_sec": self.tokens_per_sec(),
            "latency_p50_ms": self.latency_percentile_ms(0.50),
            "latency_p95_ms": self.latency_percentile_ms(0.95),
        }


__all__ = [
    "probe",
    "probe_fleet",
    "pick_healthy",
    "Telemetry",
    "_http_get",
]
