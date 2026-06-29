"""rate_limit.py — per-source request pacing, backoff, and proxy config.

cull's scrapers each hammer a different upstream (Civitai tRPC, X, Reddit,
Discord, gallery-dl, generic web). Today each one open-codes its own
``time.sleep(...)`` politeness delay and a bespoke ``if status == 429:
sleep(20)`` reaction. This module centralises that into one small, pure,
*deterministic* primitive so behaviour is consistent and testable.

Three concerns, one object (:class:`RateLimiter`):

1. **Pacing** — a rolling-window limiter keyed by source. At most
   ``max_per_min`` admissions land in any 60-second window, with an optional
   hard ``min_interval_s`` floor between consecutive calls.
   :meth:`RateLimiter.acquire` blocks (paces) until a slot is free;
   :meth:`RateLimiter.try_acquire` is the non-blocking variant.

2. **Backoff** — :meth:`RateLimiter.note_429` opens an exponential-backoff
   gate (base · 2ⁿ, capped, with jitter); :meth:`RateLimiter.note_success`
   closes it. :meth:`RateLimiter.blocked_until` / :meth:`current_backoff`
   surface the state so the dashboard can show "source X paused for N s".

3. **Proxy** — :func:`requests_kwargs` turns a config's ``proxy_url`` into the
   ``proxies={...}`` keyword a scraper passes straight to ``requests``.

Determinism is the whole point: the wall clock, ``sleep``, and the jitter RNG
are all injected. Production wiring passes ``time.monotonic`` / ``time.sleep``
/ ``random.random``; tests pass a fake clock and a fixed RNG. Nothing in the
pacing logic calls ``time.*`` directly.

This is library code, so it logs via :mod:`pipeline_logging` rather than
printing. It has no side effects beyond the injected ``sleep``.
"""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Mapping

from pipeline_logging import get_logger

logger = get_logger(__name__)

# A 60-second rolling window is the natural unit for "per minute" budgets.
_WINDOW_SECONDS: float = 60.0

# Fallback cap so a misconfigured source can never back off unboundedly.
_DEFAULT_MAX_BACKOFF_S: float = 300.0
_DEFAULT_BASE_BACKOFF_S: float = 1.0

NowFn = Callable[[], float]
SleepFn = Callable[[float], None]
RandFn = Callable[[], float]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RateLimitConfig:
    """Per-source rate-limit / proxy / backoff settings.

    All fields are optional with permissive defaults so an unconfigured source
    is effectively unthrottled (but still backoff-aware once it sees a 429).

    Attributes:
        source: Identity of the upstream (e.g. ``"civitai"``). Used for keying
            and for log / UI labels.
        max_per_min: Max admissions per rolling 60 s window. ``None`` or ``0``
            means unbounded.
        min_interval_s: Hard floor (seconds) between consecutive admissions.
            ``0`` disables it.
        proxy_url: Optional proxy URL (``http://``/``https://``/``socks5://``)
            applied to both schemes via :func:`requests_kwargs`.
        max_backoff_s: Upper bound on the exponential backoff wait.
        base_backoff_s: First backoff step; doubles each consecutive 429.
        jitter: Fractional jitter in ``[0, 1]``. The computed backoff is scaled
            by ``1 - jitter * rand()`` so it never exceeds the cap and stays
            non-negative (full jitter, decorrelated downward).
    """

    source: str
    max_per_min: int | None = None
    min_interval_s: float = 0.0
    proxy_url: str | None = None
    max_backoff_s: float = _DEFAULT_MAX_BACKOFF_S
    base_backoff_s: float = _DEFAULT_BASE_BACKOFF_S
    jitter: float = 0.2

    @classmethod
    def from_mapping(cls, source: str, env: Mapping[str, str]) -> "RateLimitConfig":
        """Build a config from a flat env-style mapping.

        Reads ``RATE_LIMIT_<SOURCE>_{MAX_PER_MIN,MIN_INTERVAL_S,PROXY,
        MAX_BACKOFF_S}`` (source upper-cased, non-alphanumerics → ``_``).
        Unparseable numbers fall back to the field default rather than raising,
        so a typo in ``.env`` degrades to "unthrottled" instead of crashing a
        scraper at import time.
        """
        key = _env_key(source)
        return cls(
            source=source,
            max_per_min=_as_int(env.get(f"RATE_LIMIT_{key}_MAX_PER_MIN")),
            min_interval_s=_as_float(env.get(f"RATE_LIMIT_{key}_MIN_INTERVAL_S"), 0.0),
            proxy_url=_clean_proxy(env.get(f"RATE_LIMIT_{key}_PROXY")),
            max_backoff_s=_as_float(
                env.get(f"RATE_LIMIT_{key}_MAX_BACKOFF_S"), _DEFAULT_MAX_BACKOFF_S
            ),
        )


def _env_key(source: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in source).upper()


def _as_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _as_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _clean_proxy(raw: str | None) -> str | None:
    if raw is None:
        return None
    val = raw.strip()
    return val or None


# ---------------------------------------------------------------------------
# Proxy helper
# ---------------------------------------------------------------------------

def requests_kwargs(config: RateLimitConfig) -> dict[str, dict[str, str]]:
    """Return ``requests`` keyword args derived from ``config``.

    Currently just the proxy: ``{"proxies": {"http": url, "https": url}}`` when
    a proxy is set, otherwise ``{}`` (so ``**requests_kwargs(cfg)`` is always
    safe to splat into a ``requests`` call).
    """
    proxy = _clean_proxy(config.proxy_url)
    if not proxy:
        return {}
    return {"proxies": {"http": proxy, "https": proxy}}


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

@dataclass
class _BackoffState:
    """Mutable backoff bookkeeping for one source."""

    consecutive_429: int = 0
    blocked_until: float | None = None
    current: float = 0.0


class RateLimiter:
    """Rolling-window limiter + exponential backoff for a single source.

    Construct one per source (the supervisor / scraper owns the instance). All
    timing is injected:

        RateLimiter(cfg, now=time.monotonic, sleep=time.sleep, rand=random.random)

    The defaults wire those production callables, so ``RateLimiter(cfg)`` works
    out of the box; tests pass a fake clock for determinism.
    """

    def __init__(
        self,
        config: RateLimitConfig,
        *,
        now: NowFn = time.monotonic,
        sleep: SleepFn = time.sleep,
        rand: RandFn = random.random,
    ) -> None:
        self.config = config
        self._now = now
        self._sleep = sleep
        self._rand = rand
        # Admission timestamps inside the current rolling window (monotonic).
        self._admissions: Deque[float] = deque()
        self._last_admit: float | None = None
        self._backoff = _BackoffState()

    # -- public: pacing ----------------------------------------------------

    def acquire(self) -> None:
        """Block until a slot is available, then record the admission.

        Waits out (in order): any active backoff gate, the rolling-window
        budget, and the ``min_interval_s`` floor. Re-checks after each sleep so
        a single call can satisfy several constraints.
        """
        while True:
            wait = self._wait_needed()
            if wait <= 0.0:
                self._record_admission()
                return
            self._sleep(wait)

    def try_acquire(self) -> bool:
        """Non-blocking admission. Return ``True`` and record it if a slot is
        free *right now*; otherwise return ``False`` and change nothing."""
        if self._wait_needed() > 0.0:
            return False
        self._record_admission()
        return True

    # -- public: backoff ---------------------------------------------------

    def note_429(self, retry_after: float | None = None) -> None:
        """Record a rate-limit / throttle response and open the backoff gate.

        Args:
            retry_after: Optional server-provided ``Retry-After`` in seconds. If
                given (and non-negative) it overrides the computed exponential
                wait for this step.
        """
        st = self._backoff
        st.consecutive_429 += 1
        if retry_after is not None and retry_after >= 0.0:
            wait = float(retry_after)
        else:
            wait = self._compute_backoff(st.consecutive_429)
        st.current = wait
        st.blocked_until = self._now() + wait
        logger.warning(
            "rate_limit[%s]: 429 #%d → backing off %.2fs",
            self.config.source, st.consecutive_429, wait,
        )

    def note_success(self) -> None:
        """Record a successful response and reset the backoff gate."""
        if self._backoff.consecutive_429:
            logger.info("rate_limit[%s]: success → backoff cleared", self.config.source)
        self._backoff = _BackoffState()

    # -- public: introspection (for the UI) --------------------------------

    def current_backoff(self) -> float:
        """Seconds of backoff currently imposed (0.0 when not backing off)."""
        return self._backoff.current

    def blocked_until(self) -> float | None:
        """Monotonic timestamp the source is paused until, or ``None``.

        Returns ``None`` once the gate has elapsed so callers can treat
        "no value" as "go".
        """
        until = self._backoff.blocked_until
        if until is None:
            return None
        if until <= self._now():
            return None
        return until

    def remaining(self) -> int | None:
        """Admissions still available in the current rolling window.

        ``None`` means unbounded (no ``max_per_min`` configured).
        """
        cap = self._cap()
        if cap is None:
            return None
        self._evict_expired()
        return max(0, cap - len(self._admissions))

    def snapshot(self) -> dict[str, object]:
        """A compact dict for the dashboard to render per-source status."""
        return {
            "source": self.config.source,
            "remaining": self.remaining(),
            "max_per_min": self._cap(),
            "min_interval_s": self.config.min_interval_s,
            "backoff_s": round(self.current_backoff(), 3),
            "blocked_until": self.blocked_until(),
            "proxied": bool(_clean_proxy(self.config.proxy_url)),
        }

    def requests_kwargs(self) -> dict[str, dict[str, str]]:
        """Convenience: this limiter's proxy kwargs for ``requests``."""
        return requests_kwargs(self.config)

    # -- internals ---------------------------------------------------------

    def _cap(self) -> int | None:
        cap = self.config.max_per_min
        return cap if cap and cap > 0 else None

    def _evict_expired(self) -> None:
        """Drop admissions that have aged out of the rolling window."""
        horizon = self._now() - _WINDOW_SECONDS
        adm = self._admissions
        while adm and adm[0] <= horizon:
            adm.popleft()

    def _wait_needed(self) -> float:
        """Seconds to wait before the next admission may proceed (0 = now).

        The max of three independent gates: backoff, window, min-interval.
        """
        now = self._now()
        waits = [0.0]

        # 1) Backoff gate.
        until = self._backoff.blocked_until
        if until is not None and until > now:
            waits.append(until - now)

        # 2) Rolling-window budget: if full, wait for the oldest to age out.
        cap = self._cap()
        if cap is not None:
            self._evict_expired()
            if len(self._admissions) >= cap:
                waits.append((self._admissions[0] + _WINDOW_SECONDS) - now)

        # 3) Minimum spacing between consecutive calls.
        gap = self.config.min_interval_s
        if gap > 0.0 and self._last_admit is not None:
            earliest = self._last_admit + gap
            if earliest > now:
                waits.append(earliest - now)

        return max(waits)

    def _record_admission(self) -> None:
        now = self._now()
        self._evict_expired()
        self._admissions.append(now)
        self._last_admit = now

    def _compute_backoff(self, attempt: int) -> float:
        """Exponential backoff for the n-th consecutive 429, capped + jittered.

        Wait = min(base · 2^(n-1), cap), then scaled by ``1 - jitter·rand()``
        (full jitter biased downward) so the result is always in ``[0, cap]``.
        """
        base = max(0.0, self.config.base_backoff_s)
        cap = max(0.0, self.config.max_backoff_s)
        raw = base * (2.0 ** max(0, attempt - 1))
        capped = min(raw, cap)
        jitter = min(max(self.config.jitter, 0.0), 1.0)
        if jitter:
            capped *= 1.0 - jitter * self._rand()
        return max(0.0, capped)


__all__ = [
    "RateLimitConfig",
    "RateLimiter",
    "requests_kwargs",
]
