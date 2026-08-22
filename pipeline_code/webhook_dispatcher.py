"""Per-job lifecycle webhook dispatcher.

Persists webhook subscriptions in each job's ``overrides.webhooks`` list and
sends HTTP POSTs to them off a background thread pool so a slow (or hostile)
receiver can never block the supervisor / dashboard.

Design constraints:

* **SSRF guarded at save time.** URLs are validated with
  ``scheduler._is_public_http_url`` before they're persisted. A private-address
  URL is rejected outright — we never store one we wouldn't dispatch to.
* **Signed if a secret is set.** ``X-Cull-Webhook-Signature: sha256=<hex>`` is
  an HMAC-SHA256 over the raw request body. Receivers verify with their shared
  secret.
* **Retries only on transient failure.** 5xx / connection errors → three
  attempts with 2s / 4s / 8s backoff; 4xx is a permanent-config problem, one
  log line, no retry storm.
* **Best-effort logging.** Never raise up into the caller. The dispatcher is
  observability, not the critical path.

Events emitted:

* ``job.completed`` — supervisor drained a job's queue.
* ``job.sorted_threshold_hit`` — the ``data/sorted/<slug>/<category>/`` folder
  reached a configured count. Polled by ``ThresholdWatcher`` every 30s.
* ``job.scraper_auth_failed`` — a scraper subprocess exited with
  ``credentials.MissingCredentialError`` (EX_CONFIG / 78). Supervisor calls
  ``dispatch`` from its cooldown path.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import job_config
import scheduler
from pipeline_logging import get_logger

logger = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

EVENT_JOB_COMPLETED = "job.completed"
EVENT_SORTED_THRESHOLD_HIT = "job.sorted_threshold_hit"
EVENT_SCRAPER_AUTH_FAILED = "job.scraper_auth_failed"

SUPPORTED_EVENTS: frozenset[str] = frozenset({
    EVENT_JOB_COMPLETED,
    EVENT_SORTED_THRESHOLD_HIT,
    EVENT_SCRAPER_AUTH_FAILED,
})

# One config-sourced pool for the whole process. 4 workers is plenty for the
# hundreds-per-day event volume this dashboard sees; the pool caps concurrency
# so a stuck receiver can't spawn unbounded goroutines.
_MAX_WORKERS = 4
_DISPATCH_TIMEOUT_SECS = 10.0
_RETRY_BACKOFFS: tuple[float, ...] = (2.0, 4.0, 8.0)

# Poll interval for the sorted-threshold watcher. Deliberately coarse (30s) so
# the watcher doesn't hammer the filesystem.
DEFAULT_THRESHOLD_POLL_SECS = 30.0

# Signature header format matches Stripe / GitHub-style webhooks.
SIGNATURE_HEADER = "X-Cull-Webhook-Signature"


# ── Storage ──────────────────────────────────────────────────────────────────

_WEBHOOKS_PATH = "webhooks"  # dotted-path in ``job.overrides``


@dataclass(frozen=True)
class Webhook:
    """A single subscription. ``secret`` is optional; presence enables HMAC."""

    event: str
    url: str
    secret: str = ""
    # For ``job.sorted_threshold_hit`` only: which category / count triggers it.
    threshold_category: str = ""
    threshold_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"event": self.event, "url": self.url}
        if self.secret:
            out["secret"] = self.secret
        if self.threshold_category:
            out["threshold_category"] = self.threshold_category
        if self.threshold_count > 0:
            out["threshold_count"] = self.threshold_count
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Webhook":
        if not isinstance(payload, dict):
            raise ValueError("webhook must be an object")
        event = str(payload.get("event") or "").strip()
        url = str(payload.get("url") or "").strip()
        if event not in SUPPORTED_EVENTS:
            raise ValueError(f"unknown event: {event!r}")
        if not url:
            raise ValueError("url is required")
        secret = str(payload.get("secret") or "").strip()
        cat = str(payload.get("threshold_category") or "").strip()
        try:
            count = int(payload.get("threshold_count") or 0)
        except (TypeError, ValueError):
            count = 0
        return cls(
            event=event,
            url=url,
            secret=secret,
            threshold_category=cat,
            threshold_count=max(0, count),
        )


def list_webhooks(slug: str) -> list[Webhook]:
    """Return every webhook subscribed on ``slug``.

    Returns an empty list when the job doesn't exist or has no ``webhooks``
    override.
    """
    job = job_config.get_job(slug)
    if job is None:
        return []
    raw = (job.overrides or {}).get(_WEBHOOKS_PATH)
    if not isinstance(raw, list):
        return []
    out: list[Webhook] = []
    for entry in raw:
        try:
            out.append(Webhook.from_dict(entry))
        except (TypeError, ValueError):
            continue
    return out


def save_webhooks(slug: str, hooks: Iterable[Webhook]) -> list[Webhook]:
    """Persist the webhook list on ``slug`` and return the saved values.

    URLs are re-validated with :func:`scheduler._is_public_http_url` before
    write — an invalid URL raises ``ValueError`` with a fixed generic message
    (never leaks the URL back through an exception string).
    """
    job = job_config.get_job(slug)
    if job is None:
        raise ValueError(f"job not found: {slug}")

    cleaned: list[Webhook] = []
    seen: set[tuple[str, str]] = set()
    for hook in hooks:
        if not isinstance(hook, Webhook):
            raise ValueError("hooks must be Webhook instances")
        if hook.event not in SUPPORTED_EVENTS:
            raise ValueError("invalid event")
        if not scheduler._is_public_http_url(hook.url):  # noqa: SLF001
            raise ValueError("url must resolve to a public https endpoint")
        key = (hook.event, hook.url)
        if key in seen:
            continue  # fold duplicates
        seen.add(key)
        cleaned.append(hook)

    updated = job_config.set_override(
        job, _WEBHOOKS_PATH, [h.to_dict() for h in cleaned]
    )
    job_config.save_job(updated)
    return cleaned


# ── Signing ──────────────────────────────────────────────────────────────────

def sign_body(secret: str, body: bytes) -> str:
    """Return the ``sha256=<hex>`` signature for ``body`` under ``secret``."""
    if not secret:
        return ""
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def verify_signature(secret: str, body: bytes, signature: str) -> bool:
    """Constant-time signature verification — used by the webhook tests."""
    if not secret or not signature:
        return False
    expected = sign_body(secret, body)
    return hmac.compare_digest(expected, signature)


# ── Dispatch ─────────────────────────────────────────────────────────────────

_pool_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
# Injectable HTTP client so tests can bypass ``requests``. Signature:
#   fn(url, *, data, headers, timeout) -> object with .status_code
_http_post: Callable[..., Any] | None = None


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=_MAX_WORKERS,
                thread_name_prefix="cull-webhook",
            )
        return _pool


def set_http_post(fn: Callable[..., Any] | None) -> None:
    """Swap the HTTP client (tests inject a fake here)."""
    global _http_post
    _http_post = fn


def _http_post_default(url: str, *, data: bytes, headers: dict[str, str], timeout: float):
    """Lazy-import ``requests`` so the module stays cheap to load in tests."""
    import requests  # noqa: PLC0415

    return requests.post(
        url,
        data=data,
        headers=headers,
        timeout=timeout,
        allow_redirects=False,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_payload(event: str, slug: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": event,
        "slug": slug,
        "timestamp_iso": _now_iso(),
        "data": data or {},
    }


def _post_once(hook: Webhook, body: bytes, extra_headers: dict[str, str]) -> tuple[int, str]:
    """POST ``body`` to ``hook`` once. Returns ``(status_code, error_desc)``."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "cull-webhook/0.1",
        **extra_headers,
    }
    if hook.secret:
        headers[SIGNATURE_HEADER] = sign_body(hook.secret, body)

    client = _http_post or _http_post_default
    try:
        resp = client(
            hook.url,
            data=body,
            headers=headers,
            timeout=_DISPATCH_TIMEOUT_SECS,
        )
    except Exception as exc:  # noqa: BLE001 - webhook receiver is untrusted
        return 0, f"transport error: {type(exc).__name__}"
    status = int(getattr(resp, "status_code", 0) or 0)
    return status, ""


def _dispatch_with_retry(hook: Webhook, payload: dict[str, Any]) -> bool:
    """Retry loop: 3 attempts, exponential backoff on 5xx / network errors.

    Returns ``True`` on any 2xx; ``False`` after exhausting retries.
    """
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    for attempt, backoff in enumerate((*_RETRY_BACKOFFS, None), start=1):
        status, err = _post_once(hook, body, {})
        if 200 <= status < 300:
            logger.info(
                "webhook %s → %s: %d (attempt %d)",
                hook.event, hook.url, status, attempt,
            )
            return True
        if 400 <= status < 500:
            # Client error — configuration problem, don't retry.
            logger.warning(
                "webhook %s → %s: HTTP %d, giving up (client error)",
                hook.event, hook.url, status,
            )
            return False
        logger.warning(
            "webhook %s → %s: %s (attempt %d)",
            hook.event, hook.url, err or f"HTTP {status}", attempt,
        )
        if backoff is None:
            break
        time.sleep(backoff)
    return False


def dispatch(event: str, slug: str, data: dict[str, Any] | None = None) -> int:
    """Fan out ``event`` to every matching webhook. Returns fired-count.

    Non-blocking: hands each POST off to the shared thread pool. Returns the
    number of subscriptions matched (i.e. how many futures were scheduled), not
    the number that succeeded — success is asynchronous.
    """
    if event not in SUPPORTED_EVENTS:
        logger.warning("webhook: refusing to dispatch unknown event %r", event)
        return 0
    hooks = [h for h in list_webhooks(slug) if h.event == event]
    if not hooks:
        return 0
    payload = _build_payload(event, slug, data or {})
    pool = _get_pool()
    for hook in hooks:
        pool.submit(_dispatch_with_retry, hook, payload)
    return len(hooks)


def dispatch_sync(event: str, slug: str, data: dict[str, Any] | None = None) -> list[bool]:
    """Synchronous variant — used by tests to observe the retry loop."""
    if event not in SUPPORTED_EVENTS:
        return []
    hooks = [h for h in list_webhooks(slug) if h.event == event]
    payload = _build_payload(event, slug, data or {})
    return [_dispatch_with_retry(h, payload) for h in hooks]


# ── Sorted-threshold watcher ─────────────────────────────────────────────────

@dataclass
class ThresholdWatcher:
    """Poll ``data/sorted/<slug>/<category>/`` counts and fire webhooks.

    A single background thread walks every job's threshold-webhooks. Each
    ``(slug, category, count)`` combination fires **once per crossing** — the
    watcher remembers the last-observed count and only re-fires when the count
    drops below the threshold and climbs back over it.
    """

    sorted_root_fn: Callable[[str], Path]
    poll_interval: float = DEFAULT_THRESHOLD_POLL_SECS
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _last_state: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="cull-webhook-threshold",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 1.0)
            self._thread = None

    def tick(self) -> None:
        """One pass — exposed so tests can drive it without sleeping."""
        try:
            jobs = job_config.list_jobs()
        except Exception:  # noqa: BLE001 - never let a bad job file crash the loop
            return
        for job in jobs:
            for hook in list_webhooks(job.slug):
                if hook.event != EVENT_SORTED_THRESHOLD_HIT:
                    continue
                if not hook.threshold_category or hook.threshold_count <= 0:
                    continue
                count = self._count_for(job.slug, hook.threshold_category)
                key = (job.slug, hook.threshold_category)
                last = self._last_state.get(key, -1)
                self._last_state[key] = count
                # Fire on a fresh crossing: previous < threshold, now >= threshold.
                if last < hook.threshold_count <= count:
                    dispatch(EVENT_SORTED_THRESHOLD_HIT, job.slug, {
                        "category": hook.threshold_category,
                        "count": count,
                        "threshold": hook.threshold_count,
                    })

    def _count_for(self, slug: str, category: str) -> int:
        try:
            root = self.sorted_root_fn(slug)
        except Exception:  # noqa: BLE001
            return 0
        target = root / category
        if not target.is_dir():
            return 0
        try:
            # Count image-like files only — cheap heuristic, avoids .txt / .json.
            return sum(
                1
                for p in target.rglob("*")
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
            )
        except OSError:
            return 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                logger.warning("threshold watcher tick failed: %s", exc)
            self._stop.wait(self.poll_interval)


__all__ = [
    "EVENT_JOB_COMPLETED",
    "EVENT_SORTED_THRESHOLD_HIT",
    "EVENT_SCRAPER_AUTH_FAILED",
    "SUPPORTED_EVENTS",
    "SIGNATURE_HEADER",
    "Webhook",
    "ThresholdWatcher",
    "dispatch",
    "dispatch_sync",
    "list_webhooks",
    "save_webhooks",
    "set_http_post",
    "sign_body",
    "verify_signature",
]
