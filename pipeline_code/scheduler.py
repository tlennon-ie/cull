"""Per-job scheduling + completion webhooks / desktop notifications.

This module owns two small, independent concerns:

1. **Scheduling.** A :class:`Schedule` pairs a job ``slug`` with a ``cadence``
   and an ``action`` (``scrape`` / ``curate`` / ``export``). The list persists
   to ``data/schedules.json`` (atomic write, tolerant load). The supervisor is
   expected to tick :func:`run_due` on its reconcile loop; everything that
   touches the clock takes ``now`` as a parameter so the selection logic
   (:func:`due`) is pure and trivially testable.

2. **Notifications.** :func:`notify` POSTs a JSON envelope to a webhook (so a
   completed job can fan out to Slack / Discord / a home-automation hook) and,
   if enabled, fires a best-effort desktop notification. Both paths are
   best-effort: a network error or a missing desktop-notification library never
   raises into the caller.

Design rules (load-bearing):

* ``now`` is ALWAYS injected into pure logic. Nothing here calls
  ``datetime.now()`` / ``time.time()`` at import or inside :func:`due`. The two
  convenience wrappers that *do* need a wall clock (:func:`run_due_now`) read it
  explicitly and pass it down.
* The optional desktop-notification backend (``plyer`` → ``win10toast``) is
  imported lazily inside a ``try/except`` so the base install stays lean and CI
  never depends on a GUI library.
* Persistence mirrors ``seen_store`` / ``categories``: a single JSON file under
  the data root, written atomically via a tempfile + ``os.replace``.

Public API:
    Schedule                      # dataclass {slug, cadence, action, enabled, last_run}
    cadence_seconds(cadence)      # parse the small cron-ish subset -> seconds
    due(schedules, now)           # PURE selector -> schedules to run
    run_due(now, runner)          # call the injected runner for due schedules
    run_due_now(runner)           # thin wall-clock wrapper around run_due
    list_schedules()              # load persisted schedules
    add_schedule(schedule)        # upsert by (slug, action)
    update_schedule(slug, action, **fields)
    remove_schedule(slug, action) # -> bool
    notify(event, payload, webhook_url=None)  # webhook POST + optional desktop
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import requests

import digest as _digest
from paths import base_dir as _default_base_dir
from pipeline_logging import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: The job actions a schedule may trigger. Kept here (not inlined) so callers /
#: the dashboard editor can offer exactly these and nothing else.
ACTIONS: tuple[str, ...] = ("scrape", "curate", "export", "digest")

#: Timeout used when POSTing a digest webhook. Kept small so a slow hook can
#: never stall a scheduler tick.
_DIGEST_TIMEOUT = 15.0

#: Accepted digest webhook payload styles.
_DIGEST_STYLES: frozenset[str] = frozenset({"discord", "slack"})

#: Named cron-ish aliases → period in seconds.
_ALIASES: dict[str, int] = {
    "@hourly": 3600,
    "@daily": 86400,
    "@weekly": 7 * 86400,
}

#: Interval-unit suffixes → seconds-per-unit. Minimal on purpose.
_UNIT_SECONDS: dict[str, int] = {"m": 60, "h": 3600}

#: ``every 30m`` / ``every 2h`` / bare ``30m`` — captures count + unit.
_INTERVAL_RE = re.compile(r"^(?:every\s+)?(\d+)\s*([mh])$", re.IGNORECASE)

#: Default webhook timeout (seconds). Small — a slow hook must not stall a tick.
_WEBHOOK_TIMEOUT = 8.0

#: Filename for the persisted schedule list, under the data root.
_FILENAME = "schedules.json"

_lock = threading.Lock()


# ── Cadence parsing (pure) ────────────────────────────────────────────────────

def cadence_seconds(cadence: str) -> int:
    """Translate a cadence string into a period in **seconds**.

    Supports a deliberately tiny subset:

    * Aliases: ``@hourly``, ``@daily``, ``@weekly``.
    * Intervals: ``every 30m``, ``every 2h``, or the bare ``30m`` / ``2h`` form.

    Raises :class:`ValueError` for anything else, so a bad cadence fails fast at
    the edit boundary rather than silently never firing.
    """
    if not isinstance(cadence, str):
        raise ValueError(f"cadence must be a string, got {type(cadence).__name__}")
    key = cadence.strip().lower()
    if not key:
        raise ValueError("cadence is empty")
    if key in _ALIASES:
        return _ALIASES[key]
    match = _INTERVAL_RE.match(key)
    if match is None:
        raise ValueError(f"unrecognised cadence: {cadence!r}")
    count = int(match.group(1))
    if count <= 0:
        raise ValueError(f"cadence interval must be positive: {cadence!r}")
    return count * _UNIT_SECONDS[match.group(2).lower()]


# ── Schedule dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Schedule:
    """One scheduled action for a job.

    * ``slug`` — the job slug this fires for.
    * ``cadence`` — a string accepted by :func:`cadence_seconds`.
    * ``action`` — one of :data:`ACTIONS`.
    * ``enabled`` — disabled schedules are never selected by :func:`due`.
    * ``last_run`` — ``datetime`` of the last successful run, or ``None`` if it
      has never run (in which case it is due immediately).

    Frozen so it stays immutable; mutate via :func:`dataclasses.replace`.
    """

    slug: str
    cadence: str = "@daily"
    action: str = "scrape"
    enabled: bool = True
    last_run: datetime | None = field(default=None)
    # Free-form action config. Kept out of ``__post_init__`` validation because
    # each action gets to interpret its own keys; today only the ``digest``
    # action consumes it (``webhook_url`` / ``webhook_style`` / ``since_hours``
    # / ``top_n``). Frozen dataclass → we hold a tuple of items for hashability
    # and expose a dict via :attr:`config`.
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(
                f"unknown action {self.action!r}; expected one of {ACTIONS}"
            )
        # Validate the cadence eagerly so a bad value can't be persisted.
        cadence_seconds(self.cadence)

    # ── (de)serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (datetime → ISO-8601 string)."""
        out: dict[str, Any] = {
            "slug": self.slug,
            "cadence": self.cadence,
            "action": self.action,
            "enabled": bool(self.enabled),
            "last_run": self.last_run.isoformat() if self.last_run else None,
        }
        # Only serialise config when non-empty so existing schedule files stay
        # byte-identical after a load / save round trip.
        if self.config:
            out["config"] = dict(self.config)
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Schedule":
        """Rebuild a Schedule from persisted JSON, tolerating loose types."""
        cfg = raw.get("config")
        if not isinstance(cfg, dict):
            cfg = {}
        return cls(
            slug=str(raw.get("slug", "")),
            cadence=str(raw.get("cadence", "@daily")),
            action=str(raw.get("action", "scrape")),
            enabled=bool(raw.get("enabled", True)),
            last_run=_parse_dt(raw.get("last_run")),
            config=dict(cfg),
        )


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO-8601 string (or pass through a datetime) → datetime | None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        logger.warning("schedule: unparseable last_run %r; treating as never-run", value)
        return None


# ── Due selection (pure) ──────────────────────────────────────────────────────

def _as_epoch(moment: datetime) -> float:
    """Seconds-since-epoch for a datetime, treating naive values as UTC.

    Keeps :func:`due` robust to a mix of naive (freshly-constructed) and aware
    (round-tripped) timestamps without forcing every caller to pick one.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def is_due(schedule: Schedule, now: datetime) -> bool:
    """Return True if ``schedule`` should run at ``now`` (pure).

    A schedule is due when it is enabled AND (it has never run OR at least one
    cadence period has elapsed since ``last_run``).
    """
    if not schedule.enabled:
        return False
    if schedule.last_run is None:
        return True
    elapsed = _as_epoch(now) - _as_epoch(schedule.last_run)
    return elapsed >= cadence_seconds(schedule.cadence)


def due(schedules: Iterable[Schedule], now: datetime) -> list[Schedule]:
    """Filter ``schedules`` down to those due at ``now``.

    Pure: ``now`` is injected and no input is mutated. Order is preserved from
    the input iterable.
    """
    return [s for s in schedules if is_due(s, now)]


# ── Persistence (add / update / remove / list) ────────────────────────────────

def _store_path() -> Path:
    """Absolute path to the persisted schedule list under the data root."""
    return _default_base_dir() / _FILENAME


def list_schedules() -> list[Schedule]:
    """Load the persisted schedules. Tolerant: a missing or corrupt file yields
    an empty list rather than raising."""
    path = _store_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("schedules file corrupt or unreadable, ignoring: %s", path)
        return []
    if not isinstance(payload, list):
        logger.warning("schedules file is not a list, ignoring: %s", path)
        return []
    out: list[Schedule] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        try:
            out.append(Schedule.from_dict(raw))
        except ValueError as exc:
            logger.warning("skipping invalid persisted schedule %r: %s", raw, exc)
    return out


def _write(schedules: list[Schedule]) -> None:
    """Persist ``schedules`` atomically (tempfile + ``os.replace``)."""
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps([s.to_dict() for s in schedules], indent=2, ensure_ascii=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def add_schedule(schedule: Schedule) -> Schedule:
    """Insert or update ``schedule``, keyed by ``(slug, action)``.

    Re-adding the same ``(slug, action)`` overwrites the existing entry (so the
    list never accumulates duplicates), but different actions for the same slug
    coexist. Returns the stored schedule.
    """
    with _lock:
        current = list_schedules()
        kept = [s for s in current
                if not (s.slug == schedule.slug and s.action == schedule.action)]
        kept.append(schedule)
        _write(kept)
    return schedule


def update_schedule(slug: str, action: str, **fields: Any) -> Schedule | None:
    """Patch the schedule identified by ``(slug, action)``.

    ``fields`` may include ``cadence`` / ``enabled`` / ``last_run`` / ``slug`` /
    ``action``. Returns the updated schedule, or ``None`` if no match exists.
    """
    with _lock:
        current = list_schedules()
        updated: Schedule | None = None
        out: list[Schedule] = []
        for s in current:
            if s.slug == slug and s.action == action and updated is None:
                updated = replace(s, **fields)
                out.append(updated)
            else:
                out.append(s)
        if updated is not None:
            _write(out)
        return updated


def remove_schedule(slug: str, action: str) -> bool:
    """Delete the schedule identified by ``(slug, action)``.

    Returns True if something was removed, False if no match existed.
    """
    with _lock:
        current = list_schedules()
        kept = [s for s in current if not (s.slug == slug and s.action == action)]
        if len(kept) == len(current):
            return False
        _write(kept)
        return True


# ── run_due — drive the injected runner ───────────────────────────────────────

Runner = Callable[[str, str], Any]


def _dispatch(sched: Schedule, runner: Runner) -> None:
    """Run one schedule's action.

    Scheduler-owned actions (``digest``) are handled here directly — they are
    pure reads over :mod:`index_store` and do not touch the supervisor's
    process fleet. Everything else is delegated to the injected ``runner``.
    """
    if sched.action == "digest":
        cfg = sched.config or {}
        run_digest(
            sched.slug,
            since_hours=int(cfg.get("since_hours", 24) or 24),
            top_n=int(cfg.get("top_n", 20) or 20),
            webhook_url=cfg.get("webhook_url") or None,
            webhook_style=str(cfg.get("webhook_style", "discord") or "discord"),
        )
        return
    runner(sched.slug, sched.action)


def run_due(now: datetime, runner: Runner) -> list[Schedule]:
    """Run every persisted schedule that is due at ``now`` via ``runner``.

    ``runner(slug, action)`` is the injected side-effecting callback — the
    supervisor passes one that actually kicks the pipeline; tests pass a stub.
    Each schedule that runs has its ``last_run`` stamped to ``now`` and the list
    is re-persisted, so the same schedule won't re-fire until its cadence
    elapses again.

    A ``runner`` exception is logged and swallowed so one failing job can't abort
    the others or crash the supervisor's tick. Returns the schedules whose runner
    completed without raising.
    """
    with _lock:
        current = list_schedules()
        to_run = due(current, now)
        if not to_run:
            return []
        run_keys = {(s.slug, s.action) for s in to_run}
        succeeded: list[Schedule] = []
        for sched in to_run:
            try:
                _dispatch(sched, runner)
            except Exception as exc:  # noqa: BLE001 - best-effort, never abort the tick
                logger.warning(
                    "scheduled %s for %r failed: %s", sched.action, sched.slug, exc
                )
                continue
            succeeded.append(sched)
        # Stamp last_run for everything we *attempted* (so a persistently failing
        # job backs off on its cadence instead of hammering every tick).
        stamped = [
            replace(s, last_run=now) if (s.slug, s.action) in run_keys else s
            for s in current
        ]
        _write(stamped)
    return succeeded


def run_due_now(runner: Runner) -> list[Schedule]:
    """Wall-clock convenience wrapper around :func:`run_due`.

    Reads the clock here (the one place it's allowed) and delegates to the pure
    core. The supervisor can call this directly from its reconcile loop.
    """
    return run_due(datetime.now(timezone.utc), runner)


# ── SSRF guard ────────────────────────────────────────────────────────────────

def _is_public_http_url(url: str) -> bool:
    """Return True iff ``url`` is a well-formed http(s) URL whose host resolves
    exclusively to public unicast addresses.

    Rejects: non-http(s) schemes, missing hostnames, hosts in the private /
    loopback / link-local / multicast / reserved ranges, and hosts whose DNS
    resolution touches any of those. The DNS check catches naive attempts to
    dodge the string check with a domain that points at ``127.0.0.1``.

    Best-effort: a DNS failure returns False (fail-closed) — the digest hook
    is a webhook, not a critical path, so we prefer "silently skip" over
    "silently deliver to localhost".
    """
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme.lower() not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    # Explicit belt-and-braces string check for the most common gotchas.
    if host in ("localhost", "0.0.0.0", "broadcasthost", "ip6-localhost", "ip6-loopback"):
        return False

    # Every resolved address must be a public unicast address. socket.getaddrinfo
    # covers both IPv4 and IPv6.
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False
    if not infos:
        return False
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            return False
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


# ── Digest action ─────────────────────────────────────────────────────────────

def _digest_webhook_body(markdown: str, style: str) -> dict[str, Any]:
    """Format ``markdown`` for the target webhook style.

    Discord expects ``{"content": ...}``; Slack expects ``{"text": ...}``.
    Anything else falls back to Discord's shape (matches the parameter default).
    """
    key = "text" if style == "slack" else "content"
    return {key: markdown}


def run_digest(
    slug: str,
    *,
    since_hours: int = 24,
    top_n: int = 20,
    webhook_url: str | None = None,
    webhook_style: str = "discord",
) -> dict[str, Any]:
    """Build a digest payload for ``slug`` and optionally POST it to a webhook.

    Always logs a one-line summary regardless of webhook configuration so the
    supervisor's tick log records that the digest fired. When ``webhook_url``
    is set, the URL is validated by :func:`_is_public_http_url` (SSRF guard)
    and the Markdown body is POSTed with a 15s cap. A rejected URL is logged
    as a warning and the payload is still returned to the caller.
    """
    hours = max(1, int(since_hours or 24))
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    payload = _digest.build_digest(slug, since=since, top_n=int(top_n or 20))

    totals = payload.get("totals", {})
    logger.info(
        "digest: slug=%s window=%dh sample=%d kept=%d borderline=%d off_topic=%d discarded=%d",
        slug,
        payload.get("window_hours", hours),
        payload.get("sample_count", 0),
        totals.get("kept", 0),
        totals.get("borderline", 0),
        totals.get("off_topic", 0),
        totals.get("discarded", 0),
    )

    if webhook_url:
        style = (webhook_style or "discord").strip().lower()
        if style not in _DIGEST_STYLES:
            logger.warning(
                "digest: unknown webhook_style %r for slug=%s, falling back to discord",
                webhook_style, slug,
            )
            style = "discord"
        if not _is_public_http_url(webhook_url):
            logger.warning(
                "digest: refusing to POST to non-public webhook URL for slug=%s", slug,
            )
        else:
            body = _digest_webhook_body(_digest.render_markdown(payload), style)
            try:
                resp = requests.post(webhook_url, json=body, timeout=_DIGEST_TIMEOUT)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("digest: webhook POST failed for slug=%s: %s", slug, exc)
            except Exception as exc:  # noqa: BLE001 - best-effort, never crash a tick
                logger.warning("digest: webhook POST errored for slug=%s: %s", slug, exc)

    return payload


# ── Notifications ─────────────────────────────────────────────────────────────

def _env_truthy(key: str) -> bool:
    """Return True if env var ``key`` is set to a truthy string."""
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _desktop_notify(title: str, message: str) -> bool:
    """Fire a best-effort desktop notification. Returns True on success.

    The backend is imported lazily and guarded: if neither ``plyer`` nor
    ``win10toast`` is installed (the base case), this is a no-op that returns
    False instead of raising. Declared as a module function so tests can stub it.
    """
    # Preferred: cross-platform plyer.
    try:
        from plyer import notification as _plyer  # type: ignore

        _plyer.notify(title=title, message=message, timeout=5)
        return True
    except Exception:  # noqa: BLE001 - missing lib OR backend failure: stay silent
        pass
    # Windows fallback: win10toast.
    try:
        from win10toast import ToastNotifier  # type: ignore

        ToastNotifier().show_toast(title, message, duration=5, threaded=True)
        return True
    except Exception:  # noqa: BLE001 - missing lib OR backend failure: stay silent
        return False


def notify(
    event: str,
    payload: dict[str, Any] | None = None,
    webhook_url: str | None = None,
) -> bool:
    """Emit a completion notification. Best-effort; never raises.

    Builds a JSON envelope ``{"event": <event>, **payload}`` and:

    * POSTs it to ``webhook_url`` (or ``$WEBHOOK_URL`` when ``webhook_url`` is
      ``None``). Passing an explicit empty string suppresses the webhook.
    * Optionally fires a desktop notification when ``$DESKTOP_NOTIFICATIONS`` is
      truthy.

    Returns True if *either* channel reported success, False if nothing was
    delivered (e.g. no webhook configured and desktop disabled / unavailable).
    """
    body: dict[str, Any] = {"event": event}
    if payload:
        body.update(payload)

    delivered = False

    # Resolve the webhook target. ``None`` means "fall back to env"; an explicit
    # empty string means "deliberately no webhook".
    target = os.environ.get("WEBHOOK_URL", "") if webhook_url is None else webhook_url
    target = (target or "").strip()
    if target:
        delivered = _post_webhook(target, body) or delivered

    if _env_truthy("DESKTOP_NOTIFICATIONS"):
        title = f"cull: {event}"
        message = _summarise(body)
        delivered = _desktop_notify(title, message) or delivered

    return delivered


#: Only http(s) is a legitimate webhook scheme. Reject ``file://``, ``gopher://``,
#: ``ftp://`` and friends so a hostile / typo'd WEBHOOK_URL cannot read local
#: files or open unexpected protocols through requests / urllib3.
_ALLOWED_WEBHOOK_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _is_safe_webhook_url(url: str) -> bool:
    """Return True iff ``url`` is a syntactically-valid http(s) URL.

    Basic scheme guard; we deliberately do NOT block private/loopback hosts here
    because the webhook is user-configured and pointing it at a LAN service
    (e.g. a home automation hub or a Docker sidecar) is a legitimate use case.
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return False
    return parsed.scheme.lower() in _ALLOWED_WEBHOOK_SCHEMES and bool(parsed.netloc)


def _post_webhook(url: str, body: dict[str, Any]) -> bool:
    """POST ``body`` as JSON to ``url``. Swallows all network errors → bool.

    Rejects any URL whose scheme is not http/https (SSRF guard against
    ``file://`` / ``gopher://`` etc.), and disables redirect-following so a
    hostile responder cannot 302-bounce the POST at a metadata endpoint.
    """
    if not _is_safe_webhook_url(url):
        # Log only the scheme — the full URL may carry secret query-string
        # tokens (Slack/Discord webhooks routinely do), and log lines outlive
        # their SSRF-refusal context.
        try:
            scheme = urlparse(url).scheme or "(none)"
        except Exception:  # noqa: BLE001
            scheme = "(unparseable)"
        logger.warning("webhook POST refused: scheme=%r is not http(s)", scheme)
        return False
    try:
        resp = requests.post(
            url,
            json=body,
            timeout=_WEBHOOK_TIMEOUT,
            allow_redirects=False,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("webhook POST to %s failed: %s", url, exc)
        return False
    except Exception as exc:  # noqa: BLE001 - never let a notification crash a tick
        logger.warning("webhook POST to %s errored: %s", url, exc)
        return False


def _summarise(body: dict[str, Any]) -> str:
    """A compact one-line summary of the payload for desktop toasts."""
    parts = [f"{k}={v}" for k, v in body.items() if k != "event"]
    return ", ".join(parts) if parts else body.get("event", "")


__all__ = [
    "ACTIONS",
    "Schedule",
    "cadence_seconds",
    "is_due",
    "due",
    "list_schedules",
    "add_schedule",
    "update_schedule",
    "remove_schedule",
    "run_due",
    "run_due_now",
    "run_digest",
    "notify",
]
