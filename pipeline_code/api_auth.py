"""API-key authentication + rate-limiting middleware for the cull dashboard.

Layered on top of the existing loopback gate (``_wave_is_loopback_request``) so
zero-config local installs keep working: with ``CULL_API_KEYS`` unset every
request goes straight through as before. Once tokens are configured, any
non-loopback caller must present one via ``Authorization: Bearer <token>`` or
``X-Cull-API-Key: <token>``.

The middleware is installed with a single ``install(app, ...)`` call from
``dashboard_enhanced.py`` and does its work in a ``before_request`` hook so no
handler has to know it exists.

Env vars (all optional):

* ``CULL_API_KEYS`` — comma-separated tokens (each 32+ chars). Empty / unset =
  auth disabled.
* ``CULL_API_KEY_SCOPES`` — optional JSON ``{token: [scope, ...]}``. Scopes are
  ``read`` / ``curator`` / ``admin``. Missing = ``curator``.
* ``CULL_API_KEY_RATE_LIMIT`` — per-token per-minute cap on write endpoints
  (default 120).

Scope model:

* ``GET`` / ``HEAD`` need ``read``.
* ``POST`` / ``PUT`` / ``DELETE`` / ``PATCH`` need ``curator`` (writes).
* ``admin``-flagged endpoints (settings, credentials, publish flows) need
  ``admin``.

Everything is expressed as pure helpers on top of the ``Flask`` request context;
tests exercise the classifier + bucket without spinning the dashboard up.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from flask import Flask, jsonify, request

from pipeline_logging import get_logger

logger = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# The three scopes, ordered by privilege.
SCOPE_READ = "read"
SCOPE_CURATOR = "curator"
SCOPE_ADMIN = "admin"
_ALL_SCOPES: frozenset[str] = frozenset({SCOPE_READ, SCOPE_CURATOR, SCOPE_ADMIN})

# Minimum entropy floor for a token. 32 chars is a good default for random
# hex/base64 tokens (~192 bits of entropy at hex, plenty at base64).
MIN_TOKEN_LEN = 32

# Default per-token per-minute write cap when the env var is unset / invalid.
DEFAULT_RATE_LIMIT = 120

# Header names we accept. Both are supported so a client can pick whichever it
# prefers; the ``Authorization`` header is idiomatic HTTP, the custom header is
# convenient for tooling that clashes with it.
_HEADER_BEARER = "Authorization"
_HEADER_APIKEY = "X-Cull-API-Key"

# Regex patterns for admin-tier endpoint URLs. Kept as compiled regexes so we
# don't rebuild them on every request. Order does not matter; anything that
# matches ANY entry is admin.
_ADMIN_URL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^/api/settings(?:/.*|/?)$"),
    re.compile(r"^/api/credentials(?:/.*|/?)$"),
    re.compile(r"^/api/presets/[^/]+/publish/?$"),
    re.compile(r"^/api/themes/[^/]+/publish/?$"),
)

# Substrings that mark a path as admin-tier even if it doesn't match the URL
# patterns above (defensive: any endpoint that reads/writes credentials by
# convention has "credential" in its path fragment).
_ADMIN_URL_SUBSTRINGS: tuple[str, ...] = ("credential",)

# HTTP methods that count as reads for scope purposes.
_READ_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuthConfig:
    """Parsed auth config. ``tokens`` is empty when auth is disabled.

    ``scopes`` maps token → allowed scope set (already lower-cased); a token
    missing from the map defaults to ``{read, curator}`` (everything except
    ``admin``).
    """

    tokens: frozenset[str] = field(default_factory=frozenset)
    scopes: dict[str, frozenset[str]] = field(default_factory=dict)
    rate_limit_per_min: int = DEFAULT_RATE_LIMIT

    @property
    def enabled(self) -> bool:
        return bool(self.tokens)


def _parse_tokens(raw: str | None) -> list[str]:
    """Return the sanitised token list. Rejects tokens shorter than the floor.

    Duplicates are folded so a token accidentally listed twice doesn't get two
    independent rate-limit buckets.
    """
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token or token in seen:
            continue
        if len(token) < MIN_TOKEN_LEN:
            logger.warning(
                "api_auth: ignoring CULL_API_KEYS entry shorter than %d chars",
                MIN_TOKEN_LEN,
            )
            continue
        seen.add(token)
        out.append(token)
    return out


def _parse_scopes(raw: str | None, tokens: Iterable[str]) -> dict[str, frozenset[str]]:
    """Parse ``CULL_API_KEY_SCOPES`` into a token → scope-set map.

    Only entries for known tokens are kept, so a rotated token can't linger in
    the scope map. Invalid scope names are silently dropped.
    """
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("api_auth: CULL_API_KEY_SCOPES is not valid JSON; ignoring")
        return {}
    if not isinstance(payload, dict):
        logger.warning("api_auth: CULL_API_KEY_SCOPES must be a JSON object; ignoring")
        return {}

    known = set(tokens)
    out: dict[str, frozenset[str]] = {}
    for token, scopes in payload.items():
        if not isinstance(token, str) or token not in known:
            continue
        if not isinstance(scopes, list):
            continue
        cleaned = {
            str(s).strip().lower()
            for s in scopes
            if isinstance(s, str) and s.strip().lower() in _ALL_SCOPES
        }
        if cleaned:
            out[token] = frozenset(cleaned)
    return out


def _parse_rate_limit(raw: str | None) -> int:
    """Return a positive int rate cap; fall back to the default on garbage."""
    if raw is None or str(raw).strip() == "":
        return DEFAULT_RATE_LIMIT
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_RATE_LIMIT
    return max(1, value)


def load_config_from_env(env: dict[str, str] | None = None) -> AuthConfig:
    """Build an ``AuthConfig`` from ``os.environ`` (or an injected mapping)."""
    src = env if env is not None else os.environ
    tokens = _parse_tokens(src.get("CULL_API_KEYS"))
    scopes = _parse_scopes(src.get("CULL_API_KEY_SCOPES"), tokens)
    rate = _parse_rate_limit(src.get("CULL_API_KEY_RATE_LIMIT"))
    return AuthConfig(
        tokens=frozenset(tokens),
        scopes=scopes,
        rate_limit_per_min=rate,
    )


# ── Token lookup ─────────────────────────────────────────────────────────────

def constant_time_match(candidate: str, tokens: Iterable[str]) -> str | None:
    """Return the token that matches ``candidate`` or ``None``.

    Compares every candidate to every configured token with
    :func:`hmac.compare_digest` so timing analysis can't distinguish a wrong
    token from an early-rejected one.
    """
    if not isinstance(candidate, str) or not candidate:
        return None
    match: str | None = None
    # Iterate every token regardless of an early hit so the loop's runtime is
    # constant in the number of configured tokens.
    for token in tokens:
        if hmac.compare_digest(candidate, token):
            match = token
    return match


def _extract_token(headers) -> str | None:
    """Pull a token out of the request headers.

    Accepts ``Authorization: Bearer <token>`` or the custom ``X-Cull-API-Key``
    header. Case-insensitive on the scheme, whitespace-tolerant on the value.
    """
    api_key = headers.get(_HEADER_APIKEY, "").strip()
    if api_key:
        return api_key
    auth = headers.get(_HEADER_BEARER, "").strip()
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, value = parts
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


# ── Scope classification ─────────────────────────────────────────────────────

def required_scope(method: str, path: str) -> str:
    """Return the scope needed to reach ``method path``.

    Admin-tier URLs override everything (a ``GET`` on ``/api/settings`` still
    needs ``admin``). Otherwise reads → ``read``, writes → ``curator``.
    """
    lowered_path = path or ""
    if any(p.match(lowered_path) for p in _ADMIN_URL_PATTERNS):
        return SCOPE_ADMIN
    if any(frag in lowered_path for frag in _ADMIN_URL_SUBSTRINGS):
        return SCOPE_ADMIN
    if (method or "").upper() in _READ_METHODS:
        return SCOPE_READ
    return SCOPE_CURATOR


def scope_grants(needed: str, granted: frozenset[str]) -> bool:
    """Whether the ``granted`` scope set satisfies ``needed``.

    ``admin`` implicitly grants ``curator`` and ``read``; ``curator`` grants
    ``read``. Everything else is exact.
    """
    if not granted:
        return False
    if needed == SCOPE_READ:
        return bool(granted & {SCOPE_READ, SCOPE_CURATOR, SCOPE_ADMIN})
    if needed == SCOPE_CURATOR:
        return bool(granted & {SCOPE_CURATOR, SCOPE_ADMIN})
    if needed == SCOPE_ADMIN:
        return SCOPE_ADMIN in granted
    return False


def default_scopes_for(token: str, config: AuthConfig) -> frozenset[str]:
    """Return the scope set for ``token``, falling back to the default."""
    explicit = config.scopes.get(token)
    if explicit:
        return explicit
    return frozenset({SCOPE_READ, SCOPE_CURATOR})


# ── Rate limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Per-token, per-minute-bucket in-memory rate limiter.

    Simple fixed-window counter — sufficient for a single-machine dashboard.
    Keyed by ``(token, minute_bucket)`` so cleanup is cheap: an old bucket falls
    out on the first call after the minute rolls over.
    """

    def __init__(self, per_min: int = DEFAULT_RATE_LIMIT) -> None:
        self.per_min = max(1, int(per_min))
        self._lock = threading.Lock()
        # {(token, minute_bucket): count}
        self._counters: dict[tuple[str, int], int] = {}

    def _bucket(self, now: float | None = None) -> int:
        return int((now if now is not None else time.time()) // 60)

    def check_and_add(self, token: str, *, now: float | None = None) -> bool:
        """Consume 1 slot for ``token``. Returns ``True`` when allowed."""
        if not token:
            return True  # loopback / disabled → no limit
        bucket = self._bucket(now)
        key = (token, bucket)
        with self._lock:
            # Drop stale buckets for this token so the map can't grow unbounded.
            stale = [k for k in self._counters if k[0] == token and k[1] != bucket]
            for k in stale:
                self._counters.pop(k, None)
            count = self._counters.get(key, 0) + 1
            if count > self.per_min:
                return False
            self._counters[key] = count
            return True

    def reset(self) -> None:
        """Drop every counter — tests call this between cases."""
        with self._lock:
            self._counters.clear()


# ── Middleware ───────────────────────────────────────────────────────────────

def _err_response(message: str, code: int):
    """Return a JSON error envelope matching ``_err`` in the dashboard.

    Kept local so this module has no runtime dependency on ``dashboard_enhanced``
    (avoids a circular import when tests reload the dashboard).
    """
    resp = jsonify({"success": False, "data": None, "error": message})
    resp.status_code = code
    return resp


def _is_api_path(path: str) -> bool:
    """API routes are the only surface auth applies to."""
    return isinstance(path, str) and path.startswith("/api/")


def install(
    app: Flask,
    *,
    config: AuthConfig | None = None,
    is_loopback: Callable[[], bool] | None = None,
    limiter: RateLimiter | None = None,
) -> RateLimiter:
    """Wire the auth middleware into ``app``.

    ``config`` / ``is_loopback`` / ``limiter`` are injectable so tests can drive
    behaviour without touching env vars. Returns the rate limiter so tests can
    inspect / reset it. Idempotent per-app: repeated calls replace the previous
    installation cleanly.
    """
    cfg = config if config is not None else load_config_from_env()
    loopback_fn = is_loopback if is_loopback is not None else _default_loopback

    rate = limiter if limiter is not None else RateLimiter(cfg.rate_limit_per_min)
    if limiter is None and cfg.rate_limit_per_min != rate.per_min:
        rate.per_min = cfg.rate_limit_per_min

    # Stash config on the app so introspection / tests can read it.
    app.extensions.setdefault("cull_api_auth", {})
    app.extensions["cull_api_auth"] = {
        "config": cfg,
        "limiter": rate,
        "is_loopback": loopback_fn,
    }

    @app.before_request
    def _cull_api_auth_gate():  # noqa: D401 - short, single-purpose
        # Non-/api/* routes (the HTML shell, static-ish endpoints) are outside
        # this middleware — the loopback gate on sensitive endpoints stays the
        # authority there.
        if not _is_api_path(request.path or ""):
            return None
        # Auth disabled → behave exactly as before (loopback checks in handlers
        # still fire on sensitive endpoints).
        if not cfg.enabled:
            return None
        # Loopback callers always pass — keeps localhost tooling working when
        # auth is on (an operator running curl from the same machine).
        try:
            if loopback_fn():
                return None
        except Exception:  # noqa: BLE001 - never let auth crash the app
            logger.warning("api_auth: loopback check raised; treating as remote")

        token = _extract_token(request.headers)
        matched = constant_time_match(token or "", cfg.tokens) if token else None
        if matched is None:
            return _err_response("invalid API key", 401)

        needed = required_scope(request.method, request.path or "")
        granted = default_scopes_for(matched, cfg)
        if not scope_grants(needed, granted):
            return _err_response("insufficient scope", 403)

        # Rate-limit writes only (reads stay free so a dashboard tab polling
        # /api/status can't lock itself out).
        if needed != SCOPE_READ:
            if not rate.check_and_add(matched):
                return _err_response("rate limit exceeded", 429)

        return None

    return rate


def _default_loopback() -> bool:
    """Fallback loopback classifier when the dashboard's helper isn't injected.

    Kept minimal to avoid the circular import — the dashboard passes its own
    ``_wave_is_loopback_request`` via ``install(..., is_loopback=...)``.
    """
    addr = (getattr(request, "remote_addr", "") or "").strip()
    if not addr:
        return False
    if addr in ("127.0.0.1", "::1", "localhost"):
        return True
    return addr.startswith("::ffff:127.")


__all__ = [
    "AuthConfig",
    "RateLimiter",
    "SCOPE_READ",
    "SCOPE_CURATOR",
    "SCOPE_ADMIN",
    "MIN_TOKEN_LEN",
    "DEFAULT_RATE_LIMIT",
    "constant_time_match",
    "default_scopes_for",
    "install",
    "load_config_from_env",
    "required_scope",
    "scope_grants",
]
