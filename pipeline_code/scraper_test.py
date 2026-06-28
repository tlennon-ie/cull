"""scraper_test.py — Live connectivity and auth tests for each cull scraper.

Public API
----------
test_scraper(name, config=None, env=None) -> dict
    Return {"ok": bool, "message": str, "latency_ms": int | None, "detail": str}.
    Never raises. `name` must be one of SUPPORTED.

SUPPORTED : tuple[str, ...]
    Names this module can test.

Design
------
Two-stage check per scraper:

  Stage 1 (offline) — credential / config present and well-formed?
      If not -> ok=False, precise message, latency_ms=None, zero network.

  Stage 2 (live) — cheap authenticated HTTP call via _http_request().
      Tests monkeypatch _http_request; production code calls it for real.

All HTTP is routed through _http_request() so tests never touch the network.
Timeout is 8 seconds; responses are mapped to clear user-facing messages.
"""
from __future__ import annotations

import base64
import ipaddress
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from pipeline_logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------

SUPPORTED: tuple[str, ...] = (
    "X.com",
    "Discord-1",
    "Civitai-Com",
    "Civitai-Red",
    "Reddit",
    "Web",
    "Gallery-DL",
    "Local",
)

_TIMEOUT = 8.0

# Sentinel for "no latency measured"
_NO_LATENCY: None = None


# ---------------------------------------------------------------------------
# Private HTTP helper — monkeypatched in tests
# ---------------------------------------------------------------------------

def _http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = _TIMEOUT,
) -> tuple[int, str]:
    """Execute a real HTTP request; return (status_code, body_snippet).

    Tests replace this function via monkeypatch so no real network is hit.
    """
    resp = requests.request(method, url, headers=headers or {}, timeout=timeout)
    return resp.status_code, resp.text[:500]


# ---------------------------------------------------------------------------
# Result builder helpers
# ---------------------------------------------------------------------------

def _ok(message: str, latency_ms: int | None = _NO_LATENCY, detail: str = "") -> dict:
    return {"ok": True, "message": message, "latency_ms": latency_ms, "detail": detail}


def _fail(message: str, latency_ms: int | None = _NO_LATENCY, detail: str = "") -> dict:
    return {"ok": False, "message": message, "latency_ms": latency_ms, "detail": detail}


def _unsupported(name: str) -> dict:
    return _fail(f"unsupported scraper: {name!r}")


# ---------------------------------------------------------------------------
# Credential resolution (uses caller-supplied env dict, falls back to os.environ)
# ---------------------------------------------------------------------------

def _get(key: str, env: dict[str, Any] | None) -> str | None:
    """Return env[key] stripped, or None if absent/empty/non-string."""
    source = env if env is not None else os.environ
    val = source.get(key)
    if not isinstance(val, str):
        return None
    val = val.strip()
    return val or None


def _is_safe_public_url(url: str) -> bool:
    """True only for an http(s) URL aimed at a public host.

    Blocks ``file://`` and other schemes, ``localhost``, and literal
    loopback / private / link-local / reserved IPs so the per-scraper Test
    button can't be turned into an SSRF primitive against the host's own
    services or a cloud metadata endpoint. We deliberately do NOT resolve DNS
    (keeps the check hermetic and fast); a hostname that *resolves* to a private
    IP is out of scope for a single-machine localhost tool.
    """
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001 - never raise from a validator
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip()
    if not host:
        return False
    low = host.lower()
    if low == "localhost" or low.endswith(".localhost"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (ip.is_private or ip.is_loopback or ip.is_link_local
                           or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
        return False
    return True


# ---------------------------------------------------------------------------
# HTTP outcome mapper (status -> (ok, message))
# ---------------------------------------------------------------------------

def _status_to_result(
    status: int,
    latency_ms: int,
    *,
    detail: str = "",
) -> dict:
    if 200 <= status < 300:
        return _ok("authenticated OK", latency_ms=latency_ms, detail=detail)
    if status in (401, 403):
        return _fail(
            f"auth rejected (HTTP {status}) — check key/cookies/token",
            latency_ms=latency_ms,
            detail=detail,
        )
    return _fail(
        f"unexpected HTTP {status}",
        latency_ms=latency_ms,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Live call helper — wraps _http_request, catches all exceptions
# ---------------------------------------------------------------------------

def _live_call(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    detail: str = "",
) -> dict:
    """Call _http_request, measure latency, and return a typed result dict.

    Exceptions from _http_request are caught so test_scraper never raises.
    """
    t0 = time.monotonic()
    try:
        status, body = _http_request(method, url, headers=headers, timeout=_TIMEOUT)
        latency_ms = int((time.monotonic() - t0) * 1000)
        # Never forward the raw response body to the client by default — it could
        # echo token hints or internal error text. Callers pass an explicit detail.
        return _status_to_result(status, latency_ms, detail=detail or "")
    except requests.exceptions.Timeout:
        return _fail(
            "could not connect (timed out)",
            detail=f"GET {url} timed out after {_TIMEOUT}s",
        )
    except requests.exceptions.ConnectionError as exc:
        return _fail(
            "could not connect (connection error)",
            detail=str(exc)[:200],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("_live_call unexpected error for %s: %s", url, exc)
        return _fail(
            f"could not connect (error: {type(exc).__name__})",
            detail=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# Per-scraper check functions
# ---------------------------------------------------------------------------

def _check_civitai_com(config: dict, env: dict | None) -> dict:
    """Civitai.com — requires CIVITAI_API_KEY.

    Live call: GET https://civitai.com/api/v1/images?limit=1
    with Authorization: Bearer <key>.
    """
    key = _get("CIVITAI_API_KEY", env)
    if key is None:
        return _fail("CIVITAI_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}"}
    return _live_call(
        "GET",
        "https://civitai.com/api/v1/images?limit=1",
        headers=headers,
        detail="civitai.com images API",
    )


def _check_civitai_red(config: dict, env: dict | None) -> dict:
    """Civitai.red — prefers CIVITAI_API_RED_KEY, falls back to CIVITAI_API_KEY.

    Live call: GET https://civitai.red/api/v1/images?limit=1
    with Authorization: Bearer <key>.
    """
    key = _get("CIVITAI_API_RED_KEY", env) or _get("CIVITAI_API_KEY", env)
    if key is None:
        return _fail("CIVITAI_API_RED_KEY (or CIVITAI_API_KEY) not set")
    headers = {"Authorization": f"Bearer {key}"}
    return _live_call(
        "GET",
        "https://civitai.red/api/v1/images?limit=1",
        headers=headers,
        detail="civitai.red images API",
    )


# Public web bearer token embedded in x.com's own JavaScript. NOT a secret — it
# is sent by every browser on every request — but the v1.1 API rejects a request
# that carries only the auth cookies. The authenticated web call needs all three:
# this bearer, the x-csrf-token header (== the ct0 cookie value), and the cookies.
_TWITTER_WEB_BEARER = (
    "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs="
    "1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)


def _parse_cookie_pairs(raw: str) -> dict[str, str]:
    """Parse a raw ``name=value; name2=value2`` Cookie header into a dict."""
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" in part:
            name, value = part.split("=", 1)
            out[name.strip()] = value.strip()
    return out


def _check_x_com(config: dict, env: dict | None) -> dict:
    """X.com — verify a logged-in cookie string actually authenticates.

    Stage 1 (offline): the cookie string must contain both ``auth_token`` and
    ``ct0`` (paste the FULL ``Cookie:`` header from a logged-in x.com request).
    Stage 2 (live): GET v1.1 ``account/verify_credentials.json`` with the public
    web bearer + ``x-csrf-token: <ct0>`` + the cookies. A cookie-ONLY request
    always 401s, which is why the old check failed for valid cookies; with the
    bearer + csrf, 200 == the session is live, 401/403 == expired/invalid.
    """
    raw_cookies = _get("TWITTER_COOKIES", env)
    if raw_cookies is None:
        return _fail("TWITTER_COOKIES not set")

    cookies = _parse_cookie_pairs(raw_cookies)
    if "auth_token" not in cookies:
        return _fail(
            "TWITTER_COOKIES missing 'auth_token' — paste the FULL Cookie header "
            "from a logged-in x.com request (auth_token=...; ct0=...; ...)"
        )
    if not cookies.get("ct0"):
        return _fail(
            "TWITTER_COOKIES missing 'ct0' — paste the FULL Cookie header from a "
            "logged-in x.com request (auth_token=...; ct0=...; ...)"
        )

    headers = {
        "Authorization": _TWITTER_WEB_BEARER,
        "x-csrf-token": cookies["ct0"],
        "Cookie": raw_cookies,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
        ),
    }
    # Best-effort LIVE verification. X has deprecated most v1.1 endpoints (they now
    # 404), so we try the surviving web-app paths and DEGRADE GRACEFULLY:
    #   * 2xx          -> session is live (cookies valid).
    #   * 401/403      -> cookies rejected (invalid/expired) — a real failure.
    #   * 404/5xx/err  -> the endpoint moved or blocks plain HTTP; do NOT condemn
    #                     structurally-valid cookies — the scraper signs in with a
    #                     real browser (Playwright), not this probe.
    t0 = time.monotonic()
    last_status: int | None = None
    for url in (
        "https://api.x.com/1.1/account/settings.json",
        "https://x.com/i/api/1.1/account/settings.json",
        "https://api.twitter.com/1.1/account/verify_credentials.json",
    ):
        try:
            status, _body = _http_request("GET", url, headers=headers, timeout=_TIMEOUT)
        except Exception:  # noqa: BLE001 - try the next endpoint
            continue
        last_status = status
        latency_ms = int((time.monotonic() - t0) * 1000)
        if 200 <= status < 300:
            return _ok("authenticated — session is live", latency_ms=latency_ms,
                       detail="x.com account settings")
        if status in (401, 403):
            return _fail(
                f"auth rejected (HTTP {status}) — cookies are invalid or expired; "
                "re-copy the full Cookie header from a logged-in x.com tab",
                latency_ms=latency_ms, detail="x.com account settings")
        # 404 / other -> endpoint moved; fall through to the next candidate.
    latency_ms = int((time.monotonic() - t0) * 1000)
    note = f"x.com API returned HTTP {last_status}" if last_status else "x.com API unreachable"
    return _ok(
        "cookies look valid (auth_token + ct0 present) — could NOT verify live "
        f"({note}); the scraper signs in with a real browser, so valid cookies "
        "should still work",
        latency_ms=latency_ms, detail="structural check only")


def _check_discord(config: dict, env: dict | None) -> dict:
    """Discord — requires DISCORD_BOT_TOKEN.

    Auth mode (DISCORD_AUTH_MODE): 'bot' -> 'Bot <token>', 'user' -> raw token.
    Default ('auto') starts as 'Bot <token>' per scraper_discord.py behaviour.

    Live call: GET https://discord.com/api/v10/users/@me
    """
    token = _get("DISCORD_BOT_TOKEN", env)
    if token is None:
        return _fail("DISCORD_BOT_TOKEN not set")

    auth_mode = (_get("DISCORD_AUTH_MODE", env) or "auto").lower()
    if auth_mode == "user":
        auth_header = token
    else:
        # "bot" or "auto" both start as Bot prefix (mirrors scraper_discord.py)
        auth_header = f"Bot {token}"

    headers = {"Authorization": auth_header}
    return _live_call(
        "GET",
        "https://discord.com/api/v10/users/@me",
        headers=headers,
        detail=f"discord users/@me (mode={auth_mode})",
    )


def _check_reddit(config: dict, env: dict | None) -> dict:
    """Reddit — REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET enable OAuth token fetch.

    If credentials are absent, do a simple public ping of the Reddit JSON API
    to confirm basic connectivity (scraper_web.py uses unauthenticated requests).

    With credentials: POST https://www.reddit.com/api/v1/access_token
    Without:          GET  https://www.reddit.com/r/all.json?limit=1 (public ping)

    Both paths are routed through _http_request (or _http_request_with_basic_auth)
    so tests can monkeypatch either helper.
    """
    client_id = _get("REDDIT_CLIENT_ID", env)
    client_secret = _get("REDDIT_CLIENT_SECRET", env)
    user_agent = _get("REDDIT_USER_AGENT", env) or "cull/test"

    if client_id and client_secret:
        # OAuth client-credentials token fetch via the testable helper
        t0 = time.monotonic()
        try:
            status, body = _http_request(
                "POST",
                "https://www.reddit.com/api/v1/access_token",
                headers={
                    "User-Agent": user_agent,
                    "Content-Type": "application/x-www-form-urlencoded",
                    # Basic auth encoded into the Authorization header so the
                    # monkeypatched _http_request receives it without needing
                    # requests.post(auth=...) which can't be intercepted.
                    "Authorization": "Basic " + base64.b64encode(
                        f"{client_id}:{client_secret}".encode()
                    ).decode(),
                },
                timeout=_TIMEOUT,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            return _status_to_result(status, latency_ms, detail="reddit OAuth token")
        except requests.exceptions.Timeout:
            return _fail("could not connect (timed out)", detail="reddit OAuth token endpoint")
        except requests.exceptions.ConnectionError as exc:
            return _fail("could not connect (connection error)", detail=str(exc)[:200])
        except Exception as exc:  # noqa: BLE001
            return _fail(f"could not connect (error: {type(exc).__name__})", detail=str(exc)[:200])
    else:
        # No credentials — unauthenticated public ping
        result = _live_call(
            "GET",
            "https://www.reddit.com/r/all.json?limit=1",
            headers={"User-Agent": user_agent},
            detail="reddit public JSON API (no credentials)",
        )
        if result["ok"]:
            result = dict(result)
            result["message"] = "public API reachable (no OAuth credentials configured)"
        return result


def _check_web(config: dict, env: dict | None) -> dict:
    """Generic web scraper — no dedicated auth.

    If config contains a 'target_url', probe it. Otherwise report that
    there is no live auth to test.
    """
    target = None
    if isinstance(config, dict):
        target = config.get("target_url") or config.get("url")

    if not target:
        return _ok(
            "no live auth to test (Web scraper uses unauthenticated public URLs)",
            detail="no target_url configured",
        )

    if not _is_safe_public_url(str(target)):
        return _fail(
            "refusing to probe a non-public or non-HTTP URL",
            detail="Web test only allows http(s) URLs to public hosts",
        )
    return _live_call("GET", str(target), detail=f"web target: {target}")


def _check_gallery_dl(config: dict, env: dict | None) -> dict:
    """Gallery-DL — offline checks only (import + optional file paths).

    1. gallery_dl package must be importable.
    2. If config['cookies_file'] is set, the file must exist on disk.
    3. If config['config_path'] is set, the file must exist on disk.

    No live HTTP (gallery-dl handles site auth internally).
    """
    try:
        import gallery_dl  # noqa: F401
    except Exception:  # noqa: BLE001 - corrupt install / missing transitive dep
        return _fail(
            "gallery-dl not importable — ensure it is in requirements.txt and "
            "your virtualenv is active"
        )

    if isinstance(config, dict):
        cookies_file = config.get("cookies_file")
        if cookies_file and isinstance(cookies_file, str):
            if not Path(cookies_file).exists():
                return _fail(
                    f"cookies_file not found: {cookies_file}",
                    detail="check GALLERY_DL_COOKIES_FILE path",
                )

        config_path = config.get("config_path")
        if config_path and isinstance(config_path, str):
            if not Path(config_path).exists():
                return _fail(
                    f"config_path not found: {config_path}",
                    detail="check GALLERY_DL_CONFIG_PATH",
                )

    return _ok("gallery-dl installed and config paths verified")


def _check_local(config: dict, env: dict | None) -> dict:
    """Local folder import — checks dir exists and is readable.

    config['dir'] must be a string pointing to an existing directory.
    """
    folder: Any = None
    if isinstance(config, dict):
        folder = config.get("dir")

    if not folder or not isinstance(folder, str):
        return _fail(
            "dir not configured — pass config={'dir': '/path/to/folder'}",
            detail="no dir key in config",
        )

    p = Path(folder)
    if not p.exists():
        return _fail(f"dir does not exist: {folder}", detail="path not found")
    if not p.is_dir():
        return _fail(f"dir is not a directory: {folder}", detail="path is a file")

    # Quick readability check: try listing the directory
    try:
        list(p.iterdir())
    except PermissionError:
        return _fail(f"dir is not readable: {folder}", detail="permission denied")

    return _ok(f"dir is accessible: {folder}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_CHECKERS = {
    "Civitai-Com": _check_civitai_com,
    "Civitai-Red": _check_civitai_red,
    "X.com":        _check_x_com,
    "Discord-1":    _check_discord,
    "Reddit":       _check_reddit,
    "Web":          _check_web,
    "Gallery-DL":   _check_gallery_dl,
    "Local":        _check_local,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def test_scraper(
    name: str,
    config: dict | None = None,
    env: dict | None = None,
) -> dict:
    """Return {"ok": bool, "message": str, "latency_ms": int | None, "detail": str}.

    Never raises. `name` must be one of SUPPORTED. `config` carries per-job
    scraper sub-config (keys: x_accounts, reddit_subreddits, discord_channels_json,
    civitai_domains, gallery_dl{enabled,urls,limit_per_url,cookies_file,config_path},
    local_imports[{name,dir,enabled}], target_url).

    `env` overrides os.environ for credential lookup (defaults to os.environ).
    """
    checker = _CHECKERS.get(name)
    if checker is None:
        return _unsupported(name)

    safe_config: dict = config if isinstance(config, dict) else {}
    try:
        return checker(safe_config, env)
    except Exception as exc:  # noqa: BLE001
        # Log the full exception server-side; never leak str(exc) (may carry a
        # credential value or path) into the user-facing message.
        logger.exception("unexpected error in test_scraper(%r): %s", name, exc)
        return _fail(
            f"internal error: {type(exc).__name__}",
            detail="unexpected exception — check logs",
        )


__all__ = ["test_scraper", "SUPPORTED", "_http_request"]
