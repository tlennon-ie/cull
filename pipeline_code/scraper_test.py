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
import concurrent.futures
import ipaddress
import json
import os
import tempfile
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
#
# ``verified`` and ``warning`` are additive fields that the dashboard already
# renders. ``verified=True`` means the check exchanged a real authenticated
# round-trip with the target service (not just a structural sanity pass);
# ``warning`` carries an amber note for "ok but not fully verified" results so
# a user can see WHY a green tick has an asterisk next to it. Both default to
# safe values so every legacy caller (and every existing test that only reads
# ``ok``/``message``/``latency_ms``/``detail``) sees the same behaviour.

def _ok(
    message: str,
    latency_ms: int | None = _NO_LATENCY,
    detail: str = "",
    *,
    verified: bool = True,
    warning: str = "",
) -> dict:
    return {
        "ok": True,
        "message": message,
        "latency_ms": latency_ms,
        "detail": detail,
        "verified": verified,
        "warning": warning,
    }


def _fail(
    message: str,
    latency_ms: int | None = _NO_LATENCY,
    detail: str = "",
    *,
    warning: str = "",
) -> dict:
    return {
        "ok": False,
        "message": message,
        "latency_ms": latency_ms,
        "detail": detail,
        "verified": False,
        "warning": warning,
    }


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
    host: str = "",
) -> dict:
    """Map an HTTP status to a scraper test result.

    ``host`` is included in the error message so a failed check tells the
    operator WHICH host was hit (invaluable when a scraper has multiple
    upstreams / a CDN split / a moving API version).
    """
    if 200 <= status < 300:
        return _ok("authenticated OK", latency_ms=latency_ms, detail=detail)
    host_suffix = f" [host: {host}]" if host else ""
    if status in (401, 403):
        return _fail(
            f"auth rejected (HTTP {status}) — check key/cookies/token{host_suffix}",
            latency_ms=latency_ms,
            detail=detail,
        )
    return _fail(
        f"unexpected HTTP {status}{host_suffix}",
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
    Network errors carry the host name in the user-facing message so a failed
    live probe tells the operator which upstream is unreachable.
    """
    host = ""
    try:
        host = (urlparse(url).hostname or "")
    except Exception:  # noqa: BLE001 - never let a URL parse block a probe
        host = ""
    host_suffix = f" [host: {host}]" if host else ""
    t0 = time.monotonic()
    try:
        status, body = _http_request(method, url, headers=headers, timeout=_TIMEOUT)
        latency_ms = int((time.monotonic() - t0) * 1000)
        # Never forward the raw response body to the client by default — it could
        # echo token hints or internal error text. Callers pass an explicit detail.
        return _status_to_result(status, latency_ms, detail=detail or "", host=host)
    except requests.exceptions.Timeout:
        return _fail(
            f"could not connect (timed out){host_suffix}",
            detail=f"GET {url} timed out after {_TIMEOUT}s",
        )
    except requests.exceptions.ConnectionError as exc:
        return _fail(
            f"could not connect (connection error){host_suffix}",
            detail=str(exc)[:200],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("_live_call unexpected error for %s: %s", url, exc)
        return _fail(
            f"could not connect (error: {type(exc).__name__}){host_suffix}",
            detail=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# Per-scraper check functions
# ---------------------------------------------------------------------------

_CIVITAI_NSFW_HINT = (
    "NSFW visibility depends on the 'Mature content' toggle in your "
    "Civitai account settings (civitai.com/user/account) — flip it on "
    "there if the scraper is only returning SFW results."
)


def _check_civitai_com(config: dict, env: dict | None) -> dict:
    """Civitai.com — requires CIVITAI_API_KEY.

    Live call: GET https://civitai.com/api/v1/images?limit=1
    with Authorization: Bearer <key>. A 200 confirms the KEY works; note that
    NSFW content visibility is a separate per-account toggle on Civitai (see
    ``_CIVITAI_NSFW_HINT``) that no API probe can flip from here — we surface
    that fact in the detail string so a puzzled user knows where to look.
    """
    key = _get("CIVITAI_API_KEY", env)
    if key is None:
        return _fail("CIVITAI_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}"}
    result = _live_call(
        "GET",
        "https://civitai.com/api/v1/images?limit=1",
        headers=headers,
        detail="civitai.com images API",
    )
    if result.get("ok"):
        # Preserve caller-facing message; extend detail with the NSFW hint.
        result = dict(result)
        base_detail = result.get("detail") or "civitai.com images API"
        result["detail"] = f"{base_detail} — {_CIVITAI_NSFW_HINT}"
    return result


def _check_civitai_red(config: dict, env: dict | None) -> dict:
    """Civitai.red — prefers CIVITAI_API_RED_KEY, falls back to CIVITAI_API_KEY.

    Live call: GET https://civitai.red/api/v1/images?limit=1
    with Authorization: Bearer <key>. Same NSFW-toggle caveat as .com applies.
    """
    key = _get("CIVITAI_API_RED_KEY", env) or _get("CIVITAI_API_KEY", env)
    if key is None:
        return _fail("CIVITAI_API_RED_KEY (or CIVITAI_API_KEY) not set")
    headers = {"Authorization": f"Bearer {key}"}
    result = _live_call(
        "GET",
        "https://civitai.red/api/v1/images?limit=1",
        headers=headers,
        detail="civitai.red images API",
    )
    if result.get("ok"):
        result = dict(result)
        base_detail = result.get("detail") or "civitai.red images API"
        result["detail"] = f"{base_detail} — {_CIVITAI_NSFW_HINT}"
    return result


# Public web bearer token embedded in x.com's own JavaScript. NOT a secret — it
# is sent by every browser on every request — but the v1.1 API rejects a request
# that carries only the auth cookies. The authenticated web call needs all three:
# this bearer, the x-csrf-token header (== the ct0 cookie value), and the cookies.
_TWITTER_WEB_BEARER = (
    "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs="
    "1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)


def _parse_cookie_pairs(raw: str) -> dict[str, str]:
    """Parse a raw ``name=value; name2=value2`` Cookie header into a dict.

    Tolerates a ``Cookie:`` prefix (users often paste the entire request header
    line, header name and all) so we don't reject a well-formed paste on a
    trivial format quirk.
    """
    raw = (raw or "").strip()
    if raw[:7].lower() == "cookie:":
        raw = raw[7:].strip()
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" in part:
            name, value = part.split("=", 1)
            out[name.strip()] = value.strip()
    return out


def _read_netscape_cookies(path: str) -> tuple[dict[str, str], int | None, str | None]:
    """Read a Netscape cookies.txt file into (name→value, nearest_expiry, err).

    Never raises. On malformed content, returns whatever we could parse plus an
    error string so the caller can surface it as a warning (rather than
    crashing the test). ``nearest_expiry`` is the SMALLEST non-zero ``expires``
    timestamp across all cookies, or ``None`` if no cookie declared one — used
    to warn the user when their session is about to lapse.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return {}, None, f"cookies file not found: {p}"
    if not p.is_file():
        return {}, None, f"cookies path is not a file: {p}"

    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {}, None, f"could not read cookies file ({exc})"

    first_line = raw.splitlines()[0] if raw else ""
    header_ok = first_line.strip().lower().startswith("# netscape http cookie file")

    cookies: dict[str, str] = {}
    nearest: int | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Netscape format: domain \t include_subdomains \t path \t secure \t expires \t name \t value
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        try:
            expires = int(parts[4])
        except ValueError:
            expires = 0
        name = parts[5].strip()
        value = parts[6].strip()
        if not name:
            continue
        cookies[name] = value
        # 0 = session cookie (no persistent expiry); skip it for the "nearest"
        # bookkeeping so a mix of session + persistent cookies doesn't spam us.
        if expires > 0 and (nearest is None or expires < nearest):
            nearest = expires

    err = None if header_ok else (
        "cookies file present but missing the '# Netscape HTTP Cookie File' "
        "header — parsed anyway, but re-export from your browser if this fails"
    )
    return cookies, nearest, err


def _load_x_cookies(env: dict | None) -> tuple[dict[str, str], str, int | None, str | None]:
    """Resolve X cookies from either an env-string OR an on-disk Netscape file.

    Returns (cookies_map, raw_cookie_header, nearest_expiry_unix, error_or_None).
    The raw header is what the live HTTP probe will send in the ``Cookie:``
    header; if the source was a file, we reconstruct it as ``k=v; k2=v2``.
    """
    txt_path = _get("X_COOKIES_TXT", env)
    if txt_path:
        cookies, nearest, err = _read_netscape_cookies(txt_path)
        header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        return cookies, header, nearest, err

    raw = _get("TWITTER_COOKIES", env)
    if raw is None:
        return {}, "", None, "TWITTER_COOKIES not set"

    stripped = raw.strip()
    if stripped[:7].lower() == "cookie:":
        stripped = stripped[7:].strip()
    return _parse_cookie_pairs(stripped), stripped, None, None


def _check_x_com(config: dict, env: dict | None) -> dict:
    """X.com — verify a logged-in cookie string actually authenticates.

    Two ways to supply the session:
      * ``TWITTER_COOKIES`` — raw ``name=value; name2=value2`` header (what most
        users paste from a browser's Dev Tools). Sessions have no on-disk
        expiry metadata; only the live probe can tell us if they're stale.
      * ``X_COOKIES_TXT`` — a Netscape cookies.txt path. Preferred when
        available because we can also parse the ``expires`` column and warn
        early ("cookies present but session appears expired — re-export").

    Stage 1 (offline): the cookie set must contain both ``auth_token`` and
    ``ct0`` (paste the FULL ``Cookie:`` header from a logged-in x.com request,
    or export a full cookies.txt).
    Stage 2 (live): GET v1.1 ``account/settings.json`` with the public web
    bearer + ``x-csrf-token: <ct0>`` + the cookies. A cookie-ONLY request
    always 401s, which is why the old check failed for valid cookies; with the
    bearer + csrf, 200 == the session is live, 401/403 == expired/invalid.
    """
    cookies, raw_cookie_header, nearest_expiry, load_err = _load_x_cookies(env)

    # If the operator explicitly pointed us at a cookies.txt file that we could
    # not read, that's the failure — say so BEFORE the structural check runs.
    if load_err and _get("X_COOKIES_TXT", env):
        return _fail(load_err, detail="X_COOKIES_TXT")

    if not cookies and load_err:
        return _fail(load_err)

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

    # Early expiry warning (only possible when we have persistent expiry data,
    # i.e. cookies came from a Netscape file). Sub-7-days = amber; already
    # expired = red — but only when nearest_expiry actually pins auth_token
    # / ct0. To avoid crying wolf on a session cookie in the mix we required
    # ``expires > 0`` in the parser.
    expiry_warning = ""
    if nearest_expiry is not None:
        now = int(time.time())
        remaining = nearest_expiry - now
        if remaining <= 0:
            return _fail(
                "cookies present but session appears expired — re-export from "
                "browser (cookies.txt)",
                detail=f"expired {abs(remaining)}s ago",
            )
        if remaining < 7 * 86400:
            days = max(1, remaining // 86400)
            expiry_warning = (
                f"cookies expire in ~{days} day(s) — re-export soon from browser"
            )

    headers = {
        "Authorization": _TWITTER_WEB_BEARER,
        "x-csrf-token": cookies["ct0"],
        "Cookie": raw_cookie_header,
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
            return _ok(
                "authenticated — session is live",
                latency_ms=latency_ms,
                detail="x.com account settings",
                verified=True,
                warning=expiry_warning,
            )
        if status in (401, 403):
            return _fail(
                f"auth rejected (HTTP {status}) — cookies are invalid or expired; "
                "re-copy the full Cookie header from a logged-in x.com tab",
                latency_ms=latency_ms, detail="x.com account settings")
        # 404 / other -> endpoint moved; fall through to the next candidate.
    latency_ms = int((time.monotonic() - t0) * 1000)
    note = f"x.com API returned HTTP {last_status}" if last_status else "x.com API unreachable"
    amber = (
        "structural only, live not verified — the scraper signs in with a "
        "real browser (Playwright) so valid cookies should still work"
    )
    if expiry_warning:
        amber = f"{amber} · {expiry_warning}"
    return _ok(
        "cookies look valid (auth_token + ct0 present) — could NOT verify live "
        f"({note}); the scraper signs in with a real browser, so valid cookies "
        "should still work",
        latency_ms=latency_ms,
        detail="structural check only",
        verified=False,
        warning=amber,
    )


def _check_discord(config: dict, env: dict | None) -> dict:
    """Discord — requires DISCORD_BOT_TOKEN.

    Auth mode (DISCORD_AUTH_MODE): 'bot' -> 'Bot <token>', 'user' -> raw token.
    Default ('auto') starts as 'Bot <token>' per scraper_discord.py behaviour.

    Live call: GET https://discord.com/api/v10/users/@me — returns the
    authenticated user object (bot or user). A 401 here means the token is
    wrong for the current mode; the operator's usual fix is to flip
    DISCORD_AUTH_MODE between ``bot`` / ``user`` and retry.
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
    result = _live_call(
        "GET",
        "https://discord.com/api/v10/users/@me",
        headers=headers,
        detail=f"discord users/@me (mode={auth_mode})",
    )
    # On a 401 in "bot" or "auto" mode, the token is very likely a personal
    # user token — surface the actionable fix (flip mode) in the warning so
    # the operator doesn't need to grep the scraper docs.
    if not result["ok"] and auth_mode in ("bot", "auto"):
        msg = result.get("message", "")
        if "401" in msg or "auth rejected" in msg.lower():
            result = dict(result)
            result["warning"] = (
                "token rejected as a bot — if this is a user account token, "
                "set DISCORD_AUTH_MODE=user in Global Settings"
            )
    return result


def _check_reddit(config: dict, env: dict | None) -> dict:
    """Reddit — REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET enable OAuth token fetch.

    If credentials are absent, do a simple public ping of a KNOWN-GOOD listing
    to confirm both connectivity AND that Reddit isn't 403-ing our default UA.
    Reddit's anti-bot filter blocks generic UAs — we use a real Chrome UA plus
    an ``Accept: application/json`` header so the probe mirrors the browser
    session the scraper actually uses at runtime.

    With credentials: POST https://www.reddit.com/api/v1/access_token
    Without:          GET  https://www.reddit.com/r/pics/hot.json?limit=1
    """
    client_id = _get("REDDIT_CLIENT_ID", env)
    client_secret = _get("REDDIT_CLIENT_SECRET", env)
    user_agent = _get("REDDIT_USER_AGENT", env) or (
        # Match the real scraper's default UA (scraper_web.py). Reddit 403s the
        # generic "python-requests/*" and "cull/test" UAs; a Chrome UA passes.
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

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
        # No credentials — unauthenticated public ping against a known-good
        # subreddit (r/pics is huge, public, and not quarantined), with the
        # ``Accept`` header the scraper actually sends. This catches Reddit's
        # UA-based 403 filter that a bare ``/r/all.json`` probe would miss.
        result = _live_call(
            "GET",
            "https://www.reddit.com/r/pics/hot.json?limit=1",
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
            detail="reddit public JSON API (no credentials)",
        )
        if result["ok"]:
            result = dict(result)
            result["message"] = (
                "public API reachable (no OAuth credentials configured) — "
                "the scraper uses a real browser (Playwright) at runtime, "
                "which bypasses Reddit's raw-UA 403s automatically"
            )
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

    1. ``gallery_dl`` package must be importable (and its version resolvable —
       a truly corrupt install imports but has no ``__version__``).
    2. If config['cookies_file'] is set, the file must exist AND parse as
       Netscape format (the format gallery-dl accepts).
    3. If config['config_path'] is set, the file must exist AND be valid JSON.

    No live HTTP (gallery-dl handles site auth internally). Per-URL live
    verification is a separate entry point — ``_check_gallery_dl_url`` — used
    by the dashboard's "Test this URL" button.
    """
    try:
        import gallery_dl  # noqa: F401
    except ImportError as exc:
        return _fail(
            "gallery-dl not importable — install it via `pip install -r "
            f"requirements.txt` in your virtualenv (import error: {exc})"
        )
    except Exception as exc:  # noqa: BLE001 - corrupt install / missing transitive dep
        return _fail(
            "gallery-dl import failed unexpectedly — reinstall via "
            f"`pip install --force-reinstall gallery-dl` ({type(exc).__name__}: {exc})"
        )

    version = getattr(gallery_dl, "__version__", None) or "unknown"

    if isinstance(config, dict):
        cookies_file = config.get("cookies_file")
        if cookies_file and isinstance(cookies_file, str):
            cookies_path = Path(cookies_file).expanduser()
            if not cookies_path.exists():
                return _fail(
                    f"cookies_file not found: {cookies_file}",
                    detail="check GALLERY_DL_COOKIES_FILE path",
                )
            if not cookies_path.is_file():
                return _fail(
                    f"cookies_file is not a regular file: {cookies_file}",
                    detail="check GALLERY_DL_COOKIES_FILE path",
                )
            # Best-effort Netscape header check — non-fatal if the header is
            # missing (gallery-dl itself is tolerant) but surface as a warning.
            try:
                head = cookies_path.read_text(encoding="utf-8", errors="replace")[:512]
            except OSError as exc:
                return _fail(
                    f"cookies_file unreadable: {exc}",
                    detail=str(cookies_path),
                )
            if not head.lstrip().lower().startswith("# netscape http cookie file"):
                return _ok(
                    f"gallery-dl v{version} installed; cookies_file present "
                    "but missing 'Netscape HTTP Cookie File' header",
                    verified=False,
                    warning=(
                        "cookies_file lacks the Netscape header — re-export "
                        "from your browser as cookies.txt if auth fails"
                    ),
                )

        config_path = config.get("config_path")
        if config_path and isinstance(config_path, str):
            cfg = Path(config_path).expanduser()
            if not cfg.exists():
                return _fail(
                    f"config_path not found: {config_path}",
                    detail="check GALLERY_DL_CONFIG_PATH",
                )
            if not cfg.is_file():
                return _fail(
                    f"config_path is not a regular file: {config_path}",
                    detail="check GALLERY_DL_CONFIG_PATH",
                )
            # JSON validation — a syntactically broken config is a very common
            # failure mode we should catch here rather than at scraper startup.
            try:
                json.loads(cfg.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError) as exc:
                return _fail(
                    f"config_path is not valid JSON: {exc}",
                    detail=str(cfg),
                )

    return _ok(f"gallery-dl v{version} installed and config paths verified")


_GALLERY_DL_URL_TIMEOUT = 10.0


def _iter_gallery_dl_extractor(extr: Any, max_items: int) -> list[str]:
    """Iterate a gallery-dl extractor, returning up to `max_items` file URLs.

    Filters for ``Message.Url`` entries (skips Directory / Queue messages).
    """
    from gallery_dl.extractor.message import Message  # local import

    urls: list[str] = []
    for msg in extr:
        if not isinstance(msg, tuple) or len(msg) < 2:
            continue
        kind = msg[0]
        if kind == Message.Url:
            url_val = msg[1]
            if isinstance(url_val, str):
                urls.append(url_val)
                if len(urls) >= max_items:
                    break
        elif kind == Message.Queue:
            # Sub-gallery link — count the queued URL as a sample too so users
            # can see a listing page yields links, without descending into it.
            queued = msg[1]
            if isinstance(queued, str):
                urls.append(queued)
                if len(urls) >= max_items:
                    break
    return urls


# gallery-dl config keys that can read/write arbitrary files or invoke shell
# commands. A dry-run URL check never needs these — they only exist for real
# download runs and are the classic path-traversal / RCE surface if a hostile
# config_json is accepted from the dashboard.
_GALLERY_DL_UNSAFE_KEYS: frozenset[str] = frozenset({
    "postprocessors",   # can include the "exec" postprocessor for shell exec
    "exec",
    "output",
    "base-directory",
    "directory",
    "filename",
    "archive",          # writes to arbitrary paths
    "log",              # log-file overrides
    "logfile",
    "download-archive",
    "cache",
})


def _scrub_gallery_dl_config(cfg: dict) -> dict:
    """Return a copy of ``cfg`` with any file-access / shell-exec keys removed.

    Recursively walks nested dicts because gallery-dl namespaces settings
    (``{"extractor": {"reddit": {"postprocessors": [...]}}}``). Lists are
    walked so a postprocessors-inside-a-list is caught too. The dry-run URL
    check only ever needs metadata / rate-limit / user-agent shaping; the
    stripped keys have no legitimate role here.
    """
    if isinstance(cfg, dict):
        return {
            k: _scrub_gallery_dl_config(v)
            for k, v in cfg.items()
            if k not in _GALLERY_DL_UNSAFE_KEYS
        }
    if isinstance(cfg, list):
        return [_scrub_gallery_dl_config(item) for item in cfg]
    return cfg


def _check_gallery_dl_url(
    url: str,
    cookies_txt: str | None = None,
    config_json: str | None = None,
    max_items_estimate: int = 5,
) -> dict:
    """Verify a URL via gallery-dl in dry-run mode.

    Runs the URL through gallery-dl's extractor machinery WITHOUT downloading
    any files. Returns a dict describing which extractor matched, its category,
    and a small sample of file URLs the extractor would have downloaded.

    Parameters
    ----------
    url : str
        The URL to probe. Must be http(s).
    cookies_txt : str | None
        Optional Netscape cookies.txt contents. Written to a temp file for the
        duration of the probe; the file is deleted in a ``finally`` block so
        cookie material never leaks past this call.
    config_json : str | None
        Optional inline gallery-dl config (JSON object string). Merged over
        the module defaults for the duration of the probe.
    max_items_estimate : int
        Cap the sample of URLs returned. Iteration stops as soon as we have
        this many.

    Returns
    -------
    dict with keys: ok, extractor, category, sample_urls, message, error.
    """
    empty: dict = {
        "ok": False,
        "extractor": None,
        "category": None,
        "sample_urls": [],
        "message": "",
        "error": None,
    }

    # ---- Basic validation --------------------------------------------------
    if not isinstance(url, str) or not url.strip():
        return {**empty, "error": "url is required", "message": "url is required"}
    url = url.strip()
    try:
        scheme = urlparse(url).scheme.lower()
    except Exception:  # noqa: BLE001
        return {**empty, "error": "invalid url", "message": "invalid url"}
    if scheme not in ("http", "https"):
        return {
            **empty,
            "error": "only http(s) URLs are allowed",
            "message": "only http(s) URLs are allowed",
        }

    # ---- Import gallery-dl -------------------------------------------------
    try:
        from gallery_dl import extractor as gdl_extractor
        from gallery_dl import config as gdl_config
    except Exception as exc:  # noqa: BLE001 - not installed / corrupt install
        return {
            **empty,
            "error": "gallery-dl not installed",
            "message": f"gallery-dl not importable ({type(exc).__name__})",
        }

    # ---- Wire optional cookies + config inside a scoped try/finally --------
    cookies_path: str | None = None
    config_path: str | None = None
    # Snapshot the module config so we can restore it on exit — gallery-dl's
    # config is process-global, so leaking probe settings into a concurrent
    # scraper run would misroute downloads.
    original_config = None
    try:
        try:
            original_config = json.loads(json.dumps(gdl_config._config))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - private attr fallback
            original_config = None

        if cookies_txt and isinstance(cookies_txt, str) and cookies_txt.strip():
            fd, cookies_path = tempfile.mkstemp(
                prefix="cull_gdl_cookies_", suffix=".txt"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(cookies_txt)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass  # fdopen already consumed the fd
                raise
            gdl_config.set(("extractor",), "cookies", cookies_path)

        if config_json and isinstance(config_json, str) and config_json.strip():
            try:
                cfg_data = json.loads(config_json)
            except (ValueError, TypeError) as exc:
                return {
                    **empty,
                    "error": f"config_json is not valid JSON: {exc}",
                    "message": "config_json is not valid JSON",
                }
            if not isinstance(cfg_data, dict):
                return {
                    **empty,
                    "error": "config_json must be a JSON object",
                    "message": "config_json must be a JSON object",
                }
            # Strip gallery-dl config keys that can read/write arbitrary files
            # or execute shell commands. Dry-run URL preflight has no legitimate
            # need for these — but a hostile config_json could otherwise ship
            # `postprocessors: [{"name": "exec", "command": "..."}]` or point
            # `base-directory` at a system path.
            cfg_data = _scrub_gallery_dl_config(cfg_data)
            fd, config_path = tempfile.mkstemp(
                prefix="cull_gdl_config_", suffix=".json"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(cfg_data, fh)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass  # fdopen already consumed the fd
                raise
            try:
                gdl_config.load([config_path])
            except Exception as exc:  # noqa: BLE001
                return {
                    **empty,
                    "error": f"could not load config_json: {exc}",
                    "message": "could not apply config_json",
                }

        # ---- Identify the extractor ---------------------------------------
        try:
            extr = gdl_extractor.find(url)
        except Exception as exc:  # noqa: BLE001
            return {
                **empty,
                "error": f"extractor.find failed: {type(exc).__name__}",
                "message": "gallery-dl could not process this URL",
            }
        if extr is None:
            return {
                **empty,
                "error": "URL not recognised",
                "message": "no gallery-dl extractor matches this URL",
            }

        extractor_name = type(extr).__name__.lower()
        category = getattr(extr, "category", None) or None

        # ---- Iterate under a hard timeout ---------------------------------
        try:
            max_items = max(1, int(max_items_estimate))
        except (TypeError, ValueError):
            max_items = 5

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_iter_gallery_dl_extractor, extr, max_items)
                sample_urls = future.result(timeout=_GALLERY_DL_URL_TIMEOUT)
        except concurrent.futures.TimeoutError:
            return {
                "ok": False,
                "extractor": extractor_name,
                "category": category,
                "sample_urls": [],
                "message": (
                    f"extractor '{extractor_name}' matched, but the site did "
                    f"not respond within {int(_GALLERY_DL_URL_TIMEOUT)}s"
                ),
                "error": "timed out",
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("gallery-dl iteration error for %s: %s", url, exc)
            return {
                "ok": False,
                "extractor": extractor_name,
                "category": category,
                "sample_urls": [],
                "message": (
                    f"extractor '{extractor_name}' matched but iteration failed"
                ),
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }

        if not sample_urls:
            return {
                "ok": True,
                "extractor": extractor_name,
                "category": category,
                "sample_urls": [],
                "message": (
                    f"extractor '{extractor_name}' matched — no items to preview "
                    "(gallery may be empty, private, or paginated)"
                ),
                "error": None,
            }

        return {
            "ok": True,
            "extractor": extractor_name,
            "category": category,
            "sample_urls": sample_urls,
            "message": (
                f"extractor '{extractor_name}' matched — "
                f"{len(sample_urls)} sample item(s) previewed"
            ),
            "error": None,
        }
    finally:
        # Restore config so a probe never leaks cookies/config into a live run.
        try:
            if original_config is not None:
                gdl_config._config.clear()  # type: ignore[attr-defined]
                gdl_config._config.update(original_config)  # type: ignore[attr-defined]
            else:
                gdl_config.clear()
        except Exception:  # noqa: BLE001 - restoration is best-effort
            pass
        for path in (cookies_path, config_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass  # temp file already gone


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
# Per-URL test registry (parallel to _CHECKERS / SUPPORTED)
# ---------------------------------------------------------------------------
# Some sources aren't scrapers per se — they're generic URL probes. gallery-dl
# is the first: any URL a user pastes into the dashboard can be pre-flighted
# to confirm gallery-dl recognises it and to preview a few items. The registry
# below carries `kind="per_url"` so dashboards can route these to the right
# UI (an input field for a URL, not a per-scraper Test button).
SUPPORTED_URL_TESTS: dict[str, dict[str, Any]] = {
    "gallery_dl_url": {
        "label": "gallery-dl URL",
        "check": _check_gallery_dl_url,
        "requires": [],
        "kind": "per_url",
    },
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


__all__ = [
    "test_scraper",
    "SUPPORTED",
    "SUPPORTED_URL_TESTS",
    "_check_gallery_dl_url",
    "_http_request",
]
