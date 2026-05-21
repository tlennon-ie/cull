"""
credentials.py — Single source of truth for reading secrets / API keys from the
environment (`.env` via python-dotenv, or the OS environment).

Scrapers and workers MUST go through this module rather than calling
``os.environ`` directly. Centralising the lookups gives us:

  * Consistent error messages when a required key is missing.
  * A single ``MissingCredentialError`` that the supervisor (run_pipeline.py)
    recognises as a "configuration" failure and applies a cooldown to, instead
    of restarting the scraper in a tight loop.
  * One place to plug in a future secret-manager backend (Vault, 1Password,
    keyring, etc.) without touching every call site.

The module is intentionally tiny — it owns NO state, NO secrets, and NO I/O
beyond ``os.environ`` reads. All values come from the environment at call time,
so reloading ``.env`` (``load_dotenv(override=True)``) takes effect immediately.

Public API:
    MissingCredentialError      # SystemExit subclass — supervisor-aware
    get_required(key, scraper)  # raise MissingCredentialError if missing/empty
    get_optional(key, default)  # return env value or ``default`` (None if absent)
    get_keylist(primary, fallback)  # first non-empty value from a list of keys
    has_any(keys)               # True if at least one key has a non-empty value
    warn_if_missing(key, scraper, logger)  # log a one-line warning; never raises
"""
from __future__ import annotations

import os
from typing import Iterable


class MissingCredentialError(SystemExit):
    """Raised when a scraper / worker is missing a required credential.

    Inherits from ``SystemExit`` so that an uncaught instance terminates the
    subprocess cleanly with a non-zero code. ``run_pipeline.py`` treats a
    non-zero exit as a configuration failure and backs off before restarting,
    rather than restarting a broken scraper hundreds of times per minute.

    The exit code is fixed at ``78`` (EX_CONFIG from sysexits.h) so the
    supervisor can distinguish "you forgot to set an API key" from a generic
    crash.
    """

    EXIT_CODE = 78

    def __init__(self, key: str, scraper: str | None = None) -> None:
        self.key = key
        self.scraper = scraper
        scope = f"[{scraper}] " if scraper else ""
        msg = (
            f"{scope}Missing required environment variable: {key}\n"
            f"  Set it in .env (see .env.example for the full list of supported keys)."
        )
        super().__init__(msg)
        # Override the SystemExit code so callers see EX_CONFIG, not the message.
        self.code = self.EXIT_CODE


def _read(key: str) -> str | None:
    """Return env[key] stripped of whitespace, or None if missing/empty."""
    raw = os.environ.get(key)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def get_required(key: str, scraper: str | None = None) -> str:
    """Return the value of ``key`` from the environment.

    Raises ``MissingCredentialError`` if the variable is unset or empty (after
    stripping whitespace). The ``scraper`` argument is purely cosmetic — it
    shows up in the error message so the operator knows which agent crashed.
    """
    value = _read(key)
    if value is None:
        raise MissingCredentialError(key, scraper=scraper)
    return value


def get_optional(key: str, default: str | None = None) -> str | None:
    """Return env[key] if set and non-empty, otherwise ``default``.

    Note: returns ``default`` for both "unset" and "set to empty string", which
    is the behaviour every scraper in this repo relies on. If you need to
    distinguish those cases, read ``os.environ`` directly.
    """
    value = _read(key)
    return value if value is not None else default


def get_keylist(primary: str, fallback: str | None = None) -> str | None:
    """Return the first non-empty value among ``primary`` then ``fallback``.

    Handy for the common "prefer site-specific key, fall back to the generic
    one" pattern, e.g. ``get_keylist("CIVITAI_API_RED_KEY", "CIVITAI_API_KEY")``.
    """
    value = _read(primary)
    if value is not None:
        return value
    if fallback:
        return _read(fallback)
    return None


def has_any(keys: Iterable[str]) -> bool:
    """Return True if at least one of ``keys`` is set to a non-empty value."""
    return any(_read(k) is not None for k in keys)


def warn_if_missing(
    key: str,
    scraper: str | None = None,
    logger=None,
) -> bool:
    """Log a one-line warning if ``key`` is missing. Returns True if present.

    Never raises — use this for soft requirements where the scraper can still
    do *something* useful without the key (e.g. unauthenticated reads of a
    public API at a lower rate limit).
    """
    if _read(key) is not None:
        return True
    scope = f"[{scraper}] " if scraper else ""
    msg = f"{scope}WARNING: optional environment variable {key} is not set"
    if logger is not None:
        logger.warning(msg)
    else:
        # Match the subprocess logging convention used elsewhere in the repo
        # (see pipeline_logging.py — workers print with flush=True so the
        # supervisor can label stdout cleanly).
        print(msg, flush=True)
    return False


__all__ = [
    "MissingCredentialError",
    "get_required",
    "get_optional",
    "get_keylist",
    "has_any",
    "warn_if_missing",
]
