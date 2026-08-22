"""Regression tests for the security hardening CodeQL's suite motivated.

Each block below pins one invariant that was *not* holding before, so a
regression fails a test rather than quietly re-opening the hole:

  A) py/log-injection — a newline inside an interpolated log value used to emit
     what looked like a second, fully-formed record. ``ScrubbingFormatter``
     escapes CR / LF / ESC in the message while leaving a real ``exc_info``
     traceback multi-line.
  B) Validator anchors — Python's ``$`` also matches immediately *before* a
     trailing newline, so every ``^...$`` identity regex accepted ``value\\n``
     and passed it on to a filesystem path / outbound repo id / output
     filename. The whole validator family is anchored with ``\\Z`` now.
  C) py/full-ssrf — the digest webhook POST was the one outbound call missing
     ``allow_redirects=False``, so a cleared host could 302 the request at a
     private address.
  D) The ``_err`` shadow — a tuple-unpack target named ``_err`` inside
     ``api_presets_publish`` turned every earlier ``return _err(...)`` in that
     route into an ``UnboundLocalError``.

Hermetic: no network, no real ``.env``, no wall-clock reads in assertions.

Runnable:
    pytest tests/test_security_hardening.py -q
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise ``dotenv.load_dotenv`` so a developer's real ``.env`` can't
    leak into these hermetic tests."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


# ── A) log injection: CR / LF / ESC never survive into a formatted record ─────

def _format(msg: str, *args: object, exc_info: object = None) -> str:
    import pipeline_logging
    formatter = pipeline_logging.ScrubbingFormatter("%(levelname)s %(message)s")
    record = logging.LogRecord(
        "test", logging.WARNING, __file__, 1, msg, args, exc_info,  # type: ignore[arg-type]
    )
    return formatter.format(record)


def test_newline_in_interpolated_value_cannot_forge_a_record() -> None:
    """The classic payload: a slug whose value closes the line and opens a new
    one that reads like a genuine log entry."""
    hostile = "ok\n[INFO] 00:00:00 audit: pipeline stopped by admin"
    out = _format("activating job %s", hostile)
    assert "\n" not in out
    assert "\\n[INFO]" in out


def test_carriage_return_and_escape_are_neutralised() -> None:
    out = _format("probe %s", "host\rSPOOF\x1b[2J")
    assert "\r" not in out and "\x1b" not in out
    assert "\\r" in out and "\\x1b" in out


def test_scrub_applies_to_a_preformatted_message_too() -> None:
    """f-string call sites interpolate before logging, so the taint lands in
    ``msg`` rather than ``args`` — that path must be scrubbed as well."""
    out = _format("activating job ok\nforged")
    assert "\n" not in out


def test_exception_traceback_stays_multi_line() -> None:
    """Only the message is scrubbed. A traceback is trusted output appended by
    the base formatter, and must keep its line structure to stay readable."""
    try:
        raise ValueError("boom")
    except ValueError:
        out = _format("failed", exc_info=sys.exc_info())
    assert "Traceback (most recent call last)" in out
    assert out.count("\n") >= 2


def test_configure_root_installs_the_scrubbing_formatter() -> None:
    """The wiring matters as much as the class: the stderr handler the whole
    process tree logs through is the one that must scrub."""
    import pipeline_logging
    pipeline_logging = importlib.reload(pipeline_logging)
    pipeline_logging.configure_root()
    handlers = logging.getLogger().handlers
    assert handlers, "configure_root must install a handler"
    assert any(
        isinstance(h.formatter, pipeline_logging.ScrubbingFormatter) for h in handlers
    )


# ── B) validator anchors: a trailing newline is not a valid identity ──────────

def test_slug_validators_reject_a_trailing_newline() -> None:
    import job_config
    import paths
    assert job_config.JOB_SLUG_RE.match("default") is not None
    assert job_config.JOB_SLUG_RE.match("default\n") is None
    assert paths.validate_slug("default") == "default"
    with pytest.raises(ValueError):
        paths.validate_slug("default\n")


def test_name_validators_reject_a_trailing_newline() -> None:
    import config_io
    import job_config
    import queue_manager
    import theme_config
    cases = (
        (job_config.PRESET_NAME_RE, "My Preset"),
        (theme_config.THEME_NAME_RE, "midnight"),
        (queue_manager._SAFE_SOURCE, "civitai"),
        (config_io._CAT_NAME_RE, "Keep"),
        (config_io._LOCAL_NAME_RE, "drop-box"),
    )
    for pattern, good in cases:
        assert pattern.match(good) is not None, f"{pattern.pattern} rejected {good!r}"
        assert pattern.match(good + "\n") is None, \
            f"{pattern.pattern} accepted a trailing newline"


def test_dashboard_path_validators_reject_a_trailing_newline() -> None:
    import dashboard_enhanced as dash
    cases = (
        (dash._HF_REPO_RE, "owner/dataset"),
        (dash._COMMUNITY_FILENAME_RE, "aerial.preset.json"),
        (dash._COOKIES_NAME_RE, "cookies.txt"),
        (dash._PRESET_THUMB_KEY_RE, "aerial"),
        (dash._CAT_NAME_RE, "Keep"),
    )
    for pattern, good in cases:
        assert pattern.match(good) is not None, f"{pattern.pattern} rejected {good!r}"
        assert pattern.match(good + "\n") is None, \
            f"{pattern.pattern} accepted a trailing newline"
    assert dash._valid_slug_or_400("default") is True
    assert dash._valid_slug_or_400("default\n") is False


# ── C) SSRF: the digest webhook POST refuses to follow a redirect ─────────────

def test_digest_webhook_post_disables_redirects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A host that passes ``_is_public_http_url`` must not be able to 302 the
    POST at a private address afterwards."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path / "data"))
    import scheduler as _sched
    scheduler = importlib.reload(_sched)

    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(scheduler.requests, "post", fake_post)
    monkeypatch.setattr(scheduler, "_is_public_http_url", lambda _url: True)

    scheduler.run_digest("default", webhook_url="https://example.com/hook")

    assert captured.get("url") == "https://example.com/hook"
    assert captured.get("allow_redirects") is False


# ── D) the _err shadow in api_presets_publish ────────────────────────────────

def test_no_dashboard_function_shadows_the_err_helper() -> None:
    """``_err`` must stay the module-level helper everywhere: a local of the
    same name (a ``_rc, out, _err = _git_run(...)`` unpack, say) makes every
    earlier ``return _err(...)`` in that function an UnboundLocalError instead
    of the 500 it was written to be.

    Asserted over the whole module by AST rather than for one route, because
    this bug landed twice — once in the preset publish flow and once in the
    theme publish flow.
    """
    import ast
    source = (PIPELINE_CODE / "dashboard_enhanced.py").read_text(encoding="utf-8")
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if (isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
                    and child.id == "_err"):
                offenders.append(f"{node.name} (line {child.lineno})")
    assert not offenders, f"these functions bind a local named _err: {offenders}"

    import dashboard_enhanced as dash
    assert "_err" in dash.api_presets_publish.__code__.co_names, \
        "the publish route should still call the _err helper"


if __name__ == "__main__":  # python tests/test_security_hardening.py
    raise SystemExit(pytest.main([__file__, "-q"]))
