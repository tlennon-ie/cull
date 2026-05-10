#!/usr/bin/env python3
"""LM Studio admin helpers — currently just `unload_all`.

LM Studio's REST API does NOT expose a stable unload endpoint across versions
(the user-reported `POST /api/v0/models/unload` returns "Unexpected endpoint or
method"). The `lms` CLI is the documented, version-stable path.

Usage:
    from lmstudio_admin import unload_all
    ok, detail = unload_all()
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pipeline_logging import get_logger

logger = get_logger(__name__)

CLI_TIMEOUT_SECONDS = 15


def _candidate_lms_executables() -> list[str]:
    """Return likely paths to the `lms` CLI in priority order.

    LM Studio installs `lms` to `~/.lmstudio/bin/lms` on first launch but does
    NOT always add it to PATH. We check PATH first, then the documented install
    location.
    """
    candidates: list[str] = []
    on_path = shutil.which("lms")
    if on_path:
        candidates.append(on_path)
    # Documented per-user install location — same on win/mac/linux.
    home_bin = Path.home() / ".lmstudio" / "bin"
    suffix = ".exe" if os.name == "nt" else ""
    home_lms = home_bin / f"lms{suffix}"
    if home_lms.exists() and str(home_lms) not in candidates:
        candidates.append(str(home_lms))
    return candidates


@dataclass(frozen=True)
class UnloadResult:
    ok: bool
    detail: str
    method: str  # "cli-all" | "cli-per-model" | "none"


def _run_cli(executable: str, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        [executable, *args],
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def unload_all() -> UnloadResult:
    """Unload every loaded LM Studio model. Best-effort; never raises.

    Strategy:
      1. `lms unload --all` (works on recent LM Studio versions).
      2. Fallback: `lms ps --json` -> `lms unload <id>` per model, for older
         versions that don't accept `--all`.
    """
    candidates = _candidate_lms_executables()
    if not candidates:
        msg = "lms CLI not found on PATH or in ~/.lmstudio/bin/"
        logger.warning("unload skipped: %s", msg)
        return UnloadResult(False, msg, "none")

    executable = candidates[0]

    # Path 1: bulk unload.
    try:
        rc, out, err = _run_cli(executable, ["unload", "--all"])
        if rc == 0:
            detail = out or "lms unload --all returned 0"
            logger.info("unload_all ok via lms unload --all: %s", detail)
            return UnloadResult(True, detail, "cli-all")
        # Older lms returns non-zero on unrecognised flags. Fall through.
        logger.info("lms unload --all rc=%d stderr=%s; falling back to per-model", rc, err[:200])
    except subprocess.TimeoutExpired:
        return UnloadResult(False, "lms unload --all timed out", "cli-all")
    except OSError as exc:
        return UnloadResult(False, f"lms invocation failed: {exc}", "none")

    # Path 2: enumerate then unload one-by-one.
    try:
        rc, out, err = _run_cli(executable, ["ps", "--json"])
        if rc != 0:
            return UnloadResult(False, f"lms ps --json rc={rc}: {err[:200]}", "cli-per-model")
        try:
            payload = json.loads(out) if out else []
        except json.JSONDecodeError:
            return UnloadResult(False, f"lms ps --json returned non-JSON: {out[:200]}", "cli-per-model")
        models = payload if isinstance(payload, list) else payload.get("models", [])
        if not models:
            return UnloadResult(True, "no models loaded", "cli-per-model")
        unloaded: list[str] = []
        failures: list[str] = []
        for entry in models:
            ident = entry.get("identifier") or entry.get("modelKey") or entry.get("path")
            if not ident:
                continue
            rc2, _out2, err2 = _run_cli(executable, ["unload", ident])
            if rc2 == 0:
                unloaded.append(ident)
            else:
                failures.append(f"{ident} ({err2[:80]})")
        if failures:
            return UnloadResult(
                bool(unloaded),
                f"unloaded {len(unloaded)}, failed {len(failures)}: {failures[:3]}",
                "cli-per-model",
            )
        return UnloadResult(True, f"unloaded {len(unloaded)} model(s)", "cli-per-model")
    except subprocess.TimeoutExpired:
        return UnloadResult(False, "lms ps/unload timed out", "cli-per-model")
    except OSError as exc:
        return UnloadResult(False, f"lms invocation failed: {exc}", "cli-per-model")


if __name__ == "__main__":
    result = unload_all()
    print(f"ok={result.ok} method={result.method} detail={result.detail}")
