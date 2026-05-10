#!/usr/bin/env python3
"""
Launcher - boots ONLY the dashboard.

The dashboard now owns the pipeline lifecycle (Start/Stop buttons spawn/kill
run_pipeline.py via /api/pipeline/start and /api/pipeline/stop). Starting the
pipeline automatically here would fight that, so we intentionally don't.

Paths resolved from claude/.env. No hardcodes.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("launcher")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def _resolve_pipeline_code_dir() -> Path:
    """Resolve PIPELINE_CODE_DIR with a path-mismatch sanity check.

    Common foot-gun: a user clones cull into a new path (e.g. after the
    rebrand) but their `.env` still has PIPELINE_CODE_DIR pointing at the
    old location. The launcher would then spawn the dashboard FROM the old
    code while the user thinks they're on latest. We detect that here and
    warn loudly. The override still wins — power users may legitimately
    point at a worktree — but the warning makes the choice explicit.
    """
    expected = Path(__file__).resolve().parent
    raw = os.environ.get("PIPELINE_CODE_DIR")
    if not raw:
        return expected
    configured = Path(raw).resolve()
    if configured == expected:
        return configured
    if not configured.exists():
        logger.warning(
            "PIPELINE_CODE_DIR=%s does not exist; falling back to launcher's directory %s",
            configured, expected,
        )
        return expected
    print("\n" + "=" * 70, flush=True)
    print("[launcher] WARNING: PIPELINE_CODE_DIR mismatch", flush=True)
    print(f"  .env says:        {configured}", flush=True)
    print(f"  Launcher lives in: {expected}", flush=True)
    print("  The dashboard will run from the .env path, NOT the launcher's.", flush=True)
    print("  If you cloned into a new location, edit .env to match — or", flush=True)
    print("  delete PIPELINE_CODE_DIR / WORKSPACE_ROOT to use the launcher's", flush=True)
    print("  directory automatically.", flush=True)
    print("=" * 70 + "\n", flush=True)
    return configured


PIPELINE_CODE_DIR: Path = _resolve_pipeline_code_dir()
FLASK_PORT: int = int(os.environ.get("FLASK_PORT", 5000))

processes: list[tuple[str, subprocess.Popen]] = []


def start_dashboard() -> subprocess.Popen | None:
    try:
        kwargs = {"cwd": str(PIPELINE_CODE_DIR)}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen([sys.executable, "-u", "dashboard_enhanced.py"], **kwargs)
        processes.append(("dashboard", proc))
        logger.info("dashboard started (PID %s)", proc.pid)
        return proc
    except Exception as exc:
        logger.error("failed to start dashboard: %s", exc)
        return None


def cleanup(sig: object = None, frame: object = None) -> None:
    print("\nStopping dashboard...", flush=True)
    for name, proc in processes:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def main() -> None:
    print("Launcher - dashboard-only mode")
    print(f"Code dir: {PIPELINE_CODE_DIR}")
    print(f"Dashboard: http://localhost:{FLASK_PORT}")
    print("Pipeline will NOT auto-start - use the Start button in the dashboard.\n")

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    start_dashboard()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()
        sys.exit(0)


if __name__ == "__main__":
    main()
