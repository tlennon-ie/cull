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

PIPELINE_CODE_DIR: Path = Path(os.environ.get("PIPELINE_CODE_DIR", Path(__file__).parent))
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
