#!/usr/bin/env python3
"""Cross-platform system-tray launcher for cull.

Wraps :mod:`pipeline_code.integrated_launcher` (the combined dashboard boot
script) and adds an OS system-tray icon with quick-access controls. This is
purely a convenience shell — all pipeline logic still lives in the integrated
launcher, the dashboard, and the supervisor. This file only:

* Spawns ``integrated_launcher`` as a subprocess.
* Renders a tray icon whose menu drives the dashboard's existing HTTP API.
* Polls ``/api/status`` in the background to keep the tooltip current.
* Graceful-shutdowns the subprocess on quit (SIGTERM equivalent, force after 10s).

The tray dependencies (``pystray``, ``Pillow``) are OPTIONAL project extras.
If they are missing this module refuses to run and prints the exact install
command — following the CLAUDE.md rule that we never auto-``pip install`` at
runtime.

The dashboard exposes ``/api/pipeline/start`` and ``/api/pipeline/stop`` but
does NOT expose pause/resume/restart routes (see ``dashboard_enhanced.py``).
The "Pause / Resume" menu items therefore fall back to stop and start, and
"Restart" is implemented as stop + start.
"""
from __future__ import annotations

import io
import logging
import os
import platform
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import requests

# -- Optional tray deps -----------------------------------------------------
# pystray / Pillow are OPTIONAL. Do NOT auto-pip-install (CLAUDE.md rule).
# Import lazily so `import tray_launcher` succeeds even without the extras;
# we only hard-fail when `main()` is invoked.
try:  # pragma: no cover - import-time gate
    import pystray  # type: ignore
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _TRAY_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover - optional dep missing
    pystray = None  # type: ignore
    Image = ImageDraw = ImageFont = None  # type: ignore
    _TRAY_IMPORT_ERROR = _exc


def _require_tray_deps() -> None:
    if _TRAY_IMPORT_ERROR is None:
        return
    sys.stderr.write(
        "cull tray launcher requires the optional 'pystray' and 'Pillow' "
        "packages.\n"
        f"Missing import: {_TRAY_IMPORT_ERROR}\n"
        "Install them with:\n"
        "    pip install pystray Pillow\n"
    )
    sys.exit(1)


# -- Paths ------------------------------------------------------------------
# The tray launcher sits next to integrated_launcher.py inside pipeline_code/.
# We resolve everything relative to this file to avoid depending on CWD.
_THIS_FILE = Path(__file__).resolve()
_PIPELINE_CODE_DIR = _THIS_FILE.parent
_REPO_ROOT = _PIPELINE_CODE_DIR.parent

# Try to reuse cull's canonical paths module for the logs dir. Fall back to a
# sensible default if the import fails (e.g. running standalone).
try:
    sys.path.insert(0, str(_PIPELINE_CODE_DIR))
    from paths import log_dir as _paths_log_dir  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive fallback
    _paths_log_dir = None  # type: ignore[assignment]


DASHBOARD_PORT = int(os.environ.get("FLASK_PORT", 5000))
DASHBOARD_URL = f"http://localhost:{DASHBOARD_PORT}"
STATUS_POLL_SECONDS = 5.0
SHUTDOWN_GRACE_SECONDS = 10.0
TRAY_ICON_ASSET = _REPO_ROOT / "assets" / "tray_icon.png"

logger = logging.getLogger("tray_launcher")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


# -- Dashboard HTTP helpers -------------------------------------------------
def _api_get(path: str, timeout: float = 2.0) -> dict[str, Any] | None:
    """GET a dashboard endpoint; return parsed JSON or None on any failure."""
    try:
        resp = requests.get(f"{DASHBOARD_URL}{path}", timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def _api_post(path: str, timeout: float = 5.0) -> bool:
    """POST an empty body to a dashboard endpoint; return success flag."""
    try:
        resp = requests.post(f"{DASHBOARD_URL}{path}", timeout=timeout)
        return resp.status_code < 400
    except requests.RequestException as exc:
        logger.warning("POST %s failed: %s", path, exc)
        return False


def _pipeline_running() -> bool | None:
    """Return True/False from /api/status.pipeline.running; None if dashboard down."""
    payload = _api_get("/api/status")
    if payload is None:
        return None
    pipeline = payload.get("pipeline") or {}
    running = pipeline.get("running")
    return bool(running) if running is not None else None


# -- Tray icon image --------------------------------------------------------
def _load_or_generate_icon() -> "Image.Image":
    """Load the bundled tray icon; otherwise generate a distinctive fallback.

    The fallback is a 32x32 purple square with a white 'C' glyph, drawn in
    memory. No files are written to disk.
    """
    if TRAY_ICON_ASSET.exists():
        try:
            return Image.open(TRAY_ICON_ASSET).convert("RGBA")
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("failed to load %s: %s", TRAY_ICON_ASSET, exc)

    size = 32
    img = Image.new("RGBA", (size, size), (105, 65, 195, 255))  # cull purple
    draw = ImageDraw.Draw(img)
    # Try to use a system font; fall back to the PIL default if unavailable.
    font = None
    for candidate in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(candidate, 22)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    # Center the glyph. textbbox is present on modern Pillow; guard defensively.
    try:
        bbox = draw.textbbox((0, 0), "C", font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pos = ((size - text_w) // 2 - bbox[0], (size - text_h) // 2 - bbox[1])
    except Exception:  # pragma: no cover - very old Pillow
        pos = (8, 4)
    draw.text(pos, "C", fill=(255, 255, 255, 255), font=font)
    return img


# -- Subprocess lifecycle ---------------------------------------------------
class LauncherProcess:
    """Owns the integrated_launcher subprocess and its shutdown protocol."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                logger.info("integrated_launcher already running (PID %s)", self._proc.pid)
                return

            kwargs: dict[str, Any] = {"cwd": str(_REPO_ROOT)}
            if sys.platform == "win32":
                # New process group so terminate() sends CTRL_BREAK to the tree.
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                # New session so the child leads its own group — lets us signal
                # the whole tree without hitting the tray process itself.
                kwargs["start_new_session"] = True

            cmd = [sys.executable, "-u", "-m", "pipeline_code.integrated_launcher"]
            logger.info("launching: %s", " ".join(cmd))
            self._proc = subprocess.Popen(cmd, **kwargs)
            logger.info("integrated_launcher started (PID %s)", self._proc.pid)

    def stop(self, grace_seconds: float = SHUTDOWN_GRACE_SECONDS) -> None:
        with self._lock:
            proc = self._proc
            if proc is None:
                return
            if proc.poll() is not None:
                self._proc = None
                return

            logger.info("terminating integrated_launcher (PID %s)", proc.pid)
            try:
                proc.terminate()
            except Exception as exc:  # pragma: no cover - platform-specific
                logger.warning("terminate() failed: %s", exc)
            try:
                proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "integrated_launcher did not exit within %.1fs — killing",
                    grace_seconds,
                )
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception as exc:  # pragma: no cover - platform-specific
                    logger.error("kill() failed: %s", exc)
            self._proc = None

    def is_alive(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None


# -- Menu actions -----------------------------------------------------------
def _open_dashboard(icon: "pystray.Icon", item: "pystray.MenuItem") -> None:  # noqa: ARG001
    webbrowser.open(DASHBOARD_URL)


def _open_logs(icon: "pystray.Icon", item: "pystray.MenuItem") -> None:  # noqa: ARG001
    log_path = _paths_log_dir() if _paths_log_dir is not None else _REPO_ROOT / "data" / "logs"
    log_path = Path(log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(log_path))  # type: ignore[attr-defined]  # noqa: S606
        elif system == "Darwin":
            subprocess.run(["open", str(log_path)], check=False)
        else:
            subprocess.run(["xdg-open", str(log_path)], check=False)
    except Exception as exc:
        logger.error("failed to open logs dir %s: %s", log_path, exc)


def _pause_pipeline(icon: "pystray.Icon", item: "pystray.MenuItem") -> None:  # noqa: ARG001
    # No dedicated /api/pipeline/pause route exists — fall back to stop.
    _api_post("/api/pipeline/stop")


def _resume_pipeline(icon: "pystray.Icon", item: "pystray.MenuItem") -> None:  # noqa: ARG001
    # No dedicated /api/pipeline/resume route exists — fall back to start.
    _api_post("/api/pipeline/start")


def _restart_pipeline(icon: "pystray.Icon", item: "pystray.MenuItem") -> None:  # noqa: ARG001
    # No /api/pipeline/restart — stop then start.
    _api_post("/api/pipeline/stop")
    # Give the supervisor a moment to release its resources before respawning.
    time.sleep(1.0)
    _api_post("/api/pipeline/start")


def _make_quit(launcher: LauncherProcess) -> Any:
    def _quit(icon: "pystray.Icon", item: "pystray.MenuItem") -> None:  # noqa: ARG001
        logger.info("tray quit requested")
        launcher.stop()
        icon.stop()

    return _quit


# -- Status poll ------------------------------------------------------------
def _status_poll_loop(icon: "pystray.Icon", stop_event: threading.Event) -> None:
    """Background thread: refresh the tray tooltip every STATUS_POLL_SECONDS."""
    while not stop_event.is_set():
        running = _pipeline_running()
        if running is True:
            icon.title = "cull — pipeline running"
        elif running is False:
            icon.title = "cull — pipeline stopped"
        else:
            icon.title = "cull — dashboard unreachable"
        stop_event.wait(STATUS_POLL_SECONDS)


# -- Entry point ------------------------------------------------------------
def main() -> None:
    _require_tray_deps()
    launcher = LauncherProcess()
    launcher.start()

    image = _load_or_generate_icon()
    menu = pystray.Menu(
        pystray.MenuItem("Open dashboard", _open_dashboard, default=True),
        pystray.MenuItem("Pause pipeline", _pause_pipeline),
        pystray.MenuItem("Resume pipeline", _resume_pipeline),
        pystray.MenuItem("Restart pipeline", _restart_pipeline),
        pystray.MenuItem("Show logs", _open_logs),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", _make_quit(launcher)),
    )
    icon = pystray.Icon(
        "cull",
        icon=image,
        title="cull — starting",
        menu=menu,
    )

    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=_status_poll_loop,
        args=(icon, stop_event),
        name="cull-tray-status-poll",
        daemon=True,
    )
    poll_thread.start()

    try:
        icon.run()  # blocks until icon.stop()
    except KeyboardInterrupt:  # pragma: no cover - user-driven
        logger.info("KeyboardInterrupt — shutting down")
    finally:
        stop_event.set()
        launcher.stop()


if __name__ == "__main__":
    main()
