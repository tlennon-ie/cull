"""capture_screenshots.py - Boot the demo dashboard, snap the README shots.

The script is a self-contained Playwright driver that:

1. Seeds the demo dataset in an isolated ``PIPELINE_BASE_DIR`` (so the demo
   data never contaminates whatever the developer has in ``data/`` locally,
   and so the indexer's dot-prefix archive filter cannot swallow the seed
   just because the repo happens to live under ``.claude/worktrees/...``).
2. Launches ``pipeline_code/dashboard_enhanced.py`` on a private port
   pointed at that scratch base dir.
3. Waits for ``/api/status`` to answer HTTP 200.
4. Captures the primary README screenshots at 1440x900, plus a theme gallery
   (one canonical view per shipped theme).
5. Compresses each PNG so it fits under the 500 KB README budget.
6. Kills the dashboard subprocess (and any indexer thread) on exit —
   including on Ctrl-C or any unhandled exception.

The output lands in ``docs/screenshots/`` — the README references the exact
filenames this script writes.

Usage:

    python tools/capture_screenshots.py             # full run
    python tools/capture_screenshots.py --only 01,02  # subset (index ids)
    python tools/capture_screenshots.py --keep-data # don't wipe scratch dir

Playwright + a chromium build are required — install with:

    pip install -e ".[dev]"      # (playwright is already in requirements.txt)
    playwright install chromium
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
PIPELINE_CODE: Path = REPO_ROOT / "pipeline_code"
DOCS_SCREENSHOTS: Path = REPO_ROOT / "docs" / "screenshots"
THEMES_OUT: Path = DOCS_SCREENSHOTS / "themes"

# The scratch base dir lives outside the repo so the ``.claude/worktrees/``
# path prefix (or any other dot-prefixed dir) cannot make the indexer's
# ``_is_in_archive`` filter drop our seed. It also MUST NOT start with a dot
# for the same reason — a leading '.' in any path segment marks it as
# archive-only. See index_store._is_in_archive.
import tempfile as _tempfile

DEFAULT_SCRATCH: Path = Path(
    os.environ.get("CULL_SHOT_SCRATCH")
    or (Path(_tempfile.gettempdir()) / "cull_shots_data")
).expanduser().resolve()

VIEWPORT: dict[str, int] = {"width": 1440, "height": 900}
DEVICE_PIXEL_RATIO: float = 1.5  # crisper text in the PNGs

# Shipped themes we snap for the "themes gallery" figure. The 5 JSON-backed
# themes plus the 3 CSS-only cores (see theme_config.CORE_THEMES).
CORE_THEMES: tuple[str, ...] = ("dark", "light", "hc")
JSON_THEMES: tuple[str, ...] = ("ai-slop", "beige", "cyberpunk", "forest", "wood")
ALL_THEMES: tuple[str, ...] = CORE_THEMES + JSON_THEMES


@dataclass
class Shot:
    """One primary README screenshot."""

    slug: str          # "01-jobs" — becomes docs/screenshots/<slug>.png
    view: str          # 'jobs' or 'job'
    active: str        # sidebar section id
    setup: Callable | None = None   # optional page-level tweak
    settle_ms: int = 900            # extra wait for network / animation
    theme: str = "dark"             # theme to force before snapping


# ── Playwright driving helpers ───────────────────────────────────────────────


def _wait_reachable(port: int, timeout: float = 30.0) -> bool:
    """Poll ``/api/status`` until it answers 200 or we time out."""
    url = f"http://127.0.0.1:{port}/api/status"
    deadline = time.time() + timeout
    last_err: str | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionResetError, socket.timeout) as exc:
            last_err = str(exc)
        time.sleep(0.4)
    print(f"[capture] dashboard did not become reachable: {last_err}", file=sys.stderr)
    return False


def _pick_free_port(preferred: int) -> int:
    """Return ``preferred`` if free, else a random free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _launch_dashboard(port: int, base_dir: Path) -> subprocess.Popen:
    """Spawn the dashboard as a subprocess, inheriting a scoped env."""
    env = os.environ.copy()
    env["PIPELINE_BASE_DIR"] = str(base_dir)
    env["PIPELINE_QUEUE"] = str(base_dir / "queue")
    env["PIPELINE_SORTED"] = str(base_dir / "sorted")
    env["LOG_DIR"] = str(base_dir / "logs")
    env["FLASK_HOST"] = "127.0.0.1"
    env["FLASK_PORT"] = str(port)
    # Faster indexer settle in the demo so all rows land before we snap.
    env["INDEX_REFRESH_SECONDS"] = "5"
    # Kill any inherited PIPELINE_SLUG so the dashboard resolves the active
    # slug through job_config (which the seeder just multi-activated).
    env.pop("PIPELINE_SLUG", None)
    # Silence rich progress output; simpler stdout for the shot log.
    env.setdefault("LOG_LEVEL", "WARNING")

    creation = 0
    if os.name == "nt":
        # Detach into its own process group so KeyboardInterrupt / cleanup
        # only kill the dashboard, never the driver.
        creation = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    # NOTE: never wire the dashboard's stdout to a PIPE we don't drain — the
    # dashboard is a chatty subprocess (indexer progress, HTTP access logs,
    # SSE keepalives) and the OS pipe buffer fills within a couple of tab
    # switches. When it fills, the dashboard blocks on write() and Playwright
    # eventually times out with TargetClosedError as the browser context
    # dies. DEVNULL discards the noise cleanly.
    proc = subprocess.Popen(
        [sys.executable, str(PIPELINE_CODE / "dashboard_enhanced.py")],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation,
    )
    return proc


def _terminate_dashboard(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        pass
    if proc.poll() is None:
        proc.kill()


# ── Alpine.js state pokes ────────────────────────────────────────────────────


_ROOT_STATE = "document.querySelector('[x-data]')._x_dataStack[0]"


def _js_go_to(view: str, active: str, slug: str | None = None) -> str:
    """Return a Javascript snippet that navigates the SPA to (view, active)."""
    slug_expr = f"'{slug}'" if slug else "null"
    return f"""
        (() => {{
          const s = {_ROOT_STATE};
          const target = {slug_expr};
          if (target) {{ s.currentJob = target; }}
          s.view = '{view}';
          s.active = '{active}';
          s.sidebarOpen = false;
          s.welcome && (s.welcome.dismissed = true);
        }})()
    """


def _js_set_theme(name: str) -> str:
    return f"{_ROOT_STATE}.setTheme('{name}')"


def _js_toggle_show_nsfw(on: bool) -> str:
    """Force the global NSFW-blur toggle so gradient tiles in the NSFW bucket
    don't ship as blurred rectangles in the screenshots."""
    v = "true" if on else "false"
    return f"""
        (() => {{
          const s = {_ROOT_STATE};
          s.showNsfw = {v};
          try {{ localStorage.setItem('cull_show_nsfw', s.showNsfw ? '1' : '0'); }} catch (_) {{}}
        }})()
    """


def _js_open_filter_panel(panel: str) -> str:
    return f"{_ROOT_STATE}.filterPanel = '{panel}'"


_STICKY_SIDEBAR_CSS = (
    " const aside = document.querySelector('aside');"
    " if (aside && !aside.__pinned) {"
    "   aside.__pinned = true;"
    "   aside.style.position = 'fixed';"
    "   aside.style.top = '0';"
    "   aside.style.left = '0';"
    "   aside.style.height = '100vh';"
    "   aside.style.overflowY = 'auto';"
    "   aside.style.zIndex = '40';"
    "   const main = document.querySelector('main');"
    "   if (main) main.style.marginLeft = aside.getBoundingClientRect().width + 'px';"
    " }"
)


def _JS_SCROLL_MAIN_TO(match_expr: str) -> str:
    """Pin the sidebar so it stays visible, then scroll the document to the
    first ``.card`` whose <h3> textContent matches ``match_expr``.

    The dashboard's ``<main>`` is a flex-child that grows past 100vh, so it
    never scrolls internally — the document does. Pinning the aside as
    fixed keeps it in the shot while document scroll takes the target into
    view.
    """
    return (
        "(() => {"
        + _STICKY_SIDEBAR_CSS +
        " const cards = Array.from(document.querySelectorAll('.card'));"
        " const target = cards.find(c => {"
        "   const h = c.querySelector('h3');"
        "   const t = (h && h.textContent || '').trim();"
        f"   return {match_expr};"
        " });"
        " if (!target) return;"
        " const rect = target.getBoundingClientRect();"
        " const y = window.scrollY + rect.top - 12;"
        " window.scrollTo({ top: Math.max(0, y), behavior: 'instant' });"
        "})()"
    )


def _JS_SCROLL_TO_HEADING(text: str) -> str:
    """Wrapper: scroll main to the card whose <h3> equals ``text`` exactly."""
    safe = text.replace("'", "\\'")
    return _JS_SCROLL_MAIN_TO(f"t === '{safe}'")


def _JS_SCROLL_TO_HEADING_PREFIX(prefix: str) -> str:
    """Wrapper: scroll main to the card whose <h3> starts with ``prefix``."""
    safe = prefix.replace("'", "\\'")
    return _JS_SCROLL_MAIN_TO(f"t.indexOf('{safe}') === 0")


def _js_open_theme_editor() -> str:
    """Install + open a customised copy of a shipped theme so the theme
    editor detail view has content to render.

    Bases the seed on a JSON-backed builtin (dark / light / hc live in CSS,
    NOT in ``themes.list`` — the API only returns disk-based themes — so
    those would silently no-op the ``find`` and we'd land back on the grid).
    ``loadThemes()`` is awaited so ``themes.list`` is populated before the
    find runs.
    """
    return f"""
        (async () => {{
          const s = {_ROOT_STATE};
          if (typeof s.loadThemes === 'function') {{
            await s.loadThemes();
          }}
          const seed = 'demo_custom';
          const list = s.themes.list || [];
          // Prefer a JSON-backed builtin so we have real vars/font to seed
          // from. Fall back to whatever the first entry is.
          const base = list.find(t => t.name === 'ai-slop')
                    || list.find(t => t.source === 'builtin')
                    || list[0];
          if (!base) return;
          await fetch('/api/themes/' + encodeURIComponent(seed), {{
            method: 'PUT',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ font_family: base.font_family || '', vars: base.vars || {{}} }}),
          }});
          await s.loadThemes();
          s.openThemeEditor(seed);
        }})()
    """


def _js_reset_transient_ui() -> str:
    """Close any popover / modal, unpin the sidebar, and reset scroll so
    nothing from the previous shot bleeds into the next one."""
    return f"""
        (() => {{
          const s = {_ROOT_STATE};
          try {{ s.filterPanel = null; }} catch (_) {{}}
          try {{ s.themeEditor.view = 'grid'; s.themeEditor.editing = ''; }} catch (_) {{}}
          try {{ s.showShortcuts = false; }} catch (_) {{}}
          const aside = document.querySelector('aside');
          if (aside && aside.__pinned) {{
            aside.__pinned = false;
            aside.style.position = '';
            aside.style.top = '';
            aside.style.left = '';
            aside.style.height = '';
            aside.style.overflowY = '';
            aside.style.zIndex = '';
            const main = document.querySelector('main');
            if (main) main.style.marginLeft = '';
          }}
          window.scrollTo({{ top: 0, behavior: 'instant' }});
        }})()
    """


# ── Compression: PNG -> smaller PNG ──────────────────────────────────────────


def _shrink_png(src: bytes, target_kb: int = 480) -> bytes:
    """Return a compressed PNG under ``target_kb`` KB where possible.

    Uses Pillow's quantize when the raw PNG is too big — Pillow is already a
    hard dep of cull, so no new import surface.
    """
    from PIL import Image

    if len(src) <= target_kb * 1024:
        return src
    with Image.open(io.BytesIO(src)) as im:
        # Downshift the palette until it fits. 256 -> 128 -> 64 keeps the
        # dashboard readable while ~halving the byte count each step.
        for colors in (256, 192, 128, 96, 64):
            pal = im.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
            buf = io.BytesIO()
            pal.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            if len(data) <= target_kb * 1024:
                return data
        return data  # best-effort — still smaller than the original


def _write_shot(dest: Path, png: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shrunk = _shrink_png(png)
    dest.write_bytes(shrunk)
    size_kb = len(shrunk) // 1024
    print(f"  wrote {dest.relative_to(REPO_ROOT)} ({size_kb} KB)")


# ── Screenshot plan ──────────────────────────────────────────────────────────


def _plan() -> list[Shot]:
    """The 9 primary README shots + their setup."""
    return [
        Shot("01-jobs", view="jobs", active="jobs"),
        Shot("02-global-stats", view="jobs", active="gstats", settle_ms=1500),
        Shot("03-global-gallery", view="jobs", active="globalGallery", settle_ms=1500),
        Shot(
            "04-gallery-filters",
            view="job",
            active="gallery",
            # Wait for the first API response to populate the filter facets
            # (source list, categories, resolution buckets) BEFORE opening
            # the popover — otherwise the panel reads "No sources indexed yet".
            setup=lambda page: (
                page.wait_for_timeout(1500),
                page.evaluate(_js_open_filter_panel("gallery")),
            ),
            settle_ms=1400,
        ),
        Shot("05-preset-marketplace", view="jobs", active="presets", settle_ms=1200),
        Shot("06-themes-picker", view="jobs", active="settings", settle_ms=1200,
             # Deep-scroll to the Themes card once the Settings tab is up.
             setup=lambda page: page.evaluate(_JS_SCROLL_TO_HEADING("Themes"))),
        Shot(
            "07-theme-editor",
            view="jobs",
            active="settings",
            setup=lambda page: (
                page.evaluate(_js_open_theme_editor()),
                page.wait_for_timeout(600),
                page.evaluate(_JS_SCROLL_TO_HEADING_PREFIX("Editing theme")),
            ),
            settle_ms=1200,
        ),
        Shot("08-scrapers", view="job", active="scrapers", settle_ms=800),
        Shot("09-vision", view="job", active="vision", settle_ms=800),
    ]


# ── Main driver ──────────────────────────────────────────────────────────────


def _seed_dataset(base_dir: Path) -> None:
    """Import and run the seeder inside the scoped ``PIPELINE_BASE_DIR``."""
    # Ensure the seeder resolves paths under the scratch dir.
    os.environ["PIPELINE_BASE_DIR"] = str(base_dir)
    os.environ["PIPELINE_QUEUE"] = str(base_dir / "queue")
    os.environ["PIPELINE_SORTED"] = str(base_dir / "sorted")
    os.environ["LOG_DIR"] = str(base_dir / "logs")

    # Force sys.path so we can import the seeder AND pipeline_code helpers.
    for p in (str(REPO_ROOT), str(PIPELINE_CODE)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Import lazily — seed_demo_data itself imports pipeline_code modules.
    import importlib

    seeder = importlib.import_module("tools.seed_demo_data")
    # Wipe first for a deterministic set on every run.
    seeder.reset_all()
    seeder.seed_all(force=True)


def _run(only: set[str] | None, scratch: Path, keep_data: bool, port_hint: int) -> int:
    from playwright.sync_api import sync_playwright, Browser, BrowserContext

    if scratch.exists():
        # Keep any user-configured dir intact if --keep-data set; otherwise
        # start clean so the demo is deterministic on every run.
        if not keep_data:
            shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"[capture] scratch base dir: {scratch}")
    print("[capture] seeding demo dataset...")
    _seed_dataset(scratch)

    port = _pick_free_port(port_hint)
    print(f"[capture] launching dashboard on http://127.0.0.1:{port}")
    proc = _launch_dashboard(port, scratch)

    exit_code = 0
    try:
        if not _wait_reachable(port, timeout=30.0):
            print(
                "[capture] dashboard never became reachable. Re-run with the"
                " dashboard visible (drop --port to a spare port and run"
                " `python pipeline_code/dashboard_enhanced.py` in a"
                " separate shell) to see its logs.",
                file=sys.stderr,
            )
            return 2

        # Small extra wait so the indexer's first sweep finishes committing.
        # ``INDEX_REFRESH_SECONDS=5`` in _launch_dashboard means one full cycle
        # completes inside 6-7s.
        time.sleep(8)

        with sync_playwright() as pw:
            browser: Browser = pw.chromium.launch(headless=True)
            context: BrowserContext = browser.new_context(
                viewport=VIEWPORT,
                device_scale_factor=DEVICE_PIXEL_RATIO,
            )

            # Pre-seed localStorage so the welcome banner + NSFW blur don't
            # sit on top of the shots. Doing it via addInitScript means every
            # page load in the context already has the values set.
            context.add_init_script(
                """
                try {
                  localStorage.setItem('cull_welcome_dismissed', '1');
                  localStorage.setItem('cull_show_nsfw', '1');
                } catch (_) {}
                """
            )

            page = context.new_page()
            page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
            # Alpine bootstraps async — wait for the root state to exist.
            page.wait_for_function(
                "() => document.querySelector('[x-data]') "
                "&& document.querySelector('[x-data]')._x_dataStack"
            )
            page.wait_for_timeout(500)
            # Re-assert NSFW toggle on the live component (the addInitScript
            # value is only read on start(); flipping here also nudges Alpine).
            page.evaluate(_js_toggle_show_nsfw(True))

            for shot in _plan():
                if only and shot.slug.split("-", 1)[0] not in only:
                    continue
                dest = DOCS_SCREENSHOTS / f"{shot.slug}.png"
                print(f"[capture] -> {shot.slug}")

                # Clear anything the previous shot's setup left open — a
                # gallery filter popover, an open theme editor, etc.
                page.evaluate(_js_reset_transient_ui())
                page.evaluate(_js_set_theme(shot.theme))
                # For the 'job' scope, aim at the first demo slug so the URL
                # / sidebar shows real content.
                slug = "car_ads" if shot.view == "job" else None
                page.evaluate(_js_go_to(shot.view, shot.active, slug))
                # Kick a fresh themes load whenever we head into Settings so
                # the Themes card is populated before the shot's setup fires.
                if shot.active == "settings":
                    page.evaluate(f"{_ROOT_STATE}.loadThemes && {_ROOT_STATE}.loadThemes()")
                    page.wait_for_timeout(300)
                page.wait_for_timeout(400)
                if shot.setup is not None:
                    shot.setup(page)
                page.wait_for_timeout(shot.settle_ms)

                # Snap only the viewport so every shot is 1440x900. Full-page
                # would produce a 20MB file for the settings tab.
                png = page.screenshot(full_page=False)
                _write_shot(dest, png)

            # ── Theme gallery: one canonical view per shipped theme. ────
            print("[capture] rendering theme gallery...")
            THEMES_OUT.mkdir(parents=True, exist_ok=True)
            for theme in ALL_THEMES:
                dest = THEMES_OUT / f"{theme}.png"
                page.evaluate(_js_set_theme(theme))
                page.evaluate(_js_go_to("jobs", "gstats"))
                page.wait_for_timeout(1200)
                png = page.screenshot(full_page=False)
                _write_shot(dest, png)

            context.close()
            browser.close()
    except KeyboardInterrupt:
        print("[capture] interrupted", file=sys.stderr)
        exit_code = 130
    except Exception as exc:  # noqa: BLE001 — top-level driver
        print(f"[capture] failed: {exc!r}", file=sys.stderr)
        exit_code = 1
    finally:
        _terminate_dashboard(proc)
        if not keep_data:
            shutil.rmtree(scratch, ignore_errors=True)

    return exit_code


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture README screenshots against a seeded demo dashboard."
    )
    parser.add_argument(
        "--only",
        help="Comma-separated shot ids to capture (e.g. '01,04,06'). Default: all.",
    )
    parser.add_argument(
        "--scratch",
        default=str(DEFAULT_SCRATCH),
        help=f"Scratch data dir for the demo. Default: {DEFAULT_SCRATCH}.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Do not delete the scratch dir on exit (useful for debugging).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5090,
        help="Preferred loopback port for the dashboard (default 5090).",
    )
    args = parser.parse_args(argv)

    only: set[str] | None = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}

    scratch = Path(args.scratch).expanduser().resolve()
    return _run(only=only, scratch=scratch, keep_data=args.keep_data, port_hint=args.port)


if __name__ == "__main__":
    sys.exit(main())
