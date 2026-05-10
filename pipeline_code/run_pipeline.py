"""
run_pipeline.py - Topic-driven pipeline orchestrator.

Now supervised: a background loop reconciles *desired* state (derived from .env)
with *actual* state (spawned subprocesses) every few seconds. Toggle a scraper
or vision worker on in the dashboard -> its process starts within the next
reconcile tick. Toggle off -> process gets terminated. No pipeline restart.

All paths resolved from .env via paths.py. No hardcodes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from paths import base_dir

load_dotenv()

logger = logging.getLogger("pipeline")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

PY: str = sys.executable
PIPELINE_CODE_DIR: Path = Path(os.environ.get("PIPELINE_CODE_DIR", Path(__file__).parent))
BASE_DIR: Path = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))

RECONCILE_SECONDS: int = int(os.environ.get("PIPELINE_RECONCILE_SECONDS", 2))

ENV_PATH: Path = Path(os.environ.get("WORKSPACE_ROOT", PIPELINE_CODE_DIR.parent)) / ".env"

# Env vars whose CHANGE means in-flight children read stale config and must
# be respawned. Toggling SCRAPER_DISABLED or PIPELINE_VISION_WORKERS isn't
# in this list — those drive the desired-set, not in-process behaviour, so
# they are handled by the normal start/stop reconcile without restarting
# unrelated children.
STRUCTURAL_ENV_KEYS: tuple[str, ...] = (
    "PIPELINE_TOPIC", "PIPELINE_SLUG", "PIPELINE_BASE_DIR",
    "PIPELINE_QUEUE", "PIPELINE_SORTED", "LOG_DIR",
    "X_ACCOUNTS", "REDDIT_SUBREDDITS", "TOPIC_KEYWORDS_EXTRA",
    "TOPIC_BANNED_KEYWORDS", "TOPIC_GENERATION_HINTS", "MIN_PROMPT_LENGTH",
    "LMSTUDIO_PRIMARY_URL", "LMSTUDIO_PRIMARY_MODEL", "LMSTUDIO_PRIMARY_TIMEOUT",
    "LMSTUDIO_SECONDARY_URL", "LMSTUDIO_SECONDARY_MODEL", "LMSTUDIO_SECONDARY_TIMEOUT",
    "GROQ_API_KEYS", "GROQ_API_KEY", "GROQ_MODEL",
    "GEMINI_API_KEY", "GEMINI_MODEL",
    "VISION_OVR_MIN_SCORE", "VISION_REL_MIN_SCORE", "VISION_SCORE_NOTES",
    "CIVITAI_API_KEY", "CIVITAI_API_RED_KEY", "CIVITAI_DOMAINS",
    "CIVITAI_SEARCH_URL", "CIVITAI_SEARCH_HOST", "CIVITAI_TRPC_BASE",
    "ZFORFREE_LOCAL_SRC", "LOCAL_IMPORT_DIR", "LOCAL_IMPORT_NAME",
    "LOCAL_IMPORT_MIGRATE_FROM", "TWITTER_COOKIES",
    "DISCORD_BOT_TOKEN", "DISCORD_AUTH_MODE",
    "GALLERY_DL_URLS", "GALLERY_DL_COOKIES_FILE", "GALLERY_DL_CONFIG_PATH",
    "GALLERY_DL_LIMIT_PER_URL",
    "REQUIRE_PROMPT",
    "AUTO_CAPTION_ENABLED", "AUTO_CAPTION_STYLE", "AUTO_CAPTION_OVERWRITE",
)


def _structural_env_snapshot() -> dict[str, str]:
    return {key: os.environ.get(key, "") for key in STRUCTURAL_ENV_KEYS}


# ── Static scrapers ────────────────────────────────────────────────────────────

CHANNEL_GROUPS: list[list[dict]] = [
    [
        {"id": "1013128131764305930", "name": "UD #photorealistic",  "guild": "Unstable Diffusion", "kind": "png_embed"},
        {"id": "1053354830921486498", "name": "UD #photography",     "guild": "Unstable Diffusion", "kind": "png_embed"},
        {"id": "1011063627488440401", "name": "UD #women-only",      "guild": "Unstable Diffusion", "kind": "png_embed"},
        {"id": "1054191232466833478", "name": "UD #requests-sfw",    "guild": "Unstable Diffusion", "kind": "png_embed"},
        {"id": "1011861076570275840", "name": "UD #prompts-woman",   "guild": "Unstable Diffusion", "kind": "png_embed"},
    ],
]


@dataclass
class AgentSpec:
    """Blueprint for one long-running child process."""
    label: str
    script: str
    args: list[str] = field(default_factory=list)
    env_override: dict[str, str] = field(default_factory=dict)
    loop_sleep: int = 300  # seconds between respawns if the child exits on its own


# ── Helpers ────────────────────────────────────────────────────────────────────

def topic_slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


def _reload_env() -> None:
    """Reload .env from disk so dashboard edits are picked up live."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)


def disabled_scrapers() -> set[str]:
    raw = os.environ.get("SCRAPER_DISABLED", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def vision_worker_list() -> list[str]:
    raw = os.environ.get("PIPELINE_VISION_WORKERS", "").strip()
    if raw:
        return [w.strip() for w in raw.split(",") if w.strip()]
    single = os.environ.get("PIPELINE_VISION_WORKER", "").strip()
    return [single] if single else []


# ── Desired-state computation ─────────────────────────────────────────────────

def _vision_spec(worker: str) -> AgentSpec | None:
    """Map a vision worker name to its script + env overrides.

    Looks up the registry in ``vision_workers.py``. Unknown names return
    ``None`` (the supervisor's `add()` filter then drops them silently).
    """
    from vision_workers import WORKERS
    spec = WORKERS.get(worker)
    if spec is None:
        return None
    return AgentSpec(
        label=f"Vision-{worker}",
        script=spec.script,
        env_override=spec.env_override(),
        loop_sleep=10,
    )


def compute_desired_agents(topic: str) -> dict[str, AgentSpec]:
    """Build {label: AgentSpec} for everything that *should* be running right now."""
    disabled = disabled_scrapers()
    agents: dict[str, AgentSpec] = {}

    def add(spec: AgentSpec) -> None:
        if spec.label in disabled:
            return
        if not (PIPELINE_CODE_DIR / spec.script).exists():
            return
        agents[spec.label] = spec

    # Scrapers
    add(AgentSpec(label="X.com",   script="scraper_x.py",   loop_sleep=1800))
    for idx, group in enumerate(CHANNEL_GROUPS):
        add(AgentSpec(label=f"Discord-{idx + 1}", script="scraper_discord.py",
                      args=[json.dumps(group)], loop_sleep=1800))
    for domain in (d.strip() for d in os.environ.get("CIVITAI_DOMAINS", "civitai.com,civitai.red").split(",") if d.strip()):
        domain_label = "Civitai-Red" if domain == "civitai.red" else "Civitai-Com"
        add(AgentSpec(label=domain_label, script="scraper_civitai_search.py",
                      env_override={"CIVITAI_DOMAIN": domain}, loop_sleep=600))
    add(AgentSpec(label="Web", script="scraper_web.py", loop_sleep=1800))

    # ZFF-Local honours its own flag AND topic gating
    if os.environ.get("ZFORFREE_LOCAL_ENABLED", "false").lower() == "true":
        human_keywords = ("influencer", "instagram", "woman", "girl", "female", "male", "man",
                          "person", "portrait", "model", "instagrammer")
        if any(kw in topic.lower() for kw in human_keywords):
            add(AgentSpec(label="ZFF-Local", script="feed_zforfree_local.py", loop_sleep=3600))

    # Generic local importer - label follows LOCAL_IMPORT_NAME so toggles match
    if os.environ.get("LOCAL_IMPORT_ENABLED", "false").lower() == "true":
        local_name = (os.environ.get("LOCAL_IMPORT_NAME", "local") or "local").strip() or "local"
        add(AgentSpec(label=f"Local-{local_name}", script="feed_local_folder.py", loop_sleep=3600))

    # gallery-dl URL-based scraper (Pixiv, DeviantArt, booru sites, ArtStation,
    # Tumblr, Newgrounds, X, Reddit, Imgur, Flickr — anything gallery-dl knows).
    # Only desired when both the toggle is on AND at least one URL is configured;
    # otherwise the supervisor would respawn an empty agent every loop_sleep.
    if (
        os.environ.get("GALLERY_DL_ENABLED", "false").lower() == "true"
        and (os.environ.get("GALLERY_DL_URLS", "") or "").strip()
    ):
        add(AgentSpec(label="Gallery-DL", script="scraper_gallery_dl.py", loop_sleep=1800))

    # Vision workers (the Vision-* labels are also filterable via SCRAPER_DISABLED
    # so admins can force everything off with one bulk call.)
    for worker in vision_worker_list():
        spec = _vision_spec(worker)
        if spec is not None:
            add(spec)

    return agents


# ── Supervisor ────────────────────────────────────────────────────────────────

class Supervisor:
    """Reconciles desired agents with actually-running subprocesses."""

    def __init__(self, topic: str, base_env: dict[str, str], log_file) -> None:
        self.topic = topic
        self.base_env = base_env
        self.log_file = log_file
        self._lock = threading.Lock()
        self._active: dict[str, subprocess.Popen] = {}  # label -> proc
        self._desired_snapshot: dict[str, AgentSpec] = {}
        self._stop = threading.Event()
        # When a child exits on its own, we hold off respawning until the
        # spec's loop_sleep window has elapsed - otherwise a fast-failing
        # scraper (e.g. Discord 401-looping every second) gets respawned on
        # every reconcile tick and floods the log.
        self._cooldown_until: dict[str, float] = {}
        # mtime of the .env file the last time we spawned / restarted. When the
        # dashboard writes new values the mtime changes, which we detect here
        # and use to trigger a soft restart so every child picks up fresh env.
        self._env_mtime: float = self._current_env_mtime()
        self._categories_mtime: float = self._current_categories_mtime()
        self._struct_snapshot: dict[str, str] = _structural_env_snapshot()
        self._queue_dir: Path | None = None  # set by run_topic before start()

    @staticmethod
    def _current_env_mtime() -> float:
        try:
            return ENV_PATH.stat().st_mtime
        except OSError:
            return 0.0

    @staticmethod
    def _current_categories_mtime() -> float:
        """mtime of the user's categories file. Soft-restart workers when it
        changes so the JSON-schema enum + per-category prompt hints land in
        every child's process."""
        from categories import ACTIVE_PATH
        try:
            return ACTIVE_PATH.stat().st_mtime
        except OSError:
            return 0.0

    def _spawn(self, spec: AgentSpec) -> None:
        script_path = PIPELINE_CODE_DIR / spec.script
        run_env = {**self.base_env, **spec.env_override}
        args = [PY, "-u", str(script_path)] + spec.args
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env)
        self._active[spec.label] = proc
        print(f"  [+] Started {spec.label}", flush=True)
        threading.Thread(
            target=self._stream_output, args=(proc, spec.label), daemon=True,
        ).start()

    def _stream_output(self, proc: subprocess.Popen, label: str) -> None:
        try:
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                try:
                    print(f"[{label}] {text}", flush=True)
                except Exception:
                    print(f"[{label}] {text.encode('ascii', 'replace').decode()}", flush=True)
                if self.log_file:
                    self.log_file.write(f"[{label}] {text}\n")
                    self.log_file.flush()
        except Exception as exc:
            logger.warning("[%s] monitor thread error: %s", label, exc)

    def _terminate(self, label: str) -> None:
        proc = self._active.pop(label, None)
        if proc is None or proc.poll() is not None:
            return
        print(f"  [-] Stopping {label} (pid {proc.pid})", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _sweep_stale_processing(self) -> None:
        """Revert `.processing` files back to their original name before respawning.

        Vision workers rename an image to `<stem>.processing` while classifying.
        If we terminate mid-flight, the rename is orphaned and the image never
        gets reprocessed. After a restart we put them back so they re-enter the
        queue on the next poll.
        """
        if self._queue_dir is None or not self._queue_dir.exists():
            return
        reverted = 0
        for proc_file in self._queue_dir.glob("**/*.processing"):
            try:
                original = proc_file.with_suffix("")
                if not original.exists():
                    proc_file.rename(original)
                    reverted += 1
            except OSError:
                pass
        if reverted:
            print(f"  [env-reload] restored {reverted} in-flight image(s) to the queue", flush=True)

    def reconcile(self) -> None:
        _reload_env()
        structural_changed = False
        current_mtime = self._current_env_mtime()
        if current_mtime and current_mtime != self._env_mtime:
            self._env_mtime = current_mtime
            self.base_env = {**self.base_env, **os.environ}
            new_snapshot = _structural_env_snapshot()
            if new_snapshot != self._struct_snapshot:
                structural_changed = True
                # Identify exactly which keys flipped — useful in the supervisor log.
                diff = [
                    f"{k}: {self._struct_snapshot.get(k, '')!r} -> {new_snapshot.get(k, '')!r}"
                    for k in STRUCTURAL_ENV_KEYS
                    if self._struct_snapshot.get(k) != new_snapshot.get(k)
                ]
                self._struct_snapshot = new_snapshot
                print("  [env-reload] structural change detected; soft-restarting children", flush=True)
                for line in diff[:6]:
                    print(f"    - {line}", flush=True)
            else:
                # SCRAPER_DISABLED / PIPELINE_VISION_WORKERS only — start/stop deltas
                # are handled by the normal reconcile loop below; no restart needed.
                print("  [env-reload] toggle change picked up (no soft-restart needed)", flush=True)

        # Categories file edits also force a soft restart so workers pick up
        # the new schema + prompt at their next spawn.
        cats_mtime = self._current_categories_mtime()
        if cats_mtime and cats_mtime != self._categories_mtime:
            self._categories_mtime = cats_mtime
            structural_changed = True
            print("  [env-reload] categories file changed; soft-restarting children", flush=True)

        desired = compute_desired_agents(self.topic)
        self._desired_snapshot = desired

        with self._lock:
            if structural_changed:
                for label in list(self._active.keys()):
                    self._terminate(label)
                self._sweep_stale_processing()

            # Stop anything that isn't desired or has exited.
            for label in list(self._active.keys()):
                proc = self._active[label]
                exited = proc.poll() is not None
                if label not in desired:
                    self._terminate(label)
                elif exited:
                    # Mark the cooldown window so we don't respawn next tick.
                    self._active.pop(label, None)
                    spec = desired.get(label)
                    cooldown = float(spec.loop_sleep if spec else 60)
                    self._cooldown_until[label] = time.monotonic() + cooldown
                    print(
                        f"  [·] {label} exited (code {proc.returncode}); "
                        f"next respawn in {int(cooldown)}s",
                        flush=True,
                    )

            # Start anything desired that isn't active and isn't cooling down.
            now = time.monotonic()
            for label, spec in desired.items():
                if label in self._active:
                    continue
                until = self._cooldown_until.get(label, 0.0)
                if until > now:
                    continue  # still in cooldown
                if until and until <= now:
                    self._cooldown_until.pop(label, None)
                self._spawn(spec)

    def run(self) -> None:
        """Reconcile loop with fast-path env-change detection.

        Background: full reconciles run every RECONCILE_SECONDS, but we ALSO
        poll .env's mtime every 0.5s. When the dashboard writes a setting the
        mtime bumps within milliseconds, so toggles feel instant (~1s) instead
        of waiting for the next full reconcile tick.
        """
        print(
            f"\nSupervisor online. Full reconcile every {RECONCILE_SECONDS}s; "
            "env-change polled at 0.5s for instant toggles.",
            flush=True,
        )
        env_poll_interval = 0.5
        try:
            self.reconcile()  # initial
            last_full = time.monotonic()
            while not self._stop.is_set():
                time.sleep(env_poll_interval)
                now = time.monotonic()
                env_bumped = self._current_env_mtime() != self._env_mtime
                cats_bumped = self._current_categories_mtime() != self._categories_mtime
                if env_bumped or cats_bumped or (now - last_full) >= RECONCILE_SECONDS:
                    self.reconcile()
                    last_full = now
        except KeyboardInterrupt:
            print("\nPipeline stopped by user.", flush=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        for label in list(self._active.keys()):
            self._terminate(label)


# ── Topic runner ──────────────────────────────────────────────────────────────

def run_topic(topic: str, vision_worker: str = "balanced-groq") -> None:
    slug = topic_slug(topic)
    print(f"\n{'=' * 60}", flush=True)
    print(f"=== TOPIC: {topic} (slug: {slug}) ===", flush=True)
    print(f"=== VISION WORKER LIST: {vision_worker_list() or [vision_worker]} ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    queue_root = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
    sorted_root = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted")))
    queue_dir = queue_root if queue_root.name == slug else queue_root / slug
    sorted_dir = sorted_root if sorted_root.name == slug else sorted_root / slug
    queue_dir.mkdir(parents=True, exist_ok=True)
    # Pre-create category folders so the first vision-worker write doesn't
    # race against mkdir. Read live so user-edited taxonomy is picked up at
    # supervisor (re)start. Workers also mkdir lazily in _finalise as a
    # belt-and-braces guard against new categories added mid-run.
    from categories import get_all_categories
    for cat in get_all_categories():
        (sorted_dir / cat).mkdir(parents=True, exist_ok=True)

    log_dir = Path(os.environ.get("LOG_DIR", str(BASE_DIR / "logs_test")))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{slug}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    base_env = {
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "PIPELINE_TOPIC": topic,
        "PIPELINE_SLUG": slug,
        "PIPELINE_QUEUE": str(queue_dir),
        "PIPELINE_SORTED": str(sorted_dir),
        **os.environ,
    }

    # If vision_worker was passed via CLI and PIPELINE_VISION_WORKERS env is empty,
    # seed the list so the initial reconcile has something to work with.
    if not vision_worker_list() and vision_worker:
        os.environ["PIPELINE_VISION_WORKERS"] = vision_worker

    supervisor = Supervisor(topic=topic, base_env=base_env, log_file=log_file)
    supervisor._queue_dir = queue_dir  # so stale .processing cleanup knows where to look

    def _handle_sigint(_sig, _frame):
        supervisor.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        supervisor.run()
    finally:
        log_file.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Topic-driven image prompt pipeline")
    parser.add_argument("--topic", action="append", dest="topics",
                        help="Topic to scrape (can be repeated for multiple topics)")
    parser.add_argument("--vision-worker", dest="vision_worker",
                        default=os.environ.get("PIPELINE_VISION_WORKER", "balanced-groq"),
                        help="Default vision worker (used only if PIPELINE_VISION_WORKERS is empty)")
    args = parser.parse_args()

    topics = args.topics or [os.environ.get("PIPELINE_TOPIC", "Realistic Female Influencer")]

    print("=== Pipeline Orchestrator ===", flush=True)
    print(f"Pipeline code dir: {PIPELINE_CODE_DIR}", flush=True)
    print(f"Topics: {topics}", flush=True)

    for topic in topics:
        run_topic(topic, vision_worker=args.vision_worker)

    print("\n=== All topics complete ===", flush=True)


if __name__ == "__main__":
    main()
