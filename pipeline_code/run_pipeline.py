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

RECONCILE_SECONDS: int = int(os.environ.get("PIPELINE_RECONCILE_SECONDS", 5))

ENV_PATH: Path = Path(os.environ.get("WORKSPACE_ROOT", PIPELINE_CODE_DIR.parent)) / ".env"


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

    `balanced-lm-secondary` maps to the same worker script as `balanced-lm`
    but forces the primary LMStudio env vars to the SECONDARY values, so both
    LMStudios can run in parallel on different labels.
    """
    if worker == "lm-autodetect":
        return AgentSpec(label=f"Vision-{worker}", script="vision_worker_lm_autodetect.py", loop_sleep=10)
    if worker == "local":
        return AgentSpec(label=f"Vision-{worker}", script="vision_worker.py", loop_sleep=10)
    if worker == "balanced-lm-secondary":
        return AgentSpec(
            label="Vision-balanced-lm-secondary",
            script="vision_worker_balanced_lm.py",
            env_override={
                "LMSTUDIO_PRIMARY_URL":     os.environ.get("LMSTUDIO_SECONDARY_URL", ""),
                "LMSTUDIO_PRIMARY_MODEL":   os.environ.get("LMSTUDIO_SECONDARY_MODEL", ""),
                "LMSTUDIO_PRIMARY_TIMEOUT": os.environ.get("LMSTUDIO_SECONDARY_TIMEOUT", "60"),
            },
            loop_sleep=10,
        )
    script = f"vision_worker_{worker.replace('-', '_')}.py"
    return AgentSpec(label=f"Vision-{worker}", script=script, loop_sleep=10)


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
        # mtime of the .env file the last time we spawned / restarted. When the
        # dashboard writes new values the mtime changes, which we detect here
        # and use to trigger a soft restart so every child picks up fresh env.
        self._env_mtime: float = self._current_env_mtime()
        self._queue_dir: Path | None = None  # set by run_topic before start()

    @staticmethod
    def _current_env_mtime() -> float:
        try:
            return ENV_PATH.stat().st_mtime
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
        env_changed = False
        current_mtime = self._current_env_mtime()
        if current_mtime and current_mtime != self._env_mtime:
            env_changed = True
            print("  [env-reload] .env changed - restarting all child processes so new settings take effect", flush=True)
            self._env_mtime = current_mtime

        desired = compute_desired_agents(self.topic)
        self._desired_snapshot = desired

        with self._lock:
            if env_changed:
                # Hard reset: refresh base_env, kill everything, sweep stale state.
                # The normal reconcile loop below will respawn desired agents with
                # the new env on the same tick.
                self.base_env = {**self.base_env, **os.environ}
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
                    # Respect the per-spec loop_sleep cooldown before respawn.
                    self._active.pop(label, None)
                    print(f"  [·] {label} exited (code {proc.returncode}), will restart on next tick", flush=True)

            # Start anything desired that isn't active.
            for label, spec in desired.items():
                if label not in self._active:
                    self._spawn(spec)

    def run(self) -> None:
        print(f"\nSupervisor online. Reconciling every {RECONCILE_SECONDS}s.", flush=True)
        try:
            while not self._stop.is_set():
                self.reconcile()
                time.sleep(RECONCILE_SECONDS)
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
    for cat in ("InstagramInfluencer", "NSFW", "Professional", "Amateur", "Unknown", "CORRUPT", "DISCARD"):
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
