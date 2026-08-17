"""
run_pipeline.py - Job-driven pipeline orchestrator.

Now supervised: a background loop reconciles *desired* state with *actual* state
(spawned subprocesses) every few seconds. Toggle a scraper or vision worker on in
the dashboard -> its process starts within the next reconcile tick. Toggle off ->
process gets terminated. No pipeline restart.

Jobs model (see docs/jobs-model-design.md §7): the supervisor runs the ACTIVE job
from ``job_config``. A job is the source of truth; *activating* it projects its
config down into the two contracts the runtime already consumes —

  1. env vars (``job_config.resolve_env``) merged over the global ``.env`` when we
     spawn children, and
  2. the active taxonomy file ``cull_categories.json`` (``project_categories``).

Switching the active job reuses the EXISTING hot-reload machinery: we watch
``data/jobs/_index.json`` mtime alongside ``.env`` and ``cull_categories.json``,
and when the active slug changes we re-resolve env + re-project categories and
trigger the same structural restart the supervisor already does on a structural
``.env`` change (the resolved keys are already in STRUCTURAL_ENV_KEYS).

Global concerns (credentials, model endpoints, vision worker selection, throttle)
still come from ``.env`` and keep their own mtime watch.

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

import job_config
from paths import base_dir
from paths import validate_slug as _validate_slug

load_dotenv()

logger = logging.getLogger("pipeline")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

PY: str = sys.executable
PIPELINE_CODE_DIR: Path = Path(os.environ.get("PIPELINE_CODE_DIR", Path(__file__).parent))
BASE_DIR: Path = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))

RECONCILE_SECONDS: int = int(os.environ.get("PIPELINE_RECONCILE_SECONDS", 2))

ENV_PATH: Path = Path(os.environ.get("WORKSPACE_ROOT", PIPELINE_CODE_DIR.parent)) / ".env"

# The jobs index — its mtime bumps whenever the dashboard activates / advances a
# job. Watched alongside ENV_PATH so a job switch triggers a structural restart.
JOBS_INDEX_PATH: Path = job_config.jobs_dir() / "_index.json"

# How long the idle loop waits between checks when no job is active. Kept short so
# activating a job from the dashboard starts the pipeline within ~1s.
IDLE_POLL_SECONDS: float = 1.0

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
    # Local folders are projected as a single JSON blob; changing the set of
    # folders (add/remove/retarget) must restart the feeders. The per-folder
    # LOCAL_IMPORT_* vars are no longer job-global — the supervisor sets them
    # per-feeder-agent from LOCAL_IMPORTS_JSON, so they're not tracked here.
    "LOCAL_IMPORTS_JSON",
    # The local vision fleet is projected as a single JSON blob; changing the set
    # of workers (add/remove/retarget/rekey) must restart the vision workers. The
    # per-instance OPENAI_COMPAT_*/OLLAMA_* vars are set per-agent from this blob,
    # so they aren't tracked individually.
    "VISION_WORKERS_JSON",
    "TWITTER_COOKIES",
    "DISCORD_BOT_TOKEN", "DISCORD_AUTH_MODE",
    "GALLERY_DL_URLS", "GALLERY_DL_COOKIES_FILE", "GALLERY_DL_CONFIG_PATH",
    "GALLERY_DL_LIMIT_PER_URL",
    "YT_DLP_URLS", "YT_DLP_COOKIES", "YT_DLP_LIMIT",
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
    """Blueprint for one long-running child process.

    ``env`` is merged OVER the supervisor's base spawn env when this agent is
    launched, so each agent can carry per-agent overrides (e.g. a civitai
    domain, a vision-worker's keepalive flag, or a local folder's
    LOCAL_IMPORT_DIR/NAME fanned out from LOCAL_IMPORTS_JSON).
    """
    label: str
    script: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
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


def _local_import_folders() -> list[dict]:
    """Parse LOCAL_IMPORTS_JSON into the list of enabled local-folder dicts.

    job_config.resolve_env emits LOCAL_IMPORTS_JSON as a JSON array of
    ``{name, dir, migrate_from}`` (already filtered to ENABLED folders). We parse
    it defensively: empty / missing / malformed JSON → ``[]``, non-list payload →
    ``[]``, and any non-dict element or an entry with no ``dir`` is skipped so a
    garbled value can never spawn a broken feeder.
    """
    raw = (os.environ.get("LOCAL_IMPORTS_JSON", "") or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    folders: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if not str(entry.get("dir", "") or "").strip():
            continue  # a folder with no path can't be imported
        folders.append(entry)
    return folders


# ── Jobs model: active-job resolution (pure helpers, unit-tested) ──────────────

def desired_active_slug() -> str | None:
    """The slug of the job the supervisor SHOULD be running, or None to idle.

    Thin wrapper over ``job_config.get_active_slug`` so the supervisor's
    active-slug change detection has one named seam to test/mock.
    """
    return job_config.get_active_slug()


def active_job_env(slug: str | None = None) -> dict[str, str] | None:
    """Build the spawn/base environment for the active (or given) job.

    Returns ``{**os.environ, **job_config.resolve_env(job)}`` — the job's
    resolved env-var names overlaid on the process env so scrapers + vision
    workers receive the job's PIPELINE_TOPIC/SLUG, SCRAPER_DISABLED, X_ACCOUNTS,
    scoring, captioning, etc. while still inheriting global ``.env`` values
    (credentials, model endpoints) that the job never stores.

    Returns ``None`` when there is no active job, or when the active slug points
    at a job file that no longer exists — the caller idles instead of spawning
    children against a stale global config.
    """
    slug = slug if slug is not None else desired_active_slug()
    if not slug:
        return None
    job = job_config.get_job(slug)
    if job is None:
        return None
    return {**os.environ, **job_config.resolve_env(job)}


# ── Optional scheduler tick (gated SCHEDULER_ENABLED, default OFF) ─────────────
#
# When enabled, the supervisor ticks the persisted schedules on its reconcile
# loop. Each due schedule fans out to an existing supervisor primitive via
# ``_schedule_runner``. Everything here is best-effort: a missing optional
# ``scheduler`` module, or any runner error, is logged and swallowed so a
# schedule failure can NEVER stop the supervisor. With SCHEDULER_ENABLED unset
# the tick is a no-op and behaviour is byte-identical.

def _scheduler_enabled() -> bool:
    return os.environ.get("SCHEDULER_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _schedule_runner(slug: str, action: str) -> None:
    """Map a due ``(slug, action)`` schedule onto existing supervisor primitives.

    * ``scrape``  -> activate the job (projects its config + scraper toggles down
      via ``job_config.activate``; the supervisor's index-watch then restarts
      into it on the next reconcile).
    * ``curate``  -> ensure the job's vision workers run, which (in the single
      active-job model) means activating it so its fleet becomes desired.
    * ``export``  -> best-effort dataset export via ``export_profiles``.

    Kept tiny and side-effect-only so ``scheduler.run_due`` (which already
    swallows runner exceptions) drives it; we still guard ``export`` locally.
    """
    if action in ("scrape", "curate"):
        job_config.activate(slug)
    elif action == "export":
        try:
            import export_profiles
            out_dir = BASE_DIR / "exports" / slug
            export_profiles.export_dataset(slug, "folders", out_dir)
        except Exception as exc:  # noqa: BLE001 - export is best-effort
            logger.warning("scheduled export for %r failed: %s", slug, exc)


def _scheduler_tick() -> None:
    """Run any due schedules. Gated + fully defensive (never raises).

    Lazily imports ``scheduler`` so the supervisor still runs if the optional
    module/deps are absent. A no-op unless ``SCHEDULER_ENABLED`` is truthy.
    """
    if not _scheduler_enabled():
        return
    try:
        import scheduler
        scheduler.run_due_now(_schedule_runner)
    except Exception as exc:  # noqa: BLE001 - a schedule failure must not stop the supervisor
        logger.warning("scheduler tick failed: %s", exc)


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
        env=spec.env_override(),  # WorkerSpec.env_override() -> per-agent env dict
        loop_sleep=10,
    )


# Local-LLM provider -> worker script. lmstudio + llama.cpp speak the OpenAI
# /v1 API (one script); ollama uses its native API.
_VISION_PROVIDER_SCRIPTS: dict[str, str] = {
    "lmstudio": "vision_worker_balanced_openai.py",
    "llamacpp": "vision_worker_balanced_openai.py",
    "ollama": "vision_worker_balanced_ollama.py",
}


def _vision_fleet() -> list[dict]:
    """Parse VISION_WORKERS_JSON (projected by job_config.resolve_env) into the
    list of usable local worker instances. Defensive: empty / missing / malformed
    JSON, non-list payloads, non-dict entries, blank base_url, and unknown
    providers are all dropped so a garbled value can never spawn a broken worker.
    """
    raw = (os.environ.get("VISION_WORKERS_JSON", "") or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    fleet: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if not str(entry.get("base_url", "") or "").strip():
            continue
        if entry.get("provider") not in _VISION_PROVIDER_SCRIPTS:
            continue
        fleet.append(entry)
    return fleet


def _failover_enabled() -> bool:
    """Gate for health-probed failover (default OFF).

    When unset/false the supervisor fans out EVERY enabled fleet instance exactly
    as before — no probing, byte-identical behaviour. When truthy, the fleet is
    health-probed and unreachable endpoints are skipped before fan-out.
    """
    return os.environ.get("VISION_FAILOVER_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _probe_timeout() -> float:
    """Per-endpoint liveness-probe timeout in seconds (failover path only).

    Reads ``VISION_PROBE_TIMEOUT`` (default 5). A malformed value falls back to
    the default rather than crashing the reconcile loop.
    """
    try:
        return float(os.environ.get("VISION_PROBE_TIMEOUT", "5"))
    except (TypeError, ValueError):
        return 5.0


def _healthy_fleet(fleet: list[dict]) -> list[dict]:
    """Filter ``fleet`` to reachable endpoints when failover is enabled.

    OFF (default): returns ``fleet`` unchanged — no import, no network, the
    spawn set is identical to today. ON: lazily imports ``fleet_health``, probes
    every endpoint, and drops the ones whose probe failed (logging each skip).
    Best-effort — if the optional module is missing or probing blows up, we log
    and fall back to the unfiltered fleet so a probe fault can never starve the
    pipeline of workers.
    """
    if not fleet or not _failover_enabled():
        return fleet
    try:
        import fleet_health
        probes = fleet_health.probe_fleet(fleet, timeout=_probe_timeout())
        healthy = fleet_health.pick_healthy(fleet, probes)
    except Exception as exc:  # noqa: BLE001 - probing must never stop fan-out
        logger.warning("vision failover probe failed; spawning full fleet: %s", exc)
        return fleet
    healthy_ids = {str(ep.get("id", "") or "") for ep in healthy}
    for ep in fleet:
        eid = str(ep.get("id", "") or "")
        if eid not in healthy_ids:
            name = str(ep.get("name", "") or eid or "?")
            print(f"  [failover] skipping unhealthy endpoint {name}", flush=True)
    return healthy


def _vision_fleet_specs() -> list[AgentSpec]:
    """Fan VISION_WORKERS_JSON out into one Vision-<name> AgentSpec per instance,
    each carrying its own endpoint env so the (unchanged) worker script reads its
    URL/model/key from its own process env — mirrors the LOCAL_IMPORTS_JSON
    fan-out for local folders.

    When ``VISION_FAILOVER_ENABLED`` is set, the fleet is first health-probed and
    unreachable endpoints are dropped (see ``_healthy_fleet``); when unset the
    fleet is fanned out unchanged.
    """
    specs: list[AgentSpec] = []
    seen: set[str] = set()
    for i, w in enumerate(_healthy_fleet(_vision_fleet())):
        provider = w["provider"]
        name = (str(w.get("name", "") or w.get("id", "") or f"w{i}")).strip() or f"w{i}"
        label = base = f"Vision-{name}"
        n = 2
        while label in seen:                     # keep labels unique on name clash
            label, n = f"{base}-{n}", n + 1
        seen.add(label)
        env = {"VISION_INSTANCE_ID": str(w.get("id", "") or name),
               "VISION_INSTANCE_NAME": name}
        base_url = str(w.get("base_url", "") or "").strip()
        model = str(w.get("model", "") or "")
        api_key = str(w.get("api_key", "") or "")
        if provider in ("lmstudio", "llamacpp"):
            env.update({"OPENAI_COMPAT_URL": base_url, "OPENAI_COMPAT_MODEL": model,
                        "OPENAI_COMPAT_API_KEY": api_key})
        else:  # ollama
            env.update({"OLLAMA_URL": base_url, "OLLAMA_MODEL": model,
                        "OLLAMA_API_KEY": api_key})
        specs.append(AgentSpec(label=label, script=_VISION_PROVIDER_SCRIPTS[provider],
                               env=env, loop_sleep=10))
    return specs


def compute_desired_agents(topic: str) -> dict[str, AgentSpec]:
    """Build {label: AgentSpec} for everything that *should* be running right now.

    ``topic`` is retained for signature stability (the supervisor passes
    ``self.topic``); it is no longer consulted directly now that local folders
    are driven by LOCAL_IMPORTS_JSON rather than topic-keyword gating.
    """
    _ = topic  # reserved; see docstring
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
                      env={"CIVITAI_DOMAIN": domain}, loop_sleep=600))
    add(AgentSpec(label="Web", script="scraper_web.py", loop_sleep=1800))

    # Local-folder importers. The job (via job_config.resolve_env) projects every
    # ENABLED folder into LOCAL_IMPORTS_JSON as a JSON array of
    # {name, dir, migrate_from}. We fan that out into one Local-<name> agent per
    # folder, each carrying its own LOCAL_IMPORT_* env so the (unchanged)
    # feed_local_folder.py reads its folder from its own process env. (The legacy
    # single-folder LOCAL_IMPORT_* branch is gone — local folders are the one
    # mechanism now.)
    for folder in _local_import_folders():
        name = (folder.get("name") or "local").strip() or "local"
        add(AgentSpec(
            label=f"Local-{name}",
            script="feed_local_folder.py",
            loop_sleep=3600,
            env={
                "LOCAL_IMPORT_DIR": str(folder.get("dir", "") or ""),
                "LOCAL_IMPORT_NAME": name,
                "LOCAL_IMPORT_ENABLED": "true",
                "LOCAL_IMPORT_MIGRATE_FROM": str(folder.get("migrate_from", "") or ""),
            },
        ))

    # gallery-dl URL-based scraper (Pixiv, DeviantArt, booru sites, ArtStation,
    # Tumblr, Newgrounds, X, Reddit, Imgur, Flickr — anything gallery-dl knows).
    # Only desired when both the toggle is on AND at least one URL is configured;
    # otherwise the supervisor would respawn an empty agent every loop_sleep.
    if (
        os.environ.get("GALLERY_DL_ENABLED", "false").lower() == "true"
        and (os.environ.get("GALLERY_DL_URLS", "") or "").strip()
    ):
        add(AgentSpec(label="Gallery-DL", script="scraper_gallery_dl.py", loop_sleep=1800))

    # yt-dlp video scraper (YouTube, TikTok, X, Reddit, Vimeo, Twitch clips —
    # anything yt-dlp knows). Gated identically to gallery-dl: desired only when
    # the toggle is on AND at least one URL is configured, otherwise the
    # supervisor would respawn an empty agent every loop_sleep.
    if (
        os.environ.get("YT_DLP_ENABLED", "false").lower() == "true"
        and (os.environ.get("YT_DLP_URLS", "") or "").strip()
    ):
        add(AgentSpec(label="YT-DLP", script="scraper_yt_dlp.py", loop_sleep=1800))

    # Local vision-worker fleet (LM Studio / llama.cpp / Ollama) — one worker per
    # enabled instance in VISION_WORKERS_JSON, fanned out like local folders above.
    for spec in _vision_fleet_specs():
        add(spec)

    # Registry vision workers (cloud Groq + any legacy names still in
    # PIPELINE_VISION_WORKERS). The Vision-* labels are also filterable via
    # SCRAPER_DISABLED so admins can force everything off with one bulk call.
    for worker in vision_worker_list():
        spec = _vision_spec(worker)
        if spec is not None:
            add(spec)

    return agents


# ── Supervisor ────────────────────────────────────────────────────────────────

class Supervisor:
    """Reconciles desired agents with actually-running subprocesses."""

    def __init__(self, topic: str, base_env: dict[str, str], log_file,
                 job_slug: str | None = None) -> None:
        self.topic = topic
        self.base_env = base_env
        self.log_file = log_file
        # The job slug this supervisor is currently running. None in the legacy
        # CLI/topic path; set in the jobs path so we can detect when the
        # dashboard switches the active job and restart into the new one.
        self.job_slug = job_slug
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
        # mtime of data/jobs/_index.json — bumps when the dashboard activates /
        # advances a job. A change whose active slug differs from job_slug means
        # we must re-project the new job and structurally restart into it.
        self._jobs_index_mtime: float = self._current_jobs_index_mtime()
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

    @staticmethod
    def _current_jobs_index_mtime() -> float:
        """mtime of data/jobs/_index.json. Bumps when the dashboard activates or
        advances a job; we use it to detect active-slug switches cheaply (no
        JSON read on the hot poll path)."""
        try:
            return JOBS_INDEX_PATH.stat().st_mtime
        except OSError:
            return 0.0

    def _apply_active_job(self, slug: str) -> bool:
        """Project the job ``slug`` into the runtime env + taxonomy.

        Resolves the job's env over ``os.environ`` (so ``compute_desired_agents``
        and the next ``_spawn`` both see PIPELINE_TOPIC/SLUG, SCRAPER_DISABLED,
        scoring, captioning, …) and writes its categories into
        ``cull_categories.json``. Returns True on success. On a missing job file
        (orphaned active pointer) returns False so the caller can idle rather
        than spawn against stale config.

        We intentionally update ``os.environ`` rather than threading a separate
        dict through ``compute_desired_agents``: that function already reads the
        process env for every toggle (SCRAPER_DISABLED, GALLERY_DL_*, …), so
        overlaying the resolved job env is the smallest correct wiring and keeps
        the legacy CLI/topic path untouched.
        """
        job = job_config.get_job(slug)
        if job is None:
            return False
        resolved = job_config.resolve_env(job)
        os.environ.update(resolved)
        self.topic = resolved.get("PIPELINE_TOPIC", self.topic) or self.topic
        self.job_slug = slug
        # Project the new taxonomy BEFORE pre-creating folders so the new job's
        # category dirs exist for the about-to-respawn workers.
        job_config.project_categories(job)
        queue_dir, sorted_dir = _prepare_slug_dirs(slug)
        # Point spawn env + the stale-.processing sweep at the new job's dirs.
        self.base_env = {
            **self.base_env, **resolved,
            "PIPELINE_QUEUE": str(queue_dir),
            "PIPELINE_SORTED": str(sorted_dir),
        }
        self._queue_dir = queue_dir
        return True

    def _spawn(self, spec: AgentSpec) -> None:
        script_path = PIPELINE_CODE_DIR / spec.script
        # spec.env is merged OVER the base spawn env so per-agent overrides win
        # (civitai domain, vision keepalive flag, a local folder's LOCAL_IMPORT_*).
        run_env = {**self.base_env, **spec.env}
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

        # Jobs index edits (dashboard activated / advanced a job, or edited the
        # ACTIVE job's config and bumped the index — see design §8). We re-apply
        # the active job's projection in two cases:
        #   * the active slug CHANGED → force a structural restart into the new
        #     job's env + taxonomy (reuse the structural-restart path), or
        #   * the active slug is the SAME → re-resolve env + re-project so per-job
        #     edits (scraper toggles, scoring, captioning, categories) on the
        #     running job take effect. Whether that warrants a restart is then
        #     decided by the existing struct-snapshot / categories-mtime diffs
        #     below, so a mere SCRAPER_DISABLED change flows through the normal
        #     start/stop reconcile without a disruptive restart.
        idx_mtime = self._current_jobs_index_mtime()
        if idx_mtime != self._jobs_index_mtime:
            self._jobs_index_mtime = idx_mtime
            new_slug = desired_active_slug()
            if not new_slug and self.job_slug:
                # Active job cleared (dashboard stop / advance past end). Signal
                # the run loop to exit so the top-level jobs driver terminates
                # the children and returns to its idle wait. We do NOT auto-
                # advance — the dashboard owns the queue.
                self.job_slug = None
                self._stop.set()
                print("  [job-switch] active job cleared; stopping pipeline and idling",
                      flush=True)
                return
            if new_slug:
                slug_changed = new_slug != self.job_slug
                if self._apply_active_job(new_slug):
                    if slug_changed:
                        structural_changed = True
                        print(f"  [job-switch] active job -> {new_slug!r}; "
                              "restarting children with its env + categories", flush=True)
                    else:
                        print(f"  [job-reload] active job {new_slug!r} config changed; "
                              "re-projected env + categories", flush=True)
                    # Re-baseline the categories mtime so re-projecting (which
                    # rewrote cull_categories.json with identical-or-new content)
                    # doesn't double-trigger the categories watch below; the diff
                    # against the OLD struct snapshot still drives a restart when
                    # a structural key actually changed.
                    self._categories_mtime = self._current_categories_mtime()
                    new_struct = _structural_env_snapshot()
                    if not slug_changed and new_struct != self._struct_snapshot:
                        structural_changed = True
                        print("  [job-reload] structural key changed; soft-restarting children",
                              flush=True)
                    self._struct_snapshot = new_struct
                elif slug_changed:
                    print(f"  [job-switch] active slug {new_slug!r} has no job file; "
                          "ignoring", flush=True)

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

        # Optional per-job scheduler (gated SCHEDULER_ENABLED, default OFF). Runs
        # outside the agent lock since it only touches job_config state; fully
        # swallowed so a schedule failure can never stop the supervisor.
        _scheduler_tick()

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
                idx_bumped = self._current_jobs_index_mtime() != self._jobs_index_mtime
                if env_bumped or cats_bumped or idx_bumped or (now - last_full) >= RECONCILE_SECONDS:
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


# ── Slug directory setup (shared by the topic + jobs runners) ──────────────────

def _prepare_slug_dirs(slug: str) -> tuple[Path, Path]:
    """Create the queue + sorted (incl. category) folders for ``slug`` and return
    ``(queue_dir, sorted_dir)``. Reads the taxonomy live so a (re)start picks up
    the active job's categories; workers also mkdir lazily as a belt-and-braces."""
    # Sanitise the slug before it becomes a per-slug queue/sorted path component;
    # the charset barrier rejects '/', '\\' and '.' so no traversal is possible.
    slug = _validate_slug(slug)
    queue_root = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
    sorted_root = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted")))
    queue_dir = queue_root if queue_root.name == slug else queue_root / slug
    sorted_dir = sorted_root if sorted_root.name == slug else sorted_root / slug
    queue_dir.mkdir(parents=True, exist_ok=True)
    from categories import get_all_categories
    for cat in get_all_categories():
        (sorted_dir / cat).mkdir(parents=True, exist_ok=True)
    return queue_dir, sorted_dir


def _open_slug_log(slug: str):
    log_dir = Path(os.environ.get("LOG_DIR", str(BASE_DIR / "logs_test")))
    log_dir.mkdir(parents=True, exist_ok=True)
    return open(log_dir / f"pipeline_{slug}.log", "w", encoding="utf-8")


def _install_sigint(supervisor: "Supervisor") -> None:
    def _handle_sigint(_sig, _frame):
        supervisor.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_sigint)


# ── Topic runner (legacy CLI path) ─────────────────────────────────────────────

def run_topic(topic: str, vision_worker: str = "balanced-groq") -> None:
    slug = topic_slug(topic)
    print(f"\n{'=' * 60}", flush=True)
    print(f"=== TOPIC: {topic} (slug: {slug}) ===", flush=True)
    print(f"=== VISION WORKER LIST: {vision_worker_list() or [vision_worker]} ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    queue_dir, sorted_dir = _prepare_slug_dirs(slug)
    log_file = _open_slug_log(slug)

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
    _install_sigint(supervisor)

    try:
        supervisor.run()
    finally:
        log_file.close()


# ── Jobs runner (default path) ─────────────────────────────────────────────────

def run_active_job(slug: str, vision_worker: str = "balanced-groq") -> None:
    """Run the active job ``slug`` under the supervisor.

    Projects the job into env + taxonomy, sets up its slug dirs, then hands the
    Supervisor a ``job_slug`` so it watches ``_index.json`` and restarts itself
    into a different job when the dashboard switches the active pointer — no
    return to the idle loop needed for a job→job switch.
    """
    job = job_config.get_job(slug)
    if job is None:
        print(f"  [jobs] active slug {slug!r} has no job file; skipping", flush=True)
        return

    # Project the job down into the runtime contracts BEFORE we read env/taxonomy.
    resolved = job_config.resolve_env(job)
    os.environ.update(resolved)
    job_config.project_categories(job)

    topic = resolved.get("PIPELINE_TOPIC", "") or job.name
    print(f"\n{'=' * 60}", flush=True)
    print(f"=== JOB: {job.name} (slug: {slug}) ===", flush=True)
    print(f"=== TOPIC: {topic} ===", flush=True)
    print(f"=== VISION WORKER LIST: {vision_worker_list() or [vision_worker]} ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    queue_dir, sorted_dir = _prepare_slug_dirs(slug)
    log_file = _open_slug_log(slug)

    base_env = {
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "PIPELINE_QUEUE": str(queue_dir),
        "PIPELINE_SORTED": str(sorted_dir),
        **os.environ,  # already carries the resolved job env (PIPELINE_TOPIC/SLUG/…)
    }

    if not vision_worker_list() and vision_worker:
        os.environ["PIPELINE_VISION_WORKERS"] = vision_worker

    supervisor = Supervisor(topic=topic, base_env=base_env, log_file=log_file, job_slug=slug)
    supervisor._queue_dir = queue_dir
    _install_sigint(supervisor)

    try:
        supervisor.run()
    finally:
        log_file.close()


def run_jobs_loop(vision_worker: str = "balanced-groq") -> None:
    """Top-level driver for the jobs model.

    Idles gracefully while no job is active (watching ``_index.json`` for one to
    appear), then runs the active job. ``run_active_job`` handles job→job
    switches internally via the supervisor's index watch; this loop only regains
    control when the active job is *cleared* (stop / advance-past-end), at which
    point it idles again. The dashboard drives advance — we never auto-advance.

    TODO(scalability): jobs run STRICTLY SEQUENTIALLY today — one active
    supervisor instance drives one active job at a time (see CLAUDE.md Jobs
    model §"sequential queue"). Concurrent multi-job execution would need
    either (a) one Supervisor per active slug (with disjoint queue/sorted
    roots — already the case) sharing the .env + credentials pool, or (b) a
    shared worker fleet that fair-shares across slugs. Vision workers ARE
    already parallel within a single job (fleet fan-out + ThreadPoolExecutor);
    the sequential ceiling is the JOB dimension only.
    """
    announced_idle = False
    while True:
        slug = desired_active_slug()
        if not slug:
            if not announced_idle:
                print("  [jobs] no active job; waiting for the dashboard to "
                      "activate one...", flush=True)
                announced_idle = True
            try:
                time.sleep(IDLE_POLL_SECONDS)
            except KeyboardInterrupt:
                print("\nPipeline stopped by user.", flush=True)
                return
            continue
        announced_idle = False
        if job_config.get_job(slug) is None:
            # Orphaned active pointer — don't spin hot; wait for the dashboard to
            # fix it (advance() skips orphans, so this is a transient state).
            print(f"  [jobs] active slug {slug!r} has no job file; waiting...", flush=True)
            try:
                time.sleep(IDLE_POLL_SECONDS)
            except KeyboardInterrupt:
                return
            continue
        run_active_job(slug, vision_worker=vision_worker)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Job-driven image prompt pipeline")
    parser.add_argument("--topic", action="append", dest="topics",
                        help="Legacy: run a specific topic instead of the active job "
                             "(can be repeated). Bypasses the jobs model.")
    parser.add_argument("--vision-worker", dest="vision_worker",
                        default=os.environ.get("PIPELINE_VISION_WORKER", "balanced-groq"),
                        help="Default vision worker (used only if PIPELINE_VISION_WORKERS is empty)")
    args = parser.parse_args()

    # Idempotent — safe even though the dashboard may also call it on startup.
    try:
        job_config.migrate_env_to_default_job()
        job_config.migrate_legacy_vision_to_fleet()
    except Exception as exc:  # never let migration block the supervisor booting
        logger.warning("migrate_env_to_default_job failed: %s", exc)

    print("=== Pipeline Orchestrator ===", flush=True)
    print(f"Pipeline code dir: {PIPELINE_CODE_DIR}", flush=True)

    # Legacy escape hatch: an explicit --topic bypasses the jobs model entirely
    # (ad-hoc CLI runs). The default path runs the active job.
    if args.topics:
        print(f"Topics (legacy CLI mode): {args.topics}", flush=True)
        for topic in args.topics:
            run_topic(topic, vision_worker=args.vision_worker)
        print("\n=== All topics complete ===", flush=True)
        return

    print(f"Active job: {desired_active_slug() or '(none yet)'}", flush=True)
    run_jobs_loop(vision_worker=args.vision_worker)


if __name__ == "__main__":
    main()
