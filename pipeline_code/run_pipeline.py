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
from typing import Any

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
    # Kohya training-set feeder — a change to root/name/toggles/mode must restart
    # the feeder so the new dataset path takes effect. KOHYA_POLL_INTERVAL is
    # read live inside the feeder loop, so it's not structural.
    "KOHYA_IMPORT_ENABLED", "KOHYA_IMPORT_DIR", "KOHYA_IMPORT_NAME",
    "KOHYA_MOVE", "KOHYA_ALLOW_FLAT",
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

    In multi-active mode, ``slug`` names which per-slug base env this agent's
    spawn should use as its base. ``None`` (the default) means "use the
    supervisor's shared base env" — that's the shared vision fleet and every
    single-active spawn.
    """
    label: str
    script: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    loop_sleep: int = 300  # seconds between respawns if the child exits on its own
    slug: str | None = None


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


def _scraper_priority() -> dict:
    """Parse SCRAPER_PRIORITY_JSON into ``{"order": [...], "weights": {...}}``.

    Uses ``job_config.clean_scraper_priority`` so unknown names are dropped,
    missing weights get PRIORITY_WEIGHT_DEFAULT, and every PRIORITY_NAME is
    covered — the exact same shape the dashboard writes. Empty / malformed env
    → the default (PRIORITY_NAMES order, every weight = PRIORITY_WEIGHT_DEFAULT).
    """
    raw = (os.environ.get("SCRAPER_PRIORITY_JSON", "") or "").strip()
    if not raw:
        return job_config.clean_scraper_priority(None)
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return job_config.clean_scraper_priority(None)
    return job_config.clean_scraper_priority(data)


def _apply_priority(agents: dict[str, "AgentSpec"]) -> dict[str, "AgentSpec"]:
    """Re-order ``agents`` so the priority-list order wins at spawn time.

    Non-priority labels (Local-*, Kohya-*, Vision-*, dynamic Discord-N shards)
    keep their relative insertion order behind the priority-ordered head. This
    preserves the "top = fires first" contract without disturbing labels that
    the priority block doesn't cover.
    """
    if not agents:
        return agents
    priority = _scraper_priority()
    order = [n for n in priority.get("order", []) if n in agents]
    trailing = [name for name in agents if name not in order]
    return {name: agents[name] for name in (order + trailing)}


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
    active-slug change detection has one named seam to test/mock. Multi-active
    callers should use :func:`desired_active_slugs` instead.
    """
    return job_config.get_active_slug()


def desired_active_slugs() -> list[str]:
    """Every slug the supervisor should be running (may be empty).

    Multi-active reader — the supervisor iterates this to spawn per-slug
    scrapers and to compute the shared vision fleet's active-slug env.
    """
    return job_config.get_active_slugs()


def active_job_priorities() -> dict[str, int]:
    """The per-slug priority map (defaults filled in for absent entries).

    Returned map covers every currently-active slug — never missing, never
    empty when there is at least one active slug — so the shared vision
    fleet's round-robin schedule is deterministic.
    """
    priorities: dict[str, int] = {}
    for slug in desired_active_slugs():
        priorities[slug] = job_config.get_job_priority(slug)
    return priorities


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

    # Kohya-style training-set feeder. Gated identically to gallery-dl: only
    # desired when both the toggle is on AND a dataset root is configured, so an
    # empty config never respawns a broken agent every loop_sleep. The feeder
    # walks ``<repeats>_<concept>`` subdirs and (optionally) flat Danbooru-style
    # folders — see feed_kohya_folder.py.
    if (
        os.environ.get("KOHYA_IMPORT_ENABLED", "false").lower() == "true"
        and (os.environ.get("KOHYA_IMPORT_DIR", "") or "").strip()
    ):
        _kohya_name = (os.environ.get("KOHYA_IMPORT_NAME", "") or "kohya").strip() or "kohya"
        add(AgentSpec(
            label=f"Kohya-{_kohya_name}",
            script="feed_kohya_folder.py",
            loop_sleep=3600,
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

    # Per-job priority: reorder the agents dict so the top-priority scraper
    # spawns first. Non-priority labels (Local-*, Kohya-*, Vision-*, dynamic
    # Discord-N > 1) keep their insertion order behind the priority head — see
    # _apply_priority for the contract. Ordering is decided here so
    # queue_manager stays untouched.
    return _apply_priority(agents)


# ── Multi-active helpers (shared vision fleet + per-slug scrapers) ────────────
#
# In the multi-active model the supervisor spawns:
#   * one scraper subprocess per (active slug, scraper), labelled
#     f"{slug}::{scraper}", each with the SLUG's resolved env; and
#   * one shared vision-fleet subprocess per unique (provider, base_url, model)
#     across every active slug's vision.workers, with PIPELINE_ACTIVE_SLUGS_JSON
#     projected so queue_manager's shim pops weighted round-robin across the
#     active slugs' queues.
#
# The active-slug JSON blob is the ONE new env contract:
#   [[slug, weight], [slug, weight], …]
# — read by queue_manager._active_slugs_from_env().

def _scraper_agents_for_env(env: dict[str, str]) -> dict[str, AgentSpec]:
    """Return only the SCRAPER agents (no vision workers) driven by ``env``.

    Reuses :func:`compute_desired_agents` under a temporarily-overlaid
    ``os.environ`` and strips vision-worker labels off the result.
    ``compute_desired_agents`` was designed for a single active env; this
    helper lets us call it once per active slug with that slug's env.
    """
    saved = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update({k: str(v) for k, v in env.items() if v is not None})
        # Force the registry-vision path OFF for the per-slug call: the shared
        # fleet is built separately across all slugs' vision.workers.
        os.environ["PIPELINE_VISION_WORKERS"] = ""
        all_agents = compute_desired_agents(env.get("PIPELINE_TOPIC", "") or "")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return {label: spec for label, spec in all_agents.items()
            if not label.startswith("Vision-")}


def _shared_vision_fleet_specs(
    slug_envs: dict[str, dict[str, str]],
    priorities: dict[str, int],
) -> list[AgentSpec]:
    """Union the ``vision.workers`` from every active slug's env, deduped by
    ``(provider, base_url, model)``, and return one ``AgentSpec`` per unique
    endpoint carrying the shared multi-active projection env.

    Each spec's ``env`` includes ``PIPELINE_ACTIVE_SLUGS_JSON`` so the worker's
    queue-poll shim fair-shares across every active slug's queue by weight.
    The Vision label matches the historic ``Vision-<name>`` scheme so the
    supervisor's start/stop reconcile treats them like any other agent.
    """
    seen: set[tuple[str, str, str]] = set()
    labels_used: set[str] = set()
    specs: list[AgentSpec] = []
    active_json = json.dumps([[slug, int(priorities.get(slug, 1) or 1)]
                              for slug in slug_envs.keys()])
    for slug, env in slug_envs.items():
        raw = (env.get("VISION_WORKERS_JSON", "") or "").strip()
        if not raw:
            continue
        try:
            fleet = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(fleet, list):
            continue
        for i, w in enumerate(fleet):
            if not isinstance(w, dict):
                continue
            provider = str(w.get("provider", "") or "").strip().lower()
            base_url = str(w.get("base_url", "") or "").strip()
            model = str(w.get("model", "") or "").strip()
            if provider not in _VISION_PROVIDER_SCRIPTS or not base_url:
                continue
            key = (provider, base_url, model)
            if key in seen:
                continue
            seen.add(key)
            name = (str(w.get("name", "") or w.get("id", "") or f"w{i}")).strip() or f"w{i}"
            label = base = f"Vision-{name}"
            n = 2
            while label in labels_used:
                label, n = f"{base}-{n}", n + 1
            labels_used.add(label)
            worker_env = {
                "VISION_INSTANCE_ID": str(w.get("id", "") or name),
                "VISION_INSTANCE_NAME": name,
                # Multi-active projection: the queue-shim reads this to
                # weighted-round-robin across active slugs' queues; the
                # vision-worker base reads it to route sorted outputs by
                # per-image slug rather than the (undefined) PIPELINE_SLUG.
                "PIPELINE_ACTIVE_SLUGS_JSON": active_json,
            }
            api_key = str(w.get("api_key", "") or "")
            if provider in ("lmstudio", "llamacpp"):
                worker_env.update({"OPENAI_COMPAT_URL": base_url,
                                   "OPENAI_COMPAT_MODEL": model,
                                   "OPENAI_COMPAT_API_KEY": api_key})
            else:  # ollama
                worker_env.update({"OLLAMA_URL": base_url,
                                   "OLLAMA_MODEL": model,
                                   "OLLAMA_API_KEY": api_key})
            specs.append(AgentSpec(
                label=label,
                script=_VISION_PROVIDER_SCRIPTS[provider],
                env=worker_env,
                loop_sleep=10,
                slug=None,  # shared — spawned against the supervisor base env
            ))
    return specs


def compute_desired_agents_multi(
    slug_envs: dict[str, dict[str, str]],
    priorities: dict[str, int],
) -> dict[str, AgentSpec]:
    """Combined desired-agent set across every active slug.

    Layout of the returned dict:
      * per-slug scrapers labelled ``"{slug}::{label}"`` (Local/Kohya/gallery-
        dl/discord/etc. shards included), each tagged with ``spec.slug`` so
        the supervisor spawns them under that slug's resolved env; then
      * shared vision-fleet workers labelled ``Vision-<name>`` (deduped by
        provider + base_url + model), each carrying PIPELINE_ACTIVE_SLUGS_JSON
        for weighted-round-robin queue polling and per-image sorted-dir
        routing.

    Weights beyond the vision fleet's round-robin are TODO: for v1 the per-job
    priority weight also LOGS an intended scraper-spawn intensity multiplier;
    each scraper is its own subprocess whose per-source poll cadence is
    unchanged. Real per-slug scraper throttling can attach here once the
    vision fleet stops being the bottleneck.
    """
    combined: dict[str, AgentSpec] = {}
    for slug, env in slug_envs.items():
        scrapers = _scraper_agents_for_env(env)
        for label, spec in scrapers.items():
            prefixed = f"{slug}::{label}"
            combined[prefixed] = AgentSpec(
                label=prefixed, script=spec.script, args=list(spec.args),
                env=dict(spec.env), loop_sleep=spec.loop_sleep, slug=slug,
            )
    for spec in _shared_vision_fleet_specs(slug_envs, priorities):
        combined[spec.label] = spec
    return combined


# ── Supervisor ────────────────────────────────────────────────────────────────

class Supervisor:
    """Reconciles desired agents with actually-running subprocesses."""

    def __init__(self, topic: str, base_env: dict[str, str], log_file,
                 job_slug: str | None = None,
                 job_slugs: list[str] | None = None) -> None:
        self.topic = topic
        self.base_env = base_env
        self.log_file = log_file
        # The job slug this supervisor is currently running. None in the legacy
        # CLI/topic path; set in the jobs path so we can detect when the
        # dashboard switches the active job and restart into the new one.
        # In multi-active mode ``job_slugs`` holds the full list; ``job_slug``
        # is the head for legacy readers. ``_slug_envs`` maps every active
        # slug to its resolved spawn env (used by ``_spawn`` for slug-tagged
        # agents; the shared vision fleet uses ``self.base_env``).
        self.job_slug = job_slug
        self.job_slugs: list[str] = list(job_slugs) if job_slugs is not None \
            else ([job_slug] if job_slug else [])
        self._slug_envs: dict[str, dict[str, str]] = {}
        self._slug_queue_dirs: dict[str, Path] = {}
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

    def _compute_desired_agents(self) -> dict[str, AgentSpec]:
        """Route to the multi-active builder when >1 slug is active.

        Single-slug (or legacy CLI/topic) supervisor runs keep hitting the
        v1 ``compute_desired_agents`` so nothing changes there. As soon as
        two or more jobs are active, ``compute_desired_agents_multi`` takes
        over: per-slug scrapers with per-slug env, plus a shared vision
        fleet fanned out across every slug's ``vision.workers`` (deduped by
        provider+base_url+model) carrying PIPELINE_ACTIVE_SLUGS_JSON.
        """
        if len(self.job_slugs) > 1 and self._slug_envs:
            priorities = {slug: max(1, int(env.get("_JOB_PRIORITY", "1") or 1))
                          for slug, env in self._slug_envs.items()}
            # Fall back to the live index priorities so tests / callers that
            # don't populate the env sidecar still get weighted round-robin.
            for slug in self.job_slugs:
                if priorities.get(slug, 1) <= 1:
                    priorities[slug] = job_config.get_job_priority(slug)
            return compute_desired_agents_multi(self._slug_envs, priorities)
        return compute_desired_agents(self.topic)

    def _apply_active_slugs(self, slugs: list[str]) -> list[str]:
        """Project EVERY active slug's env into ``self._slug_envs`` for spawn.

        Returns the list of slugs that were successfully applied (missing
        job files are skipped and logged). The supervisor's shared base_env
        is updated with a ``PIPELINE_ACTIVE_SLUGS_JSON`` blob so any child
        that reads the process env (queue_manager shim, shared vision fleet)
        sees the multi-active projection without needing per-agent env.
        """
        applied: list[str] = []
        self._slug_envs.clear()
        self._slug_queue_dirs.clear()
        priorities: list[list[Any]] = []
        for slug in slugs:
            job = job_config.get_job(slug)
            if job is None:
                logger.warning("multi-active: skipping unknown slug %r", slug)
                continue
            resolved = job_config.resolve_env(job)
            queue_dir, sorted_dir = _prepare_slug_dirs(slug)
            job_config.project_categories(job)
            per_slug_env = {
                **os.environ,
                **resolved,
                "PIPELINE_QUEUE": str(queue_dir),
                "PIPELINE_SORTED": str(sorted_dir),
            }
            self._slug_envs[slug] = per_slug_env
            self._slug_queue_dirs[slug] = queue_dir
            weight = job_config.get_job_priority(slug)
            priorities.append([slug, weight])
            applied.append(slug)
            # TODO(scalability): the per-job priority weight also implies
            # scraper spawn intensity ("heavier weight → more turns per
            # cycle"). For v1 we only wire it into the vision-fleet
            # round-robin (via PIPELINE_ACTIVE_SLUGS_JSON) — the scraper
            # spawn cadence remains per-source. Extend queue_manager /
            # supervisor here to fair-share scraper starts by weight once
            # the vision fleet stops being the bottleneck.
            logger.info("multi-active: slug=%s weight=%d", slug, weight)
        # Point spawn env + stale-.processing sweep at the active-set:
        # PIPELINE_ACTIVE_SLUGS_JSON lets the shared vision fleet poll
        # queues across every active slug via queue_manager's shim.
        self.base_env = {
            **self.base_env,
            "PIPELINE_ACTIVE_SLUGS_JSON": json.dumps(priorities),
        }
        # Head-of-list stays the "primary" for legacy readers / the stale-
        # .processing sweep (which only walks one queue dir).
        self.job_slugs = applied
        self.job_slug = applied[0] if applied else None
        if applied:
            self._queue_dir = self._slug_queue_dirs.get(applied[0])
        return applied

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

    def _base_env_for(self, spec: AgentSpec) -> dict[str, str]:
        """Pick the base spawn env for ``spec``.

        Multi-active: slug-tagged agents (per-slug scrapers) inherit their
        own slug's resolved env from ``self._slug_envs`` so each scraper
        sees the right PIPELINE_SLUG / PIPELINE_QUEUE / SCRAPER_DISABLED /
        etc. The shared vision fleet has ``spec.slug is None`` and uses the
        supervisor's ``self.base_env`` (which carries only global keys —
        credentials, model endpoints — plus the PIPELINE_ACTIVE_SLUGS_JSON
        the fleet-spec env layer sets per worker).
        """
        if spec.slug and spec.slug in self._slug_envs:
            return self._slug_envs[spec.slug]
        return self.base_env

    def _spawn(self, spec: AgentSpec) -> None:
        script_path = PIPELINE_CODE_DIR / spec.script
        # spec.env is merged OVER the base spawn env so per-agent overrides win
        # (civitai domain, vision keepalive flag, a local folder's LOCAL_IMPORT_*).
        run_env = {**self._base_env_for(spec), **spec.env}
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
            new_slugs = desired_active_slugs()
            if not new_slugs and self.job_slugs:
                # Every active job cleared — stop the pipeline and idle.
                self.job_slugs = []
                self.job_slug = None
                self._stop.set()
                print("  [job-switch] active jobs cleared; stopping pipeline and idling",
                      flush=True)
                return
            if new_slugs:
                slugs_changed = new_slugs != self.job_slugs
                if len(new_slugs) > 1 or slugs_changed:
                    # Multi-active path OR active-set membership changed → apply
                    # the full active set and mark a structural restart if the
                    # membership shifted (so children pick up the new spawn env).
                    if self._apply_active_slugs(new_slugs):
                        if slugs_changed:
                            structural_changed = True
                            print(f"  [job-switch] active set -> {new_slugs}; "
                                  "restarting children with per-slug env + shared vision fleet",
                                  flush=True)
                        else:
                            print(f"  [job-reload] active set {new_slugs} config changed; "
                                  "re-projected env + categories", flush=True)
                        self._categories_mtime = self._current_categories_mtime()
                        new_struct = _structural_env_snapshot()
                        if not slugs_changed and new_struct != self._struct_snapshot:
                            structural_changed = True
                            print("  [job-reload] structural key changed; soft-restarting children",
                                  flush=True)
                        self._struct_snapshot = new_struct
                else:
                    # Single-active path: preserve historic single-slug wiring
                    # (the pre-existing _apply_active_job overlays os.environ so
                    # compute_desired_agents sees the same view as v1 today).
                    only = new_slugs[0]
                    if self._apply_active_job(only):
                        self.job_slugs = [only]
                        print(f"  [job-reload] active job {only!r} config changed; "
                              "re-projected env + categories", flush=True)
                        self._categories_mtime = self._current_categories_mtime()
                        new_struct = _structural_env_snapshot()
                        if new_struct != self._struct_snapshot:
                            structural_changed = True
                            print("  [job-reload] structural key changed; soft-restarting children",
                                  flush=True)
                        self._struct_snapshot = new_struct
                    else:
                        print(f"  [job-switch] active slug {only!r} has no job file; "
                              "ignoring", flush=True)

        desired = self._compute_desired_agents()
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
                if until:
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
    """Run a single active job ``slug`` under the supervisor (legacy shim).

    Kept for callers that still ask to run one specific slug. The dashboard-
    driven multi-active loop is :func:`run_active_jobs`.
    """
    run_active_jobs([slug], vision_worker=vision_worker)


def run_active_jobs(slugs: list[str], vision_worker: str = "balanced-groq") -> None:
    """Run one or more active jobs under a single Supervisor.

    Single-slug case (``len(slugs) == 1``): projects that slug's env into
    ``os.environ`` and reuses the historic single-active spawn path — the
    Supervisor keeps its ``compute_desired_agents`` call unchanged.

    Multi-slug case: each slug's env is prepared and held per-slug in the
    Supervisor's ``_slug_envs`` map. The Supervisor spawns per-slug scrapers
    (labelled ``"{slug}::{label}"``) plus a SHARED vision fleet (union across
    all slugs' vision.workers, deduped by provider+base_url+model), the
    fleet carrying PIPELINE_ACTIVE_SLUGS_JSON so its queue-poll shim fair-
    shares across every active slug.
    """
    resolved_slugs = [s for s in slugs if job_config.get_job(s) is not None]
    if not resolved_slugs:
        print(f"  [jobs] no runnable slugs in {slugs!r}; skipping", flush=True)
        return

    head = resolved_slugs[0]
    head_job = job_config.get_job(head)

    # For the SINGLE-slug case we keep the historic wiring (env projected into
    # os.environ) so compute_desired_agents' env-reading path is unchanged.
    single = len(resolved_slugs) == 1
    if single:
        resolved = job_config.resolve_env(head_job)
        os.environ.update(resolved)
        job_config.project_categories(head_job)
        topic = resolved.get("PIPELINE_TOPIC", "") or head_job.name
    else:
        topic = head_job.name
        # Multi-active: DO NOT overlay any one slug's env on os.environ — each
        # slug's env is passed per-agent via _slug_envs. Still project the
        # HEAD job's taxonomy so cull_categories.json isn't empty (the shared
        # vision fleet reads schema from that file). Subsequent slugs' per-
        # image sorted routing uses the queue-slug path derivation.
        job_config.project_categories(head_job)

    print(f"\n{'=' * 60}", flush=True)
    if single:
        print(f"=== JOB: {head_job.name} (slug: {head}) ===", flush=True)
    else:
        print(f"=== ACTIVE JOBS ({len(resolved_slugs)}): "
              f"{', '.join(resolved_slugs)} ===", flush=True)
    print(f"=== TOPIC: {topic} ===", flush=True)
    print(f"=== VISION WORKER LIST: {vision_worker_list() or [vision_worker]} ===",
          flush=True)
    print(f"{'=' * 60}\n", flush=True)

    queue_dir, sorted_dir = _prepare_slug_dirs(head)
    log_file = _open_slug_log(head)

    base_env = {
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "PIPELINE_QUEUE": str(queue_dir),
        "PIPELINE_SORTED": str(sorted_dir),
        **os.environ,
    }

    if not vision_worker_list() and vision_worker:
        os.environ["PIPELINE_VISION_WORKERS"] = vision_worker

    supervisor = Supervisor(
        topic=topic, base_env=base_env, log_file=log_file,
        job_slug=head, job_slugs=resolved_slugs,
    )
    supervisor._queue_dir = queue_dir
    if not single:
        # Prime per-slug env + PIPELINE_ACTIVE_SLUGS_JSON before the first
        # reconcile so the initial spawn is multi-active-aware.
        supervisor._apply_active_slugs(resolved_slugs)
    _install_sigint(supervisor)

    try:
        supervisor.run()
    finally:
        log_file.close()


def run_jobs_loop(vision_worker: str = "balanced-groq") -> None:
    """Top-level driver for the jobs model — multi-active aware.

    Idles gracefully while no job is active (watching ``_index.json`` for one
    to appear), then hands the whole active set to :func:`run_active_jobs`.
    Job-set changes are handled inside the supervisor's index watch (see
    :meth:`Supervisor.reconcile`); this loop only regains control when the
    active set is *cleared* (stop / advance-past-end), at which point it
    idles again. The dashboard drives activate/deactivate/advance — this
    loop never auto-advances.
    """
    announced_idle = False
    while True:
        slugs = desired_active_slugs()
        if not slugs:
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
        runnable = [s for s in slugs if job_config.get_job(s) is not None]
        if not runnable:
            print(f"  [jobs] active slugs {slugs!r} have no job files; waiting...",
                  flush=True)
            try:
                time.sleep(IDLE_POLL_SECONDS)
            except KeyboardInterrupt:
                return
            continue
        run_active_jobs(runnable, vision_worker=vision_worker)


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
