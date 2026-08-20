"""Tests for the supervisor's jobs-model integration (Phase B1).

Runnable two ways:
    pytest tests/test_supervisor_jobs.py
    python tests/test_supervisor_jobs.py        # falls back to pytest.main

Scope: the PURE/extractable wiring in ``run_pipeline`` that the supervisor's
big loop calls — NOT live subprocesses. Two seams are unit-tested:

  * ``active_job_env(slug=None)`` — the active job's resolved env overlaid on
    ``os.environ`` (or ``None`` when there is no runnable active job), and
  * ``desired_active_slug()`` + the "active slug changed since last snapshot"
    comparison the reconcile loop uses to decide whether to restart.

Fixture mirrors ``tests/test_job_config.py``: point all storage at a temp dir,
redirect + reset the categories cache, and reload the modules so their
import-time path constants land in the temp dir.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot/restore os.environ around every test in this file.

    Importing ``run_pipeline`` (done lazily inside ``isolated``) runs its
    module-level ``load_dotenv()``, which mutates the real ``os.environ`` (e.g.
    injects ``PIPELINE_SLUG`` from the repo ``.env``) — and monkeypatch can't
    undo a direct ``os.environ`` mutation it didn't make. As an autouse fixture
    this runs (and snapshots) BEFORE ``isolated`` triggers that import, so the
    finally-block restores a clean environment and this file never leaks state
    into sibling suites (notably ``test_job_config.py``, which assumes a clean
    ``PIPELINE_SLUG``)."""
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point all storage at a temp dir and hand back (run_pipeline, job_config, tmp).

    The two helpers under test (``desired_active_slug`` / ``active_job_env``) read
    ``job_config`` live, which resolves ``paths`` from the env on every call — so
    we pin the path env vars rather than reloading ``run_pipeline``. Reloading it
    would re-run its module-level ``load_dotenv()`` and leak the real ``.env``
    ``PIPELINE_QUEUE`` / ``PIPELINE_SORTED`` into the process env, polluting other
    tests. We therefore pin those keys to the temp dir here so the suite stays
    hermetic regardless of import order.
    """
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.delenv("PIPELINE_SLUG", raising=False)
    # categories.ACTIVE_PATH is import-time-computed from PIPELINE_BASE_DIR;
    # redirect it (and bust its cache) so projection lands in the temp dir.
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)

    import importlib
    import job_config
    importlib.reload(job_config)
    import run_pipeline  # imported once; pure helpers don't depend on its constants
    # The repo .env ships PIPELINE_CODE_DIR="" → Path(".") at import, so the
    # add() existence check in compute_desired_agents looks in cwd (repo root)
    # and finds no scripts. In the real pipeline the launcher runs with
    # cwd=pipeline_code; here we pin the constant to the actual source dir so
    # script-existence checks resolve as they do in production.
    monkeypatch.setattr(run_pipeline, "PIPELINE_CODE_DIR", PIPELINE_CODE)
    return run_pipeline, job_config, tmp_path


# ── desired_active_slug ───────────────────────────────────────────────────────

def test_desired_active_slug_none_when_no_job(isolated):
    run_pipeline, _job_config, _ = isolated
    assert run_pipeline.desired_active_slug() is None


def test_desired_active_slug_reflects_activation(isolated):
    run_pipeline, job_config, _ = isolated
    job_config.create_job("Job A")
    job_config.set_active("job_a")
    assert run_pipeline.desired_active_slug() == "job_a"


# ── active_job_env: graceful idle ─────────────────────────────────────────────

def test_active_job_env_none_when_no_active_job(isolated):
    """No active job → None so the supervisor idles instead of spawning against
    a stale global config."""
    run_pipeline, _job_config, _ = isolated
    assert run_pipeline.active_job_env() is None


def test_active_job_env_none_when_active_slug_orphaned(isolated, monkeypatch):
    """A dangling active pointer (slug with no job file) must yield None, not a
    half-populated env. Forced via the desired_active_slug seam."""
    run_pipeline, _job_config, _ = isolated
    monkeypatch.setattr(run_pipeline, "desired_active_slug", lambda: "ghost")
    assert run_pipeline.active_job_env() is None


# ── active_job_env: projection over os.environ ────────────────────────────────

def test_active_job_env_overlays_resolved_over_environ(isolated, monkeypatch):
    """The job's resolved env must win over a stale global value, while global
    keys the job does NOT define are inherited from os.environ unchanged."""
    run_pipeline, job_config, _ = isolated
    # A stale global topic + a global-only credential that jobs never store.
    monkeypatch.setenv("PIPELINE_TOPIC", "STALE GLOBAL TOPIC")
    monkeypatch.setenv("GROQ_API_KEY", "secret-key-123")

    # v2: PIPELINE_TOPIC comes from the job's subject.
    job = job_config.create_job("Fresh Job", subject="Fresh Job Topic")
    job_config.set_active("fresh_job")

    env = run_pipeline.active_job_env()
    assert env is not None
    # Job overrides the stale global topic...
    assert env["PIPELINE_TOPIC"] == "Fresh Job Topic"
    assert env["PIPELINE_SLUG"] == "fresh_job"
    # ...but global-only credentials are inherited.
    assert env["GROQ_API_KEY"] == "secret-key-123"


def test_active_job_env_matches_resolve_env_for_job_keys(isolated):
    """Every key resolve_env(job) emits must appear verbatim in active_job_env."""
    run_pipeline, job_config, _ = isolated
    job = job_config.create_job("Mapped")
    # v2: edit via sparse overrides (effective = preset ⊕ overrides).
    job = job_config.set_override(job, "scrapers.enabled", {"X.com": False, "Web": True})
    job = job_config.set_override(job, "scoring", {"ovr_min": 70, "rel_min": 60, "notes": "n"})
    job_config.save_job(job)
    job_config.set_active("mapped")

    resolved = job_config.resolve_env(job_config.get_job("mapped"))
    env = run_pipeline.active_job_env()
    assert env is not None
    for key, value in resolved.items():
        assert env[key] == value
    # Spot-check the projected scraper + scoring keys specifically.
    assert env["SCRAPER_DISABLED"] == "X.com"
    assert env["VISION_OVR_MIN_SCORE"] == "70"


def test_active_job_env_explicit_slug_overrides_active(isolated):
    """Passing slug= bypasses the active pointer (used by the supervisor when it
    re-resolves a just-switched-to job)."""
    run_pipeline, job_config, _ = isolated
    job_config.create_job("One")
    job_config.create_job("Two", subject="Two Topic")  # v2: topic via subject
    job_config.set_active("one")  # active is 'one'...

    env = run_pipeline.active_job_env("two")  # ...but we ask for 'two'
    assert env is not None
    assert env["PIPELINE_SLUG"] == "two"
    assert env["PIPELINE_TOPIC"] == "Two Topic"


def test_active_job_env_all_values_are_strings(isolated):
    run_pipeline, job_config, _ = isolated
    job_config.create_job("Strs")
    job_config.set_active("strs")
    env = run_pipeline.active_job_env()
    assert env is not None
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())


# ── active-slug-changed detection (the reconcile decision) ────────────────────

def test_active_slug_change_detection(isolated):
    """Reproduce the supervisor's decision: hold a snapshot slug, read the
    current desired slug, restart iff it differs."""
    run_pipeline, job_config, _ = isolated
    job_config.create_job("First")
    job_config.create_job("Second")

    job_config.set_active("first")
    held = run_pipeline.desired_active_slug()
    assert held == "first"

    # No change yet → no restart.
    assert run_pipeline.desired_active_slug() == held

    # Dashboard activates a different job → slug differs → restart.
    job_config.set_active("second")
    current = run_pipeline.desired_active_slug()
    assert current != held
    assert current == "second"


def test_active_slug_cleared_detection(isolated):
    """Clearing the active job (stop / advance-past-end) is observable as a slug
    going from a value to None — the supervisor uses this to idle."""
    run_pipeline, job_config, _ = isolated
    job_config.create_job("Only")
    job_config.set_active("only")
    assert run_pipeline.desired_active_slug() == "only"

    job_config.set_active(None)
    assert run_pipeline.desired_active_slug() is None


def test_advance_changes_desired_slug(isolated):
    """v2 multi-active: advance ADDS the queue head to the active set — the
    head stays the same, but ``desired_active_slugs`` grows."""
    run_pipeline, job_config, _ = isolated
    job_config.create_job("Head")
    job_config.create_job("Next")
    job_config.set_active("head")
    job_config.enqueue("next")

    assert run_pipeline.desired_active_slug() == "head"
    promoted = job_config.advance()
    assert promoted == "next"
    # Head unchanged; active set now includes both.
    assert run_pipeline.desired_active_slug() == "head"
    assert set(run_pipeline.desired_active_slugs()) == {"head", "next"}


# ── migration is invoked safely (idempotent, env-derived) ─────────────────────

def test_migrate_then_active_job_env_roundtrip(isolated, monkeypatch):
    """End-to-end of the startup path: migrate the legacy .env into a default
    job, then the supervisor's active_job_env reflects it."""
    run_pipeline, job_config, _ = isolated
    monkeypatch.setenv("PIPELINE_TOPIC", "Legacy Topic")
    monkeypatch.setenv("SCRAPER_DISABLED", "X.com")

    migrated = job_config.migrate_env_to_default_job()
    assert migrated is not None
    # Idempotent second call is a no-op (mirrors supervisor + dashboard both calling it).
    assert job_config.migrate_env_to_default_job() is None

    env = run_pipeline.active_job_env()
    assert env is not None
    assert env["PIPELINE_TOPIC"] == "Legacy Topic"
    assert env["SCRAPER_DISABLED"] == "X.com"


# ── AgentSpec.env merged at spawn ─────────────────────────────────────────────

class _FakeProc:
    """Minimal Popen stand-in: never exited, empty stdout, no real process."""
    pid = 4321
    returncode = None

    def __init__(self, *a, **kw):
        import io
        self.stdout = io.BytesIO(b"")

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def _spawn_capturing_env(run_pipeline, monkeypatch, spec, base_env):
    """Spawn ``spec`` through a real Supervisor with Popen mocked, returning the
    ``env`` dict the child WOULD have been launched with."""
    captured: dict = {}

    def _fake_popen(args, **kwargs):
        captured.update(kwargs.get("env") or {})
        return _FakeProc()

    monkeypatch.setattr(run_pipeline.subprocess, "Popen", _fake_popen)
    sup = run_pipeline.Supervisor(topic="t", base_env=base_env, log_file=None)
    sup._spawn(spec)
    return captured


def test_agentspec_env_merged_over_base_at_spawn(isolated, monkeypatch):
    """spec.env must override the base spawn env for that agent, while base keys
    the spec doesn't set are still inherited."""
    run_pipeline, _job_config, _ = isolated
    base_env = {"PIPELINE_SLUG": "base", "SHARED": "from-base"}
    spec = run_pipeline.AgentSpec(
        label="Local-photos",
        script="feed_local_folder.py",
        env={"LOCAL_IMPORT_DIR": "/imgs", "PIPELINE_SLUG": "overridden"},
    )
    env = _spawn_capturing_env(run_pipeline, monkeypatch, spec, base_env)
    assert env["LOCAL_IMPORT_DIR"] == "/imgs"     # per-agent value present
    assert env["PIPELINE_SLUG"] == "overridden"   # spec.env wins over base
    assert env["SHARED"] == "from-base"           # base key inherited


def test_agentspec_env_defaults_empty(isolated):
    """The env field is optional and defaults to an empty dict."""
    run_pipeline, _job_config, _ = isolated
    spec = run_pipeline.AgentSpec(label="X.com", script="scraper_x.py")
    assert spec.env == {}


# ── compute_desired_agents: local-folder fan-out from LOCAL_IMPORTS_JSON ───────

def _local_labels(agents: dict) -> set[str]:
    return {label for label in agents if label.startswith("Local-")}


def test_compute_desired_fans_out_one_agent_per_enabled_folder(isolated, monkeypatch):
    """Each enabled folder in LOCAL_IMPORTS_JSON → one Local-<name> agent with its
    own per-agent LOCAL_IMPORT_* env."""
    run_pipeline, _job_config, _ = isolated
    monkeypatch.setenv("PIPELINE_VISION_WORKERS", "")  # keep noise out
    monkeypatch.setenv("LOCAL_IMPORTS_JSON", json.dumps([
        {"name": "photos", "dir": "/data/photos", "migrate_from": "old_photos"},
        {"name": "renders", "dir": "/data/renders", "migrate_from": ""},
    ]))
    agents = run_pipeline.compute_desired_agents("anything")
    assert _local_labels(agents) == {"Local-photos", "Local-renders"}

    photos = agents["Local-photos"]
    assert photos.script == "feed_local_folder.py"
    assert photos.env["LOCAL_IMPORT_DIR"] == "/data/photos"
    assert photos.env["LOCAL_IMPORT_NAME"] == "photos"
    assert photos.env["LOCAL_IMPORT_ENABLED"] == "true"
    assert photos.env["LOCAL_IMPORT_MIGRATE_FROM"] == "old_photos"

    renders = agents["Local-renders"]
    assert renders.env["LOCAL_IMPORT_DIR"] == "/data/renders"
    assert renders.env["LOCAL_IMPORT_MIGRATE_FROM"] == ""


def test_compute_desired_no_local_agents_when_blob_empty(isolated, monkeypatch):
    """Empty / missing / malformed LOCAL_IMPORTS_JSON → no Local-* agents, no crash."""
    run_pipeline, _job_config, _ = isolated
    for blob in ("", "[]", "not json", '{"not": "a list"}'):
        monkeypatch.setenv("LOCAL_IMPORTS_JSON", blob)
        agents = run_pipeline.compute_desired_agents("topic")
        assert _local_labels(agents) == set(), f"blob={blob!r} produced Local agents"


def test_compute_desired_skips_malformed_folder_entries(isolated, monkeypatch):
    """Non-dict entries and entries with no dir are skipped defensively."""
    run_pipeline, _job_config, _ = isolated
    monkeypatch.setenv("LOCAL_IMPORTS_JSON", json.dumps([
        {"name": "good", "dir": "/ok", "migrate_from": ""},
        {"name": "nodir", "dir": "", "migrate_from": ""},  # no path → skip
        "not-a-dict",                                       # wrong type → skip
        {"name": "alsogood", "dir": "/ok2"},
    ]))
    agents = run_pipeline.compute_desired_agents("topic")
    assert _local_labels(agents) == {"Local-good", "Local-alsogood"}


# ── Local-folder sources only spawn via LOCAL_IMPORTS_JSON ────────────────────

def test_no_local_agent_when_no_local_imports(isolated, monkeypatch):
    """With an empty LOCAL_IMPORTS_JSON, no local-folder feeder agent is spawned,
    even when the topic matches the old human-keyword gate."""
    run_pipeline, _job_config, _ = isolated
    monkeypatch.setenv("LOCAL_IMPORTS_JSON", "[]")
    agents = run_pipeline.compute_desired_agents("realistic female influencer")
    assert not any(spec.script == "feed_local_folder.py" for spec in agents.values())


def test_scraper_names_has_no_local_folder_entry(isolated):
    """The shared scraper-name source of truth has no local-folder entry."""
    _run_pipeline, job_config, _ = isolated
    assert all("local" not in n.lower() for n in job_config.SCRAPER_NAMES)


# ── compute_desired_agents: vision fleet fan-out from VISION_WORKERS_JSON ──────

def _vision_labels(agents: dict) -> set[str]:
    return {label for label in agents if label.startswith("Vision-")}


def test_compute_desired_fans_out_one_vision_worker_per_instance(isolated, monkeypatch):
    """Each enabled instance in VISION_WORKERS_JSON → one Vision-<name> agent with
    its own per-agent endpoint env (provider-correct script + URL/model/key)."""
    run_pipeline, _jc, _ = isolated
    monkeypatch.setenv("PIPELINE_VISION_WORKERS", "")   # keep registry noise out
    monkeypatch.setenv("VISION_WORKERS_JSON", json.dumps([
        {"id": "a", "name": "LM A", "provider": "lmstudio",
         "base_url": "http://h:1234", "model": "m1", "api_key": ""},
        {"id": "b", "name": "Olla", "provider": "ollama",
         "base_url": "http://h:11434", "model": "", "api_key": "k"},
    ]))
    agents = run_pipeline.compute_desired_agents("anything")
    assert _vision_labels(agents) == {"Vision-LM A", "Vision-Olla"}

    a = agents["Vision-LM A"]
    assert a.script == "vision_worker_balanced_openai.py"
    assert a.env["OPENAI_COMPAT_URL"] == "http://h:1234"
    assert a.env["OPENAI_COMPAT_MODEL"] == "m1"
    assert a.env["VISION_INSTANCE_NAME"] == "LM A"

    b = agents["Vision-Olla"]
    assert b.script == "vision_worker_balanced_ollama.py"
    assert b.env["OLLAMA_URL"] == "http://h:11434"
    assert b.env["OLLAMA_API_KEY"] == "k"


def test_compute_desired_llamacpp_uses_openai_script(isolated, monkeypatch):
    """llama.cpp is OpenAI-compatible → routed to the openai-compat worker."""
    run_pipeline, _jc, _ = isolated
    monkeypatch.setenv("PIPELINE_VISION_WORKERS", "")
    monkeypatch.setenv("VISION_WORKERS_JSON", json.dumps([
        {"id": "c", "name": "vLLM", "provider": "llamacpp",
         "base_url": "http://h:8000", "api_key": "sk"}]))
    spec = run_pipeline.compute_desired_agents("x")["Vision-vLLM"]
    assert spec.script == "vision_worker_balanced_openai.py"
    assert spec.env["OPENAI_COMPAT_API_KEY"] == "sk"


def test_compute_desired_no_vision_fleet_when_blob_bad(isolated, monkeypatch):
    """Empty / malformed / url-less / bad-provider blobs spawn no fleet agents."""
    run_pipeline, _jc, _ = isolated
    monkeypatch.setenv("PIPELINE_VISION_WORKERS", "")
    for blob in ("", "[]", "not json", '{"x": 1}',
                 json.dumps([{"provider": "lmstudio", "base_url": ""}]),
                 json.dumps([{"provider": "bogus", "base_url": "http://h:1"}])):
        monkeypatch.setenv("VISION_WORKERS_JSON", blob)
        agents = run_pipeline.compute_desired_agents("t")
        assert _vision_labels(agents) == set(), f"blob={blob!r}"


# ── end-to-end: job projection → desired Local agents ─────────────────────────

def test_job_local_imports_project_into_desired_agents(isolated, monkeypatch):
    """Full path: a job's scrapers.local_imports → resolve_env(LOCAL_IMPORTS_JSON)
    → os.environ → compute_desired_agents fans out the Local agents. Disabled
    folders are filtered by resolve_env and never appear."""
    run_pipeline, job_config, _ = isolated
    monkeypatch.setenv("PIPELINE_VISION_WORKERS", "")
    job = job_config.create_job("Folders Job")
    job = job_config.set_override(job, "scrapers.local_imports", [
        {"name": "keepers", "dir": "/k", "enabled": True, "migrate_from": "m"},
        {"name": "off", "dir": "/o", "enabled": False, "migrate_from": ""},
    ])
    job_config.save_job(job)
    job_config.set_active("folders_job")

    # Project the active job into os.environ exactly as the supervisor does.
    env = run_pipeline.active_job_env()
    assert env is not None
    os.environ.update(env)

    agents = run_pipeline.compute_desired_agents(env.get("PIPELINE_TOPIC", ""))
    assert _local_labels(agents) == {"Local-keepers"}   # only the enabled folder
    assert agents["Local-keepers"].env["LOCAL_IMPORT_DIR"] == "/k"
    assert agents["Local-keepers"].env["LOCAL_IMPORT_MIGRATE_FROM"] == "m"


# ── priority: order + weights round-trip through the supervisor ───────────────

def test_scraper_priority_reorders_desired_agents(isolated, monkeypatch):
    """SCRAPER_PRIORITY_JSON reorders compute_desired_agents' returned dict so
    the top-priority scraper spawns first. Non-priority labels (Vision-*, etc.)
    keep their relative order behind the priority head."""
    run_pipeline, _jc, _ = isolated
    monkeypatch.setenv("PIPELINE_VISION_WORKERS", "")
    monkeypatch.setenv("SCRAPER_PRIORITY_JSON", json.dumps({
        "order": ["Web", "Civitai-Com", "X.com"],
        "weights": {"Web": 10, "Civitai-Com": 8, "X.com": 5},
    }))
    agents = run_pipeline.compute_desired_agents("topic")
    labels = list(agents.keys())
    # Web should now come first, ahead of X.com (default is X.com first).
    assert labels.index("Web") < labels.index("X.com")
    assert labels.index("Civitai-Com") < labels.index("X.com")


def test_scraper_priority_missing_env_uses_defaults(isolated, monkeypatch):
    """Empty / malformed SCRAPER_PRIORITY_JSON falls back to PRIORITY_NAMES
    order — no crash, no dropped agent."""
    run_pipeline, _jc, _ = isolated
    monkeypatch.setenv("PIPELINE_VISION_WORKERS", "")
    for blob in ("", "not json", '{"order": ["nope"]}'):
        monkeypatch.setenv("SCRAPER_PRIORITY_JSON", blob)
        agents = run_pipeline.compute_desired_agents("topic")
        assert "X.com" in agents  # X.com is first in default PRIORITY_NAMES


def test_scraper_priority_wired_from_job_override(isolated, monkeypatch):
    """End-to-end: a job's scrapers.priority override projects into env and
    reorders the supervisor's desired agents dict."""
    run_pipeline, job_config, _ = isolated
    monkeypatch.setenv("PIPELINE_VISION_WORKERS", "")
    job = job_config.create_job("Prio")
    job = job_config.set_override(job, "scrapers.priority", {
        "order": ["Civitai-Red", "Web"],
        "weights": {"Civitai-Red": 10, "Web": 8},
    })
    job_config.save_job(job)
    job_config.set_active("prio")
    env = run_pipeline.active_job_env()
    assert env is not None
    os.environ.update(env)
    payload = json.loads(env["SCRAPER_PRIORITY_JSON"])
    assert payload["order"][0] == "Civitai-Red"
    assert payload["order"][1] == "Web"
    agents = run_pipeline.compute_desired_agents(env.get("PIPELINE_TOPIC", ""))
    labels = list(agents.keys())
    assert labels.index("Civitai-Red") < labels.index("X.com")


# ── Multi-active: desired_active_slugs + shared vision fleet ────────────────

def test_desired_active_slugs_reflects_multi_activate(isolated):
    run_pipeline, jc, _ = isolated
    jc.create_job("A"); jc.create_job("B")
    jc.activate("a"); jc.activate("b")           # additive
    assert run_pipeline.desired_active_slugs() == ["a", "b"]
    assert run_pipeline.desired_active_slug() == "a"


def test_active_job_priorities_defaults_when_unset(isolated):
    run_pipeline, jc, _ = isolated
    jc.create_job("A"); jc.create_job("B")
    jc.activate("a"); jc.activate("b")
    prio = run_pipeline.active_job_priorities()
    assert set(prio.keys()) == {"a", "b"}
    # Both start at PRIORITY_WEIGHT_DEFAULT (5).
    assert all(w == jc.PRIORITY_WEIGHT_DEFAULT for w in prio.values())
    jc.set_job_priority("b", 9)
    prio = run_pipeline.active_job_priorities()
    assert prio["b"] == 9


def test_compute_desired_agents_multi_prefixes_scrapers_by_slug(isolated, monkeypatch):
    """Per-slug scrapers are labelled ``slug::label`` so two active jobs' X.com
    workers don't collide in the supervisor's ``self._active`` map."""
    run_pipeline, _jc, _ = isolated
    slug_envs = {
        "job_a": {"PIPELINE_TOPIC": "a", "PIPELINE_SLUG": "job_a"},
        "job_b": {"PIPELINE_TOPIC": "b", "PIPELINE_SLUG": "job_b"},
    }
    agents = run_pipeline.compute_desired_agents_multi(slug_envs, {"job_a": 5, "job_b": 3})
    assert "job_a::X.com" in agents
    assert "job_b::X.com" in agents
    assert agents["job_a::X.com"].slug == "job_a"
    assert agents["job_b::X.com"].slug == "job_b"


def test_compute_desired_agents_multi_shared_vision_fleet_deduped(isolated):
    """A vision worker with the same (provider, base_url, model) declared in
    both jobs' vision.workers becomes ONE shared Vision- agent, and it carries
    PIPELINE_ACTIVE_SLUGS_JSON = [[slug, weight], …]."""
    run_pipeline, _jc, _ = isolated
    fleet_a = [{"id": "a1", "name": "LM Shared", "provider": "lmstudio",
                "base_url": "http://h:1234", "model": "m", "api_key": ""}]
    fleet_b = [{"id": "b1", "name": "LM Shared", "provider": "lmstudio",
                "base_url": "http://h:1234", "model": "m", "api_key": ""},
               {"id": "b2", "name": "Olla B", "provider": "ollama",
                "base_url": "http://h:11434", "model": "", "api_key": ""}]
    slug_envs = {
        "job_a": {"VISION_WORKERS_JSON": json.dumps(fleet_a),
                  "PIPELINE_SLUG": "job_a"},
        "job_b": {"VISION_WORKERS_JSON": json.dumps(fleet_b),
                  "PIPELINE_SLUG": "job_b"},
    }
    agents = run_pipeline.compute_desired_agents_multi(slug_envs, {"job_a": 5, "job_b": 7})
    vision = {label: spec for label, spec in agents.items() if label.startswith("Vision-")}
    # LM Shared is deduped across the two jobs; Olla B is job_b only.
    assert set(vision.keys()) == {"Vision-LM Shared", "Vision-Olla B"}
    lm = vision["Vision-LM Shared"]
    assert lm.slug is None                        # shared, not tagged to a slug
    active_json = json.loads(lm.env["PIPELINE_ACTIVE_SLUGS_JSON"])
    assert active_json == [["job_a", 5], ["job_b", 7]]


def test_compute_desired_agents_multi_shared_fleet_all_have_active_slugs_env(isolated):
    """Every shared vision spec carries the active-slug projection env so the
    queue_manager shim can weighted-round-robin regardless of which worker
    pops first."""
    run_pipeline, _jc, _ = isolated
    fleet = [{"id": "x", "name": "X1", "provider": "lmstudio",
              "base_url": "http://h:1", "model": "m", "api_key": ""},
             {"id": "y", "name": "Y1", "provider": "ollama",
              "base_url": "http://h:2", "model": "", "api_key": ""}]
    slug_envs = {"only": {"VISION_WORKERS_JSON": json.dumps(fleet)}}
    agents = run_pipeline.compute_desired_agents_multi(slug_envs, {"only": 5})
    for label, spec in agents.items():
        if not label.startswith("Vision-"):
            continue
        assert "PIPELINE_ACTIVE_SLUGS_JSON" in spec.env


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
