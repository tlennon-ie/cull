"""Tests for the headless cull CLI (``pipeline_code/cull_cli.py``).

Runnable two ways:
    pytest tests/test_cull_cli.py
    python tests/test_cull_cli.py        # falls back to pytest.main

The CLI is a thin argparse wrapper over the EXISTING ``job_config`` /
``run_pipeline`` / ``paths`` public APIs — it must not reimplement job state.
Every test runs against a temp ``PIPELINE_BASE_DIR`` (mirroring the fixture in
``tests/test_job_config.py``) so nothing touches the real data dir, and the
``run`` subcommand is exercised against a MOCKED supervisor entrypoint so no
pipeline is ever spawned and no network is touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point all storage at a temp dir and hand back (cull_cli, job_config, tmp).

    Mirrors ``tests/test_job_config.py``: pin ``PIPELINE_BASE_DIR``, redirect the
    categories active-path + bust its cache, then reload ``job_config`` so its
    import-time path constants resolve into the temp dir. ``cull_cli`` is
    imported (cheaply — it must NOT import ``run_pipeline`` at module top) and
    handed back so the dispatcher can be driven directly.
    """
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.delenv("PIPELINE_SLUG", raising=False)
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import importlib
    import job_config
    importlib.reload(job_config)
    import cull_cli
    importlib.reload(cull_cli)
    return cull_cli, job_config, tmp_path


# ── parser construction ──────────────────────────────────────────────────────

def test_build_parser_returns_argument_parser(isolated):
    cli, _jc, _ = isolated
    parser = cli.build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parser_help_does_not_raise(isolated, capsys):
    """``--help`` exits 0 (argparse SystemExit) and prints usage."""
    cli, _jc, _ = isolated
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_main_without_subcommand_is_nonzero(isolated, capsys):
    """No subcommand → friendly non-zero exit, not a traceback."""
    cli, _jc, _ = isolated
    rc = cli.main([])
    assert rc != 0


def test_lazy_import_run_pipeline_not_loaded_on_import(isolated):
    """The CLI must import cheaply: ``run_pipeline`` is imported lazily inside the
    ``run`` handler, never at module top."""
    cli, _jc, _ = isolated
    # Drop any previously-imported run_pipeline so this assertion is meaningful.
    sys.modules.pop("run_pipeline", None)
    import importlib
    importlib.reload(cli)
    assert "run_pipeline" not in sys.modules


# ── jobs list ────────────────────────────────────────────────────────────────

def test_jobs_list_empty_store(isolated, capsys):
    cli, _jc, _ = isolated
    rc = cli.main(["jobs", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    # Empty store prints *something* (a header / "no jobs" line), never crashes.
    assert out.strip() != ""


def test_jobs_list_prints_slug_name_and_active_marker(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("Alpha", subject="Alpha Subject")
    jc.create_job("Beta")
    jc.set_active("alpha")
    rc = cli.main(["jobs", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "beta" in out
    assert "Alpha" in out  # the display name


# ── job create + jobs activate (end to end) ──────────────────────────────────

def test_job_create_then_activate(isolated, capsys):
    cli, jc, _ = isolated
    rc = cli.main(["job", "create", "myjob",
                   "--preset", "default", "--subject", "A Test Subject"])
    assert rc == 0
    job = jc.get_job("myjob")
    assert job is not None
    assert job.subject == "A Test Subject"
    assert job.preset == "default"

    capsys.readouterr()  # clear create output
    rc = cli.main(["jobs", "activate", "myjob"])
    assert rc == 0
    assert jc.get_active_slug() == "myjob"


def test_job_create_duplicate_is_nonzero(isolated, capsys):
    cli, jc, _ = isolated
    assert cli.main(["job", "create", "dup", "--subject", "s"]) == 0
    rc = cli.main(["job", "create", "dup", "--subject", "s"])
    assert rc != 0


def test_jobs_activate_unknown_slug_is_nonzero(isolated):
    cli, _jc, _ = isolated
    rc = cli.main(["jobs", "activate", "does_not_exist"])
    assert rc != 0


def test_job_create_projects_categories_on_activate(isolated):
    """Activating through the CLI must drive the real ``activate`` projection so
    the active taxonomy reflects the job's categories."""
    cli, jc, _ = isolated
    import categories
    job = jc.create_job("Cats", subject="S")
    job = jc.set_override(job, "categories", [{"name": "OnlyCat", "hint": ""}])
    jc.save_job(job)
    assert cli.main(["jobs", "activate", "cats"]) == 0
    assert categories.get_categories() == ("OnlyCat",)


# ── presets list ─────────────────────────────────────────────────────────────

def test_presets_list_includes_default(isolated, capsys):
    cli, _jc, _ = isolated
    rc = cli.main(["presets", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "default" in out


def test_presets_list_includes_builtins(isolated, capsys):
    cli, jc, _ = isolated
    rc = cli.main(["presets", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    for name in jc.builtin_preset_names():
        assert name in out


# ── status ───────────────────────────────────────────────────────────────────

def test_status_no_active_job(isolated, capsys):
    cli, _jc, _ = isolated
    rc = cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() != ""


def test_status_reports_active_job_and_counts(isolated, capsys):
    cli, jc, tmp = isolated
    jc.create_job("Active One", subject="S")
    jc.activate("active_one")
    # Seed a couple of queued files so the count is non-trivial + safe to read.
    qdir = tmp / "queue" / "active_one" / "civitai"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "a.png").write_bytes(b"x")
    (qdir / "b.png").write_bytes(b"y")
    rc = cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "active_one" in out


# ── run (mocked entrypoint — nothing spawns) ─────────────────────────────────

def test_run_invokes_mocked_entrypoint(isolated, monkeypatch):
    """``run`` must call run_pipeline's real entrypoint exactly once. We inject a
    fake ``run_pipeline`` module so nothing actually spawns."""
    cli, _jc, _ = isolated
    import types
    calls: dict[str, int] = {"n": 0}
    fake = types.ModuleType("run_pipeline")

    def _fake_main() -> None:
        calls["n"] += 1

    fake.main = _fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "run_pipeline", fake)

    rc = cli.main(["run"])
    assert rc == 0
    assert calls["n"] == 1


def test_run_does_not_import_run_pipeline_until_called(isolated, monkeypatch):
    """Building the parser / dispatching other commands must not import the
    supervisor. Only the ``run`` handler triggers the lazy import."""
    cli, jc, _ = isolated
    sys.modules.pop("run_pipeline", None)
    cli.main(["presets", "list"])
    assert "run_pipeline" not in sys.modules


# ── export (optional module; absent here → clean non-zero) ────────────────────

def test_export_missing_module_is_nonzero(isolated, monkeypatch, capsys):
    """``export_profiles`` is not shipped → the CLI prints a clear message and
    exits non-zero rather than throwing an ImportError traceback."""
    cli, jc, tmp = isolated
    monkeypatch.setitem(sys.modules, "export_profiles", None)  # force ImportError
    jc.create_job("Exp", subject="S")
    rc = cli.main(["export", "exp", "--profile", "sd",
                   "--out", str(tmp / "out")])
    assert rc != 0
    combined = capsys.readouterr()
    assert "export" in (combined.out + combined.err).lower()


def test_export_calls_export_dataset_when_available(isolated, monkeypatch):
    """When an ``export_profiles`` module IS importable, the CLI delegates to its
    ``export_dataset`` entrypoint with the parsed args."""
    cli, jc, tmp = isolated
    import types
    seen: dict[str, object] = {}
    fake = types.ModuleType("export_profiles")

    def _fake_export_dataset(slug, profile, out_dir):  # noqa: ANN001
        seen["slug"] = slug
        seen["profile"] = profile
        seen["out_dir"] = out_dir

    fake.export_dataset = _fake_export_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "export_profiles", fake)
    jc.create_job("Exp", subject="S")

    rc = cli.main(["export", "exp", "--profile", "booru",
                   "--out", str(tmp / "out")])
    assert rc == 0
    assert seen["slug"] == "exp"
    assert seen["profile"] == "booru"
    assert str(seen["out_dir"]) == str(tmp / "out")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
