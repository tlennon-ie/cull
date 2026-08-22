"""Happy-path tests for the new agent-facing subcommands.

Covers:
  * ``cull scoring set``
  * ``cull scrapers add-url``
  * ``cull scrapers toggle``
  * ``cull config show``
  * ``cull stats``
  * ``cull gallery sample``
  * ``cull export kohya``  (dispatched via argv rewrite)

The tests deliberately verify the persisted mutation (job overrides on disk)
rather than only the human-readable output, so a future refactor of the
formatting layer can not silently mask a regression in the semantics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.delenv("PIPELINE_SLUG", raising=False)
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH",
                        tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import importlib
    import job_config
    importlib.reload(job_config)
    import cull_cli
    importlib.reload(cull_cli)
    return cull_cli, job_config, tmp_path


# ── scoring set ─────────────────────────────────────────────────────────────

def test_scoring_set_writes_overrides(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("sc", subject="s")
    rc = cli.main([
        "scoring", "set", "--job", "sc",
        "--min-ovr", "80", "--min-rel", "65", "--require-prompt", "false",
    ])
    assert rc == 0
    saved = jc.get_job("sc")
    assert saved.overrides["scoring"]["ovr_min"] == 80
    assert saved.overrides["scoring"]["rel_min"] == 65
    assert saved.overrides["topic_filters"]["require_prompt"] is False


def test_scoring_set_requires_at_least_one_update(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("sc", subject="s")
    rc = cli.main(["scoring", "set", "--job", "sc"])
    assert rc == 2  # EXIT_BAD_ARGS


def test_scoring_set_rejects_out_of_range(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("sc", subject="s")
    rc = cli.main(["scoring", "set", "--job", "sc", "--min-ovr", "150"])
    assert rc == 2  # EXIT_BAD_ARGS


# ── scrapers add-url ────────────────────────────────────────────────────────

def test_scrapers_add_url_persists_and_enables(isolated, capsys, monkeypatch):
    cli, jc, _ = isolated
    import scheduler
    monkeypatch.setattr(scheduler, "_is_public_http_url", lambda url: True)
    jc.create_job("scr", subject="s")
    rc = cli.main([
        "scrapers", "add-url", "--job", "scr", "--source", "gallery_dl",
        "--url", "https://example.com/feed",
    ])
    assert rc == 0
    saved = jc.get_job("scr")
    urls = saved.overrides["scrapers"]["gallery_dl"]["urls"]
    assert urls == ["https://example.com/feed"]
    # First URL flips the source on so the next scheduler run picks it up.
    assert saved.overrides["scrapers"]["gallery_dl"]["enabled"] is True


def test_scrapers_add_url_ssrf_guard(isolated, capsys, monkeypatch):
    cli, jc, _ = isolated
    import scheduler
    # Simulate the guard rejecting a localhost URL (the real one already does).
    monkeypatch.setattr(scheduler, "_is_public_http_url", lambda url: False)
    jc.create_job("scr", subject="s")
    rc = cli.main([
        "scrapers", "add-url", "--job", "scr", "--source", "gallery_dl",
        "--url", "http://127.0.0.1:9000/feed",
    ])
    assert rc == 2  # EXIT_BAD_ARGS
    # The rejected URL must not have been persisted.
    saved = jc.get_job("scr")
    assert "gallery_dl" not in saved.overrides.get("scrapers", {})


def test_scrapers_add_url_bad_source(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("scr", subject="s")
    # ``choices=`` on argparse turns this into an argparse SystemExit(2).
    with pytest.raises(SystemExit):
        cli.main(["scrapers", "add-url", "--job", "scr",
                  "--source", "nonsense", "--url", "https://x/"])


# ── scrapers toggle ─────────────────────────────────────────────────────────

def test_scrapers_toggle_disables(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("tog", subject="s")
    name = jc.SCRAPER_NAMES[0]
    rc = cli.main(["scrapers", "toggle", "--job", "tog",
                   "--name", name, "--enabled", "false"])
    assert rc == 0
    saved = jc.get_job("tog")
    assert saved.overrides["scrapers"]["enabled"][name] is False


def test_scrapers_toggle_rejects_unknown_name(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("tog", subject="s")
    rc = cli.main(["scrapers", "toggle", "--job", "tog",
                   "--name", "not_a_real_scraper", "--enabled", "true"])
    assert rc == 2  # EXIT_BAD_ARGS


# ── config show ─────────────────────────────────────────────────────────────

def test_config_show_human_output(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("cfg", subject="my subject")
    rc = cli.main(["config", "show", "--job", "cfg"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cfg" in out
    assert "my subject" in out
    assert "Scoring" in out


# ── stats ───────────────────────────────────────────────────────────────────

def test_stats_human_output_lists_categories(isolated, capsys, tmp_path):
    cli, jc, tp = isolated
    jc.create_job("s", subject="s")
    jc.activate("s")
    for cat in ("Keep", "Borderline"):
        d = tp / "sorted" / "s" / cat
        d.mkdir(parents=True)
        (d / "a.png").write_bytes(b"x")
    rc = cli.main(["stats"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Keep" in out and "Borderline" in out


# ── gallery sample ─────────────────────────────────────────────────────────

def test_gallery_sample_human_output(isolated, capsys, tmp_path):
    cli, jc, tp = isolated
    jc.create_job("g", subject="s")
    d = tp / "sorted" / "g" / "Keep"
    d.mkdir(parents=True)
    for i in range(3):
        (d / f"i{i}.png").write_bytes(b"x")
    rc = cli.main(["gallery", "sample", "--job", "g", "--n", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    # 3 lines listing each relative path
    assert out.count("Keep") == 3


def test_gallery_sample_rejects_traversal_category(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("g", subject="s")
    rc = cli.main(["gallery", "sample", "--job", "g",
                   "--category", "../evil", "--n", "1"])
    assert rc == 2


# ── export kohya (dispatched via argv rewriter) ────────────────────────────

def test_export_kohya_dispatch(isolated, capsys, tmp_path, monkeypatch):
    """``cull export kohya`` should hit ``cmd_export_kohya`` and call
    ``export_profiles.export_dataset`` with the ``kohya`` profile."""
    cli, jc, tp = isolated
    jc.create_job("k", subject="s")

    import types
    seen: dict[str, object] = {}
    fake = types.ModuleType("export_profiles")

    def _fake(slug, profile, out_dir):
        seen["slug"] = slug
        seen["profile"] = profile
        seen["out_dir"] = out_dir
        return {"sample_count": 7}

    fake.export_dataset = _fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "export_profiles", fake)
    rc = cli.main(["export", "kohya", "--job", "k",
                   "--out", str(tp / "out")])
    assert rc == 0
    assert seen["profile"] == "kohya"
    assert seen["slug"] == "k"
    assert str(seen["out_dir"]) == str(tp / "out")


def test_export_hf_dispatch(isolated, capsys, monkeypatch):
    cli, jc, tp = isolated
    jc.create_job("h", subject="s")
    import types
    seen: dict[str, object] = {}
    fake = types.ModuleType("hf_export")

    def _fake(slug, repo, private=True, include_video=False):
        seen["slug"] = slug
        seen["repo"] = repo
        seen["private"] = private
        seen["include_video"] = include_video
        return {"repo_id": repo, "uploaded": 3}

    fake.push_to_hf = _fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hf_export", fake)
    rc = cli.main(["export", "hf", "--job", "h", "--repo", "user/name"])
    assert rc == 0
    assert seen["repo"] == "user/name"
    assert seen["private"] is True


def test_export_hf_bad_repo_shape(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("h", subject="s")
    rc = cli.main(["export", "hf", "--job", "h", "--repo", "no_slash"])
    assert rc == 2  # EXIT_BAD_ARGS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
