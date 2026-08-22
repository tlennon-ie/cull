"""``--json`` shape contract tests for the extended cull CLI.

Every subcommand that carries a ``--json`` flag must emit a single, parseable
JSON document on stdout with the documented top-level keys. If this test drifts
from the wire shape, downstream MCP/HTTP wrappers break silently — so keep the
assertions specific rather than "just parses".
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
    """Point every persistent path at ``tmp_path`` and reload cull_cli.

    Mirrors ``tests/test_cull_cli.py``'s fixture — every JSON assertion here
    also depends on job_config + categories being freshly wired against the
    temp dir so the CLI writes/reads under it exclusively.
    """
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


def _iter_json_docs(text: str):
    """Yield every JSON document present in ``text``.

    The CLI pretty-prints success payloads across multiple lines and may also
    emit a single-line error payload before it; a naive line-split would break
    on the multi-line case. Use ``JSONDecoder.raw_decode`` to walk the string
    document-by-document instead.
    """
    decoder = json.JSONDecoder()
    idx = 0
    text = text.lstrip()
    while idx < len(text):
        # skip whitespace between documents
        while idx < len(text) and text[idx] in " \t\r\n":
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text[idx:])
        yield obj
        idx += end


def _run_json(cli, argv, capsys):
    rc = cli.main(argv)
    out = capsys.readouterr().out
    docs = list(_iter_json_docs(out))
    return rc, docs


# ── shape assertions per subcommand ──────────────────────────────────────────

def test_jobs_list_json_shape(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("A", subject="s")
    jc.create_job("B", subject="s")
    jc.set_active("a")
    rc, docs = _run_json(cli, ["jobs", "list", "--json"], capsys)
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["active"] == "a"
    assert "active_slugs" in payload and isinstance(payload["active_slugs"], list)
    assert isinstance(payload["jobs"], list) and len(payload["jobs"]) == 2
    row = next(j for j in payload["jobs"] if j["slug"] == "a")
    assert set(row.keys()) >= {"slug", "name", "status", "subject", "preset",
                               "active"}
    assert row["active"] is True


def test_jobs_activate_json_shape(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("act", subject="s")
    rc, docs = _run_json(cli, ["jobs", "activate", "act", "--json"], capsys)
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["activated"] == "act"
    assert "act" in payload["active_slugs"]


def test_job_create_json_shape(isolated, capsys):
    cli, _, _ = isolated
    rc, docs = _run_json(
        cli, ["job", "create", "made", "--subject", "sub", "--json"], capsys,
    )
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    job = payload["job"]
    assert job["slug"] == "made" and job["subject"] == "sub"
    assert job["preset"] and job["status"]


def test_presets_list_json_shape(isolated, capsys):
    cli, _, _ = isolated
    rc, docs = _run_json(cli, ["presets", "list", "--json"], capsys)
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["default"]
    assert any(p["is_default"] for p in payload["presets"])
    assert all({"name", "builtin", "is_default"} <= set(p.keys())
               for p in payload["presets"])


def test_status_json_shape_empty(isolated, capsys):
    cli, _, _ = isolated
    rc, docs = _run_json(cli, ["status", "--json"], capsys)
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert "active_slug" in payload
    assert "queue" in payload
    assert "data_dir" in payload


def test_status_json_shape_with_active(isolated, capsys, tmp_path):
    cli, jc, tp = isolated
    jc.create_job("Live", subject="s")
    jc.activate("live")
    qdir = tp / "queue" / "live" / "civitai"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "a.png").write_bytes(b"x")
    rc, docs = _run_json(cli, ["status", "--json"], capsys)
    assert rc == 0
    payload = docs[-1]
    assert payload["active_slug"] == "live"
    assert payload["counts"]["queue"] == 1
    assert payload["counts"]["sorted"] == 0


def test_stats_json_shape(isolated, capsys, tmp_path):
    cli, jc, tp = isolated
    jc.create_job("Data", subject="s")
    jc.activate("data")
    keep = tp / "sorted" / "data" / "Keep"
    keep.mkdir(parents=True)
    (keep / "a.png").write_bytes(b"x")
    (keep / "a.vision.json").write_text(
        json.dumps({"OVR_Quality_Score": 82, "REL_Quality_Score": 74,
                    "quality_score": 8, "nsfw": False, "watermark": False}),
        encoding="utf-8",
    )
    (keep / "a.txt").write_text("a caption", encoding="utf-8")
    rc, docs = _run_json(cli, ["stats", "--json"], capsys)
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["slug"] == "data"
    assert payload["sorted_count"] == 1
    assert payload["counts_by_category"]["Keep"] == 1
    assert payload["score_distribution"]["ovr"]["80-89"] == 1
    assert payload["score_distribution"]["rel"]["70-79"] == 1
    assert payload["with_prompt"] == 1


def test_gallery_sample_json_shape(isolated, capsys, tmp_path):
    cli, jc, tp = isolated
    jc.create_job("Gal", subject="s")
    jc.activate("gal")
    keep = tp / "sorted" / "gal" / "Keep"
    keep.mkdir(parents=True)
    for i in range(3):
        (keep / f"img{i}.png").write_bytes(b"x")
        (keep / f"img{i}.vision.json").write_text(
            json.dumps({"OVR_Quality_Score": 60, "REL_Quality_Score": 50,
                        "nsfw": False}),
            encoding="utf-8",
        )
    rc, docs = _run_json(
        cli, ["gallery", "sample", "--job", "gal", "--n", "2", "--json"],
        capsys,
    )
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["requested"] == 2
    assert payload["returned"] == 2
    assert len(payload["samples"]) == 2
    sample = payload["samples"][0]
    for k in ("path", "relative_path", "category", "ovr", "rel", "nsfw",
              "watermark", "prompt", "vision_json"):
        assert k in sample


def test_scoring_set_json_shape(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("Score", subject="s")
    rc, docs = _run_json(
        cli, ["scoring", "set", "--job", "score", "--min-ovr", "70",
              "--min-rel", "60", "--json"], capsys,
    )
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["slug"] == "score"
    assert payload["updates"]["scoring.ovr_min"] == 70
    assert payload["updates"]["scoring.rel_min"] == 60
    # Overrides landed on disk (project mechanism will pick them up on activate)
    saved = jc.get_job("score")
    assert saved.overrides["scoring"]["ovr_min"] == 70


def test_scrapers_list_json_shape(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("scr", subject="s")
    rc, docs = _run_json(cli, ["scrapers", "list", "--job", "scr", "--json"],
                         capsys)
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["slug"] == "scr"
    assert isinstance(payload["scrapers"], list) and payload["scrapers"]
    for row in payload["scrapers"]:
        assert {"name", "enabled"} <= set(row.keys())
    assert set(payload["gallery_dl"].keys()) >= {"enabled", "url_count",
                                                 "limit_per_url"}


def test_scrapers_add_url_json_shape(isolated, capsys, monkeypatch):
    cli, jc, _ = isolated
    # Bypass the SSRF guard for tests — a real DNS lookup would flake in CI.
    import scheduler
    monkeypatch.setattr(scheduler, "_is_public_http_url", lambda url: True)
    jc.create_job("addurl", subject="s")
    rc, docs = _run_json(
        cli, ["scrapers", "add-url", "--job", "addurl", "--source",
              "gallery_dl", "--url", "https://example.com/foo", "--json"],
        capsys,
    )
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["source"] == "gallery_dl"
    assert payload["changed"] is True
    assert "https://example.com/foo" in payload["urls"]


def test_scrapers_toggle_json_shape(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("tog", subject="s")
    name = jc.SCRAPER_NAMES[0]
    rc, docs = _run_json(
        cli, ["scrapers", "toggle", "--job", "tog", "--name", name,
              "--enabled", "false", "--json"], capsys,
    )
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["scraper"] == name
    assert payload["enabled"] is False


def test_config_show_json_shape_masks_secrets(isolated, capsys):
    cli, jc, _ = isolated
    job = jc.create_job("Cfg", subject="s")
    # Plant a fake credential in the override tree so we can assert it is
    # masked; secret-shaped keys ('api_key', 'token', 'cookies', 'secret',
    # 'password', 'authorization') must never leak in config show output.
    job = jc.set_override(
        job, "scrapers.gallery_dl.cookies_file", "SECRET_COOKIE_VALUE",
    )
    jc.save_job(job)
    rc, docs = _run_json(cli, ["config", "show", "--job", "cfg", "--json"],
                         capsys)
    assert rc == 0
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["slug"] == "cfg"
    # The masked value is "***"; the raw value must not appear anywhere in the
    # emitted JSON envelope.
    dumped = json.dumps(payload)
    assert "SECRET_COOKIE_VALUE" not in dumped
    assert "***" in dumped


def test_missing_job_returns_documented_exit_code(isolated, capsys):
    cli, _, _ = isolated
    rc = cli.main(["stats", "--job", "nope", "--json"])
    # EXIT_MISSING_JOB is documented as 4.
    assert rc == 4
    docs = list(_iter_json_docs(capsys.readouterr().out))
    payload = docs[-1]
    assert payload["ok"] is False
    assert payload["exit_code"] == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
