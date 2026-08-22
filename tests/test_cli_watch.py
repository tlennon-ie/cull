"""Tests for the ``cull jobs watch`` predicate parser + wait loop.

Predicate grammar (see AGENTS.md § Deciding when to stop):

    sorted-count>=N | queue-count<=N | active-slug=X | elapsed>=Ns

Exit contract:
  * 0 — predicate satisfied
  * 2 — bad predicate / bad flag
  * 3 — timed out before the predicate matched
  * 4 — unknown slug
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# Local copy of the multi-JSON-doc parser (previously imported from
# `tests.test_cli_json`, which fails because `tests/` is not a package under
# pytest's default config). Duplicating a 15-line helper is cheaper than
# adding a conftest.py + a shared-helpers module for one caller.
def _local_iter_json_docs(text: str):
    decoder = json.JSONDecoder()
    idx = 0
    text = text.lstrip()
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n":
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text[idx:])
        yield obj
        idx += end


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


# ── predicate parser (unit) ─────────────────────────────────────────────────

def test_parse_sorted_count_ge(isolated):
    cli, _, _ = isolated
    pred = cli._parse_predicate("sorted-count>=100")
    assert pred({"sorted-count": 100}) is True
    assert pred({"sorted-count": 99}) is False


def test_parse_queue_count_le(isolated):
    cli, _, _ = isolated
    pred = cli._parse_predicate("queue-count<=0")
    assert pred({"queue-count": 0}) is True
    assert pred({"queue-count": 5}) is False


def test_parse_active_slug_eq(isolated):
    cli, _, _ = isolated
    pred = cli._parse_predicate("active-slug=my_job")
    assert pred({"active-slug": "my_job"}) is True
    assert pred({"active-slug": "other"}) is False


def test_parse_elapsed_ge(isolated):
    cli, _, _ = isolated
    pred = cli._parse_predicate("elapsed>=30s")
    assert pred({"elapsed": 30.0}) is True
    assert pred({"elapsed": 29.9}) is False


def test_parse_rejects_bogus(isolated):
    cli, _, _ = isolated
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_predicate("not-a-key>=1")
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_predicate("sorted-count>1")  # operator not allowed
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_predicate("elapsed>=abc")


# ── end-to-end command dispatch ─────────────────────────────────────────────

def test_watch_unknown_slug_returns_4(isolated, capsys):
    cli, _, _ = isolated
    rc = cli.main(["jobs", "watch", "--slug", "nope",
                   "--until", "sorted-count>=1"])
    assert rc == 4


def test_watch_bad_predicate_returns_2(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("w", subject="s")
    rc = cli.main(["jobs", "watch", "--slug", "w", "--until", "gibberish"])
    assert rc == 2


def test_watch_condition_met_immediately(isolated, capsys, tmp_path):
    cli, jc, tp = isolated
    jc.create_job("wait", subject="s")
    # Seed one sorted image so sorted-count>=1 fires on first tick.
    sdir = tp / "sorted" / "wait" / "Keep"
    sdir.mkdir(parents=True)
    (sdir / "a.png").write_bytes(b"x")
    rc = cli.main([
        "jobs", "watch", "--slug", "wait", "--until", "sorted-count>=1",
        "--interval", "0.1", "--timeout", "5", "--json",
    ])
    assert rc == 0
    # Parse the final JSON document
    _iter_json_docs = _local_iter_json_docs  # noqa: F841 — see helper below  # reuse robust parser
    docs = list(_iter_json_docs(capsys.readouterr().out))
    payload = docs[-1]
    assert payload["ok"] is True
    assert payload["condition_met"] is True
    assert payload["state"]["sorted-count"] == 1


def test_watch_times_out_returns_3(isolated, capsys):
    cli, jc, _ = isolated
    jc.create_job("out", subject="s")
    rc = cli.main([
        "jobs", "watch", "--slug", "out", "--until", "sorted-count>=1000",
        "--interval", "0.05", "--timeout", "0.2", "--json",
    ])
    assert rc == 3
    _iter_json_docs = _local_iter_json_docs  # noqa: F841 — see helper below
    docs = list(_iter_json_docs(capsys.readouterr().out))
    payload = docs[-1]
    assert payload["ok"] is False
    assert payload["exit_code"] == 3
    assert payload["condition_met"] is False


def test_watch_snapshot_mode_no_until_no_timeout(isolated, capsys):
    """No ``--until`` and no ``--timeout`` returns a single snapshot and exits."""
    cli, jc, _ = isolated
    jc.create_job("snap", subject="s")
    rc = cli.main(["jobs", "watch", "--slug", "snap", "--json"])
    assert rc == 0
    _iter_json_docs = _local_iter_json_docs  # noqa: F841 — see helper below
    docs = list(_iter_json_docs(capsys.readouterr().out))
    payload = docs[-1]
    assert payload["condition_met"] is None
    assert payload["state"]["slug"] == "snap"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
