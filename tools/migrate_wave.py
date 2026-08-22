"""Wave-upgrade verification + reconciliation tool.

Run this after ``git pull``-ing the user-acquisition wave on top of an install
that was tracking ``main``. It is READ-ONLY by default (``--dry-run``): it
verifies every on-disk state file the wave touched still parses, reports any
soft-cap breaches, and previews the one-shot migrations (legacy vision env →
fleet, legacy .env → default job). Nothing is written until you re-run
without ``--dry-run``.

Safe and idempotent. Never deletes or overwrites user data — a legacy preset
that exceeds the current soft cap is REPORTED, not truncated. Trim it via the
dashboard editor when you're ready.

Usage::

    # Preview the upgrade (default)
    python tools/migrate_wave.py

    # Actually run the one-shot migrations (jobs adoption, vision fleet fold-in)
    python tools/migrate_wave.py --apply

    # Read-only verification, useful in CI or a post-deploy hook
    python tools/migrate_wave.py --check-only
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CODE = REPO_ROOT / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import job_config  # noqa: E402  (after sys.path + dotenv)
import paths  # noqa: E402
import scheduler  # noqa: E402


#: Soft ceiling the dashboard save path enforces for categories. Presets
#: above this still LOAD (the wave dropped the validator cap from 40 to 12
#: in the earlier draft — this tool checks the read is still tolerant), but
#: the UI editor refuses to persist a new save. See config_io._MAX_CATEGORIES
#: and job_config._SOFT_CATEGORY_LIMIT for the actual gates.
_SOFT_CATEGORY_LIMIT = 12


def _fmt(ok: bool) -> str:
    return "OK  " if ok else "WARN"


def _check_presets() -> tuple[bool, list[str]]:
    """Parse the presets file and flag anything the UI editor will refuse.

    Returns ``(all_ok, notes)`` where ``notes`` is a human-readable list of
    findings — each note is either "OK" or "WARN" prefixed.
    """
    notes: list[str] = []
    all_ok = True
    try:
        lib = job_config.list_presets()
    except Exception as exc:  # noqa: BLE001 - report + move on
        return False, [f"WARN presets: unreadable — {exc}"]
    presets = lib.get("presets") or {}
    notes.append(f"OK   presets: {len(presets)} defined, default={lib.get('default')!r}")
    for name, cfg in sorted(presets.items()):
        cats = cfg.get("categories") or [] if isinstance(cfg, dict) else []
        if isinstance(cats, list) and len(cats) > _SOFT_CATEGORY_LIMIT:
            all_ok = False
            notes.append(
                f"WARN preset {name!r}: {len(cats)} categories "
                f"(>{_SOFT_CATEGORY_LIMIT} — dashboard save will refuse; trim before editing)"
            )
    # Kohya import default seeding is silent (deep-merged into every preset
    # via _default_preset_cfg), so just confirm the shape landed on default.
    default_name = lib.get("default", "default")
    default_cfg = job_config.get_preset(default_name)
    ko = (default_cfg.get("scrapers") or {}).get("kohya_import")
    if isinstance(ko, dict):
        notes.append(f"OK   scrapers.kohya_import: seeded (enabled={ko.get('enabled', False)})")
    else:
        notes.append("WARN scrapers.kohya_import: NOT seeded — check job_config._default_preset_cfg")
        all_ok = False
    return all_ok, notes


def _check_jobs() -> tuple[bool, list[str]]:
    notes: list[str] = []
    try:
        jobs = job_config.list_jobs()
    except Exception as exc:  # noqa: BLE001
        return False, [f"WARN jobs: list_jobs failed — {exc}"]
    active = job_config.get_active_slug()
    notes.append(f"OK   jobs: {len(jobs)} on disk, active={active!r}")
    for job in jobs:
        # from_dict() silently upgrades v1 files → we only see failures via
        # unreadable / bad-JSON files.
        notes.append(
            f"OK   job {job.slug!r}: preset={job.preset!r}, "
            f"status={job.status!r}, subject={(job.subject or '')[:40]!r}"
        )
    return True, notes


def _check_schedules() -> tuple[bool, list[str]]:
    notes: list[str] = []
    try:
        rows = scheduler.list_schedules()
    except Exception as exc:  # noqa: BLE001
        return False, [f"WARN schedules: {exc}"]
    notes.append(f"OK   schedules: {len(rows)} loaded")
    # The wave added the ``digest`` action; older schedule files never used
    # it and just don't appear here.
    return True, notes


def _check_index_db() -> tuple[bool, list[str]]:
    notes: list[str] = []
    db_path = paths.base_dir() / "cull_index.sqlite3"
    if not db_path.is_file():
        return True, [f"OK   sqlite index: not created yet ({db_path.name})"]
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.execute("PRAGMA table_info(images)")
            cols = {row[1] for row in cur.fetchall()}
            row_count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        return False, [f"WARN sqlite index: {exc}"]
    # The wave broadened the ingestion filter (MEDIA_EXTS = images + video),
    # which is purely additive discovery — no schema change. If ``phash`` is
    # missing the loader's _migrate_add_columns will add it on next open.
    missing_phash = "phash" not in cols
    notes.append(
        f"OK   sqlite index: {row_count} rows, {len(cols)} columns "
        f"(phash column {'present' if not missing_phash else 'will auto-add on load'})"
    )
    return True, notes


def _check_active_taxonomy() -> tuple[bool, list[str]]:
    path = paths.base_dir() / "cull_categories.json"
    if not path.is_file():
        return True, ["OK   cull_categories.json: not projected yet"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"WARN cull_categories.json: {exc}"]
    cats = data.get("categories") if isinstance(data, dict) else None
    if not isinstance(cats, list):
        return False, ["WARN cull_categories.json: 'categories' missing or wrong shape"]
    return True, [f"OK   cull_categories.json: {len(cats)} active categories"]


def _check_env_seeds() -> tuple[bool, list[str]]:
    """Legacy single-endpoint vision env vars can be folded into the fleet."""
    import os
    keys = ("LMSTUDIO_PRIMARY_URL", "LMSTUDIO_SECONDARY_URL", "OLLAMA_URL",
            "OPENAI_COMPAT_URL")
    present = [k for k in keys if os.environ.get(k)]
    if not present:
        return True, ["OK   legacy vision env: none set (fleet already authoritative)"]
    return True, [
        f"OK   legacy vision env: {len(present)} endpoint(s) will fold into the "
        f"default preset's vision.workers fleet on --apply ({', '.join(present)})"
    ]


def run(*, apply: bool, check_only: bool) -> int:
    print("cull - wave upgrade audit", flush=True)
    print(f"  data dir: {paths.base_dir()}", flush=True)
    print(f"  mode:     {'APPLY' if apply else ('CHECK-ONLY' if check_only else 'DRY-RUN')}",
          flush=True)
    print()

    sections: list[tuple[str, tuple[bool, list[str]]]] = [
        ("presets",           _check_presets()),
        ("jobs",              _check_jobs()),
        ("schedules",         _check_schedules()),
        ("sqlite index",      _check_index_db()),
        ("active taxonomy",   _check_active_taxonomy()),
        ("env vision seeds",  _check_env_seeds()),
    ]
    any_warn = False
    for title, (ok, notes) in sections:
        print(f"[{title}]", flush=True)
        for note in notes:
            print(f"  {note}", flush=True)
            if note.startswith("WARN"):
                any_warn = True
        print(flush=True)

    if check_only:
        return 1 if any_warn else 0

    if not apply:
        print("(dry-run) no changes written. Re-run with --apply to fold in "
              "legacy .env / vision endpoints and adopt discovered on-disk "
              "slugs as jobs.", flush=True)
        return 0

    # Apply-mode: idempotent migrations that already exist in job_config.
    created = job_config.migrate_existing_data()
    if created:
        print("Adopted:", flush=True)
        for j in created:
            print(f"  + {j.slug}  (preset={j.preset!r})", flush=True)
    else:
        print("No new jobs adopted.", flush=True)
    if job_config.migrate_legacy_vision_to_fleet():
        print("Folded legacy vision env vars into the default preset's fleet.",
              flush=True)
    else:
        print("Vision fleet already configured — no fold-in needed.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true",
        help="Run the one-shot migrations (default is dry-run).",
    )
    ap.add_argument(
        "--check-only", action="store_true",
        help="Report only; exit code 1 if any WARN found. Never migrates.",
    )
    args = ap.parse_args()
    if args.apply and args.check_only:
        ap.error("--apply and --check-only are mutually exclusive")
    return run(apply=args.apply, check_only=args.check_only)


if __name__ == "__main__":
    raise SystemExit(main())
