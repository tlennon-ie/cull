"""Headless command-line interface for cull.

Run cull on a remote GPU box with no dashboard. This is a *thin* argparse
wrapper over the EXISTING public APIs — it never reimplements job state:

  * ``job_config`` owns jobs/presets/the active pointer (single source of truth).
  * ``run_pipeline`` owns the supervisor (imported LAZILY so this module stays
    cheap to import and unit-testable without ever spawning a pipeline).
  * ``paths`` owns the filesystem layout used by ``status``.
  * ``index_store`` owns the SQLite index used by ``stats`` / ``gallery``.
  * ``hf_export`` / ``export_profiles`` own dataset export.

Subcommands::

    cull jobs list [--json]
    cull jobs activate <slug>
    cull jobs watch --slug SLUG [--until PREDICATE] [--interval S] [--json]
    cull job create <slug> [--preset NAME] [--subject "text"]
    cull presets list [--json]
    cull status [--json]
    cull stats [--job SLUG] [--json]
    cull gallery sample --job SLUG [--category NAME] [--n N] [--json]
    cull scoring set --job SLUG [--min-ovr N] [--min-rel N] [--require-prompt BOOL]
    cull scrapers list [--job SLUG] [--json]
    cull scrapers add-url --job SLUG --source gallery_dl --url URL
    cull scrapers toggle --job SLUG --name NAME --enabled BOOL
    cull config show [--job SLUG] [--json]
    cull run
    cull export <slug> --profile P --out DIR [--json]        # legacy profile export
    cull export kohya --job SLUG --out DIR [--json]          # convenience wrapper
    cull export hf --job SLUG --repo user/name [--json]      # HuggingFace push

Every subcommand supports ``--json`` for machine-readable output. Every handler
returns an ``int`` exit code with the following contract:

  * 0 — success
  * 2 — bad arguments / usage error
  * 3 — watch condition timed out (only ``jobs watch``)
  * 4 — missing job or preset
  * 5 — subprocess / export failure

Errors are reported to stderr with a clear, non-traceback message.
"""
from __future__ import annotations

import argparse
import json as _json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import job_config
import paths
from pipeline_logging import get_logger

logger = get_logger(__name__)

_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
)

# ── Exit codes (documented contract) ─────────────────────────────────────────
EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_WATCH_TIMEOUT = 3
EXIT_MISSING_JOB = 4
EXIT_SUBPROCESS_FAIL = 5


# ── small output helpers ─────────────────────────────────────────────────────

def _err(message: str) -> None:
    """Print a user-facing error to stderr (no traceback)."""
    print(f"error: {message}", file=sys.stderr, flush=True)


def _emit(payload: Any, *, as_json: bool, text: str | None = None) -> None:
    """Print either JSON (``--json``) or a human-readable ``text`` line.

    When ``text`` is None we fall back to ``str(payload)``.
    """
    if as_json:
        print(_json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(text if text is not None else str(payload))


def _emit_error_json(message: str, *, code: int, as_json: bool) -> int:
    """Return ``code`` after printing an error in the requested format."""
    if as_json:
        # JSON error rides stdout so a caller can uniformly parse both success
        # and failure; a duplicate line still lands on stderr for humans.
        print(_json.dumps({"ok": False, "error": message, "exit_code": code},
                          ensure_ascii=False))
    _err(message)
    return code


def _bool_arg(value: str) -> bool:
    """Parse a permissive boolean string (true/false, 1/0, yes/no, on/off)."""
    if isinstance(value, bool):
        return value
    s = str(value or "").strip().lower()
    if s in ("true", "1", "yes", "on", "y", "t"):
        return True
    if s in ("false", "0", "no", "off", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean (true/false/yes/no/1/0), got {value!r}"
    )


def _count_images(root: Path) -> int:
    """Cheaply count image files under ``root`` (recursive), tolerating a missing
    directory or permission errors — ``status`` must never crash."""
    if not root.is_dir():
        return 0
    total = 0
    try:
        for path in root.rglob("*"):
            try:
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    total += 1
            except OSError:
                continue
    except OSError:
        return total
    return total


def _job_or_error(slug: str | None, *, as_json: bool) -> tuple[job_config.Job | None, int]:
    """Resolve ``slug`` (falling back to the active job) or return an error code.

    Returns ``(job, 0)`` on success, ``(None, exit_code)`` on failure.
    """
    resolved = (slug or "").strip() or job_config.get_active_slug()
    if not resolved:
        return None, _emit_error_json(
            "no job specified and no active job; pass --job SLUG or run "
            "'cull jobs activate <slug>' first",
            code=EXIT_MISSING_JOB, as_json=as_json,
        )
    if not job_config.JOB_SLUG_RE.fullmatch(resolved):
        return None, _emit_error_json(
            f"invalid job slug: {resolved!r}",
            code=EXIT_BAD_ARGS, as_json=as_json,
        )
    job = job_config.get_job(resolved)
    if job is None:
        return None, _emit_error_json(
            f"unknown job: {resolved!r}",
            code=EXIT_MISSING_JOB, as_json=as_json,
        )
    return job, EXIT_OK


# ── jobs list ────────────────────────────────────────────────────────────────

def cmd_jobs_list(args: argparse.Namespace) -> int:
    jobs = job_config.list_jobs()
    active = job_config.get_active_slug()
    active_slugs = set(job_config.get_active_slugs())
    if getattr(args, "json", False):
        _emit(
            {
                "ok": True,
                "active": active,
                "active_slugs": sorted(active_slugs),
                "jobs": [
                    {
                        "slug": j.slug, "name": j.name, "status": j.status,
                        "subject": j.subject, "preset": j.preset,
                        "active": j.slug in active_slugs,
                    }
                    for j in jobs
                ],
            },
            as_json=True,
        )
        return EXIT_OK
    if not jobs:
        print("No jobs. Create one with: cull job create <slug> --subject \"...\"")
        return EXIT_OK
    print(f"{'':2}{'SLUG':<24} {'STATUS':<10} NAME")
    for job in jobs:
        marker = "* " if job.slug == active else "  "
        print(f"{marker}{job.slug:<24} {job.status:<10} {job.name}")
    return EXIT_OK


# ── jobs activate ────────────────────────────────────────────────────────────

def cmd_jobs_activate(args: argparse.Namespace) -> int:
    slug = args.slug
    try:
        job_config.activate(slug)
    except ValueError as exc:
        return _emit_error_json(str(exc), code=EXIT_MISSING_JOB,
                                as_json=getattr(args, "json", False))
    if getattr(args, "json", False):
        _emit({"ok": True, "activated": slug,
               "active_slugs": list(job_config.get_active_slugs())}, as_json=True)
    else:
        print(f"Activated job {slug!r} (env + categories projected).")
    return EXIT_OK


# ── job create ───────────────────────────────────────────────────────────────

def cmd_job_create(args: argparse.Namespace) -> int:
    name = args.slug
    try:
        job = job_config.create_job(
            name, subject=args.subject, preset=args.preset
        )
    except ValueError as exc:
        return _emit_error_json(str(exc), code=EXIT_BAD_ARGS,
                                as_json=getattr(args, "json", False))
    if getattr(args, "json", False):
        _emit({"ok": True, "job": {
            "slug": job.slug, "name": job.name, "preset": job.preset,
            "subject": job.subject, "status": job.status,
        }}, as_json=True)
    else:
        print(f"Created job {job.slug!r} (preset={job.preset!r}, "
              f"subject={job.subject!r}).")
    return EXIT_OK


# ── presets list ─────────────────────────────────────────────────────────────

def cmd_presets_list(args: argparse.Namespace) -> int:
    lib = job_config.list_presets()
    presets = lib.get("presets", {})
    default = lib.get("default", "")
    builtins = set(job_config.builtin_preset_names())
    if getattr(args, "json", False):
        _emit(
            {
                "ok": True, "default": default,
                "presets": [
                    {"name": n, "builtin": n in builtins,
                     "is_default": n == default}
                    for n in sorted(presets)
                ],
            },
            as_json=True,
        )
        return EXIT_OK
    if not presets:
        print("No presets.")
        return EXIT_OK
    print(f"{'':2}{'PRESET':<24} KIND")
    for name in sorted(presets):
        marker = "* " if name == default else "  "
        kind = "builtin" if name in builtins else "custom"
        print(f"{marker}{name:<24} {kind}")
    return EXIT_OK


# ── status ───────────────────────────────────────────────────────────────────

def cmd_status(args: argparse.Namespace) -> int:
    active = job_config.get_active_slug()
    index = job_config.get_index()
    queue = index.get("queue", []) or []
    active_all = list(job_config.get_active_slugs())
    if getattr(args, "json", False):
        payload: dict[str, Any] = {
            "ok": True,
            "active_slug": active,
            "active_slugs": active_all,
            "queue": queue,
            "data_dir": str(paths.base_dir()),
        }
        if active:
            payload["counts"] = {
                "queue": _count_images(paths.queue_dir(active)),
                "sorted": _count_images(paths.sorted_dir(active)),
            }
        _emit(payload, as_json=True)
        return EXIT_OK
    print(f"Active job : {active or '(none)'}")
    print(f"Queued     : {', '.join(queue) if queue else '(empty)'}")
    print(f"Data dir   : {paths.base_dir()}")
    if active:
        q = _count_images(paths.queue_dir(active))
        s = _count_images(paths.sorted_dir(active))
        print(f"Queue images (slug={active})  : {q}")
        print(f"Sorted images (slug={active}) : {s}")
    return EXIT_OK


# ── run (lazy import of the supervisor) ──────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    """Start the supervisor. ``run_pipeline`` is imported HERE (not at module
    top) so the CLI module stays cheap to import and tests never spawn it."""
    try:
        import run_pipeline  # lazy: only when actually running the pipeline
    except ImportError as exc:  # pragma: no cover - defensive
        return _emit_error_json(
            f"could not import the supervisor (run_pipeline): {exc}",
            code=EXIT_SUBPROCESS_FAIL, as_json=getattr(args, "json", False),
        )
    run_pipeline.main()
    return EXIT_OK


# ── export (legacy profile export) ───────────────────────────────────────────

def cmd_export(args: argparse.Namespace) -> int:
    try:
        import export_profiles  # optional, may not be installed
    except ImportError:
        return _emit_error_json(
            "export module not available (export_profiles is not installed); "
            "nothing was exported.",
            code=EXIT_SUBPROCESS_FAIL, as_json=getattr(args, "json", False),
        )
    export_dataset = getattr(export_profiles, "export_dataset", None)
    if export_dataset is None:
        return _emit_error_json(
            "export module is present but exposes no export_dataset(); "
            "nothing was exported.",
            code=EXIT_SUBPROCESS_FAIL, as_json=getattr(args, "json", False),
        )
    try:
        summary = export_dataset(args.slug, args.profile, Path(args.out))
    except Exception as exc:  # surface a clean message, not a traceback
        return _emit_error_json(
            f"export failed: {exc}",
            code=EXIT_SUBPROCESS_FAIL, as_json=getattr(args, "json", False),
        )
    if getattr(args, "json", False):
        payload = {"ok": True, "slug": args.slug, "profile": args.profile,
                   "out": str(args.out)}
        if isinstance(summary, dict):
            payload["summary"] = summary
        _emit(payload, as_json=True)
    else:
        count = summary.get("sample_count") if isinstance(summary, dict) else None
        suffix = f" ({count} sample(s))" if count is not None else ""
        print(f"Exported job {args.slug!r} (profile={args.profile!r}) -> "
              f"{args.out}{suffix}")
    return EXIT_OK


# ── stats ────────────────────────────────────────────────────────────────────

def _score_bucket(score: Any) -> str:
    """Group a 0-100 score into 10-point buckets for a compact distribution."""
    if score is None:
        return "unknown"
    try:
        s = int(score)
    except (TypeError, ValueError):
        return "unknown"
    s = max(0, min(100, s))
    lo = (s // 10) * 10
    hi = lo + 9 if lo < 100 else 100
    return f"{lo}-{hi}"


def _collect_stats(slug: str) -> dict[str, Any]:
    """Walk the sorted tree for ``slug`` and produce counts + score distribution.

    Uses direct filesystem iteration (bounded to one slug's tree) so it works
    without the dashboard's SQLite index. This is a read-only aggregation — no
    files are moved or rewritten.
    """
    sorted_root = paths.sorted_dir(slug)
    queue_root = paths.queue_dir(slug)
    counts: dict[str, int] = {}
    ovr_dist: dict[str, int] = {}
    rel_dist: dict[str, int] = {}
    total_sorted = 0
    nsfw = 0
    watermark = 0
    with_prompt = 0
    if sorted_root.is_dir():
        try:
            for category_dir in sorted_root.iterdir():
                if not category_dir.is_dir():
                    continue
                cat = category_dir.name
                cat_count = 0
                for path in category_dir.rglob("*"):
                    if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
                        continue
                    cat_count += 1
                    total_sorted += 1
                    meta = path.with_name(f"{path.stem}.vision.json")
                    if meta.exists():
                        try:
                            payload = _json.loads(meta.read_text(encoding="utf-8"))
                        except (OSError, _json.JSONDecodeError):
                            payload = {}
                        ovr = payload.get("OVR_Quality_Score")
                        rel = payload.get("REL_Quality_Score")
                        ovr_dist[_score_bucket(ovr)] = \
                            ovr_dist.get(_score_bucket(ovr), 0) + 1
                        rel_dist[_score_bucket(rel)] = \
                            rel_dist.get(_score_bucket(rel), 0) + 1
                        if payload.get("nsfw"):
                            nsfw += 1
                        if payload.get("watermark"):
                            watermark += 1
                    if path.with_suffix(".txt").exists():
                        with_prompt += 1
                counts[cat] = cat_count
        except OSError:
            pass
    return {
        "slug": slug,
        "queue_count": _count_images(queue_root),
        "sorted_count": total_sorted,
        "counts_by_category": counts,
        "score_distribution": {
            "ovr": ovr_dist,
            "rel": rel_dist,
        },
        "nsfw_count": nsfw,
        "watermark_count": watermark,
        "with_prompt": with_prompt,
    }


def cmd_stats(args: argparse.Namespace) -> int:
    job, code = _job_or_error(getattr(args, "job", None),
                              as_json=getattr(args, "json", False))
    if job is None:
        return code
    data = _collect_stats(job.slug)
    if getattr(args, "json", False):
        _emit({"ok": True, **data}, as_json=True)
        return EXIT_OK
    print(f"Job       : {job.slug}")
    print(f"Queue     : {data['queue_count']}")
    print(f"Sorted    : {data['sorted_count']}")
    if data["counts_by_category"]:
        print("By category:")
        for cat, n in sorted(data["counts_by_category"].items(),
                             key=lambda kv: (-kv[1], kv[0])):
            print(f"  {cat:<20} {n}")
    if data["score_distribution"]["ovr"]:
        print("OVR distribution:")
        for bucket, n in sorted(data["score_distribution"]["ovr"].items()):
            print(f"  {bucket:<10} {n}")
    if data["score_distribution"]["rel"]:
        print("REL distribution:")
        for bucket, n in sorted(data["score_distribution"]["rel"].items()):
            print(f"  {bucket:<10} {n}")
    if data["nsfw_count"] or data["watermark_count"]:
        print(f"NSFW      : {data['nsfw_count']}")
        print(f"Watermark : {data['watermark_count']}")
    return EXIT_OK


# ── gallery sample ──────────────────────────────────────────────────────────

def _load_vision_record(image_path: Path) -> dict[str, Any]:
    meta = image_path.with_name(f"{image_path.stem}.vision.json")
    payload: dict[str, Any] = {}
    if meta.exists():
        try:
            payload = _json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError):
            payload = {}
    return payload


def _sample_gallery(slug: str, *, category: str | None, n: int) -> list[dict[str, Any]]:
    """Return up to ``n`` records from ``sorted/<slug>[/<category>]`` (random)."""
    sorted_root = paths.sorted_dir(slug)
    if not sorted_root.is_dir():
        return []
    candidates: list[Path] = []
    if category:
        # Only look inside the requested category subtree; never traverse
        # anywhere else. ``paths.validate_slug`` already sanitised ``slug``.
        root = sorted_root / category
        if not root.is_dir():
            return []
        try:
            resolved = root.resolve()
            base = sorted_root.resolve()
            if base not in resolved.parents and resolved != base:
                return []
        except OSError:
            return []
        walk_roots = [root]
    else:
        walk_roots = [d for d in sorted_root.iterdir() if d.is_dir()] \
            if sorted_root.is_dir() else []
    for root in walk_roots:
        try:
            for p in root.rglob("*"):
                try:
                    if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                        candidates.append(p)
                except OSError:
                    continue
        except OSError:
            continue
    if not candidates:
        return []
    n = max(1, min(int(n), len(candidates)))
    picked = random.sample(candidates, n)
    out: list[dict[str, Any]] = []
    for path in picked:
        rec = _load_vision_record(path)
        # rel_to_sorted keeps output stable even if the user later moves the
        # sorted root; the caller can rejoin against ``paths.sorted_dir(slug)``.
        try:
            rel = path.relative_to(sorted_root)
        except ValueError:
            rel = path
        prompt_path = path.with_suffix(".txt")
        prompt = ""
        if prompt_path.exists():
            try:
                prompt = prompt_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                prompt = ""
        out.append({
            "path": str(path),
            "relative_path": str(rel),
            "category": rel.parts[0] if rel.parts else "",
            "ovr": rec.get("OVR_Quality_Score"),
            "rel": rec.get("REL_Quality_Score"),
            "quality_score": rec.get("quality_score"),
            "nsfw": bool(rec.get("nsfw")),
            "watermark": bool(rec.get("watermark")),
            "prompt": prompt,
            "vision_json": rec,
        })
    return out


def cmd_gallery_sample(args: argparse.Namespace) -> int:
    job, code = _job_or_error(getattr(args, "job", None),
                              as_json=getattr(args, "json", False))
    if job is None:
        return code
    category = (args.category or "").strip() or None
    # Category name must not contain a path separator (defense-in-depth on top
    # of ``paths.validate_slug`` running on the slug).
    if category and re.search(r"[\\/]|\.\.", category):
        return _emit_error_json(
            f"invalid category name: {category!r}",
            code=EXIT_BAD_ARGS, as_json=getattr(args, "json", False),
        )
    try:
        n = int(args.n)
    except (TypeError, ValueError):
        return _emit_error_json(
            f"invalid --n value: {args.n!r}",
            code=EXIT_BAD_ARGS, as_json=getattr(args, "json", False),
        )
    if n <= 0:
        return _emit_error_json("--n must be positive",
                                code=EXIT_BAD_ARGS,
                                as_json=getattr(args, "json", False))
    samples = _sample_gallery(job.slug, category=category, n=n)
    if getattr(args, "json", False):
        _emit({"ok": True, "slug": job.slug, "category": category,
               "requested": n, "returned": len(samples),
               "samples": samples}, as_json=True)
        return EXIT_OK
    if not samples:
        print(f"No sorted samples for slug={job.slug}"
              + (f" category={category}" if category else ""))
        return EXIT_OK
    for rec in samples:
        print(f"{rec['relative_path']:<60}  ovr={rec['ovr']}  rel={rec['rel']}")
    return EXIT_OK


# ── scoring set ──────────────────────────────────────────────────────────────

def cmd_scoring_set(args: argparse.Namespace) -> int:
    job, code = _job_or_error(getattr(args, "job", None),
                              as_json=getattr(args, "json", False))
    if job is None:
        return code
    updates: dict[str, Any] = {}
    if args.min_ovr is not None:
        if not 0 <= int(args.min_ovr) <= 100:
            return _emit_error_json("--min-ovr must be 0-100",
                                    code=EXIT_BAD_ARGS,
                                    as_json=getattr(args, "json", False))
        job = job_config.set_override(job, "scoring.ovr_min", int(args.min_ovr))
        updates["scoring.ovr_min"] = int(args.min_ovr)
    if args.min_rel is not None:
        if not 0 <= int(args.min_rel) <= 100:
            return _emit_error_json("--min-rel must be 0-100",
                                    code=EXIT_BAD_ARGS,
                                    as_json=getattr(args, "json", False))
        job = job_config.set_override(job, "scoring.rel_min", int(args.min_rel))
        updates["scoring.rel_min"] = int(args.min_rel)
    if args.require_prompt is not None:
        job = job_config.set_override(
            job, "topic_filters.require_prompt", bool(args.require_prompt)
        )
        updates["topic_filters.require_prompt"] = bool(args.require_prompt)
    if not updates:
        return _emit_error_json(
            "no changes: pass at least one of --min-ovr / --min-rel / "
            "--require-prompt",
            code=EXIT_BAD_ARGS, as_json=getattr(args, "json", False),
        )
    saved = job_config.save_job(job)
    if getattr(args, "json", False):
        _emit({"ok": True, "slug": saved.slug, "updates": updates,
               "overrides": saved.overrides}, as_json=True)
    else:
        for k, v in updates.items():
            print(f"set {k} = {v}")
    return EXIT_OK


# ── scrapers list / add-url / toggle ─────────────────────────────────────────

def cmd_scrapers_list(args: argparse.Namespace) -> int:
    job, code = _job_or_error(getattr(args, "job", None),
                              as_json=getattr(args, "json", False))
    if job is None:
        return code
    eff = job_config.effective_config(job)
    scrapers = eff.get("scrapers", {}) or {}
    enabled = scrapers.get("enabled", {}) or {}
    rows = [{"name": name, "enabled": bool(enabled.get(name, True))}
            for name in job_config.SCRAPER_NAMES]
    gd = scrapers.get("gallery_dl") or {}
    yt = scrapers.get("yt_dlp") or {}
    if getattr(args, "json", False):
        _emit({
            "ok": True, "slug": job.slug,
            "scrapers": rows,
            "gallery_dl": {
                "enabled": bool(gd.get("enabled", False)),
                "url_count": len(gd.get("urls") or []),
                "limit_per_url": int(gd.get("limit_per_url", 200) or 200),
            },
            "yt_dlp": {
                "enabled": bool(yt.get("enabled", False)),
                "url_count": len(yt.get("urls") or []),
            },
            "local_imports": [
                {"name": li.get("name", ""), "enabled": bool(li.get("enabled")),
                 "dir": li.get("dir", "")}
                for li in scrapers.get("local_imports") or []
                if isinstance(li, dict)
            ],
            "x_accounts": list(scrapers.get("x_accounts") or []),
            "reddit_subreddits": list(scrapers.get("reddit_subreddits") or []),
        }, as_json=True)
        return EXIT_OK
    print(f"Scrapers for job {job.slug!r}:")
    for row in rows:
        state = "on " if row["enabled"] else "off"
        print(f"  [{state}] {row['name']}")
    print(f"  gallery-dl: {'on' if gd.get('enabled') else 'off'} "
          f"({len(gd.get('urls') or [])} URL(s))")
    return EXIT_OK


_SCRAPERS_WITH_URLS: dict[str, str] = {
    # user-facing name -> effective_config().scrapers.<key>.urls list
    "gallery_dl": "gallery_dl",
    "yt_dlp": "yt_dlp",
}


def cmd_scrapers_add_url(args: argparse.Namespace) -> int:
    job, code = _job_or_error(getattr(args, "job", None),
                              as_json=getattr(args, "json", False))
    if job is None:
        return code
    key = _SCRAPERS_WITH_URLS.get((args.source or "").lower())
    if key is None:
        return _emit_error_json(
            f"--source must be one of {sorted(_SCRAPERS_WITH_URLS)}",
            code=EXIT_BAD_ARGS, as_json=getattr(args, "json", False),
        )
    url = (args.url or "").strip()
    if not url:
        return _emit_error_json("--url is required",
                                code=EXIT_BAD_ARGS,
                                as_json=getattr(args, "json", False))
    # SSRF guard: the CLI is a trusted-input surface but a scheduled/looped
    # agent could still be tricked into feeding a localhost or link-local URL
    # into a scraper subprocess. Block obviously local/private hosts here.
    try:
        from scheduler import _is_public_http_url  # local import (avoid cycle)
    except ImportError:  # pragma: no cover - scheduler always importable
        _is_public_http_url = None  # type: ignore[assignment]
    if _is_public_http_url is not None and not _is_public_http_url(url):
        return _emit_error_json(
            "URL is not a public http(s) URL (rejected by SSRF guard)",
            code=EXIT_BAD_ARGS, as_json=getattr(args, "json", False),
        )
    eff = job_config.effective_config(job)
    existing = list((eff.get("scrapers", {}).get(key) or {}).get("urls") or [])
    if url in existing:
        if getattr(args, "json", False):
            _emit({"ok": True, "slug": job.slug, "source": key,
                   "url": url, "changed": False, "urls": existing}, as_json=True)
        else:
            print(f"URL already present in {key}.urls")
        return EXIT_OK
    updated = [*existing, url]
    job = job_config.set_override(job, f"scrapers.{key}.urls", updated)
    # Turn the source on when the user is adding the first URL — otherwise the
    # scraper stays gated behind its ``enabled`` flag.
    if not existing:
        job = job_config.set_override(job, f"scrapers.{key}.enabled", True)
    saved = job_config.save_job(job)
    if getattr(args, "json", False):
        _emit({"ok": True, "slug": saved.slug, "source": key,
               "url": url, "changed": True, "urls": updated}, as_json=True)
    else:
        print(f"added URL to {key}.urls ({len(updated)} total)")
    return EXIT_OK


def cmd_scrapers_toggle(args: argparse.Namespace) -> int:
    job, code = _job_or_error(getattr(args, "job", None),
                              as_json=getattr(args, "json", False))
    if job is None:
        return code
    name = (args.name or "").strip()
    if name not in job_config.SCRAPER_NAMES:
        return _emit_error_json(
            f"--name must be one of {list(job_config.SCRAPER_NAMES)}",
            code=EXIT_BAD_ARGS, as_json=getattr(args, "json", False),
        )
    # NOTE: several scraper names contain a ``.`` (``X.com``, ``Civitai-Com``…)
    # which would collide with ``set_override``'s dotted-path splitter — instead
    # of using ``scrapers.enabled.<name>``, read the current effective ``enabled``
    # map, mutate that key, and re-project the whole dict as one override.
    eff = job_config.effective_config(job)
    enabled = dict((eff.get("scrapers") or {}).get("enabled") or {})
    enabled[name] = bool(args.enabled)
    job = job_config.set_override(job, "scrapers.enabled", enabled)
    saved = job_config.save_job(job)
    if getattr(args, "json", False):
        _emit({"ok": True, "slug": saved.slug, "scraper": name,
               "enabled": bool(args.enabled)}, as_json=True)
    else:
        print(f"scraper {name} -> {'enabled' if args.enabled else 'disabled'}")
    return EXIT_OK


# ── config show ──────────────────────────────────────────────────────────────

def _mask_secrets(node: Any) -> Any:
    """Recursively mask fields whose keys look like a credential.

    We NEVER surface a raw ``api_key`` / ``token`` / ``cookies`` value in
    ``config show`` — a JSON caller would otherwise scrape them into a log.
    """
    sensitive = ("api_key", "apikey", "token", "cookies", "cookie", "secret",
                 "password", "authorization")
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            lk = str(k).lower()
            if any(s in lk for s in sensitive) and isinstance(v, str) and v:
                out[k] = "***"
            else:
                out[k] = _mask_secrets(v)
        return out
    if isinstance(node, list):
        return [_mask_secrets(item) for item in node]
    return node


def cmd_config_show(args: argparse.Namespace) -> int:
    job, code = _job_or_error(getattr(args, "job", None),
                              as_json=getattr(args, "json", False))
    if job is None:
        return code
    eff = job_config.effective_config(job)
    masked_eff = _mask_secrets(eff)
    masked_ov = _mask_secrets(job.overrides or {})
    if getattr(args, "json", False):
        _emit({
            "ok": True,
            "slug": job.slug, "name": job.name, "preset": job.preset,
            "subject": job.subject, "status": job.status,
            "overrides": masked_ov,
            "effective": masked_eff,
        }, as_json=True)
        return EXIT_OK
    print(f"Job     : {job.slug} ({job.name})")
    print(f"Preset  : {job.preset}")
    print(f"Subject : {job.subject}")
    print(f"Status  : {job.status}")
    scoring = eff.get("scoring") or {}
    print(f"Scoring : ovr_min={scoring.get('ovr_min', 0)} "
          f"rel_min={scoring.get('rel_min', 0)}")
    tf = eff.get("topic") or {}
    print(f"Prompts : require_prompt={tf.get('require_prompt', True)} "
          f"min_length={tf.get('min_prompt_length', 0)}")
    return EXIT_OK


# ── jobs watch (predicate-driven wait loop) ──────────────────────────────────

_PREDICATE_RE = re.compile(
    r"^\s*(?P<key>sorted-count|queue-count|active-slug|elapsed)"
    r"\s*(?P<op>>=|<=|=|==)\s*(?P<val>.+?)\s*$"
)


def _parse_predicate(expr: str) -> Callable[[dict[str, Any]], bool]:
    """Compile a ``--until`` predicate into a callable.

    Supported grammar (see AGENTS.md § Deciding when to stop):

      sorted-count>=N
      queue-count<=N
      active-slug=X
      elapsed>=Ns
    """
    m = _PREDICATE_RE.match(str(expr or ""))
    if not m:
        raise argparse.ArgumentTypeError(
            f"invalid --until predicate: {expr!r}. Grammar: "
            "sorted-count>=N | queue-count<=N | active-slug=X | elapsed>=Ns"
        )
    key = m.group("key")
    op = m.group("op")
    raw = m.group("val").strip()
    if key in ("sorted-count", "queue-count"):
        if op not in (">=", "<=", "=", "=="):
            raise argparse.ArgumentTypeError(
                f"operator {op!r} not supported for {key}"
            )
        try:
            target = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"expected integer for {key}, got {raw!r}"
            ) from exc

        def check_count(state: dict[str, Any], *, k=key, o=op, t=target) -> bool:
            v = int(state.get(k, 0) or 0)
            return {">=": v >= t, "<=": v <= t, "=": v == t, "==": v == t}[o]

        return check_count
    if key == "active-slug":
        if op not in ("=", "=="):
            raise argparse.ArgumentTypeError(
                "active-slug only supports the = operator"
            )

        def check_active(state: dict[str, Any], *, expected=raw) -> bool:
            return str(state.get("active-slug") or "") == expected

        return check_active
    # elapsed
    if op != ">=":
        raise argparse.ArgumentTypeError("elapsed only supports >= operator")
    if not raw.endswith("s"):
        raise argparse.ArgumentTypeError(
            f"elapsed value must be in seconds (e.g. 60s), got {raw!r}"
        )
    try:
        seconds = float(raw[:-1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid elapsed seconds: {raw!r}"
        ) from exc

    def check_elapsed(state: dict[str, Any], *, s=seconds) -> bool:
        return float(state.get("elapsed", 0.0)) >= s

    return check_elapsed


def _watch_state(slug: str, *, elapsed: float) -> dict[str, Any]:
    return {
        "slug": slug,
        "sorted-count": _count_images(paths.sorted_dir(slug)),
        "queue-count": _count_images(paths.queue_dir(slug)),
        "active-slug": job_config.get_active_slug() or "",
        "elapsed": elapsed,
    }


def cmd_jobs_watch(args: argparse.Namespace) -> int:
    slug = (args.slug or "").strip()
    as_json = getattr(args, "json", False)
    if not slug or not job_config.JOB_SLUG_RE.fullmatch(slug):
        return _emit_error_json(
            f"invalid --slug: {slug!r}",
            code=EXIT_BAD_ARGS, as_json=as_json,
        )
    if job_config.get_job(slug) is None:
        return _emit_error_json(
            f"unknown job: {slug!r}",
            code=EXIT_MISSING_JOB, as_json=as_json,
        )
    predicate: Callable[[dict[str, Any]], bool] | None = None
    if args.until:
        try:
            predicate = _parse_predicate(args.until)
        except argparse.ArgumentTypeError as exc:
            return _emit_error_json(str(exc), code=EXIT_BAD_ARGS, as_json=as_json)
    try:
        interval = max(0.1, float(args.interval))
    except (TypeError, ValueError):
        return _emit_error_json(
            f"invalid --interval: {args.interval!r}",
            code=EXIT_BAD_ARGS, as_json=as_json,
        )
    try:
        timeout = float(args.timeout) if args.timeout else 0.0
    except (TypeError, ValueError):
        return _emit_error_json(
            f"invalid --timeout: {args.timeout!r}",
            code=EXIT_BAD_ARGS, as_json=as_json,
        )
    started = time.monotonic()
    ticks = 0
    last_state: dict[str, Any] = _watch_state(slug, elapsed=0.0)
    # Non-blocking, single-tick mode when neither --until nor --timeout is set:
    # emit one state read and exit 0 so a scripted caller can poll.
    if predicate is None and timeout <= 0:
        if as_json:
            _emit({"ok": True, "condition_met": None, "state": last_state},
                  as_json=True)
        else:
            print(f"slug={slug} sorted={last_state['sorted-count']} "
                  f"queue={last_state['queue-count']} "
                  f"active={last_state['active-slug']}")
        return EXIT_OK
    while True:
        elapsed = time.monotonic() - started
        last_state = _watch_state(slug, elapsed=elapsed)
        ticks += 1
        if predicate is not None and predicate(last_state):
            if as_json:
                _emit({"ok": True, "condition_met": True, "ticks": ticks,
                       "elapsed": elapsed, "state": last_state}, as_json=True)
            else:
                print(f"condition met after {elapsed:.1f}s ({ticks} tick(s)): "
                      f"{args.until}")
            return EXIT_OK
        if timeout > 0 and elapsed >= timeout:
            if as_json:
                _emit({"ok": False, "condition_met": False, "ticks": ticks,
                       "elapsed": elapsed, "state": last_state,
                       "error": "watch timed out",
                       "exit_code": EXIT_WATCH_TIMEOUT}, as_json=True)
            else:
                _err(f"watch timed out after {elapsed:.1f}s")
            return EXIT_WATCH_TIMEOUT
        if not as_json:
            print(f"[{elapsed:6.1f}s] sorted={last_state['sorted-count']} "
                  f"queue={last_state['queue-count']} "
                  f"active={last_state['active-slug']}", flush=True)
        time.sleep(interval)


# ── export kohya + export hf (convenience wrappers) ──────────────────────────

def cmd_export_kohya(args: argparse.Namespace) -> int:
    """Export a job's sorted set as a Kohya training folder."""
    as_json = getattr(args, "json", False)
    job, code = _job_or_error(getattr(args, "job", None), as_json=as_json)
    if job is None:
        return code
    try:
        import export_profiles
    except ImportError:
        return _emit_error_json(
            "export module not available (export_profiles is not installed)",
            code=EXIT_SUBPROCESS_FAIL, as_json=as_json,
        )
    export_dataset = getattr(export_profiles, "export_dataset", None)
    if export_dataset is None:
        return _emit_error_json(
            "export module is present but exposes no export_dataset()",
            code=EXIT_SUBPROCESS_FAIL, as_json=as_json,
        )
    try:
        summary = export_dataset(job.slug, "kohya", Path(args.out))
    except Exception as exc:
        return _emit_error_json(
            f"kohya export failed: {exc}",
            code=EXIT_SUBPROCESS_FAIL, as_json=as_json,
        )
    if as_json:
        _emit({"ok": True, "slug": job.slug, "profile": "kohya",
               "out": str(args.out),
               "summary": summary if isinstance(summary, dict) else {}},
              as_json=True)
    else:
        count = summary.get("sample_count") if isinstance(summary, dict) else None
        suffix = f" ({count} sample(s))" if count is not None else ""
        print(f"Kohya export {job.slug!r} -> {args.out}{suffix}")
    return EXIT_OK


def cmd_export_hf(args: argparse.Namespace) -> int:
    """Push the job's curated set to a private HuggingFace dataset repo."""
    as_json = getattr(args, "json", False)
    job, code = _job_or_error(getattr(args, "job", None), as_json=as_json)
    if job is None:
        return code
    if not args.repo or "/" not in args.repo:
        return _emit_error_json(
            "--repo must be user/name",
            code=EXIT_BAD_ARGS, as_json=as_json,
        )
    try:
        import hf_export
    except ImportError:
        return _emit_error_json(
            "hf_export module not available",
            code=EXIT_SUBPROCESS_FAIL, as_json=as_json,
        )
    push_to_hf = getattr(hf_export, "push_to_hf", None)
    if push_to_hf is None:
        return _emit_error_json(
            "hf_export.push_to_hf is not available",
            code=EXIT_SUBPROCESS_FAIL, as_json=as_json,
        )
    try:
        result = push_to_hf(
            job.slug, args.repo,
            private=not args.public,
            include_video=bool(args.include_video),
        )
    except Exception as exc:
        return _emit_error_json(
            f"HuggingFace push failed: {exc}",
            code=EXIT_SUBPROCESS_FAIL, as_json=as_json,
        )
    if as_json:
        payload = {"ok": True, "slug": job.slug, "repo": args.repo}
        if isinstance(result, dict):
            payload["result"] = _mask_secrets(result)
        _emit(payload, as_json=True)
    else:
        print(f"HuggingFace push {job.slug!r} -> {args.repo}: OK")
    return EXIT_OK


# ── parser ───────────────────────────────────────────────────────────────────

def _build_export_kohya_parser() -> argparse.ArgumentParser:
    """Standalone parser for ``cull export kohya …``. Dispatched pre-argparse
    from :func:`main` so it does not clash with the legacy positional form of
    ``cull export`` (argparse can't cleanly mix a sub-parser with a trailing
    optional positional at the same level)."""
    p = argparse.ArgumentParser(
        prog="cull export kohya",
        description="Export a job's sorted set as a Kohya training folder.",
    )
    p.add_argument("--job", default=None,
                   help="Job slug (default: active job).")
    p.add_argument("--out", required=True, help="Output directory.")
    _add_json_flag(p)
    return p


def _build_export_hf_parser() -> argparse.ArgumentParser:
    """Standalone parser for ``cull export hf …`` (see kohya sibling)."""
    p = argparse.ArgumentParser(
        prog="cull export hf",
        description="Push the job's curated set to a HuggingFace dataset repo.",
    )
    p.add_argument("--job", default=None,
                   help="Job slug (default: active job).")
    p.add_argument("--repo", required=True,
                   help="Target HuggingFace dataset repo (user/name).")
    p.add_argument("--public", action="store_true",
                   help="Create the repo as public (default: private).")
    p.add_argument("--include-video", action="store_true",
                   help="Include video files in the push.")
    _add_json_flag(p)
    return p


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON on stdout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cull",
        description="Headless control for cull (no dashboard) - drive jobs, "
                    "presets, scrapers, scoring, stats and exports from the "
                    "command line. Every subcommand supports --json.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # jobs (group: list / activate / watch)
    p_jobs = sub.add_parser("jobs", help="List, activate, or watch jobs.")
    jobs_sub = p_jobs.add_subparsers(dest="jobs_command", metavar="<subcommand>")

    p_jobs_list = jobs_sub.add_parser("list", help="List all jobs.")
    _add_json_flag(p_jobs_list)
    p_jobs_list.set_defaults(func=cmd_jobs_list)

    p_jobs_activate = jobs_sub.add_parser(
        "activate", help="Activate a job (projects env + categories).")
    p_jobs_activate.add_argument("slug", help="Job slug to activate.")
    _add_json_flag(p_jobs_activate)
    p_jobs_activate.set_defaults(func=cmd_jobs_activate)

    p_jobs_watch = jobs_sub.add_parser(
        "watch", help="Poll a job's counts until a predicate is met.")
    p_jobs_watch.add_argument("--slug", required=True, help="Job slug to watch.")
    p_jobs_watch.add_argument(
        "--until", default=None,
        help='Stop-condition predicate. Examples: "sorted-count>=500", '
             '"queue-count<=0", "active-slug=my_job", "elapsed>=300s".')
    p_jobs_watch.add_argument("--interval", type=float, default=5.0,
                              help="Poll interval in seconds (default 5).")
    p_jobs_watch.add_argument("--timeout", type=float, default=0.0,
                              help="Max seconds to wait (0 = forever).")
    _add_json_flag(p_jobs_watch)
    p_jobs_watch.set_defaults(func=cmd_jobs_watch)

    p_jobs.set_defaults(func=lambda a: _require_subcommand(p_jobs))

    # job create
    p_job = sub.add_parser("job", help="Create or manage a single job.")
    job_sub = p_job.add_subparsers(dest="job_command", metavar="<subcommand>")
    p_job_create = job_sub.add_parser("create", help="Create a job from a preset.")
    p_job_create.add_argument("slug", help="Job name/slug (a-z, 0-9, _).")
    p_job_create.add_argument(
        "--preset", default=None,
        help="Preset to inherit (default: the library default preset).")
    p_job_create.add_argument(
        "--subject", default=None,
        help="Subject / topic line for the job (defaults to the name).")
    _add_json_flag(p_job_create)
    p_job_create.set_defaults(func=cmd_job_create)
    p_job.set_defaults(func=lambda a: _require_subcommand(p_job))

    # presets (group: list)
    p_presets = sub.add_parser("presets", help="List presets.")
    presets_sub = p_presets.add_subparsers(
        dest="presets_command", metavar="<subcommand>")
    p_presets_list = presets_sub.add_parser("list", help="List preset names.")
    _add_json_flag(p_presets_list)
    p_presets_list.set_defaults(func=cmd_presets_list)
    p_presets.set_defaults(func=lambda a: _require_subcommand(p_presets))

    # status
    p_status = sub.add_parser(
        "status", help="Show the active job + basic queue/sorted counts.")
    _add_json_flag(p_status)
    p_status.set_defaults(func=cmd_status)

    # stats
    p_stats = sub.add_parser(
        "stats", help="Category counts + score distribution for a job.")
    p_stats.add_argument("--job", default=None,
                         help="Job slug (default: active job).")
    _add_json_flag(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    # gallery sample
    p_gallery = sub.add_parser(
        "gallery", help="Sample the classified gallery.")
    gal_sub = p_gallery.add_subparsers(dest="gallery_command",
                                       metavar="<subcommand>")
    p_gal_sample = gal_sub.add_parser("sample", help="Random sample of records.")
    p_gal_sample.add_argument("--job", required=True, help="Job slug.")
    p_gal_sample.add_argument("--category", default=None,
                              help="Restrict to this category folder.")
    p_gal_sample.add_argument("--n", type=int, default=20,
                              help="Sample size (default 20).")
    _add_json_flag(p_gal_sample)
    p_gal_sample.set_defaults(func=cmd_gallery_sample)
    p_gallery.set_defaults(func=lambda a: _require_subcommand(p_gallery))

    # scoring set
    p_scoring = sub.add_parser("scoring", help="Adjust score gates for a job.")
    sc_sub = p_scoring.add_subparsers(dest="scoring_command",
                                      metavar="<subcommand>")
    p_sc_set = sc_sub.add_parser("set", help="Write scoring overrides.")
    p_sc_set.add_argument("--job", default=None,
                          help="Job slug (default: active job).")
    p_sc_set.add_argument("--min-ovr", type=int, default=None,
                          help="Minimum OVR score (0-100).")
    p_sc_set.add_argument("--min-rel", type=int, default=None,
                          help="Minimum REL score (0-100).")
    p_sc_set.add_argument("--require-prompt", type=_bool_arg, default=None,
                          help="Require a scraped prompt (true/false).")
    _add_json_flag(p_sc_set)
    p_sc_set.set_defaults(func=cmd_scoring_set)
    p_scoring.set_defaults(func=lambda a: _require_subcommand(p_scoring))

    # scrapers list / add-url / toggle
    p_scrapers = sub.add_parser("scrapers", help="List / configure scrapers.")
    scr_sub = p_scrapers.add_subparsers(dest="scrapers_command",
                                        metavar="<subcommand>")
    p_scr_list = scr_sub.add_parser("list", help="Show configured scrapers.")
    p_scr_list.add_argument("--job", default=None,
                            help="Job slug (default: active job).")
    _add_json_flag(p_scr_list)
    p_scr_list.set_defaults(func=cmd_scrapers_list)

    p_scr_add = scr_sub.add_parser("add-url",
                                   help="Append a URL to a URL-driven scraper.")
    p_scr_add.add_argument("--job", default=None,
                           help="Job slug (default: active job).")
    p_scr_add.add_argument("--source", required=True,
                           choices=sorted(_SCRAPERS_WITH_URLS),
                           help="URL-driven scraper key.")
    p_scr_add.add_argument("--url", required=True, help="Public http(s) URL.")
    _add_json_flag(p_scr_add)
    p_scr_add.set_defaults(func=cmd_scrapers_add_url)

    p_scr_tog = scr_sub.add_parser("toggle",
                                   help="Enable/disable a scraper for the job.")
    p_scr_tog.add_argument("--job", default=None,
                           help="Job slug (default: active job).")
    p_scr_tog.add_argument("--name", required=True,
                           help=f"Scraper name (one of "
                                f"{list(job_config.SCRAPER_NAMES)}).")
    p_scr_tog.add_argument("--enabled", type=_bool_arg, required=True,
                           help="true / false.")
    _add_json_flag(p_scr_tog)
    p_scr_tog.set_defaults(func=cmd_scrapers_toggle)

    p_scrapers.set_defaults(func=lambda a: _require_subcommand(p_scrapers))

    # config show
    p_config = sub.add_parser("config", help="Inspect job configuration.")
    cfg_sub = p_config.add_subparsers(dest="config_command",
                                      metavar="<subcommand>")
    p_cfg_show = cfg_sub.add_parser("show",
                                    help="Print the effective config for a job.")
    p_cfg_show.add_argument("--job", default=None,
                            help="Job slug (default: active job).")
    _add_json_flag(p_cfg_show)
    p_cfg_show.set_defaults(func=cmd_config_show)
    p_config.set_defaults(func=lambda a: _require_subcommand(p_config))

    # run
    p_run = sub.add_parser("run", help="Start the supervisor (runs the active job).")
    _add_json_flag(p_run)
    p_run.set_defaults(func=cmd_run)

    # export — the legacy positional form (``cull export <slug> --profile P
    # --out D``) stays intact so shipped docs/tests keep working; the two new
    # agent-friendly variants ``cull export kohya …`` / ``cull export hf …`` are
    # dispatched from :func:`main` via their own standalone parsers (argparse
    # cannot cleanly mix a subparser with a trailing optional positional at the
    # same level, and ``help=argparse.SUPPRESS`` leaks the hidden command name
    # into the top-level ``--help`` output).
    p_export = sub.add_parser(
        "export", help="Export a job's dataset (legacy: --profile P --out D). "
                       "See 'cull export kohya' / 'cull export hf'.")
    p_export.add_argument("slug", help="Job slug to export.")
    p_export.add_argument("--profile", required=True, help="Export profile name.")
    p_export.add_argument("--out", required=True, help="Output directory.")
    _add_json_flag(p_export)
    p_export.set_defaults(func=cmd_export)

    return parser


def _require_subcommand(group_parser: argparse.ArgumentParser) -> int:
    """A bare group (e.g. ``cull jobs``) with no subcommand: print its help and
    return a non-zero code instead of silently doing nothing."""
    group_parser.print_help(sys.stderr)
    return EXIT_BAD_ARGS


# ── dispatch ─────────────────────────────────────────────────────────────────

def _dispatch_export_subcommand(rest: list[str]) -> int | None:
    """Handle ``cull export {kohya,hf} …`` via a dedicated standalone parser.

    Returns the handler's exit code, or ``None`` if ``rest`` does not name one
    of the recognised sub-commands (caller then falls through to the legacy
    ``cull export <slug> --profile P --out D`` shape).
    """
    if not rest:
        return None
    head = rest[0]
    tail = rest[1:]
    if head == "kohya":
        args = _build_export_kohya_parser().parse_args(tail)
        return cmd_export_kohya(args)
    if head == "hf":
        args = _build_export_hf_parser().parse_args(tail)
        return cmd_export_hf(args)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(argv) if argv is not None else list(sys.argv[1:])
    # ``cull export kohya …`` / ``cull export hf …`` are dispatched to their
    # own standalone parsers so they do not compete with the legacy
    # ``cull export <slug> --profile P --out D`` positional form.
    if argv_list and argv_list[0] == "export" and len(argv_list) >= 2 \
            and argv_list[1] in ("kohya", "hf"):
        try:
            rc = _dispatch_export_subcommand(argv_list[1:])
        except KeyboardInterrupt:  # pragma: no cover - interactive only
            print("\nInterrupted.", file=sys.stderr, flush=True)
            return 130
        # None means the sub-command didn't match — should be unreachable given
        # the guard above, but fall through for safety.
        if rc is not None:
            return rc
    args = parser.parse_args(argv_list)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return EXIT_BAD_ARGS
    try:
        return func(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\nInterrupted.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
