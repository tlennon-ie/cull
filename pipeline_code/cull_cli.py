"""Headless command-line interface for cull.

Run cull on a remote GPU box with no dashboard. This is a *thin* argparse
wrapper over the EXISTING public APIs — it never reimplements job state:

  * ``job_config`` owns jobs/presets/the active pointer (single source of truth).
  * ``run_pipeline`` owns the supervisor (imported LAZILY so this module stays
    cheap to import and unit-testable without ever spawning a pipeline).
  * ``paths`` owns the filesystem layout used by ``status``.

Subcommands::

    cull jobs list
    cull jobs activate <slug>
    cull job create <slug> --preset <name> --subject "<text>"
    cull presets list
    cull status
    cull run
    cull export <slug> --profile <p> --out <dir>     # optional module

All handlers return an ``int`` exit code; ``main`` returns it so callers (and
``sys.exit``) propagate failures. Errors are reported to stderr with a clear,
non-traceback message.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import job_config
import paths
from pipeline_logging import get_logger

logger = get_logger(__name__)

_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
)


# ── small output helpers ─────────────────────────────────────────────────────

def _err(message: str) -> None:
    """Print a user-facing error to stderr (no traceback)."""
    print(f"error: {message}", file=sys.stderr, flush=True)


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


# ── jobs list ────────────────────────────────────────────────────────────────

def cmd_jobs_list(args: argparse.Namespace) -> int:
    jobs = job_config.list_jobs()
    active = job_config.get_active_slug()
    if not jobs:
        print("No jobs. Create one with: cull job create <slug> --subject \"...\"")
        return 0
    print(f"{'':2}{'SLUG':<24} {'STATUS':<10} NAME")
    for job in jobs:
        marker = "* " if job.slug == active else "  "
        print(f"{marker}{job.slug:<24} {job.status:<10} {job.name}")
    return 0


# ── jobs activate ────────────────────────────────────────────────────────────

def cmd_jobs_activate(args: argparse.Namespace) -> int:
    slug = args.slug
    try:
        job_config.activate(slug)
    except ValueError as exc:
        _err(str(exc))
        return 1
    print(f"Activated job {slug!r} (env + categories projected).")
    return 0


# ── job create ───────────────────────────────────────────────────────────────

def cmd_job_create(args: argparse.Namespace) -> int:
    name = args.slug
    try:
        job = job_config.create_job(
            name, subject=args.subject, preset=args.preset
        )
    except ValueError as exc:
        _err(str(exc))
        return 1
    print(f"Created job {job.slug!r} (preset={job.preset!r}, "
          f"subject={job.subject!r}).")
    return 0


# ── presets list ─────────────────────────────────────────────────────────────

def cmd_presets_list(args: argparse.Namespace) -> int:
    lib = job_config.list_presets()
    presets = lib.get("presets", {})
    default = lib.get("default", "")
    builtins = set(job_config.builtin_preset_names())
    if not presets:
        print("No presets.")
        return 0
    print(f"{'':2}{'PRESET':<24} KIND")
    for name in sorted(presets):
        marker = "* " if name == default else "  "
        kind = "builtin" if name in builtins else "custom"
        print(f"{marker}{name:<24} {kind}")
    return 0


# ── status ───────────────────────────────────────────────────────────────────

def cmd_status(args: argparse.Namespace) -> int:
    active = job_config.get_active_slug()
    index = job_config.get_index()
    queue = index.get("queue", []) or []
    print(f"Active job : {active or '(none)'}")
    print(f"Queued     : {', '.join(queue) if queue else '(empty)'}")
    print(f"Data dir   : {paths.base_dir()}")
    if active:
        q = _count_images(paths.queue_dir(active))
        s = _count_images(paths.sorted_dir(active))
        print(f"Queue images (slug={active})  : {q}")
        print(f"Sorted images (slug={active}) : {s}")
    return 0


# ── run (lazy import of the supervisor) ──────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    """Start the supervisor. ``run_pipeline`` is imported HERE (not at module
    top) so the CLI module stays cheap to import and tests never spawn it."""
    try:
        import run_pipeline  # lazy: only when actually running the pipeline
    except ImportError as exc:  # pragma: no cover - defensive
        _err(f"could not import the supervisor (run_pipeline): {exc}")
        return 1
    run_pipeline.main()
    return 0


# ── export (optional module; do not hard-depend on it) ───────────────────────

def cmd_export(args: argparse.Namespace) -> int:
    try:
        import export_profiles  # optional, may not be installed
    except ImportError:
        _err("export module not available (export_profiles is not installed); "
             "nothing was exported.")
        return 1
    export_dataset = getattr(export_profiles, "export_dataset", None)
    if export_dataset is None:
        _err("export module is present but exposes no export_dataset(); "
             "nothing was exported.")
        return 1
    try:
        summary = export_dataset(args.slug, args.profile, Path(args.out))
    except Exception as exc:  # surface a clean message, not a traceback
        _err(f"export failed: {exc}")
        return 1
    count = summary.get("sample_count") if isinstance(summary, dict) else None
    suffix = f" ({count} sample(s))" if count is not None else ""
    print(f"Exported job {args.slug!r} (profile={args.profile!r}) -> "
          f"{args.out}{suffix}")
    return 0


# ── parser ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cull",
        description="Headless control for cull (no dashboard) - drive jobs, "
                    "presets, and the supervisor from the command line.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # jobs (group: list / activate)
    p_jobs = sub.add_parser("jobs", help="List or activate jobs.")
    jobs_sub = p_jobs.add_subparsers(dest="jobs_command", metavar="<subcommand>")
    p_jobs_list = jobs_sub.add_parser("list", help="List all jobs.")
    p_jobs_list.set_defaults(func=cmd_jobs_list)
    p_jobs_activate = jobs_sub.add_parser(
        "activate", help="Activate a job (projects env + categories).")
    p_jobs_activate.add_argument("slug", help="Job slug to activate.")
    p_jobs_activate.set_defaults(func=cmd_jobs_activate)
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
    p_job_create.set_defaults(func=cmd_job_create)
    p_job.set_defaults(func=lambda a: _require_subcommand(p_job))

    # presets (group: list)
    p_presets = sub.add_parser("presets", help="List presets.")
    presets_sub = p_presets.add_subparsers(
        dest="presets_command", metavar="<subcommand>")
    p_presets_list = presets_sub.add_parser("list", help="List preset names.")
    p_presets_list.set_defaults(func=cmd_presets_list)
    p_presets.set_defaults(func=lambda a: _require_subcommand(p_presets))

    # status
    p_status = sub.add_parser(
        "status", help="Show the active job + basic queue/sorted counts.")
    p_status.set_defaults(func=cmd_status)

    # run
    p_run = sub.add_parser("run", help="Start the supervisor (runs the active job).")
    p_run.set_defaults(func=cmd_run)

    # export (optional)
    p_export = sub.add_parser(
        "export", help="Export a job's dataset (requires the export module).")
    p_export.add_argument("slug", help="Job slug to export.")
    p_export.add_argument("--profile", required=True, help="Export profile name.")
    p_export.add_argument("--out", required=True, help="Output directory.")
    p_export.set_defaults(func=cmd_export)

    return parser


def _require_subcommand(group_parser: argparse.ArgumentParser) -> int:
    """A bare group (e.g. ``cull jobs``) with no subcommand: print its help and
    return a non-zero code instead of silently doing nothing."""
    group_parser.print_help(sys.stderr)
    return 2


# ── dispatch ─────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return func(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\nInterrupted.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
