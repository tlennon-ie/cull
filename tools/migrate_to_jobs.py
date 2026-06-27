"""One-shot migration to the jobs model for users upgrading from a pre-jobs cull.

Safe and idempotent. It does NOT move or rename any files — existing queues and
sorted output are already namespaced per-slug on disk
(``data/queue/<slug>/`` and ``data/sorted/<slug>/``), so migration simply:

  1. captures your current ``.env`` config as a job (adopting ``PIPELINE_SLUG`` so
     it inherits the existing queue/sorted folders), and
  2. adopts every other slug that already has data on disk as its own job.

Run from the repo root (inside the venv)::

    python tools/migrate_to_jobs.py

Re-running is harmless — it reports "nothing new to migrate".
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CODE = REPO_ROOT / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

load_dotenv()  # bring the legacy .env config into os.environ

import job_config  # noqa: E402  (after sys.path + dotenv)
import paths  # noqa: E402


def _count_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def main() -> int:
    print("cull - migrate to jobs model", flush=True)
    print(f"  data dir: {paths.base_dir()}", flush=True)

    existing_before = {j.slug for j in job_config.list_jobs()}
    on_disk = job_config.discover_data_slugs()
    if on_disk:
        print(f"  found existing data slugs on disk: {', '.join(on_disk)}", flush=True)
    else:
        print("  no existing per-slug data folders found", flush=True)

    created = job_config.migrate_existing_data()

    if not created:
        active = job_config.get_active_slug()
        print(
            f"\nNothing new to migrate - already on the jobs model "
            f"({len(existing_before)} job(s), active={active!r}).",
            flush=True,
        )
        return 0

    print("\nCreated jobs:", flush=True)
    for job in created:
        q = _count_files(paths.queue_dir(job.slug))
        s = _count_files(paths.sorted_dir(job.slug))
        print(
            f"  - {job.slug:<28} name={job.name!r}  "
            f"(queued files: {q}, sorted files: {s})",
            flush=True,
        )

    print(f"\nActive job: {job_config.get_active_slug()!r}", flush=True)
    print(
        "\nDone. Your existing queues and sorted output are untouched and now "
        "belong to the jobs above. Open the dashboard to manage them.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
