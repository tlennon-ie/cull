"""Shared logging configuration for the pipeline.

The supervisor (run_pipeline.py) launches each scraper / vision worker as a
subprocess and captures its stdout/stderr verbatim into ``logs/pipeline_<slug>.log``.
That captured stream is the one humans tail, and it deliberately preserves
each child's ``print(...)`` statements unchanged — switching every worker
to ``logging.getLogger(...)`` would add per-process formatter prefixes that
double up with the supervisor's own ``[label]`` prefix and make the file
harder to read, not easier.

So the convention is:
  * Subprocess workers (scraper_*.py, vision_worker_*.py, feed_*.py) keep
    their ``print(..., flush=True)`` calls. They run with PYTHONUNBUFFERED=1
    and the supervisor labels their output.
  * Supervisor + dashboard + library code (queue_manager.py, topic_filter.py,
    paths.py, run_pipeline.py, dashboard_enhanced.py) use ``logging.getLogger(__name__)``
    via this module so log levels and format are consistent.

Call ``configure_root()`` once at process startup. Subsequent calls are
no-ops.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"
_DATE_FORMAT: Final[str] = "%H:%M:%S"
_configured: bool = False


def configure_root(default_level: str = "INFO") -> None:
    """Configure the root logger once. Honors $LOG_LEVEL override.

    Idempotent: subsequent calls are no-ops, so a child process can call this
    safely even if the parent already configured handlers.
    """
    global _configured
    if _configured:
        return
    level_name = os.environ.get("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    # Replace any pre-existing handlers (basicConfig debris) with our own so
    # the format is consistent across the whole process tree.
    root.handlers[:] = [handler]
    root.setLevel(level)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Use ``__name__`` as the argument."""
    if not _configured:
        configure_root()
    return logging.getLogger(name)


__all__ = ["configure_root", "get_logger"]
