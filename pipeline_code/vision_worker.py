#!/usr/bin/env python3
"""Single-thread LM Studio worker (debug / fallback variant).

Identical to BalancedLMWorker except parallel_workers=1, useful when you
want deterministic ordering or are debugging a flaky model. For production
parallelism, use ``balanced-lm`` instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vision_worker_balanced_lm import BalancedLMWorker
from vision_worker_base import run_subclass


class LocalLMWorker(BalancedLMWorker):
    name = "local"
    parallel_workers = 1


def run_vision_worker() -> None:
    run_subclass(LocalLMWorker)


if __name__ == "__main__":
    run_vision_worker()
