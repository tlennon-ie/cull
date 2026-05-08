#!/usr/bin/env python3
"""LM Studio worker that pings the model every 15s to prevent auto-unload.

Identical to BalancedLMWorker except for ``setup()`` which spawns a daemon
thread sending a 1-token request every 15 seconds. Useful when LM Studio is
configured with idle-unload and the queue can be empty for minutes.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_logging import get_logger
from vision_worker_balanced_lm import BalancedLMWorker
from vision_worker_base import run_subclass

logger = get_logger(__name__)

KEEPALIVE_INTERVAL_SECONDS = 15


class LMKeepaliveWorker(BalancedLMWorker):
    name = "lm-keepalive"

    def setup(self) -> None:
        super().setup()
        thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        thread.start()
        logger.info("keepalive thread started (every %ds)", KEEPALIVE_INTERVAL_SECONDS)
        # Give LM Studio a moment to load the model on first ping.
        time.sleep(2)

    def _keepalive_loop(self) -> None:
        while True:
            time.sleep(KEEPALIVE_INTERVAL_SECONDS)
            try:
                requests.post(
                    f"{self.lms_url}/v1/chat/completions",
                    json={
                        "model": self.lms_model,
                        "messages": [{"role": "user", "content": " "}],
                        "max_tokens": 1,
                        "temperature": 0.1,
                    },
                    timeout=30,
                )
            except requests.RequestException:
                # Silent failure - the model might be in the middle of loading,
                # or the network might have hiccuped. Try again next interval.
                pass


def run_vision_worker() -> None:
    run_subclass(LMKeepaliveWorker)


if __name__ == "__main__":
    run_vision_worker()
