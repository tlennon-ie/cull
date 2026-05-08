#!/usr/bin/env python3
"""Groq cloud vision worker - llama-4-scout (fast, handles NSFW).

Uses round-robin key rotation across ``GROQ_API_KEYS`` (or a single
``GROQ_API_KEY``). Keys hit by 429 are cooled down for 60s before the
manager hands them out again.
"""
from __future__ import annotations

import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Allow direct invocation as a script.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from pipeline_logging import get_logger
from vision_worker_base import (
    BaseVisionWorker,
    build_response_format,
    extract_message_text,
    _safe_parse_vision_json,
    run_subclass,
)

load_dotenv()

logger = get_logger(__name__)

try:
    from groq import Groq
except ImportError as exc:
    raise SystemExit(
        "groq SDK is not installed. Install pipeline dependencies first:\n"
        "    pip install -r requirements.txt\n"
        "(or `pip install groq` for just this worker)."
    ) from exc


class _KeyManager:
    """Round-robins API keys; cools down a key for 60s after a 429."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = list(keys)
        self.clients = {k: Groq(api_key=k) for k in self.keys}
        self.cooldowns: dict[str, float] = {k: 0.0 for k in self.keys}
        self._lock = threading.Lock()

    def get_client(self) -> tuple[Groq | None, str | None]:
        with self._lock:
            now = time.time()
            available = [k for k in self.keys if self.cooldowns[k] < now]
            if not available:
                soonest = min(self.cooldowns.values())
                wait = soonest - now
                if wait > 0:
                    time.sleep(min(wait, 5))
                return None, None
            key = random.choice(available)
            return self.clients[key], key

    def mark_rate_limited(self, key: str) -> None:
        with self._lock:
            self.cooldowns[key] = time.time() + 60
            logger.warning("rate-limited key ...%s, cooling down 60s", key[-4:])


class BalancedGroqWorker(BaseVisionWorker):
    name = "balanced-groq"
    parallel_workers = 6
    poll_interval = 1.0
    # Groq accepts up to 4 MB images; resize ceiling is generous compared to LM Studio.
    max_image_size = 1024

    def __init__(self) -> None:
        super().__init__()
        raw_keys = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY", "")
        keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not keys:
            raise SystemExit(
                "No GROQ_API_KEY / GROQ_API_KEYS configured. Set one in .env "
                "or via the dashboard Settings tab."
            )
        self.model_id: str = os.environ.get(
            "GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        )
        self._keys = _KeyManager(keys)

    def banner(self) -> None:
        logger.info("=== Vision Worker (Groq %s) ===", self.name)
        logger.info("  Model:   %s", self.model_id)
        logger.info("  Keys:    %d (round-robin)", len(self._keys.keys))
        logger.info("  Workers: %d", self.parallel_workers)

    def classify_image_bytes(
        self,
        b64_jpeg: str,
        prompt_instruction: str,
    ) -> dict[str, Any] | None:
        client, key = self._keys.get_client()
        if not client or not key:
            logger.info("all Groq keys cooling down; signalling RETRY")
            return None

        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_instruction},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_jpeg}",
                                },
                            },
                        ],
                    }
                ],
                model=self.model_id,
                temperature=0.1,
                max_tokens=2000,
                response_format=build_response_format(),
            )
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate" in err.lower():
                self._keys.mark_rate_limited(key)
                return None
            logger.warning("Groq request failed: %s", err[:200])
            return None

        message = response.choices[0].message
        msg_dict = (
            message.model_dump() if hasattr(message, "model_dump") else dict(message)
        )
        raw = extract_message_text(msg_dict)
        parsed = _safe_parse_vision_json(raw)
        if parsed is None:
            preview = (raw or "")[:300].replace("\n", " ")
            logger.warning("empty/invalid JSON from Groq; raw=%r", preview)
            return None
        return parsed


def run_vision_worker() -> None:
    run_subclass(BalancedGroqWorker)


if __name__ == "__main__":
    run_vision_worker()
