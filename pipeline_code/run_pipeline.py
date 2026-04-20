"""
run_pipeline.py - Topic-driven pipeline orchestrator

Usage:
    python run_pipeline.py --topic "Realistic Female Influencer"
    python run_pipeline.py --topic "Realistic Male Instagrammer" --topic "Classic Race Cars"
    python run_pipeline.py  # uses default topic from PIPELINE_TOPIC env var

Each topic gets its own:
  - sorted/<slug>/ output folder
  - seen_<slug>.json dedup tracking
  - queue/<slug>/ subfolder
  - pipeline_<slug>.log

All scrapers receive the topic and generate relevant search terms automatically.

All paths resolved from .env (never hardcoded).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from paths import base_dir

load_dotenv()

logger = logging.getLogger("pipeline")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

PY: str = sys.executable
PIPELINE_CODE_DIR: Path = Path(os.environ.get("PIPELINE_CODE_DIR", Path(__file__).parent))
BASE_DIR: Path = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))


CHANNEL_GROUPS: list[list[dict]] = [
    [
        {"id": "1013128131764305930", "name": "UD #photorealistic",  "guild": "Unstable Diffusion", "kind": "png_embed"},
        {"id": "1053354830921486498", "name": "UD #photography",     "guild": "Unstable Diffusion", "kind": "png_embed"},
        {"id": "1011063627488440401", "name": "UD #women-only",      "guild": "Unstable Diffusion", "kind": "png_embed"},
        {"id": "1054191232466833478", "name": "UD #requests-sfw",    "guild": "Unstable Diffusion", "kind": "png_embed"},
        {"id": "1011861076570275840", "name": "UD #prompts-woman",   "guild": "Unstable Diffusion", "kind": "png_embed"},
    ],
]


def topic_slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


def stream_output(proc: subprocess.Popen, label: str, log_file=None) -> None:
    try:
        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            try:
                print(f"[{label}] {text}", flush=True)
            except Exception:
                safe = text.encode("ascii", "replace").decode()
                print(f"[{label}] {safe}", flush=True)
            if log_file:
                log_file.write(f"[{label}] {text}\n")
                log_file.flush()
    except Exception as exc:
        logger.warning("[%s] monitor thread error: %s", label, exc)


def disabled_scrapers() -> set[str]:
    raw = os.environ.get("SCRAPER_DISABLED", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def run_topic(topic: str, vision_worker: str = "balanced-groq") -> None:
    slug = topic_slug(topic)
    print(f"\n{'=' * 60}", flush=True)
    print(f"=== TOPIC: {topic} (slug: {slug}) ===", flush=True)
    print(f"=== VISION WORKER: {vision_worker} ===", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    queue_root = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
    sorted_root = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted")))
    queue_dir = queue_root if queue_root.name == slug else queue_root / slug
    sorted_dir = sorted_root if sorted_root.name == slug else sorted_root / slug
    queue_dir.mkdir(parents=True, exist_ok=True)
    for cat in ("InstagramInfluencer", "NSFW", "Professional", "Amateur", "Unknown", "CORRUPT", "DISCARD"):
        (sorted_dir / cat).mkdir(parents=True, exist_ok=True)

    log_dir = Path(os.environ.get("LOG_DIR", str(BASE_DIR / "logs_test")))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{slug}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    env = {
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "PIPELINE_TOPIC": topic,
        "PIPELINE_SLUG": slug,
        "PIPELINE_QUEUE": str(queue_dir),
        "PIPELINE_SORTED": str(sorted_dir),
        **os.environ,
    }

    disabled = disabled_scrapers()
    procs: list[tuple[subprocess.Popen, str]] = []

    def launch(script: str, label: str, extra_args: list[str] | None = None,
               loop: bool = False, loop_sleep: int = 300, env_override: dict | None = None) -> None:
        if label in disabled:
            print(f"  Skipped {label} (disabled via SCRAPER_DISABLED)", flush=True)
            return

        script_path = PIPELINE_CODE_DIR / script
        if not script_path.exists():
            print(f"  WARNING: {script_path} not found; skipping {label}", flush=True)
            return

        run_env = {**env, **(env_override or {})}

        def _run() -> None:
            while True:
                args = [PY, "-u", str(script_path)] + (extra_args or [])
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env)
                procs.append((proc, label))
                stream_output(proc, label, log_file)
                proc.wait()
                print(f"  [{label}] exited (code {proc.returncode})", flush=True)
                if not loop:
                    break
                print(f"  [{label}] restarting in {loop_sleep}s...", flush=True)
                time.sleep(loop_sleep)

        threading.Thread(target=_run, daemon=True).start()
        print(f"  Started {label} (loop={loop})", flush=True)

    launch("scraper_x.py", "X.com", loop=True, loop_sleep=1800)

    for idx, group in enumerate(CHANNEL_GROUPS):
        launch("scraper_discord.py", f"Discord-{idx + 1}", [json.dumps(group)], loop=True, loop_sleep=1800)

    # Civitai: run BOTH domains in parallel via env override
    civitai_domains = [d.strip() for d in os.environ.get("CIVITAI_DOMAINS", "civitai.com,civitai.red").split(",") if d.strip()]
    for domain in civitai_domains:
        domain_label = "Civitai-Red" if domain == "civitai.red" else "Civitai-Com"
        launch(
            "scraper_civitai_search.py",
            domain_label,
            loop=True,
            loop_sleep=600,
            env_override={"CIVITAI_DOMAIN": domain},
        )

    launch("scraper_web.py", "Web", loop=True, loop_sleep=1800)

    human_keywords = ("influencer", "instagram", "woman", "girl", "female", "male", "man",
                      "person", "portrait", "model", "instagrammer")
    topic_is_human = any(kw in topic.lower() for kw in human_keywords)

    if os.environ.get("ZFORFREE_LOCAL_ENABLED", "false").lower() == "true":
        if topic_is_human:
            launch("feed_zforfree_local.py", "ZFF-Local", loop=True, loop_sleep=3600)
        else:
            print(f"  Skipping ZFF-Local (topic '{topic}' is not human/photo focused)", flush=True)

    # Generic admin-configured local folder mirror (any path, any label).
    if os.environ.get("LOCAL_IMPORT_ENABLED", "false").lower() == "true":
        label = os.environ.get("LOCAL_IMPORT_NAME", "local").strip() or "local"
        launch("feed_local_folder.py", f"Local-{label}", loop=True, loop_sleep=3600)

    # Vision workers: PIPELINE_VISION_WORKERS is a comma-separated list,
    # so we can run e.g. "balanced-groq,lm-autodetect" in parallel. Falls back
    # to the legacy single value if the list isn't set.
    raw_workers = os.environ.get("PIPELINE_VISION_WORKERS", "").strip()
    if raw_workers:
        worker_list = [w.strip() for w in raw_workers.split(",") if w.strip()]
    else:
        worker_list = [vision_worker]

    for worker in worker_list:
        if worker == "lm-autodetect":
            vision_script = "vision_worker_lm_autodetect.py"
        elif worker == "local":
            vision_script = "vision_worker.py"
        else:
            vision_script = f"vision_worker_{worker.replace('-', '_')}.py"

        if not (PIPELINE_CODE_DIR / vision_script).exists():
            print(f"  WARNING: {vision_script} not found for worker '{worker}', skipping", flush=True)
            continue
        launch(vision_script, f"Vision-{worker}", loop=True, loop_sleep=10)

    print(f"\nAll agents running for topic: {topic}", flush=True)

    try:
        while True:
            time.sleep(30)
            sorted_counts = {
                c.name: len(list(c.glob("*.vision.json")))
                for c in sorted_dir.iterdir() if c.is_dir()
            }
            total = sum(sorted_counts.values())
            summary = " | ".join(f"{k}:{v}" for k, v in sorted(sorted_counts.items()) if v > 0)
            print(f"  [Stats] Total:{total} | {summary}", flush=True)
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.", flush=True)
    finally:
        log_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic-driven image prompt pipeline")
    parser.add_argument("--topic", action="append", dest="topics",
                        help="Topic to scrape (can be repeated for multiple topics)")
    parser.add_argument("--vision-worker", dest="vision_worker",
                        default=os.environ.get("PIPELINE_VISION_WORKER", "balanced-groq"),
                        choices=["local", "groq", "antigravity", "lm-studio",
                                 "balanced-lm", "balanced-groq", "lm-autodetect", "gemini"],
                        help="Vision worker to use")
    args = parser.parse_args()

    topics = args.topics or [os.environ.get("PIPELINE_TOPIC", "Realistic Female Influencer")]

    print("=== Pipeline Orchestrator ===", flush=True)
    print(f"Pipeline code dir: {PIPELINE_CODE_DIR}", flush=True)
    print(f"Topics: {topics}", flush=True)
    print(f"Vision worker: {args.vision_worker}", flush=True)

    for topic in topics:
        run_topic(topic, vision_worker=args.vision_worker)

    print("\n=== All topics complete ===", flush=True)


if __name__ == "__main__":
    main()
