"""tests/test_packaging.py — static asserts for the container packaging.

TDD RED phase: written before the Dockerfile / docker-compose.yml / .dockerignore
exist. These are *static* checks — they parse the files as text and never invoke
`docker build` (which may be unavailable or slow in CI). They lock in the
load-bearing facts an operator relies on:

  * the image installs the project's declared deps + ffmpeg (video lane needs it),
  * it copies the source and exposes the dashboard port,
  * it launches the REAL entrypoint (pipeline_code/integrated_launcher.py),
  * compose mounts ./data and reads secrets from .env at run time, and
  * .dockerignore keeps data/, the venv, and .env OUT of the image.

Run with:
    python -m pytest tests/test_packaging.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Repo root is the parent of tests/ — the Docker files live there.
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
COMPOSE = REPO_ROOT / "docker-compose.yml"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

# The real entrypoint, learned from launch.sh (exec ... integrated_launcher.py)
# and confirmed in pipeline_code/integrated_launcher.py.
ENTRYPOINT = "integrated_launcher.py"


def _read(path: Path) -> str:
    assert path.is_file(), f"expected {path.name} at repo root: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [DOCKERFILE, COMPOSE, DOCKERIGNORE])
def test_packaging_file_exists(path: Path) -> None:
    assert path.is_file(), f"missing packaging file: {path}"


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------

def test_dockerfile_uses_a_python_base() -> None:
    text = _read(DOCKERFILE)
    assert re.search(r"(?im)^\s*FROM\s+python:", text), \
        "Dockerfile should base on an official python image"


def test_dockerfile_installs_requirements() -> None:
    text = _read(DOCKERFILE)
    assert "requirements.txt" in text, "Dockerfile must COPY requirements.txt"
    assert re.search(r"pip\s+install[^\n]*requirements\.txt", text), \
        "Dockerfile must `pip install -r requirements.txt`"


def test_dockerfile_copies_pipeline_code() -> None:
    text = _read(DOCKERFILE)
    assert re.search(r"(?im)^\s*COPY\s+[^\n]*pipeline_code", text), \
        "Dockerfile must COPY the pipeline_code/ source tree"


def test_dockerfile_installs_ffmpeg() -> None:
    # ffmpeg is required by the upcoming video lane (frame extraction).
    text = _read(DOCKERFILE)
    assert re.search(r"\bffmpeg\b", text), \
        "Dockerfile must install ffmpeg (needed for video frame extraction)"


def test_dockerfile_exposes_dashboard_port() -> None:
    text = _read(DOCKERFILE)
    assert re.search(r"(?im)^\s*EXPOSE\s+5000\b", text), \
        "Dockerfile must EXPOSE 5000 (the dashboard port)"


def test_dockerfile_references_real_entrypoint() -> None:
    text = _read(DOCKERFILE)
    assert ENTRYPOINT in text, \
        f"Dockerfile CMD must launch the real entrypoint ({ENTRYPOINT})"


def test_dockerfile_sets_pipeline_base_dir() -> None:
    text = _read(DOCKERFILE)
    assert re.search(r"(?im)ENV\s+PIPELINE_BASE_DIR", text), \
        "Dockerfile should set PIPELINE_BASE_DIR for the mounted data volume"


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------

def test_compose_maps_data_volume() -> None:
    text = _read(COMPOSE)
    assert "./data:/data" in text, \
        "compose must bind-mount ./data to the container data dir"


def test_compose_exposes_port_5000() -> None:
    text = _read(COMPOSE)
    assert re.search(r'["\']?5000:5000["\']?', text), \
        "compose must publish 5000:5000"


def test_compose_uses_env_file() -> None:
    text = _read(COMPOSE)
    assert re.search(r"(?im)env_file\s*:", text), "compose must declare env_file"
    assert ".env" in text, "compose env_file must reference .env (secrets at run time)"


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("needle", ["data/", ".venv", ".env"])
def test_dockerignore_excludes(needle: str) -> None:
    lines = {ln.strip() for ln in _read(DOCKERIGNORE).splitlines()}
    assert any(needle in ln for ln in lines), \
        f".dockerignore must exclude {needle!r} (keep it out of the image)"
