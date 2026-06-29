"""tests/test_pyproject.py — static asserts for the PEP 621 packaging metadata.

TDD RED phase: written before ``pyproject.toml`` exists. These checks parse the
file (preferring ``tomllib`` on Python 3.11+, falling back to a forgiving text
scan on 3.10) and never invoke ``pip``/``build`` — so they stay fast and run on
any interpreter. They lock in the load-bearing facts an installer relies on:

  * the project is named ``cull`` and declares ``requires-python >= 3.10``;
  * the build backend is setuptools (so ``pip install -e .`` works);
  * the console script ``cull`` resolves to ``cull_cli:main`` (the headless CLI
    at ``pipeline_code/cull_cli.py``); and
  * the optional-dependency extras the README/requirements advertise actually
    exist: ``cloud`` / ``video`` / ``ml`` / ``dev``.

Run with:
    python -m pytest tests/test_pyproject.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# pyproject.toml lives at the repo root — the parent of tests/.
REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The console script must resolve to the headless CLI's main().
EXPECTED_SCRIPT_NAME = "cull"
EXPECTED_SCRIPT_TARGET = "cull_cli:main"

# Optional-dependency extras the README + requirements.txt advertise.
EXPECTED_EXTRAS = ("cloud", "video", "ml", "dev")


def _read() -> str:
    assert PYPROJECT.is_file(), f"expected pyproject.toml at repo root: {PYPROJECT}"
    return PYPROJECT.read_text(encoding="utf-8")


def _load_toml() -> dict | None:
    """Parse pyproject.toml with tomllib (3.11+). Return ``None`` on 3.10 so the
    caller can fall back to the text-based assertions."""
    if sys.version_info < (3, 11):
        return None
    import tomllib

    return tomllib.loads(_read())


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------

def test_pyproject_exists() -> None:
    assert PYPROJECT.is_file(), f"missing packaging file: {PYPROJECT}"


# ---------------------------------------------------------------------------
# [project] core metadata (PEP 621)
# ---------------------------------------------------------------------------

def test_project_core_metadata() -> None:
    data = _load_toml()
    if data is None:  # 3.10 fallback — text scan.
        text = _read()
        assert re.search(r'(?im)^\s*name\s*=\s*["\']cull["\']', text), \
            "[project] name must be \"cull\""
        assert re.search(r'requires-python\s*=\s*["\'][^"\']*3\.10', text), \
            "[project] requires-python must allow 3.10+"
        return
    project = data.get("project", {})
    assert project.get("name") == "cull", '[project] name must be "cull"'
    assert "3.10" in str(project.get("requires-python", "")), \
        "[project] requires-python must allow 3.10+"
    assert project.get("version"), "[project] must declare a version"


def test_build_system_is_setuptools() -> None:
    data = _load_toml()
    if data is None:  # 3.10 fallback — text scan.
        text = _read()
        assert re.search(r"(?im)^\s*\[build-system\]", text), \
            "pyproject must declare a [build-system] table"
        assert "setuptools" in text, "build backend should be setuptools"
        return
    build = data.get("build-system", {})
    requires = " ".join(build.get("requires", []))
    assert "setuptools" in requires, "[build-system].requires must include setuptools"
    assert "setuptools" in str(build.get("build-backend", "")), \
        "[build-system].build-backend must be setuptools"


# ---------------------------------------------------------------------------
# [project.scripts] — the console entry point
# ---------------------------------------------------------------------------

def test_console_script_cull_exists() -> None:
    """``[project.scripts] cull = "cull_cli:main"`` must exist so the installed
    ``cull`` command resolves to the headless CLI."""
    data = _load_toml()
    if data is None:  # 3.10 fallback — text scan.
        text = _read()
        assert re.search(r"(?im)^\s*\[project\.scripts\]", text), \
            "pyproject must declare a [project.scripts] table"
        assert re.search(
            r'(?im)^\s*cull\s*=\s*["\']cull_cli:main["\']', text
        ), 'pyproject must map cull = "cull_cli:main"'
        return
    scripts = data.get("project", {}).get("scripts", {})
    assert EXPECTED_SCRIPT_NAME in scripts, \
        '[project.scripts] must define a "cull" entry point'
    assert scripts[EXPECTED_SCRIPT_NAME] == EXPECTED_SCRIPT_TARGET, \
        f'[project.scripts] cull must point at "{EXPECTED_SCRIPT_TARGET}"'


# ---------------------------------------------------------------------------
# [project.optional-dependencies] — the extras
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extra", EXPECTED_EXTRAS)
def test_optional_dependency_extra_exists(extra: str) -> None:
    """Each advertised extra (cloud / video / ml / dev) must be declared."""
    data = _load_toml()
    if data is None:  # 3.10 fallback — text scan.
        text = _read()
        assert re.search(r"(?im)^\s*\[project\.optional-dependencies\]", text), \
            "pyproject must declare a [project.optional-dependencies] table"
        assert re.search(rf"(?im)^\s*{re.escape(extra)}\s*=", text), \
            f"optional-dependencies must define the {extra!r} extra"
        return
    extras = data.get("project", {}).get("optional-dependencies", {})
    assert extra in extras, \
        f"[project.optional-dependencies] must define the {extra!r} extra"
    assert extras[extra], f"the {extra!r} extra must list at least one dependency"


def test_all_expected_extras_present_together() -> None:
    """Belt-and-braces: the full set of extras is present (catches a partial
    table even if the parametrized test above is filtered)."""
    data = _load_toml()
    if data is None:
        text = _read()
        for extra in EXPECTED_EXTRAS:
            assert re.search(rf"(?im)^\s*{re.escape(extra)}\s*=", text), \
                f"optional-dependencies missing {extra!r}"
        return
    extras = data.get("project", {}).get("optional-dependencies", {})
    missing = [e for e in EXPECTED_EXTRAS if e not in extras]
    assert not missing, f"optional-dependencies missing extras: {missing}"


if __name__ == "__main__":  # python tests/test_pyproject.py
    raise SystemExit(pytest.main([__file__, "-q"]))
