"""tests/test_ci_config.py — static asserts for the GitHub Actions CI/CD config.

TDD RED phase: written before .github/workflows/ci.yml, .github/workflows/security.yml
and .github/dependabot.yml exist. These are *static, tolerant* checks — they parse
the YAML files as plain TEXT with regexes and never import PyYAML (which is NOT a
project dependency; do not add it). They never invoke `act` or the GitHub runner.

They lock in the load-bearing facts the project relies on so the pipeline keeps
"running correctly for users + no dependency vulnerabilities" (roadmap req #5):

  * ci.yml runs the Python version matrix (3.10/3.11/3.12), installs requirements,
    runs the CLAUDE.md import-smoke check, and runs the pytest suite,
  * security.yml runs pip-audit (fails on known CVEs), CodeQL for Python, and a
    secret scan, on PRs + a weekly schedule,
  * dependabot.yml watches both the pip (requirements.txt) and github-actions
    ecosystems on a weekly cadence.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_ci_config.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Repo root is the parent of tests/ — the .github tree lives there.
REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_DIR = REPO_ROOT / ".github"
CI_YML = GITHUB_DIR / "workflows" / "ci.yml"
SECURITY_YML = GITHUB_DIR / "workflows" / "security.yml"
DEPENDABOT_YML = GITHUB_DIR / "dependabot.yml"


# ---------------------------------------------------------------------------
# Helpers — tolerant text loading (no YAML parser).
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    """Read a config file as text, failing loudly if it is missing."""
    assert path.is_file(), f"expected config file is missing: {path}"
    return path.read_text(encoding="utf-8")


def _norm(text: str) -> str:
    """Lower-case + collapse whitespace for forgiving substring matching."""
    return re.sub(r"\s+", " ", text).lower()


# ---------------------------------------------------------------------------
# Fixtures — read each file once.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ci_text() -> str:
    return _read(CI_YML)


@pytest.fixture(scope="module")
def security_text() -> str:
    return _read(SECURITY_YML)


@pytest.fixture(scope="module")
def dependabot_text() -> str:
    return _read(DEPENDABOT_YML)


# ---------------------------------------------------------------------------
# ci.yml
# ---------------------------------------------------------------------------

class TestCIWorkflow:
    def test_file_exists(self) -> None:
        assert CI_YML.is_file(), f"missing {CI_YML}"

    def test_triggers_on_push_and_pull_request(self, ci_text: str) -> None:
        norm = _norm(ci_text)
        assert "push:" in norm or "push :" in norm or " push" in norm
        assert "pull_request" in norm

    def test_branches_main_and_feat(self, ci_text: str) -> None:
        # main + the feat/** glob used by the roadmap branches.
        assert re.search(r"\bmain\b", ci_text)
        assert "feat/**" in ci_text

    def test_runs_on_ubuntu(self, ci_text: str) -> None:
        assert "ubuntu-latest" in ci_text

    def test_python_version_matrix(self, ci_text: str) -> None:
        # All three supported interpreters must appear in a matrix list.
        for ver in ("3.10", "3.11", "3.12"):
            assert ver in ci_text, f"python {ver} missing from matrix"
        assert "matrix" in _norm(ci_text), "no build matrix declared"

    def test_uses_pinned_checkout_and_setup_python(self, ci_text: str) -> None:
        # Actions must be version-pinned, not floating on a branch.
        assert "actions/checkout@v4" in ci_text
        assert "actions/setup-python@v5" in ci_text

    def test_pip_cache_enabled(self, ci_text: str) -> None:
        # setup-python's built-in pip cache keeps the gallery-dl git dep fast-ish.
        assert re.search(r"cache:\s*['\"]?pip['\"]?", ci_text), "pip cache not enabled"

    def test_installs_requirements(self, ci_text: str) -> None:
        assert "requirements.txt" in ci_text
        assert re.search(r"pip install .*-r .*requirements\.txt", _norm(ci_text)) \
            or re.search(r"pip install\b[^\n]*-r[^\n]*requirements\.txt", ci_text)

    def test_runs_import_smoke_check(self, ci_text: str) -> None:
        # The CLAUDE.md import sanity check — a handful of its modules must appear,
        # proving the real one-liner is wired into CI.
        norm = _norm(ci_text)
        assert "importlib" in norm
        for mod in ("run_pipeline", "dashboard_enhanced", "vision_prompt", "job_config"):
            assert mod in ci_text, f"import-smoke module {mod!r} missing"

    def test_runs_pytest(self, ci_text: str) -> None:
        assert re.search(r"pytest\b", ci_text), "pytest step missing"

    def test_pytest_scoped_to_tests_dir(self, ci_text: str) -> None:
        # Must target tests/ — a bare `pytest` walks the whole tree and would
        # collect pipeline_code/test_harness.py (a live runner, not a unit test).
        assert re.search(r"pytest\b[^\n]*\btests/?\b", ci_text), \
            "pytest must be scoped to tests/ to avoid the live test_harness.py"

    def test_runs_ruff_lint(self, ci_text: str) -> None:
        # Lint step present (may be non-blocking when no ruff config ships).
        assert "ruff" in _norm(ci_text), "ruff lint step missing"


# ---------------------------------------------------------------------------
# security.yml
# ---------------------------------------------------------------------------

class TestSecurityWorkflow:
    def test_file_exists(self) -> None:
        assert SECURITY_YML.is_file(), f"missing {SECURITY_YML}"

    def test_pip_audit_present(self, security_text: str) -> None:
        norm = _norm(security_text)
        assert "pip-audit" in norm, "pip-audit step missing"
        # It must audit the pinned requirements file.
        assert "requirements.txt" in security_text

    def test_codeql_python(self, security_text: str) -> None:
        norm = _norm(security_text)
        assert "codeql" in norm, "CodeQL not configured"
        # Both init + analyze stages of the CodeQL action.
        assert "codeql-action/init" in security_text
        assert "codeql-action/analyze" in security_text
        # Language must target python.
        assert "python" in norm

    def test_secret_scan_present(self, security_text: str) -> None:
        norm = _norm(security_text)
        assert ("gitleaks" in norm) or ("trufflehog" in norm), \
            "no secret-scanning action (gitleaks/trufflehog) found"

    def test_triggers_pull_request_and_schedule(self, security_text: str) -> None:
        norm = _norm(security_text)
        assert "pull_request" in norm, "security scans should run on PRs"
        assert "schedule" in norm and "cron" in norm, "weekly cron schedule missing"

    def test_actions_are_version_pinned(self, security_text: str) -> None:
        # Every `uses:` reference must carry an @ref pin (no floating actions).
        uses_refs = re.findall(r"uses:\s*([^\s]+)", security_text)
        assert uses_refs, "no actions referenced in security.yml"
        for ref in uses_refs:
            assert "@" in ref, f"action not version-pinned: {ref}"


# ---------------------------------------------------------------------------
# dependabot.yml
# ---------------------------------------------------------------------------

class TestDependabotConfig:
    def test_file_exists(self) -> None:
        assert DEPENDABOT_YML.is_file(), f"missing {DEPENDABOT_YML}"

    def test_version_two(self, dependabot_text: str) -> None:
        assert re.search(r"version:\s*2", dependabot_text), "dependabot must be v2"

    def test_pip_ecosystem(self, dependabot_text: str) -> None:
        assert re.search(r'package-ecosystem:\s*["\']?pip["\']?', dependabot_text), \
            "pip ecosystem missing"

    def test_github_actions_ecosystem(self, dependabot_text: str) -> None:
        assert re.search(
            r'package-ecosystem:\s*["\']?github-actions["\']?', dependabot_text
        ), "github-actions ecosystem missing"

    def test_weekly_schedule(self, dependabot_text: str) -> None:
        assert re.search(r'interval:\s*["\']?weekly["\']?', dependabot_text), \
            "weekly update interval missing"

    def test_open_pr_limit_set(self, dependabot_text: str) -> None:
        # A sensible cap so Dependabot does not flood the PR queue.
        assert "open-pull-requests-limit" in dependabot_text, \
            "open-pull-requests-limit not set"
