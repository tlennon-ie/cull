# Contributing to cull

Small fixes welcome. Larger changes (new scraper source, new vision provider,
new dashboard tab) should start as an issue so we can talk shape before you
write the code.

## Ground rules

cull has a tight architecture and a small number of load-bearing seams. Before
opening a PR, please read:

- [`README.md`](README.md) — what cull does and how it fits together.
- [`CLAUDE.md`](CLAUDE.md) — the **must-follow** conventions (categories, vision
  worker registry, queue Protocol, `seen_store`, `credentials`, `safe_inside`,
  the atomic `.processing` rename, the strict JSON schema on every worker).
- [`SECURITY.md`](SECURITY.md) — the threat model and the defensive controls
  that must not regress.

If you're using an AI coding agent (Claude Code, Cursor, Aider, Codex, …), the
project ships a Claude-style skill at
[`.claude/skills/cull-helper/SKILL.md`](.claude/skills/cull-helper/SKILL.md) so
the agent knows the load-bearing seams.

## Development setup

```bash
git clone https://github.com/tlennon-ie/cull.git
cd cull
./launch.sh                       # macOS / Linux — creates .venv, installs deps, boots the dashboard
# launch.bat                      # Windows
```

Prefer to install once and run separately:

```bash
./install.sh                      # or install.bat / install.ps1
python pipeline_code/dashboard_enhanced.py
```

Install the dev extras (pytest + ruff + pip-audit):

```bash
pip install -e ".[dev]"
```

## Running the checks

Before opening a PR, run the same three gates CI runs:

```bash
# 1. Fast import smoke test (mirrors CLAUDE.md's canonical one-liner)
python -c "import sys; sys.path.insert(0, 'pipeline_code'); import importlib; [importlib.import_module(m) for m in (
  'paths','pipeline_logging','categories','job_config','builtin_presets','vision_workers','vision_prompt',
  'queue_manager','topic_filter','seen_store','credentials','scraper_test',
  'feed_local_folder',
  'scraper_civitai','scraper_civitai_search','scraper_x','scraper_discord','scraper_web',
  'scraper_gallery_dl',
  'vision_worker_base','vision_worker_balanced_lm','vision_worker_balanced_groq',
  'vision_worker_lm_autodetect','vision_worker_lm_keepalive','vision_worker',
  'run_pipeline','integrated_launcher','dashboard_enhanced')]; print('OK')"

# 2. Unit tests (scoped to tests/ on purpose — pipeline_code/test_harness.py
#    is a live end-to-end runner, not a unit test)
pytest -q tests/

# 3. Lint (non-blocking today, but keep new code clean)
ruff check .
```

The security workflow (`pip-audit`, CodeQL, TruffleHog) runs on the PR
automatically. If `pip-audit` flags a dependency, bump the pin in
`requirements.txt` and mention the CVE in your commit message.

## Making a change

1. **Discuss the shape first** for anything beyond a small fix. Use an issue.
2. **Follow the seams.** See `CLAUDE.md`. Adding a scraper is a copy of
   `scraper_civitai.py` + a row in `run_pipeline.compute_desired_agents` +
   a description in `_SCRAPER_DESCRIPTIONS`. Do not roll your own dedup, path
   guard, or credential loader.
3. **Small commits.** Each commit should stand on its own and describe *why*.
4. **Small files.** Aim for 200-400 lines per module, 800 lines max.
5. **No new secrets in committed files.** `.env` is the only place.
6. **Do not weaken the security controls.** In particular:
   - Every user-supplied path in a dashboard endpoint goes through
     `safe_inside()`.
   - Every outbound HTTP call passes `allow_redirects=False` unless the caller
     needs redirects.
   - Every scraper uses `credentials.get_required(...)` / `SeenStore(...)`.
   - Errors are surfaced through `_err()`, not `str(exc)`.
7. **Add tests** for anything non-trivial. See `tests/test_dashboard_security.py`
   for the security-regression style.
8. **Update docs.** A new env var goes into `.env.example`; a new endpoint or
   feature belongs in `README.md`; a new load-bearing convention goes into
   `CLAUDE.md`.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <short summary>

<optional body — explain the *why*, not the *what*>
```

Types we use: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.

Examples from the log:

- `feat(gallery-dl): global cookies + global/per-job custom args + supported-sites link`
- `fix(security): scrub fleet keys, SSRF guards, mask sentinel, video-discard`
- `refactor(dashboard): delegate inheritable-config validation to config_io`

## Pull request checklist

Copy this into your PR description and check what applies:

- [ ] Import smoke test passes locally.
- [ ] `pytest -q tests/` passes locally.
- [ ] `pip-audit -r requirements.txt` reports no new vulnerabilities.
- [ ] Any new dashboard endpoint that reads a user-supplied path uses
      `safe_inside()`.
- [ ] Any new outbound HTTP call sets `allow_redirects=False`.
- [ ] Any new env var appears in `.env.example` with a comment.
- [ ] User-facing behavior change is reflected in `README.md`.
- [ ] Load-bearing architecture change is reflected in `CLAUDE.md`.

## Code of conduct

By participating in this project you agree to abide by the
[Contributor Covenant](CODE_OF_CONDUCT.md). Be kind, be constructive, and
assume good faith.
