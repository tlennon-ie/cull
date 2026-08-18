<!-- Thanks for contributing to cull! Please fill in the sections below. Skip
sections that genuinely don't apply. -->

## Summary

<!-- One or two sentences: what does this change and why. Focus on the *why*. -->

## Change type

- [ ] `feat` — user-visible new capability
- [ ] `fix` — bug fix
- [ ] `refactor` — internal restructure, no behavior change
- [ ] `docs` — docs / comments only
- [ ] `test` — tests only
- [ ] `chore` / `perf` / `ci` — housekeeping

## Related issues / discussion

<!-- e.g. Fixes #123, addresses discussion in #456 -->

## What changed

<!-- Bulleted list of the concrete changes. Group by file or area if it helps. -->

-

## Test plan

<!-- How did you verify this? What should a reviewer run to reproduce your check? -->

- [ ] Import smoke test (`python -c "..."` — see CONTRIBUTING.md) passes.
- [ ] `pytest -q tests/` passes locally.
- [ ] Manual verification steps:
  - <!-- e.g. "Started the dashboard, opened Settings, saved a new Groq key, saw the mask return." -->

## Security checklist (for anything touching endpoints, scrapers, workers, or dependencies)

- [ ] No new secret is committed. `.env` remains gitignored.
- [ ] Any new user-supplied path in a dashboard endpoint goes through
      `safe_inside()`.
- [ ] Any new outbound HTTP call passes `allow_redirects=False` (or documents
      why redirects are needed).
- [ ] Any error response goes through `_err()` (fixed message, detail logged
      server-side), not `str(exc)`.
- [ ] Any new env var is added to `.env.example` with a comment.
- [ ] `pip-audit -r requirements.txt` reports no new vulnerabilities.

## Docs

- [ ] User-facing behavior change is reflected in `README.md`.
- [ ] Load-bearing architecture change (a new registry, a new seam) is
      reflected in `CLAUDE.md`.
- [ ] New skill / agent contract change is reflected in `.claude/skills/`.

## Backward compatibility

<!-- Any migration steps? New required env vars? Data-on-disk shape change?
     If yes: state exactly what a user upgrading from `main` has to do. -->

## Screenshots (for dashboard changes)

<!-- Attach before/after screenshots for anything visible in the UI. -->
