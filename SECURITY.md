# Security Policy

## Threat model

cull is a **single-machine, single-user admin tool**. The dashboard binds a Flask
server on `0.0.0.0:5000` by default so Docker port-mapping and RunPod tunneling
just work — but that means the dashboard trusts anyone who can reach the port.
The design assumes:

- The host is trusted (your laptop, your GPU box, your own RunPod).
- The port is only exposed to trusted networks (loopback, a private LAN,
  Tailscale, an SSH tunnel).
- The `.env` file is treated like any other credential store (gitignored,
  never committed, readable only by the running user).

If you need to expose cull to the open internet, put a real reverse proxy with
auth in front of it — cull does not ship built-in authentication.

### Loopback-only deployment

Set `FLASK_HOST=127.0.0.1` in `.env` to bind the dashboard to loopback only.
This keeps everything working on the same machine while refusing every other
network interface.

## What we already do

The dashboard and pipeline ship a number of defensive controls that should not
regress:

| Layer | Control | Where |
|-|-|-|
| Path handling | Every user-supplied path funnels through `safe_inside()` (realpath + `commonpath` containment). Traversal payloads return 400. | `pipeline_code/dashboard_enhanced.py` |
| Secret handling | Credentials never leave the server in plaintext; the settings API returns `********` for any set secret; a POST echoing the mask preserves the stored value. | `SECRET_KEYS` / `SECRET_MASK` in `dashboard_enhanced.py` |
| Error responses | Every endpoint routes exceptions through `_err()`, which returns a fixed generic message and logs the detail server-side. Prevents stack-trace leakage. | `_err()` in `dashboard_enhanced.py` |
| Outbound HTTP | `/api/vision/test`, `/api/scrapers/test`, and the scheduler webhook all use `allow_redirects=False` so a hostile responder cannot 302-bounce the probe at a metadata URL. The webhook additionally rejects non-http(s) schemes. | `dashboard_enhanced.py`, `scheduler.py` |
| Log integrity | Every record formatted by `pipeline_logging.configure_root()`'s handler escapes CR / LF / ESC in the interpolated message, so a job slug, preset name, scraper URL or webhook host containing a newline cannot forge a second log record or repaint the operator's terminal. `exc_info` tracebacks stay multi-line. | `ScrubbingFormatter` in `pipeline_code/pipeline_logging.py` |
| Identity validation | Every `^...$` identity regex (job slug, preset name, theme name, HF repo id, community filename, cookies filename) is anchored with `\Z`, not `$` — Python's `$` also matches immediately before a trailing newline, which would otherwise let `value\n` through into a filesystem path or an outbound repo id. | `job_config.py`, `paths.py`, `theme_config.py`, `config_io.py`, `queue_manager.py`, `dashboard_enhanced.py` |
| Response headers | `after_request` attaches CSP, X-Frame-Options: DENY, X-Content-Type-Options: nosniff, Referrer-Policy: no-referrer, and a restrictive Permissions-Policy to every response. | `_apply_security_headers` in `dashboard_enhanced.py` |
| Credential resolution | `credentials.py` is the single source of truth; scrapers/workers never read `os.environ` directly. A missing required key exits cleanly with `EX_CONFIG` so the supervisor applies a cooldown instead of restart-looping. | `pipeline_code/credentials.py` |
| Dependencies | `pip-audit`, CodeQL (Python), and TruffleHog run in CI on every PR and on a weekly cron. | `.github/workflows/security.yml` |
| CI hardening | GitHub Actions are pinned to major versions (`@v7`, `@v6`, `@v4`, `@v3.95.6`). Dependabot bumps them weekly. | `.github/workflows/*.yml`, `.github/dependabot.yml` |

## Static analysis: accepted findings

CodeQL's `security-and-quality` suite runs over the whole tree, including
`tests/` and `tools/`. A number of its alerts are known, reviewed, and
deliberately not "fixed" — the code they point at is either already guarded by
a sanitiser CodeQL cannot model, or the flagged behaviour is the feature. They
are listed here so a triager does not re-litigate them each sweep.

| Rule | Why it stands |
|-|-|
| `py/path-injection` | Every flagged sink is reached only through `safe_inside()` (realpath + `commonpath` containment), a whitelist (`_BRAND_ASSETS`), or an anchored identity regex (`JOB_SLUG_RE`, `_COMMUNITY_FILENAME_RE`, `THEME_NAME_RE`). CodeQL models neither the regex validators nor `_valid_slug_or_400` as sanitisers, so the taint appears to survive them. `tests/test_path_safety.py` and `tests/test_dashboard_security.py` assert the guards directly. |
| `py/log-injection` | Fixed centrally rather than per call site: `ScrubbingFormatter` escapes CR / LF / ESC in every formatted record (see the table above). CodeQL flags the call sites, not the formatter, so the alert count does not move. `tests/test_security_hardening.py` asserts the scrub. |
| `py/full-ssrf`, `py/partial-ssrf` | Both outbound lanes validate the URL with `scheduler._is_public_http_url` (scheme allowlist + DNS resolution rejected for private / loopback / link-local / metadata addresses) and pass `allow_redirects=False`. The remaining alerts are the guard not being recognised. |
| `py/command-line-injection` | `_git_run` is `shell=False` with an argv list; the only user-derived values are a branch name and a commit message, both prefixed by literals (so neither can present as an option) and built from regex-validated keys. `git add` already uses a `--` separator. |
| `py/stack-trace-exposure` | The survivors are two deliberate classes: (a) a `config_io.ValidationError` message, which is authored validation text and IS the preset editor's API contract — suppressing it would leave the user with "invalid config" and no reason; and (b) the connection-test endpoints (`/api/vision/test`, `/api/scrapers/test`, the yt-dlp probe), whose entire purpose is telling the operator why their local endpoint did not answer. Everything that leaked an *unexpected* internal failure now routes through `_err()`. |
| `py/clear-text-logging-sensitive-data` | `pipeline_code/test_harness.py` prints a present/missing status literal chosen by `_configured(bool)`. The credential value, and anything derived from it (not even its length), never reaches the sink. The alert follows the variable's name, not its value. |
| `py/uninitialized-local-variable`, `py/mixed-returns` (thumbnail route) | Flask's `abort()` raises; CodeQL does not treat it as `NoReturn`, so the code after it looks reachable with an unbound local. |
| `py/catch-base-exception` (two dashboard sites) | Intentional and commented: `credentials.MissingCredentialError` is a `SystemExit` subclass, so a plain `except Exception` would let it kill the worker thread silently instead of being mapped to a friendly status. |

Everything else CodeQL raised in this suite has been fixed rather than
accepted. If you add a finding to this table, say why in the same terms — "the
guard exists and the tool cannot see it", or "the behaviour is the feature" —
and point at the test that pins it.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Email the maintainer at **thomaslennon86@gmail.com** with:

1. A concise description of the issue.
2. The affected commit / release.
3. Reproduction steps or a proof-of-concept.
4. Any suggested remediation.

We aim to acknowledge within 72 hours and to ship a fix (or a public advisory)
within 30 days for confirmed high-severity issues. Coordinated disclosure is
appreciated; credit will be given in the changelog unless you'd rather stay
anonymous.

## Scope

**In scope**

- Path traversal, SSRF, secret leakage, XSS, CSRF affecting the dashboard.
- Injection or race conditions in `pipeline_code/`.
- Vulnerable dependency versions we ship as required or as an extra.
- CI or release-artifact tampering surfaces.

**Out of scope**

- Findings that require a compromised host (physical / OS-level).
- Denial of service against a single-user local tool by that user.
- Vulnerabilities in optional third-party services (Groq, LM Studio, gallery-dl
  extractors, etc.) — please report those upstream.

## Supported versions

cull is pre-1.0. Only the current `main` branch receives security fixes.
