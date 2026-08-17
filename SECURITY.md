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
| Response headers | `after_request` attaches CSP, X-Frame-Options: DENY, X-Content-Type-Options: nosniff, Referrer-Policy: no-referrer, and a restrictive Permissions-Policy to every response. | `_apply_security_headers` in `dashboard_enhanced.py` |
| Credential resolution | `credentials.py` is the single source of truth; scrapers/workers never read `os.environ` directly. A missing required key exits cleanly with `EX_CONFIG` so the supervisor applies a cooldown instead of restart-looping. | `pipeline_code/credentials.py` |
| Dependencies | `pip-audit`, CodeQL (Python), and TruffleHog run in CI on every PR and on a weekly cron. | `.github/workflows/security.yml` |
| CI hardening | GitHub Actions are pinned to major versions (`@v7`, `@v6`, `@v4`, `@v3.95.6`). Dependabot bumps them weekly. | `.github/workflows/*.yml`, `.github/dependabot.yml` |

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
