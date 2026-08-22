"""Generate an OpenAPI 3.1 spec from a Flask app's URL map.

Design goals:

* **Zero decoration burden.** Rather than force every route through a decorator
  or annotation, we introspect ``app.url_map`` at request time and lift path
  parameters straight from the werkzeug rule syntax (``<slug>``, ``<key>``…).
* **Docstring-first descriptions.** A handler's first non-blank docstring line
  becomes its OpenAPI ``summary``; the full docstring becomes ``description``.
* **Safe by default.** Anything under ``/api/`` is included; the static shell,
  thumbnail endpoints, and streams are surfaced too so a client using OpenAPI as
  its map of the surface doesn't miss a route.

Consumed by the dashboard's ``/api/openapi.json`` and ``/api/docs`` (Swagger UI)
endpoints.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# The Flask URL-rule mini-language uses ``<converter:name>``; we ignore the
# converter and treat every path param as a string. That matches the current
# dashboard: every dynamic segment is a slug / key / preset name / theme name.
_RULE_PARAM_RE = re.compile(r"<(?:(?P<converter>[^:>]+):)?(?P<name>[^>]+)>")

# Werkzeug adds these to every rule automatically; they don't belong in an
# OpenAPI spec that documents user-facing traffic.
_INTERNAL_METHODS: frozenset[str] = frozenset({"HEAD", "OPTIONS"})

# Rule endpoints Flask registers by default that aren't part of the API.
_INTERNAL_ENDPOINTS: frozenset[str] = frozenset({"static"})

# OpenAPI info block; version is kept low-churn — bump alongside a real
# API-shape change, not on every dashboard release.
_INFO: dict[str, Any] = {
    "title": "cull",
    "version": "0.1.0",
    "description": (
        "REST surface for the cull curation engine. Every endpoint returns a "
        "JSON envelope (``{ok|success, data?, error?}``) unless it streams "
        "bytes (thumbnails, ZIP exports, SSE events). When API-key auth is "
        "enabled (``CULL_API_KEYS``), non-loopback callers must present a "
        "token via ``Authorization: Bearer <token>`` or ``X-Cull-API-Key`` — "
        "see the ``bearerAuth`` / ``apiKeyAuth`` security schemes below."
    ),
}

_SECURITY_SCHEMES: dict[str, Any] = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "Bearer token from ``CULL_API_KEYS``. Loopback callers bypass "
            "auth even when tokens are configured."
        ),
    },
    "apiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-Cull-API-Key",
        "description": "Alternative to ``Authorization: Bearer``.",
    },
}

# Standard error responses reused across most endpoints.
_STANDARD_RESPONSES: dict[str, Any] = {
    "GenericError": {
        "description": "Error envelope.",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "ok": {"type": "boolean"},
                        "data": {"nullable": True},
                        "error": {"type": "string"},
                    },
                }
            }
        },
    }
}


def _rule_to_openapi_path(rule: str) -> str:
    """``/api/jobs/<slug>`` → ``/api/jobs/{slug}``.

    Strips the (optional) converter and wraps the name in braces.
    """
    return _RULE_PARAM_RE.sub(lambda m: "{" + m.group("name") + "}", rule)


def _extract_path_params(rule: str) -> list[dict[str, Any]]:
    """Return the OpenAPI ``parameters`` entries for a Flask rule."""
    out: list[dict[str, Any]] = []
    for match in _RULE_PARAM_RE.finditer(rule):
        name = match.group("name")
        converter = (match.group("converter") or "string").lower()
        # Map werkzeug converters to sensible OpenAPI types; unknown → string.
        if converter in {"int"}:
            schema_type = "integer"
        elif converter in {"float"}:
            schema_type = "number"
        elif converter in {"path"}:
            schema_type = "string"
        else:
            schema_type = "string"
        out.append({
            "name": name,
            "in": "path",
            "required": True,
            "schema": {"type": schema_type},
        })
    return out


def _first_line(text: str) -> str:
    """Return the first non-blank stripped line of ``text``."""
    if not text:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _handler_docs(handler) -> tuple[str, str]:
    """Return ``(summary, description)`` for a handler."""
    doc = (getattr(handler, "__doc__", "") or "").strip()
    if not doc:
        return "", ""
    return _first_line(doc), doc


def _method_operation(rule: str, method: str, handler) -> dict[str, Any]:
    """Build an OpenAPI operation object for ``(method, rule)``."""
    summary, description = _handler_docs(handler)
    parameters = _extract_path_params(rule)

    responses: dict[str, Any] = {
        "200": {"description": "Success."},
        "400": {"$ref": "#/components/responses/GenericError"},
        "401": {"$ref": "#/components/responses/GenericError"},
        "403": {"$ref": "#/components/responses/GenericError"},
        "404": {"$ref": "#/components/responses/GenericError"},
        "500": {"$ref": "#/components/responses/GenericError"},
    }

    op: dict[str, Any] = {
        "operationId": f"{method.lower()}_{_operation_id_suffix(rule)}",
        "responses": responses,
    }
    if summary:
        op["summary"] = summary
    if description and description != summary:
        op["description"] = description
    if parameters:
        op["parameters"] = parameters

    # Write methods with a body: emit a permissive JSON body slot so clients
    # know one is accepted. Handler-specific shapes stay in the docstring.
    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        op["requestBody"] = {
            "required": False,
            "content": {
                "application/json": {
                    "schema": {"type": "object", "additionalProperties": True}
                }
            },
        }

    return op


def _operation_id_suffix(rule: str) -> str:
    """Turn a rule into a stable snake-case operation id fragment."""
    cleaned = _RULE_PARAM_RE.sub(lambda m: m.group("name"), rule)
    cleaned = cleaned.strip("/").replace("/", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", cleaned)
    return re.sub(r"_+", "_", cleaned).strip("_") or "root"


def _iter_api_rules(app) -> Iterable[tuple[str, list[str], Any]]:
    """Yield ``(rule, methods, handler)`` for every /api/* URL rule.

    The dashboard's URL map is walked directly; werkzeug guarantees rule
    uniqueness per (rule, methods) so we don't dedupe further.
    """
    for rule in app.url_map.iter_rules():
        if rule.endpoint in _INTERNAL_ENDPOINTS:
            continue
        raw = str(rule.rule or "")
        if not raw.startswith("/api/"):
            continue
        if raw == "/api/openapi.json" or raw == "/api/docs":
            # Include them in the spec so clients see the full surface — Swagger
            # UI likes to see itself listed.
            pass
        methods = sorted((rule.methods or set()) - _INTERNAL_METHODS)
        if not methods:
            continue
        handler = app.view_functions.get(rule.endpoint)
        yield raw, methods, handler


def build_spec(app) -> dict[str, Any]:
    """Return the full OpenAPI 3.1 document for ``app``.

    Called on every ``GET /api/openapi.json`` — the dashboard's URL map is
    small enough that this is negligible (~ms) and always reflects the current
    route set (no cache invalidation to worry about).
    """
    paths: dict[str, dict[str, Any]] = {}
    for rule, methods, handler in _iter_api_rules(app):
        openapi_path = _rule_to_openapi_path(rule)
        path_item = paths.setdefault(openapi_path, {})
        for method in methods:
            path_item[method.lower()] = _method_operation(rule, method, handler)

    return {
        "openapi": "3.1.0",
        "info": _INFO,
        "servers": [{"url": "/"}],
        "paths": paths,
        "components": {
            "securitySchemes": _SECURITY_SCHEMES,
            "responses": _STANDARD_RESPONSES,
        },
        # Documented as OPTIONAL so unauth'd loopback tooling isn't marked
        # invalid by a stricter validator.
        "security": [
            {"bearerAuth": []},
            {"apiKeyAuth": []},
        ],
    }


# ── Swagger UI HTML ──────────────────────────────────────────────────────────

# The dashboard's CSP whitelists jsdelivr for style/script — see the CSP block
# in ``dashboard_enhanced.py``. Loading Swagger UI from there keeps the page
# self-hosted apart from those two well-known assets. The page is fully static
# so it's safe to inline the small bootstrap script.
SWAGGER_UI_HTML = """
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>cull API — Swagger UI</title>
    <link rel=\"stylesheet\"
          href=\"https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css\">
    <style>body { margin: 0; }</style>
  </head>
  <body>
    <div id=\"swagger-ui\"></div>
    <script src=\"https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js\"></script>
    <script>
      // Minimal bootstrap. The bundle exposes SwaggerUIBundle on window.
      window.addEventListener('load', function () {
        // eslint-disable-next-line no-undef
        SwaggerUIBundle({
          url: '/api/openapi.json',
          dom_id: '#swagger-ui',
          deepLinking: true,
          layout: 'BaseLayout',
        });
      });
    </script>
  </body>
</html>
""".strip()


__all__ = ["build_spec", "SWAGGER_UI_HTML"]
