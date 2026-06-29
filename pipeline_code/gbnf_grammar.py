"""gbnf_grammar.py — convert the cull JSON response schema into a llama.cpp GBNF
grammar.

``llama.cpp``'s ``llama-server`` can constrain generation to a GBNF grammar
(``grammar`` field / ``--grammar``), which is a stricter, more portable lever
than the OpenAI ``response_format`` JSON-schema envelope on some builds. This
module turns the schema returned by ``vision_prompt.build_response_format()``
(``["json_schema"]["schema"]``) into an equivalent grammar string so a
``provider == llamacpp`` fleet worker can force valid, schema-shaped JSON.

``json_schema_to_gbnf(schema)`` is a **pure function**: it never mutates the
input and performs no IO. It supports the JSON-Schema subset cull actually uses:

  * ``object`` with ``properties`` + ``required`` (required keys emitted in
    order; ``additionalProperties: false`` is the assumed, enforced shape)
  * ``string`` (escaped-char class; ``enum`` strings become literal alternations)
  * ``integer`` / ``number``
  * ``boolean``
  * ``array`` with ``items``
  * ``enum`` on any type (string / number) → literal alternation

Anything unrecognised degrades to a permissive ``value`` rule rather than
raising, so a schema evolution never hard-fails the worker.

Pure stdlib. No secrets, no runtime pip.
"""
from __future__ import annotations

import json
from typing import Any

# Shared terminal rules every grammar gets. ``ws`` allows optional insignificant
# whitespace between JSON tokens; ``string`` / ``number`` / ``boolean`` / ``null``
# mirror the JSON grammar shipped in llama.cpp's grammar examples.
_PRELUDE_RULES: dict[str, str] = {
    "ws": r"""[ \t\n\r]*""",
    "string": r'''"\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws''',
    "integer": r'''("-"? ([0-9] | [1-9] [0-9]*)) ws''',
    "number": r'''("-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?) ws''',
    "boolean": r'''("true" | "false") ws''',
    "null": r'''"null" ws''',
    # Permissive fallback for any schema node we don't specialise.
    "value": r'''(object | array | string | number | boolean | null) ws''',
    "object": r'''"{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws''',
    "array": r'''"[" ws ( value ("," ws value)* )? "]" ws''',
}


def _gbnf_quote(literal: str) -> str:
    """Render a Python string as a GBNF double-quoted literal (escape \\ and ")."""
    escaped = literal.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _key_literal(key: str) -> str:
    """GBNF that matches a JSON object key, e.g. ``description`` →
    ``"\\"" "description" "\\""`` — the surrounding JSON quote marks as their own
    literals so the bare key name stays readable in the grammar (greppable, and
    no double-escaping). Keys in cull's schema are plain identifiers; an unusual
    key with a quote/backslash falls back to the fully-escaped JSON literal."""
    if '"' in key or "\\" in key:
        return _gbnf_quote(json.dumps(key))
    return f'"\\"" {_gbnf_quote(key)} "\\""'


class _Grammar:
    """Accumulates named rules and hands out fresh names for anonymous subrules."""

    def __init__(self) -> None:
        self._rules: dict[str, str] = {}
        self._counter: int = 0

    def add(self, name: str, body: str) -> str:
        self._rules[name] = body
        return name

    def fresh(self, hint: str) -> str:
        safe = "".join(ch if (ch.isalnum() or ch == "-") else "-" for ch in hint.lower())
        safe = safe.strip("-") or "rule"
        self._counter += 1
        return f"{safe}-{self._counter}"

    def render(self, root_body: str) -> str:
        lines = [f"root ::= {root_body}"]
        for name, body in self._rules.items():
            lines.append(f"{name} ::= {body}")
        for name, body in _PRELUDE_RULES.items():
            lines.append(f"{name} ::= {body}")
        return "\n".join(lines) + "\n"


def _enum_rule(g: _Grammar, values: list[Any], hint: str) -> str:
    """A rule matching exactly one of the JSON-encoded ``values`` (literals)."""
    alts = " | ".join(_gbnf_quote(json.dumps(v)) for v in values) or '""'
    return g.add(g.fresh(hint), f"({alts}) ws")


def _node_to_ref(g: _Grammar, schema: dict[str, Any], hint: str) -> str:
    """Map a schema node to the name of a rule that matches it.

    Returns a rule reference (terminal name or a freshly-registered subrule).
    Unknown shapes fall back to the permissive ``value`` rule.
    """
    if not isinstance(schema, dict):
        return "value"

    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return _enum_rule(g, schema["enum"], hint)

    node_type = schema.get("type")
    if node_type == "object":
        return _object_rule(g, schema, hint)
    if node_type == "array":
        return _array_rule(g, schema, hint)
    if node_type == "string":
        return "string"
    if node_type == "integer":
        return "integer"
    if node_type == "number":
        return "number"
    if node_type == "boolean":
        return "boolean"
    if node_type == "null":
        return "null"
    return "value"


def _object_rule(g: _Grammar, schema: dict[str, Any], hint: str) -> str:
    """Build a rule for an object, emitting its required keys in order.

    cull's schema is strict (``additionalProperties: false``) with every property
    required, so we generate a fixed key sequence: ``"k1" : v1 , "k2" : v2 ...``.
    Properties not listed in ``required`` (rare here) are appended as required
    too — the vision schema marks them all required anyway, and a fixed shape is
    what makes the grammar useful for forcing valid output.
    """
    props: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = list(schema.get("required") or [])
    # Preserve required-order first, then any leftover declared properties.
    ordered_keys = required + [k for k in props if k not in required]

    if not ordered_keys:
        # An object with no known properties → the generic JSON object rule.
        return "object"

    members: list[str] = []
    for key in ordered_keys:
        value_ref = _node_to_ref(g, props.get(key, {}), f"{hint}-{key}")
        members.append(f'{_key_literal(key)} ws ":" ws {value_ref}')
    body = ' "," ws '.join(members)
    return g.add(g.fresh(hint or "object"), f'"{{" ws {body} "}}" ws')


def _array_rule(g: _Grammar, schema: dict[str, Any], hint: str) -> str:
    """Build a rule for an array of ``items`` (homogeneous; tuple-form ignored)."""
    items = schema.get("items")
    if isinstance(items, list):  # tuple validation — first item shape is enough
        items = items[0] if items else {}
    item_ref = _node_to_ref(g, items or {}, f"{hint}-item")
    name = g.fresh(hint or "array")
    return g.add(name, f'"[" ws ( {item_ref} ("," ws {item_ref})* )? "]" ws')


def json_schema_to_gbnf(schema: dict[str, Any]) -> str:
    """Convert a JSON Schema ``dict`` into a llama.cpp GBNF grammar ``str``.

    Pure: the input is never mutated. The returned grammar defines a ``root``
    rule plus every supporting rule, suitable for llama.cpp's ``grammar`` field.

    Supports the subset cull's vision schema uses: object/string/integer/number/
    boolean/array/enum. Unrecognised nodes degrade to a permissive JSON ``value``
    so schema drift can't hard-fail a worker.
    """
    g = _Grammar()
    if not isinstance(schema, dict):
        # Degenerate input → accept any JSON value, leading whitespace allowed.
        return g.render("ws value")
    root_ref = _node_to_ref(g, schema, "schema")
    return g.render(f"ws {root_ref}")


__all__ = ["json_schema_to_gbnf"]
