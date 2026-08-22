"""Topic-aware content filter shared by every scraper.

Drives what counts as an in-topic post based on `.env`:

    PIPELINE_TOPIC            e.g. "Realistic Female Influencer"
    TOPIC_KEYWORDS_EXTRA      Required keywords (see grammar below).
                              Legacy comma-list still works. Default:
                              auto-derived from PIPELINE_TOPIC.
    TOPIC_BANNED_KEYWORDS     Ban phrases (see grammar below).
                              Legacy comma-list still works.
    TOPIC_GENERATION_HINTS    Prompt-craft words (e.g. "photorealistic,cfg").
                              At least one match required in the prompt.
                              Empty -> gate disabled.
    MIN_PROMPT_LENGTH         int, default 40.

Keyword query grammar
---------------------
Any of the three keyword fields (``keywords_extra``, ``banned_keywords``,
``generation_hints``) may now be either the LEGACY list-of-strings (implicit
OR of substring matches) or a QUERY EXPRESSION using ``AND`` / ``OR`` / ``NOT``
and parentheses. Bare terms remain substring-matched (case-insensitive) so
existing setups keep behaving.

Grammar::

    expr    := or_expr
    or_expr := and_expr ( 'OR' and_expr )*
    and_expr:= not_expr ( 'AND' not_expr )*
    not_expr:= 'NOT' term | term
    term    := '(' expr ')' | quoted | word
    quoted  := '"' <chars> '"'
    word    := any non-whitespace, non-paren, non-quote sequence

Whitespace between two operand-terms with no operator between them is treated
as an implicit ``OR`` (backward compatibility with today's list-of-terms
behaviour, so ``foo bar baz`` behaves like ``foo OR bar OR baz``). Operators
are case-insensitive (``and`` = ``AND``). Bare terms are substring-matched
against the (lowercased) prompt text. A malformed expression never raises —
it logs a warning and returns a permissive predicate so a typo can't take a
scraper down. An empty expression yields a predicate that returns ``False``
(safe default when explicitly queried).

Examples::

    "product AND packshot"                     both terms required
    "product OR packshot"                       at least one
    "(product AND packshot) OR (catalog AND studio)"
    "product AND NOT nsfw"                      required + forbidden in one line
    "packshot \"studio lighting\""              quoted phrase with spaces

Everything loads at import time from the current environment. Scrapers call
`passes(title_body, prompt)` once per candidate post before queueing.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("topic_filter")

# Words that show up in every topic but carry no signal.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "and", "or", "for", "with", "ai", "realistic",
    "real", "ultra", "generated", "style", "image", "images", "prompt",
    "prompts",
})


# ── query language: tokenizer + parser + evaluator ────────────────────────────

class _ParseError(ValueError):
    """Raised internally by the parser; callers get a permissive predicate."""


# Detect whether raw env text looks like a query expression vs a legacy CSV
# list. Parens, quotes or a bare AND/OR/NOT word flip it to query mode.
_QUERY_MARKER_RE = re.compile(r'[()"]|(?:\b(?:and|or|not)\b)', re.IGNORECASE)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    """Split ``expr`` into (kind, value) tokens.

    Kinds: ``LPAREN``, ``RPAREN``, ``AND``, ``OR``, ``NOT``, ``TERM``.
    Quoted strings become one ``TERM`` (contents lowercased later).
    """
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(("LPAREN", "("))
            i += 1
            continue
        if ch == ")":
            tokens.append(("RPAREN", ")"))
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and expr[j] != '"':
                j += 1
            # Unterminated quoted string: consume to end (permissive).
            tokens.append(("TERM", expr[i + 1:j]))
            i = j + 1 if j < n else n
            continue
        # A plain word: read until whitespace or a structural char.
        j = i
        while j < n and not expr[j].isspace() and expr[j] not in '()"':
            j += 1
        word = expr[i:j]
        upper = word.upper()
        if upper == "AND":
            tokens.append(("AND", word))
        elif upper == "OR":
            tokens.append(("OR", word))
        elif upper == "NOT":
            tokens.append(("NOT", word))
        else:
            tokens.append(("TERM", word))
        i = j
    return tokens


def _inject_implicit_or(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Insert ``OR`` between two operand-adjacent tokens with no operator.

    An operand ENDS on ``TERM`` or ``RPAREN``; a new operand STARTS on
    ``TERM``, ``LPAREN`` or ``NOT``. This is the load-bearing rule that
    keeps ``"foo bar baz"`` behaving like ``"foo OR bar OR baz"``.
    """
    if not tokens:
        return tokens
    out: list[tuple[str, str]] = [tokens[0]]
    prev_kind = tokens[0][0]
    for kind, val in tokens[1:]:
        if prev_kind in ("TERM", "RPAREN") and kind in ("TERM", "LPAREN", "NOT"):
            out.append(("OR", "OR"))
        out.append((kind, val))
        prev_kind = kind
    return out


# ── AST nodes ────────────────────────────────────────────────────────────────

# AST evaluation is a plain in-string substring test walk — NO Python `eval()`
# or code execution anywhere. The ``evaluate`` method name is used deliberately
# so scanners don't confuse it with the builtin.


@dataclass(frozen=True)
class _Word:
    text: str

    def evaluate(self, hay: str) -> bool:
        return self.text in hay


@dataclass(frozen=True)
class _Not:
    child: Any  # any of the four node types

    def evaluate(self, hay: str) -> bool:
        return not self.child.evaluate(hay)


@dataclass(frozen=True)
class _And:
    left: Any
    right: Any

    def evaluate(self, hay: str) -> bool:
        return self.left.evaluate(hay) and self.right.evaluate(hay)


@dataclass(frozen=True)
class _Or:
    left: Any
    right: Any

    def evaluate(self, hay: str) -> bool:
        return self.left.evaluate(hay) or self.right.evaluate(hay)


class _Parser:
    """Recursive-descent parser for the keyword-query grammar."""

    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> tuple[str | None, str | None]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def _consume(self) -> tuple[str | None, str | None]:
        tok = self._peek()
        self.pos += 1
        return tok

    def parse_expr(self):
        return self._parse_or()

    def _parse_or(self):
        node = self._parse_and()
        while self._peek()[0] == "OR":
            self._consume()
            node = _Or(node, self._parse_and())
        return node

    def _parse_and(self):
        node = self._parse_not()
        while self._peek()[0] == "AND":
            self._consume()
            node = _And(node, self._parse_not())
        return node

    def _parse_not(self):
        if self._peek()[0] == "NOT":
            self._consume()
            return _Not(self._parse_not())
        return self._parse_term()

    def _parse_term(self):
        kind, val = self._peek()
        if kind == "LPAREN":
            self._consume()
            node = self.parse_expr()
            if self._peek()[0] != "RPAREN":
                raise _ParseError("unclosed parenthesis")
            self._consume()
            return node
        if kind == "TERM":
            self._consume()
            return _Word((val or "").lower())
        raise _ParseError(f"unexpected token: {kind}={val!r}")


def parse_query(expr: str) -> Callable[[str], bool]:
    """Return a predicate over lowercased prompt text.

    Empty input yields ``lambda _: False`` (safe default when the caller
    explicitly asks whether an empty query matches). Malformed input logs a
    warning and yields ``lambda _: True`` (permissive — never take a scraper
    down on a typo). The returned predicate expects text that is ALREADY
    lowercased; callers should call ``.lower()`` themselves so we don't
    duplicate the work on every candidate post.
    """
    if not isinstance(expr, str) or not expr.strip():
        return lambda _hay: False
    try:
        tokens = _inject_implicit_or(_tokenize(expr))
        if not tokens:
            return lambda _hay: False
        parser = _Parser(tokens)
        ast = parser.parse_expr()
        if parser.pos != len(tokens):
            raise _ParseError(
                f"unexpected trailing token(s) at position {parser.pos}")
    except _ParseError as exc:
        logger.warning(
            "invalid keyword query %r: %s (treating as permissive)", expr, exc)
        return lambda _hay: True
    return ast.evaluate


def matches_query(query: Any, text_lc: str) -> bool:
    """Evaluate ``query`` (list, str, or None) against lowercased ``text_lc``.

    * A LIST is treated as an implicit-OR of case-insensitive substring
      matches (the legacy comma-list behaviour).
    * A STRING is parsed as a query expression.
    * Anything else (None, empty) → ``False``.
    """
    if isinstance(query, str):
        return parse_query(query)(text_lc)
    if isinstance(query, (list, tuple)):
        for term in query:
            if not isinstance(term, str):
                continue
            t = term.lower().strip()
            if t and t in text_lc:
                return True
        return False
    return False


def _has_query(query: Any) -> bool:
    """True when ``query`` is a non-empty list or non-empty string."""
    if isinstance(query, str):
        return bool(query.strip())
    if isinstance(query, (list, tuple)):
        return any(isinstance(t, str) and t.strip() for t in query)
    return False


def _split_csv(raw: str) -> list[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _parse_env_field(raw: str) -> list[str] | str:
    """Turn a single env-var string into either a legacy list or a query string.

    Detection rule: any ``(``, ``)``, ``"``, or a bare ``AND``/``OR``/``NOT``
    word flips it into query mode; otherwise it's a comma-list. Empty input
    yields ``[]`` (no filter defined).
    """
    text = (raw or "").strip()
    if not text:
        return []
    if _QUERY_MARKER_RE.search(text):
        return text
    return _split_csv(text)


def _auto_keywords(topic: str) -> list[str]:
    """Derive default required keywords from the topic string."""
    tokens = re.findall(r"[A-Za-z][A-Za-z-]+", topic.lower())
    kept = [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]
    # Add a few common synonyms for the known "influencer" family so existing
    # setups keep behaving, but do not impose anything else.
    out = list(dict.fromkeys(kept))
    if any(t in kept for t in ("influencer", "instagram", "instagrammer")):
        out += ["woman", "girl", "female", "model", "portrait", "selfie"]
    if "male" in kept or "man" in kept:
        out += ["man", "male", "guy", "portrait"]
    return out


@dataclass(frozen=True)
class TopicConfig:
    topic: str
    # Each of the three now holds EITHER a list[str] (legacy) or a str (query).
    # ``load_config`` picks the shape based on env-var content; direct callers
    # can pass whichever they prefer and matches_query handles both.
    required_keywords: Any
    banned_keywords: Any
    generation_hints: Any
    min_prompt_length: int


_DEFAULT_BANNED = (
    # Self-promotion spam
    "link in bio", "dm me", "for sale", "patreon", "onlyfans", "only fans",
    "subscribe", "follow me", "commission", "workflow+lora", "workflow + lora",
    "check my", "check out my", "my pack", "my preset", "rate this",
    # Meta posts that never contain real prompts
    "what do you think", "feedback please", "drop your",
)

_DEFAULT_GENERATION_HINTS = (
    "photorealistic", "hyperrealistic", "cinematic", "8k", "4k", "bokeh",
    "depth of field", "dof", "masterpiece", "best quality", "ultra realistic",
    "skin texture", "natural light", "studio light", "golden hour",
    "portrait", "photograph", "camera", "lens", "aperture", "f/",
    "lora", "checkpoint", "negative prompt", "cfg", "sampler", "steps",
)


def load_config() -> TopicConfig:
    topic = os.environ.get("PIPELINE_TOPIC", "").strip()

    raw_extra = os.environ.get("TOPIC_KEYWORDS_EXTRA", "")
    parsed_extra = _parse_env_field(raw_extra)
    if isinstance(parsed_extra, list) and not parsed_extra:
        # Empty explicit setting -> auto-derive from the topic (legacy default).
        required: Any = _auto_keywords(topic)
    else:
        required = parsed_extra

    raw_banned = os.environ.get("TOPIC_BANNED_KEYWORDS", "")
    parsed_banned = _parse_env_field(raw_banned)
    if isinstance(parsed_banned, list) and not parsed_banned:
        banned: Any = list(_DEFAULT_BANNED)
    else:
        banned = parsed_banned

    hints_raw = os.environ.get("TOPIC_GENERATION_HINTS")
    if hints_raw is None:
        hints: Any = list(_DEFAULT_GENERATION_HINTS)
    else:
        hints = _parse_env_field(hints_raw)

    try:
        min_len = int(os.environ.get("MIN_PROMPT_LENGTH", 40))
    except ValueError:
        min_len = 40

    return TopicConfig(
        topic=topic,
        required_keywords=required,
        banned_keywords=banned,
        generation_hints=hints,
        min_prompt_length=min_len,
    )


def prompt_optional() -> bool:
    """Whether scrapers may queue images without a usable prompt.

    Driven by `REQUIRE_PROMPT` (default ``"true"``). When set to ``"false"``,
    scrapers ingest images regardless of prompt length / generation hints,
    and the vision worker's auto-caption step (when enabled) fills in a
    `.txt` afterwards.
    """
    return os.environ.get("REQUIRE_PROMPT", "true").strip().lower() == "false"


def passes(
    context: str,
    prompt: str,
    *,
    cfg: TopicConfig | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason). `context` is title+body/source text, `prompt` is the image prompt.

    A post passes when:
      1. `prompt` is at least MIN_PROMPT_LENGTH chars (skipped if REQUIRE_PROMPT=false).
      2. No banned phrase appears in either context or prompt.
      3. At least one required keyword appears in (context + prompt).
      4. At least one generation hint appears in prompt (skipped if REQUIRE_PROMPT=false).
    """
    cfg = cfg or load_config()
    prompt = (prompt or "").strip()
    haystack = f"{context or ''}\n{prompt}"
    haystack_lc = haystack.lower()
    prompt_lc = prompt.lower()
    require_prompt = not prompt_optional()

    if require_prompt and len(prompt) < cfg.min_prompt_length:
        return False, f"prompt too short ({len(prompt)} < {cfg.min_prompt_length})"

    if _has_query(cfg.banned_keywords) and matches_query(cfg.banned_keywords, haystack_lc):
        return False, "banned phrase"

    if _has_query(cfg.required_keywords) and not matches_query(cfg.required_keywords, haystack_lc):
        return False, "missing required keyword"

    if (
        require_prompt
        and _has_query(cfg.generation_hints)
        and not matches_query(cfg.generation_hints, prompt_lc)
    ):
        return False, "no generation-style keywords in prompt"

    return True, "ok"


def log_rejection(source: str, reason: str, preview: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    clipped = preview[:80].replace("\n", " ")
    logger.debug("[%s] drop (%s): %s", source, reason, clipped)


__all__ = [
    "TopicConfig",
    "load_config",
    "passes",
    "log_rejection",
    "prompt_optional",
    "parse_query",
    "matches_query",
]
