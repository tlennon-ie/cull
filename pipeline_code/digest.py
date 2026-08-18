"""Daily-digest payload assembly for the active (or any) job.

The digest reads exclusively from :mod:`index_store` — the SQLite index that
already tracks every sorted image alongside its parsed ``.vision.json``. No
filesystem walks, no re-parsing of audit records.

Public API:

* :func:`build_digest` — assemble the payload dict for one slug over a time
  window (default: last 24h).
* :func:`render_markdown` — format the payload as Markdown (Discord/Slack).
* :func:`render_plain` — format it as plain text (email-friendly).

Design rules (load-bearing):

* Read-only against ``index_store``; the digest never regenerates audit
  records or touches the vision workers' outputs.
* All timestamps produced by the digest are UTC ISO-8601.
* An empty index for the slug returns the same shape with zeros — never raises.
* KEEP-vs-terminal buckets are derived from :mod:`categories` so a taxonomy
  edit reshapes the digest without code changes.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

import index_store
from categories import get_all_categories, get_categories, get_terminal_categories
from pipeline_logging import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: Default window length when the caller does not pass ``since``.
_DEFAULT_WINDOW_HOURS = 24

#: A "long-reason" outlier is one whose ``reason`` exceeds this length.
_LONG_REASON_CHARS = 200

#: A score-band outlier sits inside this half-open OVR range (0-1 scale).
_OUTLIER_SCORE_LO = 0.4
_OUTLIER_SCORE_HI = 0.6

#: Maximum outliers surfaced in the payload — the digest is a summary, not a
#: log dump.
_MAX_OUTLIERS = 3

#: OVR / REL are stored as 0-100 ints in the index; the payload exposes them
#: on a 0-1 float scale to match how humans usually talk about "score 0.5".
_SCORE_SCALE = 100.0

#: Category name for images the model chose to discard.
_DISCARD = "DISCARD"


# ── Categorisation helpers ───────────────────────────────────────────────────

def _bucket(category: str | None) -> str:
    """Fold a category name into one of the four totals buckets.

    * ``kept`` — any KEEP-bucket category in the active taxonomy.
    * ``borderline`` — a category whose name literally contains "borderline"
      (case-insensitive). Presets like ``general_dataset`` use it explicitly.
    * ``off_topic`` — categories that look like off-topic buckets (name
      contains "off" and "topic").
    * ``discarded`` — DISCARD / CORRUPT / any terminal bucket.
    """
    if not category:
        return "discarded"
    name = category.strip()
    if name in get_terminal_categories():
        return "discarded"
    lowered = name.lower()
    if "borderline" in lowered:
        return "borderline"
    if "off" in lowered and "topic" in lowered:
        return "off_topic"
    if name in get_categories():
        return "kept"
    # Unknown category (edited-out taxonomy row) — treat as discarded so the
    # totals still add up.
    return "discarded"


# ── Vision-payload helpers ───────────────────────────────────────────────────

def _score(vision_json: dict[str, Any] | None, key: str) -> float:
    """Fetch a score from the audit record and normalise to 0-1.

    Returns ``0.0`` when the key is missing or unparseable.
    """
    if not isinstance(vision_json, dict):
        return 0.0
    raw = vision_json.get(key)
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value / _SCORE_SCALE if value > 1 else value


def _text_field(vision_json: dict[str, Any] | None, key: str) -> str:
    """Safe extractor for freeform string fields (reason / caption)."""
    if not isinstance(vision_json, dict):
        return ""
    value = vision_json.get(key)
    return str(value) if value is not None else ""


# ── Time window ──────────────────────────────────────────────────────────────

def _now_utc() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _as_utc(moment: _dt.datetime) -> _dt.datetime:
    """Coerce a datetime to UTC, treating naive values as UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=_dt.timezone.utc)
    return moment.astimezone(_dt.timezone.utc)


def _empty_payload(slug: str, window_hours: int, now: _dt.datetime) -> dict[str, Any]:
    """The zero-shaped payload returned when the slug has no indexed rows."""
    return {
        "slug": slug,
        "generated_at": now.isoformat(),
        "window_hours": window_hours,
        "totals": {"kept": 0, "borderline": 0, "off_topic": 0, "discarded": 0},
        "top_keeps": [],
        "category_breakdown": [],
        "outliers": [],
        "avg_scores": {"OVR_Quality_Score": 0.0, "REL_Quality_Score": 0.0},
        "sample_count": 0,
    }


# ── Public: build_digest ─────────────────────────────────────────────────────

def build_digest(
    slug: str,
    since: _dt.datetime | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    """Assemble the digest payload for ``slug`` over ``[since, now]``.

    ``since`` defaults to 24 hours ago (UTC). Reads exclusively from
    :mod:`index_store`; if the index is empty for the slug the zero payload
    is returned instead of raising.
    """
    now = _now_utc()
    if since is None:
        since = now - _dt.timedelta(hours=_DEFAULT_WINDOW_HOURS)
    since = _as_utc(since)
    window_hours = max(1, int(round((now - since).total_seconds() / 3600)))

    try:
        with index_store.with_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM images "
                "WHERE status = 'sorted' AND topic_slug = ? AND mtime >= ? "
                "ORDER BY mtime DESC",
                (slug, since.timestamp()),
            )
            rows = [index_store.IndexedImage.from_row(row) for row in cur.fetchall()]
    except RuntimeError:
        # index_store.configure() was never called — treat as empty rather
        # than crash a scheduler tick.
        logger.warning(
            "digest: index_store not configured for slug=%r; returning empty payload",
            slug,
        )
        return _empty_payload(slug, window_hours, now)
    except Exception as exc:  # noqa: BLE001 - best-effort read, never raise
        logger.warning("digest: index_store query failed for slug=%r: %s", slug, exc)
        return _empty_payload(slug, window_hours, now)

    if not rows:
        return _empty_payload(slug, window_hours, now)

    totals = {"kept": 0, "borderline": 0, "off_topic": 0, "discarded": 0}
    category_counts: dict[str, int] = {}
    ovr_sum = 0.0
    rel_sum = 0.0
    scored = 0

    keeps: list[tuple[float, index_store.IndexedImage]] = []
    outliers_pool: list[tuple[str, index_store.IndexedImage, float]] = []
    keep_names = set(get_categories())
    all_names = set(get_all_categories())

    for row in rows:
        cat = row.category or "unknown"
        category_counts[cat] = category_counts.get(cat, 0) + 1
        totals[_bucket(row.category)] += 1

        ovr = _score(row.vision_json, "OVR_Quality_Score")
        rel = _score(row.vision_json, "REL_Quality_Score")
        if ovr > 0 or rel > 0:
            ovr_sum += ovr
            rel_sum += rel
            scored += 1

        if row.category in keep_names:
            keeps.append((ovr, row))

        # Outlier signals: a long-reason justification, OR a score sitting in
        # the ambiguity band. Both are worth surfacing to a human reviewer.
        reason = _text_field(row.vision_json, "reason")
        if len(reason) > _LONG_REASON_CHARS:
            outliers_pool.append(("long_reason", row, ovr))
        elif _OUTLIER_SCORE_LO <= ovr <= _OUTLIER_SCORE_HI and row.category in all_names:
            outliers_pool.append(("score_band", row, ovr))

    keeps.sort(key=lambda pair: pair[0], reverse=True)
    top_n = max(0, int(top_n))
    top_keeps = [
        {
            "path": row.path,
            "category": row.category or "",
            "score": round(score, 4),
            "reason": _text_field(row.vision_json, "reason"),
            "caption": _text_field(row.vision_json, "caption"),
        }
        for score, row in keeps[:top_n]
    ]

    category_breakdown = sorted(
        ({"category": name, "count": count} for name, count in category_counts.items()),
        key=lambda entry: (-entry["count"], entry["category"]),
    )

    outliers = [
        {
            "path": row.path,
            "category": row.category or "",
            "reason": _text_field(row.vision_json, "reason"),
            "score": round(score, 4),
        }
        for _kind, row, score in outliers_pool[:_MAX_OUTLIERS]
    ]

    avg_ovr = round(ovr_sum / scored, 4) if scored else 0.0
    avg_rel = round(rel_sum / scored, 4) if scored else 0.0

    return {
        "slug": slug,
        "generated_at": now.isoformat(),
        "window_hours": window_hours,
        "totals": totals,
        "top_keeps": top_keeps,
        "category_breakdown": category_breakdown,
        "outliers": outliers,
        "avg_scores": {
            "OVR_Quality_Score": avg_ovr,
            "REL_Quality_Score": avg_rel,
        },
        "sample_count": len(rows),
    }


# ── Renderers ────────────────────────────────────────────────────────────────

def _truncate(text: str, limit: int = 160) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _basename(path: str) -> str:
    if not path:
        return ""
    # Normalise both separators so this works cross-platform without pulling
    # in os.path (path may have been recorded on the other OS).
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def render_markdown(payload: dict[str, Any]) -> str:
    """Format ``payload`` as a Markdown digest suitable for Discord/Slack."""
    totals = payload.get("totals", {})
    avg = payload.get("avg_scores", {})
    lines: list[str] = [
        f"**cull digest — `{payload.get('slug', '?')}`**",
        (
            f"_window: {payload.get('window_hours', 0)}h · "
            f"generated: {payload.get('generated_at', '')} · "
            f"sample: {payload.get('sample_count', 0)}_"
        ),
        "",
        "**Totals**",
        (
            f"- kept: {totals.get('kept', 0)}  ·  "
            f"borderline: {totals.get('borderline', 0)}  ·  "
            f"off-topic: {totals.get('off_topic', 0)}  ·  "
            f"discarded: {totals.get('discarded', 0)}"
        ),
        (
            f"- avg OVR: {avg.get('OVR_Quality_Score', 0.0):.2f}  ·  "
            f"avg REL: {avg.get('REL_Quality_Score', 0.0):.2f}"
        ),
    ]

    breakdown = payload.get("category_breakdown", [])
    if breakdown:
        lines.extend(["", "**Category breakdown**"])
        for entry in breakdown:
            lines.append(f"- `{entry['category']}`: {entry['count']}")

    top_keeps = payload.get("top_keeps", [])
    if top_keeps:
        lines.extend(["", "**Top keeps**"])
        for i, item in enumerate(top_keeps, start=1):
            reason = _truncate(item.get("reason", ""), 120)
            reason_tail = f" — {reason}" if reason else ""
            lines.append(
                f"{i}. `{_basename(item['path'])}` "
                f"[{item.get('category', '?')} · {item.get('score', 0):.2f}]"
                f"{reason_tail}"
            )

    outliers = payload.get("outliers", [])
    if outliers:
        lines.extend(["", "**Outliers to review**"])
        for item in outliers:
            reason = _truncate(item.get("reason", ""), 160)
            reason_tail = f" — {reason}" if reason else ""
            lines.append(
                f"- `{_basename(item['path'])}` "
                f"[{item.get('category', '?')} · {item.get('score', 0):.2f}]"
                f"{reason_tail}"
            )

    return "\n".join(lines).rstrip() + "\n"


def render_plain(payload: dict[str, Any]) -> str:
    """Format ``payload`` as an email-friendly plain-text digest."""
    totals = payload.get("totals", {})
    avg = payload.get("avg_scores", {})
    lines: list[str] = [
        f"cull digest — {payload.get('slug', '?')}",
        "=" * 60,
        (
            f"window : {payload.get('window_hours', 0)}h\n"
            f"stamp  : {payload.get('generated_at', '')}\n"
            f"sample : {payload.get('sample_count', 0)}"
        ),
        "",
        "Totals",
        "------",
        f"  kept       : {totals.get('kept', 0)}",
        f"  borderline : {totals.get('borderline', 0)}",
        f"  off-topic  : {totals.get('off_topic', 0)}",
        f"  discarded  : {totals.get('discarded', 0)}",
        "",
        "Averages",
        "--------",
        f"  OVR : {avg.get('OVR_Quality_Score', 0.0):.2f}",
        f"  REL : {avg.get('REL_Quality_Score', 0.0):.2f}",
    ]

    breakdown = payload.get("category_breakdown", [])
    if breakdown:
        lines.extend(["", "Category breakdown", "------------------"])
        for entry in breakdown:
            lines.append(f"  {entry['category']:<24} {entry['count']}")

    top_keeps = payload.get("top_keeps", [])
    if top_keeps:
        lines.extend(["", "Top keeps", "---------"])
        for i, item in enumerate(top_keeps, start=1):
            reason = _truncate(item.get("reason", ""), 140)
            lines.append(
                f"  {i:>2}. {_basename(item['path'])} "
                f"[{item.get('category', '?')} · {item.get('score', 0):.2f}]"
            )
            if reason:
                lines.append(f"      {reason}")

    outliers = payload.get("outliers", [])
    if outliers:
        lines.extend(["", "Outliers to review", "------------------"])
        for item in outliers:
            reason = _truncate(item.get("reason", ""), 160)
            lines.append(
                f"  - {_basename(item['path'])} "
                f"[{item.get('category', '?')} · {item.get('score', 0):.2f}]"
            )
            if reason:
                lines.append(f"    {reason}")

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "build_digest",
    "render_markdown",
    "render_plain",
]
