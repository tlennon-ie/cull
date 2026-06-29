"""Active-learning loop — turn user corrections into taste signals.

cull's positioning is *automating taste*. Every time a user overrides the
vision worker — accepting, rejecting, or recategorising an image in the
dashboard — that correction is a cheap, high-signal label. This module records
those corrections per-slug and distils them into two **deterministic,
explainable** outputs:

  1. ``suggest_threshold_adjustments(slug)`` — bounded nudges to the per-preset
     OVR / REL score thresholds, derived from an EWMA of the scores on
     *false-keeps* (model kept, user dropped → bar too low → push up) and
     *false-drops* (model dropped, user kept → bar too high → push down).
  2. ``exemplars(slug, category, k)`` — the most recent corrections targeting a
     category, shaped as few-shot positives/negatives the classification prompt
     can inject.

Plus ``stats(slug)`` (correction counts per corrected category, for the
dashboard) and ``record_correction(...)`` (the write path the dashboard /
worker call on a manual re-label).

These are **pure module functions** — the dashboard and vision workers import
and call them; this module never imports them back. Persistence lives at
``data/active_learning/<slug>.json`` (``paths.base_dir()``), written atomically
and read tolerantly so a torn or corrupt file degrades to "no signal" rather
than crashing the pipeline.

Design notes (why it's not a black box):
  * "Dropped" == the corrected category is terminal (DISCARD / CORRUPT). The
    terminal set is read from ``categories`` so it tracks the taxonomy.
  * A recategorisation between two *non-terminal* buckets is a taxonomy fix,
    not a quality-threshold signal — it updates stats + exemplars but never
    moves a threshold.
  * EWMA (newest-weighted) so recent taste dominates stale corrections, with a
    minimum-support gate (``MIN_SUPPORT``) and a hard cap (``MAX_DELTA``) so a
    handful of clicks can't lurch the thresholds.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paths import base_dir as _default_base_dir
from paths import validate_slug as _validate_slug
from pipeline_logging import get_logger

try:  # taxonomy-aware terminal set; degrade gracefully if unavailable
    from categories import SYSTEM_TERMINAL as _SYSTEM_TERMINAL
except Exception:  # pragma: no cover - categories always importable in practice
    _SYSTEM_TERMINAL = ("DISCARD", "CORRUPT")

logger = get_logger(__name__)

# ── Tunables (named constants — no magic numbers) ────────────────────────────
SCHEMA_VERSION = 1
EWMA_ALPHA = 0.3          # newest-weighted smoothing factor for score EWMAs
MIN_SUPPORT = 3           # need at least this many qualifying corrections to move
MAX_DELTA = 15            # hard cap (points) on any single threshold suggestion
DEFAULT_EXEMPLAR_K = 3    # few-shot exemplars returned per category by default
MAX_CORRECTIONS = 500     # ring-buffer cap per slug so the file stays small
# Base nudge (points) applied once MIN_SUPPORT is reached, plus an extra step
# per qualifying correction beyond the floor. Both feed a MAX_DELTA-capped sum,
# so more consistent corrections → a larger (but bounded) move.
BASE_STEP = 2.0
STEP_PER_CORRECTION = 1.0

# Terminal categories define "dropped" (rejected). Anything else is a keep.
TERMINAL: frozenset[str] = frozenset(_SYSTEM_TERMINAL)

# One global lock — writes are infrequent (human clicks) so contention is nil.
_lock = threading.Lock()


# ── Persistence ──────────────────────────────────────────────────────────────

def _store_dir() -> Path:
    """Directory holding per-slug correction logs. Under the global base dir."""
    return _default_base_dir() / "active_learning"


def _store_path(slug: str) -> Path:
    # Sanitise the slug at the single point where it becomes a filesystem path —
    # the charset barrier rejects '/', '\\' and '.' so no traversal is possible.
    safe_slug = _validate_slug(slug)
    return _store_dir() / f"{safe_slug}.json"


def _empty_state() -> dict[str, Any]:
    return {"version": SCHEMA_VERSION, "corrections": []}


def _load(slug: str) -> dict[str, Any]:
    """Read a slug's state. Tolerant: missing/corrupt → empty state."""
    path = _store_path(slug)
    if not path.exists():
        return _empty_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("active_learning state corrupt, ignoring: %s", path)
        return _empty_state()
    if not isinstance(payload, dict) or not isinstance(payload.get("corrections"), list):
        return _empty_state()
    return payload


def _save(slug: str, state: dict[str, Any]) -> None:
    """Persist a slug's state atomically (tempfile + os.replace)."""
    path = _store_path(slug)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError as exc:  # pragma: no cover - disk-full / permission edge
        logger.warning("active_learning state flush failed for %s: %s", slug, exc)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _coerce_score(scores: dict[str, Any], *keys: str) -> float | None:
    """Pull the first present, parseable score from ``scores`` under ``keys``.

    Accepts both the vision-schema keys (``OVR_Quality_Score``) and the index
    aliases (``ovr``) so either caller shape works.
    """
    if not isinstance(scores, dict):
        return None
    for key in keys:
        if key in scores and scores[key] is not None:
            try:
                return max(0.0, min(100.0, float(scores[key])))
            except (TypeError, ValueError):
                continue
    return None


def _is_terminal(category: str) -> bool:
    return (category or "").strip() in TERMINAL


def _bounded_int(value: float) -> int:
    """Round to an int and clamp to ``[-MAX_DELTA, MAX_DELTA]``."""
    return max(-MAX_DELTA, min(MAX_DELTA, int(round(value))))


def _ewma(values: list[float]) -> float | None:
    """Exponentially-weighted moving average, oldest→newest.

    ``values`` must be in chronological order; the most recent value carries the
    most weight. Returns None for an empty list.
    """
    acc: float | None = None
    for v in values:
        acc = v if acc is None else (EWMA_ALPHA * v + (1.0 - EWMA_ALPHA) * acc)
    return acc


# ── Public API ───────────────────────────────────────────────────────────────

def record_correction(
    slug: str,
    image_key: str,
    predicted_category: str,
    corrected_category: str,
    scores: dict[str, Any],
    *,
    caption: str | None = None,
    note: str | None = None,
) -> None:
    """Record one user correction for ``slug``.

    ``image_key`` is the stable identity of the image (filename stem / path);
    re-recording the same key replaces the prior entry so a flip-flopped
    decision reflects only the latest call. Blank keys are ignored. Scores are
    stored if parseable (schema keys or ``ovr``/``rel`` aliases) and simply
    omitted otherwise — a correction with no usable scores still counts toward
    stats and exemplars, it just can't inform the threshold EWMA.
    """
    key = (image_key or "").strip()
    if not key:
        return

    ovr = _coerce_score(scores, "OVR_Quality_Score", "ovr")
    rel = _coerce_score(scores, "REL_Quality_Score", "rel")

    entry: dict[str, Any] = {
        "image_key": key,
        "predicted": (predicted_category or "").strip(),
        "corrected": (corrected_category or "").strip(),
        "ovr": ovr,
        "rel": rel,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if caption:
        entry["caption"] = caption
    if note:
        entry["note"] = note

    with _lock:
        state = _load(slug)
        corrections: list[dict[str, Any]] = [
            c for c in state.get("corrections", [])
            if isinstance(c, dict) and c.get("image_key") != key
        ]
        corrections.append(entry)
        # Ring-buffer: keep the most recent MAX_CORRECTIONS so the file stays small.
        if len(corrections) > MAX_CORRECTIONS:
            corrections = corrections[-MAX_CORRECTIONS:]
        state["corrections"] = corrections
        state["version"] = SCHEMA_VERSION
        _save(slug, state)


def stats(slug: str) -> dict[str, int]:
    """Correction counts keyed by the *corrected* category (for the dashboard)."""
    out: dict[str, int] = {}
    for c in _load(slug).get("corrections", []):
        corrected = (c.get("corrected") or "").strip()
        if not corrected:
            continue
        out[corrected] = out.get(corrected, 0) + 1
    return out


def suggest_threshold_adjustments(slug: str) -> dict[str, Any]:
    """Suggest bounded, explainable nudges to the OVR / REL score thresholds.

    Returns ``{"ovr_min_delta": int, "rel_min_delta": int, "rationale": str}``.

    Logic (deterministic):
      * **false-keep** — model assigned a *non-terminal* category, user moved it
        to a *terminal* one (DISCARD/CORRUPT). Its score cleared the bar yet the
        user rejected it ⇒ the bar was too low ⇒ push UP.
      * **false-drop** — model assigned a *terminal* category, user moved it to a
        *keep* bucket. Its score fell below the bar yet the user wanted it ⇒ the
        bar was too high ⇒ push DOWN.
      * Recategorisations between two non-terminal buckets are taxonomy fixes and
        are ignored here.

    The magnitude grows with how many qualifying corrections support the move
    (a base step once ``MIN_SUPPORT`` is reached, plus a per-correction step),
    weighted by the EWMA of the population's scores so a more decisive model
    miss nudges harder, all clamped to ``MAX_DELTA``. Fewer than ``MIN_SUPPORT``
    qualifying corrections ⇒ no movement.
    """
    corrections = _load(slug).get("corrections", [])

    fk_ovr: list[float] = []
    fk_rel: list[float] = []
    fd_ovr: list[float] = []
    fd_rel: list[float] = []

    for c in corrections:
        predicted = (c.get("predicted") or "").strip()
        corrected = (c.get("corrected") or "").strip()
        pred_terminal = _is_terminal(predicted)
        corr_terminal = _is_terminal(corrected)
        ovr = c.get("ovr")
        rel = c.get("rel")

        if (not pred_terminal) and corr_terminal:
            # false-keep
            if isinstance(ovr, (int, float)):
                fk_ovr.append(float(ovr))
            if isinstance(rel, (int, float)):
                fk_rel.append(float(rel))
        elif pred_terminal and (not corr_terminal):
            # false-drop
            if isinstance(ovr, (int, float)):
                fd_ovr.append(float(ovr))
            if isinstance(rel, (int, float)):
                fd_rel.append(float(rel))
        # else: keep↔keep recategorisation → no threshold signal

    ovr_delta, ovr_reason = _axis_delta(fk_ovr, fd_ovr, "OVR")
    rel_delta, rel_reason = _axis_delta(fk_rel, fd_rel, "REL")

    reasons = [r for r in (ovr_reason, rel_reason) if r]
    if not reasons:
        rationale = (
            "No threshold movement: not enough qualifying corrections "
            f"(need {MIN_SUPPORT} false-keeps or false-drops with scores)."
        )
    else:
        rationale = " ".join(reasons)

    return {
        "ovr_min_delta": ovr_delta,
        "rel_min_delta": rel_delta,
        "rationale": rationale,
    }


def _population_pressure(scores: list[float]) -> float:
    """Unsigned nudge magnitude for one population of qualifying corrections.

    Zero below ``MIN_SUPPORT`` (not enough evidence). Otherwise a base step plus
    a per-correction step for each correction beyond the floor, weighted by the
    EWMA of the population's scores (a more confident model miss → a firmer
    nudge). The caller clamps the signed sum to ``MAX_DELTA``.
    """
    n = len(scores)
    if n < MIN_SUPPORT:
        return 0.0
    ewma = _ewma(scores) or 0.0
    # Weight in [~0.1, 1.0] so a genuine signal never rounds to zero even when
    # the scores cluster near the bottom of the axis.
    weight = max(0.1, ewma / 100.0)
    raw = BASE_STEP + STEP_PER_CORRECTION * (n - MIN_SUPPORT)
    return raw * weight


def _axis_delta(
    false_keeps: list[float],
    false_drops: list[float],
    label: str,
) -> tuple[int, str]:
    """Compute one axis's bounded delta + a human-readable reason fragment.

    Positive favours raising the bar (driven by false-keeps), negative favours
    lowering it (driven by false-drops). The net is whichever population has the
    stronger, better-supported signal; equal-and-opposite signals cancel.
    """
    up = _population_pressure(false_keeps)
    down = _population_pressure(false_drops)
    reason_bits: list[str] = []

    if up > 0:
        reason_bits.append(
            f"{label}: {len(false_keeps)} false-keeps (model kept, user sent to "
            f"DISCARD) EWMA {(_ewma(false_keeps) or 0.0):.0f} → raise the bar."
        )
    if down > 0:
        reason_bits.append(
            f"{label}: {len(false_drops)} false-drops (model dropped, user kept) "
            f"EWMA {(_ewma(false_drops) or 0.0):.0f} → lower the bar."
        )

    delta = _bounded_int(up - down)
    if delta == 0 and (up or down):
        reason_bits.append(f"{label}: signals balanced → net zero.")
    return delta, " ".join(reason_bits)


def exemplars(slug: str, category: str, k: int = DEFAULT_EXEMPLAR_K) -> list[dict[str, Any]]:
    """Return up to ``k`` recent corrections that targeted ``category``.

    Shaped as ``{"image_key", "caption"?, "note"}`` for few-shot injection into
    the classification prompt — positives when ``category`` is a keep bucket,
    negatives when it's terminal. Most-recent first (the corrections list is
    chronological; we walk it from the end).
    """
    want = (category or "").strip()
    if not want or k <= 0:
        return []

    out: list[dict[str, Any]] = []
    for c in reversed(_load(slug).get("corrections", [])):
        if (c.get("corrected") or "").strip() != want:
            continue
        ex: dict[str, Any] = {"image_key": c.get("image_key", "")}
        caption = c.get("caption")
        if caption:
            ex["caption"] = caption
        # Always provide a note documenting the predicted→corrected transition;
        # it's the cheapest possible explanation of *why* this is an exemplar.
        note = c.get("note")
        if not note:
            predicted = (c.get("predicted") or "?").strip() or "?"
            note = f"user moved {predicted} → {want}"
        ex["note"] = note
        out.append(ex)
        if len(out) >= k:
            break
    return out


__all__ = [
    "SCHEMA_VERSION",
    "EWMA_ALPHA",
    "MIN_SUPPORT",
    "MAX_DELTA",
    "DEFAULT_EXEMPLAR_K",
    "MAX_CORRECTIONS",
    "TERMINAL",
    "record_correction",
    "stats",
    "suggest_threshold_adjustments",
    "exemplars",
]
