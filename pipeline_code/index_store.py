"""SQLite-backed index of every queue + sorted image.

The dashboard's old approach was to glob ``PIPELINE_SORTED/**/*.vision.json``
on every status / gallery / history call. At 100k images that's 100k stat()s
per request, and ``/api/status`` polls every 5 seconds — F5 was taking
minutes. This module replaces that with a SQLite index updated by a
background poll thread; the dashboard reads stats and gallery pages from
indexed queries (sub-millisecond) and the indexer takes the IO hit once.

Schema (single table):

    images(
        path TEXT PRIMARY KEY,
        status TEXT NOT NULL,           -- 'queue' | 'sorted'
        topic_slug TEXT,                -- subdirectory name under sorted/
        source TEXT NOT NULL,           -- civitai / x / discord / gallery_dl / ...
        category TEXT,                  -- folder under sorted/<slug>/<category>/
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        width INTEGER,
        height INTEGER,
        ovr INTEGER,                    -- vision OVR_Quality_Score
        rel INTEGER,                    -- vision REL_Quality_Score
        quality INTEGER,                -- vision quality_score (1-10)
        nsfw INTEGER,                   -- 0 / 1
        prompt TEXT,                    -- contents of <stem>.txt
        vision_json TEXT                -- raw JSON of <stem>.vision.json
    )

Connection model: one connection per thread (sqlite3 connections are
not safe to share across threads without serialising). The module
exposes a ``with_conn()`` context manager that handles this internally.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

DB_FILENAME = "cull_index.sqlite3"
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# Per-thread connections so each Flask request handler / indexer thread has
# its own connection without locking each other out.
_local = threading.local()
_db_path: Path | None = None
_init_lock = threading.Lock()


# ── Connection management ───────────────────────────────────────────────────

def configure(db_path: Path) -> None:
    """Set the database path. Call once at process start."""
    global _db_path
    _db_path = Path(db_path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure schema on the calling thread's connection.
    with with_conn() as conn:
        _ensure_schema(conn)


@contextmanager
def with_conn() -> Iterator[sqlite3.Connection]:
    """Yield this thread's SQLite connection. Lazily opens + initialises."""
    if _db_path is None:
        raise RuntimeError("index_store.configure(db_path) must be called first")
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(_db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        # WAL mode lets readers (Flask request handlers) and the writer
        # (indexer thread) work concurrently without blocking each other.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        _local.conn = conn
    yield conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    with _init_lock:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                path        TEXT PRIMARY KEY,
                status      TEXT NOT NULL,
                topic_slug  TEXT,
                source      TEXT NOT NULL,
                category    TEXT,
                mtime       REAL NOT NULL,
                size        INTEGER NOT NULL,
                width       INTEGER,
                height      INTEGER,
                ovr         INTEGER,
                rel         INTEGER,
                quality     INTEGER,
                nsfw        INTEGER,
                prompt      TEXT,
                vision_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_status            ON images(status);
            CREATE INDEX IF NOT EXISTS idx_status_category   ON images(status, category);
            CREATE INDEX IF NOT EXISTS idx_status_source     ON images(status, source);
            CREATE INDEX IF NOT EXISTS idx_status_mtime      ON images(status, mtime DESC);
            CREATE INDEX IF NOT EXISTS idx_topic_category    ON images(topic_slug, category);

            CREATE TABLE IF NOT EXISTS scan_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )


# ── Public DTO ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndexedImage:
    path: str
    status: str
    topic_slug: str | None
    source: str
    category: str | None
    mtime: float
    size: int
    width: int | None
    height: int | None
    ovr: int | None
    rel: int | None
    quality: int | None
    nsfw: bool | None
    prompt: str
    vision_json: dict[str, Any] | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "IndexedImage":
        vj = row["vision_json"]
        try:
            parsed = json.loads(vj) if vj else None
        except (TypeError, json.JSONDecodeError):
            parsed = None
        return cls(
            path=row["path"],
            status=row["status"],
            topic_slug=row["topic_slug"],
            source=row["source"],
            category=row["category"],
            mtime=row["mtime"],
            size=row["size"],
            width=row["width"],
            height=row["height"],
            ovr=row["ovr"],
            rel=row["rel"],
            quality=row["quality"],
            nsfw=bool(row["nsfw"]) if row["nsfw"] is not None else None,
            prompt=row["prompt"] or "",
            vision_json=parsed,
        )


# ── Stats queries (the perf-critical ones) ──────────────────────────────────

def total(status: str) -> int:
    with with_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM images WHERE status = ?", (status,))
        return int(cur.fetchone()[0])


def count_by_source(status: str) -> dict[str, int]:
    """{source_name: count} for the given status. Used by /api/status queue panel."""
    with with_conn() as conn:
        cur = conn.execute(
            "SELECT source, COUNT(*) FROM images WHERE status = ? GROUP BY source",
            (status,),
        )
        return {row[0]: int(row[1]) for row in cur.fetchall()}


def count_queue_by_topic_source() -> dict[str, int]:
    """{<topic_slug>/<source>: count} matching the legacy get_queue_stats shape."""
    with with_conn() as conn:
        cur = conn.execute(
            "SELECT topic_slug, source, COUNT(*) FROM images WHERE status = 'queue' GROUP BY topic_slug, source"
        )
        return {f"{row[0] or 'default'}/{row[1]}": int(row[2]) for row in cur.fetchall()}


def count_sorted_by_topic_category() -> dict[str, dict[str, int]]:
    """{topic: {category: count}} matching legacy get_sorted_stats shape."""
    with with_conn() as conn:
        cur = conn.execute(
            "SELECT topic_slug, category, COUNT(*) FROM images "
            "WHERE status = 'sorted' AND category IS NOT NULL "
            "GROUP BY topic_slug, category"
        )
        out: dict[str, dict[str, int]] = {}
        for topic, cat, count in cur.fetchall():
            out.setdefault(topic or "default", {})[cat] = int(count)
        return out


# ── Listing queries (gallery + history + activity) ──────────────────────────

def list_sorted(
    *,
    sources: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    sort: str = "newest",
    limit: int = 60,
    offset: int = 0,
    nsfw: bool | None = None,
    min_ovr: int | None = None,
    min_rel: int | None = None,
) -> tuple[list[IndexedImage], int]:
    """Paginated listing of sorted images with optional filters.

    Returns (items, total_after_filter). Uses indexed queries; doesn't hit the
    filesystem at all.
    """
    sort_clause = {
        "newest": "mtime DESC",
        "oldest": "mtime ASC",
        "ovr": "ovr DESC NULLS LAST, mtime DESC",
        "rel": "rel DESC NULLS LAST, mtime DESC",
        "quality": "quality DESC NULLS LAST, mtime DESC",
    }.get(sort, "mtime DESC")

    where: list[str] = ["status = 'sorted'"]
    params: list[Any] = []
    if sources:
        srcs = list(sources)
        if srcs:
            where.append("source IN (" + ",".join("?" * len(srcs)) + ")")
            params.extend(srcs)
    if categories:
        cats = list(categories)
        if cats:
            where.append("category IN (" + ",".join("?" * len(cats)) + ")")
            params.extend(cats)
    if nsfw is True:
        where.append("nsfw = 1")
    elif nsfw is False:
        where.append("(nsfw = 0 OR nsfw IS NULL)")
    if min_ovr is not None:
        where.append("ovr >= ?")
        params.append(min_ovr)
    if min_rel is not None:
        where.append("rel >= ?")
        params.append(min_rel)
    where_sql = " AND ".join(where)

    with with_conn() as conn:
        total_count = int(conn.execute(
            f"SELECT COUNT(*) FROM images WHERE {where_sql}", params,
        ).fetchone()[0])
        cur = conn.execute(
            f"SELECT * FROM images WHERE {where_sql} "
            f"ORDER BY {sort_clause} LIMIT ? OFFSET ?",
            [*params, int(limit), int(offset)],
        )
        items = [IndexedImage.from_row(row) for row in cur.fetchall()]
        return items, total_count


def list_recent_queue(limit: int = 60) -> list[IndexedImage]:
    """Newest queued images (used by the Queue tab)."""
    with with_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM images WHERE status = 'queue' ORDER BY mtime DESC LIMIT ?",
            (int(limit),),
        )
        return [IndexedImage.from_row(row) for row in cur.fetchall()]


def list_recent_sorted(limit: int = 12) -> list[IndexedImage]:
    """Newest classified images (used by the Activity panel)."""
    with with_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM images WHERE status = 'sorted' ORDER BY mtime DESC LIMIT ?",
            (int(limit),),
        )
        return [IndexedImage.from_row(row) for row in cur.fetchall()]


def distinct_values(column: str, status: str = "sorted") -> list[str]:
    """Distinct non-null values of `column` for the given status (sources/categories)."""
    if column not in {"source", "category", "topic_slug"}:
        raise ValueError(f"unsupported column: {column}")
    with with_conn() as conn:
        cur = conn.execute(
            f"SELECT DISTINCT {column} FROM images WHERE status = ? AND {column} IS NOT NULL ORDER BY {column}",
            (status,),
        )
        return [row[0] for row in cur.fetchall()]


# ── Mutators (used by indexer + writers) ────────────────────────────────────

def upsert(image: dict[str, Any]) -> None:
    """INSERT OR REPLACE one image row."""
    with with_conn() as conn:
        conn.execute(
            """
            INSERT INTO images
              (path, status, topic_slug, source, category, mtime, size,
               width, height, ovr, rel, quality, nsfw, prompt, vision_json)
            VALUES (:path, :status, :topic_slug, :source, :category, :mtime, :size,
                    :width, :height, :ovr, :rel, :quality, :nsfw, :prompt, :vision_json)
            ON CONFLICT(path) DO UPDATE SET
              status      = excluded.status,
              topic_slug  = excluded.topic_slug,
              source      = excluded.source,
              category    = excluded.category,
              mtime       = excluded.mtime,
              size        = excluded.size,
              width       = excluded.width,
              height      = excluded.height,
              ovr         = excluded.ovr,
              rel         = excluded.rel,
              quality     = excluded.quality,
              nsfw        = excluded.nsfw,
              prompt      = excluded.prompt,
              vision_json = excluded.vision_json
            """,
            image,
        )


def upsert_many(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with with_conn() as conn:
        conn.execute("BEGIN")
        try:
            conn.executemany(
                """
                INSERT INTO images
                  (path, status, topic_slug, source, category, mtime, size,
                   width, height, ovr, rel, quality, nsfw, prompt, vision_json)
                VALUES (:path, :status, :topic_slug, :source, :category, :mtime, :size,
                        :width, :height, :ovr, :rel, :quality, :nsfw, :prompt, :vision_json)
                ON CONFLICT(path) DO UPDATE SET
                  status      = excluded.status,
                  topic_slug  = excluded.topic_slug,
                  source      = excluded.source,
                  category    = excluded.category,
                  mtime       = excluded.mtime,
                  size        = excluded.size,
                  width       = excluded.width,
                  height      = excluded.height,
                  ovr         = excluded.ovr,
                  rel         = excluded.rel,
                  quality     = excluded.quality,
                  nsfw        = excluded.nsfw,
                  prompt      = excluded.prompt,
                  vision_json = excluded.vision_json
                """,
                rows,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def delete_paths(paths: Iterable[str]) -> int:
    """Bulk-delete rows for paths that no longer exist on disk."""
    paths = list(paths)
    if not paths:
        return 0
    with with_conn() as conn:
        conn.execute("BEGIN")
        try:
            cur = conn.executemany("DELETE FROM images WHERE path = ?", [(p,) for p in paths])
            conn.execute("COMMIT")
            return cur.rowcount or 0
        except Exception:
            conn.execute("ROLLBACK")
            raise


def existing_paths(status: str | None = None) -> dict[str, float]:
    """{path: mtime} for every row matching status (or all rows if None)."""
    with with_conn() as conn:
        if status is None:
            cur = conn.execute("SELECT path, mtime FROM images")
        else:
            cur = conn.execute("SELECT path, mtime FROM images WHERE status = ?", (status,))
        return {row[0]: row[1] for row in cur.fetchall()}


def set_meta(key: str, value: str) -> None:
    with with_conn() as conn:
        conn.execute(
            "INSERT INTO scan_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def get_meta(key: str) -> str | None:
    with with_conn() as conn:
        cur = conn.execute("SELECT value FROM scan_meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None


# ── Indexer (filesystem -> SQLite) ──────────────────────────────────────────
#
# Single-pass scan. For each image found on disk, look up its mtime in the
# index; if missing or stale, read the .vision.json + .txt and upsert. After
# the scan, delete index rows whose path no longer exists on disk.

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _is_in_archive(path: Path) -> bool:
    """Skip dot-prefixed dirs and the `.archive` sweep directory."""
    return any(part.startswith(".") for part in path.parts) or "_archive" in path.parts


def _read_vision_payload(meta_path: Path) -> dict[str, Any]:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_prompt(image_path: Path) -> str:
    txt = image_path.with_suffix(".txt")
    if not txt.exists():
        return ""
    try:
        return txt.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _resolve_dims(image_path: Path) -> tuple[int | None, int | None]:
    """PIL is the only reliable way; called sparingly during initial backfill."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (None, None)


def _row_for_sorted(image_path: Path, sorted_root: Path) -> dict[str, Any] | None:
    """Build an index row for a sorted image. Returns None if it's invalid."""
    rel = image_path.relative_to(sorted_root)
    parts = rel.parts
    if len(parts) < 3:
        return None  # expected: <topic_slug>/<category>/<source>/<file>
    topic_slug, category, source = parts[0], parts[1], parts[2]
    meta = image_path.parent / f"{image_path.stem}.vision.json"
    payload = _read_vision_payload(meta) if meta.exists() else {}
    width, height = (
        int(payload["width"]) if "width" in payload else None,
        int(payload["height"]) if "height" in payload else None,
    )
    if width is None or height is None:
        # Backfill dimensions once; PIL is slow but only fires on new rows.
        width, height = _resolve_dims(image_path)
    try:
        stat = image_path.stat()
    except OSError:
        return None
    return {
        "path": str(image_path),
        "status": "sorted",
        "topic_slug": topic_slug,
        "source": source,
        "category": category,
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "width": width,
        "height": height,
        "ovr": payload.get("OVR_Quality_Score"),
        "rel": payload.get("REL_Quality_Score"),
        "quality": payload.get("quality_score"),
        "nsfw": 1 if payload.get("nsfw") else 0,
        "prompt": _read_prompt(image_path),
        "vision_json": json.dumps(payload, ensure_ascii=False) if payload else None,
    }


def _row_for_queue(image_path: Path, queue_root: Path) -> dict[str, Any] | None:
    rel = image_path.relative_to(queue_root)
    parts = rel.parts
    if len(parts) < 2:
        return None  # expected: <topic_slug>/<source>/<file>
    topic_slug = parts[0] if len(parts) >= 3 else None
    source = parts[1] if len(parts) >= 3 else parts[0]
    try:
        stat = image_path.stat()
    except OSError:
        return None
    return {
        "path": str(image_path),
        "status": "queue",
        "topic_slug": topic_slug,
        "source": source,
        "category": None,
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "width": None,
        "height": None,
        "ovr": None,
        "rel": None,
        "quality": None,
        "nsfw": None,
        "prompt": _read_prompt(image_path),
        "vision_json": None,
    }


@dataclass
class ScanReport:
    sorted_added: int = 0
    sorted_updated: int = 0
    queue_added: int = 0
    queue_updated: int = 0
    deleted: int = 0
    duration_seconds: float = 0.0


def scan(*, queue_root: Path, sorted_root: Path) -> ScanReport:
    """Single-pass index update. Idempotent and safe to run repeatedly."""
    start = time.monotonic()
    report = ScanReport()
    existing = existing_paths()
    seen: set[str] = set()
    upserts: list[dict[str, Any]] = []

    if sorted_root.exists():
        for image_path in sorted_root.glob("**/*"):
            if not _is_image(image_path) or _is_in_archive(image_path):
                continue
            try:
                if not image_path.is_file():
                    continue
            except OSError:
                continue
            key = str(image_path)
            seen.add(key)
            try:
                mtime = image_path.stat().st_mtime
            except OSError:
                continue
            cached_mtime = existing.get(key)
            if cached_mtime is not None and cached_mtime >= mtime - 1e-6:
                continue  # unchanged
            row = _row_for_sorted(image_path, sorted_root)
            if row is None:
                continue
            upserts.append(row)
            if cached_mtime is None:
                report.sorted_added += 1
            else:
                report.sorted_updated += 1

    if queue_root.exists():
        for image_path in queue_root.glob("**/*"):
            if not _is_image(image_path) or _is_in_archive(image_path):
                continue
            try:
                if not image_path.is_file():
                    continue
            except OSError:
                continue
            # Skip in-flight files (workers rename to <stem>.processing during classification).
            if image_path.suffix == ".processing":
                continue
            key = str(image_path)
            seen.add(key)
            try:
                mtime = image_path.stat().st_mtime
            except OSError:
                continue
            cached_mtime = existing.get(key)
            if cached_mtime is not None and cached_mtime >= mtime - 1e-6:
                continue
            row = _row_for_queue(image_path, queue_root)
            if row is None:
                continue
            upserts.append(row)
            if cached_mtime is None:
                report.queue_added += 1
            else:
                report.queue_updated += 1

    # Bulk write in batches of 500 to keep transactions snappy.
    BATCH = 500
    for i in range(0, len(upserts), BATCH):
        upsert_many(upserts[i:i + BATCH])

    # Anything in the index that wasn't seen on disk has been deleted/moved.
    stale = [p for p in existing if p not in seen]
    report.deleted = delete_paths(stale)

    set_meta("last_scan_at", str(time.time()))
    set_meta("last_scan_report", json.dumps({
        "sorted_added": report.sorted_added,
        "sorted_updated": report.sorted_updated,
        "queue_added": report.queue_added,
        "queue_updated": report.queue_updated,
        "deleted": report.deleted,
    }))
    report.duration_seconds = time.monotonic() - start
    return report


# ── Background indexer thread ───────────────────────────────────────────────

_indexer_thread: threading.Thread | None = None
_indexer_stop = threading.Event()


def start_background_indexer(
    *, queue_root: Path, sorted_root: Path, interval_seconds: float = 30.0,
) -> threading.Thread:
    """Spawn a daemon thread that re-scans every interval_seconds.

    Re-entrant: a second call is a no-op if a thread is already running.
    """
    global _indexer_thread
    if _indexer_thread is not None and _indexer_thread.is_alive():
        return _indexer_thread

    def _loop() -> None:
        # Initial backfill — likely the slow run on first start.
        try:
            r = scan(queue_root=queue_root, sorted_root=sorted_root)
            logger.info(
                "indexer initial scan: +%d sorted, +%d queue, ~%d updated, -%d removed in %.1fs",
                r.sorted_added, r.queue_added,
                r.sorted_updated + r.queue_updated, r.deleted, r.duration_seconds,
            )
        except Exception as exc:
            logger.warning("indexer initial scan failed: %s", exc)

        while not _indexer_stop.wait(interval_seconds):
            try:
                r = scan(queue_root=queue_root, sorted_root=sorted_root)
                if r.sorted_added or r.queue_added or r.deleted:
                    logger.info(
                        "indexer tick: +%d sorted, +%d queue, -%d removed in %.1fs",
                        r.sorted_added, r.queue_added, r.deleted, r.duration_seconds,
                    )
            except Exception as exc:
                logger.warning("indexer tick failed: %s", exc)

    _indexer_thread = threading.Thread(target=_loop, name="cull-indexer", daemon=True)
    _indexer_thread.start()
    return _indexer_thread


def stop_background_indexer() -> None:
    _indexer_stop.set()


__all__ = [
    "DB_FILENAME",
    "IndexedImage",
    "ScanReport",
    "configure",
    "with_conn",
    "total",
    "count_by_source",
    "count_queue_by_topic_source",
    "count_sorted_by_topic_category",
    "list_sorted",
    "list_recent_queue",
    "list_recent_sorted",
    "distinct_values",
    "upsert",
    "upsert_many",
    "delete_paths",
    "existing_paths",
    "scan",
    "start_background_indexer",
    "stop_background_indexer",
    "get_meta",
    "set_meta",
]
