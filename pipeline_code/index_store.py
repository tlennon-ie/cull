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
        vision_json TEXT,               -- raw JSON of <stem>.vision.json
        phash TEXT                      -- 16-char hex dHash (nullable; see phash_dedup)
    )

The ``phash`` column is added by a backward-safe migration (``ALTER TABLE ADD
COLUMN`` guarded by ``PRAGMA table_info``) so databases created before this
column shipped keep working — existing rows simply get ``NULL``.

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
                vision_json TEXT,
                phash       TEXT
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
        _migrate_add_columns(conn)


# Backward-safe migrations. Each entry is (column_name, column_def). Adding a
# nullable column via ``ALTER TABLE ADD COLUMN`` is the one schema change SQLite
# applies in-place without a table rewrite, so old databases upgrade cheaply and
# pre-existing rows get NULL for the new column.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("phash", "TEXT"),
)


def _migrate_add_columns(conn: sqlite3.Connection) -> None:
    """Add any missing columns from ``_ADDED_COLUMNS`` to the images table.

    Idempotent: inspects ``PRAGMA table_info`` and only issues ``ALTER TABLE``
    for columns that don't already exist, so running it on every connection /
    process start is harmless.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    for column, definition in _ADDED_COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE images ADD COLUMN {column} {definition}")


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


def set_phash(key: str, phash: str | None) -> None:
    """Store the perceptual hash for an already-indexed image row.

    No-op if the row doesn't exist (the indexer upserts rows; this only
    annotates them). Pass ``None`` to clear a previously-stored hash.
    """
    with with_conn() as conn:
        conn.execute(
            "UPDATE images SET phash = ? WHERE path = ?",
            (phash, key),
        )


def iter_phashes(
    slug: str | None = None, *, status: str | None = None,
) -> Iterator[tuple[str, str | None]]:
    """Yield ``(path, phash)`` for indexed images, optionally filtered.

    Used by the near-duplicate scan (``phash_dedup.find_near_duplicates``).
    Filter by ``slug`` (``topic_slug``) and/or ``status`` (``queue`` / ``sorted``).
    Rows whose ``phash`` was never computed yield ``None`` so callers can decide
    whether to skip or backfill them.
    """
    where: list[str] = []
    params: list[Any] = []
    if slug is not None:
        where.append("topic_slug = ?")
        params.append(slug)
    if status is not None:
        where.append("status = ?")
        params.append(status)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    with with_conn() as conn:
        cur = conn.execute(
            f"SELECT path, phash FROM images{where_sql}", params,
        )
        for row in cur.fetchall():
            yield row[0], row[1]


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


# Commit pending rows to SQLite every BATCH_SIZE upserts so observers (the
# dashboard, the user) see counts climb during the scan instead of waiting
# for the entire walk to finish in memory. Also emit a console line every
# PROGRESS_LOG_INTERVAL files so a user watching the launcher terminal can
# see the cold backfill making progress.
_BATCH_SIZE = 500
_PROGRESS_LOG_INTERVAL = 1000


def scan(
    *,
    queue_root: Path,
    sorted_root: Path,
    progress_callback: Any = None,
    log_progress: bool = True,
) -> ScanReport:
    """Single-pass index update. Idempotent, streams commits during the walk.

    progress_callback(files_seen, files_added) is invoked every BATCH_SIZE
    new rows and at end of scan. Pass None to disable; the indexer thread
    uses it to update scan_meta keys for the API.
    """
    start = time.monotonic()
    report = ScanReport()
    existing = existing_paths()
    seen: set[str] = set()
    pending: list[dict[str, Any]] = []
    files_seen = 0
    files_added = 0
    last_log = 0

    set_meta("scan_in_progress", "1")
    set_meta("scan_started_at", str(time.time()))
    set_meta("scan_files_seen", "0")
    set_meta("scan_files_added", "0")

    def _flush() -> None:
        if not pending:
            return
        upsert_many(pending)
        pending.clear()

    def _maybe_progress() -> None:
        nonlocal last_log
        set_meta("scan_files_seen", str(files_seen))
        set_meta("scan_files_added", str(files_added))
        if progress_callback:
            try:
                progress_callback(files_seen, files_added)
            except Exception:
                pass
        if log_progress and files_seen - last_log >= _PROGRESS_LOG_INTERVAL:
            print(
                f"[indexer] scanned {files_seen:>7} files, "
                f"{files_added:>6} new/changed rows committed",
                flush=True,
            )
            last_log = files_seen

    def _ingest(image_path: Path, row_builder: Any, kind: str) -> None:
        nonlocal files_seen, files_added
        try:
            if not image_path.is_file():
                return
        except OSError:
            return
        key = str(image_path)
        seen.add(key)
        files_seen += 1
        try:
            mtime = image_path.stat().st_mtime
        except OSError:
            return
        cached_mtime = existing.get(key)
        if cached_mtime is not None and cached_mtime >= mtime - 1e-6:
            return  # unchanged — skip the expensive row build
        row = row_builder(image_path)
        if row is None:
            return
        pending.append(row)
        files_added += 1
        if cached_mtime is None:
            if kind == "sorted":
                report.sorted_added += 1
            else:
                report.queue_added += 1
        else:
            if kind == "sorted":
                report.sorted_updated += 1
            else:
                report.queue_updated += 1
        if len(pending) >= _BATCH_SIZE:
            _flush()
            _maybe_progress()

    try:
        if sorted_root.exists():
            for image_path in sorted_root.glob("**/*"):
                if not _is_image(image_path) or _is_in_archive(image_path):
                    continue
                _ingest(
                    image_path,
                    lambda p: _row_for_sorted(p, sorted_root),
                    "sorted",
                )

        if queue_root.exists():
            for image_path in queue_root.glob("**/*"):
                if not _is_image(image_path) or _is_in_archive(image_path):
                    continue
                if image_path.suffix == ".processing":
                    continue  # in-flight; vision worker owns it
                _ingest(
                    image_path,
                    lambda p: _row_for_queue(p, queue_root),
                    "queue",
                )

        _flush()

        # Anything in the index that wasn't seen on disk has been deleted/moved.
        stale = [p for p in existing if p not in seen]
        report.deleted = delete_paths(stale)
    finally:
        # Always release the in-progress flag, even if the walk crashed.
        set_meta("scan_in_progress", "0")

    report.duration_seconds = time.monotonic() - start
    set_meta("last_scan_at", str(time.time()))
    set_meta("last_scan_report", json.dumps({
        "sorted_added": report.sorted_added,
        "sorted_updated": report.sorted_updated,
        "queue_added": report.queue_added,
        "queue_updated": report.queue_updated,
        "deleted": report.deleted,
        "duration_seconds": round(report.duration_seconds, 1),
        "files_seen": files_seen,
    }))
    if log_progress:
        print(
            f"[indexer] scan complete: +{report.sorted_added} sorted, "
            f"+{report.queue_added} queue, ~{report.sorted_updated + report.queue_updated} updated, "
            f"-{report.deleted} removed in {report.duration_seconds:.1f}s",
            flush=True,
        )
    _maybe_progress()
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
        # Initial backfill — likely the slow run on first start. The console
        # progress lines come out of scan() itself; here we just bracket them.
        existing_total = total("queue") + total("sorted")
        if existing_total == 0:
            print(
                "[indexer] cold backfill starting — scanning the queue + sorted "
                "trees for the first time. Stats and gallery will populate as "
                "rows commit; you can watch progress at /api/index/status.",
                flush=True,
            )
        else:
            print(
                f"[indexer] resuming with {existing_total:,} rows already indexed; "
                f"checking for new / changed files",
                flush=True,
            )
        try:
            scan(queue_root=queue_root, sorted_root=sorted_root)
        except Exception as exc:
            print(f"[indexer] initial scan FAILED: {exc}", flush=True)
            logger.warning("indexer initial scan failed: %s", exc)

        while not _indexer_stop.wait(interval_seconds):
            try:
                r = scan(queue_root=queue_root, sorted_root=sorted_root, log_progress=False)
                # Quiet by default on incremental ticks — only print when
                # something actually changed, otherwise the launcher log
                # would fill with no-op lines every 30s.
                if r.sorted_added or r.queue_added or r.deleted:
                    print(
                        f"[indexer] tick: +{r.sorted_added} sorted, "
                        f"+{r.queue_added} queue, -{r.deleted} removed "
                        f"in {r.duration_seconds:.1f}s",
                        flush=True,
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
    "set_phash",
    "iter_phashes",
    "scan",
    "start_background_indexer",
    "stop_background_indexer",
    "get_meta",
    "set_meta",
]
