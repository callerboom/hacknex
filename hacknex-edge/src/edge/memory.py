"""
Persistent Conversation Memory (HNX26EPS08).
SQLite + FTS5, thread-safe, zero-latency (all writes are daemon-threaded).
Falls back silently to None — pipeline never breaks if DB is missing.
"""

import sqlite3
import threading
import datetime
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory")


class Memory:
    """
    Persistent per-session conversation store backed by SQLite FTS5.

    Schema
    ------
    turns(id INTEGER PK, ts TEXT, day TEXT, user TEXT, bot TEXT)
    turns_fts  -- FTS5 virtual table shadowing user + bot columns

    API
    ---
    save(user, bot)              -- called from a daemon thread; never blocks TTS
    recent(n=3)                  -- last n (user, bot) pairs for in-context history
    on_day(day)                  -- all turns where day == 'YYYY-MM-DD'
    first_day()                  -- earliest recorded day (str or None)
    search(query, k=2)           -- FTS5 ranked search, returns list of (user, bot)
    clear()                      -- delete all rows ("forget everything")
    """

    def __init__(self, path: str = "data/memory.db"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con: Optional[sqlite3.Connection] = None
        self._available = False
        self._init_db()

    # ------------------------------------------------------------------ init

    def _init_db(self) -> None:
        try:
            self._con = sqlite3.connect(
                str(self._path),
                check_same_thread=False,   # we guard with our own lock
                isolation_level=None,       # autocommit
            )
            cur = self._con.cursor()

            # Enable WAL for concurrent read performance
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")

            # Main turns table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts   TEXT NOT NULL,
                    day  TEXT NOT NULL,
                    user TEXT NOT NULL,
                    bot  TEXT NOT NULL
                )
            """)

            # FTS5 virtual table (content= mirrors main table)
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts
                USING fts5(user, bot, content=turns, content_rowid=id)
            """)

            # Triggers to keep FTS in sync
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                    INSERT INTO turns_fts(rowid, user, bot)
                    VALUES (new.id, new.user, new.bot);
                END
            """)
            cur.execute("""
                CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                    INSERT INTO turns_fts(turns_fts, rowid, user, bot)
                    VALUES ('delete', old.id, old.user, old.bot);
                END
            """)

            self._available = True
            logger.info(f"[Memory] DB ready at {self._path}")
        except Exception as e:
            logger.warning(f"[Memory] DB init failed: {e} — memory disabled")
            self._con = None
            self._available = False

    # ------------------------------------------------------------------ write

    def save(self, user: str, bot: str, _fake_day: Optional[str] = None) -> None:
        """
        Persist one (user, bot) exchange.
        Designed to be called inside a daemon thread — never called before TTS.
        """
        if not self._available or not self._con:
            return
        if "low-power mode" in bot.lower() or "simpler questions" in bot.lower():
            return
        now = datetime.datetime.now()
        ts  = now.isoformat(timespec="seconds")
        day = _fake_day or now.strftime("%Y-%m-%d")
        try:
            with self._lock:
                self._con.execute(
                    "INSERT INTO turns (ts, day, user, bot) VALUES (?, ?, ?, ?)",
                    (ts, day, user, bot),
                )
        except Exception as e:
            logger.warning(f"[Memory] save() failed: {e}")

    # ------------------------------------------------------------------ read

    def recent(self, n: int = 3) -> list[tuple[str, str]]:
        """Return last n (user, bot) pairs, oldest-first, excluding error fallbacks."""
        if not self._available or not self._con:
            return []
        try:
            with self._lock:
                rows = self._con.execute(
                    "SELECT user, bot FROM turns WHERE bot NOT LIKE '%low-power mode%' AND bot NOT LIKE '%simpler questions%' ORDER BY id DESC LIMIT ?", (n,)
                ).fetchall()
            return list(reversed(rows))
        except Exception as e:
            logger.warning(f"[Memory] recent() failed: {e}")
            return []

    def on_day(self, day: str) -> list[tuple[str, str]]:
        """Return all (user, bot) pairs recorded on day (YYYY-MM-DD)."""
        if not self._available or not self._con:
            return []
        try:
            with self._lock:
                rows = self._con.execute(
                    "SELECT user, bot FROM turns WHERE day = ? ORDER BY id", (day,)
                ).fetchall()
            return rows
        except Exception as e:
            logger.warning(f"[Memory] on_day() failed: {e}")
            return []

    def first_day(self) -> Optional[str]:
        """Return earliest recorded day string, or None if empty."""
        if not self._available or not self._con:
            return None
        try:
            with self._lock:
                row = self._con.execute(
                    "SELECT day FROM turns ORDER BY id ASC LIMIT 1"
                ).fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.warning(f"[Memory] first_day() failed: {e}")
            return None

    def search(self, query: str, k: int = 2) -> list[tuple[str, str]]:
        """FTS5 ranked search over user+bot. Returns up to k (user, bot) pairs."""
        if not self._available or not self._con:
            return []
        if not query.strip():
            return []
        try:
            # Escape FTS5 special chars (simple approach)
            safe_q = query.replace('"', '""')
            with self._lock:
                rows = self._con.execute(
                    """SELECT t.user, t.bot
                       FROM turns_fts f
                       JOIN turns t ON t.id = f.rowid
                       WHERE turns_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (safe_q, k),
                ).fetchall()
            return rows
        except Exception as e:
            # FTS errors (e.g. malformed query) must never crash pipeline
            logger.debug(f"[Memory] search() failed: {e}")
            return []

    def clear(self) -> None:
        """Delete all conversation history ('forget everything')."""
        if not self._available or not self._con:
            return
        try:
            with self._lock:
                self._con.execute("DELETE FROM turns")
                # Rebuild FTS index
                self._con.execute("INSERT INTO turns_fts(turns_fts) VALUES ('rebuild')")
            logger.info("[Memory] All history cleared.")
        except Exception as e:
            logger.warning(f"[Memory] clear() failed: {e}")

    @property
    def is_available(self) -> bool:
        return self._available
