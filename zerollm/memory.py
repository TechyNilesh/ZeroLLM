"""Memory management — session history + SQLite persistent storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path


MEMORY_DB = Path.home() / ".cache" / "zerollm" / "memory.db"


class Memory:
    """Manages conversation history with optional persistence.

    - Session memory: in-memory list of recent messages
    - Persistent memory: SQLite storage for summaries across sessions
    """

    def __init__(self, persist: bool = False, session_id: str = "default"):
        self.messages: list[dict[str, str]] = []
        self.persist = persist
        self.session_id = session_id
        self._db: sqlite3.Connection | None = None

        if persist:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for persistent memory."""
        MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(MEMORY_DB))
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                turn_start INTEGER NOT NULL,
                turn_end INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._db.commit()

    def add(self, role: str, content: str) -> None:
        """Add a message to the session history."""
        self.messages.append({"role": role, "content": content})

    def add_system(self, content: str) -> None:
        """Set or update the system prompt (always first message)."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def get_context(self, max_messages: int = 20) -> list[dict[str, str]]:
        """Get recent messages that fit in context.

        Keeps the system prompt (if any) + last N messages.
        """
        if not self.messages:
            return []

        # Always keep system prompt if present
        if self.messages[0]["role"] == "system":
            system = [self.messages[0]]
            history = self.messages[1:]
        else:
            system = []
            history = self.messages

        # Take the most recent messages
        recent = history[-max_messages:]

        return system + recent

    def get_full_history(self) -> list[dict[str, str]]:
        """Get all messages (for display/export)."""
        return list(self.messages)

    def save_summary(self, summary: str, turn_start: int, turn_end: int) -> None:
        """Save a summary of old turns to persistent storage."""
        if not self._db:
            return
        self._db.execute(
            "INSERT INTO summaries (session_id, summary, turn_start, turn_end) VALUES (?, ?, ?, ?)",
            (self.session_id, summary, turn_start, turn_end),
        )
        self._db.commit()

    def load_summaries(self) -> list[str]:
        """Load past summaries for this session."""
        if not self._db:
            return []
        cursor = self._db.execute(
            "SELECT summary FROM summaries WHERE session_id = ? ORDER BY turn_start",
            (self.session_id,),
        )
        return [row[0] for row in cursor.fetchall()]

    def clear(self) -> None:
        """Clear session history (keeps persistent summaries)."""
        # Preserve system prompt if present
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def clear_all(self) -> None:
        """Clear everything including persistent storage."""
        self.messages = []
        if self._db:
            self._db.execute(
                "DELETE FROM summaries WHERE session_id = ?",
                (self.session_id,),
            )
            self._db.commit()

    @property
    def turn_count(self) -> int:
        """Number of user messages in history."""
        return sum(1 for m in self.messages if m["role"] == "user")

    def __del__(self):
        if self._db:
            self._db.close()
