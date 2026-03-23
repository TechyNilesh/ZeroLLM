"""Memory management — session history + auto-summarization + persistent storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path


MEMORY_DB = Path.home() / ".cache" / "zerollm" / "memory.db"


class Memory:
    """Manages conversation history with optional persistence and auto-summarization.

    - Session memory: in-memory list of recent messages
    - Auto-summarization: when history exceeds max_messages, old turns get summarized
    - Persistent memory: SQLite storage for summaries across sessions
    """

    def __init__(
        self,
        persist: bool = False,
        session_id: str = "default",
        max_messages: int = 20,
        summarize_after: int = 16,
    ):
        """Initialize memory.

        Args:
            persist: Enable SQLite persistent storage for summaries.
            session_id: Unique session ID for persistent storage.
            max_messages: Maximum messages to keep in context.
            summarize_after: Trigger summarization when history exceeds this count.
        """
        self.messages: list[dict[str, str]] = []
        self.persist = persist
        self.session_id = session_id
        self.max_messages = max_messages
        self.summarize_after = summarize_after
        self._summaries: list[str] = []
        self._summarizer = None  # lazy-loaded LLM for summarization
        self._db: sqlite3.Connection | None = None
        self._total_turns: int = 0

        if persist:
            self._init_db()
            self._summaries = self.load_summaries()

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
        if role == "user":
            self._total_turns += 1

    def add_system(self, content: str) -> None:
        """Set or update the system prompt (always first message)."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def get_context(self, max_messages: int | None = None) -> list[dict[str, str]]:
        """Get messages for the LLM context window.

        Includes:
        1. System prompt (if any)
        2. Summaries of old conversations (if any)
        3. Recent messages (last max_messages)
        """
        limit = max_messages or self.max_messages

        if not self.messages:
            return []

        # Separate system prompt from history
        if self.messages[0]["role"] == "system":
            system = [self.messages[0]]
            history = self.messages[1:]
        else:
            system = []
            history = self.messages

        # Build context with summaries
        context = list(system)

        # Inject summaries as a system-level context message
        if self._summaries:
            summary_text = "\n".join(self._summaries)
            context.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{summary_text}",
            })

        # Add recent messages
        recent = history[-limit:]
        context.extend(recent)

        return context

    def maybe_summarize(self, backend=None) -> bool:
        """Auto-summarize old messages if history is too long.

        Call this after each turn. If history exceeds summarize_after,
        the oldest messages get summarized and removed.

        Args:
            backend: HFBackend instance to use for summarization.
                     If None, summarization is skipped.

        Returns:
            True if summarization happened.
        """
        # Count non-system messages
        history = [m for m in self.messages if m["role"] != "system"]
        if len(history) <= self.summarize_after:
            return False

        if backend is None:
            return False

        # Take the oldest half of messages to summarize
        n_to_summarize = len(history) // 2
        old_messages = history[:n_to_summarize]
        keep_messages = history[n_to_summarize:]

        # Format old messages for summarization
        text = "\n".join(
            f"{m['role']}: {m['content'][:300]}" for m in old_messages
        )

        summary = backend.generate(
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this conversation concisely, keeping key facts and decisions:\n\n{text}"
                ),
            }],
            max_tokens=256,
            temperature=0.1,
        )

        self._summaries.append(summary)

        # Save to persistent storage
        turn_start = self._total_turns - len(history)
        turn_end = turn_start + n_to_summarize
        self.save_summary(summary, turn_start, turn_end)

        # Remove old messages, keep system prompt + recent
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]] + keep_messages
        else:
            self.messages = keep_messages

        return True

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
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
        self._total_turns = 0

    def clear_all(self) -> None:
        """Clear everything including persistent storage."""
        self.messages = []
        self._summaries = []
        self._total_turns = 0
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

    @property
    def has_summaries(self) -> bool:
        """Whether there are any stored summaries."""
        return len(self._summaries) > 0

    def __del__(self):
        if self._db:
            self._db.close()
