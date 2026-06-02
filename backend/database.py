"""
database.py — SQLite persistence layer using Python's built-in sqlite3.
Manages task lifecycle and stores JSON results without any ORM dependency.
"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "tasks.db"


def get_connection() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection with row factory set."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # concurrent read safety
    return conn


def init_db() -> None:
    """Create schema if it does not already exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id     TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                error       TEXT,
                result_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
        """)


# ── CRUD helpers ────────────────────────────────────────────────────────────

def create_task(task_id: str, filename: str) -> dict:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO tasks (task_id, filename, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            (task_id, filename, "pending", now, now),
        )
    return get_task(task_id)


def update_task_status(task_id: str, status: str, error: str | None = None) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE tasks SET status=?, error=?, updated_at=? WHERE task_id=?",
            (status, error, now, task_id),
        )


def save_task_result(task_id: str, result: dict) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE tasks SET status='completed', result_json=?, updated_at=? WHERE task_id=?",
            (json.dumps(result), now, task_id),
        )


def get_task(task_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def list_tasks() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT task_id, filename, status, created_at, updated_at, error FROM tasks ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
