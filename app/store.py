import sqlite3
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from .models import TaskStatus

DB_PATH = Path(__file__).parent.parent / "data" / "tasks.db"

_init_lock = threading.Lock()
_initialized = False


def _init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            result TEXT,
            error TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
    conn.commit()


@contextmanager
def _get_conn():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _ensure_init():
    global _initialized
    if _initialized and DB_PATH.exists():
        return
    with _init_lock:
        if _initialized and DB_PATH.exists():
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _get_conn() as conn:
            _init_db(conn)
        _initialized = True


def reset_stale_processing():
    _ensure_init()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET status = ? WHERE status = ?",
            (TaskStatus.PENDING, TaskStatus.PROCESSING),
        )
        conn.commit()


def create_task(task_id: str, file_name: str, file_path: str) -> dict:
    _ensure_init()
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO tasks (task_id, status, file_name, file_path, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, TaskStatus.PENDING, file_name, file_path, now),
        )
        conn.commit()
    return {"task_id": task_id, "status": TaskStatus.PENDING, "created_at": now, "file_name": file_name}


def get_task(task_id: str) -> dict | None:
    _ensure_init()
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if not row:
        return None
    task = dict(row)
    if task["result"]:
        task["result"] = json.loads(task["result"])
    task.pop("file_path", None)
    return task


def claim_next_pending() -> dict | None:
    _ensure_init()
    with _get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM tasks WHERE status = ? ORDER BY created_at LIMIT 1",
            (TaskStatus.PENDING,),
        ).fetchone()
        if not row:
            conn.execute("ROLLBACK")
            return None
        conn.execute(
            "UPDATE tasks SET status = ? WHERE task_id = ?",
            (TaskStatus.PROCESSING, row["task_id"]),
        )
        conn.commit()
    task = dict(row)
    task["status"] = TaskStatus.PROCESSING
    return task


def complete_task(task_id: str, result: dict):
    _ensure_init()
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET status = ?, result = ?, completed_at = ? WHERE task_id = ?",
            (TaskStatus.COMPLETED, json.dumps(result, ensure_ascii=False), now, task_id),
        )
        conn.commit()


def fail_task(task_id: str, error: str):
    _ensure_init()
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE tasks SET status = ?, error = ?, completed_at = ? WHERE task_id = ?",
            (TaskStatus.FAILED, error, now, task_id),
        )
        conn.commit()


def list_tasks(limit: int = 20, status: str | None = None) -> list[dict]:
    _ensure_init()
    with _get_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT task_id, status, file_name, created_at, completed_at, error FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT task_id, status, file_name, created_at, completed_at, error FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]
