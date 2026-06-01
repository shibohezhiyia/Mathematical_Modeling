"""
Model Versioning (lightweight)

Save/restore model snapshots via SQLite BLOB.
"""
import io
import os
import pickle
import sqlite3
import time
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'experiments.db')


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_table() -> None:
    conn = _get_conn()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS model_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            model_key TEXT,
            timestamp TEXT,
            snapshot BLOB,
            metadata TEXT
        )
    ''')
    conn.commit()
    conn.close()


_init_table()


def save_snapshot(experiment_id: int, model_key: str, model: Any, metadata: Optional[Dict] = None) -> int:
    buf = io.BytesIO()
    pickle.dump(model, buf)
    conn = _get_conn()
    cursor = conn.execute(
        '''INSERT INTO model_snapshots (experiment_id, model_key, timestamp, snapshot, metadata)
           VALUES (?, ?, ?, ?, ?)''',
        (experiment_id, model_key, time.strftime('%Y-%m-%d %H:%M:%S'), buf.getvalue(), str(metadata or {}))
    )
    conn.commit()
    sid = cursor.lastrowid
    conn.close()
    return sid


def list_snapshots(experiment_id: Optional[int] = None, limit: int = 20) -> List[Dict]:
    conn = _get_conn()
    if experiment_id:
        rows = conn.execute(
            'SELECT id, experiment_id, model_key, timestamp, metadata FROM model_snapshots WHERE experiment_id = ? ORDER BY id DESC LIMIT ?',
            (experiment_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT id, experiment_id, model_key, timestamp, metadata FROM model_snapshots ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_snapshot(snapshot_id: int) -> Optional[Any]:
    conn = _get_conn()
    row = conn.execute('SELECT snapshot FROM model_snapshots WHERE id = ?', (snapshot_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return pickle.loads(row['snapshot'])


def delete_snapshot(snapshot_id: int) -> bool:
    conn = _get_conn()
    cursor = conn.execute('DELETE FROM model_snapshots WHERE id = ?', (snapshot_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0
