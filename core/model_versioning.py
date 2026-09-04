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
            owner_id TEXT,
            model_key TEXT,
            timestamp TEXT,
            snapshot BLOB,
            metadata TEXT
        )
    ''')
    columns = {row[1] for row in conn.execute('PRAGMA table_info(model_snapshots)').fetchall()}
    if 'owner_id' not in columns:
        conn.execute('ALTER TABLE model_snapshots ADD COLUMN owner_id TEXT')
    conn.commit()
    conn.close()


_init_table()


def save_snapshot(
    experiment_id: int,
    model_key: str,
    model: Any,
    metadata: Optional[Dict] = None,
    owner_id: Optional[str] = None,
) -> int:
    buf = io.BytesIO()
    pickle.dump(model, buf)
    conn = _get_conn()
    cursor = conn.execute(
        '''INSERT INTO model_snapshots (experiment_id, owner_id, model_key, timestamp, snapshot, metadata)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (experiment_id, owner_id, model_key, time.strftime('%Y-%m-%d %H:%M:%S'), buf.getvalue(), str(metadata or {}))
    )
    conn.commit()
    sid = cursor.lastrowid
    conn.close()
    return sid


def list_snapshots(
    experiment_id: Optional[int] = None,
    limit: int = 20,
    owner_id: Optional[str] = None,
) -> List[Dict]:
    conn = _get_conn()
    clauses = []
    params = []
    if experiment_id is not None:
        clauses.append('experiment_id = ?')
        params.append(experiment_id)
    if owner_id is not None:
        clauses.append('owner_id = ?')
        params.append(owner_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ''
    params.append(max(1, min(int(limit), 1000)))
    rows = conn.execute(
        f'SELECT id, experiment_id, owner_id, model_key, timestamp, metadata '
        f'FROM model_snapshots{where} ORDER BY id DESC LIMIT ?',
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_snapshot(snapshot_id: int, owner_id: Optional[str] = None) -> Optional[Any]:
    conn = _get_conn()
    if owner_id is None:
        row = conn.execute('SELECT snapshot FROM model_snapshots WHERE id = ?', (snapshot_id,)).fetchone()
    else:
        row = conn.execute(
            'SELECT snapshot FROM model_snapshots WHERE id = ? AND owner_id = ?',
            (snapshot_id, owner_id),
        ).fetchone()
    conn.close()
    if row is None:
        return None
    return pickle.loads(row['snapshot'])


def delete_snapshot(snapshot_id: int, owner_id: Optional[str] = None) -> bool:
    conn = _get_conn()
    if owner_id is None:
        cursor = conn.execute('DELETE FROM model_snapshots WHERE id = ?', (snapshot_id,))
    else:
        cursor = conn.execute(
            'DELETE FROM model_snapshots WHERE id = ? AND owner_id = ?',
            (snapshot_id, owner_id),
        )
    conn.commit()
    conn.close()
    return cursor.rowcount > 0
