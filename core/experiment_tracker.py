"""
实验追踪引擎（MLflow-lite）

用 SQLite 本地存储每次实验的完整配置与结果，
支持历史查询、对比、复现。
"""

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'experiments.db')


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_conn()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            dataset_name TEXT,
            task_type TEXT,
            config TEXT,
            result TEXT,
            best_score REAL,
            best_model TEXT,
            duration REAL
        )
    ''')
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_exp_time ON experiments(timestamp)
    ''')
    conn.commit()
    conn.close()


_init_db()


def log_experiment(
    config: Dict[str, Any],
    result: Dict[str, Any],
    duration: float,
    dataset_name: str = ''
) -> int:
    """记录一次实验，返回实验 ID"""
    best_score = 0.0
    best_model = ''
    task_type = result.get('task_type', '')
    
    leaderboard = result.get('leaderboard', [])
    if leaderboard and len(leaderboard) > 0:
        first = leaderboard[0]
        # 取主指标分数
        score_keys = [k for k in first.keys() if k.endswith('_mean')]
        if score_keys:
            best_score = float(first.get(score_keys[0], 0))
        best_model = first.get('model_name', first.get('model_key', ''))
    
    conn = _get_conn()
    cursor = conn.execute(
        '''INSERT INTO experiments (timestamp, dataset_name, task_type, config, result, best_score, best_model, duration)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            time.strftime('%Y-%m-%d %H:%M:%S'),
            dataset_name,
            task_type,
            json.dumps(config, ensure_ascii=False, default=str),
            json.dumps(result, ensure_ascii=False, default=str),
            best_score,
            best_model,
            duration
        )
    )
    conn.commit()
    exp_id = cursor.lastrowid
    conn.close()
    return exp_id


def list_experiments(limit: int = 50, task_type: Optional[str] = None) -> List[Dict]:
    """列出历史实验"""
    conn = _get_conn()
    if task_type:
        rows = conn.execute(
            'SELECT id, timestamp, dataset_name, task_type, best_score, best_model, duration FROM experiments WHERE task_type = ? ORDER BY id DESC LIMIT ?',
            (task_type, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT id, timestamp, dataset_name, task_type, best_score, best_model, duration FROM experiments ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_experiment(exp_id: int) -> Optional[Dict]:
    """获取单个实验详情"""
    conn = _get_conn()
    row = conn.execute('SELECT * FROM experiments WHERE id = ?', (exp_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    result = dict(row)
    result['config'] = json.loads(result['config']) if result['config'] else {}
    result['result'] = json.loads(result['result']) if result['result'] else {}
    return result


def compare_experiments(exp_ids: List[int]) -> List[Dict]:
    """对比多个实验"""
    results = []
    for eid in exp_ids:
        exp = get_experiment(eid)
        if exp:
            results.append({
                'id': exp['id'],
                'timestamp': exp['timestamp'],
                'task_type': exp['task_type'],
                'best_score': exp['best_score'],
                'best_model': exp['best_model'],
                'duration': exp['duration'],
                'config_summary': _summarize_config(exp['config']),
            })
    return results


def delete_experiment(exp_id: int) -> bool:
    """删除实验"""
    conn = _get_conn()
    cursor = conn.execute('DELETE FROM experiments WHERE id = ?', (exp_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def _summarize_config(config: Dict) -> str:
    """生成配置摘要"""
    parts = []
    if config.get('model_keys'):
        parts.append(f"models={len(config['model_keys'])}")
    if config.get('ensemble'):
        parts.append(f"ens={config['ensemble']}")
    if config.get('optimize_hyperparams'):
        parts.append(f"hpo={config['hyperparam_trials']}")
    if config.get('feature_engineering'):
        parts.append('fe=on')
    if config.get('pseudo_labeling'):
        parts.append('pl=on')
    return ', '.join(parts) if parts else 'default'
