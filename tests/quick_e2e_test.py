"""
快速 E2E 验证脚本 - 测试训练透明化功能
禁用深度学习模型和超参优化，使用少量快速模型
"""
import requests
import time
import sys

BASE = "http://localhost:5000"
CSV_PATH = r"I:\exercise_data\Apple_stock.csv"
ERRORS = []
SESSION = requests.Session()

def log(msg):
    line = f"[QuickE2E] {msg}\n"
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.buffer.write(line.encode('utf-8', errors='replace'))
        sys.stdout.buffer.flush()

def call(method, endpoint, **kwargs):
    url = f"{BASE}{endpoint}"
    try:
        if method == "GET":
            r = SESSION.get(url, timeout=30, **kwargs)
        else:
            r = SESSION.post(url, timeout=300, **kwargs)
        return r.status_code, r.json() if r.headers.get('content-type', '').startswith('application/json') else r.text
    except Exception as e:
        ERRORS.append(f"{method} {endpoint}: {e}")
        return None, str(e)

def main():
    # 1. Upload
    log("=== 1. Upload Data ===")
    with open(CSV_PATH, 'rb') as f:
        status, data = call("POST", "/api/upload", files={"files": f})
    log(f"upload status={status}, success={data.get('success') if isinstance(data, dict) else False}")
    if status != 200:
        log("Upload failed, aborting")
        return 1

    # 2. Train with minimal config (fast models only, no hyperopt, no DL)
    config = {
        'target_col': 'Close',
        'task_type': 'regression',
        'optimize_hyperparams': False,
        'model_keys': ['lr', 'ridge', 'dt'],  # 只训练3个快速模型
        'n_splits': 3,
        'ensemble': 'weighted',
    }
    log(f"=== 2. Train (fast config) ===")
    status, data = call("POST", "/api/model/train", json=config)
    log(f"train status={status}, success={data.get('success') if isinstance(data, dict) else False}")
    if status != 200:
        log("Train failed, aborting")
        return 1

    # 3. Poll status + events + live-results
    log("=== 3. Polling with events ===")
    since_id = -1
    for i in range(60):  # max 120s
        time.sleep(2)
        
        # Status
        s, d = call("GET", "/api/model/status")
        status_str = d.get('status') if isinstance(d, dict) else 'unknown'
        
        # Events
        es, ed = call("GET", f"/api/model/train-events?since_id={since_id}")
        if es == 200 and isinstance(ed, dict) and ed.get('events'):
            for ev in ed['events']:
                log(f"  EVENT [{ev['step']}] {ev['message']}")
            since_id = ed.get('latest_id', since_id)
        
        # Live results
        ls, ld = call("GET", "/api/model/live-results")
        if ls == 200 and isinstance(ld, dict) and ld.get('results'):
            log(f"  LIVE RESULTS: {len(ld['results'])} models done")
        
        if status_str == 'done':
            log(f"Train DONE, polled {i+1} times")
            break
        if status_str == 'error':
            log(f"Train ERROR: {d}")
            ERRORS.append(f"train error: {d}")
            break
        if i % 5 == 0:
            progress = d.get('progress') if isinstance(d, dict) else None
            log(f"  polling... status={status_str}, progress={progress}")
    else:
        log("Train timeout after 120s")
        ERRORS.append("train timeout")

    # 4. Check result
    log("=== 4. Get Result ===")
    status, data = call("GET", "/api/model/result")
    log(f"result status={status}")
    if status == 200 and isinstance(data, dict) and data.get('success'):
        leaderboard = data.get('leaderboard', [])
        log(f"Leaderboard: {len(leaderboard)} models")
        for row in leaderboard[:3]:
            log(f"  {row.get('rank')}. {row.get('model_name')}: {row.get('r2_mean', 'N/A')}")
    else:
        ERRORS.append(f"result failed: {data}")

    # Summary
    log("=== SUMMARY ===")
    if ERRORS:
        log(f"ERRORS ({len(ERRORS)}):")
        for e in ERRORS:
            log(f"  - {e}")
        return 1
    else:
        log("All checks PASSED")
        return 0

if __name__ == '__main__':
    sys.exit(main())
