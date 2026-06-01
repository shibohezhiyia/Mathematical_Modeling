"""
快速 E2E 测试：验证训练过程透明化
"""
import requests
import time
import sys

BASE = "http://localhost:5000"
CSV_PATH = r"I:\exercise_data\Apple_stock.csv"
SESSION = requests.Session()

def log(msg):
    line = f"[E2E-Transparency] {msg}\n"
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
        log(f"ERROR {method} {endpoint}: {e}")
        return None, str(e)

def test_transparency():
    # 1. Upload
    log("=== 1. Upload ===")
    with open(CSV_PATH, 'rb') as f:
        status, data = call("POST", "/api/upload", files={"files": f})
    assert status == 200 and data.get('success'), f"upload failed: {data}"
    log("upload OK")

    # 2. Train
    log("=== 2. Train ===")
    config = {
        "target_col": "Close",
        "task_type": "regression",
        "optimize_hyperparams": False,
        "n_splits": 3,
        "model_keys": ["lr", "ridge"],
        "encoding": "auto",
        "feature_selection": "none",
        "ensemble": "best_single",
    }
    status, data = call("POST", "/api/model/train", json=config)
    assert status == 200 and data.get('success'), f"train failed: {data}"
    log("train started")

    # 3. Poll events and live results
    log("=== 3. Poll Events & Live Results ===")
    since_id = -1
    events_received = []
    live_results_received = []
    for i in range(60):
        time.sleep(2)
        s, d = call("GET", "/api/model/status")
        if s == 200 and d.get('success'):
            status_val = d.get('status')
            # Events
            es, ed = call("GET", f"/api/model/train-events?since_id={since_id}")
            if es == 200 and ed.get('success') and ed.get('events'):
                events = ed['events']
                for ev in events:
                    log(f"  EVENT [{ev['step']}] {ev['message']}")
                events_received.extend(events)
                since_id = ed['latest_id']
            # Live results
            ls, ld = call("GET", "/api/model/live-results")
            if ls == 200 and ld.get('success') and ld.get('results'):
                live_results_received = ld['results']
                log(f"  LIVE RESULTS: {len(live_results_received)} models")
            if status_val in ('done', 'error'):
                break

    # 4. Verify
    log("=== 4. Verify ===")
    assert len(events_received) > 0, "No events received!"
    log(f"Total events: {len(events_received)}")

    steps = set(e['step'] for e in events_received)
    log(f"Steps seen: {steps}")
    assert 'preprocessing' in steps, "Missing preprocessing events"
    assert 'model_done' in steps, "Missing model_done events"

    assert len(live_results_received) > 0, "No live results received!"
    log(f"Live results: {len(live_results_received)} models")

    # 5. Final result
    log("=== 5. Final Result ===")
    s, d = call("GET", "/api/model/result")
    assert s == 200 and d.get('success'), f"result failed: {d}"
    result = d.get('result', {})
    log(f"Result task_type: {result.get('task_type')}")
    log(f"Leaderboard rows: {len(result.get('leaderboard', []))}")
    log(f"Best model: {result.get('best_model_key')}")
    log("ALL PASSED!")

if __name__ == "__main__":
    test_transparency()
