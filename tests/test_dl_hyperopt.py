"""
验证 TorchMLP 超参优化改进
"""
import requests
import time
import sys

BASE = "http://localhost:5000"
CSV_PATH = r"I:\exercise_data\Apple_stock.csv"
SESSION = requests.Session()

def log(msg):
    line = f"[DL-Hyperopt] {msg}\n"
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
            r = SESSION.post(url, timeout=600, **kwargs)
        return r.status_code, r.json() if r.headers.get('content-type', '').startswith('application/json') else r.text
    except Exception as e:
        return None, str(e)

def main():
    # Upload
    log("=== Upload ===")
    with open(CSV_PATH, 'rb') as f:
        status, data = call("POST", "/api/upload", files={"files": f})
    assert status == 200, f"upload failed: {status}"
    log("upload OK")

    # Train with TorchMLP hyperopt
    log("=== Train with TorchMLP hyperopt ===")
    config = {
        "target_col": "Close",
        "task_type": "regression",
        "optimize_hyperparams": True,
        "hyperparam_trials": 5,
        "n_splits": 3,
        "model_keys": ["torch_mlp"],
        "deep_learning": {"enabled": True, "models": ["torch_mlp"]},
    }
    
    start = time.time()
    status, data = call("POST", "/api/model/train", json=config)
    assert status == 200 and data.get('success'), f"train failed: {data}"
    log(f"train started, elapsed={time.time()-start:.1f}s")
    
    # Poll
    for i in range(120):
        time.sleep(3)
        s, d = call("GET", "/api/model/status")
        if s == 200 and d.get('status') == 'done':
            elapsed = time.time() - start
            log(f"DONE in {elapsed:.1f}s!")
            break
        if s == 200 and d.get('status') == 'error':
            log(f"ERROR: {d.get('error')}")
            break
        if i % 5 == 0:
            ev = call("GET", "/api/model/train-events?since_id=-1")
            if ev[0] == 200:
                events = ev[1].get('events', [])
                log(f"  poll {i+1}: status={d.get('status')}, events={len(events)}")
    
    # Result
    s, d = call("GET", "/api/model/result")
    if s == 200 and d.get('success'):
        result = d.get('result', {})
        log(f"Result: task_type={result.get('task_type')}, leaderboard={len(result.get('leaderboard', []))}")
        log(f"Best model: {result.get('best_model_key')}")
    else:
        log(f"Result failed: {s}, {d}")

if __name__ == "__main__":
    main()
