"""
端到端全功能集成测试
使用 Apple_stock.csv 跑通上传→训练→解释→公平性→结果 全流程
"""
import requests
import json
import time
import pandas as pd
import traceback

BASE = "http://localhost:5000"
CSV_PATH = r"I:\exercise_data\Apple_stock.csv"
ERRORS = []
SESSION = requests.Session()

import sys

def log(msg):
    line = f"[E2E] {msg}\n"
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

def test_upload():
    log("=== 1. Upload Data ===")
    with open(CSV_PATH, 'rb') as f:
        status, data = call("POST", "/api/upload", files={"files": f})
    log(f"upload status={status}")
    if status != 200 or not data.get('success'):
        ERRORS.append(f"upload failed: {data}")
        return False
    return True

def do_train(train_config):
    log(f"=== 2. Train (config={train_config}) ===")
    status, data = call("POST", "/api/model/train", json=train_config)
    log(f"train status={status}")
    if status != 200 or not data.get('success'):
        ERRORS.append(f"train failed: {data}")
        return False
    
    # 轮询状态
    for i in range(300):
        time.sleep(2)
        s, d = call("GET", "/api/model/status")
        if s == 200 and d.get('status') == 'done':
            log(f"Train done, polled {i+1} times")
            return True
        if s == 200 and d.get('status') == 'error':
            ERRORS.append(f"train error: {d}")
            return False
        if i % 10 == 0:
            log(f"  polling... status={d.get('status')}, progress={d.get('progress')}")
    ERRORS.append("train timeout after 600s")
    return False

def test_result():
    log("=== 3. Get Result ===")
    status, data = call("GET", "/api/model/result")
    log(f"result status={status}")
    if status != 200:
        ERRORS.append(f"result failed: {data}")
        return None
    return data

def test_hyperopt_history():
    log("=== 4. Hyperopt History ===")
    status, data = call("GET", "/api/model/hyperopt-history")
    log(f"hyperopt-history status={status}")
    if status != 200:
        ERRORS.append(f"hyperopt-history failed: {data}")

def do_explain(model_key):
    log(f"=== 5. Explain: {model_key} ===")
    status, data = call("POST", "/api/model/explain", json={"model_key": model_key, "instance_index": 0})
    log(f"explain status={status}, success={data.get('success') if isinstance(data, dict) else False}")
    if status != 200 or (isinstance(data, dict) and not data.get('success')):
        ERRORS.append(f"explain {model_key} failed: {data}")

def do_fairness(model_key):
    log(f"=== 6. Fairness: {model_key} ===")
    status, data = call("POST", "/api/model/fairness", json={"model_key": model_key})
    log(f"fairness status={status}, success={data.get('success') if isinstance(data, dict) else False}")
    if status != 200 or (isinstance(data, dict) and not data.get('success')):
        ERRORS.append(f"fairness {model_key} failed: {data}")

def run_all():
    # 配置 1: 回归 + 超参优化 + PCA + 自编码器
    cfg = {
        "target_col": "Close",
        "task_type": "regression",
        "optimize_hyperparams": True,
        "hyperparam_trials": 5,
        "hyperparam_sampler": "tpe",
        "optimizer": "bayesian",
        "feature_selection": "pca",
        "dim_reduction": "autoencoder",
        "ensemble": "weighted",
        "n_splits": 3,
        "explainability": True,
        "deep_learning": {"enabled": True, "models": ["torch_mlp"]}
    }
    
    if not test_upload():
        return
    
    if not do_train(cfg):
        return
    
    result = test_result()
    test_hyperopt_history()
    
    if result and result.get('success'):
        cv_results = result.get('result', {}).get('cv_results', [])
        model_keys = [r['model_key'] for r in cv_results[:3]] if cv_results else ['knn', 'rf']
        for mk in model_keys:
            do_explain(mk)
            do_fairness(mk)
    
    log("\n=== Error Summary ===")
    if ERRORS:
        for e in ERRORS:
            log(f"  [ERROR] {e}")
    else:
        log("  No errors!")

if __name__ == "__main__":
    run_all()
