"""Web research endpoint regression tests."""

from unittest.mock import patch
import json
import threading
import time
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from web.app import (app, user_sessions, _prepare_research_frame, _read_large_csv_representation,
                     _read_large_parquet_representation, _read_large_xlsx_representation,
                     _read_file_with_sheets, df_to_dict)
from core.artifact_manager import RunArtifactManager


def test_method_plan_endpoint_routes_external_methods_without_execution():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/method-plan', json={
            'profile': {
                'numeric_data': True,
                'time_series': True,
                'known_dynamics': True,
                'noise_expected': True,
            },
            'enabled_methods': ['sindy', 'weak_sindy', 'ude', 'llm_sr'],
        })
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    methods = {row['method'] for row in payload['result']['candidates']}
    assert {'sindy', 'weak_sindy', 'ude', 'llm_sr'} <= methods
    assert all(row['dependency_status'] == 'not_assessed' for row in payload['result']['candidates'])


def test_dynamic_compile_async_endpoint_exposes_owner_scoped_task():
    app.config.update(TESTING=True)
    graph = {
        'nodes': [
            {'id': 'x', 'op': 'variable', 'inputs': [], 'kind': 'quantity', 'dimensions': {}, 'attributes': {'name': 'x'}},
            {'id': 'two', 'op': 'constant', 'inputs': [], 'kind': 'quantity', 'dimensions': {}, 'attributes': {'value': 2}},
            {'id': 'y', 'op': 'multiply', 'inputs': ['x', 'two'], 'kind': 'quantity', 'dimensions': {}, 'attributes': {}},
        ], 'bindings': {'x': 3}, 'output_ids': ['y'],
    }
    with app.test_client() as client:
        response = client.post('/api/research/dynamic-compile', json={'kind': 'primitive_graph', 'payload': graph, 'async': True})
        assert response.status_code == 202
        task = response.get_json()['task']
        task_id = task['task_id']
        for _ in range(100):
            status = client.get(f'/api/research/dynamic-compile/status/{task_id}').get_json()['task']
            if status['status'] in {'completed', 'failed', 'cancelled'}:
                break
            time.sleep(0.005)
        assert status['status'] == 'completed'
        assert status['result']['result']['outputs']['y'] == 6
        assert client.post(f'/api/research/dynamic-compile/cancel/{task_id}').get_json()['task']['status'] == 'completed'


def test_evaluation_sources_endpoint_exposes_audit_without_source_content():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get('/api/research/evaluation-sources')
    assert response.status_code == 200
    result = response.get_json()['result']
    assert result['eligible_source_count'] == 0
    assert all('question' not in row and 'answer' not in row for row in result['sources'])


def test_holdout_intake_endpoint_returns_not_ready_for_incomplete_metadata():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/holdout-intake', json={
            'schema_version': 'mathmodel.holdout-intake/v1',
            'protocol_id': 'p1',
            'development_freeze_digest': '0' * 64,
            'rubric_digest': '1' * 64,
            'independent_author_attested': True,
            'independent_evaluator_attested': False,
            'cases': [],
        })
    assert response.status_code == 200
    result = response.get_json()['result']
    assert result['status'] == 'not_ready'
    assert 'minimum_case_count' in result['missing']


def test_holdout_bind_endpoint_rejects_missing_payload():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/holdout-bind', json={})
    assert response.status_code == 400
    assert response.get_json()['success'] is False


def test_method_execute_endpoint_keeps_llm_code_in_proposal_state():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/method-execute', json={
            'method': 'llm_sr',
            'payload': {'source': "__import__('os').system('x')"},
        })
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['result']['status'] == 'proposal_only'


def test_method_cegis_endpoint_runs_typed_candidate_loop():
    app.config.update(TESTING=True)
    x = [index / 31 for index in range(32)]
    y = [2.0 * value + 1.0 for value in x]
    candidate = {
        'id': 'linear', 'method': 'llm_sr', 'payload': {
            'target': y, 'features': [[value] for value in x],
            'expression': {'op': 'add', 'left': {'op': 'multiply', 'left': {'op': 'param', 'name': 'a'}, 'right': {'op': 'var', 'name': 'x'}}, 'right': {'op': 'param', 'name': 'b'}},
            'parameter_bounds': {'a': [-5, 5], 'b': [-5, 5]}, 'max_nfev': 80,
        },
    }
    with app.test_client() as client:
        response = client.post('/api/research/method-cegis', json={
            'method': 'llm_sr', 'candidates': [candidate],
            'cases': [{'id': 'holdout', 'max_metric': 0.1}],
        })
    assert response.status_code == 200
    result = response.get_json()['result']
    assert result['adapter_family'] == 'llm_sr'
    assert result['records']


def test_method_compare_endpoint_runs_paired_grid_and_keeps_proposal_arm():
    app.config.update(TESTING=True)
    from core.external_method_comparison import build_external_method_manifest
    manifest = build_external_method_manifest(
        task_ids=['case-a'], fixed_budget={'seconds': 30, 'seed': 1},
        final_test_fingerprint='locked-final-v1',
    )
    times = [index / 8 for index in range(64)]
    states = [[1.0 + 0.01 * index, 2.0 + 0.02 * index] for index in range(64)]
    with app.test_client() as client:
        response = client.post('/api/research/method-compare', json={
            'manifest': manifest,
            'tasks': [{
                'id': 'case-a',
                'payloads': {
                    'sindy': {'times': times, 'states': states, 'state_names': ['x', 'y'],
                              'polynomial_degree': 1, 'residual_tolerance': 10.0},
                    'weak_sindy': {'times': times, 'states': states, 'state_names': ['x', 'y'],
                                   'polynomial_degree': 1, 'residual_tolerance': 10.0,
                                   'window_count': 8},
                    'ude': {'target_rhs': list(range(16)),
                            'known_rhs': list(range(16)),
                            'correction_features': [[1.0, index / 16] for index in range(16)]},
                },
            }],
        })
    assert response.status_code == 200
    result = response.get_json()['result']
    assert result['expected_rows'] == result['observed_rows'] == 5
    assert result['status_counts']['completed'] == 3
    assert result['status_counts']['rejected'] == 2


def test_blind_accuracy_endpoint_rejects_missing_sealed_inputs():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/blind-accuracy', json={
            'baseline_system': 'baseline-v1', 'treatment_system': 'treatment-v1',
        })
    assert response.status_code == 400
    assert response.get_json()['success'] is False


def test_uncertainty_audit_endpoint_keeps_layer_spread_separate_from_calibration():
    evaluations = [
        {
            'semantic_id': 's1', 'structure_id': 'm1', 'parameter_id': 'p1',
            'numerical_id': f'n{i}', 'value': float(i), 'unit_signature': 'm'
        }
        for i in range(10)
    ]
    with app.test_client() as client:
        response = client.post('/api/research/uncertainty-audit', json={
            'evaluations': evaluations,
            'interval': {
                'actual': list(range(10)),
                'lower': [i - 0.5 for i in range(10)],
                'upper': [i + 0.5 for i in range(10)],
            },
        })
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['result']['propagation']['status'] == 'ok'
    assert payload['result']['calibration']['sample_count'] == 10


class _ImmediateThread:
    def __init__(self, target, daemon=True):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


def test_large_time_fact_is_exactly_preaggregated_instead_of_head_sampled():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    frame = pd.DataFrame([
        {
            "观测日期": date,
            "站点编号": station,
            "需求量": float(transaction + 1),
            "服务单价": float(10 + transaction),
        }
        for date in dates
        for station in range(4)
        for transaction in range(3)
    ])
    prepared = _prepare_research_frame(frame, max_rows=100)

    assert prepared.attrs["aggregation_complete"] is True
    assert prepared.attrs["research_representation"] == "exact_daily_dimension_aggregation"
    assert prepared.attrs["source_rows"] == len(frame)
    assert len(prepared) == len(dates) * 4
    assert prepared["观测日期"].min() == dates.min()
    assert prepared["观测日期"].max() == dates.max()
    assert prepared["需求量"].sum() == frame["需求量"].sum()
    assert prepared["服务单价"].iloc[0] == pytest.approx(68.0 / 6.0)


def test_research_sensitivity_endpoint_aggregates_only_trusted_evaluations():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/sensitivity', json={
            'base_evaluation': {'decision': 'A', 'objective': 1.0},
            'variants': [
                {'id': 'low', 'overrides': {'alpha': 0.1},
                 'evaluation': {'decision': 'B', 'objective': 0.8}},
            ],
        })
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['result']['decision_change_count'] == 1
    assert payload['policy'] == 'aggregation_only_trusted_evaluator_required'


def test_hypothesis_plan_endpoint_returns_downstream_recompute_plan():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/hypothesis-plan', json={
            'graph': {'nodes': [
                {'id': 'a', 'inputs': []}, {'id': 'b', 'inputs': ['a']},
                {'id': 'c', 'inputs': ['b']},
            ]},
            'controls': {
                'node_ids': ['a', 'b', 'c'],
                'controls': [{'id': 'alpha', 'min': 0, 'max': 1, 'default': .5,
                              'step': .1, 'affected_nodes': ['a']}],
            },
        })
    assert response.status_code == 200
    body = response.get_json()
    assert body['success'] is True
    assert body['plans'][0]['recompute_nodes'] == ['a', 'b', 'c']
    assert body['policy'].startswith('plan_only')


def test_hypothesis_preview_recomputes_typed_graph_with_slider_value():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/hypothesis-preview', json={
            'controls': [{'id': 'gain', 'parameter_id': 'k', 'version': 2,
                          'min': 0, 'max': 2, 'default': 1, 'step': .1,
                          'affected_nodes': ['k']}],
            'values': {'gain': 1.5}, 'bindings': {'k': 1.0, 'x': 2.0},
            'graph': {'output_ids': ['out'], 'nodes': [
                {'id': 'k', 'op': 'variable', 'inputs': [], 'kind': 'quantity', 'dimensions': {'Q': 1}, 'attributes': {}},
                {'id': 'x', 'op': 'variable', 'inputs': [], 'kind': 'quantity', 'dimensions': {'Q': 1}, 'attributes': {}},
                {'id': 'out', 'op': 'add', 'inputs': ['k', 'x'], 'kind': 'quantity', 'dimensions': {'Q': 1}, 'attributes': {}},
            ]},
        })
    assert response.status_code == 200
    body = response.get_json()
    assert body['success'] is True
    assert body['preview']['bindings']['k'] == 1.5
    assert body['execution']['outputs']['out'] == 3.5


def test_research_cancel_endpoint_signals_running_worker_without_force_kill():
    sid = "research-cancel-contract-test"
    cancel_event = threading.Event()
    user_sessions[sid] = {
        "research_status": "running",
        "research_cancel_event": cancel_event,
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            response = client.post("/api/research/cancel")
            payload = response.get_json()
            assert response.status_code == 200
            assert payload["status"] == "cancelling"
            assert payload["cancellation_requested"] is True
            assert cancel_event.is_set()
            status = client.get("/api/research/status").get_json()
            assert status["status"] == "cancelling"
            assert status["cancellation_requested"] is True
    finally:
        user_sessions.pop(sid, None)


def test_large_csv_representation_is_bounded_and_records_non_complete_source(tmp_path):
    path = tmp_path / "large.csv"
    pd.DataFrame({"id": range(100), "value": [x * 2 for x in range(100)]}).to_csv(path, index=False)
    frame = _read_large_csv_representation(path, ".csv", max_rows=11)
    assert len(frame) == 11
    assert frame.attrs["source_rows"] == 100
    assert frame.attrs["research_representation"] == "deterministic_coverage_sample"
    assert frame.attrs["aggregation_complete"] is False
    info = df_to_dict(frame)
    assert info["source_shape"] == [100, 2]
    assert info["display_rows"] == 11
    assert info["bounded_representation"] is True


def test_file_reader_can_switch_to_bounded_csv_view(tmp_path, monkeypatch):
    path = tmp_path / "large.csv"
    pd.DataFrame({"x": range(30)}).to_csv(path, index=False)
    monkeypatch.setattr("web.app.MAX_SESSION_FILE_BYTES", 1)
    frame, sheets = _read_file_with_sheets(path, ".csv")
    assert sheets is None
    assert len(frame) == 30 or frame.attrs["source_rows"] == 30
    assert frame.attrs["source_rows"] == 30


def test_large_xlsx_representation_uses_read_only_rows(tmp_path):
    path = tmp_path / "large.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"id": range(30), "value": [x * 3 for x in range(30)]}).to_excel(
            writer, sheet_name="data", index=False
        )
    frame = _read_large_xlsx_representation(path, "data", max_rows=7)
    assert len(frame) == 7
    assert frame.attrs["source_rows"] == 30
    assert frame.attrs["aggregation_complete"] is False
    assert frame.iloc[0]["id"] == 0
    assert frame.iloc[-1]["id"] == 29


def test_large_parquet_representation_reads_batches(tmp_path):
    path = tmp_path / "large.parquet"
    frame = pd.DataFrame({"id": range(30), "value": [x * 4 for x in range(30)]})
    frame.to_parquet(path, index=False, row_group_size=7)
    sampled = _read_large_parquet_representation(path, max_rows=8)
    assert len(sampled) == 8
    assert sampled.attrs["source_rows"] == 30
    assert sampled.attrs["research_representation"] == "deterministic_coverage_sample"
    assert sampled.iloc[0]["id"] == 0
    assert sampled.iloc[-1]["id"] == 29


def test_quality_endpoint_uses_source_for_bounded_csv_and_autofix_refuses_preview(tmp_path):
    path = tmp_path / "source.csv"
    source = pd.DataFrame({"value": range(30), "group": ["a", "b"] * 15})
    source.to_csv(path, index=False)
    sampled = _read_large_csv_representation(path, ".csv", max_rows=7)
    sid = "stream-quality-contract-test"
    user_sessions[sid] = {
        "df": sampled,
        "active_file_index": 0,
        "uploaded_files": [{"path": str(path), "ext": ".csv", "filename": "source.csv"}],
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            response = client.post("/api/data/quality", json={"target_col": "value"})
            payload = response.get_json()
            assert response.status_code == 200
            assert payload["report"]["n_rows"] == 30
            assert payload["report"]["streaming"] is True
            autofix = client.post("/api/data/autofix", json={})
            assert autofix.status_code == 400
            assert "受限预览" in autofix.get_json()["error"]
    finally:
        user_sessions.pop(sid, None)


def test_merge_endpoint_uses_explicit_sheet_selection_and_rejects_stale_sources(tmp_path):
    sid = "sheet-merge-contract-test"
    workbook = tmp_path / "multi_sheet.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_excel(
            writer, sheet_name="north", index=False
        )
        pd.DataFrame({"id": [3], "value": [30]}).to_excel(
            writer, sheet_name="south", index=False
        )
    user_sessions[sid] = {
        "uploaded_files": [{
            "filename": "multi_sheet.xlsx",
            "path": str(workbook),
            "ext": ".xlsx",
            "shape": [2, 2],
            "sheets": ["north", "south"],
            "active_sheet": "north",
            "columns": ["id", "value"],
        }],
        "train_events": [],
        "train_live_results": [],
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            response = client.post("/api/upload/merge", json={
                "sources": [
                    {"file_index": 0, "sheet_name": "north"},
                    {"file_index": 0, "sheet_name": "south"},
                ],
                "axis": 0,
            })
            payload = response.get_json()
            assert response.status_code == 200
            assert payload["shape"] == [3, 2]
            assert payload["merge_diagnostics"]["source_count"] == 2
            assert payload["merge_diagnostics"]["schemas_identical"] is True

            duplicate = client.post("/api/upload/merge", json={
                "sources": [
                    {"file_index": 0, "sheet_name": "north"},
                    {"file_index": 0, "sheet_name": "north"},
                ],
                "axis": 0,
            })
            assert duplicate.status_code == 400
            assert "重复" in duplicate.get_json()["error"]

            stale = client.post("/api/upload/merge", json={
                "sources": [
                    {"file_index": 0, "sheet_name": "north"},
                    {"file_index": 0, "sheet_name": "missing"},
                ],
                "axis": 0,
            })
            assert stale.status_code == 400
            assert "Sheet不存在" in stale.get_json()["error"]
    finally:
        user_sessions.pop(sid, None)


def test_merge_endpoint_supports_cross_file_single_table_sources(tmp_path):
    sid = "cross-file-merge-contract-test"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({"id": [1, 2], "value": [10, 20]}).to_csv(first, index=False)
    pd.DataFrame({"id": [3], "score": [0.8]}).to_csv(second, index=False)
    user_sessions[sid] = {
        "uploaded_files": [
            {
                "filename": first.name,
                "path": str(first),
                "ext": ".csv",
                "shape": [2, 2],
                "sheets": None,
                "columns": ["id", "value"],
            },
            {
                "filename": second.name,
                "path": str(second),
                "ext": ".csv",
                "shape": [1, 2],
                "sheets": None,
                "columns": ["id", "score"],
            },
        ],
        "train_events": [],
        "train_live_results": [],
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            response = client.post("/api/upload/merge", json={
                "sources": [
                    {"file_index": 0, "sheet_name": None},
                    {"file_index": 1, "sheet_name": None},
                ],
                "axis": 0,
            })
            payload = response.get_json()
            assert response.status_code == 200
            assert payload["shape"] == [3, 3]
            assert payload["merge_diagnostics"]["schemas_identical"] is False
            assert payload["merge_diagnostics"]["common_columns"] == ["id"]
            assert payload["merge_diagnostics"]["union_columns"] == ["id", "score", "value"]

            invalid_axis = client.post("/api/upload/merge", json={
                "sources": [
                    {"file_index": 0, "sheet_name": None},
                    {"file_index": 1, "sheet_name": None},
                ],
                "axis": "0",
            })
            assert invalid_axis.status_code == 400
            assert "合并方向" in invalid_axis.get_json()["error"]
    finally:
        user_sessions.pop(sid, None)


def test_join_supports_different_multi_keys_and_blocks_cartesian_expansion(tmp_path):
    sid = "multi-key-join-contract-test"
    left_path = tmp_path / "demand.csv"
    right_path = tmp_path / "cost.csv"
    pd.DataFrame({
        "地区": ["甲", "甲", "乙"],
        "产品": ["A", "B", "A"],
        "需求": [10, 20, 30],
    }).to_csv(left_path, index=False)
    pd.DataFrame({
        "区域": ["甲", "甲", "乙"],
        "货号": ["A", "B", "A"],
        "成本": [3.0, 4.0, 5.0],
    }).to_csv(right_path, index=False)
    user_sessions[sid] = {
        "uploaded_files": [
            {"filename": left_path.name, "path": str(left_path), "ext": ".csv", "shape": [3, 3], "sheets": None, "columns": ["地区", "产品", "需求"]},
            {"filename": right_path.name, "path": str(right_path), "ext": ".csv", "shape": [3, 3], "sheets": None, "columns": ["区域", "货号", "成本"]},
        ],
        "train_events": [],
        "train_live_results": [],
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            response = client.post("/api/upload/join", json={
                "left": {"file_index": 0, "sheet_name": None},
                "right": {"file_index": 1, "sheet_name": None},
                "left_on": ["地区", "产品"],
                "right_on": ["区域", "货号"],
                "how": "left",
                "validate": "one_to_one",
            })
            payload = response.get_json()
            assert response.status_code == 200
            assert payload["shape"] == [3, 6]
            assert payload["join_diagnostics"]["inferred_relation"] == "one_to_one"
            assert payload["join_diagnostics"]["estimated_rows"] == 3
            assert payload["join_diagnostics"]["actual_rows"] == 3

            repeated_left = tmp_path / "repeated_left.csv"
            repeated_right = tmp_path / "repeated_right.csv"
            pd.DataFrame({"key": ["same"] * 60, "x": range(60)}).to_csv(repeated_left, index=False)
            pd.DataFrame({"key": ["same"] * 60, "y": range(60)}).to_csv(repeated_right, index=False)
            user_sessions[sid]["uploaded_files"] = [
                {"filename": repeated_left.name, "path": str(repeated_left), "ext": ".csv", "shape": [60, 2], "sheets": None, "columns": ["key", "x"]},
                {"filename": repeated_right.name, "path": str(repeated_right), "ext": ".csv", "shape": [60, 2], "sheets": None, "columns": ["key", "y"]},
            ]
            rejected = client.post("/api/upload/join", json={
                "left": {"file_index": 0, "sheet_name": None},
                "right": {"file_index": 1, "sheet_name": None},
                "left_on": ["key"],
                "right_on": ["key"],
                "how": "inner",
            })
            assert rejected.status_code == 400
            assert "笛卡尔" in rejected.get_json()["error"]
    finally:
        user_sessions.pop(sid, None)


def test_merge_ui_renders_visible_checkboxes_for_single_and_multi_sheet_sources():
    template = Path("web/templates/index.html").read_text(encoding="utf-8")
    script = Path("web/static/js/app.js").read_text(encoding="utf-8")
    styles = Path("web/static/css/style.css").read_text(encoding="utf-8")

    assert 'id="execute-merge-btn"' in template
    assert "merge-sheet-checkbox" in script
    assert "file.sheets && file.sheets.length ? file.sheets : [null]" in script
    assert "sheetSelectionKey" in script
    assert '.merge-sheet-table input[type="checkbox"]' in styles
    assert "display: inline-block" in styles


def test_async_research_returns_status_and_completed_result():
    sid = "research-async-test"
    app.config.update(TESTING=True)
    user_sessions[sid] = {
        "df": pd.DataFrame({"x": range(40), "target": range(40)}),
        "train_events": [],
        "train_live_results": [],
    }
    fake_result = {
        "problem": "预测 target",
        "charts": [],
        "model_results": [],
        "report_path": "ignored.md",
        "output_dir": "ignored",
    }

    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            with (
                patch("web.app.threading.Thread", _ImmediateThread),
                patch("core.modeling_assistant.MathModelingAssistant") as assistant_class,
            ):
                assistant_class.return_value.run.return_value.to_dict.return_value = fake_result
                response = client.post("/api/research/run", json={
                    "description": "预测 target",
                    "async": True,
                    "run_modeling": False,
                    "generate_plots": False,
                })
                assert response.status_code == 200
                assert response.get_json()["status"] == "running"

                status = client.get("/api/research/status")
                payload = status.get_json()
                assert status.status_code == 200
                assert payload["status"] == "done"
                assert payload["result"]["problem"] == "预测 target"
                assert payload["result"]["report_url"] == "/api/research/report"
                assert assistant_class.call_args.kwargs["credibility_audit"] is True
    finally:
        user_sessions.pop(sid, None)


def test_async_research_keeps_two_sessions_responsive_under_concurrent_workers():
    """A bounded service-level pressure slice: sessions may run concurrently,
    status/cancel state stays isolated, and both workers can finish."""
    started = threading.Event()
    release = threading.Event()
    active_lock = threading.Lock()
    active = {"count": 0, "peak": 0}

    class Result:
        def to_dict(self):
            return {"problem": "并发测试", "charts": [], "model_results": [], "specialized_results": {}}

    class SlowAssistant:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            with active_lock:
                active["count"] += 1
                active["peak"] = max(active["peak"], active["count"])
                if active["count"] >= 2:
                    started.set()
            assert release.wait(timeout=5)
            with active_lock:
                active["count"] -= 1
            return Result()

    sessions = ["research-pressure-a", "research-pressure-b"]
    for sid in sessions:
        user_sessions[sid] = {"train_events": [], "train_live_results": []}
    try:
        with patch("core.modeling_assistant.MathModelingAssistant", side_effect=lambda **kwargs: SlowAssistant()):
            clients = []
            for sid in sessions:
                client = app.test_client()
                with client.session_transaction() as flask_session:
                    flask_session["sid"] = sid
                response = client.post("/api/research/run", json={
                    "description": "并发压力测试", "async": True,
                    "run_modeling": False, "generate_plots": False,
                })
                assert response.status_code == 200
                clients.append(client)
            assert started.wait(timeout=5)
            for client in clients:
                status = client.get("/api/research/status").get_json()
                assert status["status"] in {"running", "done"}
            release.set()
            for client in clients:
                for _ in range(100):
                    status = client.get("/api/research/status").get_json()
                    if status["status"] == "done":
                        break
                    import time
                    time.sleep(0.01)
                assert status["status"] == "done"
        assert active["peak"] == 2
    finally:
        for sid in sessions:
            user_sessions.pop(sid, None)


def test_research_accepts_problem_statement_without_uploaded_dataset():
    sid = "research-no-dataset-test"
    app.config.update(TESTING=True)
    user_sessions[sid] = {"train_events": [], "train_live_results": []}
    fake_result = {
        "problem": "建立动力学模型",
        "input_mode": "mechanistic_no_dataset",
        "charts": [],
        "model_results": [],
        "report_path": "ignored.md",
        "output_dir": "ignored",
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            with patch("core.modeling_assistant.MathModelingAssistant") as assistant_class:
                assistant_class.return_value.run.return_value.to_dict.return_value = fake_result
                response = client.post("/api/research/run", json={
                    "description": "建立动力学模型",
                    "run_modeling": False,
                    "generate_plots": False,
                })
                assert response.status_code == 200
                payload = response.get_json()
                assert payload["success"] is True
                assert payload["result"]["input_mode"] == "mechanistic_no_dataset"
                assert assistant_class.return_value.run.call_args.kwargs["datasets"] == {}
    finally:
        user_sessions.pop(sid, None)


def test_external_data_benchmark_endpoint_uses_pinned_project_catalog(tmp_path):
    app.config.update(TESTING=True)
    with patch('core.external_data_benchmark.run_external_data_benchmark') as runner:
        runner.return_value = {
            'schema_version': 'mathmodel.external-data-benchmark/v1',
            'status': 'completed', 'case_count': 0, 'completed_count': 0,
            'failed_count': 0, 'evaluation_status': 'descriptive_public_labels',
        }
        with app.test_client() as client:
            response = client.post('/api/research/external-data-benchmark', json={'seed': 7})
        assert response.status_code == 200
        assert response.get_json()['result']['evaluation_status'] == 'descriptive_public_labels'
        args = runner.call_args.args
        assert str(args[0]).endswith('examples\\external_dataset_catalog.json') or str(args[0]).endswith('examples/external_dataset_catalog.json')
        assert str(args[1]).endswith('data\\external') or str(args[1]).endswith('data/external')


def test_external_data_benchmark_endpoint_rejects_unbounded_seed():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/external-data-benchmark', json={'seed': -1})
        assert response.status_code == 400


def test_graph_search_endpoint_accepts_only_versioned_bundle(monkeypatch, tmp_path):
    app.config.update(TESTING=True)
    fake_result = {"schema_version": "mathmodel.graph-search-result/v1", "termination": "budget_exhausted", "pareto_candidates": []}
    with patch('core.graph_search_artifacts.run_search_bundle') as runner:
        runner.return_value = (fake_result, tmp_path / "run-1")
        with app.test_client() as client:
            response = client.post('/api/research/graph-search', json={"bundle": {"schema_version": "mathmodel.graph-search-bundle/v1"}})
    assert response.status_code == 200
    assert response.get_json()['result']['termination'] == 'budget_exhausted'
    assert runner.call_args.kwargs['output_root'].name == 'graph_search_runs'


def test_graph_search_endpoint_rejects_non_object_bundle():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/graph-search', json={"bundle": []})
    assert response.status_code == 400


def test_primitive_graph_endpoint_executes_typed_arithmetic():
    app.config.update(TESTING=True)
    nodes = [
        {"id": "out", "op": "add", "inputs": ["x", "one"], "kind": "quantity",
         "dimensions": {"L": 1}, "attributes": {}},
        {"id": "one", "op": "constant", "inputs": [], "kind": "quantity",
         "dimensions": {"L": 1}, "attributes": {"value": 1}},
        {"id": "x", "op": "variable", "inputs": [], "kind": "quantity",
         "dimensions": {"L": 1}, "attributes": {}},
    ]
    with app.test_client() as client:
        response = client.post('/api/research/primitive-graph', json={
            'nodes': nodes, 'bindings': {'x': [1, 2]}, 'output_ids': ['out'],
        })
    assert response.status_code == 200
    assert response.get_json()['result']['outputs']['out'] == [2.0, 3.0]


def test_structure_candidate_endpoint_preserves_execution_boundary():
    app.config.update(TESTING=True)
    candidate = {
        'id': 'candidate-1', 'status': 'proposed_not_executed',
        'primitive_graph': {'output_ids': ['out'], 'nodes': [
            {'id': 'out', 'op': 'add', 'inputs': ['x', 'one'], 'kind': 'quantity', 'dimensions': {'Q': 1}, 'attributes': {}},
            {'id': 'x', 'op': 'variable', 'inputs': [], 'kind': 'quantity', 'dimensions': {'Q': 1}, 'attributes': {}},
            {'id': 'one', 'op': 'constant', 'inputs': [], 'kind': 'quantity', 'dimensions': {'Q': 1}, 'attributes': {'value': 1}},
        ]},
    }
    with app.test_client() as client:
        response = client.post('/api/research/structure-candidate/execute', json={
            'candidate': candidate, 'bindings': {'x': 4},
        })
    assert response.status_code == 200
    body = response.get_json()
    assert body['result']['status'] == 'executed'
    assert body['result']['proposal_status'] == 'proposed_not_executed'


def test_structure_candidate_cegis_endpoint_repairs_constant_with_cases():
    app.config.update(TESTING=True)
    candidate = {
        'id': 'candidate-cegis', 'status': 'proposed_not_executed',
        'primitive_graph': {'output_ids': ['out'], 'nodes': [
            {'id': 'out', 'op': 'add', 'inputs': ['x', 'bias'], 'kind': 'quantity', 'dimensions': {'Q': 1}, 'attributes': {}},
            {'id': 'x', 'op': 'variable', 'inputs': [], 'kind': 'quantity', 'dimensions': {'Q': 1}, 'attributes': {}},
            {'id': 'bias', 'op': 'constant', 'inputs': [], 'kind': 'quantity', 'dimensions': {'Q': 1}, 'attributes': {'value': 0.9}},
        ]},
    }
    with app.test_client() as client:
        response = client.post('/api/research/structure-candidate/cegis', json={
            'candidate': candidate,
            'cases': [
                {'id': 'a', 'bindings': {'x': 0}, 'expected': 1},
                {'id': 'b', 'bindings': {'x': 2}, 'expected': 3},
            ],
            'config': {'max_rounds': 8, 'max_candidates': 16, 'max_repairs': 8},
        })
    assert response.status_code == 200
    result = response.get_json()['result']
    assert result['status'] == 'accepted_candidates'
    assert result['accepted_candidate_hashes']


def test_ode_cegis_endpoint_executes_shared_model_family_loop():
    app.config.update(TESTING=True)
    times = np.linspace(0.0, 1.0, 11)
    with app.test_client() as client:
        response = client.post('/api/research/ode-cegis', json={
            'candidate': {'id': 'ode-seed', 'state_dim': 1, 'basis': ['linear'], 'coefficients': [[-0.5]]},
            'cases': [{'id': 'decay', 'times': times.tolist(), 'initial': [1.0],
                       'observations': np.exp(-0.5 * times).reshape(-1, 1).tolist()}],
            'config': {'max_rounds': 4, 'max_candidates': 8, 'max_repairs': 4},
        })
    assert response.status_code == 200
    body = response.get_json()
    assert body['success'] is True
    assert body['result']['adapter_family'] == 'ode'
    assert body['execution_policy'].startswith('bounded_ode_cegis')


def test_optimization_cegis_endpoint_executes_validated_linear_program():
    app.config.update(TESTING=True)
    candidate = {
        'id': 'lp-seed', 'kind': 'linear_program', 'variables': ['x', 'y'],
        'units': {'x': 'u', 'y': 'u'}, 'objective_coefficients': [1.0, 1.0],
        'direction': 'minimize', 'bounds': [[0.0, 10.0], [0.0, 10.0]],
        'A_ub': [[-1.0, -1.0]], 'b_ub': [-1.0], 'A_eq': [], 'b_eq': [],
    }
    with app.test_client() as client:
        response = client.post('/api/research/optimization-cegis', json={
            'candidate': candidate, 'cases': [{'id': 'base', 'expected_objective': 1.0}],
            'config': {'max_rounds': 4, 'max_candidates': 8, 'max_repairs': 4},
        })
    assert response.status_code == 200
    body = response.get_json()
    assert body['success'] is True
    assert body['result']['adapter_family'] == 'linear_program'
    assert body['execution_policy'].startswith('bounded_linear_program_cegis')


def test_binding_contract_endpoint_blocks_unresolved_units():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/binding-contract', json={
            'variables': [{'id': 'x', 'role': 'input', 'dimensions': {'L': 1}}],
            'target': {'id': 'y', 'role': 'target'},
        })
    assert response.status_code == 200
    body = response.get_json()['result']
    assert body['contract']['compile_gate'] == 'blocked'
    assert body['plan']['status'] == 'blocked'


def test_data_graph_search_endpoint_binds_json_rows_and_runs(tmp_path):
    app.config.update(TESTING=True)
    fake_result = {"schema_version": "mathmodel.graph-search-result/v1",
                   "termination": "candidate_pool_exhausted", "pareto_candidates": []}
    with patch('core.graph_search_artifacts.run_search_bundle') as runner:
        runner.return_value = (fake_result, tmp_path / "run-1")
        with app.test_client() as client:
            response = client.post('/api/research/data-graph-search', json={
                "feature": "x", "target": "y",
                "rows": [{"x": 0, "y": 0}, {"x": 1, "y": 1},
                         {"x": 2, "y": 4}, {"x": 3, "y": 9}, {"x": 4, "y": 16}],
            })
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert body["audit"]["causal_status"] == "not_assessed"


def test_data_graph_search_endpoint_accepts_four_feature_contract(tmp_path):
    app.config.update(TESTING=True)
    fake_result = {"schema_version": "mathmodel.graph-search-result/v1",
                   "termination": "candidate_pool_exhausted", "pareto_candidates": []}
    rows = [{"a": i, "b": i + 1, "c": i + 2, "d": i + 3, "y": i * 2}
            for i in range(8)]
    with patch('core.graph_search_artifacts.run_search_bundle') as runner:
        runner.return_value = (fake_result, tmp_path / "run-4")
        with app.test_client() as client:
            response = client.post('/api/research/data-graph-search', json={
                "features": ["a", "b", "c", "d"], "target": "y", "rows": rows,
            })
    assert response.status_code == 200
    assert response.get_json()["audit"]["features"] == ["a", "b", "c", "d"]


def test_data_graph_search_endpoint_rejects_unbounded_rows():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post('/api/research/data-graph-search', json={
            "feature": "x", "target": "y", "rows": []
        })
    assert response.status_code == 400


def test_gnn_interaction_endpoint_returns_predictive_only_result(monkeypatch):
    app.config.update(TESTING=True)
    fake = {"schema_version": "mathmodel.gnn-interaction-screen/v1",
            "status": "executed", "edges": [], "policy": "predictive_only"}
    monkeypatch.setattr("core.gnn_interaction_screen.discover_gnn_interactions", lambda *args, **kwargs: fake)
    with app.test_client() as client:
        response = client.post('/api/research/gnn-interactions', json={
            "target": "y", "rows": [{"a": 1, "b": 2, "y": 3}] * 60,
            "epochs": 10, "restarts": 1,
        })
    assert response.status_code == 200
    assert response.get_json()["result"]["status"] == "executed"


def test_module_catalog_endpoint_exposes_maturity_status():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get('/api/research/module-catalog')
    assert response.status_code == 200
    result = response.get_json()['result']
    assert result['status'] == 'valid'
    assert result['status_counts']['experimental'] >= 1


def test_research_can_enable_local_semantic_compiler_without_persisting_api_key():
    sid = "research-semantic-model-test"
    app.config.update(TESTING=True)
    user_sessions[sid] = {"train_events": [], "train_live_results": []}
    fake_result = {
        "problem": "求解线性方程组",
        "input_mode": "mechanistic_no_dataset",
        "charts": [],
        "model_results": [],
        "report_path": "ignored.md",
        "output_dir": "ignored",
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            with patch("core.modeling_assistant.MathModelingAssistant") as assistant_class:
                assistant_class.return_value.run.return_value.to_dict.return_value = fake_result
                response = client.post("/api/research/run", json={
                    "description": "求解线性方程组",
                    "run_modeling": False,
                    "generate_plots": False,
                    "semantic_model_compiler": True,
                    "semantic_provider": "ollama",
                    "semantic_base_url": "http://localhost:11434",
                    "semantic_model_name": "qwen2.5:3b",
                    "semantic_api_key": "top-secret",
                })
                assert response.status_code == 200
                compiler = assistant_class.call_args.kwargs["semantic_compiler"]
                assert compiler.config.model_name == "qwen2.5:3b"
                assert compiler.config.api_key == "top-secret"
                assert "top-secret" not in repr(user_sessions[sid])
    finally:
        user_sessions.pop(sid, None)


def test_hypothesis_proposal_is_opt_in_and_independent_of_fact_parser():
    sid = "research-hypothesis-test"
    app.config.update(TESTING=True)
    user_sessions[sid] = {"train_events": [], "train_live_results": []}
    fake_result = {"charts": [], "model_results": [], "input_mode": "mechanistic_no_dataset"}
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            for enabled in (False, True):
                with patch("core.modeling_assistant.MathModelingAssistant") as assistant_class:
                    assistant_class.return_value.run.return_value.to_dict.return_value = dict(fake_result)
                    response = client.post("/api/research/run", json={
                        "description": "研究运动规律", "run_modeling": False, "generate_plots": False,
                        "hypothesis_generation": enabled, "semantic_model_compiler": False,
                        "semantic_provider": "ollama", "semantic_api_key": "hypothesis-test-secret",
                    })
                    assert response.status_code == 200
                    assert assistant_class.call_args.kwargs["semantic_compiler"] is None
                    proposer = assistant_class.call_args.kwargs["hypothesis_generator"]
                    assert (proposer is not None) is enabled
                    if enabled:
                        assert proposer.config.api_key == "hypothesis-test-secret"
                    assert "hypothesis-test-secret" not in repr(user_sessions[sid])
    finally:
        user_sessions.pop(sid, None)


def test_hypothesis_ui_defaults_off_and_escapes_model_text():
    template = Path("web/templates/index.html").read_text(encoding="utf-8")
    script = Path("web/static/js/app.js").read_text(encoding="utf-8")
    assert 'id="research-hypothesis-generation">' in template
    assert "hypothesis_generation: hypothesisEnabled" in script
    assert "semanticEnabled || hypothesisEnabled" in script
    assert "escapeHtml(assumption.text)" in script
    assert "尚未求解 · 不作为事实或数值证据" in script
    assert "/api/research/clarify" in script
    assert "escapeHtml(question)" in script
    assert "question_options" in script
    assert "selectResearchClarificationOption" in script
    assert "escapeHtml(option.impact || '')" in script
    assert "suppressed_questions" in script
    assert "possible_repeated_questions" in script
    assert "clarification_contract_hash" in script


def test_research_clarification_creates_server_bound_revision_and_rerun_uses_it():
    from core.model_hypotheses import ProblemContract

    sid = "research-clarification-test"
    app.config.update(TESTING=True)
    contract = ProblemContract.create("研究降温过程。")
    proposal = {
        "problem_contract": contract.public(),
        "questions": ["环境温度是否恒定？"],
        "hypotheses": [],
    }
    original_result = {
        "specialized_results": {"model_hypotheses": proposal},
    }
    user_sessions[sid] = {
        "research_result": original_result,
        "train_events": [],
        "train_live_results": [],
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            response = client.post("/api/research/clarify", json={
                "question_index": 0,
                "answer": "恒定为 20 摄氏度。",
                "hard_constraint": True,
            })
            payload = response.get_json()
            assert response.status_code == 200
            assert payload["revision"] == 2
            assert payload["policy"]["previous_result_mutated"] is False
            assert payload["policy"]["api_key_persisted"] is False
            assert "环境温度是否恒定？" in payload["contract"]["statement"]
            assert "恒定为 20 摄氏度。" in payload["contract"]["statement"]
            assert payload["contract"]["hard_constraint_ids"] == ["user_confirmation_1"]
            assert original_result["specialized_results"]["model_hypotheses"]["problem_contract"]["revision"] == 1

            fake_result = {"charts": [], "model_results": [], "specialized_results": {}}
            with patch("core.modeling_assistant.MathModelingAssistant") as assistant_class:
                assistant_class.return_value.run.return_value.to_dict.return_value = fake_result
                rerun = client.post("/api/research/run", json={
                    "description": payload["contract"]["statement"],
                    "clarification_contract_hash": payload["contract_hash"],
                    "run_modeling": False,
                    "generate_plots": False,
                })
                assert rerun.status_code == 200
                rebound = assistant_class.return_value.run.call_args.kwargs["problem_contract"]
                assert rebound.digest == payload["contract_hash"]
                assert rebound.public()["revision"] == 2
    finally:
        user_sessions.pop(sid, None)


def test_research_clarification_rejects_unknown_question_and_stale_contract_hash():
    from core.model_hypotheses import ProblemContract

    sid = "research-clarification-rejection-test"
    contract = ProblemContract.create("研究系统。")
    user_sessions[sid] = {
        "research_result": {"specialized_results": {"model_hypotheses": {
            "problem_contract": contract.public(), "questions": ["边界是什么？"],
        }}},
        "train_events": [],
        "train_live_results": [],
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            invalid = client.post("/api/research/clarify", json={
                "question_index": 1, "answer": "伪造问题的回答",
            })
            assert invalid.status_code == 400

            accepted = client.post("/api/research/clarify", json={
                "question_index": 0, "answer": "边界为闭区间。",
            }).get_json()
            stale = client.post("/api/research/run", json={
                "description": accepted["contract"]["statement"],
                "clarification_contract_hash": "0" * 64,
            })
            assert stale.status_code == 409
            assert "版本不一致" in stale.get_json()["error"]
    finally:
        user_sessions.pop(sid, None)


def test_research_evidence_endpoint_downloads_machine_readable_bundle(tmp_path):
    sid = "research-evidence-test"
    app.config.update(TESTING=True)
    payload = {
        "overall_status": "conditional",
        "claims": [{"id": "claim_1", "grade": "conditionally_supported"}],
        "writing_contract": {"enabled": False},
    }
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "evidence_bundle.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    user_sessions[sid] = {"research_output_dir": str(tmp_path)}

    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            response = client.get("/api/research/evidence")
            assert response.status_code == 200
            assert json.loads(response.data.decode("utf-8"))["writing_contract"]["enabled"] is False
    finally:
        user_sessions.pop(sid, None)


def test_research_artifact_routes_enforce_layout_and_clear_only_cache(tmp_path):
    sid = "research-artifact-layout-test"
    app.config.update(TESTING=True)
    manager = RunArtifactManager(tmp_path, run_id="web_run")
    chart = manager.path("charts", "overview.png")
    chart.write_bytes(b"not-a-real-png-but-route-safe")
    manager.register_existing(
        "chart.001.overview", "charts", chart, media_type="image/png"
    )
    evidence = manager.write_json(
        "evidence.bundle", "evidence", "evidence_bundle.json", {"keep": True}
    )
    report = manager.write_text(
        "report.argument", "reports", "mathematical_argument.md", "# keep"
    )
    cache = manager.write_cache("web", {"input": 1}, {"cached": True})
    manager.finalize()
    user_sessions[sid] = {"research_output_dir": str(tmp_path)}

    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            assert client.get("/api/research/chart/charts/overview.png").status_code == 200
            assert client.get("/api/research/chart/evidence/evidence_bundle.json").status_code == 404
            manifest_response = client.get("/api/research/manifest")
            assert manifest_response.status_code == 200
            assert json.loads(manifest_response.data)["schema_version"] == "mathmodel.run-artifacts/v1"

            cleanup = client.delete("/api/research/cache")
            assert cleanup.status_code == 200
            assert cleanup.get_json()["cleanup"]["deleted_files"] == 1
            assert not cache.exists()
            assert evidence.is_file()
            assert report.is_file()
    finally:
        user_sessions.pop(sid, None)


def test_causal_discovery_endpoint_returns_partial_dag_hypothesis():
    rng = np.random.default_rng(19)
    x = rng.normal(size=60)
    y = 1.2 * x + rng.normal(scale=0.2, size=60)
    with app.test_client() as client:
        response = client.post('/api/research/causal-discovery', json={
            'data': np.column_stack([x, y]).tolist(),
            'variable_names': ['x', 'y'], 'bootstrap': 8,
        })
    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert payload['result']['contract']['evidence_status'] == 'partial'
