"""Web research endpoint regression tests."""

from unittest.mock import patch
import json
from pathlib import Path

import pandas as pd
import pytest

from web.app import app, user_sessions, _prepare_research_frame
from core.artifact_manager import RunArtifactManager


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
