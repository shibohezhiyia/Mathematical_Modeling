"""Headless browser checks for the ordinary research result's trust labels."""
from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest
from werkzeug.serving import make_server

from web.app import app, user_sessions


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "web" / "static" / "js" / "app.js"
DIAGNOSTICS_SCRIPT = ROOT / "web" / "static" / "js" / "model_diagnostics.js"
GRAPH_SCRIPT = ROOT / "web" / "static" / "js" / "graph_lab.js"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def test_browser_renders_validated_and_underobserved_results_in_chinese():
    playwright = pytest.importorskip("playwright.sync_api")
    if not EDGE.is_file():
        pytest.skip("headless Edge unavailable")
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True, executable_path=str(EDGE))
        try:
            page = browser.new_page()
            page.set_content('<div id="research-result" class="research-result hidden"></div>')
            page.add_script_tag(path=str(DIAGNOSTICS_SCRIPT))
            page.add_script_tag(path=str(GRAPH_SCRIPT))
            page.add_script_tag(path=str(SCRIPT))
            base = {"problem_analysis": {"model_class": "代数关系", "confidence": 90},
                    "charts": [], "dataset_profiles": [], "relationships": [], "interactions": []}
            validated = {**base, "specialized_results": {"automatic_modeling": {
                "status": "completed", "result_grade": "validated_candidate",
                "recommended_action": "use_with_stated_scope",
                "model": {"family": "modeling_algebra", "structure": "affine"},
                "predictions": [2.4], "usage": {"manual_interventions": 0},
                "routing_evidence": {"selected_arm": "current_bounded_grammar",
                                     "training_row_count": 64, "validation_row_count": 16,
                                     "validation_metrics": {"current_bounded_grammar": {
                                         "nmse": 0.0, "acc_0.1": 1.0}}},
            }}}
            page.evaluate("(data) => renderResearchResult(data)", validated)
            box = page.locator("#research-result")
            assert "通过当前验证的候选" in box.inner_text()
            assert "验证依据" in box.inner_text()
            assert box.locator(".research-safe").count() >= 1

            underobserved = {**base, "specialized_results": {"automatic_modeling": {
                "status": "needs_input", "reason": "portfolio_transition_region_underobserved",
                "result_grade": "abstain",
                "recommended_action": "collect_observations_within_suggested_transition_interval",
                "routing_evidence": {"suggested_observation_interval": [-0.1, 0.1]},
            }}}
            page.evaluate("(data) => renderResearchResult(data)", underobserved)
            text = box.inner_text()
            assert "需要补充观测" in text
            assert "过渡区观测不足" in text
            assert "建议补点区间" in text
            assert "portfolio_transition_region_underobserved" not in text
            assert box.locator(".research-risk").count() >= 1
        finally:
            browser.close()


def test_browser_upload_problem_and_run_reaches_validated_result():
    playwright = pytest.importorskip("playwright.sync_api")
    if not EDGE.is_file():
        pytest.skip("headless Edge unavailable")
    was_testing = app.config.get("TESTING", False)
    app.config.update(TESTING=True)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    session_ids_before = set(user_sessions)
    try:
        with playwright.sync_playwright() as driver:
            browser = driver.chromium.launch(headless=True, executable_path=str(EDGE))
            try:
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="domcontentloaded")
                csv = "driver,measured_kw\n" + "".join(
                    f"{index / 10.0},{1.6 * (index / 10.0) + 0.4}\n"
                    for index in range(80)
                )
                page.locator("#file-input").set_input_files({
                    "name": "browser_raw_case.csv", "mimeType": "text/csv",
                    "buffer": csv.encode("utf-8"),
                })
                page.wait_for_function("uploadedFiles.length === 1", timeout=30000)
                page.locator('.step-item[data-step="1"]').click()
                page.locator("#problem-description").fill(
                    "根据原始记录建模，并预测 driver = 1.25 时的 measured_kw。"
                )
                page.locator("#research-target").fill("measured_kw")
                assert page.locator("#research-symbolic-arm-budget").input_value() == "2"
                page.locator("#research-run-model").uncheck(force=True)
                page.locator("#research-feedback-optimize").uncheck(force=True)
                page.locator("#research-credibility-audit").uncheck(force=True)
                page.evaluate("document.getElementById('research-run-btn').click()")
                page.wait_for_function(
                    "document.querySelector('#research-result')?.textContent.includes('通过当前验证的候选')",
                    timeout=90000,
                )
                result_text = page.locator("#research-result").inner_text()
                assert "验证依据" in result_text
                assert "预测" in result_text
                assert "2.4" in result_text
                page.locator('#research-result details').filter(has_text='本浏览器会话的输入负担').click()
                burden_text = page.locator('#research-result').inner_text()
                assert '本浏览器会话的输入负担' in burden_text
                assert '首尾操作间隔' in burden_text
                assert '不是人工操作耗时' in burden_text
                session_burdens = [sdata.get('research_input_burden') for sid, sdata in user_sessions.items()
                                   if sid not in session_ids_before and sdata.get('research_input_burden')]
                assert any(row.get('uploaded_files') == 1 and row.get('run_submissions') == 1
                           and row.get('explicit_target_submissions') == 1 for row in session_burdens)
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        app.config.update(TESTING=was_testing)
        for sid in set(user_sessions) - session_ids_before:
            sdata = user_sessions.pop(sid)
            for file_info in sdata.get("uploaded_files", []):
                saved = Path(file_info.get("path", ""))
                if saved.is_file() and saved.parent.resolve() == (
                    ROOT / "data" / "uploads"
                ).resolve():
                    saved.unlink()
