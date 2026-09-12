"""Optional real-browser smoke test; never install a browser during the test."""
import json
import threading

import pytest

from test_graph_lab_web import bundle
from test_graph_confirmation import holdout


@pytest.mark.parametrize("source_mode", ["json", "table"])
def test_browser_staged_graph_experiment(tmp_path, monkeypatch, source_mode):
    playwright = pytest.importorskip("playwright.sync_api")
    from werkzeug.serving import make_server
    import web.app as module
    monkeypatch.setattr(module.graph_lab, "root", tmp_path / "runs")
    monkeypatch.setattr(module.graph_lab, "registry_path", tmp_path / "usage.sqlite3")
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(module, "UPLOAD_DIR", tmp_path / "uploads")
    server = make_server("127.0.0.1", 0, module.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    with playwright.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except playwright.Error:
            try:
                browser = pw.chromium.launch(headless=True, channel="msedge")
            except playwright.Error:
                server.server_close()
                pytest.skip("No usable Chromium/Edge runtime; no download attempted")
        thread.start()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 960})
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.locator("details.graph-lab > summary").click()
            assert page.locator("#graph-lab-heldout").is_disabled()
            page.locator("#graph-lab-study").fill("browser_smoke")
            if source_mode == "json":
                page.locator("#graph-lab-input").set_input_files({"name": "development.json", "mimeType": "application/json",
                    "buffer": json.dumps(bundle()).encode()})
            else:
                from test_table_graph_compiler import frame
                response = page.request.post(f"http://127.0.0.1:{server.server_port}/api/upload", multipart={
                    "file": {"name": "table.csv", "mimeType": "text/csv", "buffer": frame().to_csv(index=False).encode("utf-8")}})
                assert response.ok
                page.locator("#graph-lab-table-builder > summary").click()
                page.get_by_role("button", name="读取当前表格字段", exact=True).click()
                playwright.expect(page.locator("#graph-lab-table-source")).to_contain_text("30 行")
                page.locator("#graph-lab-table-inputs").select_option(["时间"])
                page.locator("#graph-lab-table-target").select_option("距离")
                page.locator('#graph-lab-table-units select[data-column="时间"]').select_option("s")
                page.locator('#graph-lab-table-units select[data-column="距离"]').select_option("m")
                page.locator("#graph-lab-table-count").fill("30")
                page.locator("#graph-lab-table-problem").fill("检验距离与时间的关系。")
                page.locator("#graph-lab-table-compile").click()
                playwright.expect(page.locator("#graph-lab-table-preview")).to_contain_text("实验已准备")
            page.locator("#graph-lab-start").click()
            playwright.expect(page.locator("#graph-lab-confirm")).to_be_enabled(timeout=30000)
            assert "候选已冻结" in page.locator("#graph-lab-output").inner_text()
            if source_mode == "json":
                page.locator("#graph-lab-heldout").set_input_files({"name": "heldout.json", "mimeType": "application/json",
                    "buffer": json.dumps(holdout()).encode()})
                page.locator("#graph-lab-confirm").click()
            else:
                from test_heldout_table_mapper import validation_table
                response = page.request.post(f"http://127.0.0.1:{server.server_port}/api/upload", multipart={
                    "file": {"name": "validation.csv", "mimeType": "text/csv", "buffer": validation_table().to_csv(index=False).encode("utf-8")}})
                assert response.ok
                page.locator("#graph-lab-heldout-table > summary").click()
                page.locator("#graph-lab-heldout-fields").click()
                playwright.expect(page.locator("#graph-lab-heldout-source")).to_contain_text("已固定第 1–4 行")
                page.locator('#graph-lab-heldout-mapping select[data-node="x0"][data-role="column"]').select_option("时长分钟")
                page.locator('#graph-lab-heldout-mapping select[data-node="x0"][data-role="unit"]').select_option("min")
                page.locator('#graph-lab-heldout-mapping select[data-node="y"][data-role="column"]').select_option("长度厘米")
                page.locator('#graph-lab-heldout-mapping select[data-node="y"][data-role="unit"]').select_option("cm")
                page.locator("#graph-lab-heldout-si").check()
                page.locator("#graph-lab-confirm-table").click()
            playwright.expect(page.locator("#graph-lab-output")).to_contain_text("有限留出检查通过", timeout=30000)
            assert page.locator("#graph-lab-confirm").is_disabled()
            assert page.locator("#graph-lab-confirm-table").is_disabled()
            page.locator("details.graph-lab").screenshot(path=str(tmp_path / "graph_lab_desktop.png"))
            page.set_viewport_size({"width": 760, "height": 960})
            assert page.locator(".graph-lab-stages").evaluate("el => getComputedStyle(el).gridTemplateColumns.split(' ').length") == 1
            if source_mode == "table":
                page.locator("#graph-lab-table-atol").fill("0.2")
                assert page.locator("#graph-lab-confirm").is_disabled()
                assert page.locator("#graph-lab-output").inner_text() == ""
                assert "重新准备实验" in page.locator("#graph-lab-table-preview").inner_text()
            print(f"Graph lab screenshot: {tmp_path / 'graph_lab_desktop.png'}")
        finally:
            browser.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
