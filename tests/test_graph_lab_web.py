import json
from pathlib import Path

from flask import Flask, session
import pytest

from test_graph_search import fixture_graph, experiment
from test_graph_confirmation import holdout, local_runner
from web.graph_lab import GraphLab, graph_lab_blueprint


class InlineThread:
    def __init__(self, target, **kwargs):
        self.target = target
    def start(self):
        self.target()


def bundle():
    contract, graph = fixture_graph("multiply", parameter=True)
    return {"schema_version": "mathmodel.graph-search-bundle/v1", "problem": contract.public(),
        "experiment": experiment(contract, graph, lambda x: 2*x, parameter_bounds={"a": [0, 5, 1]}).public(),
        "candidates": [graph.payload()], "budget": {"wall_seconds": 10}, "grammar_search": True}


@pytest.fixture
def web(monkeypatch, tmp_path):
    import web.graph_lab as module
    local_runner(monkeypatch)
    monkeypatch.setattr(module.threading, "Thread", InlineThread)
    lab = GraphLab(tmp_path / "runs", tmp_path / "usage.sqlite3")
    app = Flask(__name__)
    app.secret_key = "test-only"
    app.register_blueprint(graph_lab_blueprint(lab, lambda: session.get("owner", "alice")), url_prefix="/api/graph-lab")
    return app.test_client(), lab


def start(client):
    response = client.post("/api/graph-lab/run", json={"bundle": bundle(), "study": "paired"})
    assert response.status_code == 202
    return response.json["id"]


def test_separate_development_and_holdout_requests(web):
    client, lab = web
    key = start(client)
    state = client.get(f"/api/graph-lab/{key}").json
    assert state["status"] == "ready" and state["selected_hash"]
    assert state["confirmation"] is None and lab.active == 0
    directory = lab.root / key
    assert (directory / "evidence" / "frozen_model.json").exists()
    assert not (directory / "evidence" / "heldout_input.json").exists()
    response = client.post(f"/api/graph-lab/{key}/confirm", json=holdout())
    assert response.status_code == 202
    final = client.get(f"/api/graph-lab/{key}").json
    assert final["status"] == "done"
    assert final["confirmation"]["status"] == "passed_finite_heldout_checks"
    assert final["confirmation"]["parameters_refitted"] is False
    assert client.post(f"/api/graph-lab/{key}/confirm", json=holdout()).status_code == 409
    report = client.get(f"/api/graph-lab/{key}/artifact/report")
    assert report.status_code == 200 and "attachment" in report.headers["Content-Disposition"]
    assert "留出检验通过" in report.data.decode("utf-8")


def test_optional_http_model_mutation_is_bounded_and_key_is_not_persisted(web, monkeypatch):
    from core.semantic_model_compiler import HttpSemanticBackend

    client, lab = web
    contract, graph = fixture_graph()
    mutation_bundle = {
        "schema_version": "mathmodel.graph-search-bundle/v1",
        "problem": contract.public(),
        "experiment": experiment(contract, graph).public(),
        "candidates": [graph.payload()],
        "budget": {"wall_seconds": 20, "max_candidates": 3,
                   "max_patch_attempts": 4, "max_evaluations": 100,
                   "per_candidate_evaluations": 30},
        "grammar_search": False,
    }

    def propose_square(self, messages):
        request_payload = json.loads(messages[1]["content"])
        parent = request_payload["typed_graph"]
        output = dict(parent["nodes"][-1])
        output.update(op="multiply", inputs=["x", "x"], attributes={})
        return json.dumps({"patches": [{
            "parent_hash": request_payload["parent_hash"], "id": "square_model_patch",
            "replace_nodes": [output], "add_nodes": [], "remove_node_ids": [],
        }]})

    monkeypatch.setattr(HttpSemanticBackend, "complete", propose_square)
    secret = "never-write-this-key"
    response = client.post("/api/graph-lab/run", json={
        "bundle": mutation_bundle,
        "study": "model_mutation",
        "mutation_model": {
            "provider": "ollama", "base_url": "http://localhost:11434",
            "model_name": "fixture", "api_key": secret, "timeout_seconds": 15,
        },
    })
    assert response.status_code == 202
    key = response.json["id"]
    state = client.get(f"/api/graph-lab/{key}").json
    assert state["status"] == "ready"
    assert state["model_mutation"]["enabled"] is True
    assert state["model_mutation"]["calls"] == 1
    assert state["model_mutation"]["events"][0]["accepted_count"] == 1
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (lab.root / key).rglob("*") if path.is_file() and path.suffix in {".json", ".md"}
    )
    assert secret not in persisted
    assert secret not in repr(lab.jobs[key])


def test_graph_lab_rejects_invalid_mutation_provider_without_starting_job(web):
    client, lab = web
    response = client.post("/api/graph-lab/run", json={
        "bundle": bundle(), "study": "bad_mutation",
        "mutation_model": {
            "provider": "arbitrary_untrusted_callback", "base_url": "http://localhost:1",
            "model_name": "fixture", "api_key": "", "timeout_seconds": 15,
        },
    })
    assert response.status_code == 400
    assert response.json["error"] == "invalid_mutation_model_config"
    assert not any(job["study"] == "bad_mutation" for job in lab.jobs.values())


def test_session_ownership_covers_status_cancel_confirm_and_download(web):
    client, _ = web
    key = start(client)
    with client.session_transaction() as cookie:
        cookie["owner"] = "mallory"
    for method, suffix, payload in [("get", "", None), ("get", "/artifact/frozen", None),
                                     ("get", "/heldout-fields", None), ("post", "/confirm-table", {}),
                                     ("post", "/cancel", {}), ("post", "/confirm", holdout())]:
        response = getattr(client, method)(f"/api/graph-lab/{key}{suffix}", **({"json": payload} if payload is not None else {}))
        assert response.status_code == 404


def test_final_data_not_accepted_in_development_input(web):
    client, lab = web
    data = {"bundle": bundle(), "study": "paired", "holdout": holdout()}
    assert client.post("/api/graph-lab/run", json=data).status_code == 400
    assert not lab.jobs


def test_invalid_holdout_can_be_corrected_before_any_execution(web):
    client, lab = web
    key = start(client)
    invalid = holdout()
    invalid["cases"][0]["bindings"]["x"] = 0
    assert client.post(f"/api/graph-lab/{key}/confirm", json=invalid).status_code == 400
    assert client.get(f"/api/graph-lab/{key}").json["status"] == "ready"
    assert not lab.registry_path.exists()
    assert client.post(f"/api/graph-lab/{key}/confirm", json=holdout()).status_code == 202


def test_same_study_new_job_does_not_reset_consumption(web):
    client, _ = web
    key = start(client)
    client.post(f"/api/graph-lab/{key}/confirm", json=holdout())
    new_key = start(client)
    client.post(f"/api/graph-lab/{new_key}/confirm", json=holdout())
    state = client.get(f"/api/graph-lab/{new_key}").json
    assert state["status"] == "error" and state["error"] == "holdout_already_consumed_in_study"


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://[broken"])
def test_cross_origin_mutation_is_rejected(web, origin):
    client, lab = web
    response = client.post("/api/graph-lab/run", json={"bundle": bundle(), "study": "paired"}, headers={"Origin": origin})
    assert response.status_code == 403 and not lab.jobs


def test_request_limits_and_path_allowlist(web):
    client, lab = web
    assert client.post("/api/graph-lab/run", data="x", content_type="text/plain").status_code == 415
    assert client.post("/api/graph-lab/run", data="x"*300001, content_type="application/json").status_code == 413
    key = start(client)
    assert client.get(f"/api/graph-lab/{key}/artifact/protocol.json").status_code == 404
    assert client.get(f"/api/graph-lab/{key}/artifact/../../usage.sqlite3").status_code == 404
    assert "owner" not in client.get(f"/api/graph-lab/{key}").json


def test_public_mode_disables_lab(tmp_path):
    app = Flask(__name__)
    app.register_blueprint(graph_lab_blueprint(GraphLab(tmp_path, tmp_path / "registry"), lambda: "owner", lambda: True),
                           url_prefix="/api/graph-lab")
    with app.test_client() as client:
        assert client.get("/api/graph-lab/example").status_code == 403
        assert client.post("/api/graph-lab/run", json={}).status_code == 403


def test_running_guards_and_cancellation(web, monkeypatch):
    client, lab = web
    import web.graph_lab as module
    pending = []
    class Deferred(InlineThread):
        def start(self):
            pending.append(self.target)
    monkeypatch.setattr(module.threading, "Thread", Deferred)
    key = start(client)
    assert client.post("/api/graph-lab/run", json={"bundle": bundle(), "study": "other"}).status_code == 409
    assert client.post(f"/api/graph-lab/{key}/confirm", json=holdout()).status_code == 409
    assert client.post(f"/api/graph-lab/{key}/cancel", json={}).status_code == 202
    pending.pop()()
    assert client.get(f"/api/graph-lab/{key}").json["status"] == "cancelled"
    assert lab.active == 0


def test_background_failure_releases_capacity_without_private_traceback(web, monkeypatch):
    client, lab = web
    import web.graph_lab as module
    def fail(*args, **kwargs):
        raise RuntimeError("private path / API key")
    monkeypatch.setattr(module.GraphSearchSession, "run", fail)
    key = start(client)
    state = client.get(f"/api/graph-lab/{key}")
    assert state.json["status"] == "error" and lab.active == 0
    assert b"private path" not in state.data


def test_thread_start_failure_releases_capacity(web, monkeypatch):
    client, lab = web
    import web.graph_lab as module
    class Broken(InlineThread):
        def start(self):
            raise RuntimeError("cannot start")
    monkeypatch.setattr(module.threading, "Thread", Broken)
    response = client.post("/api/graph-lab/run", json={"bundle": bundle(), "study": "paired"})
    assert response.status_code == 503 and lab.active == 0


def test_global_capacity_blocks_third_worker(web, monkeypatch):
    client, lab = web
    import web.graph_lab as module
    pending = []
    class Deferred(InlineThread):
        def start(self):
            pending.append(self.target)
    monkeypatch.setattr(module.threading, "Thread", Deferred)
    first = start(client)
    with client.session_transaction() as cookie:
        cookie["owner"] = "bob"
    start(client)
    with client.session_transaction() as cookie:
        cookie["owner"] = "charlie"
    assert client.post("/api/graph-lab/run", json={"bundle": bundle(), "study": "paired"}).status_code == 429
    assert lab.active == 2
    for target in pending:
        target()
    assert lab.active == 0


def test_confirm_cancel_consumes_holdout(web, monkeypatch):
    client, lab = web
    key = start(client)
    import web.graph_lab as module
    from core.solver_runtime import SolverProcessRunner, SolverRuntimeError
    pending = []
    class Deferred(InlineThread):
        def start(self):
            pending.append(self.target)
    monkeypatch.setattr(module.threading, "Thread", Deferred)
    assert client.post(f"/api/graph-lab/{key}/confirm", json=holdout()).status_code == 202
    assert client.post(f"/api/graph-lab/{key}/confirm", json=holdout()).status_code == 409
    assert client.post(f"/api/graph-lab/{key}/cancel", json={}).status_code == 202
    def cancelled(self, executor, payload, *, limits, cancel):
        assert cancel.is_set()
        raise SolverRuntimeError("cancelled")
    monkeypatch.setattr(SolverProcessRunner, "execute", cancelled)
    pending.pop()()
    state = client.get(f"/api/graph-lab/{key}").json
    assert state["status"] == "cancelled" and state["confirmation"]["status"] == "execution_incomplete"
    assert lab.registry_path.exists() and lab.active == 0


def test_table_compiler_uses_only_current_session_frame(tmp_path, monkeypatch):
    from test_table_graph_compiler import frame, config
    import web.graph_lab as module
    local_runner(monkeypatch)
    monkeypatch.setattr(module.threading, "Thread", InlineThread)
    lab = GraphLab(tmp_path / "runs", tmp_path / "usage.sqlite3")
    app = Flask(__name__)
    app.secret_key = "test-only"
    frames = {"alice": frame(), "bob": None}
    def owner():
        return session.get("owner", "alice")
    app.register_blueprint(graph_lab_blueprint(lab, owner, current_frame=lambda: frames[owner()]), url_prefix="/api/graph-lab")
    with app.test_client() as client:
        assert client.get("/api/graph-lab/table-fields").json["rows"] == 30
        compiled = client.post("/api/graph-lab/compile-table", json=config())
        assert compiled.status_code == 200 and not lab.jobs
        started = client.post("/api/graph-lab/run", json={"bundle": compiled.json["bundle"], "study": "compiled"})
        assert started.status_code == 202
        assert client.get(f"/api/graph-lab/{started.json['id']}").json["status"] == "ready"
        with client.session_transaction() as cookie:
            cookie["owner"] = "bob"
        assert client.get("/api/graph-lab/table-fields").status_code == 400
        assert client.post("/api/graph-lab/compile-table", json=config()).status_code == 400


def test_heldout_table_snapshot_is_immutable_and_audited(tmp_path, monkeypatch):
    from test_table_graph_compiler import frame, config
    from test_heldout_table_mapper import validation_table, mapping
    from core.table_graph_compiler import compile_table_graph
    import web.graph_lab as module
    local_runner(monkeypatch)
    monkeypatch.setattr(module.threading, "Thread", InlineThread)
    lab = GraphLab(tmp_path / "runs", tmp_path / "usage.sqlite3")
    current = {"frame": validation_table()}
    app = Flask(__name__)
    app.register_blueprint(graph_lab_blueprint(lab, lambda: "owner", current_frame=lambda: current["frame"]),
                           url_prefix="/api/graph-lab")
    with app.test_client() as client:
        bundle = compile_table_graph(frame(), config())["bundle"]
        key = client.post("/api/graph-lab/run", json={"bundle": bundle, "study": "paired"}).json["id"]
        first = client.get(f"/api/graph-lab/{key}/heldout-fields").json
        second = client.get(f"/api/graph-lab/{key}/heldout-fields").json
        stale = {**mapping(), "table_snapshot": first["table_snapshot"]}
        assert client.post(f"/api/graph-lab/{key}/confirm-table", json=stale).status_code == 400
        # Both in-place editing and replacing the current table must not change
        # the committed numeric snapshot chosen during field mapping.
        current["frame"].loc[:, "长度厘米"] = 1e9
        current["frame"] = None
        payload = {**mapping(), "table_snapshot": second["table_snapshot"]}
        response = client.post(f"/api/graph-lab/{key}/confirm-table", json=payload)
        assert response.status_code == 202
        state = client.get(f"/api/graph-lab/{key}").json
        assert state["confirmation"]["status"] == "passed_finite_heldout_checks"
        evidence = client.get(f"/api/graph-lab/{key}/artifact/heldout-mapping").json
        assert evidence["table_snapshot"] == second["table_snapshot"]
        assert evidence["selected_rows"] == 4
        assert client.get(f"/api/graph-lab/{key}/heldout-fields").status_code == 409
        assert client.post(f"/api/graph-lab/{key}/confirm-table", json=payload).status_code == 409


def test_heldout_fields_cannot_read_table_before_freeze(web, monkeypatch):
    client, lab = web
    import web.graph_lab as module
    class Deferred(InlineThread):
        def start(self):
            pass
    monkeypatch.setattr(module.threading, "Thread", Deferred)
    key = start(client)
    def must_not_read():
        pytest.fail("read table before freezing")
    with pytest.raises(module.LabError):
        lab.heldout_fields("alice", key, must_not_read)
    assert client.get(f"/api/graph-lab/{key}/heldout-fields").status_code == 409


def test_heldout_window_is_bounded_and_preserves_source_offsets(tmp_path, monkeypatch):
    from test_table_graph_compiler import frame, config
    from test_heldout_table_mapper import validation_table, mapping
    from core.table_graph_compiler import compile_table_graph
    import web.graph_lab as module
    import pandas as pd
    local_runner(monkeypatch)
    monkeypatch.setattr(module.threading, "Thread", InlineThread)
    source = pd.concat([validation_table()]*100, ignore_index=True)
    lab = GraphLab(tmp_path / "runs", tmp_path / "usage.sqlite3")
    job = lab.start("owner", compile_table_graph(frame(), config())["bundle"], "window")
    fields = lab.heldout_fields("owner", job, lambda: source, start=300)
    assert fields["rows"] == 400 and fields["snapshot_rows"] == 100 and fields["snapshot_start"] == 300
    lab.confirm_table("owner", job, {**mapping(), "start_row": 300, "table_snapshot": fields["table_snapshot"]})
    evidence = json.loads(lab.artifact("owner", job, "heldout-mapping").read_text(encoding="utf-8"))
    assert evidence["start_row"] == 300 and evidence["source_rows"] == 400
