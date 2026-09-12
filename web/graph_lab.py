"""Session-scoped asynchronous graph experiments, separate from automatic research."""
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import threading
import uuid

from flask import Blueprint, jsonify, request, send_file

from core.artifact_manager import RunArtifactManager
from core.confirmation_registry import ConfirmationRegistry, confirm_frozen_model, validate_study_id
from core.graph_benchmark import _select_development
from core.graph_confirmation import HeldoutCases
from core.graph_experiments import SearchExperiment, restore_problem
from core.graph_search import GraphSearchBudget, GraphSearchSession
from core.graph_search_artifacts import BUNDLE_VERSION, search_report
from core.model_hypotheses import HypothesisIR, HypothesisValidationError, _canonical, _keys, _require, decode_proposal


class LabError(Exception):
    def __init__(self, code, status=409):
        self.code, self.status = code, status


class GraphLab:
    """Two active workers / bounded in-memory history per server process.

    Files survive restart; session handles do not. This is a local tool, not a
    distributed job queue or filesystem/network sandbox for arbitrary code.
    """
    def __init__(self, root, registry_path):
        self.root, self.registry_path = Path(root), Path(registry_path)
        self.lock = threading.RLock()
        self.jobs = OrderedDict()
        self.active = 0

    def _owned(self, owner, key):
        job = self.jobs.get(key)
        if job is None or job["owner"] != owner:
            raise LabError("job_not_found", 404)
        return job

    def _capacity(self, owner):
        if self.active >= 2:
            raise LabError("server_busy", 429)
        if any(j["owner"] == owner and j["status"] in ("running", "confirming") for j in self.jobs.values()):
            raise LabError("session_busy")

    def start(self, owner, bundle, study, mutation_model=None):
        validate_study_id(study)
        bundle = decode_proposal(_canonical(bundle))
        _keys(bundle, {"schema_version", "problem", "experiment", "candidates", "budget"}, {"grammar_search"})
        _require(bundle["schema_version"] == BUNDLE_VERSION, "bundle_version_mismatch")
        _require(type(bundle.get("grammar_search", True)) is bool, "invalid_search_switch")
        contract = restore_problem(bundle["problem"])
        experiment = SearchExperiment.from_payload(bundle["experiment"], contract)
        _keys(bundle["budget"], set(), set(asdict(GraphSearchBudget())))
        budget = GraphSearchBudget(**bundle["budget"])
        _require(type(bundle["candidates"]) is list and 1 <= len(bundle["candidates"]) <= 16, "candidate_budget")
        for graph in bundle["candidates"]:
            HypothesisIR.from_payload(graph, contract)
        patch_generator = None
        mutation_public = None
        if mutation_model is not None:
            _keys(mutation_model, {
                "provider", "base_url", "model_name", "api_key", "timeout_seconds",
            })
            try:
                from core.graph_patch_generator import GraphPatchGenerator
                from core.semantic_model_compiler import SemanticCompilerConfig
                config = SemanticCompilerConfig(
                    provider=mutation_model["provider"],
                    base_url=mutation_model["base_url"],
                    model_name=mutation_model["model_name"],
                    api_key=mutation_model["api_key"],
                    timeout_seconds=mutation_model["timeout_seconds"],
                    max_output_tokens=4096,
                ).validate()
                patch_generator = GraphPatchGenerator(config)
                mutation_public = config.public()
            except (TypeError, ValueError):
                raise HypothesisValidationError("invalid_mutation_model_config") from None
        with self.lock:
            self._capacity(owner)
            if len(self.jobs) >= 32:
                old = next((k for k, j in self.jobs.items() if j["status"] not in ("running", "confirming")), None)
                if old is None:
                    raise LabError("server_busy", 429)
                del self.jobs[old]  # Only expire a handle, never delete evidence files.
            key = uuid.uuid4().hex
            directory = self.root / key
            directory.mkdir(parents=True, exist_ok=False)
            manager = RunArtifactManager(directory)
            manager.write_json("lab.input", "evidence", "development_input.json", bundle)
            manager.write_json("lab.protocol", "evidence", "protocol.json", {
                "study": study, "selection": "minimum_nodes_then_development_rmse_then_hash",
                "mutation_model": mutation_public,
                "external_model_is_judge": False,
                "api_key_persisted": False,
                "holdout_upload_requires_frozen_candidate": True})
            job = {"owner": owner, "id": key, "study": study, "status": "running", "cancel": threading.Event(),
                   "manager": manager, "model": None, "result": None, "confirmation": None, "error": None,
                   "mutation_model": mutation_public}
            self.jobs[key] = job
            self.active += 1
            def work():
                session = GraphSearchSession(contract, experiment, budget=budget)
                result = session.run(
                    bundle["candidates"],
                    grammar_search=bundle.get("grammar_search", True),
                    cancel=job["cancel"],
                    model_patch_generator=patch_generator,
                )
                selected = _select_development(result)
                model = session.freeze(selected) if selected else None
                if model:
                    manager.write_json("lab.frozen", "evidence", "frozen_model.json", model.public())
                manager.write_json("lab.development", "evidence", "development_result.json", result)
                manager.write_text("lab.report", "reports", "experiment.md", search_report(result), media_type="text/markdown; charset=utf-8")
                return dict(result=result, model=model, status="cancelled" if job["cancel"].is_set()
                            else "ready" if model else "no_candidate")
            self._launch(job, work)
            return key

    def _launch(self, job, work):
        def target():
            updates = {}
            try:
                updates = work()
            except HypothesisValidationError as exc:
                updates = dict(status="error", error=exc.code)
            except BaseException:
                # This handler belongs to the background worker only. A thread
                # exit must not strand its capacity slot or look like success.
                updates = dict(status="error", error="execution_failed")
            finally:
                with self.lock:
                    job.update(updates)
                    try:
                        job["manager"].finalize("failed" if job["status"] == "error" else
                                                "incomplete" if job["status"] in ("cancelled", "ready") else "complete")
                    except Exception:
                        job.update(status="error", error="artifact_write_failed")
                    self.active -= 1
        try:
            threading.Thread(target=target, daemon=True).start()
        except Exception:
            job.update(status="error", error="worker_start_failed")
            self.active -= 1
            job["manager"].finalize("failed")
            raise LabError("worker_start_failed", 503)

    def confirm(self, owner, key, payload):
        with self.lock:
            job = self._owned(owner, key)
            if job["status"] != "ready":
                raise LabError("model_not_ready_or_already_used")
            self._capacity(owner)
            # Final labels first become available after this job's private model
            # was selected and its complete snapshot persisted.
            data = HeldoutCases.from_payload(payload, job["model"])
            job["manager"].write_json("lab.heldout", "evidence", "heldout_input.json", data.public())
            job.update(status="confirming")
            self.active += 1
            def work():
                final = confirm_frozen_model(job["model"], data.public(), study=job["study"], cancel=job["cancel"],
                                             registry=ConfirmationRegistry(self.registry_path))
                job["manager"].write_json("lab.confirmation", "evidence", "confirmation.json", final)
                view = {**job["result"], "confirmation": final}
                job["manager"].write_text("lab.report", "reports", "experiment.md", search_report(view),
                                           media_type="text/markdown; charset=utf-8")
                return dict(confirmation=final, status="cancelled" if job["cancel"].is_set() else "done")
            self._launch(job, work)

    def cancel(self, owner, key):
        with self.lock:
            job = self._owned(owner, key)
            if job["status"] not in ("running", "confirming"):
                raise LabError("job_not_running")
            job["cancel"].set()

    def heldout_fields(self, owner, key, current_frame, start=0):
        from core.heldout_table_mapper import frozen_table_roles
        from core.table_graph_compiler import table_fields
        with self.lock:
            job = self._owned(owner, key)
            if job["status"] != "ready":
                raise LabError("model_not_ready_or_already_used")
            frame = current_frame()
            fields = table_fields(frame)
            _require(type(start) is int and 0 <= start < len(frame), "invalid_heldout_window")
            names = [c["name"] for c in fields["columns"] if not c["identifier"]]
            snapshot = frame.iloc[start:start+256][names].copy(deep=True)
            token = uuid.uuid4().hex
            job["table_snapshot"] = {"token": token, "frame": snapshot, "start": start, "source_rows": len(frame)}
            return {**fields, **frozen_table_roles(job["model"]), "table_snapshot": token,
                    "snapshot_start": start, "snapshot_rows": len(snapshot)}

    def confirm_table(self, owner, key, config):
        from core.heldout_table_mapper import map_heldout_table
        with self.lock:
            job = self._owned(owner, key)
            if job["status"] != "ready":
                raise LabError("model_not_ready_or_already_used")
            self._capacity(owner)
            _keys(config, {"bindings", "start_row", "row_count", "si_contract_confirmed", "table_snapshot"})
            snapshot = job.get("table_snapshot")
            _require(snapshot is not None and config["table_snapshot"] == snapshot["token"], "stale_heldout_table_snapshot")
            _require(type(config["start_row"]) is int, "invalid_heldout_window")
            local_config = {k: v for k, v in config.items() if k != "table_snapshot"}
            local_config["start_row"] -= snapshot["start"]
            data, mapping = map_heldout_table(job["model"], snapshot["frame"], local_config)
            mapping.update(start_row=config["start_row"], source_rows=snapshot["source_rows"], table_snapshot=snapshot["token"],
                           snapshot_start=snapshot["start"], snapshot_rows=len(snapshot["frame"]))
            job["manager"].write_json("lab.heldout-mapping", "evidence", "heldout_mapping.json", mapping)
            self.confirm(owner, key, data.public())

    def status(self, owner, key):
        with self.lock:
            job = self._owned(owner, key)
            result = job["result"] or {}
            model = job["model"]
            return {"id": key, "status": job["status"], "study": job["study"], "error": job["error"],
                "cancel_requested": job["cancel"].is_set(), "candidate_count": len(result.get("pareto_candidates", [])),
                "selected_hash": model.public()["development_evidence"]["hypothesis_hash"] if model else None,
                "budget": result.get("budget"), "confirmation": job["confirmation"],
                "model_mutation": {
                    "enabled": job.get("mutation_model") is not None,
                    "calls": (result.get("policy") or {}).get("external_api_calls", 0),
                    "events": [{k: event.get(k) for k in ("status", "error_code", "accepted_count")}
                               for event in result.get("external_patch_events", [])],
                    "model_is_judge": False,
                },
                "reports": [{k: r.get(k) for k in ("hypothesis_hash", "status", "search_rmse", "failure_code")}
                            for r in result.get("reports", [])],
                "downloads": [name for name, rel in FILES.items() if (job["manager"].root / rel).is_file()]}

    def artifact(self, owner, key, name):
        with self.lock:
            job = self._owned(owner, key)
            if name not in FILES:
                raise LabError("artifact_not_found", 404)
            path = job["manager"].root / FILES[name]
            if not path.is_file():
                raise LabError("artifact_not_found", 404)
            return path


FILES = {"report": "reports/experiment.md", "development": "evidence/development_result.json",
         "frozen": "evidence/frozen_model.json", "confirmation": "evidence/confirmation.json",
         "heldout-mapping": "evidence/heldout_mapping.json",
         "manifest": "artifact_manifest.json"}


def graph_lab_blueprint(lab, owner, public_mode=lambda: False, current_frame=lambda: None):
    bp = Blueprint("graph_lab", __name__)

    @bp.before_request
    def guard():
        # Resource limits are NOT a public multi-tenant security boundary.
        if public_mode():
            return jsonify(error="local_feature_only"), 403
        if request.method == "POST":
            from urllib.parse import urlsplit
            origin = request.headers.get("Origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                    same = parsed.netloc == request.host and parsed.scheme == request.scheme
                except ValueError:
                    same = False
                if not same:
                    return jsonify(error="cross_origin_rejected"), 403
            if not request.is_json:
                return jsonify(error="json_required"), 415
            if request.content_length is not None and request.content_length > 300000:
                return jsonify(error="request_size_limit"), 413

    @bp.errorhandler(LabError)
    def lab_error(exc):
        return jsonify(error=exc.code), exc.status

    @bp.after_request
    def no_cached_evidence(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @bp.errorhandler(HypothesisValidationError)
    def invalid(exc):
        return jsonify(error=exc.code), 400

    @bp.errorhandler(OSError)
    def unavailable(exc):
        return jsonify(error="artifact_storage_unavailable"), 503

    def body():
        raw = request.stream.read(300001)
        _require(len(raw) <= 300000, "request_size_limit")
        try:
            return decode_proposal(raw.decode("utf-8"))
        except UnicodeError:
            raise HypothesisValidationError("invalid_encoding") from None

    @bp.post("/run")
    def run():
        data = body()
        _keys(data, {"bundle", "study"}, {"mutation_model"})
        return jsonify(id=lab.start(
            owner(), data["bundle"], data["study"], data.get("mutation_model"),
        )), 202

    @bp.get("/<key>")
    def status(key):
        return jsonify(lab.status(owner(), key))

    @bp.post("/<key>/confirm")
    def confirm(key):
        lab.confirm(owner(), key, body())
        return jsonify(status="confirming"), 202

    @bp.post("/<key>/cancel")
    def cancel(key):
        _keys(body(), set())
        lab.cancel(owner(), key)
        return jsonify(status="cancellation_requested"), 202

    @bp.get("/<key>/artifact/<name>")
    def artifact(key, name):
        return send_file(lab.artifact(owner(), key, name), as_attachment=True)

    @bp.get("/example")
    def example():
        # Fixed, synthetic development input only; no caller-supplied path.
        return send_file(Path(__file__).resolve().parents[1] / "examples" / "graph_search_scalar.json",
                         mimetype="application/json")

    @bp.get("/table-fields")
    def fields():
        from core.table_graph_compiler import table_fields
        return jsonify(table_fields(current_frame()))

    @bp.post("/compile-table")
    def compile_table():
        from core.table_graph_compiler import compile_table_graph
        return jsonify(compile_table_graph(current_frame(), body()))

    @bp.get("/<key>/heldout-fields")
    def heldout_fields(key):
        raw_start = request.args.get("start", "0")
        _require(raw_start.isascii() and raw_start.isdigit() and len(raw_start) <= 12, "invalid_heldout_window")
        return jsonify(lab.heldout_fields(owner(), key, current_frame, int(raw_start)))

    @bp.post("/<key>/confirm-table")
    def confirm_table(key):
        lab.confirm_table(owner(), key, body())
        return jsonify(status="confirming"), 202

    return bp
