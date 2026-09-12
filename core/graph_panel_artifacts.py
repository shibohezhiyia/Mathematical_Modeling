"""File boundary for a frozen panel; holdout is opened only after snapshot save."""
from pathlib import Path
import sqlite3

from .artifact_manager import RunArtifactManager, create_run_id
from .confirmation_registry import ConfirmationRegistry, validate_study_id
from .graph_panel import FrozenModelPanel, confirm_frozen_panel, PANEL_VERSION
from .graph_search_artifacts import confirmation_report
from .model_hypotheses import HypothesisValidationError, decode_proposal, _require


def panel_report(result):
    lines = ["# 冻结模型组配对检验", "",
        "本次只检查预先固定的模型，不依据留出表现选冠军、不重新拟合。", "",
        f"面板指纹：`{result['panel_hash']}`", "",
        "完成运行不表示所有模型通过；超时、失败和未完成均需保留。",
        "相同留出样本和每臂等额额度不证明开发过程预算公平，也不构成统计显著性结论。", ""]
    if result.get("failure_code"):
        # Codes originate from internal validators, never include exception tracebacks.
        import html
        lines.extend(["本次未完成配对确认：" + html.escape(result["failure_code"]), ""])
    for arm in result.get("arms", []):
        # Labels are validated ASCII identifiers, not arbitrary markdown.
        lines.extend(["## 方法 " + arm["id"], ""])
        lines.extend(line.replace("## 冻结模型的留出检验", "### 留出检验")
                     for line in confirmation_report(arm["result"]))
    lines.extend(["", "仅限同一 study / 本地登记库的精确输入点使用记录，不认证外部快照的生成历史。",
                  "全局硬墙钟限制、时间/实体独立性和现实语义仍未验证。", ""])
    return "\n".join(lines)


def run_panel_confirmation(payload, *, holdout_path, output_root, study,
                           registry_path=None, cancel=None):
    """Validate/save entire panel, then read heldout file once, then consume it.

    This ordering applies to this entry point, not the history of externally
    supplied snapshots. A manifest checksum is not producer authentication.
    """
    validate_study_id(study)
    panel = FrozenModelPanel.from_payload(payload)
    destination = Path(output_root).resolve() / create_run_id()
    destination.mkdir(parents=True, exist_ok=False)
    manager = RunArtifactManager(destination)
    try:
        manager.write_json("panel.frozen", "evidence", "frozen_panel.json", panel.public(),
                           format_version=PANEL_VERSION, metadata={"role": "frozen_before_holdout_read"})
        # Save the study separately: it is an audit scope, not a model variable.
        manager.write_json("panel.protocol", "evidence", "panel_protocol.json", {
            "schema_version": "mathmodel.panel-file-protocol/v1", "study": study,
            "panel_hash": panel.digest, "external_history_authenticated": False,
            "holdout_results_may_feed_search": False})
        try:
            with Path(holdout_path).open("rb") as source:
                raw = source.read(256001)
            _require(len(raw) <= 256000, "holdout_file_size_limit")
            data = panel.validate_holdout(decode_proposal(raw.decode("utf-8-sig")))
            manager.write_json("panel.heldout-input", "evidence", "heldout_input.json", data.public(),
                               metadata={"role": "heldout_only_never_search_input"})
            registry = ConfirmationRegistry(registry_path or Path(__file__).resolve().parents[1] /
                                             "workspace" / "confirmation" / "usage.sqlite3")
            result = confirm_frozen_panel(panel, data.public(), registry=registry, study=study, cancel=cancel)
        except HypothesisValidationError as exc:
            result = {"status": "input_rejected", "failure_code": exc.code}
        except (OSError, UnicodeError):
            result = {"status": "execution_incomplete", "failure_code": "panel_file_unavailable"}
        except sqlite3.Error:
            result = {"status": "execution_incomplete", "failure_code": "confirmation_registry_unavailable"}
        result.update(panel_hash=panel.digest, may_feed_search=False, winner_selected=False)
        result.setdefault("schema_version", "mathmodel.panel-confirmation/v1")
        manager.write_json("panel.result", "evidence", "panel_confirmation.json", result,
                           format_version=result["schema_version"], metadata={"role": "readonly_final_stage"})
        manager.write_text("panel.report", "reports", "panel_confirmation.md", panel_report(result),
                           media_type="text/markdown; charset=utf-8")
        manager.finalize("complete" if result["status"] == "panel_completed" else "failed")
        return result, destination
    except BaseException:
        # Neither file errors nor interruption release a consumed reservation.
        manager.finalize("failed")
        raise
