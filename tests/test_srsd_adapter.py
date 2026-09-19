import json
import pickle

import numpy as np

from core.srsd_adapter import load_srsd_cases, run_srsd_pilot, select_srsd_instances


def _write_fixture(root):
    for name in ("train", "test", "true_eq"):
        (root / name).mkdir()
    (root / "README.md").write_text("license: cc-by-4.0", encoding="utf-8")
    info = {}
    for ordinal, case_id in enumerate(("case-a", "case-b")):
        x = np.linspace(1.0, 4.0, 48)
        data = np.column_stack([x, 2.0 * x + ordinal])
        np.savetxt(root / "train" / f"{case_id}.txt", data)
        np.savetxt(root / "test" / f"{case_id}.txt", data)
        with (root / "true_eq" / f"{case_id}.pkl").open("wb") as handle:
            pickle.dump(f"2*x0+{ordinal}", handle)
        info[case_id] = {"sympy_eq_str": f"2*x0+{ordinal}"}
    (root / "supp_info.json").write_text(json.dumps(info), encoding="utf-8")


def test_srsd_selection_loading_and_independent_model_scoring(tmp_path):
    _write_fixture(tmp_path)
    selected = select_srsd_instances(tmp_path, count=2, seed=9)
    cases = load_srsd_cases(tmp_path, instance_ids=selected, source_revision="a" * 40,
                            train_limit=32, test_limit=16, seed=9)
    assert len(cases) == 2
    assert "ground_truth" not in str(cases[0]["public_input"])
    def solver(payload):
        offset = 0.0 if selected[0] == "case-a" else 1.0
        return {"model": {"structure": "affine", "input_variables": ["x0"],
                          "coefficients": [2.0, offset]},
                "predictions": [2.0 * row[0] + offset for row in payload["query_inputs"]]}
    report = run_srsd_pilot([cases[0]], solver=solver)
    assert report["rows"][0]["model_reexecution_status"] == "verified"
    assert report["rows"][0]["nmse"] == 0.0
    assert "official_srsd_snapshot" in report["policy"]
