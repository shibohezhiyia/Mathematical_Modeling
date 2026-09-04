"""Regression tests for mathematically valid classification blending."""

import numpy as np

from core.modeling_engine import CVResult, EnsembleBuilder, EnsembleMethod, TaskType


def _cv_result(key, prediction, probability=None, score=0.8):
    return CVResult(
        model_key=key,
        model_name=key,
        mean_scores={"f1_weighted": score},
        oof_pred=np.asarray(prediction),
        oof_proba=None if probability is None else np.asarray(probability),
    )


def test_weighted_classification_blends_probabilities_not_class_ids():
    # The two models disagree on their hard labels, but both assign 0.51 to
    # the positive class.  Probability blending must therefore predict class 1
    # for both rows; averaging class IDs would produce meaningless 0.5 values.
    results = [
        _cv_result("first", [0, 1], [0.51, 0.51]),
        _cv_result("second", [1, 0], [0.51, 0.51]),
    ]

    blend = EnsembleBuilder(EnsembleMethod.WEIGHTED).blend(
        results, task_type=TaskType.CLASSIFICATION
    )

    np.testing.assert_array_equal(blend["oof"], np.array([1, 1]))
    np.testing.assert_allclose(blend["oof_proba"], np.array([0.51, 0.51]))


def test_blending_ignores_results_without_oof_predictions():
    valid = _cv_result("valid", [0, 1, 1], [0.1, 0.8, 0.9])
    missing = CVResult(model_key="missing", model_name="missing", mean_scores={"f1_weighted": 0.9})

    blend = EnsembleBuilder(EnsembleMethod.WEIGHTED).blend(
        [valid, missing], task_type=TaskType.CLASSIFICATION
    )

    np.testing.assert_array_equal(blend["oof"], np.array([0, 1, 1]))
    assert list(blend["weights"]) == ["valid"]
