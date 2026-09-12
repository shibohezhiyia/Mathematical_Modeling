import numpy as np

from core.surrogate_guard import SurrogateModel


def test_surrogate_falls_back_outside_training_domain():
    model = SurrogateModel.fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
    result = model.predict([[3.0]])
    assert result["status"] == "out_of_domain"
    fallback = model.predict_or_fallback([[3.0]], lambda x: 2.0 * x[:, 0] + 1.0)
    assert fallback["status"] == "full_fidelity_fallback"
    assert fallback["predictions"] == [7.0]


def test_surrogate_in_domain_and_rejects_nonfinite_training():
    model = SurrogateModel.fit([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [1.0, 2.0, 3.0])
    result = model.predict([[1.0, 1.0]])
    assert result["status"] == "in_domain"
    assert np.isfinite(result["predictions"]).all()
