"""
Incremental Learning (lightweight)

Wraps sklearn models with partial_fit for online updates.
"""
from typing import Any, Dict, Optional

import pandas as pd


PARTIAL_FIT_MODELS = {
    'sgd', 'passiveaggressive', 'mnb', 'bnb', 'cnnb',
    'torch_mlp',  # custom wrapper below
}


def supports_incremental(model_key: str) -> bool:
    return model_key.lower() in PARTIAL_FIT_MODELS


def partial_fit_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    classes: Optional[list] = None
) -> Any:
    """Incrementally train a model with new data."""
    if hasattr(model, 'partial_fit'):
        if classes is not None:
            model.partial_fit(X, y, classes=classes)
        else:
            model.partial_fit(X, y)
        return model

    # Pipeline support
    if hasattr(model, 'named_steps'):
        final = model.named_steps.get('model') or model.named_steps.get('classifier') or model.named_steps.get('regressor')
        if final and hasattr(final, 'partial_fit'):
            # Transform X through pipeline steps except final
            Xt = X
            for name, step in model.named_steps.items():
                if name in ('model', 'classifier', 'regressor'):
                    break
                if hasattr(step, 'transform'):
                    Xt = step.transform(Xt)
            if classes is not None:
                final.partial_fit(Xt, y, classes=classes)
            else:
                final.partial_fit(Xt, y)
            return model

    raise ValueError(f'Model does not support incremental learning')
