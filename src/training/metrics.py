"""Métricas de avaliação para modelos de previsão de rating."""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de regressão comparando ratings previstos e reais.

    Args:
        y_true: Valores reais de rating.
        y_pred: Valores de rating previstos pelo modelo.

    Returns:
        Dicionário com os escores ``rmse``, ``mae``, ``r2`` e ``medae``.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
    }
