"""Testes para o cálculo de métricas de avaliação."""

import numpy as np
import pytest

from src.training.metrics import evaluate_predictions


def test_evaluate_predictions_returns_expected_keys() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = evaluate_predictions(y_true, y_pred)

    assert set(metrics) == {"rmse", "mae", "r2", "medae"}
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["medae"] == 0.0


def test_evaluate_predictions_with_errors() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["rmse"] > 0.0
    assert metrics["mae"] == pytest.approx(2 / 3)
