"""Testes para a fábrica de modelos (Factory pattern)."""

import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models.factory import ModelFactory
from src.models.networks import MLPEmbedding


def test_create_mlp_model() -> None:
    model = ModelFactory.create_model("mlp", num_users=10, num_items=20)
    assert isinstance(model, MLPEmbedding)


def test_create_rf_baseline_model() -> None:
    model = ModelFactory.create_model("rf_baseline")
    assert isinstance(model, RandomForestRegressor)


def test_create_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        ModelFactory.create_model("unknown")
