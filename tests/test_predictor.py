"""Testes para o carregamento e geração de previsões da API."""

import pytest
import torch

from src.api.predictor import Predictor
from src.models.networks import MLPEmbedding


@pytest.fixture
def predictor() -> Predictor:
    """Monta um Predictor com um modelo pequeno, sem depender de arquivos em disco."""
    torch.manual_seed(0)
    api_predictor = Predictor()
    api_predictor.model = MLPEmbedding(num_users=3, num_items=4, emb_dim=8)
    api_predictor.model.eval()
    api_predictor.user_mapping = {10: 0, 20: 1, 30: 2}
    api_predictor.movie_mapping = {100: 0, 200: 1, 300: 2, 400: 3}
    api_predictor.movie_ids = list(api_predictor.movie_mapping.keys())
    api_predictor.interactions = {10: {100, 200}}
    return api_predictor


def test_predict_returns_rating_within_bounds(predictor: Predictor) -> None:
    rating = predictor.predict(user_id=10, movie_id=100)
    assert 0.0 <= rating <= 5.0


def test_predict_unknown_user_raises(predictor: Predictor) -> None:
    with pytest.raises(ValueError):
        predictor.predict(user_id=999, movie_id=100)


def test_predict_unknown_movie_raises(predictor: Predictor) -> None:
    with pytest.raises(ValueError):
        predictor.predict(user_id=10, movie_id=999)


def test_recommend_excludes_watched_movies(predictor: Predictor) -> None:
    recommendations = predictor.recommend(user_id=10, top_n=10)
    recommended_ids = {item["movie_id"] for item in recommendations}
    assert recommended_ids == {300, 400}


def test_recommend_respects_top_n(predictor: Predictor) -> None:
    recommendations = predictor.recommend(user_id=20, top_n=2)
    assert len(recommendations) == 2


def test_recommend_matches_individual_predict(predictor: Predictor) -> None:
    """O batching do recommend() deve bater com o predict() individual por filme."""
    batched = predictor.recommend(user_id=30, top_n=10)
    for item in batched:
        individual = predictor.predict(user_id=30, movie_id=item["movie_id"])
        assert item["predicted_rating"] == pytest.approx(individual, abs=1e-5)
