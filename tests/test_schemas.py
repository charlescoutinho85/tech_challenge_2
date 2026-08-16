"""Testes para a validação dos schemas Pydantic da API."""

import pytest
from pydantic import ValidationError

from src.api.schemas import RecommendationRequest


def test_recommendation_request_accepts_valid_top_n() -> None:
    request = RecommendationRequest(user_id=1, top_n=10)
    assert request.top_n == 10


def test_recommendation_request_default_top_n() -> None:
    request = RecommendationRequest(user_id=1)
    assert request.top_n == 10


def test_recommendation_request_rejects_top_n_above_limit() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(user_id=1, top_n=51)


def test_recommendation_request_rejects_top_n_below_limit() -> None:
    with pytest.raises(ValidationError):
        RecommendationRequest(user_id=1, top_n=0)
