"""Testes para as estratégias de pré-processamento (Strategy pattern)."""

import pandas as pd

from src.data.preprocessors import MovieLensPreprocessor


def test_process_adds_contiguous_indices() -> None:
    data = pd.DataFrame(
        {
            "userId": [5, 5, 9],
            "movieId": [100, 200, 100],
            "rating": [4.0, 3.5, 5.0],
        }
    )

    processed = MovieLensPreprocessor().process(data)

    assert set(processed["user_idx"]) == {0, 1}
    assert set(processed["movie_idx"]) == {0, 1}
    assert processed.loc[processed["userId"] == 5, "user_idx"].nunique() == 1


def test_process_does_not_mutate_original() -> None:
    data = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [3.0]})

    MovieLensPreprocessor().process(data)

    assert "user_idx" not in data.columns
