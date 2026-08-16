import pickle
from pathlib import Path

import pandas as pd
import torch

from src.config import settings
from src.models.networks import MLPEmbedding


class Predictor:
    """Carrega o modelo e realiza previsões e recomendações."""

    def __init__(self) -> None:
        self.model = None
        self.user_mapping: dict[int, int] = {}
        self.movie_mapping: dict[int, int] = {}
        self.movie_ids: list[int] = []
        self.interactions: dict[int, set[int]] = {}

    def load(self) -> None:
        """Carrega modelo, mapeamentos e histórico de interações."""
        models_dir = Path(settings.models_dir)
        model_path = models_dir / "model.pt"
        user_mapping_path = models_dir / "user_mappings.csv"
        movie_mapping_path = models_dir / "movie_mappings.csv"
        interactions_path = models_dir / "interactions.pkl"

        self._validate_files(
            model_path, user_mapping_path, movie_mapping_path, interactions_path
        )

        self._load_mappings(user_mapping_path, movie_mapping_path)
        self._load_model(model_path)
        self._load_interactions(interactions_path)

    @staticmethod
    def _validate_files(*paths: Path) -> None:
        """Valida a existência dos arquivos necessários."""
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado em: {path}")

    def _load_mappings(self, user_mapping_path: Path, movie_mapping_path: Path) -> None:
        """Carrega os mapeamentos dos IDs de usuário e de filme."""
        users = pd.read_csv(user_mapping_path)
        movies = pd.read_csv(movie_mapping_path)

        self.user_mapping = dict(zip(users["userId"], users["user_idx"]))
        self.movie_mapping = dict(zip(movies["movieId"], movies["movie_idx"]))
        self.movie_ids = list(self.movie_mapping.keys())

    def _load_model(self, model_path: Path) -> None:
        """Carrega os pesos do modelo PyTorch."""
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )

        self.model = MLPEmbedding(
            num_users=checkpoint["num_users"],
            num_items=checkpoint["num_items"],
            emb_dim=settings.mlp_emb_dim,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _load_interactions(self, interactions_path: Path) -> None:
        """Carrega os filmes já avaliados por cada usuário (pré-computado no treino)."""
        with open(interactions_path, "rb") as f:
            self.interactions = pickle.load(f)

    def predict(
        self,
        user_id: int,
        movie_id: int,
    ) -> float:
        """Prevê o rating de um usuário para um filme."""
        self._validate_user(user_id)
        self._validate_movie(movie_id)

        user_idx = self.user_mapping[user_id]
        movie_idx = self.movie_mapping[movie_id]

        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        movie_tensor = torch.tensor([movie_idx], dtype=torch.long)

        with torch.no_grad():
            prediction = self.model(user_tensor, movie_tensor)

        return max(0.0, min(5.0, float(prediction.item())))

    def recommend(
        self,
        user_id: int,
        top_n: int,
    ) -> list[dict[str, float | int]]:
        """Retorna os filmes com maior rating previsto."""
        self._validate_user(user_id)

        watched = self.interactions.get(user_id, set())
        candidates = [
            movie_id for movie_id in self.movie_ids if movie_id not in watched
        ]
        candidate_idx = [self.movie_mapping[movie_id] for movie_id in candidates]

        ratings = self._predict_batch(self.user_mapping[user_id], candidate_idx)
        predictions = [
            {"movie_id": movie_id, "predicted_rating": rating}
            for movie_id, rating in zip(candidates, ratings)
        ]

        predictions.sort(key=lambda item: item["predicted_rating"], reverse=True)
        return predictions[:top_n]

    def _predict_batch(self, user_idx: int, movie_indices: list[int]) -> list[float]:
        """Roda o forward pass do modelo em lote para vários filmes de uma vez."""
        user_tensor = torch.full((len(movie_indices),), user_idx, dtype=torch.long)
        movie_tensor = torch.tensor(movie_indices, dtype=torch.long)

        with torch.no_grad():
            predictions = self.model(user_tensor, movie_tensor)

        return predictions.clamp(0.0, 5.0).tolist()

    def _validate_user(self, user_id: int) -> None:
        """Valida se o usuário existe no modelo."""
        if self.model is None:
            raise RuntimeError("Modelo ainda não foi carregado.")

        if user_id not in self.user_mapping:
            raise ValueError(f"Usuário {user_id} não encontrado no modelo.")

    def _validate_movie(self, movie_id: int) -> None:
        """Valida se o filme existe no modelo."""
        if movie_id not in self.movie_mapping:
            raise ValueError(f"Filme {movie_id} não encontrado no modelo.")
