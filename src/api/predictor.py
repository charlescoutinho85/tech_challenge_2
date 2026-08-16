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
        model_path = Path("models/model.pt")
        mapping_path = Path("models/mappings.csv")

        self._validate_files(model_path, mapping_path)

        mappings = pd.read_csv(mapping_path)
        self._load_mappings(mappings)
        self._load_model(model_path)
        self._load_interactions()

    @staticmethod
    def _validate_files(
        model_path: Path,
        mapping_path: Path,
    ) -> None:
        """Valida a existência dos arquivos necessários."""
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Mapeamentos não encontrados em: {mapping_path}"
            )

    def _load_mappings(self, mappings: pd.DataFrame) -> None:
        """Carrega os mapeamentos dos IDs."""
        self.user_mapping = dict(
            zip(mappings["userId"], mappings["user_idx"])
        )
        self.movie_mapping = dict(
            zip(mappings["movieId"], mappings["movie_idx"])
        )
        self.movie_ids = list(self.movie_mapping.keys())

    def _load_model(self, model_path: Path) -> None:
        """Carrega os pesos do modelo PyTorch."""
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        self.model = MLPEmbedding(
            num_users=checkpoint["num_users"],
            num_items=checkpoint["num_items"],
            emb_dim=settings.mlp_emb_dim,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _load_interactions(self) -> None:
        """Carrega os filmes já avaliados pelos usuários."""
        data = pd.read_csv(settings.raw_data_path)

        for user_id, group in data.groupby("userId"):
            self.interactions[int(user_id)] = set(
                group["movieId"].astype(int)
            )

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
            movie_id
            for movie_id in self.movie_ids
            if movie_id not in watched
        ]

        predictions = [
            {
                "movie_id": movie_id,
                "predicted_rating": self.predict(user_id, movie_id),
            }
            for movie_id in candidates
        ]

        predictions.sort(
            key=lambda item: item["predicted_rating"],
            reverse=True,
        )

        return predictions[:top_n]

    def _validate_user(self, user_id: int) -> None:
        """Valida se o usuário existe no modelo."""
        if self.model is None:
            raise RuntimeError("Modelo ainda não foi carregado.")

        if user_id not in self.user_mapping:
            raise ValueError(
                f"Usuário {user_id} não encontrado no modelo."
            )

    def _validate_movie(self, movie_id: int) -> None:
        """Valida se o filme existe no modelo."""
        if movie_id not in self.movie_mapping:
            raise ValueError(
                f"Filme {movie_id} não encontrado no modelo."
            )