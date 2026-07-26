"""Fábrica para instanciação de modelos de recomendação."""

from typing import Any

from sklearn.ensemble import RandomForestRegressor

from src.config import settings
from src.models.networks import MLPEmbedding


class ModelFactory:
    """Fábrica para instanciação de modelos preditivos."""

    @staticmethod
    def create_model(model_type: str, **kwargs: Any) -> Any:
        """Cria e retorna o modelo correspondente a ``model_type``.

        Args:
            model_type: Identificador do modelo a criar, ex. "mlp" ou "rf_baseline".
            **kwargs: Parâmetros específicos do modelo. O tipo "mlp" exige
                ``num_users`` e ``num_items``.

        Returns:
            A instância do modelo criado.

        Raises:
            ValueError: Se ``model_type`` não for reconhecido.
        """
        if model_type == "mlp":
            return MLPEmbedding(
                num_users=kwargs["num_users"],
                num_items=kwargs["num_items"],
                emb_dim=settings.mlp_emb_dim,
            )
        if model_type == "rf_baseline":
            # Random Forest ignora num_users/num_items, usa apenas os
            # próprios hiperparâmetros
            return RandomForestRegressor(
                n_estimators=settings.rf_n_estimators,
                random_state=settings.random_seed,
            )

        raise ValueError(f"Modelo {model_type} desconhecido.")
