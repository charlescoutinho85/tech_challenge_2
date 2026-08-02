"""Integração com o MLflow: tracking de experimentos e Model Registry."""

import mlflow
import torch
from mlflow.tracking import MlflowClient

from src.config import settings


def configure_tracking() -> None:
    """Aponta o MLflow para o tracking URI configurado em `settings`."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def log_trained_model(model: object, model_type: str) -> str:
    """Loga o modelo treinado no MLflow no formato correspondente ao seu tipo.

    Args:
        model: Instância do modelo treinado (sklearn ou PyTorch).
        model_type: Identificador do modelo, ex. "mlp" ou "rf_baseline".

    Returns:
        URI do modelo logado no run atual.
    """
    artifact_path = "model"
    if model_type == "rf_baseline":
        mlflow.sklearn.log_model(model, name=artifact_path)
    else:
        example_input = (torch.tensor([0]), torch.tensor([0]))
        mlflow.pytorch.log_model(
            model,
            name=artifact_path,
            input_example=example_input,
            serialization_format="pickle",
        )
    return f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"


def promote_to_production(model_uri: str) -> None:
    """Registra o modelo no Model Registry e promove a versão para Production.

    Args:
        model_uri: URI do modelo logado no run atual (``runs:/<run_id>/model``).
    """
    client = MlflowClient()
    version = mlflow.register_model(model_uri, settings.registered_model_name)
    client.transition_model_version_stage(
        name=settings.registered_model_name,
        version=version.version,
        stage="Staging",
    )
    client.transition_model_version_stage(
        name=settings.registered_model_name,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
    )
