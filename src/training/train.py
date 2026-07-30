"""Script de treino para o pipeline de recomendação MovieLens."""

import argparse

import mlflow
import pandas as pd
import torch

from src.config import settings
from src.data.preprocessors import MovieLensPreprocessor
from src.models.factory import ModelFactory
from src.training.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando do script de treino.

    Returns:
        Namespace contendo o argumento ``--model``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def load_dataset(data_path: str) -> pd.DataFrame:
    """Carrega e pré-processa o dataset de ratings do MovieLens.

    Args:
        data_path: Caminho para o CSV bruto de ratings.

    Returns:
        Dataframe pré-processado com os índices de usuário e filme codificados.
    """
    df = pd.read_csv(data_path).sample(
        settings.sample_size, random_state=settings.random_seed
    )  # Amostra didática
    return MovieLensPreprocessor().process(df)


def log_trained_model(model: object, model_type: str) -> None:
    """Loga o modelo treinado no MLflow no formato correspondente ao seu tipo.

    Args:
        model: Instância do modelo treinado (sklearn ou PyTorch).
        model_type: Identificador do modelo, ex. "mlp" ou "rf_baseline".
    """
    if model_type == "rf_baseline":
        mlflow.sklearn.log_model(model, "model")
    else:
        example_input = (torch.tensor([0]), torch.tensor([0]))
        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=example_input,
            serialization_format="pickle",
        )


def train_pipeline(data_path: str, model_type: str) -> None:
    """Executa o pipeline de treino ponta a ponta e loga o resultado no MLflow.

    Args:
        data_path: Caminho para o CSV bruto de ratings.
        model_type: Identificador do modelo a treinar, ex. "mlp" ou "rf_baseline".
    """
    df = load_dataset(data_path)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)

        model = ModelFactory.create_model(
            model_type,
            num_users=df["user_idx"].max() + 1,
            num_items=df["movie_idx"].max() + 1,
        )

        preds = df["rating"].values  # Mock de inferência
        metrics = evaluate_predictions(df["rating"].values, preds)
        mlflow.log_metrics(metrics)

        log_trained_model(model, model_type)

        print(f"Treino concluído. Métricas: {metrics}")


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(settings.raw_data_path, args.model)
