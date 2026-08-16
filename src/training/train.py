"""Script de treino para o pipeline de recomendação MovieLens."""

import argparse
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from src.config import settings
from src.models.factory import ModelFactory
from src.tracking.mlflow_tracker import (
    configure_tracking,
    log_trained_model,
    promote_to_production,
)
from src.training.metrics import evaluate_predictions

FEATURE_COLUMNS = ["user_idx", "movie_idx"]
TARGET_COLUMN = "rating"
Tensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando do script de treino.

    Returns:
        Namespace contendo o argumento ``--model``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def load_dataset(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Carrega o dataset pré-processado e sorteia a amostra didática de treino.

    As dimensões dos embeddings são calculadas a partir do dataset completo (antes
    da amostragem), para que o modelo suporte qualquer usuário/filme do catálogo,
    independente de quais linhas caírem na amostra sorteada.

    Args:
        data_path: Caminho para o CSV pré-processado (saída do stage `preprocess`).

    Returns:
        Tupla (amostra sorteada, dataset completo, número total de usuários,
        número total de filmes).
    """
    df = pd.read_csv(data_path)
    num_users = int(df["user_idx"].max()) + 1
    num_items = int(df["movie_idx"].max()) + 1
    sample = df.sample(settings.sample_size, random_state=settings.random_seed)
    return sample, df, num_users, num_items


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o dataset amostrado em treino e teste.

    Args:
        df: Amostra pré-processada.

    Returns:
        Tupla (treino, teste).
    """
    return train_test_split(
        df, test_size=settings.test_size, random_state=settings.random_seed
    )


def _to_tensors(df: pd.DataFrame) -> Tensors:
    """Converte as colunas de usuário, filme e rating em tensores PyTorch.

    Args:
        df: Dataframe com as colunas ``user_idx``, ``movie_idx`` e ``rating``.

    Returns:
        Tupla (usuários, filmes, ratings) como tensores.
    """
    users = torch.tensor(df["user_idx"].values, dtype=torch.long)
    items = torch.tensor(df["movie_idx"].values, dtype=torch.long)
    ratings = torch.tensor(df[TARGET_COLUMN].values, dtype=torch.float32)
    return users, items, ratings


def _train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_tensors: Tensors,
    val_tensors: Tensors,
) -> float:
    """Executa uma época de treino e retorna a perda no conjunto de validação.

    Args:
        model: Modelo em treinamento.
        optimizer: Otimizador já associado aos parâmetros do modelo.
        loss_fn: Função de perda.
        train_tensors: Tupla (usuários, filmes, ratings) de treino.
        val_tensors: Tupla (usuários, filmes, ratings) de validação.

    Returns:
        Valor da perda de validação nesta época.
    """
    user_tr, item_tr, rating_tr = train_tensors
    user_val, item_val, rating_val = val_tensors

    model.train()
    optimizer.zero_grad()
    loss_fn(model(user_tr, item_tr), rating_tr).backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        return loss_fn(model(user_val, item_val), rating_val).item()


def train_mlp(
    model: nn.Module, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> nn.Module:
    """Treina o MLP com early stopping baseado na perda no conjunto de teste.

    Args:
        model: Instância não treinada do `MLPEmbedding`.
        train_df: Dados de treino.
        test_df: Dados de teste, usados para monitorar o early stopping.

    Returns:
        O modelo treinado.
    """
    train_tensors = _to_tensors(train_df)
    val_tensors = _to_tensors(test_df)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.mlp_lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(settings.mlp_epochs):
        val_loss = _train_one_epoch(
            model, optimizer, loss_fn, train_tensors, val_tensors
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.early_stopping_patience:
                break

    return model


def fit_model(
    model: object, model_type: str, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> object:
    """Treina o modelo com o procedimento adequado ao seu tipo.

    Args:
        model: Instância não treinada retornada pela `ModelFactory`.
        model_type: Identificador do modelo, ex. "mlp" ou "rf_baseline".
        train_df: Dados de treino.
        test_df: Dados de teste (usados só pelo MLP, para early stopping).

    Returns:
        O modelo treinado.
    """
    if model_type == "rf_baseline":
        model.fit(train_df[FEATURE_COLUMNS].values, train_df[TARGET_COLUMN].values)
        return model
    return train_mlp(model, train_df, test_df)


def predict(model: object, model_type: str, df: pd.DataFrame) -> np.ndarray:
    """Gera previsões de rating para o dataframe informado.

    Args:
        model: Modelo treinado.
        model_type: Identificador do modelo, ex. "mlp" ou "rf_baseline".
        df: Dados de entrada.

    Returns:
        Array de ratings previstos.
    """
    if model_type == "rf_baseline":
        return model.predict(df[FEATURE_COLUMNS].values)
    model.eval()
    with torch.no_grad():
        users, items, _ = _to_tensors(df)
        return model(users, items).numpy()


def save_model_for_api(
    model: nn.Module,
    num_users: int,
    num_items: int,
    data: pd.DataFrame,
) -> None:
    """Salva o modelo e os mapeamentos de IDs necessários para a API.

    Args:
        model: Modelo MLP treinado.
        num_users: Número total de usuários no catálogo.
        num_items: Número total de filmes no catálogo.
        data: Dataset pré-processado completo (já contém `user_idx`/`movie_idx`),
            usado para extrair os mapeamentos únicos de usuário e filme sem
            reprocessar o dataset bruto novamente.
    """
    models_dir = Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
        },
        models_dir / "model.pt",
    )

    data[["userId", "user_idx"]].drop_duplicates().to_csv(
        models_dir / "user_mappings.csv", index=False
    )
    data[["movieId", "movie_idx"]].drop_duplicates().to_csv(
        models_dir / "movie_mappings.csv", index=False
    )

    _save_interactions(data, models_dir)


def _save_interactions(data: pd.DataFrame, models_dir: Path) -> None:
    """Pré-computa e persiste os filmes já avaliados por cada usuário.

    Faz esse agrupamento uma única vez em tempo de treino, para que a API não
    precise reler o dataset bruto inteiro e reagrupá-lo a cada startup.

    Args:
        data: Dataset pré-processado completo.
        models_dir: Diretório onde os artefatos da API são salvos.
    """
    interactions = {
        int(user_id): set(group["movieId"].astype(int))
        for user_id, group in data.groupby("userId")
    }
    with open(models_dir / "interactions.pkl", "wb") as f:
        pickle.dump(interactions, f)


def train_pipeline(data_path: str, model_type: str) -> None:
    """Executa o pipeline de treino ponta a ponta e loga o resultado no MLflow.

    Args:
        data_path: Caminho para o CSV pré-processado de ratings.
        model_type: Identificador do modelo a treinar, ex. "mlp" ou "rf_baseline".
    """
    sample, full_df, num_users, num_items = load_dataset(data_path)
    train_df, test_df = split_dataset(sample)

    configure_tracking()

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("sample_size", settings.sample_size)
        mlflow.log_param("test_size", settings.test_size)

        torch.manual_seed(settings.random_seed)
        model = ModelFactory.create_model(
            model_type, num_users=num_users, num_items=num_items
        )
        model = fit_model(model, model_type, train_df, test_df)
        if model_type == "mlp":
            save_model_for_api(
                model=model,
                num_users=num_users,
                num_items=num_items,
                data=full_df,
            )

        preds = predict(model, model_type, test_df)
        metrics = evaluate_predictions(test_df[TARGET_COLUMN].values, preds)
        mlflow.log_metrics(metrics)

        model_uri = log_trained_model(model, model_type)
        if model_type == "mlp":
            promote_to_production(model_uri)

        print(f"Treino concluído. Métricas: {metrics}")


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(settings.processed_data_path, args.model)
