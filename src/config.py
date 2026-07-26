"""Configurações do pipeline, carregadas a partir de variáveis de ambiente/.env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações do pipeline de treino e pré-processamento."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    raw_data_path: str = "data/raw/ratings.csv"
    processed_data_path: str = "data/processed/clean.csv"
    sample_size: int = 1000
    mlp_emb_dim: int = 32
    rf_n_estimators: int = 50
    random_seed: int = 42
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"


settings = Settings()
