import argparse
import mlflow
import pandas as pd
import torch

from src.data.preprocessors import MovieLensPreprocessor
from src.evaluation.metrics import evaluate_predictions
from src.models.factory import ModelFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def train_pipeline(data_path: str, model_type: str) -> None:
    df = pd.read_csv(data_path).sample(1000)  # Amostra didática
    df = MovieLensPreprocessor().process(df)
    
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        
        model = ModelFactory.create_model(
            model_type, 
            num_users=df['user_idx'].max() + 1, 
            num_items=df['movie_idx'].max() + 1
        )
        
        preds = df['rating'].values  # Mock de inferência
        metrics = evaluate_predictions(df['rating'].values, preds)
        
        mlflow.log_metrics(metrics)
        
        if model_type == "rf_baseline":
            mlflow.sklearn.log_model(model, "model")
        else:
            example_input = (torch.tensor([0]), torch.tensor([0]))
            mlflow.pytorch.log_model(
                model, 
                "model", 
                input_example=example_input,
                serialization_format="pickle"
            )
            
        print(f"Treino concluído. Métricas: {metrics}")


if __name__ == "__main__":
    args = parse_args()
    train_pipeline("data/raw/ratings.csv", args.model)