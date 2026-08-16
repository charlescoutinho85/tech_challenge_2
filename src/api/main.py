from fastapi import FastAPI, HTTPException

from src.api.logging_config import setup_logging
from src.api.middleware import log_requests
from src.api.predictor import Predictor
from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
)


# Configuração do logging
setup_logging()


app = FastAPI(
    title="MovieLens Recommendation API",
    description="API para previsão de ratings e recomendações.",
    version="1.0.0",
)


# Middleware para registrar requisições e calcular latência
app.middleware("http")(log_requests)


predictor = Predictor()


@app.on_event("startup")
def load_model() -> None:
    """Carrega o modelo quando a API é iniciada."""
    predictor.load()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Verifica se a API está funcionando."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor.model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Prevê o rating de um usuário para um filme."""
    try:
        prediction = predictor.predict(
            user_id=request.user_id,
            movie_id=request.movie_id,
        )

        return PredictionResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            predicted_rating=prediction,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao realizar a previsão.",
        ) from error


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
) -> RecommendationResponse:
    """Retorna recomendações personalizadas para um usuário."""
    try:
        recommendations = predictor.recommend(
            user_id=request.user_id,
            top_n=request.top_n,
        )

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao gerar recomendações.",
        ) from error