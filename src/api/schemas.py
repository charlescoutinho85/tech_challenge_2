from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionRequest(BaseModel):
    user_id: int
    movie_id: int


class PredictionResponse(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float


class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = Field(default=10, ge=1, le=50)


class RecommendationItem(BaseModel):
    movie_id: int
    predicted_rating: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]
