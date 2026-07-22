from sklearn.ensemble import RandomForestRegressor
from src.models.networks import MLPEmbedding

class ModelFactory:
    """Fábrica para instanciação de modelos preditivos."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> any:
        """Cria e retorna o modelo especificado com os parâmetros corretos."""
        if model_type == "mlp":
            return MLPEmbedding(
                num_users=kwargs["num_users"], 
                num_items=kwargs["num_items"]
            )
        if model_type == "rf_baseline":
            # Random Forest ignora num_users/num_items, usa apenas os próprios hiperparâmetros
            return RandomForestRegressor(n_estimators=50, random_state=42)
        
        raise ValueError(f"Modelo {model_type} desconhecido.")