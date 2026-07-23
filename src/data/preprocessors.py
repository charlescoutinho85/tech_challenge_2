"""Estratégias de pré-processamento para datasets de recomendação."""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PreprocessStrategy(ABC):
    """Estratégia abstrata de pré-processamento específica por dataset."""

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforma dados brutos de interação em um dataframe pronto para o modelo.

        Args:
            data: Dataframe bruto de interações.

        Returns:
            O dataframe pré-processado.
        """


class MovieLensPreprocessor(PreprocessStrategy):
    """Estratégia de pré-processamento para o dataset MovieLens."""

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Codifica os IDs de usuário e filme do MovieLens em índices contíguos.

        Args:
            data: Dataframe bruto do MovieLens com as colunas ``userId``/``movieId``.

        Returns:
            Cópia de ``data`` com as colunas ``user_idx`` e ``movie_idx`` adicionadas.
        """
        df = data.copy()
        df["user_idx"] = LabelEncoder().fit_transform(df["userId"])
        df["movie_idx"] = LabelEncoder().fit_transform(df["movieId"])
        return df
