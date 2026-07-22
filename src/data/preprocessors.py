from abc import ABC, abstractmethod

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PreprocessStrategy(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class MovieLensPreprocessor(PreprocessStrategy):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['user_idx'] = LabelEncoder().fit_transform(df['userId'])
        df['movie_idx'] = LabelEncoder().fit_transform(df['movieId'])
        return df