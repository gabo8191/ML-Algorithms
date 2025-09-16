"""Objetivo: Centralizar parámetros de configuración del proyecto (rutas, hiperparámetros,
gráficos y logging). Bosquejo: dataclass Config con valores por defecto, post-init para
asegurar directorios, y helpers para obtener configuraciones de modelo y preprocesamiento
(`get_model_config`, `get_preprocessing_config`)."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:

    DATA_PATH: str = "data/Titanic-Dataset.csv"
    RESULTS_PATH: str = "results"
    MODELS_PATH: str = "models"

    DEFAULT_K: int = 5
    K_RANGE: tuple = (1, 21)
    CV_FOLDS: int = 5
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 300
    STYLE: str = "seaborn-v0_8"

    TARGET_COLUMN: str = "Survived"
    FEATURE_COLUMNS: Optional[list] = None

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    TITANIC_CATEGORICAL_COLS: Optional[list] = None
    TITANIC_NUMERICAL_COLS: Optional[list] = None

    def __post_init__(self):
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        os.makedirs(self.MODELS_PATH, exist_ok=True)

        if self.TITANIC_CATEGORICAL_COLS is None:
            self.TITANIC_CATEGORICAL_COLS = [
                "Pclass",
                "Sex",
                "Embarked",
            ]

        if self.TITANIC_NUMERICAL_COLS is None:
            self.TITANIC_NUMERICAL_COLS = [
                "Age",
                "SibSp",
                "Parch",
                "Fare",
            ]

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "n_neighbors": self.DEFAULT_K,
            "weights": "uniform",
            "algorithm": "auto",
            "metric": "euclidean",
        }

    def get_preprocessing_config(self) -> Dict[str, Any]:
        return {
            "test_size": self.TEST_SIZE,
            "random_state": self.RANDOM_STATE,
            "stratify": True,
            "scale_features": True,
        }
