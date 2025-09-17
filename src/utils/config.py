"""Configuración centralizada para el análisis de Machine Learning de cafeterías.

Este módulo define todos los parámetros de configuración necesarios para:
- Carga y preprocesamiento de datos de cafeterías
- Entrenamiento y evaluación de algoritmos ML
- Generación de visualizaciones y reportes
- Configuración de logging y directorios
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class Config:
    """Configuración centralizada para el proyecto de análisis de cafeterías."""

    # ================================================================
    # RUTAS Y DIRECTORIOS
    # ================================================================
    DATA_PATH: str = "data/coffee_shop_revenue.csv"
    RESULTS_PATH: str = "results"
    MODELS_PATH: str = "models"

    # ================================================================
    # PARÁMETROS DE MACHINE LEARNING
    # ================================================================
    CV_FOLDS: int = 5
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    SCORING_METRIC: str = "accuracy"
    N_JOBS: int = -1  # Usar todos los cores disponibles
    N_ITER_SEARCH: int = 50  # Para RandomizedSearchCV

    # ================================================================
    # CONFIGURACIÓN DEL DATASET DE CAFETERÍAS
    # ================================================================
    TARGET_COLUMN: str = "Successful"
    COFFEE_FEATURES: Optional[List[str]] = None

    # Umbral de éxito configurable
    SUCCESS_THRESHOLD_MODE: str = "fixed"  # "quantile" | "fixed"
    SUCCESS_THRESHOLD_VALUE: float = 2000.0  # Valor fijo de $2000

    # ================================================================
    # ALGORITMOS A COMPARAR
    # ================================================================
    ALGORITHMS_TO_COMPARE: Optional[List[str]] = None

    # ================================================================
    # CONFIGURACIÓN DE VISUALIZACIÓN
    # ================================================================
    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 300
    STYLE: str = "seaborn-v0_8"

    # ================================================================
    # CONFIGURACIÓN DE LOGGING
    # ================================================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        """Inicialización post-creación de la configuración."""
        # Crear directorios principales
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        os.makedirs(self.MODELS_PATH, exist_ok=True)

        # Configurar algoritmos a comparar
        if self.ALGORITHMS_TO_COMPARE is None:
            self.ALGORITHMS_TO_COMPARE = [
                "LogisticRegression",
                "SVM",
                "DecisionTree",
                "RandomForest",
                "NeuralNetwork",
            ]

        # Crear subdirectorios para resultados de cada algoritmo
        for algorithm in self.ALGORITHMS_TO_COMPARE:
            os.makedirs(
                os.path.join(self.RESULTS_PATH, algorithm.lower()), exist_ok=True
            )

        # Configurar características del dataset de cafeterías
        if self.COFFEE_FEATURES is None:
            self.COFFEE_FEATURES = [
                "Number_of_Customers_Per_Day",
                "Average_Order_Value",
                "Operating_Hours_Per_Day",
                "Number_of_Employees",
                "Marketing_Spend_Per_Day",
                "Location_Foot_Traffic",
            ]

    # ================================================================
    # MÉTODOS DE CONFIGURACIÓN
    # ================================================================

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Obtener configuración para preprocesamiento de datos."""
        return {
            "test_size": self.TEST_SIZE,
            "random_state": self.RANDOM_STATE,
            "stratify": True,
            "scale_features": True,
        }

    def get_hyperparameter_config(self) -> Dict[str, Any]:
        """Obtener configuración para optimización de hiperparámetros."""
        return {
            "cv_folds": self.CV_FOLDS,
            "scoring": self.SCORING_METRIC,
            "n_jobs": self.N_JOBS,
            "n_iter_search": self.N_ITER_SEARCH,
            "random_state": self.RANDOM_STATE,
        }

    def get_visualization_config(self) -> Dict[str, Any]:
        """Obtener configuración para visualizaciones."""
        return {
            "figure_size": self.FIGURE_SIZE,
            "dpi": self.DPI,
            "style": self.STYLE,
        }

    def get_success_threshold_info(self) -> Dict[str, Any]:
        """Obtener información sobre el umbral de éxito configurado."""
        return {
            "mode": self.SUCCESS_THRESHOLD_MODE,
            "value": self.SUCCESS_THRESHOLD_VALUE,
            "description": (
                f"Percentil {int(self.SUCCESS_THRESHOLD_VALUE * 100)}"
                if self.SUCCESS_THRESHOLD_MODE == "quantile"
                else f"Valor fijo ${self.SUCCESS_THRESHOLD_VALUE:,.0f}"
            ),
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """Obtener información del dataset de cafeterías."""
        return {
            "data_path": self.DATA_PATH,
            "target_column": self.TARGET_COLUMN,
            "features": self.COFFEE_FEATURES,
            "success_threshold": self.get_success_threshold_info(),
        }

    def get_algorithms_info(self) -> Dict[str, Any]:
        """Obtener información de los algoritmos a comparar."""
        algorithms = self.ALGORITHMS_TO_COMPARE or []
        return {
            "algorithms": algorithms,
            "count": len(algorithms),
            "description": "Algoritmos de Machine Learning para clasificación binaria",
        }
