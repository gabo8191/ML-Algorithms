"""Objetivo: Cargar datasets (por ejemplo, Titanic) desde archivos CSV, validar su integridad
y proporcionar resúmenes. Bosquejo: clase DataLoader con métodos para leer CSV (`load_csv`),
cargar conjuntos específicos (`load_titanic_data`, `load_airline_data`), validar estructura y
calidad de datos (`validate_data`), mostrar un resumen (`get_data_summary`) y obtener muestras
(`sample_data`). Usa `Config` para rutas y `LoggerMixin` para trazabilidad."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import warnings

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class DataLoader(LoggerMixin):
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.df: Optional[pd.DataFrame] = None

    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"El archivo {file_path} no existe")

        try:
            self.log_step(
                "CARGA_CSV",
                f"Cargar archivo CSV desde {file_path}",
                encoding=kwargs.get("encoding", "utf-8"),
                low_memory=kwargs.get("low_memory", False),
            )

            default_kwargs = {"low_memory": False, "encoding": "utf-8"}
            default_kwargs.update(kwargs)

            df = pd.read_csv(file_path, **default_kwargs)
            self.df = df

            file_size_mb = path.stat().st_size / (1024**2)
            memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)

            self.log_data_info(
                "Dataset cargado",
                df.shape,
                {
                    "Tamaño archivo": f"{file_size_mb:.1f} MB",
                    "Uso memoria": f"{memory_usage_mb:.1f} MB",
                    "Columnas": list(df.columns),
                    "Tipos de datos": df.dtypes.value_counts().to_dict(),
                },
            )

            return df

        except Exception as e:
            self.log_error_with_context(
                f"No se pudo cargar el archivo: {str(e)}",
                f"Intentando cargar: {file_path}",
                "Verificar que el archivo exista y tenga formato CSV válido",
            )
            raise ValueError(f"No se pudo cargar el archivo: {str(e)}")

    def load_titanic_data(self, file_path: Optional[str] = None) -> pd.DataFrame:

        if file_path is None:
            file_path = self.config.DATA_PATH

        self.df = self.load_csv(
            file_path,
            dtype={
                "PassengerId": "int32",
                "Survived": "int8",
                "Pclass": "int8",
                "Age": "float32",
                "SibSp": "int8",
                "Parch": "int8",
                "Fare": "float32",
            },
        )

        return self.df

    def load_airline_data(self, file_path: Optional[str] = None) -> pd.DataFrame:

        return self.load_titanic_data(file_path)

    def validate_data(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:

        if df is None:
            df = self.df

        if df is None:
            raise ValueError("No hay datos para validar")

        validation_report = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicated_rows": df.duplicated().sum(),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=["object"]).columns),
        }

        numeric_stats = {}
        for col in validation_report["numeric_columns"]:
            numeric_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "zeros": (df[col] == 0).sum(),
                "negative": (df[col] < 0).sum(),
            }
        validation_report["numeric_stats"] = numeric_stats

        categorical_stats = {}
        for col in validation_report["categorical_columns"]:
            categorical_stats[col] = {
                "unique_count": df[col].nunique(),
                "most_frequent": (
                    df[col].mode().iloc[0] if not df[col].mode().empty else None
                ),
                "most_frequent_count": (
                    df[col].value_counts().iloc[0]
                    if len(df[col].value_counts()) > 0
                    else 0
                ),
            }
        validation_report["categorical_stats"] = categorical_stats

        self.logger.info("Validación de datos completada")
        return validation_report

    def get_data_summary(self, df: Optional[pd.DataFrame] = None) -> None:

        if df is None:
            df = self.df

        if df is None:
            self.logger.warning("No hay datos para resumir")
            return

        print("=" * 60)
        print("RESUMEN DE DATOS")
        print("=" * 60)

        print(f"Forma del dataset: {df.shape}")
        print(f"Uso de memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Filas duplicadas: {df.duplicated().sum()}")

        print("\nTipos de datos:")
        print(df.dtypes.value_counts())

        print("\nValores nulos por columna:")
        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df) * 100).round(2)

        for col in df.columns:
            if null_counts[col] > 0:
                print(f"  {col}: {null_counts[col]} ({null_percentages[col]}%)")

        if null_counts.sum() == 0:
            print("  No hay valores nulos")

        print("\nColumnas numéricas:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"  {list(numeric_cols)}")

        print("\nColumnas categóricas:")
        categorical_cols = df.select_dtypes(include=["object"]).columns
        print(f"  {list(categorical_cols)}")

        print("\nEstadísticas descriptivas (primeras 5 columnas numéricas):")
        if len(numeric_cols) > 0:
            print(df[numeric_cols[:5]].describe())

        print("=" * 60)

    def sample_data(self, n: int = 1000, random_state: int = 42) -> pd.DataFrame:

        if self.df is None:
            raise ValueError("No hay datos cargados")

        if n >= len(self.df):
            self.logger.warning(
                f"Tamaño de muestra ({n}) >= tamaño del dataset ({len(self.df)})"
            )
            return self.df.copy()

        sample = self.df.sample(n=n, random_state=random_state)
        self.logger.info(f"Muestra de {n} filas creada")

        return sample
