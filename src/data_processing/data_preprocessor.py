"""Objetivo: Preparar datos para clasificación con KNN. Bosquejo: clase DataPreprocessor
que selecciona y transforma características (`select_features_for_classification`), imputa
faltantes (`handle_missing_values`), codifica categóricas (`encode_categorical_features`),
elimina outliers (`remove_outliers`), separa X/y (`prepare_features_target`), escala
características (`scale_features`), divide conjuntos (`split_data`) y orquesta el flujo en
`preprocess_pipeline`. Expone nombres de features y clases."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class DataPreprocessor(LoggerMixin):
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_names: List[str] = []
        self.target_name: str = ""

    # Eliminado: ingeniería de características específica de Titanic no requerida

    def select_features_for_classification(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True, dtype=int)

        if "Survived" not in df.columns:
            raise ValueError("Columna 'Survived' no encontrada en el dataset")

        result_df = df.copy()

        feature_columns = [col for col in df.columns if col != "Survived"]

        self.logger.info(f"Características seleccionadas: {feature_columns}")
        self.logger.info(f"Variable objetivo: Survived")
        self.logger.info(f"Total de características: {len(feature_columns)}")

        return result_df

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "median"
    ) -> pd.DataFrame:

        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if "Survived" in categorical_cols:
            categorical_cols.remove("Survived")

        if numeric_cols:
            if "numeric" not in self.imputers:
                self.imputers["numeric"] = SimpleImputer(strategy=strategy)
                df[numeric_cols] = self.imputers["numeric"].fit_transform(
                    df[numeric_cols]
                )
            else:
                df[numeric_cols] = self.imputers["numeric"].transform(df[numeric_cols])

        if categorical_cols:
            if "categorical" not in self.imputers:
                self.imputers["categorical"] = SimpleImputer(strategy="most_frequent")
                df[categorical_cols] = self.imputers["categorical"].fit_transform(
                    df[categorical_cols]
                )
            else:
                df[categorical_cols] = self.imputers["categorical"].transform(
                    df[categorical_cols]
                )

        if "Survived" in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=["Survived"])
            final_len = len(df)

            if initial_len != final_len:
                self.logger.warning(
                    f"Removidas {initial_len - final_len} filas con objetivo faltante"
                )

        self.logger.info("Valores faltantes manejados exitosamente")
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        if "Survived" in categorical_cols:
            categorical_cols.remove("Survived")

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))

        if categorical_cols:
            self.logger.info(
                f"Características categóricas codificadas: {categorical_cols}"
            )

        return df

    def remove_outliers(
        self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
    ) -> pd.DataFrame:

        df = df.copy()
        initial_len = len(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if "Survived" in df.columns:
            target_col = df["Survived"].copy()
        else:
            target_col = None

        if method == "iqr":
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        elif method == "zscore":
            self.logger.info("Usando método IQR para remoción de outliers")
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        if target_col is not None:
            df["Survived"] = target_col.reindex(df.index)

        final_len = len(df)
        removed_count = initial_len - final_len

        if removed_count > 0:
            self.logger.info(
                f"Removidos {removed_count} outliers ({removed_count/initial_len*100:.1f}%)"
            )

        return df

    def prepare_features_target(
        self, df: pd.DataFrame, target_column: str = "Survived"
    ) -> Tuple[pd.DataFrame, pd.Series]:

        if target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{target_column}' no encontrada")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X = X.astype("float64")

        if y.dtype == "object":
            if "target" not in self.label_encoders:
                self.label_encoders["target"] = LabelEncoder()
                encoded_y = self.label_encoders["target"].fit_transform(y)
                y = pd.Series(data=np.array(encoded_y), index=y.index, dtype="int64")
            else:
                encoded_y = self.label_encoders["target"].transform(y)
                y = pd.Series(data=np.array(encoded_y), index=y.index, dtype="int64")
        else:
            y = y.astype("int64")

        self.feature_names = list(X.columns)
        self.target_name = target_column

        self.logger.info(f"Características preparadas: {len(X.columns)} features")
        self.logger.info(f"Clases objetivo: {len(np.unique(y))}")

        return X, y

    def scale_features(
        self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)

        self.logger.info("Características escaladas exitosamente")

        X_train_scaled = np.asarray(X_train_scaled)
        if X_test_scaled is not None:
            X_test_scaled = np.asarray(X_test_scaled)

        return X_train_scaled, X_test_scaled

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        if test_size is None:
            test_size = self.config.TEST_SIZE
        if random_state is None:
            random_state = self.config.RANDOM_STATE

        class_counts = pd.Series(y).value_counts()
        min_class_count = class_counts.min()

        if min_class_count < 2:
            self.logger.warning(f"Clase con muy pocas muestras: {min_class_count}")
            valid_classes = class_counts[class_counts >= 2].index
            mask = pd.Series(y).isin(valid_classes)
            X = X[mask]
            y = y[mask]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            self.logger.warning("Stratification falló, dividiendo sin estratificar")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        self.logger.info(
            f"Datos divididos - Train: {X_train.shape}, Test: {X_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    def preprocess_pipeline(
        self, df: pd.DataFrame, target_column: str = "Survived"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.logger.info("Iniciando pipeline de preprocesamiento")

        df_selected = self.select_features_for_classification(df)

        df_clean = self.handle_missing_values(df_selected)

        df_encoded = df_clean

        X, y = self.prepare_features_target(df_encoded, target_column)

        X_train, X_test, y_train, y_test = self.split_data(X, y)

        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        self.logger.info("Pipeline de preprocesamiento completado")

        if X_test_scaled is None:
            raise ValueError("Error en escalado de datos de prueba")

        return (
            X_train_scaled,
            X_test_scaled,
            np.array(y_train.values),
            np.array(y_test.values),
        )

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_target_classes(self) -> Optional[List[str]]:
        if "target" in self.label_encoders:
            return list(self.label_encoders["target"].classes_)
        return None
