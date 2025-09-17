"""Clase base para todos los clasificadores de ML.
Define la interfaz común y funcionalidades compartidas entre algoritmos."""

import numpy as np
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class BaseClassifier(LoggerMixin, BaseEstimator, ClassifierMixin, ABC):
    """Clase base abstracta para todos los clasificadores"""

    _estimator_type = "classifier"

    def __init__(self, config: Optional[Config] = None, **kwargs):
        self.config = config or Config()
        self.model: Any = None
        self.best_params = {}
        self.cv_results = {}
        self.is_trained = False
        self.feature_names = []
        # Almacenar parámetros para compatibilidad con scikit-learn
        self._params = kwargs
        # Inicializar parámetros para BaseEstimator
        # _estimator_type se define como atributo de clase

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Retornar parámetros por defecto para el algoritmo"""
        pass

    @abstractmethod
    def get_param_grid(self) -> Dict[str, List]:
        """Retornar grilla de hiperparámetros para búsqueda"""
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Retornar nombre del algoritmo"""
        pass

    # Métodos requeridos por BaseEstimator
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Obtener parámetros del estimador"""
        params = self._params.copy()
        if hasattr(self, "model") and self.model is not None and deep:
            model_params = self.model.get_params(deep=deep)
            params.update(model_params)
        return params

    def set_params(self, **params) -> "BaseClassifier":
        """Establecer parámetros del estimador"""
        self._params.update(params)
        if hasattr(self, "model") and self.model is not None:
            self.model.set_params(**params)
        return self

    # Métodos requeridos por ClassifierMixin
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Método fit requerido por ClassifierMixin"""
        self.train(X, y, optimize_params=False, feature_names=self.feature_names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Método predict requerido por ClassifierMixin"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        assert self.model is not None
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Método predict_proba requerido por ClassifierMixin"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        assert self.model is not None
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Método score requerido por ClassifierMixin"""
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return float(accuracy_score(y, y_pred))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize_params: bool = True,
        use_grid_search: bool = True,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Entrenar el modelo con optimización opcional de hiperparámetros"""

        self.logger.info(f"Iniciando entrenamiento de {self.get_algorithm_name()}")

        if feature_names:
            self.feature_names = feature_names

        if optimize_params:
            self._optimize_hyperparameters(X_train, y_train, use_grid_search)
        else:
            # Usar parámetros por defecto
            params = self.get_default_params()
            self.model = self._create_model(**params)
            self.model.fit(X_train, y_train)

        self.is_trained = True
        self.logger.info(f"Entrenamiento de {self.get_algorithm_name()} completado")

    def _optimize_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, use_grid_search: bool = True
    ) -> None:
        """Optimizar hiperparámetros usando GridSearch o RandomizedSearch"""

        # Obtener configuración de hiperparámetros usando el nuevo método
        hyperparameter_config = self.config.get_hyperparameter_config()

        param_grid = self.get_param_grid()
        base_model = self._create_model()

        if use_grid_search:
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=hyperparameter_config["cv_folds"],
                scoring=hyperparameter_config["scoring"],
                n_jobs=hyperparameter_config["n_jobs"],
                verbose=1,
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=hyperparameter_config["n_iter_search"],
                cv=hyperparameter_config["cv_folds"],
                scoring=hyperparameter_config["scoring"],
                n_jobs=hyperparameter_config["n_jobs"],
                random_state=hyperparameter_config["random_state"],
                verbose=1,
            )

        self.logger.info(
            f"Optimizando hiperparámetros con {'GridSearch' if use_grid_search else 'RandomizedSearch'}"
        )
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_results = {
            "best_score": search.best_score_,
            "best_params": search.best_params_,
            "cv_results": search.cv_results_,
        }

        self.logger.info(f"Mejores parámetros encontrados: {self.best_params}")
        self.logger.info(f"Mejor score CV: {search.best_score_:.4f}")

    @abstractmethod
    def _create_model(self, **params) -> Any:
        """Crear instancia del modelo con parámetros específicos"""
        pass

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, float]:
        """Realizar validación cruzada"""
        if not self.is_trained:
            raise ValueError(
                "El modelo debe ser entrenado antes de la validación cruzada"
            )

        # Obtener configuración de hiperparámetros usando el nuevo método
        hyperparameter_config = self.config.get_hyperparameter_config()

        assert self.model is not None
        scores = cross_val_score(
            self.model, X, y, cv=cv, scoring=hyperparameter_config["scoring"]
        )

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores.tolist(),
        }

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Obtener importancia de características si está disponible"""
        if not self.is_trained:
            return None

        assert self.model is not None
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # Para modelos lineales, usar valor absoluto de coeficientes
            return np.abs(self.model.coef_).flatten()
        else:
            return None

    def save_model(self, file_path: str) -> None:
        """Guardar modelo entrenado"""
        if not self.is_trained:
            raise ValueError("No se puede guardar un modelo no entrenado")

        model_data = {
            "model": self.model,
            "algorithm": self.get_algorithm_name(),
            "best_params": self.best_params,
            "cv_results": self.cv_results,
            "feature_names": self.feature_names,
            "config": self.config.__dict__,
        }

        joblib.dump(model_data, file_path)
        self.logger.info(f"Modelo guardado en: {file_path}")

    def load_model(self, file_path: str) -> None:
        """Cargar modelo desde archivo"""
        model_data = joblib.load(file_path)

        self.model = model_data["model"]
        self.best_params = model_data.get("best_params", {})
        self.cv_results = model_data.get("cv_results", {})
        self.feature_names = model_data.get("feature_names", [])
        self.is_trained = True

        self.logger.info(f"Modelo cargado desde: {file_path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Obtener resumen del modelo"""
        if not self.is_trained:
            return {"trained": False}

        summary = {
            "algorithm": self.get_algorithm_name(),
            "trained": self.is_trained,
            "best_params": self.best_params,
            "feature_count": (
                len(self.feature_names) if self.feature_names else "Unknown"
            ),
        }

        if self.cv_results:
            summary["best_cv_score"] = self.cv_results.get("best_score")

        return summary
