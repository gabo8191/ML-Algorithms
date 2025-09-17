"""Clasificador de Máquinas de Vector de Soporte (SVM) con optimización de hiperparámetros.

Hiperparámetros principales:
- C: Parámetro de regularización (controla el trade-off entre margen y errores)
- kernel: Tipo de kernel ('linear', 'poly', 'rbf', 'sigmoid')
- gamma: Coeficiente del kernel (para kernels no lineales)
- degree: Grado del kernel polinómico
- class_weight: Pesos de las clases ('balanced', None, dict)
- probability: Habilitar estimación de probabilidades
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.svm import SVC

from .base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    """Clasificador de Máquinas de Vector de Soporte"""

    def get_algorithm_name(self) -> str:
        return "Support Vector Machine"

    def get_default_params(self) -> Dict[str, Any]:
        """Parámetros por defecto optimizados para la mayoría de casos"""
        return {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "random_state": self.config.get_hyperparameter_config()["random_state"],
            "class_weight": "balanced",
            "probability": True,  # Necesario para predict_proba
        }

    def get_param_grid(self) -> Dict[str, List]:
        """Grilla de hiperparámetros para optimización (reducida para velocidad)"""
        return {
            "C": [0.1, 1.0, 10.0],  # Reducido de 4 a 3 valores
            "kernel": ["linear", "rbf"],  # Eliminado "poly" que es muy lento
            "gamma": ["scale", 0.01, 0.1],  # Reducido de 6 a 3 valores
            "class_weight": [None, "balanced"],  # Mantenido
        }

    def _create_model(self, **params) -> SVC:
        """Crear instancia de SVC"""
        default_params = self.get_default_params()
        default_params.update(params)

        # Limpiar parámetros incompatibles
        if default_params["kernel"] != "poly" and "degree" in default_params:
            del default_params["degree"]
        elif default_params["kernel"] == "linear" and "gamma" in default_params:
            # Para kernel lineal, gamma no se usa
            if default_params["gamma"] not in ["scale", "auto"]:
                default_params["gamma"] = "scale"

        return SVC(**default_params)

    def get_support_vectors(self) -> np.ndarray:
        """Obtener vectores de soporte"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.support_vectors_

    def get_support_vector_indices(self) -> np.ndarray:
        """Obtener índices de los vectores de soporte"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.support_

    def get_n_support(self) -> np.ndarray:
        """Obtener número de vectores de soporte por clase"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.n_support_

    def get_decision_function(self, X: np.ndarray) -> np.ndarray:
        """Obtener valores de la función de decisión"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.decision_function(X)

    def plot_decision_boundary_2d(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: List[int] = [0, 1],
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Visualizar frontera de decisión en 2D (solo para las primeras 2 características)"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        if X.shape[1] < 2:
            raise ValueError(
                "Se necesitan al menos 2 características para visualizar la frontera de decisión"
            )

        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore

        # Seleccionar solo las características especificadas
        X_plot = X[:, feature_indices]

        # Crear un modelo temporal con solo 2 características
        temp_model = self._create_model(
            **self.best_params if self.best_params else self.get_default_params()
        )
        temp_model.fit(X_plot, y)

        # Crear malla para la frontera de decisión
        h = 0.02  # Tamaño del paso en la malla
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predecir en la malla
        Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))

        # Plotear la frontera de decisión
        import matplotlib.cm as cm  # type: ignore

        plt.contourf(xx, yy, Z, alpha=0.4, cmap=cm.get_cmap("RdYlBu"))

        # Plotear los puntos de datos
        scatter = plt.scatter(
            X_plot[:, 0],
            X_plot[:, 1],
            c=y,
            cmap=cm.get_cmap("RdYlBu"),
            edgecolors="black",
        )

        # Plotear vectores de soporte
        support_vectors = temp_model.support_vectors_
        plt.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=100,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
            label=f"Support Vectors ({len(support_vectors)})",
        )

        feature_names = (
            self.feature_names
            if self.feature_names
            else [f"Feature_{i}" for i in range(X.shape[1])]
        )
        plt.xlabel(f"{feature_names[feature_indices[0]]}")
        plt.ylabel(f"{feature_names[feature_indices[1]]}")
        # Obtener parámetros del modelo de forma segura
        kernel = getattr(temp_model, "kernel", "unknown")
        C = getattr(temp_model, "C", "unknown")
        plt.title(f"SVM Decision Boundary\nKernel: {kernel}, C: {C}")
        plt.colorbar(scatter)
        plt.legend()

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(
                f"Gráfico de frontera de decisión guardado en: {save_path}"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def analyze_support_vectors(self) -> Dict[str, Any]:
        """Analizar información sobre los vectores de soporte"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        n_support = self.get_n_support()
        total_support = n_support.sum()

        analysis = {
            "total_support_vectors": total_support,
            "support_vectors_per_class": n_support.tolist(),
            "support_vector_ratio": total_support / len(self.model.support_),
            "kernel": getattr(self.model, "kernel", "unknown"),
            "C": getattr(self.model, "C", "unknown"),
            "gamma": getattr(self.model, "gamma", "N/A"),
        }

        self.logger.info(f"Análisis de vectores de soporte: {analysis}")
        return analysis
