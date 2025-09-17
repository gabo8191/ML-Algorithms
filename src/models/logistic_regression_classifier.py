"""Clasificador de Regresión Logística con optimización de hiperparámetros.

Hiperparámetros principales:
- C: Parámetro de regularización (inverso de la fuerza de regularización)
- penalty: Tipo de regularización ('l1', 'l2', 'elasticnet', None)
- solver: Algoritmo de optimización ('liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga')
- max_iter: Número máximo de iteraciones
- class_weight: Pesos de las clases ('balanced', None, dict)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression

from .base_classifier import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """Clasificador de Regresión Logística"""

    def get_algorithm_name(self) -> str:
        return "Logistic Regression"

    def get_default_params(self) -> Dict[str, Any]:
        """Parámetros por defecto optimizados para la mayoría de casos"""
        return {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": self.config.get_hyperparameter_config()["random_state"],
            "class_weight": "balanced",
        }

    def get_param_grid(self) -> Dict[str, List]:
        """Grilla de hiperparámetros para optimización (reducida para velocidad)"""
        return {
            "C": [0.01, 0.1, 1.0, 10.0],  # Reducido de 6 a 4 valores
            "penalty": ["l1", "l2"],  # Reducido de 3 a 2 valores (eliminado elasticnet)
            "solver": [
                "liblinear",
                "lbfgs",
            ],  # Reducido de 3 a 2 valores (eliminado saga)
            "max_iter": [1000, 2000],  # Reducido de 3 a 2 valores
            "class_weight": [None, "balanced"],  # Mantenido
        }

    def _create_model(self, **params) -> LogisticRegression:
        """Crear instancia de LogisticRegression"""
        default_params = self.get_default_params()
        default_params.update(params)

        # Validar combinaciones de parámetros
        if default_params["penalty"] == "l1" and default_params["solver"] in [
            "lbfgs",
            "newton-cg",
        ]:
            default_params["solver"] = "liblinear"
        elif (
            default_params["penalty"] == "elasticnet"
            and default_params["solver"] != "saga"
        ):
            default_params["solver"] = "saga"
            default_params["l1_ratio"] = 0.5  # Para elasticnet

        return LogisticRegression(**default_params)

    def get_coefficients(self) -> np.ndarray:
        """Obtener coeficientes del modelo entrenado"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.coef_[0]

    def get_intercept(self) -> float:
        """Obtener intercepto del modelo entrenado"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.intercept_[0]

    def get_feature_contributions(self, X: np.ndarray) -> np.ndarray:
        """Calcular contribuciones de cada característica para las predicciones"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        # Contribución = coeficiente * valor_característica
        coefficients = self.get_coefficients()
        return X * coefficients

    def plot_coefficients(self, save_path: Optional[str] = None, show: bool = True):
        """Visualizar coeficientes del modelo"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
        import pandas as pd

        coefficients = self.get_coefficients()
        feature_names = (
            self.feature_names
            if self.feature_names
            else [f"Feature_{i}" for i in range(len(coefficients))]
        )

        # Crear DataFrame para facilitar la visualización
        coef_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coefficients,
                "Abs_Coefficient": np.abs(coefficients),
            }
        ).sort_values("Abs_Coefficient", ascending=True)

        plt.figure(figsize=(10, max(6, len(coefficients) * 0.4)))

        # Crear colores basados en el signo del coeficiente
        colors = ["red" if x < 0 else "blue" for x in coef_df["Coefficient"]]

        plt.barh(range(len(coef_df)), coef_df["Coefficient"], color=colors, alpha=0.7)
        plt.yticks(range(len(coef_df)), coef_df["Feature"])
        plt.xlabel("Coefficient Value")
        plt.title(
            f"Logistic Regression Coefficients\n(Blue: Positive impact, Red: Negative impact)"
        )
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        # Añadir valores en las barras
        for i, (idx, row) in enumerate(coef_df.iterrows()):
            plt.text(
                row["Coefficient"] + (0.01 if row["Coefficient"] >= 0 else -0.01),
                i,
                f'{row["Coefficient"]:.3f}',
                ha="left" if row["Coefficient"] >= 0 else "right",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(f"Gráfico de coeficientes guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return coef_df

    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray, save_path: Optional[str] = None, show: bool = True):
        """Visualizar curva ROC"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        import matplotlib.pyplot as plt  # type: ignore
        from sklearn.metrics import roc_curve, auc

        # Obtener probabilidades de predicción
        y_proba = self.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva

        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Logistic Regression - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(f"Curva ROC guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_calibration_curve(self, X_test: np.ndarray, y_test: np.ndarray, save_path: Optional[str] = None, show: bool = True):
        """Visualizar curva de calibración"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        import matplotlib.pyplot as plt  # type: ignore
        from sklearn.calibration import calibration_curve

        # Obtener probabilidades de predicción
        y_proba = self.predict_proba(X_test)[:, 1]

        # Calcular curva de calibración
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_proba, n_bins=10
        )

        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Logistic Regression")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve - Logistic Regression")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(f"Curva de calibración guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
