"""Clasificador Random Forest con optimización de hiperparámetros.

Hiperparámetros principales:
- n_estimators: Número de árboles en el bosque
- criterion: Función para medir la calidad de la división ('gini', 'entropy')
- max_depth: Profundidad máxima de los árboles
- min_samples_split: Número mínimo de muestras para dividir un nodo interno
- min_samples_leaf: Número mínimo de muestras en un nodo hoja
- max_features: Número de características a considerar en cada división
- bootstrap: Si usar bootstrap para construir árboles
- class_weight: Pesos de las clases ('balanced', None, dict)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier

from .base_classifier import BaseClassifier


class RandomForestClassifierCustom(BaseClassifier):
    """Clasificador Random Forest"""

    def get_algorithm_name(self) -> str:
        return "Random Forest"

    def get_default_params(self) -> Dict[str, Any]:
        """Parámetros por defecto optimizados para balance entre precisión y velocidad"""
        return {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": self.config.get_hyperparameter_config()["random_state"],
            "class_weight": "balanced",
            "n_jobs": -1,  # Usar todos los cores disponibles
        }

    def get_param_grid(self) -> Dict[str, List]:
        """Grilla de hiperparámetros para optimización (reducida para velocidad)"""
        return {
            "n_estimators": [100, 200],  # Reducido de 4 a 2 valores
            "criterion": ["gini", "entropy"],  # Mantenido
            "max_depth": [10, 15, None],  # Reducido de 5 a 3 valores
            "min_samples_split": [2, 5],  # Reducido de 3 a 2 valores
            "min_samples_leaf": [1, 2],  # Reducido de 3 a 2 valores
            "max_features": [
                "sqrt",
                "log2",
            ],  # Reducido de 3 a 2 valores (eliminado None)
            "bootstrap": [True],  # Reducido de 2 a 1 valor (True es más común)
            "class_weight": [None, "balanced"],  # Mantenido
        }

    def _create_model(self, **params) -> RandomForestClassifier:
        """Crear instancia de RandomForestClassifier"""
        default_params = self.get_default_params()
        default_params.update(params)
        return RandomForestClassifier(**default_params)

    def get_oob_score(self) -> Optional[float]:
        """Obtener el score Out-of-Bag si está disponible"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        if hasattr(self.model, "oob_score_"):
            return self.model.oob_score_
        return None

    def get_estimators(self) -> List:
        """Obtener los estimadores individuales (árboles)"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.estimators_

    def analyze_forest_structure(self) -> Dict[str, Any]:
        """Analizar la estructura del bosque"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        estimators = self.get_estimators()

        # Analizar cada árbol
        depths = []
        n_nodes = []
        n_leaves = []

        for tree in estimators:
            depths.append(tree.tree_.max_depth)
            n_nodes.append(tree.tree_.node_count)
            n_leaves.append(tree.tree_.n_leaves)

        analysis = {
            "n_estimators": len(estimators),
            "avg_depth": np.mean(depths),
            "max_depth": np.max(depths),
            "min_depth": np.min(depths),
            "avg_nodes": np.mean(n_nodes),
            "avg_leaves": np.mean(n_leaves),
            "oob_score": self.get_oob_score(),
            "feature_importance": self.model.feature_importances_.tolist(),
        }

        self.logger.info(f"Análisis de estructura del bosque: {analysis}")
        return analysis

    def plot_feature_importance(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """Visualizar importancia de características"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd

        assert self.model is not None
        feature_importance = self.model.feature_importances_
        feature_names = (
            self.feature_names
            if self.feature_names
            else [f"Feature_{i}" for i in range(len(feature_importance))]
        )

        # Crear DataFrame y ordenar por importancia
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importance}
        ).sort_values("Importance", ascending=True)

        plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))

        plt.barh(
            range(len(importance_df)),
            importance_df["Importance"],
            color="forestgreen",
            alpha=0.7,
        )
        plt.yticks(range(len(importance_df)), importance_df["Feature"])
        plt.xlabel("Feature Importance")
        plt.title("Random Forest Feature Importance")

        # Añadir valores en las barras
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            plt.text(
                row["Importance"] + 0.001,
                i,
                f'{row["Importance"]:.3f}',
                va="center",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(
                f"Gráfico de importancia de características guardado en: {save_path}"
            )

        if show:
            plt.show()
        else:
            plt.close()

        return importance_df

    def plot_trees_depth_distribution(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """Visualizar distribución de profundidades de los árboles"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        import matplotlib.pyplot as plt  # type: ignore

        estimators = self.get_estimators()
        depths = [tree.tree_.max_depth for tree in estimators]

        plt.figure(figsize=(10, 6))

        plt.hist(depths, bins=20, alpha=0.7, color="forestgreen", edgecolor="black")
        plt.xlabel("Tree Depth")
        plt.ylabel("Number of Trees")
        plt.title(
            f"Distribution of Tree Depths in Random Forest\n"
            f"Mean: {np.mean(depths):.1f}, Std: {np.std(depths):.1f}"
        )

        # Añadir línea vertical para la media
        mean_depth = float(np.mean(depths))
        plt.axvline(
            mean_depth,
            color="red",
            linestyle="--",
            label=f"Mean Depth: {mean_depth:.1f}",
        )
        plt.legend()

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(
                f"Gráfico de distribución de profundidades guardado en: {save_path}"
            )

        if show:
            plt.show()
        else:
            plt.close()

    def get_feature_importance_std(self) -> np.ndarray:
        """Calcular desviación estándar de la importancia de características entre árboles"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        estimators = self.get_estimators()
        assert self.model is not None
        n_features = len(self.model.feature_importances_)

        # Recopilar importancias de todos los árboles
        all_importances = np.zeros((len(estimators), n_features))

        for i, tree in enumerate(estimators):
            all_importances[i] = tree.feature_importances_

        # Calcular desviación estándar
        return np.std(all_importances, axis=0)

    def plot_feature_importance_with_std(
        self, save_path: Optional[str] = None, show: bool = True
    ):
        """Visualizar importancia de características con barras de error"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd

        feature_importance = self.model.feature_importances_
        feature_importance_std = self.get_feature_importance_std()
        feature_names = (
            self.feature_names
            if self.feature_names
            else [f"Feature_{i}" for i in range(len(feature_importance))]
        )

        # Crear DataFrame y ordenar por importancia
        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": feature_importance,
                "Std": feature_importance_std,
            }
        ).sort_values("Importance", ascending=True)

        plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))

        plt.barh(
            range(len(importance_df)),
            importance_df["Importance"],
            xerr=importance_df["Std"],
            color="forestgreen",
            alpha=0.7,
            capsize=3,
        )
        plt.yticks(range(len(importance_df)), importance_df["Feature"])
        plt.xlabel("Feature Importance")
        plt.title("Random Forest Feature Importance with Standard Deviation")

        # Añadir valores en las barras
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            plt.text(
                row["Importance"] + row["Std"] + 0.001,
                i,
                f'{row["Importance"]:.3f}±{row["Std"]:.3f}',
                va="center",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(
                f"Gráfico de importancia con desviación estándar guardado en: {save_path}"
            )

        if show:
            plt.show()
        else:
            plt.close()

        return importance_df
