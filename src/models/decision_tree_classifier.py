"""Clasificador de Árbol de Decisión con optimización de hiperparámetros.

Hiperparámetros principales:
- criterion: Función para medir la calidad de la división ('gini', 'entropy')
- max_depth: Profundidad máxima del árbol
- min_samples_split: Número mínimo de muestras para dividir un nodo interno
- min_samples_leaf: Número mínimo de muestras en un nodo hoja
- max_features: Número de características a considerar en cada división
- class_weight: Pesos de las clases ('balanced', None, dict)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from .base_classifier import BaseClassifier


class DecisionTreeClassifierCustom(BaseClassifier):
    """Clasificador de Árbol de Decisión"""

    def get_algorithm_name(self) -> str:
        return "Decision Tree"

    def get_default_params(self) -> Dict[str, Any]:
        """Parámetros por defecto optimizados para evitar overfitting"""
        return {
            "criterion": "gini",
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "random_state": self.config.get_hyperparameter_config()["random_state"],
            "class_weight": "balanced",
        }

    def get_param_grid(self) -> Dict[str, List]:
        """Grilla de hiperparámetros para optimización (reducida para velocidad)"""
        return {
            "criterion": ["gini", "entropy"],  # Mantenido
            "max_depth": [5, 10, 15, None],  # Reducido de 6 a 4 valores
            "min_samples_split": [2, 5, 10],  # Reducido de 4 a 3 valores
            "min_samples_leaf": [1, 2, 5],  # Reducido de 4 a 3 valores
            "max_features": [
                "sqrt",
                "log2",
            ],  # Reducido de 3 a 2 valores (eliminado None)
            "class_weight": [None, "balanced"],  # Mantenido
        }

    def _create_model(self, **params) -> DecisionTreeClassifier:
        """Crear instancia de DecisionTreeClassifier"""
        default_params = self.get_default_params()
        default_params.update(params)
        return DecisionTreeClassifier(**default_params)

    def get_tree_depth(self) -> int:
        """Obtener la profundidad real del árbol entrenado"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.tree_.max_depth

    def get_n_leaves(self) -> int:
        """Obtener el número de hojas del árbol"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.tree_.n_leaves

    def get_n_nodes(self) -> int:
        """Obtener el número total de nodos del árbol"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.tree_.node_count

    def plot_tree_visualization(
        self, max_depth: int = 3, save_path: Optional[str] = None, show: bool = True
    ):
        """Visualizar el árbol de decisión"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        import matplotlib.pyplot as plt  # type: ignore

        feature_names = self.feature_names if self.feature_names else None
        class_names = ["Not Successful", "Successful"]  # Para el problema de cafeterías

        plt.figure(figsize=(20, 10))

        assert self.model is not None
        plot_tree(
            self.model,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
        )

        plt.title(
            f"Decision Tree Visualization (Max Depth: {max_depth})\n"
            f"Actual Depth: {self.get_tree_depth()}, Nodes: {self.get_n_nodes()}, Leaves: {self.get_n_leaves()}"
        )

        if save_path is not None:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            self.logger.info(f"Visualización del árbol guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def export_tree_rules(self, save_path: Optional[str] = None) -> str:
        """Exportar las reglas del árbol en formato texto"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        feature_names = self.feature_names if self.feature_names else None
        assert self.model is not None
        tree_rules = export_text(
            self.model, feature_names=feature_names, show_weights=True
        )

        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(tree_rules)
            self.logger.info(f"Reglas del árbol exportadas a: {save_path}")

        return tree_rules

    def get_leaf_samples(self) -> np.ndarray:
        """Obtener el número de muestras en cada hoja"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        assert self.model is not None
        return self.model.tree_.n_node_samples

    def analyze_tree_structure(self) -> Dict[str, Any]:
        """Analizar la estructura del árbol entrenado"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        assert self.model is not None
        tree = self.model.tree_

        # Calcular estadísticas del árbol
        leaf_nodes = []
        internal_nodes = []

        for i in range(tree.node_count):
            if tree.children_left[i] == tree.children_right[i]:  # Es hoja
                leaf_nodes.append(i)
            else:  # Es nodo interno
                internal_nodes.append(i)

        # Obtener importancia de características
        feature_importance = self.model.feature_importances_

        analysis = {
            "max_depth": tree.max_depth,
            "n_nodes": tree.node_count,
            "n_leaves": len(leaf_nodes),
            "n_internal_nodes": len(internal_nodes),
            "feature_importance": feature_importance.tolist(),
            "most_important_features": [],
        }

        # Identificar características más importantes
        if self.feature_names and len(self.feature_names) == len(feature_importance):
            feature_importance_pairs = list(zip(self.feature_names, feature_importance))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            analysis["most_important_features"] = feature_importance_pairs[:5]

        self.logger.info(f"Análisis de estructura del árbol: {analysis}")
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
            color="skyblue",
            alpha=0.7,
        )
        plt.yticks(range(len(importance_df)), importance_df["Feature"])
        plt.xlabel("Feature Importance")
        plt.title("Decision Tree Feature Importance")

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
