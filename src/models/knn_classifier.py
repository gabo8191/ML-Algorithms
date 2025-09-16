"""Objetivo: Implementar el clasificador KNN con utilidades de optimización, entrenamiento,
predicción y análisis. Bosquejo: clase KNNClassifier que busca K óptimo por CV
(`find_optimal_k`, `plot_k_optimization`), opcional grid search (`grid_search_optimization`),
entrena (`train`), predice (`predict`, `predict_proba`), evalúa (`evaluate`), grafica matriz
de confusión (`plot_confusion_matrix`), calcula y grafica importancia de características
(`feature_importance_analysis`, `plot_feature_importance`, `calculate_feature_importance`),
persiste/carga el modelo (`save_model`, `load_model`) y entrega un resumen (`get_model_summary`).
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class KNNClassifier(LoggerMixin):

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model: Optional[KNeighborsClassifier] = None
        self.best_k: int = self.config.DEFAULT_K
        self.best_params: Dict[str, Any] = {}
        self.cv_results: Dict[str, Any] = {}
        self.is_trained: bool = False

        self.train_accuracy: float = 0.0
        self.test_accuracy: float = 0.0
        self.cv_scores: List[float] = []

    def find_optimal_k(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_range: Optional[Tuple[int, int]] = None,
        cv_folds: Optional[int] = None,
        scoring: str = "accuracy",
    ) -> int:

        if k_range is None:
            k_range = self.config.K_RANGE
        if cv_folds is None:
            cv_folds = self.config.CV_FOLDS

        self.logger.info(f"Buscando K óptimo en rango {k_range} con {cv_folds}-fold CV")

        k_values = list(range(k_range[0], k_range[1]))
        cv_scores_mean = []
        cv_scores_std = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)

            scores = cross_val_score(
                knn, X_train, y_train, cv=cv_folds, scoring=scoring, n_jobs=-1
            )

            cv_scores_mean.append(scores.mean())
            cv_scores_std.append(scores.std())

        best_idx = np.argmax(cv_scores_mean)
        self.best_k = k_values[best_idx]

        self.cv_results = {
            "k_values": k_values,
            "cv_scores_mean": cv_scores_mean,
            "cv_scores_std": cv_scores_std,
            "best_k": self.best_k,
            "best_score": cv_scores_mean[best_idx],
        }

        self.logger.info(
            f"K óptimo encontrado: {self.best_k} (Score: {cv_scores_mean[best_idx]:.4f})"
        )

        return self.best_k

    def plot_k_optimization(
        self, save_path: Optional[str] = None, show: bool = True
    ) -> None:

        if not self.cv_results:
            self.logger.warning("No hay resultados de optimización de K para graficar")
            return

        plt.figure(figsize=self.config.FIGURE_SIZE)

        plt.figtext(
            0.5,
            0.97,
            "OBJETIVO: Optimizar K para clasificación de supervivencia del Titanic (SOBREVIVIÓ/NO SOBREVIVIÓ).\n"
            "INTERPRETACIÓN: K bajos (1-5) sobreajustan, K altos (>25) pierden patrones locales.\n"
            "K óptimo balancea sesgo-varianza para mejor clasificación de supervivencia.",
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.9),
        )

        k_values = self.cv_results["k_values"]
        means = self.cv_results["cv_scores_mean"]
        stds = self.cv_results["cv_scores_std"]

        plt.errorbar(k_values, means, yerr=stds, marker="o", capsize=5, capthick=2)
        plt.axvline(
            x=self.best_k,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"K óptimo = {self.best_k}",
        )

        plt.xlabel("Valor de K")
        plt.ylabel("Accuracy (Validación Cruzada)")
        plt.title("Optimización del Hiperparámetro K")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.82)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Gráfico guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def grid_search_optimization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: Optional[int] = None,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:

        if param_grid is None:
            param_grid = {
                "n_neighbors": list(
                    range(self.config.K_RANGE[0], self.config.K_RANGE[1])
                ),
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            }

        if cv_folds is None:
            cv_folds = self.config.CV_FOLDS

        self.logger.info("Iniciando búsqueda en grilla de hiperparámetros")

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            knn, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.best_k = self.best_params["n_neighbors"]

        self.logger.info(f"Mejores parámetros: {self.best_params}")
        self.logger.info(f"Mejor score: {grid_search.best_score_:.4f}")

        return self.best_params

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize_k: bool = True,
        use_grid_search: bool = False,
    ) -> "KNNClassifier":

        self.logger.info("Iniciando entrenamiento del clasificador KNN")

        if use_grid_search:
            self.grid_search_optimization(X_train, y_train)
            model_params = self.best_params
        elif optimize_k:
            self.find_optimal_k(X_train, y_train)
            model_params = self.config.get_model_config()
            model_params["n_neighbors"] = self.best_k
        else:
            model_params = self.config.get_model_config()

        self.model = KNeighborsClassifier(**model_params)
        self.model.fit(X_train, y_train)

        y_train_pred = self.model.predict(X_train)
        self.train_accuracy = float(accuracy_score(y_train, y_train_pred))

        self.is_trained = True
        self.logger.info(
            f"Modelo entrenado exitosamente. Accuracy entrenamiento: {self.train_accuracy:.4f}"
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained or self.model is None:
            raise ValueError("El modelo no ha sido entrenado")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained or self.model is None:
            raise ValueError("El modelo no ha sido entrenado")

        proba_result = self.model.predict_proba(X)
        if isinstance(proba_result, list):
            return np.array(proba_result[0])
        return proba_result

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        if not self.is_trained or self.model is None:
            raise ValueError("El modelo no ha sido entrenado")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        self.test_accuracy = float(accuracy_score(y_test, y_pred))

        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )

        cm = confusion_matrix(y_test, y_pred)

        evaluation_results = {
            "test_accuracy": self.test_accuracy,
            "train_accuracy": self.train_accuracy,
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": y_pred,
            "probabilities": y_proba,
            "best_k": self.best_k,
        }

        self.logger.info(
            f"Evaluación completada. Test Accuracy: {self.test_accuracy:.4f}"
        )

        return evaluation_results

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names if class_names is not None else True,
            yticklabels=class_names if class_names is not None else True,
        )

        plt.title("Matriz de Confusión")
        plt.ylabel("Valores Reales")
        plt.xlabel("Valores Predichos")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Matriz de confusión guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def feature_importance_analysis(
        self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]
    ) -> Dict[str, float]:
        if not self.is_trained or self.model is None:
            raise ValueError("El modelo no ha sido entrenado")

        from sklearn.inspection import permutation_importance

        self.logger.info("Calculando importancia de características")

        perm_importance = permutation_importance(
            self.model,
            X_train,
            y_train,
            n_repeats=10,
            random_state=self.config.RANDOM_STATE,
        )

        importance_dict = {}
        importances = getattr(perm_importance, "importances_mean", None)
        for i, feature in enumerate(feature_names):
            try:
                value = float(importances[i]) if importances is not None else 0.0
                importance_dict[feature] = value
            except Exception:
                importance_dict[feature] = 0.0

        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        self.logger.info("Análisis de importancia completado")

        return importance_dict

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        features = list(importance_dict.keys())
        importances = list(importance_dict.values())

        plt.figure(figsize=self.config.FIGURE_SIZE)

        plt.figtext(
            0.5,
            0.95,
            "OBJETIVO: Determinar qué características influyen más en la supervivencia del Titanic.\n"
            "INTERPRETACIÓN: Variables con mayor importancia tienen más influencia en las predicciones. "
            "Valores más altos indican mayor contribución al rendimiento del modelo.",
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.barh(features, importances)
        plt.xlabel("Importancia")
        plt.title("Importancia de Características (Permutación)")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Gráfico de importancia guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def calculate_feature_importance(
        self, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]
    ) -> Dict[str, float]:

        if not self.is_trained or self.model is None:
            raise ValueError("El modelo no ha sido entrenado")

        self.logger.info(
            "🔍 Calculando importancia de características (relaciones entre variables)"
        )

        baseline_score = self.model.score(X_test, y_test)
        importance_values: List[float] = []

        n_repeats = 10
        for i, _feature in enumerate(feature_names):
            drops: List[float] = []
            for _ in range(n_repeats):
                X_permuted = X_test.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                permuted_score = self.model.score(X_permuted, y_test)
                drops.append(max(0.0, float(baseline_score - permuted_score)))
            importance_values.append(float(np.mean(drops)))

        imp = np.array(importance_values, dtype=float)
        imp = np.nan_to_num(imp, nan=0.0)
        imp = np.maximum(imp, 0.0)
        total = float(np.sum(imp))
        if total > 0:
            imp = imp / total

        importance_dict = {name: float(val) for name, val in zip(feature_names, imp)}

        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        self.logger.info("✅ Relaciones entre variables identificadas:")
        for feature, importance in importance_dict.items():
            self.logger.info(f"   • {feature}: {importance:.1%}")

        return importance_dict

    def save_model(self, filepath: str) -> None:

        if not self.is_trained or self.model is None:
            raise ValueError("No hay modelo entrenado para guardar")

        model_data = {
            "model": self.model,
            "best_k": self.best_k,
            "best_params": self.best_params,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "cv_results": self.cv_results,
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath: str) -> "KNNClassifier":

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.best_k = model_data["best_k"]
        self.best_params = model_data.get("best_params", {})
        self.train_accuracy = model_data.get("train_accuracy", 0.0)
        self.test_accuracy = model_data.get("test_accuracy", 0.0)
        self.cv_results = model_data.get("cv_results", {})
        self.is_trained = True

        self.logger.info(f"Modelo cargado desde: {filepath}")

        return self

    def get_model_summary(self) -> Dict[str, Any]:

        if not self.is_trained:
            return {"status": "No entrenado"}

        return {
            "status": "Entrenado",
            "best_k": self.best_k,
            "best_params": self.best_params,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "model_type": type(self.model).__name__,
        }
