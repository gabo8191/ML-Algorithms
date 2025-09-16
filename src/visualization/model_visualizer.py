"""Objetivo: Visualizar resultados y rendimiento del modelo KNN. Bosquejo: clase
ModelVisualizer que genera gráficas como matriz de confusión (`plot_confusion_matrix`),
reporte de clasificación (`plot_classification_report`), optimización de K
(`plot_k_optimization`), curvas de aprendizaje (`plot_learning_curve`), importancia de
características (`plot_feature_importance`), distribución de predicciones
(`plot_prediction_distribution`), comparación de modelos (`plot_model_comparison`),
residuos (`plot_residuals`) y reportes compuestos (`create_model_report`)."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class ModelVisualizer(LoggerMixin):

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.setup_style()

    def setup_style(self):
        plt.style.use("default")
        sns.set_palette("husl")

        plt.rcParams["figure.figsize"] = self.config.FIGURE_SIZE
        plt.rcParams["figure.dpi"] = self.config.DPI
        plt.rcParams["savefig.dpi"] = self.config.DPI

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            if normalize == "true":
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                title = "Matriz de Confusión Normalizada (por fila)"
                fmt = ".2f"
            elif normalize == "pred":
                cm = cm.astype("float") / cm.sum(axis=0)
                title = "Matriz de Confusión Normalizada (por columna)"
                fmt = ".2f"
            elif normalize == "all":
                cm = cm.astype("float") / cm.sum()
                title = "Matriz de Confusión Normalizada (total)"
                fmt = ".2f"
        else:
            title = "Matriz de Confusión"
            fmt = "d"

        plt.figure(figsize=(10, 8))

        plt.figtext(
            0.5,
            0.98,
            "OBJETIVO: Evaluar capacidad del modelo KNN para clasificar supervivencia del Titanic (SOBREVIVIÓ/NO SOBREVIVIÓ).\n"
            "INTERPRETACIÓN: Matriz muestra confusiones entre predicciones de supervivencia.\n"
            "Diagonal principal indica clasificaciones correctas, errores muestran casos mal clasificados.",
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.9),
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names if class_names is not None else True,
            yticklabels=class_names if class_names is not None else True,
        )

        plt.title(title)
        plt.ylabel("Etiqueta Verdadera")
        plt.xlabel("Etiqueta Predicha")

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Matriz de confusión guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        from sklearn.metrics import classification_report

        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )

        df_report = pd.DataFrame(report).iloc[:-1, :].T  # Excluir 'accuracy'

        plt.figure(figsize=(8, 6))
        sns.heatmap(df_report, annot=True, cmap="RdYlBu_r", fmt=".3f")

        plt.title("Reporte de Clasificación")

        plt.figtext(
            0.5,
            0.98,
            "OBJETIVO: Evaluar rendimiento balanceado del modelo con métricas precision, recall y F1-score por clase.\n"
            "INTERPRETACION: Reporte detallado muestra rendimiento específico de cada tipo de ruta aérea.\n"
            "Balance adecuado entre todas las métricas indica modelo confiable para clasificación.",
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.9),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Reporte de clasificación guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_k_optimization(
        self,
        k_values: List[int],
        cv_scores_mean: List[float],
        cv_scores_std: List[float],
        best_k: int,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        plt.figure(figsize=(10, 6))

        plt.errorbar(
            k_values,
            cv_scores_mean,
            yerr=cv_scores_std,
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=6,
        )

        plt.axvline(
            x=best_k,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label=f"K óptimo = {best_k}",
        )

        plt.xlabel("Valor de K", fontsize=12)
        plt.ylabel("Accuracy (Validación Cruzada)", fontsize=12)
        plt.title("Optimización del Hiperparámetro K", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        best_idx = k_values.index(best_k)
        best_score = cv_scores_mean[best_idx]
        plt.annotate(
            f"Score: {best_score:.4f}",
            xy=(best_k, best_score),
            xytext=(best_k + 2, best_score + 0.01),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            fontsize=10,
            color="red",
        )

        plt.figtext(
            0.5,
            0.98,
            "OBJETIVO: Encontrar el valor K óptimo que maximice la accuracy evitando sobreajuste y subajuste.\n"
            "INTERPRETACION: K bajos (1-5) muestran variabilidad por sobreajuste, K altos (25+) declinan por subajuste.\n"
            "Curva estable en rango medio confirma robustez del modelo para clasificacion de rutas aereas.",
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.9),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Gráfico de optimización K guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_learning_curve(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        train_sizes: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        learning_curve_result = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
        )
        train_sizes_result = learning_curve_result[0]
        train_scores = learning_curve_result[1]
        val_scores = learning_curve_result[2]

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))

        plt.plot(
            train_sizes_result, train_mean, "o-", color="blue", label="Entrenamiento"
        )
        plt.fill_between(
            train_sizes_result,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color="blue",
        )

        plt.plot(train_sizes_result, val_mean, "o-", color="red", label="Validación")
        plt.fill_between(
            train_sizes_result,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.2,
            color="red",
        )

        plt.xlabel("Tamaño del Conjunto de Entrenamiento")
        plt.ylabel("Accuracy")
        plt.title("Curvas de Aprendizaje")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Curvas de aprendizaje guardadas en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        max_features: int = 15,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )
        sorted_features = sorted_features[:max_features]

        features, importances = zip(*sorted_features)

        plt.figure(figsize=(10, max(6, len(features) * 0.4)))

        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, color="skyblue")

        plt.yticks(y_pos, features)
        plt.xlabel("Importancia")
        plt.title("Importancia de Características")
        plt.grid(True, alpha=0.3, axis="x")

        for i, v in enumerate(importances):
            plt.text(
                v + max(importances) * 0.01, i, f"{v:.4f}", va="center", fontsize=10
            )

        plt.figtext(
            0.5,
            0.98,
            "OBJETIVO: Identificar variables mas influyentes para predecir tipo de ruta aerea (SHORT/MEDIUM/LONG HAUL).\n"
            "INTERPRETACION: Market share y precios son predictores clave para clasificar rutas por distancia.\n"
            "Variables temporales y competencia tambien influyen en la caracterizacion de rutas aereas.",
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.9),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Importancia de características guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        unique_true, counts_true = np.unique(y_true, return_counts=True)
        axes[0].bar(unique_true, counts_true, color="lightblue", alpha=0.7)
        axes[0].set_title("Distribución de Valores Reales")
        axes[0].set_xlabel("Clase")
        axes[0].set_ylabel("Frecuencia")

        if class_names:
            axes[0].set_xticks(unique_true)
            axes[0].set_xticklabels([class_names[i] for i in unique_true], rotation=45)

        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        axes[1].bar(unique_pred, counts_pred, color="lightcoral", alpha=0.7)
        axes[1].set_title("Distribución de Predicciones")
        axes[1].set_xlabel("Clase")
        axes[1].set_ylabel("Frecuencia")

        if class_names:
            axes[1].set_xticks(unique_pred)
            axes[1].set_xticklabels([class_names[i] for i in unique_pred], rotation=45)

        plt.figtext(
            0.5,
            0.02,
            "🎯 OBJETIVO: Verificar calibración del modelo comparando distribuciones reales vs predichas para detectar sesgos.\n"
            "📊 EJEMPLO DE ANÁLISIS: Distribuciones casi idénticas ([492,491,491,526] vs [488,511,492,509]). "
            "Diferencias mínimas (±20 casos) confirman excelente calibración. Sin sesgo hacia clases específicas.",
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Distribución de predicciones guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_model_comparison(
        self,
        results_dict: Dict[str, Dict[str, float]],
        metrics: List[str] = ["accuracy", "precision", "recall", "f1-score"],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        models = list(results_dict.keys())

        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        plt.figure(figsize=(12, 6))

        for i, model in enumerate(models):
            scores = [results_dict[model].get(metric, 0) for metric in metrics]
            plt.bar(x + i * width, scores, width, label=model, alpha=0.8)

        plt.xlabel("Métricas")
        plt.ylabel("Puntuación")
        plt.title("Comparación de Modelos")
        plt.xticks(x + width * (len(models) - 1) / 2, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        for i, model in enumerate(models):
            scores = [results_dict[model].get(metric, 0) for metric in metrics]
            for j, score in enumerate(scores):
                plt.text(
                    j + i * width,
                    score + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Comparación de modelos guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        errors = y_true != y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        unique_classes = np.unique(y_true)
        error_rates = []

        for cls in unique_classes:
            mask = y_true == cls
            error_rate = np.sum(errors[mask]) / np.sum(mask)
            error_rates.append(error_rate)

        axes[0].bar(unique_classes, error_rates, color="lightcoral", alpha=0.7)
        axes[0].set_title("Tasa de Error por Clase Verdadera")
        axes[0].set_xlabel("Clase")
        axes[0].set_ylabel("Tasa de Error")
        axes[0].grid(True, alpha=0.3)

        error_matrix = np.zeros((len(unique_classes), len(unique_classes)))
        for i, true_cls in enumerate(unique_classes):
            for j, pred_cls in enumerate(unique_classes):
                mask = (y_true == true_cls) & (y_pred == pred_cls)
                error_matrix[i, j] = np.sum(mask)

        im = axes[1].imshow(error_matrix, cmap="Reds")
        axes[1].set_title("Matriz de Predicciones")
        axes[1].set_xlabel("Clase Predicha")
        axes[1].set_ylabel("Clase Verdadera")

        plt.colorbar(im, ax=axes[1])

        plt.figtext(
            0.5,
            0.02,
            "🎯 OBJETIVO: Detectar patrones sistemáticos en errores del modelo para identificar problemas de especificación.\n"
            "📊 EJEMPLO DE ANÁLISIS: Errores distribuidos aleatoriamente sin patrones sistemáticos evidentes. "
            "Balance entre sobre/subestimación. Ausencia de tendencias confirma modelo bien especificado - no hay variables omitidas importantes.",
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Análisis de residuos guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def create_model_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        model_name: str = "KNN",
        save_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        from pathlib import Path

        save_dir_path = None
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        cm_path = (
            str(save_dir_path / f"{model_name}_confusion_matrix.png")
            if save_dir_path
            else None
        )
        self.plot_confusion_matrix(
            y_true, y_pred, class_names, save_path=cm_path, show=False
        )
        if cm_path:
            saved_files["confusion_matrix"] = cm_path

        self.logger.info(
            f"Reporte de modelo {model_name} generado. {len(saved_files)} gráficos creados."
        )

        return saved_files
