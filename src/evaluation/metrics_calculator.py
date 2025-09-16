"""Objetivo: Calcular métricas de clasificación (básicas, avanzadas, por clase, y
derivadas de la matriz de confusión), validación cruzada y análisis de errores.
Bosquejo: clase MetricsCalculator con métodos para métricas globales
(`calculate_basic_metrics`, `calculate_advanced_metrics`), por clase
(`calculate_per_class_metrics`), a partir de la matriz de confusión
(`calculate_confusion_matrix_metrics`), validación cruzada
(`calculate_cross_validation_metrics`), análisis de errores
(`calculate_classification_errors`), combinar todo en un reporte diccionario
(`generate_classification_report_dict`), comparar modelos (`compare_models`),
estabilidad del modelo (`calculate_model_stability`) e impresión de resumen
(`print_metrics_summary`)."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Literal
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from sklearn.model_selection import cross_val_score, cross_validate
import warnings

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class MetricsCalculator(LoggerMixin):

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def calculate_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: Literal[
            "micro", "macro", "samples", "weighted", "binary"
        ] = "weighted",
    ) -> Dict[str, float]:
        """
        Calcula métricas básicas de clasificación


        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        self.logger.info("Métricas básicas calculadas")
        return metrics

    def calculate_advanced_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:

        metrics = {
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        }

        self.logger.info("Métricas avanzadas calculadas")
        return metrics

    def calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:

        unique_classes = np.unique(np.concatenate([y_true, y_pred]))

        if class_names is None:
            class_names = [f"Class_{i}" for i in unique_classes]
        elif len(class_names) < len(unique_classes):
            class_names.extend(
                [f"Class_{i}" for i in range(len(class_names), len(unique_classes))]
            )

        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        support_per_class = np.bincount(y_true, minlength=len(unique_classes))

        metrics_df = pd.DataFrame(
            {
                "Class": [class_names[i] for i in unique_classes],
                "Precision": precision_per_class,
                "Recall": recall_per_class,
                "F1-Score": f1_per_class,
                "Support": support_per_class,
            }
        )

        self.logger.info("Métricas por clase calculadas")
        return metrics_df

    def calculate_confusion_matrix_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:

        cm = confusion_matrix(y_true, y_pred)

        with np.errstate(all="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm_true = np.divide(cm, row_sums, where=row_sums != 0)

            col_sums = cm.sum(axis=0, keepdims=True)
            cm_norm_pred = np.divide(cm, col_sums, where=col_sums != 0)

            total = cm.sum()
            cm_norm_overall = (
                (cm / total) if total > 0 else np.zeros_like(cm, dtype=float)
            )

        total_samples = cm.sum()
        correct_predictions = np.trace(cm)  # Diagonal principal

        metrics = {
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized_by_true": cm_norm_true.tolist(),
            "confusion_matrix_normalized_by_pred": cm_norm_pred.tolist(),
            "confusion_matrix_normalized_overall": cm_norm_overall.tolist(),
            "total_samples": int(total_samples),
            "correct_predictions": int(correct_predictions),
            "incorrect_predictions": int(total_samples - correct_predictions),
            "error_rate": float((total_samples - correct_predictions) / total_samples),
            "class_distribution": cm.sum(axis=1).tolist(),
            "prediction_distribution": cm.sum(axis=0).tolist(),
        }

        self.logger.info("Métricas de matriz de confusión calculadas")
        return metrics

    def calculate_cross_validation_metrics(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        if scoring is None:
            scoring = [
                "accuracy",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
            ]

        self.logger.info(f"Iniciando validación cruzada con {cv} folds")

        cv_results = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        metrics = {}
        for metric in scoring:
            test_scores = cv_results[f"test_{metric}"]
            metrics[metric] = {
                "mean": float(np.mean(test_scores)),
                "std": float(np.std(test_scores)),
                "min": float(np.min(test_scores)),
                "max": float(np.max(test_scores)),
                "scores": test_scores.tolist(),
            }

        self.logger.info("Validación cruzada completada")
        return metrics

    def calculate_classification_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        unique_classes = np.unique(np.concatenate([y_true, y_pred]))

        if class_names is None:
            class_names = [f"Class_{i}" for i in unique_classes]

        cm = confusion_matrix(y_true, y_pred)

        error_analysis = {}

        for i, true_class in enumerate(unique_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"

            true_instances = np.sum(y_true == true_class)

            correct_predictions = cm[i, i]

            false_negatives = true_instances - correct_predictions

            false_positives = np.sum(cm[:, i]) - correct_predictions

            error_analysis[class_name] = {
                "true_instances": int(true_instances),
                "correct_predictions": int(correct_predictions),
                "false_negatives": int(false_negatives),
                "false_positives": int(false_positives),
                "accuracy_for_class": (
                    float(correct_predictions / true_instances)
                    if true_instances > 0
                    else 0.0
                ),
            }

        confusion_pairs = []
        for i in range(len(unique_classes)):
            for j in range(len(unique_classes)):
                if i != j and cm[i, j] > 0:
                    true_class_name = (
                        class_names[i] if i < len(class_names) else f"Class_{i}"
                    )
                    pred_class_name = (
                        class_names[j] if j < len(class_names) else f"Class_{j}"
                    )

                    confusion_pairs.append(
                        {
                            "true_class": true_class_name,
                            "predicted_class": pred_class_name,
                            "count": int(cm[i, j]),
                            "percentage": float(cm[i, j] / np.sum(cm[i, :]) * 100),
                        }
                    )

        confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

        result = {
            "per_class_errors": error_analysis,
            "most_common_confusions": confusion_pairs[:10],  # Top 10 confusiones
            "total_errors": int(np.sum(cm) - np.trace(cm)),
            "error_rate": float((np.sum(cm) - np.trace(cm)) / np.sum(cm)),
        }

        self.logger.info("Análisis de errores completado")
        return result

    def generate_classification_report_dict(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        sklearn_report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )

        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        advanced_metrics = self.calculate_advanced_metrics(y_true, y_pred)
        cm_metrics = self.calculate_confusion_matrix_metrics(y_true, y_pred)
        error_analysis = self.calculate_classification_errors(
            y_true, y_pred, class_names
        )

        complete_report = {
            "sklearn_classification_report": sklearn_report,
            "basic_metrics": basic_metrics,
            "advanced_metrics": advanced_metrics,
            "confusion_matrix_analysis": cm_metrics,
            "error_analysis": error_analysis,
            "summary": {
                "total_samples": len(y_true),
                "num_classes": len(np.unique(y_true)),
                "accuracy": basic_metrics["accuracy"],
                "weighted_f1": basic_metrics["f1_score"],
                "cohen_kappa": advanced_metrics["cohen_kappa"],
            },
        }

        self.logger.info("Reporte de clasificación completo generado")
        return complete_report

    def compare_models(
        self, models_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        comparison_df = pd.DataFrame(models_results).T

        if "accuracy" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("accuracy", ascending=False)
        else:
            comparison_df = comparison_df.sort_values(
                comparison_df.columns[0], ascending=False
            )

        comparison_df["rank"] = range(1, len(comparison_df) + 1)

        self.logger.info(f"Comparación de {len(models_results)} modelos completada")
        return comparison_df

    def calculate_model_stability(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 10,
        test_size: float = 0.2,
    ) -> Dict[str, float]:

        from sklearn.model_selection import train_test_split

        accuracies = []
        f1_scores = []

        for i in range(n_iterations):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=i
            )

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)

            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average="weighted"))

        stability_metrics = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "accuracy_cv": float(np.std(accuracies) / np.mean(accuracies)),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "f1_cv": float(np.std(f1_scores) / np.mean(f1_scores)),
            "n_iterations": n_iterations,
        }

        self.logger.info(
            f"Estabilidad del modelo calculada con {n_iterations} iteraciones"
        )
        return stability_metrics

    def print_metrics_summary(
        self, metrics_dict: Dict[str, Any], title: str = "Resumen de Métricas"
    ) -> None:

        print("=" * 60)
        print(f"{title:^60}")
        print("=" * 60)

        if "basic_metrics" in metrics_dict:
            print("\nMÉTRICAS BÁSICAS:")
            for metric, value in metrics_dict["basic_metrics"].items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

        if "advanced_metrics" in metrics_dict:
            print("\nMÉTRICAS AVANZADAS:")
            for metric, value in metrics_dict["advanced_metrics"].items():
                print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

        if "summary" in metrics_dict:
            print("\nRESUMEN:")
            for key, value in metrics_dict["summary"].items():
                if isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        print("=" * 60)
