"""Objetivo: Orquestar la evaluación integral del modelo y generar reportes/visualizaciones.
Bosquejo: clase ModelEvaluator que evalúa (`evaluate_model`) computando métricas, análisis
por clase y CV; crea visualizaciones (`generate_visualizations`), guarda/carga reportes
(`save_evaluation_report`, `load_evaluation_report`), compara contra baseline
(`compare_with_baseline`), genera resúmenes (`generate_summary_report`, `print_summary`),
exporta CSV (`export_results_to_csv`) y selecciona el mejor modelo (`get_best_model_comparison`).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json
import joblib
from datetime import datetime

from .metrics_calculator import MetricsCalculator
from ..visualization.model_visualizer import ModelVisualizer
from ..utils.logger import LoggerMixin
from ..utils.config import Config


class ModelEvaluator(LoggerMixin):
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.metrics_calculator = MetricsCalculator(config)
        self.model_visualizer = ModelVisualizer(config)

        self.evaluation_results: Dict[str, Any] = {}
        self.model_name: str = ""

    def evaluate_model(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "KNN",
        class_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:

        self.model_name = model_name
        self.logger.info(f"Iniciando evaluación completa del modelo: {model_name}")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        try:
            y_test_proba = model.predict_proba(X_test)
        except:
            y_test_proba = None
            self.logger.warning("El modelo no soporta predict_proba")

        train_metrics = self.metrics_calculator.calculate_basic_metrics(
            y_train, y_train_pred
        )
        test_metrics = self.metrics_calculator.calculate_basic_metrics(
            y_test, y_test_pred
        )

        advanced_metrics = self.metrics_calculator.calculate_advanced_metrics(
            y_test, y_test_pred
        )

        per_class_metrics = self.metrics_calculator.calculate_per_class_metrics(
            y_test, y_test_pred, class_names
        )

        cm_analysis = self.metrics_calculator.calculate_confusion_matrix_metrics(
            y_test, y_test_pred
        )

        error_analysis = self.metrics_calculator.calculate_classification_errors(
            y_test, y_test_pred, class_names
        )

        cv_metrics = None
        try:
            cv_metrics = self.metrics_calculator.calculate_cross_validation_metrics(
                model, X_train, y_train, cv=self.config.CV_FOLDS
            )
        except Exception as e:
            self.logger.warning(f"Validación cruzada falló: {str(e)}")

        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(
                zip(
                    feature_names or [f"feature_{i}" for i in range(X_train.shape[1])],
                    model.feature_importances_,
                )
            )
        elif feature_names:
            try:
                from sklearn.inspection import permutation_importance

                perm_importance = permutation_importance(
                    model,
                    X_test,
                    y_test,
                    n_repeats=10,
                    random_state=self.config.RANDOM_STATE,
                )

                importances = getattr(perm_importance, "importances_mean", None)

                if importances is None:
                    feature_importance = dict(
                        zip(feature_names, [0.0] * len(feature_names))
                    )
                else:
                    import numpy as np

                    imp = np.nan_to_num(importances, nan=0.0)
                    imp = np.maximum(imp, 0.0)
                    total = float(np.sum(imp))
                    if total > 0:
                        imp = imp / total
                    feature_importance = dict(
                        zip(feature_names, [float(v) for v in imp])
                    )
            except Exception as e:
                self.logger.warning(f"Cálculo de importancia falló: {str(e)}")

        self.evaluation_results = {
            "model_info": {
                "name": model_name,
                "type": type(model).__name__,
                "parameters": (
                    model.get_params() if hasattr(model, "get_params") else {}
                ),
                "evaluation_date": datetime.now().isoformat(),
            },
            "data_info": {
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "num_features": X_train.shape[1],
                "num_classes": len(np.unique(y_train)),
                "class_names": class_names,
                "feature_names": feature_names,
            },
            "performance_metrics": {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "advanced_metrics": advanced_metrics,
                "overfitting_score": train_metrics["accuracy"]
                - test_metrics["accuracy"],
            },
            "detailed_analysis": {
                "per_class_metrics": per_class_metrics.to_dict("records"),
                "confusion_matrix_analysis": cm_analysis,
                "error_analysis": error_analysis,
                "cross_validation": cv_metrics,
            },
            "feature_analysis": {"feature_importance": feature_importance},
            "predictions": {
                "y_test_true": y_test.astype("int64").tolist(),
                "y_test_pred": y_test_pred.astype("int64").tolist(),
                "y_test_proba": (
                    y_test_proba.tolist() if y_test_proba is not None else None
                ),
            },
        }

        self.logger.info("Evaluación completa finalizada")
        return self.evaluation_results

    def generate_visualizations(
        self, save_dir: Optional[str] = None, show_plots: bool = False
    ) -> Dict[str, str]:
        if not self.evaluation_results:
            raise ValueError(
                "No hay resultados de evaluación. Ejecute evaluate_model() primero."
            )

        save_dir_path = None
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)

        y_true = np.array(self.evaluation_results["predictions"]["y_test_true"])
        y_pred = np.array(self.evaluation_results["predictions"]["y_test_pred"])
        y_proba = self.evaluation_results["predictions"]["y_test_proba"]
        class_names = self.evaluation_results["data_info"]["class_names"]

        saved_files = self.model_visualizer.create_model_report(
            y_true,
            y_pred,
            y_proba=np.array(y_proba) if y_proba else None,
            class_names=class_names,
            model_name=self.model_name,
            save_dir=str(save_dir_path) if save_dir_path else None,
        )

        if self.evaluation_results["feature_analysis"]["feature_importance"]:
            importance_path = (
                str(save_dir_path / f"{self.model_name}_feature_importance.png")
                if save_dir_path
                else None
            )
            self.model_visualizer.plot_feature_importance(
                self.evaluation_results["feature_analysis"]["feature_importance"],
                save_path=importance_path,
                show=show_plots,
            )
            if importance_path:
                saved_files["feature_importance"] = importance_path

        self.logger.info(f"Visualizaciones generadas: {len(saved_files)} archivos")
        return saved_files

    def save_evaluation_report(
        self, filepath: str, include_predictions: bool = False
    ) -> None:
        if not self.evaluation_results:
            raise ValueError("No hay resultados de evaluación para guardar.")

        report_data = self.evaluation_results.copy()

        if not include_predictions:
            report_data.pop("predictions", None)

        if (
            "detailed_analysis" in report_data
            and "per_class_metrics" in report_data["detailed_analysis"]
        ):
            pass

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Reporte de evaluación guardado en: {filepath}")

    def load_evaluation_report(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            self.evaluation_results = json.load(f)

        self.model_name = self.evaluation_results.get("model_info", {}).get(
            "name", "Unknown"
        )

        self.logger.info(f"Reporte de evaluación cargado desde: {filepath}")
        return self.evaluation_results

    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        if not self.evaluation_results:
            raise ValueError("No hay resultados de evaluación actuales.")

        current_metrics = self.evaluation_results["performance_metrics"]["test_metrics"]
        baseline_metrics = baseline_results["performance_metrics"]["test_metrics"]

        comparison = {
            "model_names": {
                "current": self.evaluation_results["model_info"]["name"],
                "baseline": baseline_results["model_info"]["name"],
            },
            "metric_comparison": {},
            "improvement": {},
            "winner": {},
        }

        for metric in current_metrics:
            if metric in baseline_metrics:
                current_value = current_metrics[metric]
                baseline_value = baseline_metrics[metric]

                comparison["metric_comparison"][metric] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "difference": current_value - baseline_value,
                    "percentage_change": (
                        ((current_value - baseline_value) / baseline_value * 100)
                        if baseline_value != 0
                        else 0
                    ),
                }

                comparison["improvement"][metric] = current_value > baseline_value
                comparison["winner"][metric] = (
                    "current" if current_value > baseline_value else "baseline"
                )

        if "accuracy" in comparison["improvement"]:
            comparison["overall_winner"] = (
                "current" if comparison["improvement"]["accuracy"] else "baseline"
            )

        self.logger.info("Comparación con baseline completada")
        return comparison

    def generate_summary_report(self) -> str:
        if not self.evaluation_results:
            raise ValueError("No hay resultados de evaluación.")

        model_info = self.evaluation_results["model_info"]
        data_info = self.evaluation_results["data_info"]
        test_metrics = self.evaluation_results["performance_metrics"]["test_metrics"]
        train_metrics = self.evaluation_results["performance_metrics"]["train_metrics"]

        summary = f"""
{'='*80}
REPORTE DE EVALUACIÓN - {model_info['name'].upper()}
{'='*80}

INFORMACIÓN DEL MODELO:
  Tipo: {model_info['type']}
  Fecha de evaluación: {model_info['evaluation_date']}

INFORMACIÓN DE DATOS:
  Muestras de entrenamiento: {data_info['train_samples']:,}
  Muestras de prueba: {data_info['test_samples']:,}
  Número de características: {data_info['num_features']}
  Número de clases: {data_info['num_classes']}

MÉTRICAS DE RENDIMIENTO:
  Accuracy (Entrenamiento): {train_metrics['accuracy']:.4f}
  Accuracy (Prueba): {test_metrics['accuracy']:.4f}
  Precision (Ponderada): {test_metrics['precision']:.4f}
  Recall (Ponderado): {test_metrics['recall']:.4f}
  F1-Score (Ponderado): {test_metrics['f1_score']:.4f}

ANÁLISIS DE SOBREAJUSTE:
  Diferencia Train-Test: {self.evaluation_results['performance_metrics']['overfitting_score']:.4f}
  Estado: {'Posible sobreajuste' if self.evaluation_results['performance_metrics']['overfitting_score'] > 0.05 else 'Generalización adecuada'}

ERRORES MÁS COMUNES:
"""

        common_errors = self.evaluation_results["detailed_analysis"]["error_analysis"][
            "most_common_confusions"
        ]
        for i, error in enumerate(common_errors[:3], 1):
            summary += f"  {i}. {error['true_class']} → {error['predicted_class']}: {error['count']} casos ({error['percentage']:.1f}%)\n"

        summary += f"\n{'='*80}"

        return summary

    def print_summary(self) -> None:
        print(self.generate_summary_report())

    def get_best_model_comparison(
        self, models_results: List[Dict[str, Any]], metric: str = "accuracy"
    ) -> Dict[str, Any]:
        best_model = None
        best_score = -1

        comparison_data = []

        for result in models_results:
            test_metrics = result["performance_metrics"]["test_metrics"]
            if metric in test_metrics:
                score = test_metrics[metric]

                comparison_data.append(
                    {
                        "model_name": result["model_info"]["name"],
                        "model_type": result["model_info"]["type"],
                        metric: score,
                        "train_accuracy": result["performance_metrics"][
                            "train_metrics"
                        ]["accuracy"],
                        "overfitting": result["performance_metrics"][
                            "overfitting_score"
                        ],
                    }
                )

                if score > best_score:
                    best_score = score
                    best_model = result

        return {
            "best_model": best_model,
            "best_score": best_score,
            "comparison_table": pd.DataFrame(comparison_data).sort_values(
                metric, ascending=False
            ),
            "ranking_metric": metric,
        }

    def export_results_to_csv(self, filepath: str) -> None:
        if not self.evaluation_results:
            raise ValueError("No hay resultados de evaluación.")

        data = []

        model_info = self.evaluation_results["model_info"]
        test_metrics = self.evaluation_results["performance_metrics"]["test_metrics"]
        train_metrics = self.evaluation_results["performance_metrics"]["train_metrics"]

        row = {
            "model_name": model_info["name"],
            "model_type": model_info["type"],
            "evaluation_date": model_info["evaluation_date"],
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "overfitting_score": self.evaluation_results["performance_metrics"][
                "overfitting_score"
            ],
        }

        advanced_metrics = self.evaluation_results["performance_metrics"][
            "advanced_metrics"
        ]
        row.update(advanced_metrics)

        data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        self.logger.info(f"Resultados exportados a CSV: {filepath}")
