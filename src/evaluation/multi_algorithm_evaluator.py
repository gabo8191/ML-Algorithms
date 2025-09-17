"""Evaluador para comparar múltiples algoritmos de ML.
Ejecuta evaluación completa y genera comparaciones entre diferentes algoritmos."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
import time
import math
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.cm as cm  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import confusion_matrix

from .model_evaluator import ModelEvaluator
from ..utils.logger import LoggerMixin
from ..utils.config import Config
from ..models.base_classifier import BaseClassifier


class MultiAlgorithmEvaluator(LoggerMixin):
    """Evaluador para comparar múltiples algoritmos de ML"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.results = {}
        self.comparison_results = {}

    def evaluate_algorithm(
        self,
        algorithm_name: str,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluar un algoritmo específico"""

        self.logger.info(f"Evaluando algoritmo: {algorithm_name}")
        start_time = time.time()

        # Crear evaluador específico para este algoritmo
        evaluator = ModelEvaluator(self.config)

        # Realizar evaluación completa
        evaluation_results = evaluator.evaluate_model(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name=f"{algorithm_name}_Coffee_Shop_Classifier",
            class_names=class_names or ["Not Successful", "Successful"],
            feature_names=feature_names,
        )

        # Calcular tiempo de evaluación
        evaluation_time = time.time() - start_time

        # Agregar información específica del algoritmo
        algorithm_results = {
            "algorithm_name": algorithm_name,
            "evaluation_time": evaluation_time,
            "evaluation_results": evaluation_results,
            "model_summary": (
                model.get_model_summary() if hasattr(model, "get_model_summary") else {}
            ),
        }

        # Guardar resultados específicos del algoritmo
        self._save_algorithm_results(algorithm_name, algorithm_results, evaluator)

        # Almacenar en resultados globales
        self.results[algorithm_name] = algorithm_results

        self.logger.info(
            f"Evaluación de {algorithm_name} completada en {evaluation_time:.2f}s"
        )
        return algorithm_results

    def _save_algorithm_results(
        self, algorithm_name: str, results: Dict[str, Any], evaluator: ModelEvaluator
    ) -> None:
        """Guardar resultados específicos de un algoritmo"""

        algorithm_dir = Path(self.config.RESULTS_PATH) / algorithm_name.lower()
        algorithm_dir.mkdir(exist_ok=True)

        # Guardar reporte de evaluación
        report_path = algorithm_dir / f"{algorithm_name.lower()}_evaluation_report.json"
        evaluator.save_evaluation_report(str(report_path), include_predictions=False)

        # Guardar métricas en CSV
        csv_path = algorithm_dir / f"{algorithm_name.lower()}_metrics.csv"
        evaluator.export_results_to_csv(str(csv_path))

        # Generar visualizaciones específicas del algoritmo
        viz_dir = algorithm_dir / "visualizations"
        viz_files = evaluator.generate_visualizations(str(viz_dir), show_plots=False)

        # Generar visualizaciones adicionales específicas del algoritmo
        self._generate_algorithm_specific_visualizations(
            algorithm_name, results, evaluator, algorithm_dir
        )

        self.logger.info(
            f"Resultados de {algorithm_name} guardados en: {algorithm_dir}"
        )

    def _generate_algorithm_specific_visualizations(
        self,
        algorithm_name: str,
        results: Dict[str, Any],
        evaluator: ModelEvaluator,
        algorithm_dir: Path,
    ) -> None:
        """Generar visualizaciones específicas de cada algoritmo"""

        # Obtener el modelo del algoritmo
        model = results.get("evaluation_results", {}).get("model")
        if not model:
            return

        # Obtener datos de prueba
        X_test = results.get("evaluation_results", {}).get("X_test")
        y_test = results.get("evaluation_results", {}).get("y_test")
        y_pred = results.get("evaluation_results", {}).get("y_pred")

        if X_test is None or y_test is None or y_pred is None:
            return

        # Generar matriz de confusión específica del algoritmo
        self._generate_confusion_matrix(algorithm_name, y_test, y_pred, algorithm_dir)

        # Generar feature importance específica del algoritmo
        self._generate_feature_importance(algorithm_name, model, algorithm_dir)

    def _generate_confusion_matrix(
        self,
        algorithm_name: str,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        algorithm_dir: Path,
    ) -> None:
        """Generar matriz de confusión para un algoritmo específico"""

        # Crear matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Crear visualización
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"{algorithm_name} - Matriz de Confusión")
        plt.colorbar()

        # Añadir etiquetas
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No Exitoso", "Exitoso"])
        plt.yticks(tick_marks, ["No Exitoso", "Exitoso"])

        # Añadir valores en las celdas
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("Etiqueta Real")
        plt.xlabel("Predicción")
        plt.tight_layout()

        # Guardar en la raíz de results/ como se hacía antes
        confusion_path = (
            Path(self.config.RESULTS_PATH)
            / f"{algorithm_name}_Coffee_Shop_Classifier_confusion_matrix.png"
        )
        plt.savefig(str(confusion_path), dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(
            f"Matriz de confusión de {algorithm_name} guardada en: {confusion_path}"
        )

    def _generate_feature_importance(
        self, algorithm_name: str, model: BaseClassifier, algorithm_dir: Path
    ) -> None:
        """Generar feature importance para un algoritmo específico"""

        # Obtener feature importance
        feature_importance = model.get_feature_importance()
        if feature_importance is None:
            self.logger.warning(
                f"No se pudo obtener feature importance para {algorithm_name}"
            )
            return

        # Obtener nombres de características
        feature_names = getattr(
            model,
            "feature_names",
            [f"Feature_{i}" for i in range(len(feature_importance))],
        )

        # Crear visualización
        plt.figure(figsize=(10, 6))

        # Ordenar por importancia
        indices = np.argsort(feature_importance)[::-1]

        # Crear gráfico de barras
        bars = plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.title(f"{algorithm_name} - Importancia de Características")
        plt.xlabel("Características")
        plt.ylabel("Importancia")

        # Añadir nombres de características
        plt.xticks(
            range(len(feature_importance)),
            [feature_names[i] for i in indices],
            rotation=45,
        )

        # Añadir valores en las barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.001,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        # Guardar en la raíz de results/ como se hacía antes
        importance_path = (
            Path(self.config.RESULTS_PATH)
            / f"{algorithm_name}_Coffee_Shop_Classifier_feature_importance.png"
        )
        plt.savefig(str(importance_path), dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(
            f"Feature importance de {algorithm_name} guardada en: {importance_path}"
        )

    def compare_algorithms(self) -> Dict[str, Any]:
        """Comparar todos los algoritmos evaluados"""

        if len(self.results) < 2:
            raise ValueError(
                "Se necesitan al menos 2 algoritmos para hacer comparación"
            )

        self.logger.info("Iniciando comparación de algoritmos")

        # Extraer métricas de todos los algoritmos
        comparison_data = []

        for algorithm_name, results in self.results.items():
            metrics = results["evaluation_results"]["performance_metrics"][
                "test_metrics"
            ]
            model_summary = results.get("model_summary", {})

            row = {
                "Algorithm": algorithm_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics.get("precision", float("nan")),
                "Recall": metrics.get("recall", float("nan")),
                "F1_Score": metrics.get("f1_score", float("nan")),
                "AUC_ROC": metrics.get("auc_roc", float("nan")),
                "Average_Precision": metrics.get("average_precision", float("nan")),
                "Evaluation_Time": results["evaluation_time"],
                "Best_CV_Score": model_summary.get("best_cv_score", 0.0),
            }
            comparison_data.append(row)

        # Crear DataFrame para facilitar análisis
        comparison_df = pd.DataFrame(comparison_data)

        # Calcular rankings
        metrics_to_rank = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "AUC_ROC",
            "Average_Precision",
            "Best_CV_Score",
        ]
        for metric in metrics_to_rank:
            # Ranquear manejando NaNs como peor (colocándolos al final)
            series = comparison_df[metric]
            ranks = series.rank(ascending=False, method="min", na_option="bottom")
            comparison_df[f"{metric}_Rank"] = ranks

        # Calcular ranking promedio
        rank_columns = [col for col in comparison_df.columns if col.endswith("_Rank")]
        comparison_df["Average_Rank"] = comparison_df[rank_columns].mean(axis=1)
        comparison_df["Overall_Rank"] = comparison_df["Average_Rank"].rank(method="min")

        # Identificar mejor algoritmo
        best_algorithm = comparison_df.loc[comparison_df["Overall_Rank"].idxmin()]

        self.comparison_results = {
            "comparison_table": comparison_df.to_dict("records"),
            "best_algorithm": {
                "name": best_algorithm["Algorithm"],
                "metrics": {
                    "accuracy": best_algorithm["Accuracy"],
                    "precision": best_algorithm["Precision"],
                    "recall": best_algorithm["Recall"],
                    "f1_score": best_algorithm["F1_Score"],
                    "auc_roc": best_algorithm["AUC_ROC"],
                },
                "overall_rank": best_algorithm["Overall_Rank"],
            },
            "summary_statistics": {
                "mean_accuracy": comparison_df["Accuracy"].mean(),
                "std_accuracy": comparison_df["Accuracy"].std(),
                "max_accuracy": comparison_df["Accuracy"].max(),
                "min_accuracy": comparison_df["Accuracy"].min(),
                "total_evaluation_time": comparison_df["Evaluation_Time"].sum(),
            },
        }

        self.logger.info(
            f"Mejor algoritmo: {best_algorithm['Algorithm']} (Accuracy: {best_algorithm['Accuracy']:.4f})"
        )

        # Export comparison table as CSV for deliverable
        comparison_csv = os.path.join(
            self.config.RESULTS_PATH, "algorithm_comparison_metrics.csv"
        )
        try:
            comparison_df.to_csv(comparison_csv, index=False)
            self.logger.info(f"Tabla de comparación exportada a: {comparison_csv}")
        except Exception as export_err:
            self.logger.warning(
                f"No se pudo exportar la tabla de comparación: {export_err}"
            )
        return self.comparison_results

    def generate_comparison_visualizations(
        self, save_dir: Optional[str] = None, show_plots: bool = False
    ) -> Dict[str, str]:
        """Generar visualizaciones comparativas"""

        if not self.comparison_results:
            raise ValueError("Debe ejecutar compare_algorithms() primero")

        save_dir = save_dir or str(Path(self.config.RESULTS_PATH) / "comparisons")
        os.makedirs(save_dir, exist_ok=True)

        comparison_df = pd.DataFrame(self.comparison_results["comparison_table"])
        viz_files = {}

        # 1. Gráfico de barras de métricas principales
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Algorithm Performance Comparison - Coffee Shop Success",
            fontsize=16,
            fontweight="bold",
        )

        metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
        colors = ["skyblue", "lightgreen", "lightcoral", "lightsalmon"]

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(
                comparison_df["Algorithm"],
                comparison_df[metric],
                color=color,
                alpha=0.7,
            )
            ax.set_title(f"{metric} by Algorithm")
            ax.set_ylabel(f"{metric} (higher is better)")
            ax.tick_params(axis="x", rotation=45)

            # Añadir valores en las barras
            for bar, value in zip(bars, comparison_df[metric]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        metrics_path = os.path.join(save_dir, "metrics_comparison.png")
        plt.savefig(str(metrics_path), dpi=300, bbox_inches="tight")
        viz_files["metrics_comparison"] = metrics_path

        if show_plots:
            plt.show()
        else:
            plt.close()

        # 2. Heatmap de rankings
        plt.figure(figsize=(12, 8))
        rank_columns = [col for col in comparison_df.columns if col.endswith("_Rank")]
        rank_data = comparison_df[["Algorithm"] + rank_columns].set_index("Algorithm")

        # Limpiar nombres de columnas
        rank_data.columns = [col.replace("_Rank", "") for col in rank_data.columns]

        sns.heatmap(
            rank_data.T,
            annot=True,
            cmap="RdYlGn_r",
            center=3,
            cbar_kws={"label": "Rank (1=Best)"},
            fmt=".0f",
        )
        plt.title("Algorithm Rankings Heatmap - Coffee Shop Success")
        plt.ylabel("Metrics (1 = Best)")
        plt.xlabel("Algorithms")

        rankings_path = os.path.join(save_dir, "rankings_heatmap.png")
        plt.savefig(rankings_path, dpi=300, bbox_inches="tight")
        viz_files["rankings_heatmap"] = rankings_path

        if show_plots:
            plt.show()
        else:
            plt.close()

        # 3. Gráfico de radar para comparación multidimensional
        self._create_radar_chart(comparison_df, save_dir, show_plots)
        viz_files["radar_chart"] = os.path.join(save_dir, "radar_comparison.png")

        # 4. Tiempo de evaluación vs Accuracy
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            comparison_df["Evaluation_Time"],
            comparison_df["Accuracy"],
            s=100,
            alpha=0.7,
            c=range(len(comparison_df)),
            cmap="viridis",
        )

        # Añadir etiquetas
        for i, row in comparison_df.iterrows():
            plt.annotate(
                row["Algorithm"],
                (row["Evaluation_Time"], row["Accuracy"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        plt.xlabel("Evaluation Time (s)")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Evaluation Time - Coffee Shop Success")
        plt.grid(True, alpha=0.3)

        time_accuracy_path = os.path.join(save_dir, "time_vs_accuracy.png")
        plt.savefig(time_accuracy_path, dpi=300, bbox_inches="tight")
        viz_files["time_vs_accuracy"] = time_accuracy_path

        if show_plots:
            plt.show()
        else:
            plt.close()

        self.logger.info(
            f"Visualizaciones comparativas generadas: {len(viz_files)} archivos"
        )
        return viz_files

    def _create_radar_chart(
        self,
        comparison_df: pd.DataFrame,
        save_dir: Optional[str],
        show_plots: bool = False,
    ):
        """Crear gráfico de radar para comparación multidimensional"""

        metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC"]

        # Configurar el gráfico de radar
        angles = [n / len(metrics) * 2 * math.pi for n in range(len(metrics))]
        angles += angles[:1]  # Completar el círculo

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        colors = cm.get_cmap("Set3")(np.linspace(0, 1, len(comparison_df)))

        for i, (_, row) in enumerate(comparison_df.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Completar el círculo

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=row["Algorithm"],
                color=colors[i],
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # Configurar etiquetas
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(
            "Algorithm Performance Radar Chart", size=16, fontweight="bold", pad=20
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)

        if save_dir is not None:
            radar_path = os.path.join(save_dir, "radar_comparison.png")
            plt.savefig(str(radar_path), dpi=300, bbox_inches="tight")
        else:
            radar_path = None

        if show_plots:
            plt.show()
        else:
            plt.close()

        return radar_path

    def save_comparison_report(self, save_path: Optional[str] = None) -> None:
        """Guardar reporte completo de comparación"""

        if not self.comparison_results:
            raise ValueError("Debe ejecutar compare_algorithms() primero")

        save_path = save_path or os.path.join(
            self.config.RESULTS_PATH, "algorithm_comparison_report.json"
        )

        # Preparar reporte completo
        full_report = {
            "comparison_summary": self.comparison_results,
            "individual_results": self.results,
            "evaluation_metadata": {
                "dataset_name": "Coffee Shop Revenue",
                "target_variable": "Successful",
                "n_algorithms_compared": len(self.results),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_used": {
                    **self.config.get_hyperparameter_config(),
                    **self.config.get_preprocessing_config(),
                },
            },
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"Reporte de comparación guardado en: {save_path}")

    def print_comparison_summary(self) -> None:
        """Imprimir resumen de la comparación"""

        if not self.comparison_results:
            print(
                "No hay resultados de comparación disponibles. Ejecute compare_algorithms() primero."
            )
            return

        print("\n" + "=" * 80)
        print("RESUMEN DE COMPARACIÓN DE ALGORITMOS")
        print("=" * 80)

        best_algo = self.comparison_results["best_algorithm"]
        print(f"\n🏆 MEJOR ALGORITMO: {best_algo['name']}")
        print(f"   • Accuracy: {best_algo['metrics']['accuracy']:.4f}")
        print(f"   • Precision: {best_algo['metrics']['precision']:.4f}")
        print(f"   • Recall: {best_algo['metrics']['recall']:.4f}")
        print(f"   • F1-Score: {best_algo['metrics']['f1_score']:.4f}")

        print(f"\n📊 ESTADÍSTICAS GENERALES:")
        stats = self.comparison_results["summary_statistics"]
        print(
            f"   • Accuracy promedio: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}"
        )
        print(f"   • Mejor accuracy: {stats['max_accuracy']:.4f}")
        print(f"   • Peor accuracy: {stats['min_accuracy']:.4f}")
        print(f"   • Tiempo total de evaluación: {stats['total_evaluation_time']:.2f}s")

        print(f"\n📋 RANKING COMPLETO:")
        comparison_df = pd.DataFrame(self.comparison_results["comparison_table"])
        comparison_df = comparison_df.sort_values("Overall_Rank")

        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            print(
                f"   {i:2d}. {row['Algorithm']:<20} - Accuracy: {row['Accuracy']:.4f} (Tiempo: {row['Evaluation_Time']:.2f}s)"
            )

        print("\n" + "=" * 80)
