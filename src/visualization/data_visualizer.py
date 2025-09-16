"""Objetivo: Visualizar relaciones y patrones en datos (correlaciones, etc.). Bosquejo:
clase DataVisualizer que configura estilo (`setup_style`) y provee gráficos de matriz de
correlación (`plot_correlation_matrix`) y análisis cuantitativo de correlaciones
(`analyze_correlations`). Utiliza `Config` para tamaños y `LoggerMixin` para avisos."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class DataVisualizer(LoggerMixin):

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.setup_style()

    def setup_style(self):
        plt.style.use("default")
        sns.set_palette("husl")

        plt.rcParams["figure.figsize"] = self.config.FIGURE_SIZE
        plt.rcParams["figure.dpi"] = self.config.DPI
        plt.rcParams["savefig.dpi"] = self.config.DPI

    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            self.log_warning_with_context(
                "No se encontraron variables numéricas para correlación",
                "plot_correlation_matrix",
            )
            return

        correlation_matrix = data[numeric_cols].corr()

        if correlation_matrix.isna().all().all():
            self.log_warning_with_context(
                "La matriz de correlación contiene solo valores NaN",
                "plot_correlation_matrix",
            )
            return

        for col in numeric_cols:
            unique_vals = data[col].nunique()
            if unique_vals <= 1:
                self.log_warning_with_context(
                    f"Variable '{col}' tiene {unique_vals} valor único - no puede mostrar correlaciones",
                    "plot_correlation_matrix",
                )
            elif unique_vals < 5:
                self.log_warning_with_context(
                    f"Variable '{col}' tiene solo {unique_vals} valores únicos - correlaciones limitadas",
                    "plot_correlation_matrix",
                )

        plt.figure(figsize=(12, 10))

        plt.figtext(
            0.5,
            0.98,
            "OBJETIVO: Identificar relaciones lineales entre variables para entender dependencias.\n"
            "INTERPRETACION: Colores rojos indican correlacion positiva, azules negativa.\n"
            "Valores cercanos a ±1.0 muestran relaciones fuertes entre variables.",
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.9),
        )

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            mask=mask,
            fmt=".2f",
            linewidths=0.5,
            vmin=-1,
            vmax=1,
        )

        plt.title(
            "MATRIZ DE CORRELACION - RELACIONES ENTRE VARIABLES",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Variables", fontweight="bold")
        plt.ylabel("Variables", fontweight="bold")

        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Matriz de correlación guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def analyze_correlations(
        self, data: pd.DataFrame, threshold: float = 0.5
    ) -> Dict[str, Any]:

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return {"strong_correlations": [], "correlation_matrix": None}

        correlation_matrix = data[numeric_cols].corr()

        strong_corrs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if not np.isnan(corr_val) and abs(corr_val) > threshold:
                    var1 = correlation_matrix.columns[i]
                    var2 = correlation_matrix.columns[j]
                    strong_corrs.append(
                        {
                            "var1": var1,
                            "var2": var2,
                            "correlation": corr_val,
                            "direction": "positiva" if corr_val > 0 else "negativa",
                        }
                    )

        strong_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "strong_correlations": strong_corrs,
            "correlation_matrix": correlation_matrix,
            "numeric_columns": list(numeric_cols),
        }
