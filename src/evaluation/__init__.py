"""Objetivo: Centralizar evaluación y cálculo de métricas. Bosquejo: expone
`ModelEvaluator` y `MetricsCalculator` para evaluación integral y métricas detalladas.
"""

from .model_evaluator import ModelEvaluator
from .metrics_calculator import MetricsCalculator

__all__ = ["ModelEvaluator", "MetricsCalculator"]
