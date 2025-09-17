"""Objetivo: Centralizar evaluación y cálculo de métricas. Incluye evaluación individual
y comparación de múltiples algoritmos de ML.
"""

from .model_evaluator import ModelEvaluator
from .metrics_calculator import MetricsCalculator
from .multi_algorithm_evaluator import MultiAlgorithmEvaluator

__all__ = ["ModelEvaluator", "MetricsCalculator", "MultiAlgorithmEvaluator"]
