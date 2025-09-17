"""Objetivo: Proveer utilidades de logging consistentes para todo el proyecto. Bosquejo:
función `setup_logger` para configurar loggers por nombre y clase `LoggerMixin` que
inyecta un logger por clase y ofrece helpers estructurados (`log_step`, `log_result`,
`log_file_operation`, `log_data_info`, `log_warning_with_context`, `log_error_with_context`).
"""

import logging
import sys
from typing import Optional, Dict, Any


def setup_logger(
    name: str = "ml_pipeline",
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> logging.Logger:

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


class LoggerMixin:

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = setup_logger(
                name=f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger

    def log_step(self, step_name: str, description: str, **kwargs):
        separator = "=" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"🔄 PASO: {step_name}")
        self.logger.info(f"📋 OBJETIVO: {description}")

        if kwargs:
            self.logger.info("📊 PARÁMETROS:")
            for key, value in kwargs.items():
                self.logger.info(f"   • {key}: {value}")

        self.logger.info(f"{separator}")

    def log_result(
        self, operation: str, result: str, metrics: Optional[Dict[str, Any]] = None
    ):
        self.logger.info(f"✅ {operation.upper()}: {result}")

        if metrics:
            self.logger.info("📈 MÉTRICAS OBTENIDAS:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"   • {key}: {value:.4f}")
                else:
                    self.logger.info(f"   • {key}: {value}")

    def log_file_operation(self, operation: str, file_path: str, details: str = ""):
        self.logger.info(f"💾 {operation}: {file_path}")
        if details:
            self.logger.info(f"   ℹ️  {details}")

    def log_data_info(
        self, data_name: str, shape: tuple, details: Optional[Dict[str, Any]] = None
    ):

        self.logger.info(f"📊 DATOS PROCESADOS: {data_name}")
        self.logger.info(f"   • Forma: {shape[0]:,} filas × {shape[1]} columnas")

        if details:
            for key, value in details.items():
                if isinstance(value, (int, float)):
                    formatted_value = (
                        f"{value:,}" if isinstance(value, int) else f"{value:.4f}"
                    )
                    self.logger.info(f"   • {key}: {formatted_value}")
                else:
                    self.logger.info(f"   • {key}: {value}")

    def log_warning_with_context(
        self, warning: str, context: str, suggestion: str = ""
    ):

        self.logger.warning(f"⚠️  {warning}")
        self.logger.warning(f"   📍 Contexto: {context}")
        if suggestion:
            self.logger.warning(f"   💡 Sugerencia: {suggestion}")

    def log_error_with_context(self, error: str, context: str, solution: str = ""):

        self.logger.error(f"❌ ERROR: {error}")
        self.logger.error(f"   📍 Contexto: {context}")
        if solution:
            self.logger.error(f"   🔧 Solución: {solution}")
