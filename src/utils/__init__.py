"""Objetivo: Reunir utilidades comunes como configuración y logging. Bosquejo: expone
`Config` y `setup_logger` para consumo desde otros módulos."""

from .config import Config
from .logger import setup_logger

__all__ = ["Config", "setup_logger"]
