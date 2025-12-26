"""
Módulo de logging centralizado para o sistema de trading.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# Criar diretório de logs
Path("logs").mkdir(exist_ok=True)

# Configuração do formato
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Obtém um logger configurado para o módulo especificado.

    Args:
        name: Nome do módulo (geralmente __name__)

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)

    # Evitar duplicação de handlers
    if logger.handlers:
        return logger

    # Obter nível de log do ambiente
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)

    # Handler para arquivo (log diário)
    log_file = Path(f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Sempre salvar DEBUG em arquivo
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger
