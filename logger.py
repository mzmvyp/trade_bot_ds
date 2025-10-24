"""
Sistema de logging avançado para o trading bot
"""
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

class TradingLogger:
    """Sistema de logging estruturado para trading"""
    
    def __init__(self, name: str = "trading_system", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configurar logger principal
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remover handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Handler para arquivo de logs
        self._setup_file_handler()
        
        # Handler para console
        self._setup_console_handler()
        
        # Handler para logs estruturados (JSON)
        self._setup_json_handler()
    
    def _setup_file_handler(self):
        """Configura handler para arquivo de logs"""
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_console_handler(self):
        """Configura handler para console"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_json_handler(self):
        """Configura handler para logs estruturados em JSON"""
        json_file = self.log_dir / f"{self.name}_structured_{datetime.now().strftime('%Y%m%d')}.json"
        
        json_handler = logging.FileHandler(json_file, encoding='utf-8')
        json_handler.setLevel(logging.INFO)
        
        # Formatter customizado para JSON
        json_handler.setFormatter(JsonFormatter())
        
        self.logger.addHandler(json_handler)
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log estruturado de trades"""
        log_entry = {
            'event': 'trade_executed',
            'timestamp': datetime.now().isoformat(),
            'data': trade_data
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log estruturado de sinais"""
        log_entry = {
            'event': 'signal_generated',
            'timestamp': datetime.now().isoformat(),
            'data': signal_data
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log estruturado de erros"""
        log_entry = {
            'event': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'context': context or {}
            }
        }
        
        self.logger.error(json.dumps(log_entry, ensure_ascii=False, default=str))
    
    def log_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """Log de dados de mercado"""
        log_entry = {
            'event': 'market_data',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'data': market_data
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))
    
    def log_api_call(self, api_name: str, endpoint: str, status: str, response_time: float = None):
        """Log de chamadas de API"""
        log_entry = {
            'event': 'api_call',
            'timestamp': datetime.now().isoformat(),
            'api': api_name,
            'endpoint': endpoint,
            'status': status,
            'response_time_ms': response_time
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))

class JsonFormatter(logging.Formatter):
    """Formatter customizado para logs em JSON"""
    
    def format(self, record):
        try:
            # Se a mensagem já é JSON, retorna como está
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, TypeError):
            # Se não é JSON, cria um objeto JSON
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            return json.dumps(log_entry, ensure_ascii=False, default=str)

# Instância global do logger
trading_logger = TradingLogger()

# Funções de conveniência
def log_trade(trade_data: Dict[str, Any]):
    """Log de trade"""
    trading_logger.log_trade(trade_data)

def log_signal(signal_data: Dict[str, Any]):
    """Log de sinal"""
    trading_logger.log_signal(signal_data)

def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log de erro"""
    trading_logger.log_error(error, context)

def log_market_data(symbol: str, market_data: Dict[str, Any]):
    """Log de dados de mercado"""
    trading_logger.log_market_data(symbol, market_data)

def log_api_call(api_name: str, endpoint: str, status: str, response_time: float = None):
    """Log de chamada de API"""
    trading_logger.log_api_call(api_name, endpoint, status, response_time)
