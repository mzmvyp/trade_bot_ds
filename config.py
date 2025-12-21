"""
Configurações do sistema de trading
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    # Configurações da API Binance (pública - não precisa de chaves)
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None
    
    # Twitter removido - análise de sentimento baseada apenas em dados de mercado
    
    # Configurações do DeepSeek
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    
    # Configurações do sistema
    log_level: str = "INFO"
    trading_symbol: str = "BTCUSDT"
    
    # Configurações de Risk Management
    max_risk_per_trade: float = 0.05  # Máximo 5% de risco por trade (aumentado para paper trading)
    max_drawdown: float = 0.40  # Máximo 40% de drawdown (aumentado para paper trading)
    max_exposure: float = 0.80  # Máximo 80% de exposição total (aumentado para paper trading)
    max_daily_trades: int = 5  # Máximo 5 trades por dia
    base_risk_percentage: float = 0.02  # 2% base de risco
    
    # Configurações de Confiança
    # UNIFICADO: Sempre usar escala 0-10
    min_confidence_0_10: int = 7  # Mínimo 7/10 para executar sinais
    # DEPRECATED: min_confidence_0_5 removido - sempre usar escala 0-10
    
    # Configurações de Intervalo de Análise
    min_analysis_interval_hours: float = 1.0  # Mínimo 1 hora entre análises do mesmo símbolo
    
    # Top 10 criptomoedas para análise
    top_crypto_pairs: list = [
        "BTCUSDT",   # Bitcoin
        "ETHUSDT",   # Ethereum
        "SOLUSDT",   # Solana
        "BNBUSDT",   # Binance Coin
        "ADAUSDT",   # Cardano
        "XRPUSDT",   # Ripple
        "DOGEUSDT",  # Dogecoin
        "AVAXUSDT",  # Avalanche
        "DOTUSDT",   # Polkadot
        "LINKUSDT"   # Chainlink
    ]

    # ========================================
    # TIMEOUT POR TIPO DE OPERAÇÃO
    # ========================================
    timeout_scalp_hours: float = 0.5        # 30 minutos
    timeout_day_trade_hours: float = 8.0    # 8 horas
    timeout_swing_trade_hours: float = 120.0  # 5 dias
    timeout_position_trade_hours: float = 672.0  # 28 dias

    # ========================================
    # REAVALIAÇÃO DE SINAIS ATIVOS
    # ========================================
    reevaluation_enabled: bool = True
    reevaluation_interval_hours: float = 2.0  # Reavaliar a cada 2 horas
    reevaluation_min_time_open_hours: float = 1.0  # Só reavaliar após 1h aberta
    reevaluation_min_confidence: int = 7  # Confiança mínima para agir na reavaliação


    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignorar campos extras

settings = Settings()
