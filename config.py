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
    max_risk_per_trade: float = 0.03  # Máximo 3% de risco por trade
    max_drawdown: float = 0.15  # Máximo 15% de drawdown
    max_exposure: float = 0.10  # Máximo 10% de exposição total
    max_daily_trades: int = 5  # Máximo 5 trades por dia
    base_risk_percentage: float = 0.02  # 2% base de risco
    
    # Top 3 criptomoedas para análise (BTC, ETH, SOL)
    top_crypto_pairs: list = [
        "BTCUSDT",   # Bitcoin
        "ETHUSDT",   # Ethereum
        "SOLUSDT"    # Solana
    ]
    
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignorar campos extras

settings = Settings()
