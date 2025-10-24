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
    
    # Configurações do Twitter/X
    twitter_bearer_token: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_token_secret: Optional[str] = None
    
    # Configurações do DeepSeek
    deepseek_api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    
    # Configurações do sistema
    log_level: str = "INFO"
    trading_symbol: str = "BTCUSDT"
    
    # Top 10 criptomoedas por market cap (pares USDT)
    top_crypto_pairs: list = [
        "BTCUSDT",   # Bitcoin
        "ETHUSDT",   # Ethereum
        "BNBUSDT",   # BNB
        "SOLUSDT",   # Solana
        "XRPUSDT",   # XRP
        "ADAUSDT",   # Cardano
        "DOGEUSDT",  # Dogecoin
        "AVAXUSDT",  # Avalanche
        "DOTUSDT",   # Polkadot
        "LINKUSDT"   # Chainlink
    ]
    
    @validator("deepseek_api_key")
    def validate_deepseek_api_key(cls, v):
        if not v:
            raise ValueError("DEEPSEEK_API_KEY não configurada. Configure no arquivo .env")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
