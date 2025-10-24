"""
Cliente para integração com a API da Binance
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
import json
from config import settings

class BinanceClient:
    def __init__(self):
        # API pública da Binance Futures - não precisa de autenticação
        self.base_url = "https://fapi.binance.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """
        Obtém dados de candlesticks da Binance
        """
        url = f"{self.base_url}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Converter para tipos numéricos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    async def get_ticker_24hr(self, symbol: str) -> Dict:
        """
        Obtém estatísticas de 24h para um símbolo
        """
        url = f"{self.base_url}/fapi/v1/ticker/24hr"
        params = {'symbol': symbol}
        
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Obtém o livro de ordens
        """
        url = f"{self.base_url}/fapi/v1/depth"
        params = {'symbol': symbol, 'limit': limit}
        
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_funding_rate(self, symbol: str) -> Dict:
        """
        Obtém a taxa de funding atual
        """
        url = f"{self.base_url}/fapi/v1/premiumIndex"
        params = {'symbol': symbol}
        
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_open_interest(self, symbol: str) -> Dict:
        """
        Obtém o interesse aberto
        """
        url = f"{self.base_url}/fapi/v1/openInterest"
        params = {'symbol': symbol}
        
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_historical_klines(self, symbol: str, interval: str, 
                                  start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Obtém dados históricos de klines para backtesting
        
        Args:
            symbol: Símbolo do par
            interval: Intervalo dos candles
            start_time: Data inicial
            end_time: Data final
            
        Returns:
            DataFrame com dados históricos
        """
        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        while current_start < end_timestamp:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': min(current_start + 100 * 60 * 60 * 1000, end_timestamp),
                'limit': 1000
            }
            
            try:
                async with self.session.get(url, params=params) as response:
                    batch = await response.json()
                    if not batch:
                        break
                    all_klines.extend(batch)
                    # Próximo batch começa após o último timestamp
                    current_start = int(batch[-1][0]) + 1
            except Exception as e:
                print(f"Erro obtendo dados históricos: {e}")
                break
        
        if not all_klines:
            return pd.DataFrame()
        
        # Converter para DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Converter tipos
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filtrar por período exato
        df = df[df.index <= end_time]
        
        return df
