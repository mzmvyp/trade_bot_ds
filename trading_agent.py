"""
Agent principal do sistema de trading - Versão Corrigida
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

from binance_client import BinanceClient
from technical_analysis import TechnicalAnalyzer
from sentiment_analysis import SentimentAnalyzer
from deepseek_tool import deepseek_tool
from config import settings

class TradingAgentFixed:
    def __init__(self):
        self.binance_client = BinanceClient()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    async def get_market_data(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Obtém dados do mercado da Binance
        """
        async with self.binance_client as client:
            # Obter dados de diferentes timeframes
            data_1h = await client.get_klines(symbol, "1h", 200)
            data_4h = await client.get_klines(symbol, "4h", 200)
            data_1d = await client.get_klines(symbol, "1d", 200)
            
            # Obter dados adicionais
            ticker_24h = await client.get_ticker_24hr(symbol)
            orderbook = await client.get_orderbook(symbol)
            funding_rate = await client.get_funding_rate(symbol)
            open_interest = await client.get_open_interest(symbol)
            
            return {
                'data_1h': data_1h,
                'data_4h': data_4h,
                'data_1d': data_1d,
                'ticker_24h': ticker_24h,
                'orderbook': orderbook,
                'funding_rate': funding_rate,
                'open_interest': open_interest
            }
    
    async def analyze_technical_signals(self, market_data: Dict) -> Dict:
        """
        Analisa sinais técnicos
        """
        signals = {}
        
        for timeframe, data in [('1h', market_data['data_1h']), 
                               ('4h', market_data['data_4h']), 
                               ('1d', market_data['data_1d'])]:
            if not data.empty:
                data_with_indicators = self.technical_analyzer.calculate_indicators(data)
                signal = self.technical_analyzer.generate_trading_signal(data_with_indicators)
                signals[timeframe] = signal
        
        return signals
    
    async def analyze_sentiment(self) -> Dict:
        """
        Analisa sentimento do mercado
        """
        async with self.sentiment_analyzer as analyzer:
            sentiment_data = await analyzer.analyze_bitcoin_sentiment()
            return sentiment_data
    
    async def generate_trading_signal(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Gera sinal de trading completo usando DeepSeek
        """
        print(f"🔍 Analisando mercado para {symbol}...")
        
        # Obter dados do mercado
        market_data = await self.get_market_data(symbol)
        
        # Analisar sinais técnicos
        technical_signals = await self.analyze_technical_signals(market_data)
        
        # Analisar sentimento
        sentiment_data = await self.analyze_sentiment()
        
        # Preparar dados para o DeepSeek
        analysis_data = {
            'market_data': {
                'current_price': market_data['ticker_24h']['lastPrice'],
                'price_change_24h': market_data['ticker_24h']['priceChangePercent'],
                'volume_24h': market_data['ticker_24h']['volume'],
                'funding_rate': market_data['funding_rate']['lastFundingRate'],
                'open_interest': market_data['open_interest']['openInterest']
            },
            'technical_signals': technical_signals,
            'sentiment_data': sentiment_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Usar DeepSeek para análise
        try:
            print("🧠 Usando DeepSeek para análise...")
            deepseek_response = await deepseek_tool.analyze_trading_data(
                analysis_data['market_data'],
                technical_signals,
                sentiment_data
            )
            
            if deepseek_response:
                # Tentar extrair JSON da resposta do DeepSeek
                import re
                json_match = re.search(r'\{.*\}', deepseek_response, re.DOTALL)
                if json_match:
                    signal_data = json.loads(json_match.group())
                    signal_data['raw_data'] = analysis_data
                    signal_data['timestamp'] = datetime.now().isoformat()
                    return signal_data
                else:
                    print("⚠️  Resposta do DeepSeek não contém JSON válido")
            else:
                print("⚠️  DeepSeek não retornou resposta")
        except Exception as e:
            print(f"⚠️  Erro no DeepSeek: {e}")
        
        # Fallback para análise básica
        print("🔄 Usando análise básica como fallback...")
        signal_data = self._generate_basic_analysis(analysis_data)
        signal_data['raw_data'] = analysis_data
        signal_data['timestamp'] = datetime.now().isoformat()
        
        return signal_data
    
    def _generate_basic_analysis(self, analysis_data: Dict) -> Dict:
        """
        Gera análise básica como fallback
        """
        current_price = float(analysis_data['market_data']['current_price'])
        price_change = float(analysis_data['market_data']['price_change_24h'])
        
        # Análise simples baseada na variação de preço
        if price_change > 2:
            signal_type = "BUY"
            confidence = 7
        elif price_change < -2:
            signal_type = "SELL"
            confidence = 7
        else:
            signal_type = "HOLD"
            confidence = 5
        
        # Calcular níveis básicos
        atr = 1000  # ATR estimado
        if signal_type == "BUY":
            stop_loss = current_price - (atr * 2)
            target1 = current_price + (atr * 2)
            target2 = current_price + (atr * 4)
        elif signal_type == "SELL":
            stop_loss = current_price + (atr * 2)
            target1 = current_price - (atr * 2)
            target2 = current_price - (atr * 4)
        else:
            stop_loss = None
            target1 = None
            target2 = None
        
        return {
            "signal_type": signal_type,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "target1": target1,
            "target2": target2,
            "confidence": confidence,
            "justification": f"Análise básica baseada na variação de preço de 24h ({price_change}%). Recomendo análise mais detalhada para confirmação."
        }
    
    async def run_single_analysis(self, symbol: str = "BTCUSDT"):
        """
        Executa uma única análise
        """
        signal = await self.generate_trading_signal(symbol)
        
        print(f"\n{'='*50}")
        print(f"📈 ANÁLISE DE TRADING - {symbol}")
        print(f"{'='*50}")
        print(f"Tipo: {signal['signal_type']}")
        print(f"Entrada: {signal['entry_price']}")
        print(f"Stop Loss: {signal['stop_loss']}")
        print(f"Alvo 1: {signal['target1']}")
        print(f"Alvo 2: {signal['target2']}")
        print(f"Confiança: {signal['confidence']}/10")
        print(f"Justificativa: {signal['justification']}")
        print(f"{'='*50}\n")
        
        return signal
