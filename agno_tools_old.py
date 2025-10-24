"""
Ferramentas customizadas para o AGNO Agent - VERSÃO SÍNCRONA
"""
import json
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import requests
import asyncio
import aiohttp

def get_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Obtém dados de mercado da Binance para análise.
    
    Args:
        symbol: Par de trading (ex: BTCUSDT)
    
    Returns:
        Dados de mercado incluindo preço, volume e indicadores
    """
    try:
        # Usar requests síncrono em vez de aiohttp
        base_url = "https://fapi.binance.com"
        
        # Ticker 24h
        ticker_url = f"{base_url}/fapi/v1/ticker/24hr"
        ticker_response = requests.get(ticker_url, params={'symbol': symbol})
        ticker = ticker_response.json()
        
        # Klines
        klines_url = f"{base_url}/fapi/v1/klines"
        klines_response = requests.get(klines_url, params={
            'symbol': symbol,
            'interval': '1h',
            'limit': 100
        })
        klines = klines_response.json()
        
        # Funding rate
        funding_url = f"{base_url}/fapi/v1/premiumIndex"
        funding_response = requests.get(funding_url, params={'symbol': symbol})
        funding = funding_response.json()
        
        # Open interest
        oi_url = f"{base_url}/fapi/v1/openInterest"
        oi_response = requests.get(oi_url, params={'symbol': symbol})
        open_interest = oi_response.json()
        
        return {
            "symbol": symbol,
            "current_price": float(ticker['lastPrice']),
            "price_change_24h": float(ticker['priceChangePercent']),
            "volume_24h": float(ticker['volume']),
            "high_24h": float(ticker['highPrice']),
            "low_24h": float(ticker['lowPrice']),
            "funding_rate": float(funding.get('lastFundingRate', 0)),
            "open_interest": float(open_interest.get('openInterest', 0)),
            "klines_count": len(klines),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Erro ao obter dados de mercado: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

def analyze_technical_indicators(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Analisa indicadores técnicos para um símbolo.
    
    Args:
        symbol: Par de trading para análise
    
    Returns:
        Sinais técnicos e indicadores calculados
    """
    try:
        # Obter dados de mercado
        market_data = get_market_data(symbol)
        if "error" in market_data:
            return market_data
        
        # Simular análise técnica básica
        current_price = market_data['current_price']
        price_change = market_data['price_change_24h']
        
        # Calcular RSI simples
        rsi = 50 + (price_change * 0.5)  # Simulação básica
        rsi = max(0, min(100, rsi))
        
        # Determinar tendência
        if price_change > 2:
            trend = "bullish"
        elif price_change < -2:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Calcular suporte e resistência
        support = current_price * 0.95
        resistance = current_price * 1.05
        
        return {
            "symbol": symbol,
            "trend": trend,
            "momentum": "strong" if abs(price_change) > 3 else "weak",
            "volatility": "high" if abs(price_change) > 5 else "normal",
            "indicators": {
                "rsi": rsi,
                "macd": price_change * 0.1,
                "adx": 25 + abs(price_change),
                "atr": current_price * 0.02,
                "bb_position": 0.5
            },
            "support": support,
            "resistance": resistance,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Erro na análise técnica: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

def analyze_market_sentiment(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Analisa o sentimento do mercado para um símbolo.
    
    Args:
        symbol: Par de trading
    
    Returns:
        Análise de sentimento do mercado
    """
    try:
        # Simular análise de sentimento
        market_data = get_market_data(symbol)
        if "error" in market_data:
            return market_data
        
        price_change = market_data['price_change_24h']
        
        # Determinar sentimento baseado na mudança de preço
        if price_change > 3:
            sentiment = "very_positive"
            confidence = 0.8
        elif price_change > 1:
            sentiment = "positive"
            confidence = 0.6
        elif price_change < -3:
            sentiment = "very_negative"
            confidence = 0.8
        elif price_change < -1:
            sentiment = "negative"
            confidence = 0.6
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": confidence,
            "price_change_24h": price_change,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Erro na análise de sentimento: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

def get_deepseek_analysis(
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, Any],
    sentiment: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Obtém análise avançada do DeepSeek AI baseado em todos os dados.
    
    Args:
        market_data: Dados de mercado atuais
        technical_indicators: Indicadores técnicos calculados
        sentiment: Análise de sentimento
    
    Returns:
        Análise estruturada com sinal de trading
    """
    try:
        # Análise simples baseada nos dados
        current_price = market_data.get('current_price', 0)
        trend = technical_indicators.get('trend', 'neutral')
        sentiment_score = sentiment.get('sentiment', 'neutral')
        rsi = technical_indicators.get('indicators', {}).get('rsi', 50)
        
        # Determinar sinal
        if trend == "bullish" and sentiment_score in ["positive", "very_positive"] and rsi < 70:
            signal = "BUY"
            confidence = 8
        elif trend == "bearish" and sentiment_score in ["negative", "very_negative"] and rsi > 30:
            signal = "SELL"
            confidence = 8
        else:
            signal = "HOLD"
            confidence = 5
        
        # Calcular níveis
        support = technical_indicators.get('support', current_price * 0.95)
        resistance = technical_indicators.get('resistance', current_price * 1.05)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "entry_price": current_price,
            "stop_loss": support,
            "take_profit_1": resistance,
            "take_profit_2": resistance * 1.02,
            "reasoning": f"Trend: {trend}, Sentiment: {sentiment_score}, RSI: {rsi:.1f}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Erro na análise DeepSeek: {str(e)}",
            "signal": "HOLD",
            "confidence": 3,
            "timestamp": datetime.now().isoformat()
        }

def validate_risk_and_position(
    signal: Dict[str, Any],
    symbol: str,
    account_balance: float = 10000.0
) -> Dict[str, Any]:
    """
    Valida risco e calcula tamanho de posição apropriado.
    
    Args:
        signal: Sinal de trading a validar
        symbol: Símbolo do trade
        account_balance: Saldo da conta
    
    Returns:
        Validação de risco e tamanho de posição
    """
    try:
        if signal.get('signal') == 'HOLD':
            return {
                "can_execute": False,
                "reason": "Sinal HOLD - não executar",
                "risk_level": "low"
            }
        
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        confidence = signal.get('confidence', 5)
        
        if not entry_price or not stop_loss:
            return {
                "can_execute": False,
                "reason": "Preços de entrada ou stop loss não definidos",
                "risk_level": "high"
            }
        
        # Calcular risco
        risk_per_trade = entry_price - stop_loss
        risk_percentage = (risk_per_trade / entry_price) * 100
        
        if risk_percentage > 5:  # Mais de 5% de risco
            return {
                "can_execute": False,
                "reason": f"Risco muito alto: {risk_percentage:.2f}%",
                "risk_level": "high"
            }
        
        # Calcular tamanho da posição
        max_risk_amount = account_balance * 0.02  # 2% do saldo
        position_size = max_risk_amount / risk_per_trade
        
        return {
            "can_execute": True,
            "recommended_position_size": position_size,
            "position_value": position_size * entry_price,
            "max_risk_amount": max_risk_amount,
            "risk_percentage": risk_percentage,
            "risk_level": "acceptable"
        }
    except Exception as e:
        return {
            "can_execute": False,
            "reason": f"Erro na validação: {str(e)}",
            "risk_level": "high"
        }

def execute_paper_trade(
    signal: Dict[str, Any],
    position_size: float
) -> Dict[str, Any]:
    """
    Executa um paper trade (simulado) para teste.
    
    Args:
        signal: Sinal de trading
        position_size: Tamanho da posição
    
    Returns:
        Confirmação da execução do paper trade
    """
    try:
        from pathlib import Path
        
        # Criar pasta de trades se não existir
        Path("paper_trades").mkdir(exist_ok=True)
        
        trade = {
            "trade_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "timestamp": datetime.now().isoformat(),
            "symbol": signal.get('symbol', 'BTCUSDT'),
            "signal": signal.get('signal'),
            "entry_price": signal.get('entry_price'),
            "position_size": position_size,
            "stop_loss": signal.get('stop_loss'),
            "take_profit_1": signal.get('take_profit_1'),
            "take_profit_2": signal.get('take_profit_2'),
            "confidence": signal.get('confidence'),
            "status": "OPEN"
        }
        
        # Salvar trade
        filename = f"paper_trades/trade_{trade['trade_id']}.json"
        with open(filename, 'w') as f:
            json.dump(trade, f, indent=2)
        
        return {
            "success": True,
            "trade_id": trade['trade_id'],
            "message": f"Paper trade executed: {signal.get('signal')} {position_size:.4f} units at {signal.get('entry_price')}",
            "file": filename
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Erro ao executar paper trade: {str(e)}"
        }
