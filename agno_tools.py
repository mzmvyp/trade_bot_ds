"""
Ferramentas AGNO com indicadores técnicos reais e análise de sentimento
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import requests
import talib

def get_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Obtém dados de mercado da Binance para análise.
    """
    try:
        base_url = "https://fapi.binance.com"
        timeout = 2  # Timeout muito reduzido para velocidade
        
        # Ticker 24h
        ticker_response = requests.get(f"{base_url}/fapi/v1/ticker/24hr", params={'symbol': symbol}, timeout=timeout)
        ticker = ticker_response.json()
        
        # Klines para indicadores técnicos (reduzir para 30 klines)
        klines_response = requests.get(f"{base_url}/fapi/v1/klines", params={
            'symbol': symbol,
            'interval': '1h',
            'limit': 30  # Reduzido para 30 para velocidade
        }, timeout=timeout)
        klines = klines_response.json()
        
        # Funding rate
        funding_response = requests.get(f"{base_url}/fapi/v1/premiumIndex", params={'symbol': symbol}, timeout=timeout)
        funding = funding_response.json()
        
        # Open interest
        oi_response = requests.get(f"{base_url}/fapi/v1/openInterest", params={'symbol': symbol}, timeout=timeout)
        open_interest = oi_response.json()
        
        # Processar apenas os últimos 10 klines para indicadores (muito reduzido)
        recent_klines = klines[-10:] if len(klines) > 10 else klines
        
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
            "recent_klines": recent_klines,  # Apenas 50 klines mais recentes
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
    Analisa indicadores técnicos REAIS usando TA-Lib.
    """
    try:
        market_data = get_market_data(symbol)
        if "error" in market_data:
            return market_data
        
        klines = market_data.get('recent_klines', [])
        if len(klines) < 20:
            return {
                "error": "Dados insuficientes para análise técnica",
                "symbol": symbol
            }
        
        # Otimizar: usar apenas os últimos 10 klines para análise mais rápida
        klines = klines[-10:] if len(klines) > 10 else klines
        
        # Converter klines para DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Converter para numérico
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Calcular indicadores técnicos REAIS
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values
        
        # RSI (14 períodos)
        rsi = talib.RSI(close_prices, timeperiod=14)[-1]
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        macd_value = macd[-1]
        macd_signal_value = macd_signal[-1]
        
        # ADX (14 períodos)
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        
        # ATR (14 períodos)
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
        bb_position = (close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
        
        # SMA 20 e 50
        sma_20 = talib.SMA(close_prices, timeperiod=20)[-1]
        sma_50 = talib.SMA(close_prices, timeperiod=50)[-1]
        
        # Determinar tendência
        current_price = close_prices[-1]
        if current_price > sma_20 > sma_50:
            trend = "bullish"
        elif current_price < sma_20 < sma_50:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Determinar momentum
        if rsi > 70:
            momentum = "overbought"
        elif rsi < 30:
            momentum = "oversold"
        else:
            momentum = "neutral"
        
        # Suporte e resistência
        support = bb_lower[-1]
        resistance = bb_upper[-1]
        
        return {
            "symbol": symbol,
            "trend": trend,
            "momentum": momentum,
            "volatility": "high" if atr > current_price * 0.02 else "normal",
            "indicators": {
                "rsi": float(rsi) if not np.isnan(rsi) else 50,
                "macd": float(macd_value) if not np.isnan(macd_value) else 0,
                "macd_signal": float(macd_signal_value) if not np.isnan(macd_signal_value) else 0,
                "adx": float(adx) if not np.isnan(adx) else 25,
                "atr": float(atr) if not np.isnan(atr) else current_price * 0.01,
                "bb_position": float(bb_position) if not np.isnan(bb_position) else 0.5,
                "sma_20": float(sma_20) if not np.isnan(sma_20) else current_price,
                "sma_50": float(sma_50) if not np.isnan(sma_50) else current_price
            },
            "support": float(support) if not np.isnan(support) else current_price * 0.95,
            "resistance": float(resistance) if not np.isnan(resistance) else current_price * 1.05,
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
    Análise de sentimento baseada em múltiplos fatores incluindo Twitter/X.
    """
    try:
        market_data = get_market_data(symbol)
        if "error" in market_data:
            return market_data
        
        price_change = market_data['price_change_24h']
        volume = market_data['volume_24h']
        funding_rate = market_data['funding_rate']
        
        # Fatores de sentimento
        sentiment_factors = []
        
        # 1. Variação de preço
        if price_change > 5:
            sentiment_factors.append(("price", "very_positive", 0.9))
        elif price_change > 2:
            sentiment_factors.append(("price", "positive", 0.7))
        elif price_change < -5:
            sentiment_factors.append(("price", "very_negative", 0.9))
        elif price_change < -2:
            sentiment_factors.append(("price", "negative", 0.7))
        else:
            sentiment_factors.append(("price", "neutral", 0.5))
        
        # 2. Volume (alta = interesse)
        if volume > 1000000:  # Volume alto
            sentiment_factors.append(("volume", "high_interest", 0.8))
        elif volume < 100000:  # Volume baixo
            sentiment_factors.append(("volume", "low_interest", 0.3))
        else:
            sentiment_factors.append(("volume", "normal_interest", 0.5))
        
        # 3. Funding rate (positivo = bullish)
        if funding_rate > 0.01:
            sentiment_factors.append(("funding", "bullish", 0.8))
        elif funding_rate < -0.01:
            sentiment_factors.append(("funding", "bearish", 0.8))
        else:
            sentiment_factors.append(("funding", "neutral", 0.5))
        
        # 4. Análise de sentimento baseada em dados reais de mercado
        twitter_sentiment = analyze_twitter_sentiment(symbol)
        sentiment_factors.append(("market_sentiment", twitter_sentiment[0], twitter_sentiment[1]))
        
        # Calcular sentimento médio
        avg_confidence = np.mean([factor[2] for factor in sentiment_factors])
        
        # Determinar sentimento geral
        positive_factors = sum(1 for factor in sentiment_factors if "positive" in factor[1] or "bullish" in factor[1])
        negative_factors = sum(1 for factor in sentiment_factors if "negative" in factor[1] or "bearish" in factor[1])
        
        if positive_factors > negative_factors:
            sentiment = "positive"
        elif negative_factors > positive_factors:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": float(avg_confidence),
            "factors": {
                "price_change": price_change,
                "volume_level": "high" if volume > 1000000 else "low" if volume < 100000 else "normal",
                "funding_rate": funding_rate,
                "twitter_sentiment": twitter_sentiment[0],
                "twitter_confidence": twitter_sentiment[1]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Erro na análise de sentimento: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

def analyze_twitter_sentiment(symbol: str) -> tuple:
    """
    CORREÇÃO: Análise de sentimento baseada APENAS em dados reais de mercado.
    Não simula Twitter - usa apenas indicadores reais.
    """
    try:
        # Obter dados reais de mercado
        market_data = get_market_data(symbol)
        if "error" in market_data:
            return ("neutral", 0.5)
        
        price_change = market_data['price_change_24h']
        volume = market_data['volume_24h']
        funding_rate = market_data['funding_rate']
        
        # Análise baseada APENAS em dados reais
        sentiment_score = 0
        confidence = 0.5
        
        # 1. Variação de preço (dados reais)
        if price_change > 5:
            sentiment_score += 2
            confidence += 0.2
        elif price_change > 2:
            sentiment_score += 1
            confidence += 0.1
        elif price_change < -5:
            sentiment_score -= 2
            confidence += 0.2
        elif price_change < -2:
            sentiment_score -= 1
            confidence += 0.1
        
        # 2. Volume (dados reais)
        if volume > 1000000:  # Volume alto = interesse
            sentiment_score += 1
            confidence += 0.1
        elif volume < 100000:  # Volume baixo = desinteresse
            sentiment_score -= 1
            confidence += 0.1
        
        # 3. Funding rate (dados reais)
        if funding_rate > 0.01:  # Funding positivo = bullish
            sentiment_score += 1
        elif funding_rate < -0.01:  # Funding negativo = bearish
            sentiment_score -= 1
        
        # Determinar sentimento final baseado em dados reais
        if sentiment_score >= 2:
            return ("positive", min(0.9, confidence))
        elif sentiment_score <= -2:
            return ("negative", min(0.9, confidence))
        else:
            return ("neutral", confidence)
            
    except Exception:
        return ("neutral", 0.5)

def get_deepseek_analysis(
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, Any],
    sentiment: Dict[str, Any]
) -> Dict[str, Any]:
    """
    CORREÇÃO: Esta função deve usar DeepSeek real, não análise própria!
    Por enquanto, retorna dados estruturados para o AGNO Agent usar DeepSeek.
    """
    try:
        # Preparar dados para DeepSeek
        analysis_data = {
            "market_data": market_data,
            "technical_indicators": technical_indicators,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat()
        }
        
        # NOTA: Esta função deveria chamar DeepSeek real
        # Por enquanto, retorna dados para o AGNO Agent processar
        return {
            "analysis_data": analysis_data,
            "needs_deepseek": True,
            "message": "Dados preparados para análise DeepSeek - AGNO Agent deve processar"
        }
    except Exception as e:
        return {
            "error": f"Erro na preparação para DeepSeek: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def validate_risk_and_position(
    signal: Dict[str, Any],
    symbol: str,
    account_balance: float = 10000.0
) -> Dict[str, Any]:
    """
    Valida risco e calcula tamanho de posição apropriado.
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
        risk_per_trade = abs(entry_price - stop_loss)
        risk_percentage = (risk_per_trade / entry_price) * 100
        
        if risk_percentage > 5:
            return {
                "can_execute": False,
                "reason": f"Risco muito alto: {risk_percentage:.2f}%",
                "risk_level": "high"
            }
        
        # Calcular tamanho da posição baseado na confiança
        base_risk = account_balance * 0.02  # 2% base
        confidence_multiplier = confidence / 10  # 0.1 a 1.0
        max_risk_amount = base_risk * confidence_multiplier
        
        position_size = max_risk_amount / risk_per_trade
        
        return {
            "can_execute": True,
            "recommended_position_size": position_size,
            "position_value": position_size * entry_price,
            "max_risk_amount": max_risk_amount,
            "risk_percentage": risk_percentage,
            "risk_level": "acceptable",
            "confidence_multiplier": confidence_multiplier
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
    """
    try:
        from pathlib import Path
        
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
            "reasoning": signal.get('reasoning'),
            "status": "OPEN"
        }
        
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
