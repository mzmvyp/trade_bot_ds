"""
Ferramentas AGNO com indicadores técnicos reais e análise de sentimento
Updated with logging, constants, and improved error handling
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
import talib
import os

from logger import get_logger
from constants import *

logger = get_logger(__name__)

# Análise de sentimento baseada apenas em dados de mercado (Twitter removido)

async def get_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Obtém dados de mercado da Binance para análise (CORRIGIDO: agora async usando BinanceClient).
    """
    try:
        from binance_client import BinanceClient
        
        logger.debug(f"Fetching market data for {symbol}")

        async with BinanceClient() as client:
            # Obter ticker 24h
            ticker = await client.get_ticker_24hr(symbol)
            
            # Obter klines (apenas contagem, não os dados completos)
            klines_df = await client.get_klines(symbol, '1h', limit=DEFAULT_KLINES_LIMIT)
            
            # Obter funding rate
            funding = await client.get_funding_rate(symbol)
            
            # Obter open interest
            open_interest = await client.get_open_interest(symbol)

        result = {
            "symbol": symbol,
            "current_price": float(ticker['lastPrice']),
            "price_change_24h": float(ticker['priceChangePercent']),
            "volume_24h": float(ticker['volume']),
            "high_24h": float(ticker['highPrice']),
            "low_24h": float(ticker['lowPrice']),
            "funding_rate": float(funding.get('lastFundingRate', 0)),
            "open_interest": float(open_interest.get('openInterest', 0)),
            "klines_count": len(klines_df),
            # REMOVIDO: "recent_klines" - muito grande e causa erros de decodificação
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Market data fetched for {symbol}: ${result['current_price']:.2f}")
        return result

    except Exception as e:
        logger.exception(f"Unexpected error fetching market data for {symbol}: {e}")
        return {
            "error": f"Erro ao obter dados de mercado: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

def _analyze_market_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisa estrutura de mercado identificando suporte, resistência e estrutura de tendência.
    """
    try:
        if len(df) < 10:
            return {"structure": "insufficient_data"}
        
        # Identificar High Higher (HH) e Lower Lows (LL)
        highs = df['high'].rolling(window=5).max()
        lows = df['low'].rolling(window=5).min()
        
        # Identificar estrutura de tendência
        recent_highs = highs.tail(10).dropna()
        recent_lows = lows.tail(10).dropna()
        
        if len(recent_highs) >= 3 and len(recent_lows) >= 3:
            # Verificar se está fazendo Higher High an Lower Low (Uptrend)
            if (recent_highs.iloc[-1] > recent_highs.iloc[-3] and 
                recent_lows.iloc[-1] > recent_lows.iloc[-3]):
                structure = "UPTREND"
                strength = "strong" if (recent_highs.iloc[-1] > recent_highs.iloc[-5] and 
                                      recent_lows.iloc[-1] > recent_lows.iloc[-5]) else "moderate"
            
            # Verificar se está fazendo Lower High e Lower Low (Downtrend)
            elif (recent_highs.iloc[-1] < recent_highs.iloc[-3] and 
                  recent_lows.iloc[-1] < recent_lows.iloc[-3]):
                structure = "DOWNTREND"
                strength = "strong" if (recent_highs.iloc[-1] < recent_highs.iloc[-5] and 
                                      recent_lows.iloc[-1] < recent_lows.iloc[-5]) else "moderate"
            else:
                structure = "RANGE"
                strength = "neutral"
        else:
            structure = "RANGE"
            strength = "neutral"
        
        # Identificar níveis de suporte e resistência dinâmicos
        support_level = recent_lows.min() if len(recent_lows) > 0 else df['low'].min()
        resistance_level = recent_highs.max() if len(recent_highs) > 0 else df['high'].max()
        
        return {
            "structure": structure,
            "strength": strength,
            "support_level": float(support_level),
            "resistance_level": float(resistance_level),
            "recent_highs_count": len(recent_highs),
            "recent_lows_count": len(recent_lows)
        }
        
    except Exception as e:
        return {
            "structure": "error",
            "error": str(e)
        }

async def analyze_multiple_timeframes(symbol: str) -> Dict[str, Any]:
    """
    Análise multi-timeframe para maior precisão.
    CORRIGIDO: Agora async usando BinanceClient.
    """
    try:
        from binance_client import BinanceClient
        
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        analyses = {}
        
        async with BinanceClient() as client:
            for tf in timeframes:
                try:
                    # Obter dados para timeframe específico usando BinanceClient
                    klines_df = await client.get_klines(symbol, tf, limit=100)
                    
                    if not klines_df.empty and len(klines_df) >= 20:
                        # Resetar índice para ter timestamp como coluna
                        df = klines_df.reset_index()
                        
                        # Garantir colunas numéricas
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Calcular tendência simples
                        close_prices = df['close'].values
                        sma_20 = talib.SMA(close_prices, timeperiod=20)[-1]
                        current_price = close_prices[-1]
                        
                        if current_price > sma_20:
                            trend = "bullish"
                        elif current_price < sma_20:
                            trend = "bearish"
                        else:
                            trend = "neutral"
                        
                        analyses[tf] = {
                            "trend": trend,
                            "current_price": float(current_price),
                            "sma_20": float(sma_20)
                        }
                except Exception as e:
                    logger.warning(f"Erro no timeframe {tf}: {e}")
                    continue
        
        # Calcular confluência
        bullish_timeframes = sum(1 for tf in analyses.values() if tf['trend'] == 'bullish')
        bearish_timeframes = sum(1 for tf in analyses.values() if tf['trend'] == 'bearish')
        neutral_timeframes = sum(1 for tf in analyses.values() if tf['trend'] == 'neutral')
        
        if bullish_timeframes > bearish_timeframes:
            confluence = "bullish"
        elif bearish_timeframes > bullish_timeframes:
            confluence = "bearish"
        else:
            confluence = "neutral"
        
        return {
            "symbol": symbol,
            "timeframes": analyses,
            "confluence": confluence,
            "bullish_count": bullish_timeframes,
            "bearish_count": bearish_timeframes,
            "neutral_count": neutral_timeframes,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Erro na análise multi-timeframe: {str(e)}",
            "symbol": symbol
        }

async def analyze_order_flow(symbol: str) -> Dict[str, Any]:
    """
    Análise de fluxo de ordens e delta.
    CORRIGIDO: Agora async usando BinanceClient.
    """
    try:
        from binance_client import BinanceClient
        import aiohttp
        
        async with BinanceClient() as client:
            # Obter orderbook
            orderbook = await client.get_orderbook(symbol, limit=20)
            
            # Calcular imbalance
            bid_volume = sum([float(b[1]) for b in orderbook['bids'][:20]])
            ask_volume = sum([float(a[1]) for a in orderbook['asks'][:20]])
            
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Obter trades recentes para CVD (usando API direta pois não temos método no client)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{client.base_url}/fapi/v1/aggTrades",
                    params={'symbol': symbol, 'limit': 100},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as trades_response:
                    buy_volume = 0
                    sell_volume = 0
                    if trades_response.status == 200:
                        trades = await trades_response.json()
                        for trade in trades:
                            if trade['m']:  # isBuyerMaker
                                sell_volume += float(trade['q'])
                            else:
                                buy_volume += float(trade['q'])
        
        cvd = buy_volume - sell_volume
        
        return {
            "symbol": symbol,
            "orderbook_imbalance": imbalance,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "cvd": cvd,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_pressure": buy_volume > sell_volume * 1.2,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"Erro na análise de order flow: {e}")
        return {
            "error": f"Erro na análise de order flow: {str(e)}",
            "symbol": symbol
        }

async def analyze_technical_indicators(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Analisa indicadores técnicos REAIS usando TA-Lib.
    MELHORADO: Inclui EMA, OBV, Volume Profile e Fibonacci conforme sugestões Claude/DeepSeek.
    CORRIGIDO: Agora async usando BinanceClient.
    """
    try:
        from binance_client import BinanceClient
        
        # CORRIGIDO: Obter klines usando BinanceClient async
        async with BinanceClient() as client:
            klines_df = await client.get_klines(symbol, '1h', limit=200)  # Aumentado para ter dados suficientes para EMA 200
        
        if klines_df.empty or len(klines_df) < 50:  # Mínimo para indicadores confiáveis
            return {
                "error": "Dados insuficientes para análise técnica (mínimo 50 candles)",
                "symbol": symbol
            }
        
        # BinanceClient já retorna DataFrame com índice timestamp e colunas numéricas
        # Resetar índice para ter timestamp como coluna
        df = klines_df.reset_index()
        
        # Garantir que temos as colunas necessárias
        if 'timestamp' not in df.columns:
            df['timestamp'] = df.index if hasattr(df.index, 'values') else range(len(df))
        
        # Converter para numérico (já deve estar, mas garantir)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calcular indicadores técnicos REAIS
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values
        
        current_price = close_prices[-1]
        
        # RSI (14 períodos)
        rsi = talib.RSI(close_prices, timeperiod=14)[-1]
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        macd_value = macd[-1]
        macd_signal_value = macd_signal[-1]
        macd_histogram = macd_hist[-1]
        macd_crossover = "bullish" if macd_histogram > 0 and macd_value > macd_signal_value else "bearish" if macd_histogram < 0 else "neutral"
        
        # ADX (14 períodos)
        adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        
        # ATR (14 períodos)
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
        bb_position = (close_prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if (bb_upper[-1] - bb_lower[-1]) > 0 else 0.5
        
        # SMA 20, 50 e 200
        sma_20 = talib.SMA(close_prices, timeperiod=20)[-1]
        sma_50 = talib.SMA(close_prices, timeperiod=50)[-1]
        sma_200 = talib.SMA(close_prices, timeperiod=200) if len(close_prices) >= 200 else None
        sma_200_value = float(sma_200[-1]) if sma_200 is not None and not np.isnan(sma_200[-1]) else None
        
        # EMA 20, 50 e 200 (conforme sugestão Claude/DeepSeek)
        ema_20 = talib.EMA(close_prices, timeperiod=20)[-1]
        ema_50 = talib.EMA(close_prices, timeperiod=50)[-1]
        ema_200 = talib.EMA(close_prices, timeperiod=200) if len(close_prices) >= 200 else None
        ema_200_value = float(ema_200[-1]) if ema_200 is not None and not np.isnan(ema_200[-1]) else None
        
        # OBV (On-Balance Volume) - conforme sugestão
        obv = talib.OBV(close_prices, volume)
        obv_value = float(obv[-1]) if not np.isnan(obv[-1]) else 0
        obv_trend = "bullish" if len(obv) >= 5 and obv[-1] > obv[-5] else "bearish"
        
        # Volume Profile - identificar POC (Point of Control)
        try:
            price_ranges = np.linspace(df['low'].min(), df['high'].max(), 20)
            volume_profile = {}
            for i in range(len(price_ranges) - 1):
                mask = (df['close'] >= price_ranges[i]) & (df['close'] < price_ranges[i+1])
                volume_profile[float(price_ranges[i])] = float(df[mask]['volume'].sum())
            
            # CORRIGIDO: Proteção contra volume_profile vazio
            if volume_profile and len(volume_profile) > 0:
                poc_price = max(volume_profile, key=volume_profile.get)
            else:
                poc_price = current_price
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Erro ao calcular volume profile: {e}")
            poc_price = current_price
            volume_profile = {}
        
        # Fibonacci Retracement (conforme sugestão)
        period_high = df['high'].max()
        period_low = df['low'].min()
        fib_range = period_high - period_low
        
        fib_levels = {
            "fib_0": float(period_high),
            "fib_23.6": float(period_high - (fib_range * 0.236)),
            "fib_38.2": float(period_high - (fib_range * 0.382)),
            "fib_50": float(period_high - (fib_range * 0.50)),
            "fib_61.8": float(period_high - (fib_range * 0.618)),
            "fib_100": float(period_low)
        }
        
        # Determinar tendência melhorada (usando EMA + ADX conforme correção)
        # CORRIGIDO: Considerar ADX para determinar força da tendência
        # ADX > 25 = tendência forte, ADX <= 25 = tendência fraca
        adx_value = float(adx) if not np.isnan(adx) else 25
        
        if ema_200_value:
            if current_price > ema_20 > ema_50 > ema_200_value:
                # EMA alinhada bullish: verificar ADX para determinar força
                if adx_value > 25:
                    trend = "strong_bullish"  # Tendência forte confirmada
                else:
                    trend = "bullish"  # Alinhamento bullish mas ADX fraco
            elif current_price > ema_20 > ema_50:
                trend = "bullish"
            elif current_price < ema_20 < ema_50 < ema_200_value:
                # EMA alinhada bearish: verificar ADX para determinar força
                if adx_value > 25:
                    trend = "strong_bearish"  # Tendência forte confirmada
                else:
                    trend = "bearish"  # Alinhamento bearish mas ADX fraco
            elif current_price < ema_20 < ema_50:
                trend = "bearish"
            else:
                trend = "neutral"
        else:
            if current_price > ema_20 > ema_50:
                # Sem EMA200, verificar ADX para classificar força
                if adx_value > 25:
                    trend = "strong_bullish"
                else:
                    trend = "bullish"
            elif current_price < ema_20 < ema_50:
                # Sem EMA200, verificar ADX para classificar força
                if adx_value > 25:
                    trend = "strong_bearish"
                else:
                    trend = "bearish"
            else:
                trend = "neutral"
        
        # Determinar momentum melhorado
        if rsi > 70:
            momentum = "overbought"
        elif rsi < 30:
            momentum = "oversold"
        elif rsi > 50:
            momentum = "bullish"
        elif rsi < 50:
            momentum = "bearish"
        else:
            momentum = "neutral"
        
        # Suporte e resistência melhorados (usando Fibonacci e Volume Profile)
        support = min([fib_levels["fib_61.8"], bb_lower[-1], poc_price])
        resistance = max([fib_levels["fib_38.2"], bb_upper[-1], poc_price])
        
        # Análise de estrutura de mercado
        market_structure = _analyze_market_structure(df)
        
        return {
            "symbol": symbol,
            "trend": trend,
            "momentum": momentum,
            "volatility": "high" if atr > current_price * 0.02 else "normal",
            "market_structure": market_structure,
            "indicators": {
                "rsi": float(rsi) if not np.isnan(rsi) else 50,
                "macd": float(macd_value) if not np.isnan(macd_value) else 0,
                "macd_signal": float(macd_signal_value) if not np.isnan(macd_signal_value) else 0,
                "macd_histogram": float(macd_histogram) if not np.isnan(macd_histogram) else 0,
                "macd_crossover": macd_crossover,
                "adx": float(adx) if not np.isnan(adx) else 25,
                "atr": float(atr) if not np.isnan(atr) else current_price * 0.01,
                "bb_position": float(bb_position) if not np.isnan(bb_position) else 0.5,
                "bb_upper": float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else current_price * 1.05,
                "bb_middle": float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else current_price,
                "bb_lower": float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else current_price * 0.95,
                "sma_20": float(sma_20) if not np.isnan(sma_20) else current_price,
                "sma_50": float(sma_50) if not np.isnan(sma_50) else current_price,
                "sma_200": sma_200_value,
                "ema_20": float(ema_20) if not np.isnan(ema_20) else current_price,
                "ema_50": float(ema_50) if not np.isnan(ema_50) else current_price,
                "ema_200": ema_200_value,
                "obv": obv_value,
                "obv_trend": obv_trend
            },
            "volume_profile": {
                "poc_price": float(poc_price),
                "high_volume_zones": sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)[:3] if volume_profile and len(volume_profile) > 0 else []
            },
            "fibonacci_levels": fib_levels,
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

async def analyze_market_sentiment(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Análise de sentimento baseada em dados de mercado (preço, volume, funding rate).
    CORRIGIDO: Agora async.
    """
    try:
        market_data = await get_market_data(symbol)
        if "error" in market_data:
            return market_data
        
        price_change = market_data['price_change_24h']
        volume = market_data['volume_24h']
        funding_rate = market_data['funding_rate']
        open_interest = market_data.get('open_interest', 0)
        
        # Análise baseada em dados reais de mercado
        sentiment_score = 0
        confidence = 0.5
        
        # 1. Variação de preço (indicador principal de sentimento)
        if price_change > 8:
            sentiment_score += 3  # Forte bullish
            confidence += 0.3
        elif price_change > 3:
            sentiment_score += 2  # Bullish
            confidence += 0.2
        elif price_change > 1:
            sentiment_score += 1  # Levemente bullish
            confidence += 0.1
        elif price_change < -8:
            sentiment_score -= 3  # Forte bearish
            confidence += 0.3
        elif price_change < -3:
            sentiment_score -= 2  # Bearish
            confidence += 0.2
        elif price_change < -1:
            sentiment_score -= 1  # Levemente bearish
            confidence += 0.1
        
        # 2. Volume (alta = interesse, baixa = desinteresse)
        if volume > 2000000:  # Volume muito alto = forte interesse
            sentiment_score += 2
            confidence += 0.2
        elif volume > 1000000:  # Volume alto = interesse
            sentiment_score += 1
            confidence += 0.1
        elif volume < 50000:  # Volume baixo = desinteresse
            sentiment_score -= 1
            confidence += 0.1
        
        # 3. Funding rate (positivo = bullish, negativo = bearish)
        if funding_rate > 0.02:  # Funding muito positivo = muito bullish
            sentiment_score += 2
            confidence += 0.2
        elif funding_rate > 0.005:  # Funding positivo = bullish
            sentiment_score += 1
            confidence += 0.1
        elif funding_rate < -0.02:  # Funding muito negativo = muito bearish
            sentiment_score -= 2
            confidence += 0.2
        elif funding_rate < -0.005:  # Funding negativo = bearish
            sentiment_score -= 1
            confidence += 0.1
        
        # 4. Open Interest (alta = interesse institucional)
        if open_interest > 100000000:  # OI muito alto
            sentiment_score += 1
            confidence += 0.1
        elif open_interest < 10000000:  # OI baixo
            sentiment_score -= 1
            confidence += 0.1
        
        # Determinar sentimento final baseado em dados reais
        if sentiment_score >= 3:
            sentiment = "very_positive"
            final_confidence = min(0.95, confidence)
        elif sentiment_score >= 1:
            sentiment = "positive"
            final_confidence = min(0.9, confidence)
        elif sentiment_score <= -3:
            sentiment = "very_negative"
            final_confidence = min(0.95, confidence)
        elif sentiment_score <= -1:
            sentiment = "negative"
            final_confidence = min(0.9, confidence)
        else:
            sentiment = "neutral"
            final_confidence = confidence
        
        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": float(final_confidence),
            "factors": {
                "price_change": price_change,
                "volume_level": "high" if volume > 1000000 else "low" if volume < 100000 else "normal",
                "funding_rate": funding_rate,
                "open_interest": open_interest
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Erro na análise de sentimento: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# FUNÇÕES AUXILIARES PARA INTERPRETAÇÃO DE DADOS
# ============================================================================

def _classify_rsi(rsi: float) -> Dict[str, str]:
    """Classifica RSI em zona e hint de ação"""
    try:
        if rsi < 30:
            return {"zone": "oversold", "action_hint": "potential_buy"}
        elif rsi < 40:
            return {"zone": "approaching_oversold", "action_hint": "potential_buy"}
        elif rsi < 60:
            return {"zone": "neutral", "action_hint": "wait"}
        elif rsi < 70:
            return {"zone": "approaching_overbought", "action_hint": "potential_sell"}
        else:
            return {"zone": "overbought", "action_hint": "potential_sell"}
    except:
        return {"zone": "neutral", "action_hint": "wait"}

def _interpret_adx(adx: float) -> str:
    """Interpreta força da tendência baseado no ADX"""
    try:
        if adx < 20:
            return "no_trend"
        elif adx < 25:
            return "weak"
        elif adx < 50:
            return "moderate"
        else:
            return "strong"
    except:
        return "no_trend"

def _interpret_macd_momentum(histogram: float, prev_histogram: Optional[float] = None) -> str:
    """Determina direção do momentum do MACD"""
    try:
        if prev_histogram is None:
            if histogram > 0:
                return "accelerating_up"
            else:
                return "accelerating_down"
        
        if histogram > 0 and histogram > prev_histogram:
            return "accelerating_up"
        elif histogram > 0 and histogram < prev_histogram:
            return "decelerating_up"
        elif histogram < 0 and histogram < prev_histogram:
            return "accelerating_down"
        else:
            return "decelerating_down"
    except:
        return "neutral"

def _classify_bollinger_position(position: float) -> str:
    """Classifica posição nas Bollinger Bands"""
    try:
        if position < 0.2:
            return "lower_band"
        elif position < 0.4:
            return "below_middle"
        elif position < 0.6:
            return "middle"
        elif position < 0.8:
            return "above_middle"
        else:
            return "upper_band"
    except:
        return "middle"

def _detect_ema_alignment(ema20: float, ema50: float, ema200: Optional[float], price: float) -> str:
    """Detecta alinhamento das EMAs"""
    try:
        if ema200 is None:
            if price > ema20 > ema50:
                return "bullish_stack"
            elif price < ema20 < ema50:
                return "bearish_stack"
            else:
                return "mixed"
        else:
            if price > ema20 > ema50 > ema200:
                return "bullish_stack"
            elif price < ema20 < ema50 < ema200:
                return "bearish_stack"
            else:
                return "mixed"
    except:
        return "mixed"

def _interpret_funding_rate(rate: float) -> str:
    """Interpreta funding rate"""
    try:
        if rate > 0.01:
            return "crowded_long"
        elif rate > 0.005:
            return "slightly_long"
        elif rate > -0.005:
            return "neutral"
        elif rate > -0.01:
            return "slightly_short"
        else:
            return "crowded_short"
    except:
        return "neutral"

def _classify_orderbook_imbalance(imbalance: float) -> str:
    """Classifica pressão do orderbook"""
    try:
        if imbalance > 0.5:
            return "strong_buy_pressure"
        elif imbalance > 0.2:
            return "buy_pressure"
        elif imbalance > -0.2:
            return "neutral"
        elif imbalance > -0.5:
            return "sell_pressure"
        else:
            return "strong_sell_pressure"
    except:
        return "neutral"

def _calculate_suggested_stops(atr: float, price: float, signal_type: str = "BUY") -> Dict[str, float]:
    """Calcula stop loss e take profits sugeridos baseado em ATR"""
    try:
        atr_pct = (atr / price) * 100
        
        if signal_type == "BUY":
            stop_pct = 1.5 * atr_pct
            tp1_pct = 2.0 * atr_pct
            tp2_pct = 3.0 * atr_pct
        else:  # SELL
            stop_pct = 1.5 * atr_pct
            tp1_pct = 2.0 * atr_pct
            tp2_pct = 3.0 * atr_pct
        
        return {
            "suggested_stop_pct": stop_pct,
            "suggested_tp1_pct": tp1_pct,
            "suggested_tp2_pct": tp2_pct
        }
    except:
        return {
            "suggested_stop_pct": 2.0,
            "suggested_tp1_pct": 2.5,
            "suggested_tp2_pct": 5.0
        }

def _identify_conflicting_signals(data: Dict) -> List[str]:
    """Identifica sinais que se contradizem"""
    conflicts = []
    try:
        trend = data.get("trend_analysis", {}).get("primary_trend", "neutral")
        momentum = data.get("trend_analysis", {}).get("momentum", "neutral")
        rsi_zone = data.get("key_indicators", {}).get("rsi", {}).get("zone", "neutral")
        macd_crossover = data.get("key_indicators", {}).get("macd", {}).get("crossover", "neutral")
        
        # RSI oversold mas tendência bearish
        if rsi_zone == "oversold" and "bearish" in trend:
            conflicts.append("RSI oversold but strong bearish trend")
        
        # RSI overbought mas tendência bullish
        if rsi_zone == "overbought" and "bullish" in trend:
            conflicts.append("RSI overbought but strong bullish trend")
        
        # MACD bearish mas momentum bullish
        if macd_crossover == "bearish" and "bullish" in momentum:
            conflicts.append("MACD bearish crossover but bullish momentum")
        
        # MACD bullish mas momentum bearish
        if macd_crossover == "bullish" and "bearish" in momentum:
            conflicts.append("MACD bullish crossover but bearish momentum")
        
        # Tendência bullish mas orderbook com sell pressure
        orderbook_bias = data.get("volume_flow", {}).get("orderbook_bias", "neutral")
        if "bullish" in trend and "sell" in orderbook_bias:
            conflicts.append("Bullish trend but sell pressure in orderbook")
        
        # Tendência bearish mas orderbook com buy pressure
        if "bearish" in trend and "buy" in orderbook_bias:
            conflicts.append("Bearish trend but buy pressure in orderbook")
        
    except Exception as e:
        logger.warning(f"Erro ao identificar sinais conflitantes: {e}")
    
    return conflicts

def _calculate_overall_bias(data: Dict) -> Dict[str, Any]:
    """Calcula score agregado de -10 a +10 com interpretação"""
    try:
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        # CORRIGIDO: Proteção contra None em todas as chamadas .get().get()
        trend_analysis = data.get("trend_analysis") or {}
        trend = trend_analysis.get("primary_trend", "neutral")
        if "strong_bullish" in trend:
            bullish_count += 3
        elif "bullish" in trend:
            bullish_count += 2
        elif "strong_bearish" in trend:
            bearish_count += 3
        elif "bearish" in trend:
            bearish_count += 2
        else:
            neutral_count += 1
        
        # Analisar momentum
        momentum = trend_analysis.get("momentum", "neutral")
        if momentum == "overbought":
            bearish_count += 1
        elif momentum == "oversold":
            bullish_count += 1
        elif "bullish" in momentum:
            bullish_count += 1
        elif "bearish" in momentum:
            bearish_count += 1
        else:
            neutral_count += 1
        
        # Analisar RSI
        key_indicators = data.get("key_indicators") or {}
        rsi_data = key_indicators.get("rsi") or {}
        rsi_hint = rsi_data.get("action_hint", "wait")
        if "buy" in rsi_hint:
            bullish_count += 1
        elif "sell" in rsi_hint:
            bearish_count += 1
        else:
            neutral_count += 1
        
        # Analisar MACD
        macd_data = key_indicators.get("macd") or {}
        macd_crossover = macd_data.get("crossover", "neutral")
        if macd_crossover == "bullish":
            bullish_count += 1
        elif macd_crossover == "bearish":
            bearish_count += 1
        else:
            neutral_count += 1
        
        # Analisar EMA alignment
        ema_structure = key_indicators.get("ema_structure") or {}
        ema_alignment = ema_structure.get("ema_alignment", "mixed")
        if "bullish" in ema_alignment:
            bullish_count += 1
        elif "bearish" in ema_alignment:
            bearish_count += 1
        else:
            neutral_count += 1
        
        # Analisar orderbook
        volume_flow = data.get("volume_flow") or {}
        orderbook_bias = volume_flow.get("orderbook_bias", "neutral")
        if "buy" in orderbook_bias:
            bullish_count += 1
        elif "sell" in orderbook_bias:
            bearish_count += 1
        else:
            neutral_count += 1
        
        # Analisar sentimento
        sentiment_data = data.get("sentiment") or {}
        sentiment = sentiment_data.get("overall", "neutral")
        if "very_positive" in sentiment:
            bullish_count += 2
        elif "positive" in sentiment:
            bullish_count += 1
        elif "very_negative" in sentiment:
            bearish_count += 2
        elif "negative" in sentiment:
            bearish_count += 1
        else:
            neutral_count += 1
        
        # Calcular bias geral (-10 a +10)
        overall_bias = bullish_count - bearish_count
        overall_bias = max(-10, min(10, overall_bias))
        
        # Interpretar bias
        if overall_bias >= 7:
            interpretation = "strong_buy"
            recommended_action = "BUY"
        elif overall_bias >= 3:
            interpretation = "buy"
            recommended_action = "BUY"
        elif overall_bias <= -7:
            interpretation = "strong_sell"
            recommended_action = "SELL"
        elif overall_bias <= -3:
            interpretation = "sell"
            recommended_action = "SELL"
        else:
            interpretation = "neutral"
            recommended_action = "WAIT"
        
        return {
            "bullish_factors_count": bullish_count,
            "bearish_factors_count": bearish_count,
            "neutral_factors_count": neutral_count,
            "overall_bias": overall_bias,
            "overall_bias_interpretation": interpretation,
            "recommended_action": recommended_action
        }
    except Exception as e:
        logger.warning(f"Erro ao calcular bias geral: {e}")
        return {
            "bullish_factors_count": 0,
            "bearish_factors_count": 0,
            "neutral_factors_count": 0,
            "overall_bias": 0,
            "overall_bias_interpretation": "neutral",
            "recommended_action": "WAIT"
        }

def _interpret_confluence(bullish_count: int, bearish_count: int) -> str:
    """Interpreta alinhamento de timeframes"""
    try:
        # CORRIGIDO: Garantir que são inteiros válidos
        bullish_count = int(bullish_count) if bullish_count is not None else 0
        bearish_count = int(bearish_count) if bearish_count is not None else 0
        
        if bullish_count >= 4:
            return "strong_bullish_alignment"
        elif bullish_count >= 3:
            return "bullish_alignment"
        elif bearish_count >= 4:
            return "strong_bearish_alignment"
        elif bearish_count >= 3:
            return "bearish_alignment"
        else:
            return "mixed_signals"
    except Exception as e:
        logger.warning(f"Erro ao interpretar confluência: {e}")
        return "mixed_signals"

# ============================================================================
# FUNÇÃO PRINCIPAL: PREPARAR ANÁLISE PARA LLM
# ============================================================================

async def prepare_analysis_for_llm(symbol: str) -> Dict[str, Any]:
    """
    Prepara dados SUMARIZADOS e INTERPRETADOS para envio ao DeepSeek.
    
    REGRAS:
    - Payload máximo: 5KB
    - NUNCA incluir arrays de klines
    - SEMPRE interpretar valores numéricos em categorias
    - Incluir scores agregados pré-calculados
    
    Returns:
        Dict estruturado e compacto para a LLM
    """
    try:
        # Coletar todos os dados necessários (async)
        market_data = await get_market_data(symbol)
        technical_indicators = await analyze_technical_indicators(symbol)
        sentiment = await analyze_market_sentiment(symbol)
        multi_timeframe = await analyze_multiple_timeframes(symbol)
        order_flow = await analyze_order_flow(symbol)
        
        # Verificar erros
        if "error" in market_data or "error" in technical_indicators:
            logger.error(f"Erro ao coletar dados para {symbol}")
            return {"error": "Erro ao coletar dados de mercado"}
        
        # Extrair valores principais
        current_price = market_data.get("current_price", 0)
        price_change_24h = market_data.get("price_change_24h", 0)
        high_24h = market_data.get("high_24h", current_price)
        low_24h = market_data.get("low_24h", current_price)
        position_in_range = ((current_price - low_24h) / (high_24h - low_24h) * 100) if (high_24h - low_24h) > 0 else 50
        
        # Extrair indicadores técnicos
        indicators = technical_indicators.get("indicators", {})
        rsi_value = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        macd_crossover = indicators.get("macd_crossover", "neutral")
        adx_value = indicators.get("adx", 25)
        atr_value = indicators.get("atr", current_price * 0.01)
        bb_position = indicators.get("bb_position", 0.5)
        ema_20 = indicators.get("ema_20", current_price)
        ema_50 = indicators.get("ema_50", current_price)
        ema_200 = indicators.get("ema_200")
        obv_trend = indicators.get("obv_trend", "neutral")
        
        # Extrair níveis
        support = technical_indicators.get("support", current_price * 0.95)
        resistance = technical_indicators.get("resistance", current_price * 1.05)
        fib_levels = technical_indicators.get("fibonacci_levels", {})
        # CORRIGIDO: Proteção adicional contra None em volume_profile
        volume_profile_data = technical_indicators.get("volume_profile") or {}
        poc_price = volume_profile_data.get("poc_price", current_price)
        
        # Calcular distâncias
        distance_to_support = ((current_price - support) / current_price) * 100
        distance_to_resistance = ((resistance - current_price) / current_price) * 100
        
        # Interpretar dados
        rsi_classification = _classify_rsi(rsi_value)
        adx_interpretation = _interpret_adx(adx_value)
        macd_momentum = _interpret_macd_momentum(macd_hist)
        bb_zone = _classify_bollinger_position(bb_position)
        ema_alignment = _detect_ema_alignment(ema_20, ema_50, ema_200, current_price)
        funding_interpretation = _interpret_funding_rate(market_data.get("funding_rate", 0))
        orderbook_bias = _classify_orderbook_imbalance(order_flow.get("orderbook_imbalance", 0))
        suggested_stops = _calculate_suggested_stops(atr_value, current_price)
        
        # Analisar timeframes
        timeframe_alignment = {}
        confluence_bullish = multi_timeframe.get("bullish_count", 0)
        confluence_bearish = multi_timeframe.get("bearish_count", 0)
        confluence_score = confluence_bullish - confluence_bearish
        confluence_interpretation = _interpret_confluence(confluence_bullish, confluence_bearish)
        
        for tf in ["5m", "15m", "1h", "4h", "1d"]:
            tf_data = multi_timeframe.get("timeframes", {}).get(tf, {})
            timeframe_alignment[tf] = tf_data.get("trend", "neutral")
        
        # Determinar tendência primária
        primary_trend = technical_indicators.get("trend", "neutral")
        momentum = technical_indicators.get("momentum", "neutral")
        
        # Classificar volatilidade
        atr_pct = (atr_value / current_price) * 100
        if atr_pct > 0.05:
            volatility_level = "extreme"
        elif atr_pct > 0.03:
            volatility_level = "high"
        elif atr_pct > 0.015:
            volatility_level = "normal"
        else:
            volatility_level = "low"
        
        # Volume trend
        volume_24h = market_data.get("volume_24h", 0)
        volume_trend = "stable"  # Simplificado - poderia comparar com médias
        
        # Open interest trend
        oi_trend = "stable"  # Simplificado
        
        # Construir estrutura de análise
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            
            "price_context": {
                "current": current_price,
                "change_24h_pct": price_change_24h,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "position_in_range_pct": position_in_range
            },
            
            "trend_analysis": {
                "primary_trend": primary_trend,
                "trend_strength_adx": adx_value,
                "trend_strength_interpretation": adx_interpretation,
                "momentum": momentum,
                "timeframe_alignment": timeframe_alignment,
                "confluence_score": confluence_score,
                "confluence_interpretation": confluence_interpretation
            },
            
            "key_indicators": {
                "rsi": {
                    "value": rsi_value,
                    "zone": rsi_classification["zone"],
                    "action_hint": rsi_classification["action_hint"]
                },
                "macd": {
                    "histogram": macd_hist,
                    "crossover": macd_crossover,
                    "momentum_direction": macd_momentum
                },
                "bollinger": {
                    "position": bb_position,
                    "zone": bb_zone,
                    "squeeze_detected": False  # Simplificado
                },
                "ema_structure": {
                    "price_vs_ema20": "above" if current_price > ema_20 else "below",
                    "price_vs_ema50": "above" if current_price > ema_50 else "below",
                    "price_vs_ema200": "above" if ema_200 and current_price > ema_200 else "below" if ema_200 else "N/A",
                    "ema_alignment": ema_alignment
                },
                "obv_trend": obv_trend
            },
            
            "key_levels": {
                "immediate_support": support,
                "immediate_resistance": resistance,
                "fib_382": fib_levels.get("fib_38.2", support),
                "fib_50": fib_levels.get("fib_50", current_price),
                "fib_618": fib_levels.get("fib_61.8", resistance),
                "volume_poc": poc_price,
                "distance_to_support_pct": distance_to_support,
                "distance_to_resistance_pct": distance_to_resistance
            },
            
            "volume_flow": {
                "volume_24h": volume_24h,
                "volume_trend": volume_trend,
                "obv_trend": obv_trend,
                "orderbook_imbalance": order_flow.get("orderbook_imbalance", 0),
                "orderbook_bias": orderbook_bias,
                "cvd_direction": "positive" if order_flow.get("cvd", 0) > 0 else "negative"
            },
            
            "sentiment": {
                "overall": sentiment.get("sentiment", "neutral"),
                "confidence": sentiment.get("confidence", 0.5),
                "funding_rate": market_data.get("funding_rate", 0),
                "funding_interpretation": funding_interpretation,
                "open_interest_trend": oi_trend
            },
            
            "volatility": {
                "atr_value": atr_value,
                "atr_pct": atr_pct,
                "level": volatility_level,
                "suggested_stop_pct": suggested_stops["suggested_stop_pct"],
                "suggested_tp1_pct": suggested_stops["suggested_tp1_pct"],
                "suggested_tp2_pct": suggested_stops["suggested_tp2_pct"]
            },
            
            "conflicting_signals": [],  # Será preenchido depois
            "aggregated_scores": {}  # Será preenchido depois
        }
        
        # Identificar sinais conflitantes
        analysis["conflicting_signals"] = _identify_conflicting_signals(analysis)
        
        # Calcular scores agregados
        analysis["aggregated_scores"] = _calculate_overall_bias(analysis)
        
        # CORRIGIDO: Garantir que não estamos enviando dados muito grandes
        # Limitar high_volume_zones se muito grande (já está limitado em analyze_technical_indicators, mas garantir)
        # Nota: high_volume_zones não está diretamente em analysis, mas sim em key_levels.volume_poc
        # O volume_profile completo está em technical_indicators, mas não é incluído em analysis
        
        # Validar tamanho do payload
        payload_size = len(json.dumps(analysis))
        if payload_size > 10000:  # 10KB (com margem de segurança)
            logger.warning(f"Payload muito grande: {payload_size} bytes para {symbol}")
            # Se payload muito grande, remover dados desnecessários
            # Remover high_volume_zones se existir em algum lugar
            if "key_levels" in analysis and "volume_poc" in analysis["key_levels"]:
                # Manter apenas POC, já está otimizado
                pass
        
        return analysis
        
    except Exception as e:
        logger.exception(f"Erro ao preparar análise para LLM: {e}")
        return {
            "error": f"Erro ao preparar análise: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

def _create_analysis_prompt(analysis: Dict[str, Any]) -> str:
    """Cria prompt estruturado para o DeepSeek"""
    try:
        conflicting = analysis.get("conflicting_signals", [])
        conflicting_text = "\n".join([f"- {s}" for s in conflicting]) if conflicting else "- Nenhum conflito identificado"
        
        return f"""
Analise os dados e forneça um sinal de trading.

## DADOS DE MERCADO

- Símbolo: {analysis['symbol']}
- Preço: ${analysis['price_context']['current']:,.2f}
- Variação 24h: {analysis['price_context']['change_24h_pct']:+.2f}%
- Posição no range: {analysis['price_context']['position_in_range_pct']:.0f}%

## ANÁLISE DE TENDÊNCIA

- Tendência primária: {analysis['trend_analysis']['primary_trend']}
- Força (ADX): {analysis['trend_analysis']['trend_strength_interpretation']}
- Momentum: {analysis['trend_analysis']['momentum']}
- Alinhamento timeframes: {analysis['trend_analysis']['confluence_interpretation']}
- Score confluência: {analysis['trend_analysis']['confluence_score']}/5

## INDICADORES

- RSI: {analysis['key_indicators']['rsi']['value']:.1f} ({analysis['key_indicators']['rsi']['zone']})
- MACD: {analysis['key_indicators']['macd']['crossover']} - {analysis['key_indicators']['macd']['momentum_direction']}
- Bollinger: {analysis['key_indicators']['bollinger']['zone']}
- EMAs: {analysis['key_indicators']['ema_structure']['ema_alignment']}

## NÍVEIS CHAVE

- Suporte: ${analysis['key_levels']['immediate_support']:,.2f} ({analysis['key_levels']['distance_to_support_pct']:+.2f}%)
- Resistência: ${analysis['key_levels']['immediate_resistance']:,.2f} ({analysis['key_levels']['distance_to_resistance_pct']:+.2f}%)
- POC Volume: ${analysis['key_levels']['volume_poc']:,.2f}

## VOLUME E FLUXO

- Pressão orderbook: {analysis['volume_flow']['orderbook_bias']}
- OBV: {analysis['volume_flow']['obv_trend']}

## SENTIMENTO

- Geral: {analysis['sentiment']['overall']}
- Funding: {analysis['sentiment']['funding_interpretation']}

## VOLATILIDADE

- Nível: {analysis['volatility']['level']}
- Stop sugerido: {analysis['volatility']['suggested_stop_pct']:.2f}%
- TP1 sugerido: {analysis['volatility']['suggested_tp1_pct']:.2f}%
- TP2 sugerido: {analysis['volatility']['suggested_tp2_pct']:.2f}%

## SINAIS CONFLITANTES

{conflicting_text}

## SCORE AGREGADO

- Fatores bullish: {analysis['aggregated_scores']['bullish_factors_count']}
- Fatores bearish: {analysis['aggregated_scores']['bearish_factors_count']}
- Bias geral: {analysis['aggregated_scores']['overall_bias']}/10 ({analysis['aggregated_scores']['overall_bias_interpretation']})
- Ação recomendada: {analysis['aggregated_scores']['recommended_action']}

---

RESPONDA APENAS COM JSON:

```json
{{
    "signal": "BUY" ou "SELL" ou "NO_SIGNAL",
    "entry_price": <número>,
    "stop_loss": <número>,
    "take_profit_1": <número>,
    "take_profit_2": <número>,
    "confidence": <1-10>,
    "reasoning": "<justificativa>"
}}
```
"""
    except Exception as e:
        logger.exception(f"Erro ao criar prompt: {e}")
        return "Erro ao criar prompt de análise."

async def get_deepseek_analysis(symbol: str) -> Dict[str, Any]:
    """
    Prepara análise otimizada para o DeepSeek e chama diretamente.
    CORRIGIDO: Agora chama DeepSeek diretamente e retorna o sinal JSON processado.
    """
    try:
        import os
        from agno.models.deepseek import DeepSeek
        from agno.agent import Agent
        
        # Usar a nova função que já sumariza tudo
        analysis = await prepare_analysis_for_llm(symbol)
        
        if "error" in analysis:
            return analysis
        
        # Criar prompt estruturado
        prompt = _create_analysis_prompt(analysis)
        
        # CORRIGIDO: Chamar DeepSeek diretamente ao invés de retornar prompt
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY não encontrada, retornando apenas prompt")
            return {
                "analysis_data": analysis,
                "deepseek_prompt": prompt,
                "needs_agent_processing": True,
                "timestamp": datetime.now().isoformat()
            }
        
        # Criar agent simples apenas para chamar DeepSeek
        model = DeepSeek(id="deepseek-chat", api_key=api_key, temperature=0.3, max_tokens=1000)
        agent = Agent(
            model=model,
            instructions="Você é um trader profissional. Analise os dados e forneça um sinal de trading em formato JSON."
        )
        
        # Chamar DeepSeek diretamente
        logger.info(f"[DEEPSEEK] Chamando DeepSeek diretamente para {symbol}")
        response = await agent.arun(prompt)
        
        # Extrair conteúdo da resposta
        response_content = str(response.content) if hasattr(response, 'content') else str(response)
        
        # Tentar extrair JSON da resposta
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
        if json_match:
            try:
                signal_json = json.loads(json_match.group(1))
                logger.info(f"[DEEPSEEK] Sinal extraído: {signal_json.get('signal', 'N/A')} com confiança {signal_json.get('confidence', 0)}")
                
                # Retornar sinal processado
                return {
                    "signal": signal_json.get("signal", "NO_SIGNAL"),
                    "entry_price": signal_json.get("entry_price"),
                    "stop_loss": signal_json.get("stop_loss"),
                    "take_profit_1": signal_json.get("take_profit_1"),
                    "take_profit_2": signal_json.get("take_profit_2"),
                    "confidence": signal_json.get("confidence", 5),
                    "reasoning": signal_json.get("reasoning", ""),
                    "analysis_data": analysis,
                    "raw_response": response_content,
                    "timestamp": datetime.now().isoformat()
                }
            except json.JSONDecodeError as e:
                logger.warning(f"[DEEPSEEK] Erro ao decodificar JSON: {e}")
        
        # Se não conseguiu extrair JSON, retornar resposta bruta
        logger.warning(f"[DEEPSEEK] Não foi possível extrair JSON, retornando resposta bruta")
        return {
            "analysis_data": analysis,
            "deepseek_prompt": prompt,
            "raw_response": response_content,
            "needs_agent_processing": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"Erro na preparação para DeepSeek: {e}")
        return {
            "error": f"Erro na preparação para DeepSeek: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def _calculate_current_drawdown() -> float:
    """
    Calcula o drawdown atual baseado no histórico de trades.
    CORRIGIDO: Usa real_paper_trading ao invés de paper_trading.
    """
    try:
        from real_paper_trading import real_paper_trading
        summary = real_paper_trading.get_portfolio_summary()
        return max(0, (summary['initial_balance'] - summary['total_portfolio_value']) / summary['initial_balance'])
    except:
        return 0.0

def _calculate_total_exposure() -> float:
    """
    Calcula a exposição total atual.
    CORRIGIDO: Usa real_paper_trading ao invés de paper_trading.
    """
    try:
        from real_paper_trading import real_paper_trading
        summary = real_paper_trading.get_portfolio_summary()
        return summary['open_positions_value']
    except:
        return 0.0

def _get_daily_trades_count() -> int:
    """
    Retorna o número de trades executados hoje.
    CORRIGIDO: Usa real_paper_trading ao invés de paper_trading.
    """
    try:
        from real_paper_trading import real_paper_trading
        today = datetime.now().date()
        trades = real_paper_trading.get_trade_history()
        daily_trades = [t for t in trades if datetime.fromisoformat(t['timestamp']).date() == today]
        return len(daily_trades)
    except:
        return 0

def validate_risk_and_position(
    signal: Dict[str, Any],
    symbol: str,
    account_balance: float = 10000.0
) -> Dict[str, Any]:
    """
    Valida risco e calcula tamanho de posição apropriado com circuit breakers.
    CORRIGIDO: Verifica se já existe posição aberta para o símbolo.
    """
    try:
        if signal.get('signal') == 'HOLD' or signal.get('signal') == 'NO_SIGNAL':
            return {
                "can_execute": False,
                "reason": "Sinal HOLD/NO_SIGNAL - não executar",
                "risk_level": "low"
            }
        
        # CRÍTICO: Verificar se já existe posição aberta para este símbolo
        try:
            import json
            import os
            if os.path.exists("portfolio/state.json"):
                with open("portfolio/state.json", "r", encoding='utf-8') as f:
                    state = json.load(f)
                    positions = state.get("positions", {})
                    
                    # Verificar se existe posição BUY (chave = symbol)
                    if symbol in positions and positions[symbol].get("status") == "OPEN":
                        return {
                            "can_execute": False,
                            "reason": f"Ja existe uma posicao BUY aberta para {symbol}. Feche a posicao existente antes de abrir uma nova.",
                            "risk_level": "medium"
                        }
                    
                    # Verificar se existe posição SELL (chave = symbol_SHORT)
                    if f"{symbol}_SHORT" in positions and positions[f"{symbol}_SHORT"].get("status") == "OPEN":
                        return {
                            "can_execute": False,
                            "reason": f"Ja existe uma posicao SELL aberta para {symbol}. Feche a posicao existente antes de abrir uma nova.",
                            "risk_level": "medium"
                        }
        except Exception as e:
            # Se houver erro ao verificar, continuar (não bloquear)
            pass
        
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        confidence = signal.get('confidence', 5)
        
        # CORRIGIDO: Validar confiança antes de executar
        # Se escala 0-10: executar apenas se confiança >= 7
        # Se escala 0-5: executar apenas se confiança >= 3
        from config import settings
        
        if confidence > 5:
            # Escala 0-10
            min_confidence = settings.min_confidence_0_10
            confidence_scale = "0-10"
        else:
            # Escala 0-5
            min_confidence = settings.min_confidence_0_5
            confidence_scale = "0-5"
        
        if confidence < min_confidence:
            return {
                "can_execute": False,
                "reason": f"Confianca muito baixa: {confidence}/{confidence_scale} (minimo {min_confidence})",
                "risk_level": "medium",
                "confidence": confidence,
                "min_confidence": min_confidence
            }
        
        if not entry_price or not stop_loss:
            return {
                "can_execute": False,
                "reason": "Precos de entrada ou stop loss nao definidos",
                "risk_level": "high"
            }
        
        # Calcular risco
        risk_per_trade = abs(entry_price - stop_loss)
        risk_percentage = (risk_per_trade / entry_price) * 100
        
        # Circuit Breaker 1: Risco máximo por trade
        if risk_percentage > 3:  # Máximo 3% de risco por trade
            return {
                "can_execute": False,
                "reason": f"Risco muito alto: {risk_percentage:.2f}% (máximo 3%)",
                "risk_level": "high"
            }
        
        # Circuit Breaker 2: Verificar drawdown atual
        current_drawdown = _calculate_current_drawdown()
        if current_drawdown > 0.15:  # Máximo 15% de drawdown
            return {
                "can_execute": False,
                "reason": f"Drawdown atual muito alto: {current_drawdown:.2%} (máximo 15%)",
                "risk_level": "high"
            }
        
        # Circuit Breaker 3: Verificar exposição total
        total_exposure = _calculate_total_exposure()
        if total_exposure > account_balance * 0.1:  # Máximo 10% de exposição total
            return {
                "can_execute": False,
                "reason": f"Exposição total muito alta: {total_exposure:.2f} (máximo 10% do saldo)",
                "risk_level": "high"
            }
        
        # Circuit Breaker 4: Verificar limite diário de trades
        daily_trades = _get_daily_trades_count()
        if daily_trades >= 5:  # Máximo 5 trades por dia
            return {
                "can_execute": False,
                "reason": f"Limite diário de trades atingido: {daily_trades} (máximo 5)",
                "risk_level": "medium"
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
            "confidence_multiplier": confidence_multiplier,
            "current_drawdown": current_drawdown,
            "total_exposure": total_exposure,
            "daily_trades": daily_trades
        }
    except Exception as e:
        return {
            "can_execute": False,
            "reason": f"Erro na validação: {str(e)}",
            "risk_level": "high"
        }

async def backtest_strategy(symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Backtesting com dados históricos (FIXED: now async to avoid nested event loops).
    """
    try:
        from datetime import datetime, timedelta
        from binance_client import BinanceClient

        # Converter strings para datetime
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

        # Obter dados históricos usando o binance_client (agora com await direto)
        async with BinanceClient() as client:
            historical_data = await client.get_historical_klines(symbol, '1h', start_dt, end_dt)
        
        if historical_data.empty:
            return {"error": "Nenhum dado histórico encontrado"}
        
        # Simular estratégia
        results = []
        total_trades = 0
        winning_trades = 0
        total_return = 0
        
        # Análise simples baseada em SMA
        for i in range(20, len(historical_data)):
            current_price = historical_data['close'].iloc[i]
            sma_20 = historical_data['close'].iloc[i-20:i].mean()
            sma_50 = historical_data['close'].iloc[i-50:i].mean() if i >= 50 else sma_20
            
            # Sinal simples
            if current_price > sma_20 > sma_50:
                signal = "BUY"
                entry_price = current_price
                exit_price = historical_data['close'].iloc[min(i+24, len(historical_data)-1)]  # 24h depois
                pnl = (exit_price - entry_price) / entry_price
                
                results.append({
                    "timestamp": historical_data.index[i].isoformat(),
                    "signal": signal,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_percent": pnl * 100
                })
                
                total_trades += 1
                if pnl > 0:
                    winning_trades += 1
                total_return += pnl
        
        # Calcular métricas
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_return = total_return / total_trades if total_trades > 0 else 0
        
        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": win_rate,
            "total_return_percent": total_return * 100,
            "avg_return_percent": avg_return * 100,
            "results": results[-50:] if len(results) > 50 else results,  # Últimos 50 trades
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Erro no backtesting: {str(e)}",
            "symbol": symbol
        }

def execute_paper_trade(
    signal: Dict[str, Any],
    position_size: float
) -> Dict[str, Any]:
    """
    Executa um paper trade REAL usando o sistema completo de simulação.
    """
    try:
        from real_paper_trading import real_paper_trading
        
        # Executar trade usando o sistema REAL
        result = real_paper_trading.execute_trade(signal, position_size)
        
        if result["success"]:
            # Obter resumo do portfólio
            portfolio_summary = real_paper_trading.get_portfolio_summary()
            
            return {
                "success": True,
                "trade_id": result["trade_id"],
                "message": result["message"],
                "file": result["file"],
                "portfolio_summary": {
                    "current_balance": f"${portfolio_summary['current_balance']:.2f}",
                    "total_return": f"{portfolio_summary['total_return_percent']:.2f}%",
                    "open_positions": portfolio_summary['open_positions_count'],
                    "total_trades": portfolio_summary['total_trades']
                }
            }
        else:
            return result
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Erro ao executar paper trade: {str(e)}"
        }
