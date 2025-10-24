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
import os

# Importações para análise real do Twitter (opcional)
try:
    import tweepy
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    print("⚠️ Tweepy e VADER não disponíveis. Análise de sentimento será baseada apenas em dados de mercado.")

def get_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Obtém dados de mercado da Binance para análise.
    """
    try:
        base_url = "https://fapi.binance.com"
        timeout = 10  # Timeout aumentado para estabilidade
        
        # Ticker 24h
        ticker_response = requests.get(f"{base_url}/fapi/v1/ticker/24hr", params={'symbol': symbol}, timeout=timeout)
        ticker = ticker_response.json()
        
        # Klines para indicadores técnicos (100 klines para indicadores confiáveis)
        klines_response = requests.get(f"{base_url}/fapi/v1/klines", params={
            'symbol': symbol,
            'interval': '1h',
            'limit': 100  # Aumentado para 100 para indicadores confiáveis (SMA50)
        }, timeout=timeout)
        klines = klines_response.json()
        
        # Funding rate
        funding_response = requests.get(f"{base_url}/fapi/v1/premiumIndex", params={'symbol': symbol}, timeout=timeout)
        funding = funding_response.json()
        
        # Open interest
        oi_response = requests.get(f"{base_url}/fapi/v1/openInterest", params={'symbol': symbol}, timeout=timeout)
        open_interest = oi_response.json()
        
        # Processar todos os klines para indicadores técnicos confiáveis
        recent_klines = klines if len(klines) >= 20 else klines
        
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

def analyze_multiple_timeframes(symbol: str) -> Dict[str, Any]:
    """
    Análise multi-timeframe para maior precisão.
    """
    try:
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        analyses = {}
        
        for tf in timeframes:
            try:
                # Obter dados para timeframe específico
                base_url = "https://fapi.binance.com"
                response = requests.get(f"{base_url}/fapi/v1/klines", params={
                    'symbol': symbol,
                    'interval': tf,
                    'limit': 100
                }, timeout=5)
                
                if response.status_code == 200:
                    klines = response.json()
                    if len(klines) >= 20:
                        # Análise básica do timeframe
                        df = pd.DataFrame(klines, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
                            'taker_buy_quote_volume', 'ignore'
                        ])
                        
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                        
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
                            "current_price": current_price,
                            "sma_20": sma_20
                        }
            except Exception as e:
                print(f"⚠️ Erro no timeframe {tf}: {e}")
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

def analyze_order_flow(symbol: str) -> Dict[str, Any]:
    """
    Análise de fluxo de ordens e delta.
    """
    try:
        base_url = "https://fapi.binance.com"
        
        # Obter orderbook
        orderbook_response = requests.get(f"{base_url}/fapi/v1/depth", params={
            'symbol': symbol,
            'limit': 20
        }, timeout=5)
        
        if orderbook_response.status_code != 200:
            return {"error": "Erro ao obter orderbook"}
        
        orderbook = orderbook_response.json()
        
        # Calcular imbalance
        bid_volume = sum([float(b[1]) for b in orderbook['bids'][:20]])
        ask_volume = sum([float(a[1]) for a in orderbook['asks'][:20]])
        
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        # Obter trades recentes para CVD
        trades_response = requests.get(f"{base_url}/fapi/v1/aggTrades", params={
            'symbol': symbol,
            'limit': 100
        }, timeout=5)
        
        buy_volume = 0
        sell_volume = 0
        
        if trades_response.status_code == 200:
            trades = trades_response.json()
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
        return {
            "error": f"Erro na análise de order flow: {str(e)}",
            "symbol": symbol
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
        
        # Usar todos os klines disponíveis para indicadores técnicos confiáveis
        # klines já contém todos os dados necessários
        
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
    Análise REAL de sentimento do Twitter/X usando Tweepy e VADER.
    Se as bibliotecas não estiverem disponíveis, usa dados de mercado.
    """
    try:
        # Verificar se realmente pode usar Twitter
        if TWITTER_AVAILABLE and os.getenv("TWITTER_BEARER_TOKEN"):
            print(f"🐦 Usando análise REAL do Twitter para {symbol}")
            return _analyze_real_twitter_sentiment(symbol)
        else:
            if not TWITTER_AVAILABLE:
                print(f"⚠️ Bibliotecas Twitter não disponíveis. Usando análise baseada em dados de mercado para {symbol}")
            else:
                print(f"⚠️ Token Twitter não configurado. Usando análise baseada em dados de mercado para {symbol}")
            return _analyze_market_based_sentiment(symbol)
            
    except Exception as e:
        print(f"⚠️ Erro na análise de sentimento para {symbol}: {e}")
        return _analyze_market_based_sentiment(symbol)

def _analyze_real_twitter_sentiment(symbol: str) -> tuple:
    """
    Análise REAL de sentimento do X/Twitter usando Tweepy e VADER.
    """
    try:
        # Configurar Tweepy
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer_token:
            print("⚠️ TWITTER_BEARER_TOKEN não configurada. Usando análise baseada em dados de mercado.")
            return _analyze_market_based_sentiment(symbol)
        
        client = tweepy.Client(bearer_token=bearer_token)
        
        # Buscar tweets recentes
        query = f"#{symbol} OR ${symbol} -is:retweet lang:en"
        tweets = client.search_recent_tweets(
            query=query,
            max_results=100,
            tweet_fields=['created_at', 'public_metrics']
        )
        
        if not tweets.data:
            print(f"⚠️ Nenhum tweet encontrado para {symbol}. Usando análise baseada em dados de mercado.")
            return _analyze_market_based_sentiment(symbol)
        
        # Análise de sentimento com VADER
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        
        for tweet in tweets.data:
            scores = analyzer.polarity_scores(tweet.text)
            sentiments.append(scores['compound'])
        
        # Calcular média
        avg_sentiment = np.mean(sentiments)
        
        # Classificar
        if avg_sentiment >= 0.05:
            return ("positive", min(0.9, 0.5 + abs(avg_sentiment)))
        elif avg_sentiment <= -0.05:
            return ("negative", min(0.9, 0.5 + abs(avg_sentiment)))
        else:
            return ("neutral", 0.5)
            
    except Exception as e:
        print(f"⚠️ Erro no Twitter: {e}")
        return _analyze_market_based_sentiment(symbol)

def _analyze_market_based_sentiment(symbol: str) -> tuple:
    """
    Análise de sentimento baseada em dados reais de mercado (FALLBACK - não é Twitter real).
    """
    try:
        # Obter dados reais de mercado
        market_data = get_market_data(symbol)
        if "error" in market_data:
            return ("neutral", 0.5)
        
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
            return ("very_positive", min(0.95, confidence))
        elif sentiment_score >= 1:
            return ("positive", min(0.9, confidence))
        elif sentiment_score <= -3:
            return ("very_negative", min(0.95, confidence))
        elif sentiment_score <= -1:
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
    Análise real usando DeepSeek API através do AGNO Agent.
    Esta função prepara os dados estruturados para o AGNO Agent processar com DeepSeek.
    """
    try:
        # Preparar dados estruturados para análise
        analysis_data = {
            "market_data": market_data,
            "technical_indicators": technical_indicators,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat()
        }
        
        # Criar prompt estruturado para o AGNO Agent processar com DeepSeek
        prompt = f"""
        Analise os dados de mercado e forneça um sinal de trading:
        
        DADOS DE MERCADO:
        - Símbolo: {market_data.get('symbol', 'N/A')}
        - Preço atual: ${market_data.get('current_price', 0):.2f}
        - Variação 24h: {market_data.get('price_change_24h', 0):.2f}%
        - Volume 24h: ${market_data.get('volume_24h', 0):.0f}
        - Funding Rate: {market_data.get('funding_rate', 0):.4f}
        - Open Interest: ${market_data.get('open_interest', 0):.0f}
        
        INDICADORES TÉCNICOS:
        - RSI: {technical_indicators.get('indicators', {}).get('rsi', 50):.2f}
        - MACD: {technical_indicators.get('indicators', {}).get('macd', 0):.2f}
        - ADX: {technical_indicators.get('indicators', {}).get('adx', 25):.2f}
        - ATR: {technical_indicators.get('indicators', {}).get('atr', 0):.2f}
        - Tendência: {technical_indicators.get('trend', 'neutral')}
        - Momentum: {technical_indicators.get('momentum', 'neutral')}
        - Suporte: ${technical_indicators.get('support', 0):.2f}
        - Resistência: ${technical_indicators.get('resistance', 0):.2f}
        
        ESTRUTURA DE MERCADO:
        - Estrutura: {technical_indicators.get('market_structure', {}).get('structure', 'N/A')}
        - Força: {technical_indicators.get('market_structure', {}).get('strength', 'N/A')}
        - Nível Suporte: ${technical_indicators.get('market_structure', {}).get('support_level', 0):.2f}
        - Nível Resistência: ${technical_indicators.get('market_structure', {}).get('resistance_level', 0):.2f}
        
        SENTIMENTO:
        - Sentimento geral: {sentiment.get('sentiment', 'neutral')}
        - Confiança: {sentiment.get('confidence', 0.5):.2f}
        - Fatores: {sentiment.get('factors', {})}
        
        Forneça uma análise detalhada e um sinal de trading com:
        1. Sinal: BUY ou SELL (seja decisivo)
        2. Entrada: [preço específico]
        3. Stop Loss: [preço específico]
        4. Take Profit 1: [preço específico]
        5. Take Profit 2: [preço específico]
        6. Confiança: [1-10]
        7. Justificativa técnica detalhada
        """
        
        return {
            "analysis_data": analysis_data,
            "deepseek_prompt": prompt,
            "needs_agent_processing": True,
            "message": "Dados preparados para análise DeepSeek - AGNO Agent deve processar com DeepSeek",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Erro na preparação para DeepSeek: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def _calculate_current_drawdown() -> float:
    """
    Calcula o drawdown atual baseado no histórico de trades.
    """
    try:
        from paper_trading import paper_trading
        summary = paper_trading.get_portfolio_summary()
        return max(0, (summary['initial_balance'] - summary['total_portfolio_value']) / summary['initial_balance'])
    except:
        return 0.0

def _calculate_total_exposure() -> float:
    """
    Calcula a exposição total atual.
    """
    try:
        from paper_trading import paper_trading
        summary = paper_trading.get_portfolio_summary()
        return summary['open_positions_value']
    except:
        return 0.0

def _get_daily_trades_count() -> int:
    """
    Retorna o número de trades executados hoje.
    """
    try:
        from paper_trading import paper_trading
        today = datetime.now().date()
        trades = paper_trading.get_trade_history()
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

def backtest_strategy(symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Backtesting com dados históricos.
    """
    try:
        from datetime import datetime, timedelta
        
        # Converter strings para datetime
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Obter dados históricos usando o binance_client
        from binance_client import BinanceClient
        import asyncio
        
        async def get_historical_data():
            async with BinanceClient() as client:
                return await client.get_historical_klines(symbol, '1h', start_dt, end_dt)
        
        # Executar async
        historical_data = asyncio.run(get_historical_data())
        
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
