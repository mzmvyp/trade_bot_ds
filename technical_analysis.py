"""
Módulo de análise técnica para trading
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos os indicadores técnicos
        """
        # Dados básicos - garantir que são float
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        close = df['close'].astype(float).values
        volume = df['volume'].astype(float).values
        
        # Médias móveis
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['sma_200'] = talib.SMA(close, timeperiod=200)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Williams %R
        df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # ATR (Average True Range)
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)
        
        # Volume indicators
        df['obv'] = talib.OBV(close, volume)
        
        # Suporte e Resistência
        df['support'] = self._calculate_support_resistance(df, 'support')
        df['resistance'] = self._calculate_support_resistance(df, 'resistance')
        
        return df
    
    def _calculate_support_resistance(self, df: pd.DataFrame, level_type: str) -> pd.Series:
        """
        Calcula níveis de suporte e resistência
        """
        window = 20
        if level_type == 'support':
            return df['low'].rolling(window=window).min()
        else:
            return df['high'].rolling(window=window).max()
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        Analisa a tendência atual
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Tendência das médias móveis
        sma_trend = "bullish" if latest['sma_20'] > latest['sma_50'] > latest['sma_200'] else "bearish"
        
        # MACD
        macd_signal = "bullish" if latest['macd'] > latest['macd_signal'] else "bearish"
        
        # RSI
        rsi_signal = "overbought" if latest['rsi'] > 70 else "oversold" if latest['rsi'] < 30 else "neutral"
        
        # Bollinger Bands
        bb_signal = "overbought" if latest['close'] > latest['bb_upper'] else "oversold" if latest['close'] < latest['bb_lower'] else "neutral"
        
        # ADX (força da tendência)
        adx_strength = "strong" if latest['adx'] > 25 else "weak"
        
        return {
            'sma_trend': sma_trend,
            'macd_signal': macd_signal,
            'rsi_signal': rsi_signal,
            'bb_signal': bb_signal,
            'adx_strength': adx_strength,
            'current_price': latest['close'],
            'sma_20': latest['sma_20'],
            'sma_50': latest['sma_50'],
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'atr': latest['atr']
        }
    
    def calculate_support_resistance_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calcula níveis de suporte e resistência
        """
        # Usar últimos 100 candles para análise
        recent_data = df.tail(100)
        
        # Encontrar máximos e mínimos locais
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Suporte (mínimos locais)
        support_levels = []
        for i in range(2, len(lows)-2):
            if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and
                lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]):
                support_levels.append(lows.iloc[i])
        
        # Resistência (máximos locais)
        resistance_levels = []
        for i in range(2, len(highs)-2):
            if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and
                highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
                resistance_levels.append(highs.iloc[i])
        
        # Ordenar e pegar os mais relevantes
        support_levels = sorted(support_levels, reverse=True)[:3]
        resistance_levels = sorted(resistance_levels)[:3]
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    def generate_trading_signal(self, df: pd.DataFrame) -> Dict:
        """
        Gera sinal de trading baseado na análise técnica
        """
        analysis = self.analyze_trend(df)
        levels = self.calculate_support_resistance_levels(df)
        
        current_price = analysis['current_price']
        atr = analysis['atr']
        
        # Lógica de sinal
        signal_strength = 0
        signal_type = "HOLD"
        
        # Fatores bullish
        if analysis['sma_trend'] == "bullish":
            signal_strength += 1
        if analysis['macd_signal'] == "bullish":
            signal_strength += 1
        if analysis['rsi_signal'] == "oversold":
            signal_strength += 1
        if analysis['bb_signal'] == "oversold":
            signal_strength += 1
        if analysis['adx_strength'] == "strong":
            signal_strength += 1
        
        # Fatores bearish
        if analysis['sma_trend'] == "bearish":
            signal_strength -= 1
        if analysis['macd_signal'] == "bearish":
            signal_strength -= 1
        if analysis['rsi_signal'] == "overbought":
            signal_strength -= 1
        if analysis['bb_signal'] == "overbought":
            signal_strength -= 1
        
        # Determinar tipo de sinal
        if signal_strength >= 3:
            signal_type = "BUY"
        elif signal_strength <= -3:
            signal_type = "SELL"
        
        # Calcular níveis de entrada, stop e alvos
        if signal_type == "BUY":
            entry_price = current_price
            stop_loss = entry_price - (atr * 2)
            target1 = entry_price + (atr * 2)
            target2 = entry_price + (atr * 4)
        elif signal_type == "SELL":
            entry_price = current_price
            stop_loss = entry_price + (atr * 2)
            target1 = entry_price - (atr * 2)
            target2 = entry_price - (atr * 4)
        else:
            entry_price = current_price
            stop_loss = None
            target1 = None
            target2 = None
        
        return {
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target1': target1,
            'target2': target2,
            'analysis': analysis,
            'levels': levels,
            'timestamp': df.index[-1]
        }
