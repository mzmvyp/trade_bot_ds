"""
Análise técnica aprimorada com indicadores avançados
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib

class EnhancedTechnicalAnalyzer:
    """Analisador técnico aprimorado com indicadores avançados"""
    
    def __init__(self):
        # Pesos dos indicadores por categoria
        self.indicator_weights = {
            'trend': 0.30,      # Tendência
            'momentum': 0.25,    # Momentum
            'volatility': 0.20,  # Volatilidade
            'volume': 0.15,      # Volume
            'structure': 0.10    # Estrutura de mercado
        }
        
        # Configurações dos indicadores
        self.indicator_configs = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'adx_period': 14,
            'cci_period': 20,
            'williams_period': 14
        }
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos avançados"""
        
        # Garantir que temos dados suficientes
        if len(df) < 50:
            return df
        
        # 1. INDICADORES DE TENDÊNCIA
        df = self._calculate_trend_indicators(df)
        
        # 2. INDICADORES DE MOMENTUM
        df = self._calculate_momentum_indicators(df)
        
        # 3. INDICADORES DE VOLATILIDADE
        df = self._calculate_volatility_indicators(df)
        
        # 4. INDICADORES DE VOLUME
        df = self._calculate_volume_indicators(df)
        
        # 5. ESTRUTURA DE MERCADO
        df = self._calculate_market_structure(df)
        
        # 6. INDICADORES AVANÇADOS
        df = self._calculate_advanced_indicators_custom(df)
        
        return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores de tendência"""
        
        # Médias móveis
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=self.indicator_configs['macd_fast'],
            slowperiod=self.indicator_configs['macd_slow'],
            signalperiod=self.indicator_configs['macd_signal']
        )
        
        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.indicator_configs['adx_period'])
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=self.indicator_configs['adx_period'])
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=self.indicator_configs['adx_period'])
        
        # Ichimoku Cloud (simplificado)
        df = self._calculate_ichimoku(df)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores de momentum"""
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.indicator_configs['rsi_period'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=self.indicator_configs['williams_period'])
        
        # CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=self.indicator_configs['cci_period'])
        
        # Rate of Change
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        
        # Momentum
        df['momentum'] = talib.MOM(df['close'], timeperiod=10)
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores de volatilidade"""
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=self.indicator_configs['bb_period'],
            nbdevup=self.indicator_configs['bb_std'],
            nbdevdn=self.indicator_configs['bb_std']
        )
        
        # ATR (Average True Range)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.indicator_configs['atr_period'])
        
        # Volatilidade histórica
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(365)
        
        # Keltner Channels
        df['kc_upper'] = df['ema_20'] + (2 * df['atr'])
        df['kc_middle'] = df['ema_20']
        df['kc_lower'] = df['ema_20'] - (2 * df['atr'])
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores de volume"""
        
        # Volume médio
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Volume relativo
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # OBV (On Balance Volume)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Volume Delta
        df['volume_delta'] = df['volume'].diff()
        
        # Money Flow Index
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        return df
    
    def _calculate_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula estrutura de mercado"""
        
        # Suporte e Resistência
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        
        # Higher Highs e Lower Lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        
        # Pivot Points
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        
        # Fractals
        df['fractal_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2)) & \
                            (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))
        df['fractal_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2)) & \
                           (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2))
        
        return df
    
    def _calculate_advanced_indicators_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores avançados customizados"""
        
        # Divergências RSI
        df['rsi_divergence'] = self._detect_divergence(df['close'], df['rsi'])
        
        # Order Flow Imbalance
        df['ofi'] = self._calculate_order_flow_imbalance(df)
        
        # Market Regime
        df['market_regime'] = self._calculate_market_regime(df)
        
        # Trend Strength
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Volatility Regime
        df['volatility_regime'] = self._calculate_volatility_regime(df)
        
        return df
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula Ichimoku Cloud"""
        
        # Tenkan-sen (Conversion Line)
        df['ichimoku_tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        
        # Kijun-sen (Base Line)
        df['ichimoku_kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        df['ichimoku_senkou_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['ichimoku_chikou'] = df['close'].shift(-26)
        
        return df
    
    def _detect_divergence(self, price: pd.Series, indicator: pd.Series) -> pd.Series:
        """Detecta divergências entre preço e indicador"""
        
        # Encontrar picos e vales
        price_peaks = (price.shift(1) < price) & (price > price.shift(-1))
        price_troughs = (price.shift(1) > price) & (price < price.shift(-1))
        
        indicator_peaks = (indicator.shift(1) < indicator) & (indicator > indicator.shift(-1))
        indicator_troughs = (indicator.shift(1) > indicator) & (indicator < indicator.shift(-1))
        
        # Detectar divergências
        bullish_div = (price_troughs & (indicator > indicator.shift(5))) | \
                     (price_troughs & (indicator < indicator.shift(-5)))
        
        bearish_div = (price_peaks & (indicator < indicator.shift(5))) | \
                     (price_peaks & (indicator > indicator.shift(-5)))
        
        divergence = pd.Series(0, index=price.index)
        divergence[bullish_div] = 1
        divergence[bearish_div] = -1
        
        return divergence
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calcula Order Flow Imbalance"""
        
        # Simplificado - em produção, usar dados de ordem real
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        
        # OFI baseado em correlação preço-volume
        ofi = price_change * volume_change
        ofi = ofi.rolling(window=10).mean()
        
        return ofi
    
    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Calcula regime de mercado (tendência vs lateral)"""
        
        # Usar ADX para determinar força da tendência
        adx = df['adx']
        
        regime = pd.Series('lateral', index=df.index)
        regime[adx > 25] = 'tendencia'
        regime[adx < 20] = 'lateral'
        
        return regime
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calcula força da tendência"""
        
        # Combinar múltiplos indicadores
        rsi_strength = np.abs(df['rsi'] - 50) / 50
        macd_strength = np.abs(df['macd_hist']) / df['macd_hist'].rolling(20).std()
        adx_strength = df['adx'] / 100
        
        # Média ponderada
        trend_strength = (rsi_strength * 0.3 + macd_strength * 0.4 + adx_strength * 0.3)
        
        return trend_strength.fillna(0)
    
    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Calcula regime de volatilidade"""
        
        atr = df['atr']
        atr_ma = atr.rolling(20).mean()
        
        regime = pd.Series('normal', index=df.index)
        regime[atr > atr_ma * 1.5] = 'alta'
        regime[atr < atr_ma * 0.5] = 'baixa'
        
        return regime
    
    def generate_technical_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """Gera sinais técnicos baseados em todos os indicadores"""
        
        if len(df) < 50:
            return self._generate_basic_signals(df)
        
        signals = {}
        
        # 1. Sinais de Tendência
        signals.update(self._analyze_trend_signals(df))
        
        # 2. Sinais de Momentum
        signals.update(self._analyze_momentum_signals(df))
        
        # 3. Sinais de Volatilidade
        signals.update(self._analyze_volatility_signals(df))
        
        # 4. Sinais de Volume
        signals.update(self._analyze_volume_signals(df))
        
        # 5. Sinais de Estrutura
        signals.update(self._analyze_structure_signals(df))
        
        # 6. Sinal Combinado
        signals['combined_signal'] = self._calculate_combined_signal(signals)
        
        return signals
    
    def _generate_basic_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """Gera sinais básicos quando há poucos dados"""
        return {
            'trend': 'neutral',
            'momentum': 'neutral',
            'volatility': 'normal',
            'volume': 'normal',
            'structure': 'neutral',
            'combined_signal': 'HOLD',
            'confidence': 3
        }
    
    def _analyze_trend_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analisa sinais de tendência"""
        current_price = df['close'].iloc[-1]
        
        # Médias móveis
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        ema_12 = df['ema_12'].iloc[-1]
        ema_26 = df['ema_26'].iloc[-1]
        
        # ADX
        adx = df['adx'].iloc[-1]
        
        # Determinar tendência
        if current_price > sma_20 > sma_50 and ema_12 > ema_26 and adx > 25:
            trend = 'bullish'
        elif current_price < sma_20 < sma_50 and ema_12 < ema_26 and adx > 25:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'trend': trend,
            'trend_strength': adx,
            'ma_alignment': current_price > sma_20 > sma_50
        }
    
    def _analyze_momentum_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analisa sinais de momentum"""
        rsi = df['rsi'].iloc[-1]
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        stoch_k = df['stoch_k'].iloc[-1]
        
        # Determinar momentum
        if rsi > 50 and macd > macd_signal and stoch_k > 50:
            momentum = 'bullish'
        elif rsi < 50 and macd < macd_signal and stoch_k < 50:
            momentum = 'bearish'
        else:
            momentum = 'neutral'
        
        return {
            'momentum': momentum,
            'rsi': rsi,
            'macd_bullish': macd > macd_signal,
            'stoch_bullish': stoch_k > 50
        }
    
    def _analyze_volatility_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analisa sinais de volatilidade"""
        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Posição nas Bollinger Bands
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        # Regime de volatilidade
        atr_ma = df['atr'].rolling(20).mean().iloc[-1]
        if atr > atr_ma * 1.5:
            volatility_regime = 'high'
        elif atr < atr_ma * 0.5:
            volatility_regime = 'low'
        else:
            volatility_regime = 'normal'
        
        return {
            'volatility': volatility_regime,
            'bb_position': bb_position,
            'atr': atr,
            'bb_squeeze': (bb_upper - bb_lower) / current_price < 0.1
        }
    
    def _analyze_volume_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analisa sinais de volume"""
        volume_ratio = df['volume_ratio'].iloc[-1]
        obv = df['obv'].iloc[-1]
        obv_ma = df['obv'].rolling(20).mean().iloc[-1]
        mfi = df['mfi'].iloc[-1]
        
        # Determinar sinais de volume
        if volume_ratio > 1.5 and obv > obv_ma and mfi > 50:
            volume_signal = 'bullish'
        elif volume_ratio > 1.5 and obv < obv_ma and mfi < 50:
            volume_signal = 'bearish'
        else:
            volume_signal = 'neutral'
        
        return {
            'volume': volume_signal,
            'volume_ratio': volume_ratio,
            'obv_bullish': obv > obv_ma,
            'mfi': mfi
        }
    
    def _analyze_structure_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analisa sinais de estrutura"""
        current_price = df['close'].iloc[-1]
        support = df['support'].iloc[-1]
        resistance = df['resistance'].iloc[-1]
        pivot = df['pivot'].iloc[-1]
        
        # Posição em relação a suporte/resistência
        if current_price > resistance:
            structure = 'breakout'
        elif current_price < support:
            structure = 'breakdown'
        elif current_price > pivot:
            structure = 'bullish'
        else:
            structure = 'bearish'
        
        return {
            'structure': structure,
            'support': support,
            'resistance': resistance,
            'pivot': pivot,
            'distance_to_support': (current_price - support) / current_price,
            'distance_to_resistance': (resistance - current_price) / current_price
        }
    
    def _calculate_combined_signal(self, signals: Dict[str, any]) -> Dict[str, any]:
        """Calcula sinal combinado baseado em todos os indicadores"""
        
        # Pontuação por categoria
        trend_score = 1 if signals.get('trend') == 'bullish' else -1 if signals.get('trend') == 'bearish' else 0
        momentum_score = 1 if signals.get('momentum') == 'bullish' else -1 if signals.get('momentum') == 'bearish' else 0
        volume_score = 1 if signals.get('volume') == 'bullish' else -1 if signals.get('volume') == 'bearish' else 0
        
        # Calcular score total
        total_score = (trend_score * self.indicator_weights['trend'] + 
                      momentum_score * self.indicator_weights['momentum'] + 
                      volume_score * self.indicator_weights['volume'])
        
        # Determinar sinal
        if total_score > 0.3:
            signal = 'BUY'
            confidence = min(10, int(5 + total_score * 5))
        elif total_score < -0.3:
            signal = 'SELL'
            confidence = min(10, int(5 + abs(total_score) * 5))
        else:
            signal = 'HOLD'
            confidence = 5
        
        return {
            'signal': signal,
            'confidence': confidence,
            'total_score': total_score,
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'volume_score': volume_score
        }
