"""
Sistema de gestão de risco para trading
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class RiskParameters:
    """Parâmetros de risco configuráveis"""
    max_risk_per_trade: float = 0.02  # 2% máximo por trade
    max_daily_loss: float = 0.05  # 5% perda máxima diária
    max_concurrent_positions: int = 3  # Máximo de posições simultâneas
    max_correlation: float = 0.7  # Correlação máxima entre posições
    max_position_size: float = 0.1  # 10% máximo por posição
    min_confidence: int = 6  # Confiança mínima para executar trade

@dataclass
class Portfolio:
    """Estado atual do portfólio"""
    balance: float
    positions: List[Dict]
    daily_pnl: float
    total_pnl: float
    last_reset: datetime

class RiskManagementSystem:
    """Sistema de gestão de risco"""
    
    def __init__(self, risk_params: RiskParameters = None):
        self.params = risk_params or RiskParameters()
        self.portfolio = None
        
    def set_portfolio(self, portfolio: Portfolio):
        """Define o estado atual do portfólio"""
        self.portfolio = portfolio
    
    def calculate_position_size(self, 
                               entry_price: float,
                               stop_loss: float,
                               confidence: int) -> float:
        """
        Calcula tamanho da posição baseado em Kelly Criterion modificado
        
        Args:
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
            confidence: Nível de confiança (1-10)
            
        Returns:
            Tamanho da posição em unidades
        """
        if not self.portfolio:
            return 0.0
            
        # Calcular risco por unidade
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return 0.0
            
        # Risco máximo em valor
        max_risk_amount = self.portfolio.balance * self.params.max_risk_per_trade
        
        # Ajustar risco baseado na confiança
        confidence_multiplier = confidence / 10.0
        adjusted_risk = max_risk_amount * confidence_multiplier
        
        # Tamanho baseado no risco
        position_size = adjusted_risk / price_risk
        
        # Aplicar Kelly Criterion (versão conservadora)
        win_probability = 0.5 + (confidence - 5) * 0.05  # 0.5 a 0.75
        win_loss_ratio = 2.0  # Ratio médio de ganho/perda
        
        if win_loss_ratio > 0:
            kelly_percentage = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
            kelly_percentage = max(0, min(kelly_percentage, 0.25))  # Limitar a 25%
            
            kelly_size = self.portfolio.balance * kelly_percentage / entry_price
            position_size = min(position_size, kelly_size)
        
        # Limitar tamanho máximo da posição
        max_position_value = self.portfolio.balance * self.params.max_position_size
        max_position_units = max_position_value / entry_price
        
        return min(position_size, max_position_units)
    
    def validate_trade_risk(self, trade_signal: Dict, symbol: str) -> Dict:
        """
        Valida se o trade está dentro dos parâmetros de risco
        
        Args:
            trade_signal: Sinal de trading
            symbol: Símbolo do ativo
            
        Returns:
            Dicionário com validações e ajustes
        """
        validations = {
            'can_execute': True,
            'position_size_ok': True,
            'daily_loss_ok': True,
            'correlation_ok': True,
            'confidence_ok': True,
            'volatility_ok': True,
            'adjustments': [],
            'warnings': []
        }
        
        if not self.portfolio:
            validations['can_execute'] = False
            validations['adjustments'].append("Portfólio não inicializado")
            return validations
        
        # Verificar confiança mínima
        if trade_signal.get('confidence', 0) < self.params.min_confidence:
            validations['confidence_ok'] = False
            validations['can_execute'] = False
            validations['adjustments'].append(f"Confiança muito baixa: {trade_signal.get('confidence', 0)}/10")
        
        # Verificar perda diária
        if self.portfolio.daily_pnl <= -self.params.max_daily_loss * self.portfolio.balance:
            validations['daily_loss_ok'] = False
            validations['can_execute'] = False
            validations['adjustments'].append("Limite de perda diária atingido")
        
        # Verificar número de posições
        if len(self.portfolio.positions) >= self.params.max_concurrent_positions:
            validations['can_execute'] = False
            validations['adjustments'].append(f"Máximo de posições atingido: {len(self.portfolio.positions)}/{self.params.max_concurrent_positions}")
        
        # Verificar correlação com posições existentes
        if self.portfolio.positions:
            correlation = self._calculate_correlation(symbol, self.portfolio.positions)
            if correlation > self.params.max_correlation:
                validations['correlation_ok'] = False
                validations['warnings'].append(f"Alta correlação com posições existentes: {correlation:.2f}")
        
        # Verificar volatilidade
        atr = trade_signal.get('atr', 0)
        if atr > self.portfolio.balance * 0.001:  # ATR > 0.1% do saldo
            validations['volatility_ok'] = False
            validations['warnings'].append("Alta volatilidade detectada")
        
        return validations
    
    def _calculate_correlation(self, symbol: str, positions: List[Dict]) -> float:
        """
        Calcula correlação média com posições existentes
        (Simplificado - em produção, usar dados históricos reais)
        """
        # Mapeamento simplificado de correlações conhecidas
        correlation_map = {
            'BTCUSDT': {'ETHUSDT': 0.8, 'BNBUSDT': 0.6, 'ADAUSDT': 0.7},
            'ETHUSDT': {'BTCUSDT': 0.8, 'BNBUSDT': 0.7, 'ADAUSDT': 0.6},
            'BNBUSDT': {'BTCUSDT': 0.6, 'ETHUSDT': 0.7, 'ADAUSDT': 0.5}
        }
        
        if symbol not in correlation_map:
            return 0.3  # Correlação baixa para ativos não mapeados
        
        correlations = []
        for position in positions:
            pos_symbol = position.get('symbol', '')
            if pos_symbol in correlation_map[symbol]:
                correlations.append(correlation_map[symbol][pos_symbol])
        
        return np.mean(correlations) if correlations else 0.3
    
    def calculate_stop_loss(self, entry_price: float, signal_type: str, atr: float = None) -> float:
        """
        Calcula stop loss baseado em ATR ou porcentagem
        
        Args:
            entry_price: Preço de entrada
            signal_type: Tipo de sinal (BUY/SELL)
            atr: Average True Range (opcional)
            
        Returns:
            Preço do stop loss
        """
        if atr:
            # Usar ATR para stop loss dinâmico
            stop_distance = atr * 2.0  # 2x ATR
        else:
            # Usar porcentagem fixa
            stop_distance = entry_price * self.params.max_risk_per_trade * 5  # 10% do preço
        
        if signal_type == 'BUY':
            return entry_price - stop_distance
        else:  # SELL
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             risk_reward_ratio: float = 2.0) -> Tuple[float, float]:
        """
        Calcula níveis de take profit baseado no risk/reward
        
        Args:
            entry_price: Preço de entrada
            stop_loss: Preço do stop loss
            risk_reward_ratio: Ratio risco/recompensa
            
        Returns:
            Tuple com take profit 1 e take profit 2
        """
        risk = abs(entry_price - stop_loss)
        
        # Take profit 1: 1:1 risk/reward
        tp1 = entry_price + (risk * 1.0) if entry_price > stop_loss else entry_price - (risk * 1.0)
        
        # Take profit 2: Ratio configurado
        tp2 = entry_price + (risk * risk_reward_ratio) if entry_price > stop_loss else entry_price - (risk * risk_reward_ratio)
        
        return tp1, tp2
    
    def should_close_position(self, position: Dict, current_price: float) -> bool:
        """
        Verifica se uma posição deve ser fechada por gestão de risco
        
        Args:
            position: Dados da posição
            current_price: Preço atual
            
        Returns:
            True se deve fechar a posição
        """
        # Verificar stop loss
        if position.get('stop_loss'):
            if position['side'] == 'BUY' and current_price <= position['stop_loss']:
                return True
            elif position['side'] == 'SELL' and current_price >= position['stop_loss']:
                return True
        
        # Verificar take profit
        if position.get('take_profit_1'):
            if position['side'] == 'BUY' and current_price >= position['take_profit_1']:
                return True
            elif position['side'] == 'SELL' and current_price <= position['take_profit_1']:
                return True
        
        # Verificar tempo máximo de posição (24 horas)
        if 'timestamp' in position:
            position_time = datetime.fromisoformat(position['timestamp'])
            if datetime.now() - position_time > timedelta(hours=24):
                return True
        
        return False
    
    def get_risk_summary(self) -> Dict:
        """
        Retorna resumo do risco atual
        
        Returns:
            Dicionário com métricas de risco
        """
        if not self.portfolio:
            return {'error': 'Portfólio não inicializado'}
        
        total_exposure = sum(pos.get('size', 0) * pos.get('entry_price', 0) 
                           for pos in self.portfolio.positions)
        
        return {
            'balance': self.portfolio.balance,
            'total_pnl': self.portfolio.total_pnl,
            'daily_pnl': self.portfolio.daily_pnl,
            'active_positions': len(self.portfolio.positions),
            'total_exposure': total_exposure,
            'exposure_percentage': (total_exposure / self.portfolio.balance) * 100,
            'risk_limits': {
                'max_risk_per_trade': self.params.max_risk_per_trade * 100,
                'max_daily_loss': self.params.max_daily_loss * 100,
                'max_position_size': self.params.max_position_size * 100
            }
        }
