"""
Sistema de backtesting para estratégias de trading
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from binance_client import BinanceClient
from technical_analysis import EnhancedTechnicalAnalyzer
from risk_management import RiskManagementSystem, RiskParameters, Portfolio

@dataclass
class Trade:
    """Representa uma operação de trading"""
    symbol: str
    side: str  # 'BUY' ou 'SELL'
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    return_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""

@dataclass
class BacktestResult:
    """Resultado do backtest"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_trade_duration: float
    trades: List[Trade]

class BacktestingEngine:
    """Motor de backtesting para estratégias de trading"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.binance_client = BinanceClient()
        self.technical_analyzer = EnhancedTechnicalAnalyzer()
        self.risk_manager = RiskManagementSystem()
        
        # Configurações
        self.commission_rate = 0.001  # 0.1% por trade
        self.slippage_rate = 0.0005  # 0.05% slippage
        
    async def backtest_strategy(self, 
                              symbol: str, 
                              start_date: str, 
                              end_date: str,
                              strategy_params: Dict[str, Any] = None) -> BacktestResult:
        """
        Executa backtest de uma estratégia
        
        Args:
            symbol: Símbolo para backtest
            start_date: Data de início (YYYY-MM-DD)
            end_date: Data de fim (YYYY-MM-DD)
            strategy_params: Parâmetros da estratégia
            
        Returns:
            Resultado do backtest
        """
        print(f"🚀 Iniciando backtest para {symbol}")
        print(f"📅 Período: {start_date} até {end_date}")
        print(f"💰 Capital inicial: ${self.initial_capital:,.2f}")
        
        # Obter dados históricos
        historical_data = await self._get_historical_data(symbol, start_date, end_date)
        
        if historical_data.empty:
            raise ValueError(f"Nenhum dado histórico encontrado para {symbol}")
        
        print(f"📊 Dados obtidos: {len(historical_data)} candles")
        
        # Configurar portfólio inicial
        portfolio = Portfolio(
            balance=self.initial_capital,
            positions=[],
            daily_pnl=0.0,
            total_pnl=0.0,
            last_reset=datetime.now()
        )
        
        self.risk_manager.set_portfolio(portfolio)
        
        # Executar backtest
        trades = await self._run_backtest(historical_data, symbol, strategy_params or {})
        
        # Calcular métricas
        result = self._calculate_metrics(trades, self.initial_capital)
        
        print(f"✅ Backtest concluído!")
        print(f"📈 Total de trades: {result.total_trades}")
        print(f"🎯 Taxa de acerto: {result.win_rate:.1f}%")
        print(f"💰 Retorno total: {result.total_return:.1f}%")
        print(f"📉 Drawdown máximo: {result.max_drawdown:.1f}%")
        
        return result
    
    async def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Obtém dados históricos da Binance"""
        
        # Converter datas
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Calcular número de dias
        days_diff = (end_dt - start_dt).days
        
        # Escolher timeframe baseado no período
        if days_diff <= 7:
            interval = '1h'
        elif days_diff <= 30:
            interval = '4h'
        else:
            interval = '1d'
        
        print(f"📊 Usando timeframe: {interval}")
        
        # Obter dados
        async with self.binance_client:
            data = await self.binance_client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_dt,
                end_time=end_dt
            )
        
        if not data:
            return pd.DataFrame()
        
        # Converter para DataFrame
        df = pd.DataFrame(data)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                     'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        
        # Converter tipos
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calcular indicadores técnicos
        df = self.technical_analyzer.calculate_advanced_indicators(df)
        
        return df
    
    async def _run_backtest(self, data: pd.DataFrame, symbol: str, strategy_params: Dict) -> List[Trade]:
        """Executa o backtest"""
        
        trades = []
        current_position = None
        
        for i, (timestamp, candle) in data.iterrows():
            # Pular se não temos dados suficientes
            if i < 50:
                continue
            
            # Verificar se devemos fechar posição atual
            if current_position:
                should_close, reason = self._should_close_position(current_position, candle, timestamp)
                if should_close:
                    trade = self._close_position(current_position, candle, timestamp, reason)
                    trades.append(trade)
                    current_position = None
            
            # Gerar sinal se não temos posição
            if not current_position:
                signal = await self._generate_signal(candle, data.iloc[:i+1], strategy_params)
                
                if signal and signal.get('signal') in ['BUY', 'SELL']:
                    # Validar risco
                    risk_validation = self.risk_manager.validate_trade_risk(signal, symbol)
                    
                    if risk_validation['can_execute']:
                        current_position = self._open_position(signal, candle, timestamp)
        
        # Fechar posição final se existir
        if current_position:
            last_candle = data.iloc[-1]
            trade = self._close_position(current_position, last_candle, data.index[-1], "Fim do período")
            trades.append(trade)
        
        return trades
    
    async def _generate_signal(self, candle: pd.Series, historical_data: pd.DataFrame, params: Dict) -> Optional[Dict]:
        """Gera sinal de trading"""
        
        # Usar análise técnica para gerar sinal
        technical_signals = self.technical_analyzer.generate_technical_signals(historical_data)
        
        if not technical_signals.get('combined_signal'):
            return None
        
        signal_data = technical_signals['combined_signal']
        
        if signal_data['signal'] == 'HOLD':
            return None
        
        # Calcular níveis de entrada, stop loss e take profit
        entry_price = candle['close']
        atr = candle.get('atr', entry_price * 0.02)  # 2% se ATR não disponível
        
        # Stop loss baseado em ATR
        if signal_data['signal'] == 'BUY':
            stop_loss = entry_price - (atr * 2)
            take_profit = entry_price + (atr * 4)  # Risk:Reward 1:2
        else:  # SELL
            stop_loss = entry_price + (atr * 2)
            take_profit = entry_price - (atr * 4)
        
        return {
            'signal': signal_data['signal'],
            'confidence': signal_data['confidence'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasoning': {
                'technical': f"Score: {signal_data['total_score']:.2f}",
                'trend': technical_signals.get('trend', 'neutral'),
                'momentum': technical_signals.get('momentum', 'neutral'),
                'volume': technical_signals.get('volume', 'neutral')
            }
        }
    
    def _open_position(self, signal: Dict, candle: pd.Series, timestamp: datetime) -> Dict:
        """Abre uma nova posição"""
        
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        
        # Calcular tamanho da posição
        position_size = self.risk_manager.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=signal['confidence']
        )
        
        return {
            'symbol': 'BTCUSDT',  # Simplificado
            'side': signal['signal'],
            'entry_price': entry_price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': signal['take_profit'],
            'entry_time': timestamp,
            'confidence': signal['confidence']
        }
    
    def _should_close_position(self, position: Dict, candle: pd.Series, timestamp: datetime) -> Tuple[bool, str]:
        """Verifica se deve fechar a posição"""
        
        current_price = candle['close']
        
        # Verificar stop loss
        if position['side'] == 'BUY' and current_price <= position['stop_loss']:
            return True, "Stop loss atingido"
        elif position['side'] == 'SELL' and current_price >= position['stop_loss']:
            return True, "Stop loss atingido"
        
        # Verificar take profit
        if position['side'] == 'BUY' and current_price >= position['take_profit']:
            return True, "Take profit atingido"
        elif position['side'] == 'SELL' and current_price <= position['take_profit']:
            return True, "Take profit atingido"
        
        # Verificar tempo máximo (24 horas)
        if (timestamp - position['entry_time']).total_seconds() > 86400:
            return True, "Tempo máximo excedido"
        
        return False, ""
    
    def _close_position(self, position: Dict, candle: pd.Series, timestamp: datetime, reason: str) -> Trade:
        """Fecha uma posição"""
        
        exit_price = candle['close']
        entry_price = position['entry_price']
        size = position['size']
        
        # Calcular P&L
        if position['side'] == 'BUY':
            pnl = (exit_price - entry_price) * size
        else:  # SELL
            pnl = (entry_price - exit_price) * size
        
        # Aplicar comissões
        commission = (entry_price + exit_price) * size * self.commission_rate
        pnl -= commission
        
        # Calcular retorno percentual
        return_pct = (pnl / (entry_price * size)) * 100
        
        return Trade(
            symbol=position['symbol'],
            side=position['side'],
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            entry_time=position['entry_time'],
            exit_time=timestamp,
            pnl=pnl,
            return_pct=return_pct,
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            reason=reason
        )
    
    def _calculate_metrics(self, trades: List[Trade], initial_capital: float) -> BacktestResult:
        """Calcula métricas do backtest"""
        
        if not trades:
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, total_return=0, max_drawdown=0,
                sharpe_ratio=0, profit_factor=0, avg_win=0, avg_loss=0,
                max_win=0, max_loss=0, avg_trade_duration=0, trades=[]
            )
        
        # Métricas básicas
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L
        total_pnl = sum(t.pnl for t in trades)
        total_return = (total_pnl / initial_capital) * 100
        
        # Wins e losses
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        max_win = max(wins) if wins else 0
        max_loss = min(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Drawdown
        max_drawdown = self._calculate_max_drawdown(trades, initial_capital)
        
        # Sharpe ratio
        returns = [t.return_pct for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Duração média
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]  # em horas
        avg_trade_duration = np.mean(durations) if durations else 0
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            avg_trade_duration=avg_trade_duration,
            trades=trades
        )
    
    def _calculate_max_drawdown(self, trades: List[Trade], initial_capital: float) -> float:
        """Calcula drawdown máximo"""
        
        if not trades:
            return 0.0
        
        # Calcular equity curve
        equity = initial_capital
        equity_curve = [equity]
        peak = equity
        
        for trade in trades:
            equity += trade.pnl
            equity_curve.append(equity)
            peak = max(peak, equity)
        
        # Calcular drawdown
        drawdowns = []
        for equity in equity_curve:
            drawdown = (peak - equity) / peak * 100
            drawdowns.append(drawdown)
        
        return max(drawdowns) if drawdowns else 0.0
    
    def print_backtest_summary(self, result: BacktestResult):
        """Imprime resumo do backtest"""
        
        print("\n" + "="*60)
        print("📊 RESUMO DO BACKTEST")
        print("="*60)
        print(f"💰 Capital inicial: ${self.initial_capital:,.2f}")
        print(f"📈 Total de trades: {result.total_trades}")
        print(f"✅ Trades vencedores: {result.winning_trades}")
        print(f"❌ Trades perdedores: {result.losing_trades}")
        print(f"🎯 Taxa de acerto: {result.win_rate:.1f}%")
        print(f"💰 P&L total: ${result.total_pnl:,.2f}")
        print(f"📊 Retorno total: {result.total_return:.1f}%")
        print(f"📉 Drawdown máximo: {result.max_drawdown:.1f}%")
        print(f"⚡ Sharpe ratio: {result.sharpe_ratio:.2f}")
        print(f"🔥 Profit factor: {result.profit_factor:.2f}")
        print(f"📈 Ganho médio: ${result.avg_win:,.2f}")
        print(f"📉 Perda média: ${result.avg_loss:,.2f}")
        print(f"🏆 Maior ganho: ${result.max_win:,.2f}")
        print(f"💥 Maior perda: ${result.max_loss:,.2f}")
        print(f"⏱️  Duração média: {result.avg_trade_duration:.1f}h")
        print("="*60)
