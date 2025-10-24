"""
Agent de trading aprimorado com todas as melhorias
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Removido Agno - usando integração direta

from binance_client import BinanceClient
from technical_analysis import EnhancedTechnicalAnalyzer
from sentiment_analysis import SentimentAnalyzer
from deepseek_tool import EnhancedDeepSeekTool
from risk_management import RiskManagementSystem, Portfolio
from logger import log_signal, log_error, log_market_data, log_api_call
from config import settings

class EnhancedTradingAgent:
    """Agent de trading aprimorado com todas as melhorias"""
    
    def __init__(self):
        # Clientes e analisadores
        self.binance_client = BinanceClient()
        self.technical_analyzer = EnhancedTechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.deepseek_tool = EnhancedDeepSeekTool()
        self.risk_manager = RiskManagementSystem()
        
        # Portfolio inicial
        self.portfolio = Portfolio(
            balance=10000.0,  # Capital inicial
            positions=[],
            daily_pnl=0.0,
            total_pnl=0.0,
            last_reset=datetime.now()
        )
        
        self.risk_manager.set_portfolio(self.portfolio)
        
        # Sistema simplificado sem Agno
    
    async def run_single_analysis(self, symbol: str = None) -> Dict[str, Any]:
        """
        Executa análise única para um símbolo
        
        Args:
            symbol: Símbolo para análise (padrão: BTCUSDT)
            
        Returns:
            Dicionário com sinal de trading
        """
        if not symbol:
            symbol = settings.trading_symbol
        
        try:
            print(f"🔍 Analisando mercado para {symbol}...")
            
            # 1. Coletar dados de mercado
            market_data = await self._collect_market_data(symbol)
            log_market_data(symbol, market_data)
            
            # 2. Análise técnica avançada
            technical_signals = await self._analyze_technical(symbol, market_data)
            
            # 3. Análise de sentimento
            sentiment_data = await self._analyze_sentiment(symbol)
            
            # 4. Análise com DeepSeek
            deepseek_analysis = await self._analyze_with_deepseek(
                market_data, technical_signals, sentiment_data
            )
            
            # 5. Validação de risco
            risk_validation = self._validate_risk(deepseek_analysis, symbol)
            
            # 6. Gerar sinal final
            final_signal = self._generate_final_signal(
                deepseek_analysis, risk_validation, market_data
            )
            
            # 7. Log do sinal
            log_signal(final_signal)
            
            return final_signal
            
        except Exception as e:
            log_error(e, {"symbol": symbol, "action": "single_analysis"})
            
            # Fallback para análise básica
            return self._generate_fallback_signal(symbol, market_data)
    
    async def _collect_market_data(self, symbol: str) -> Dict[str, Any]:
        """Coleta dados de mercado"""
        try:
            async with self.binance_client:
                # Dados básicos
                ticker = await self.binance_client.get_ticker_24hr(symbol)
                
                # Dados de candlestick
                klines = await self.binance_client.get_klines(symbol, "1h", limit=100)
                
                # Dados de funding
                funding_rate = await self.binance_client.get_funding_rate(symbol)
                
                # Dados de open interest
                open_interest = await self.binance_client.get_open_interest(symbol)
                
                return {
                    'symbol': symbol,
                    'current_price': float(ticker['lastPrice']),
                    'price_change_24h': float(ticker['priceChangePercent']),
                    'volume_24h': float(ticker['volume']),
                    'funding_rate': float(funding_rate.get('fundingRate', 0)),
                    'open_interest': float(open_interest.get('openInterest', 0)),
                    'klines': klines,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            log_error(e, {"symbol": symbol, "action": "collect_market_data"})
            return {
                'symbol': symbol,
                'current_price': 0,
                'price_change_24h': 0,
                'volume_24h': 0,
                'funding_rate': 0,
                'open_interest': 0,
                'klines': [],
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_technical(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Análise técnica avançada"""
        try:
            klines = market_data.get('klines', [])
            if not klines:
                return self._generate_basic_technical_signals()
            
            # Converter para DataFrame
            import pandas as pd
            df = pd.DataFrame(klines)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                         'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            
            # Converter tipos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Verificar se temos dados suficientes
            if len(df) < 50:
                return self._generate_basic_technical_signals()
            
            # Calcular indicadores avançados
            try:
                df = self.technical_analyzer.calculate_advanced_indicators(df)
            except Exception as e:
                log_error(e, {"symbol": symbol, "action": "technical_indicators"})
                return self._generate_basic_technical_signals()
            
            # Gerar sinais
            signals = self.technical_analyzer.generate_technical_signals(df)
            
            return signals
            
        except Exception as e:
            log_error(e, {"symbol": symbol, "action": "technical_analysis"})
            return self._generate_basic_technical_signals()
    
    async def _analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Análise de sentimento"""
        try:
            # Usar analisador de sentimento existente
            sentiment_data = await self.sentiment_analyzer.analyze_sentiment(symbol)
            return sentiment_data
            
        except Exception as e:
            log_error(e, {"symbol": symbol, "action": "sentiment_analysis"})
            return {
                'overall_sentiment': 'neutral',
                'score': 0.5,
                'sources': {},
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_with_deepseek(self, market_data: Dict, technical_signals: Dict, sentiment_data: Dict) -> Dict[str, Any]:
        """Análise com DeepSeek aprimorada"""
        try:
            print("🧠 Usando DeepSeek para análise...")
            
            # Preparar dados para DeepSeek
            analysis_data = {
                'market_data': market_data,
                'technical_signals': technical_signals,
                'sentiment_data': sentiment_data
            }
            
            # Chamar DeepSeek aprimorado
            result = await self.deepseek_tool.analyze_with_structured_output(analysis_data)
            
            print("✅ DeepSeek retornou análise")
            return result
            
        except Exception as e:
            log_error(e, {"action": "deepseek_analysis"})
            print("⚠️  DeepSeek não retornou resposta")
            return None
    
    def _validate_risk(self, signal: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Validação de risco"""
        if not signal:
            return {'can_execute': False, 'reason': 'Nenhum sinal disponível'}
        
        try:
            # Validar com sistema de risco
            risk_validation = self.risk_manager.validate_trade_risk(signal, symbol)
            return risk_validation
            
        except Exception as e:
            log_error(e, {"action": "risk_validation"})
            return {'can_execute': False, 'reason': 'Erro na validação de risco'}
    
    def _generate_final_signal(self, deepseek_analysis: Dict, risk_validation: Dict, market_data: Dict) -> Dict[str, Any]:
        """Gera sinal final"""
        
        if not deepseek_analysis:
            return self._generate_fallback_signal(market_data.get('symbol', 'BTCUSDT'), market_data)
        
        # Aplicar validações de risco
        if not risk_validation.get('can_execute', False):
            # Ajustar sinal baseado em validação de risco
            if risk_validation.get('confidence_ok') == False:
                deepseek_analysis['confidence'] = max(1, deepseek_analysis.get('confidence', 5) - 2)
            
            if risk_validation.get('warnings'):
                deepseek_analysis['warnings'] = deepseek_analysis.get('warnings', []) + risk_validation['warnings']
        
        # Calcular position size
        if deepseek_analysis.get('signal') in ['BUY', 'SELL']:
            position_size = self.risk_manager.calculate_position_size(
                entry_price=deepseek_analysis.get('entry_price', market_data.get('current_price', 0)),
                stop_loss=deepseek_analysis.get('stop_loss', 0),
                confidence=deepseek_analysis.get('confidence', 5)
            )
            deepseek_analysis['position_size'] = position_size
        
        # Adicionar metadados
        deepseek_analysis['timestamp'] = datetime.now().isoformat()
        deepseek_analysis['risk_validation'] = risk_validation
        deepseek_analysis['market_data'] = market_data
        
        return deepseek_analysis
    
    def _generate_fallback_signal(self, symbol: str, market_data: Dict = None) -> Dict[str, Any]:
        """Gera sinal de fallback quando DeepSeek falha"""
        
        # Verificar se market_data existe
        if not market_data:
            market_data = {
                'current_price': 0,
                'price_change_24h': 0,
                'volume_24h': 0
            }
        
        current_price = market_data.get('current_price', 0)
        price_change = market_data.get('price_change_24h', 0)
        
        # Análise básica baseada na variação de preço
        if price_change > 2:
            signal_type = 'BUY'
            confidence = 6
        elif price_change < -2:
            signal_type = 'SELL'
            confidence = 6
        else:
            signal_type = 'HOLD'
            confidence = 5
        
        return {
            'signal': signal_type,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit_1': None,
            'take_profit_2': None,
            'risk_reward_ratio': 0,
            'reasoning': {
                'technical': f"Análise básica baseada na variação de preço de 24h ({price_change:.2f}%)",
                'sentiment': 'Dados de sentimento não disponíveis',
                'market_structure': 'Análise limitada - usando fallback'
            },
            'warnings': ['Análise limitada - DeepSeek indisponível'],
            'timeframe': 'curto prazo',
            'timestamp': datetime.now().isoformat(),
            'fallback': True
        }
    
    def _generate_basic_technical_signals(self) -> Dict[str, Any]:
        """Gera sinais técnicos básicos quando não há dados suficientes"""
        return {
            'trend': 'neutral',
            'momentum': 'neutral',
            'volatility': 'normal',
            'volume': 'normal',
            'structure': 'neutral',
            'combined_signal': {
                'signal': 'HOLD',
                'confidence': 3,
                'total_score': 0
            }
        }
    
    async def run_backtest(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Executa backtest para um símbolo"""
        try:
            from backtesting_engine import BacktestingEngine
            
            engine = BacktestingEngine(initial_capital=self.portfolio.balance)
            result = await engine.backtest_strategy(symbol, start_date, end_date)
            
            # Imprimir resumo
            engine.print_backtest_summary(result)
            
            return {
                'success': True,
                'result': result,
                'summary': {
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'total_return': result.total_return,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio
                }
            }
            
        except Exception as e:
            log_error(e, {"symbol": symbol, "action": "backtest"})
            return {
                'success': False,
                'error': str(e)
            }
