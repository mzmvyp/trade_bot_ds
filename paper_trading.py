"""
Sistema de Paper Trading Completo
Simula execução de trades com rastreamento de performance
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

class PaperTradingSystem:
    def __init__(self, initial_balance: float = 10000.0):
        """
        Inicializa o sistema de paper trading.
        
        Args:
            initial_balance: Saldo inicial em USDT
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}  # {symbol: position_data}
        self.trade_history = []
        self.portfolio_value = initial_balance
        
        # Criar diretórios
        Path("paper_trades").mkdir(exist_ok=True)
        Path("portfolio").mkdir(exist_ok=True)
        
        # Carregar estado se existir
        self._load_state()
    
    def _load_state(self):
        """Carrega estado anterior do sistema"""
        try:
            if os.path.exists("portfolio/state.json"):
                with open("portfolio/state.json", "r") as f:
                    state = json.load(f)
                    self.current_balance = state.get("current_balance", self.initial_balance)
                    self.positions = state.get("positions", {})
                    self.trade_history = state.get("trade_history", [])
        except Exception as e:
            print(f"⚠️ Erro ao carregar estado: {e}")
    
    def _save_state(self):
        """Salva estado atual do sistema"""
        try:
            state = {
                "current_balance": self.current_balance,
                "positions": self.positions,
                "trade_history": self.trade_history,
                "last_update": datetime.now().isoformat()
            }
            with open("portfolio/state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Erro ao salvar estado: {e}")
    
    def execute_trade(self, signal: Dict[str, Any], position_size: float = None) -> Dict[str, Any]:
        """
        Executa um paper trade baseado no sinal.
        
        Args:
            signal: Dados do sinal (symbol, signal, entry_price, stop_loss, take_profit_1, take_profit_2)
            position_size: Tamanho da posição (se None, calcula automaticamente)
        
        Returns:
            Resultado da execução
        """
        try:
            symbol = signal.get("symbol")
            signal_type = signal.get("signal")
            entry_price = signal.get("entry_price")
            stop_loss = signal.get("stop_loss")
            take_profit_1 = signal.get("take_profit_1")
            take_profit_2 = signal.get("take_profit_2")
            confidence = signal.get("confidence", 5)
            
            # Validar dados obrigatórios
            if not all([symbol, signal_type, entry_price]):
                return {
                    "success": False,
                    "error": "Dados obrigatórios ausentes (symbol, signal, entry_price)"
                }
            
            if signal_type not in ["BUY", "SELL"]:
                return {
                    "success": False,
                    "error": f"Sinal {signal_type} não é executável (apenas BUY/SELL)"
                }
            
            # Calcular tamanho da posição
            if position_size is None:
                # Usar 2% do saldo por confiança
                risk_percentage = 0.02 * (confidence / 10)
                max_risk_amount = self.current_balance * risk_percentage
                
                if stop_loss:
                    risk_per_unit = abs(entry_price - stop_loss)
                    position_size = max_risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                else:
                    # Sem stop loss, usar 1% do saldo
                    position_size = (self.current_balance * 0.01) / entry_price
            
            # Calcular valor da posição
            position_value = position_size * entry_price
            
            # Verificar se tem saldo suficiente
            if position_value > self.current_balance:
                return {
                    "success": False,
                    "error": f"Saldo insuficiente. Necessário: ${position_value:.2f}, Disponível: ${self.current_balance:.2f}"
                }
            
            # Criar trade
            trade_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trade = {
                "trade_id": trade_id,
                "symbol": symbol,
                "signal": signal_type,
                "entry_price": entry_price,
                "position_size": position_size,
                "position_value": position_value,
                "stop_loss": stop_loss,
                "take_profit_1": take_profit_1,
                "take_profit_2": take_profit_2,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "status": "OPEN"
            }
            
            # Executar trade
            if signal_type == "BUY":
                self.current_balance -= position_value
                self.positions[symbol] = trade
            elif signal_type == "SELL":
                # Para SELL, assumimos que temos a posição (short)
                self.current_balance += position_value
                self.positions[f"{symbol}_SHORT"] = trade
            
            # Adicionar ao histórico
            self.trade_history.append(trade)
            
            # Salvar trade individual
            self._save_trade(trade)
            
            # Salvar estado
            self._save_state()
            
            return {
                "success": True,
                "trade_id": trade_id,
                "message": f"Paper trade executado: {signal_type} {position_size:.2f} unidades a ${entry_price:.2f}",
                "file": f"paper_trades/trade_{trade_id}.json"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Erro ao executar trade: {str(e)}"
            }
    
    def _save_trade(self, trade: Dict[str, Any]):
        """Salva trade individual"""
        try:
            filename = f"paper_trades/trade_{trade['trade_id']}.json"
            with open(filename, "w") as f:
                json.dump(trade, f, indent=2)
        except Exception as e:
            print(f"⚠️ Erro ao salvar trade: {e}")
    
    def close_position(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """
        Fecha uma posição aberta.
        
        Args:
            symbol: Símbolo da posição
            current_price: Preço atual para fechamento
        
        Returns:
            Resultado do fechamento
        """
        try:
            if symbol not in self.positions:
                return {
                    "success": False,
                    "error": f"Posição {symbol} não encontrada"
                }
            
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            position_size = position["position_size"]
            signal_type = position["signal"]
            
            # Calcular P&L
            if signal_type == "BUY":
                pnl = (current_price - entry_price) * position_size
                self.current_balance += current_price * position_size
            else:  # SELL
                pnl = (entry_price - current_price) * position_size
                self.current_balance -= current_price * position_size
            
            # Atualizar trade
            position["close_price"] = current_price
            position["pnl"] = pnl
            position["status"] = "CLOSED"
            position["close_timestamp"] = datetime.now().isoformat()
            
            # Remover da posição ativa
            del self.positions[symbol]
            
            # Salvar estado
            self._save_state()
            
            return {
                "success": True,
                "pnl": pnl,
                "message": f"Posição {symbol} fechada. P&L: ${pnl:.2f}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Erro ao fechar posição: {str(e)}"
            }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Retorna resumo do portfólio"""
        try:
            # Calcular valor total das posições abertas
            open_positions_value = 0
            for position in self.positions.values():
                open_positions_value += position["position_value"]
            
            # Calcular P&L total
            total_pnl = 0
            for trade in self.trade_history:
                if trade.get("status") == "CLOSED":
                    total_pnl += trade.get("pnl", 0)
            
            # Calcular performance
            total_return = (self.current_balance + open_positions_value - self.initial_balance) / self.initial_balance * 100
            
            return {
                "initial_balance": self.initial_balance,
                "current_balance": self.current_balance,
                "open_positions_value": open_positions_value,
                "total_portfolio_value": self.current_balance + open_positions_value,
                "total_pnl": total_pnl,
                "total_return_percent": total_return,
                "open_positions_count": len(self.positions),
                "total_trades": len(self.trade_history),
                "winning_trades": len([t for t in self.trade_history if t.get("pnl", 0) > 0]),
                "losing_trades": len([t for t in self.trade_history if t.get("pnl", 0) < 0])
            }
            
        except Exception as e:
            return {
                "error": f"Erro ao calcular resumo: {str(e)}"
            }
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Retorna posições abertas"""
        return list(self.positions.values())
    
    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna histórico de trades"""
        return self.trade_history[-limit:] if limit else self.trade_history
    
    def reset_portfolio(self):
        """Reseta o portfólio para o estado inicial"""
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trade_history = []
        self._save_state()
        print("🔄 Portfólio resetado para estado inicial")
    
    def export_performance_report(self) -> str:
        """Exporta relatório de performance"""
        try:
            summary = self.get_portfolio_summary()
            history = self.get_trade_history()
            
            report = {
                "report_date": datetime.now().isoformat(),
                "summary": summary,
                "trade_history": history,
                "open_positions": self.get_open_positions()
            }
            
            filename = f"portfolio/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)
            
            return filename
            
        except Exception as e:
            return f"Erro ao exportar relatório: {str(e)}"

# Instância global do sistema
paper_trading = PaperTradingSystem()
