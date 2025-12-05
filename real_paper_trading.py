"""
Sistema de Paper Trading REAL com Simulação Completa
Monitora preços, executa stop loss/take profit automaticamente
"""

import json
import os
import asyncio
import aiohttp
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from logger import get_logger

# Setup logger
logger = get_logger(__name__)

class RealPaperTradingSystem:
    def __init__(self, initial_balance: float = 10000.0):
        """
        Sistema de paper trading REAL que simula execução completa.
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}  # {symbol: position_data}
        self.trade_history = []
        self.portfolio_value = initial_balance
        self.is_monitoring = False
        self.monitor_task = None
        
        # CORREÇÃO: Adicionar lock para evitar race condition no monitoramento
        import threading
        self._monitoring_lock = threading.Lock()
        
        # Criar diretórios
        Path("paper_trades").mkdir(exist_ok=True)
        Path("portfolio").mkdir(exist_ok=True)
        Path("simulation_logs").mkdir(exist_ok=True)
        
        # Carregar estado se existir
        self._load_state()
    
    def _load_state(self):
        """Carrega estado anterior do sistema"""
        try:
            if os.path.exists("portfolio/state.json"):
                with open("portfolio/state.json", "r", encoding='utf-8') as f:
                    state = json.load(f)
                    self.current_balance = state.get("current_balance", self.initial_balance)
                    self.positions = state.get("positions", {})
                    self.trade_history = state.get("trade_history", [])
                    logger.info(f"Estado carregado: {len(self.positions)} posições abertas, saldo ${self.current_balance:.2f}")
                    
                    # CRÍTICO: Iniciar monitoramento se houver posições abertas
                    if len(self.positions) > 0 and not self.is_monitoring:
                        logger.warning(f"[CRITICO] Posicoes abertas encontradas mas monitoramento nao esta ativo. Iniciando monitoramento...")
                        self.start_monitoring()
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Erro ao decodificar state.json: {e}")
        except IOError as e:
            logger.error(f"Erro ao ler arquivo de estado: {e}")
        except Exception as e:
            logger.exception(f"Erro inesperado ao carregar estado: {e}")
    
    def _save_state(self):
        """Salva estado atual do sistema usando escrita atômica"""
        try:
            state = {
                "current_balance": self.current_balance,
                "positions": self.positions,
                "trade_history": self.trade_history,
                "last_update": datetime.now().isoformat()
            }

            # Atomic write to prevent corruption
            state_file = Path("portfolio/state.json")
            fd, temp_path = tempfile.mkstemp(
                dir=state_file.parent,
                prefix='.state_',
                suffix='.json.tmp'
            )

            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(state, f, indent=2)

                # Atomic rename
                os.replace(temp_path, state_file)
                logger.debug(f"Estado salvo atomicamente: {len(self.positions)} posições, saldo ${self.current_balance:.2f}")
            except Exception:
                # Cleanup temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        except IOError as e:
            logger.error(f"Erro de I/O ao salvar estado: {e}")
        except Exception as e:
            logger.exception(f"Erro inesperado ao salvar estado: {e}")
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual do símbolo"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://fapi.binance.com/fapi/v1/ticker/price"
                params = {'symbol': symbol}

                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        logger.warning(f"API retornou status {response.status} para {symbol}")
                        return None

                    data = await response.json()
                    return float(data['price'])

        except aiohttp.ClientError as e:
            logger.error(f"Erro de conexão ao obter preço de {symbol}: {e}")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout ao obter preço de {symbol}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Erro ao processar resposta para {symbol}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Erro inesperado ao obter preço de {symbol}: {e}")
            return None
    
    def execute_trade(self, signal: Dict[str, Any], position_size: float = None) -> Dict[str, Any]:
        """
        Executa um paper trade REAL com monitoramento automático.
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
                risk_percentage = 0.02 * (confidence / 10)
                max_risk_amount = self.current_balance * risk_percentage
                
                if stop_loss:
                    risk_per_unit = abs(entry_price - stop_loss)
                    position_size = max_risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                else:
                    position_size = (self.current_balance * 0.01) / entry_price
            
            # Calcular valor da posição
            position_value = position_size * entry_price
            
            # CRÍTICO: Verificar se já existe posição aberta para este símbolo (BUY ou SELL)
            # Não permitir duas posições no mesmo ativo simultaneamente
            existing_position = None
            existing_key = None
            
            # Verificar se existe posição BUY (chave = symbol)
            if symbol in self.positions:
                existing_position = self.positions[symbol]
                existing_key = symbol
            # Verificar se existe posição SELL (chave = symbol_SHORT)
            elif f"{symbol}_SHORT" in self.positions:
                existing_position = self.positions[f"{symbol}_SHORT"]
                existing_key = f"{symbol}_SHORT"
            
            if existing_position and existing_position.get("status") == "OPEN":
                existing_signal = existing_position.get("signal", "UNKNOWN")
                return {
                    "success": False,
                    "error": f"Ja existe uma posicao {existing_signal} aberta para {symbol}. Feche a posicao existente antes de abrir uma nova."
                }
            
            # Verificar se tem saldo suficiente
            if position_value > self.current_balance:
                return {
                    "success": False,
                    "error": f"Saldo insuficiente. Necessario: ${position_value:.2f}, Disponivel: ${self.current_balance:.2f}"
                }
            
            # Criar trade
            trade_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trade = {
                "trade_id": trade_id,
                "symbol": symbol,
                "signal": signal_type,
                "entry_price": entry_price,
                "position_size": position_size,
                "original_position_size": position_size,  # Guardar tamanho original para cálculos de fechamento parcial
                "position_value": position_value,
                "stop_loss": stop_loss,
                "take_profit_1": take_profit_1,
                "take_profit_2": take_profit_2,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "status": "OPEN",
                "max_profit_reached": 0.0,
                "max_loss_reached": 0.0,
                "tp1_partial_closed": False  # Flag para indicar se TP1 já foi parcialmente fechado
            }
            
            # Executar trade
            if signal_type == "BUY":
                # Para BUY: deduzir valor da posição do saldo
                self.current_balance -= position_value
                self.positions[symbol] = trade
            elif signal_type == "SELL":
                # CORRIGIDO: Para SELL (SHORT), precisamos RESERVAR margem (não receber dinheiro)
                # Em uma posição SHORT, você "empresta" o ativo e vende, mas precisa ter margem para cobrir
                # Deduzir margem do saldo (similar a BUY)
                self.current_balance -= position_value
                self.positions[f"{symbol}_SHORT"] = trade
            
            # Adicionar ao histórico
            self.trade_history.append(trade)
            
            # Salvar trade individual
            self._save_trade(trade)
            
            # Salvar estado
            self._save_state()
            
            # Iniciar monitoramento se não estiver ativo
            if not self.is_monitoring:
                self.start_monitoring()
            
            return {
                "success": True,
                "trade_id": trade_id,
                "message": f"Paper trade REAL executado: {signal_type} {position_size:.2f} unidades a ${entry_price:.2f}",
                "file": f"paper_trades/trade_{trade_id}.json",
                "monitoring": "Iniciado monitoramento automático"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Erro ao executar trade: {str(e)}"
            }
    
    def start_monitoring(self):
        """Inicia monitoramento automático de posições (CORRIGIDO: race condition fix com lock)"""
        # CORREÇÃO: Usar lock para evitar race condition
        with self._monitoring_lock:
            if not self.is_monitoring:
                self.is_monitoring = True

                # FIX: Check for running event loop before creating task
                try:
                    loop = asyncio.get_running_loop()
                    self.monitor_task = loop.create_task(self._monitor_positions())
                    logger.info("[MONITOR] Monitoramento automatico iniciado (async context)")
                except RuntimeError:
                    # No event loop running - create one
                    logger.warning("Sem event loop ativo, criando nova thread de monitoramento")
                    import threading

                    def run_monitoring():
                        asyncio.run(self._monitor_positions())

                    self.monitor_thread = threading.Thread(target=run_monitoring, daemon=True)
                    self.monitor_thread.start()
                    logger.info("[MONITOR] Monitoramento automatico iniciado (thread separada)")
    
    def stop_monitoring(self):
        """Para monitoramento automático"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_task:
                self.monitor_task.cancel()
            logger.info("[PARADO] Monitoramento automatico parado")
    
    async def _monitor_positions(self):
        """Monitora posições abertas e executa stop loss/take profit"""
        logger.warning(f"[MONITOR] Iniciando monitoramento de {len(self.positions)} posicoes")
        while self.is_monitoring and self.positions:
            try:
                for symbol, position in list(self.positions.items()):
                    current_price = await self.get_current_price(symbol.replace("_SHORT", ""))
                    
                    if current_price is None:
                        logger.warning(f"[MONITOR] Nao foi possivel obter preco para {symbol}")
                        continue
                    
                    # Log detalhado a cada verificação (INFO para visibilidade)
                    clean_symbol = symbol.replace("_SHORT", "")
                    entry_price = position.get("entry_price", 0)
                    tp1 = position.get("take_profit_1", 0)
                    signal_type = position.get("signal", "UNKNOWN")
                    logger.info(f"[MONITOR] {clean_symbol} ({signal_type}): Preco ${current_price:.2f} | Entry ${entry_price:.2f} | TP1 ${tp1:.2f}")
                    
                    # Calcular P&L atual
                    entry_price = position["entry_price"]
                    position_size = position["position_size"]
                    signal_type = position["signal"]
                    
                    if signal_type == "BUY":
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                        pnl_amount = (current_price - entry_price) * position_size
                    else:  # SELL
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                        pnl_amount = (entry_price - current_price) * position_size
                    
                    # Atualizar máximo lucro/prejuízo e SALVAR estado
                    if pnl_amount > position.get("max_profit_reached", 0):
                        position["max_profit_reached"] = pnl_amount
                        # Salvar estado quando atinge novo máximo
                        self._save_state()
                    if pnl_amount < position.get("max_loss_reached", 0):
                        position["max_loss_reached"] = pnl_amount
                        # Salvar estado quando atinge novo mínimo
                        self._save_state()
                    
                    # Verificar stop loss
                    if position.get("stop_loss"):
                        sl = position["stop_loss"]
                        if signal_type == "BUY" and current_price <= sl:
                            logger.warning(f"[STOP LOSS] Stop Loss atingido para {symbol}: ${current_price:.2f} (SL: ${sl:.2f})")
                            await self._close_position_auto(symbol, current_price, "STOP_LOSS")
                            continue
                        elif signal_type == "SELL" and current_price >= sl:
                            logger.warning(f"[STOP LOSS] Stop Loss atingido para {symbol}: ${current_price:.2f} (SL: ${sl:.2f})")
                            await self._close_position_auto(symbol, current_price, "STOP_LOSS")
                            continue

                    # Verificar take profit 1 (CORRIGIDO: fechar apenas 50% da posição)
                    if position.get("take_profit_1") and not position.get("tp1_partial_closed", False):
                        tp1 = position["take_profit_1"]
                        if signal_type == "BUY" and current_price >= tp1:
                            # Verificar se o preço REAL atingiu o take profit
                            logger.warning(f"[TAKE PROFIT 1] Take Profit 1 atingido para {symbol}: ${current_price:.2f} (TP: ${tp1:.2f})")
                            await self._close_position_partial(symbol, current_price, "TAKE_PROFIT_1", partial_percent=0.5)
                            continue
                        elif signal_type == "SELL" and current_price <= tp1:
                            logger.warning(f"[TAKE PROFIT 1] Take Profit 1 atingido para {symbol}: ${current_price:.2f} (TP: ${tp1:.2f})")
                            await self._close_position_partial(symbol, current_price, "TAKE_PROFIT_1", partial_percent=0.5)
                            continue

                    # Verificar take profit 2 (fechar 50% restante quando TP2 for atingido)
                    if position.get("take_profit_2") and position.get("tp1_partial_closed", False):
                        tp2 = position["take_profit_2"]
                        if signal_type == "BUY" and current_price >= tp2:
                            logger.warning(f"[TAKE PROFIT 2] Take Profit 2 atingido para {symbol}: ${current_price:.2f} (TP: ${tp2:.2f}) - Fechando 50% restante")
                            await self._close_position_auto(symbol, current_price, "TAKE_PROFIT_2")
                            continue
                        elif signal_type == "SELL" and current_price <= tp2:
                            logger.warning(f"[TAKE PROFIT 2] Take Profit 2 atingido para {symbol}: ${current_price:.2f} (TP: ${tp2:.2f}) - Fechando 50% restante")
                            await self._close_position_auto(symbol, current_price, "TAKE_PROFIT_2")
                            continue
                    
                    # Log de monitoramento
                    self._log_monitoring(symbol, current_price, pnl_percent, pnl_amount)
                
                # Pausa entre verificações
                await asyncio.sleep(5)  # Verifica a cada 5 segundos

            except asyncio.CancelledError:
                logger.info("Monitoramento cancelado pelo usuário")
                break
            except Exception as e:
                logger.exception(f"❌ Erro no monitoramento: {e}")
                await asyncio.sleep(10)
    
    async def _close_position_partial(self, symbol: str, current_price: float, reason: str, partial_percent: float = 0.5):
        """
        Fecha parcialmente uma posição (ex: 50% no TP1, 50% restante no TP2)
        
        Args:
            symbol: Símbolo da posição
            current_price: Preço atual para fechamento
            reason: Motivo do fechamento (TAKE_PROFIT_1, TAKE_PROFIT_2, STOP_LOSS)
            partial_percent: Porcentagem da posição a fechar (0.5 = 50%)
        """
        try:
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            original_position_size = position.get("original_position_size", position["position_size"])
            current_position_size = position["position_size"]
            signal_type = position["signal"]
            
            # Calcular quantidade a fechar
            size_to_close = current_position_size * partial_percent
            size_remaining = current_position_size - size_to_close
            
            # Calcular P&L da parte fechada
            if signal_type == "BUY":
                pnl_partial = (current_price - entry_price) * size_to_close
                # Recuperamos o valor da posição vendida
                self.current_balance += current_price * size_to_close
            else:  # SELL
                # CORRIGIDO: Para SELL (SHORT) parcial
                pnl_partial = (entry_price - current_price) * size_to_close
                # Devolver margem proporcional + lucro/prejuízo
                self.current_balance += entry_price * size_to_close  # Devolver margem reservada proporcional
                self.current_balance -= current_price * size_to_close  # Pagar compra de volta proporcional
                # Resultado líquido: +pnl_partial
            
            # Registrar fechamento parcial no histórico
            partial_close_entry = {
                "trade_id": f"{position.get('trade_id')}_partial_{reason}",
                "symbol": symbol.replace("_SHORT", ""),
                "signal": signal_type,
                "entry_price": entry_price,
                "close_price": current_price,
                "position_size": size_to_close,
                "pnl": pnl_partial,
                "status": "CLOSED_PARTIAL",
                "close_timestamp": datetime.now().isoformat(),
                "close_reason": reason,
                "partial_percent": partial_percent * 100
            }
            self.trade_history.append(partial_close_entry)
            
            # Atualizar posição: reduzir tamanho e marcar que TP1 foi parcialmente fechado
            position["position_size"] = size_remaining
            position["position_value"] = size_remaining * entry_price  # Valor baseado no entry original
            position["tp1_partial_closed"] = True
            position["partial_close_price"] = current_price
            position["partial_close_pnl"] = pnl_partial
            
            # Salvar estado
            self._save_state()
            
            # Log de fechamento parcial
            logger.warning(f"[FECHADO PARCIAL] {reason}: {symbol} - {partial_percent*100:.0f}% fechado a ${current_price:.2f} | P&L: ${pnl_partial:.2f} | Restante: {size_remaining:.6f}")
            self._log_trade_close(f"{symbol}_partial", current_price, pnl_partial, f"{reason}_PARTIAL")

        except KeyError as e:
            logger.error(f"Posição não encontrada: {symbol} - {e}")
        except Exception as e:
            logger.exception(f"❌ Erro ao fechar parcialmente posição {symbol}: {e}")
    
    async def _close_position_auto(self, symbol: str, current_price: float, reason: str):
        """Fecha posição automaticamente (100% - usado para TP2 final ou Stop Loss)"""
        try:
            position = self.positions[symbol]
            entry_price = position["entry_price"]
            position_size = position["position_size"]
            signal_type = position["signal"]
            
            # Calcular P&L final
            if signal_type == "BUY":
                # Para BUY: vendemos a posição ao preço atual
                pnl = (current_price - entry_price) * position_size
                # Recuperamos o valor da posição vendida (que foi pago na abertura)
                self.current_balance += current_price * position_size
            else:  # SELL
                # CORRIGIDO: Para SELL (SHORT)
                # Na abertura: reservamos margem (deduzimos position_value do saldo)
                # No fechamento: compramos de volta ao preço atual
                # Lucro = (preço de venda - preço de compra) * quantidade
                pnl = (entry_price - current_price) * position_size
                # Devolver margem reservada + lucro/prejuízo
                # Margem reservada = entry_price * position_size (foi deduzida na abertura)
                # Preço de compra de volta = current_price * position_size
                # Saldo final = saldo_atual + margem_reservada - preco_compra_volta + pnl
                # Simplificando: saldo_atual + entry_price * position_size - current_price * position_size
                # Como pnl = (entry_price - current_price) * position_size:
                # saldo_final = saldo_atual + pnl
                self.current_balance += entry_price * position_size  # Devolver margem reservada
                self.current_balance -= current_price * position_size  # Pagar compra de volta
                # Resultado líquido: saldo_atual + (entry_price - current_price) * position_size = saldo_atual + pnl
            
            # Calcular P&L total (incluindo fechamento parcial anterior se houver)
            total_pnl = pnl
            if position.get("partial_close_pnl"):
                total_pnl += position["partial_close_pnl"]
            
            # Atualizar trade no histórico ANTES de remover
            # Encontrar trade no histórico e atualizar
            for trade in self.trade_history:
                if trade.get("trade_id") == position.get("trade_id") and trade.get("status") != "CLOSED":
                    trade["close_price"] = current_price
                    trade["pnl"] = total_pnl  # P&L total incluindo parcial
                    trade["status"] = "CLOSED"
                    trade["close_timestamp"] = datetime.now().isoformat()
                    trade["close_reason"] = reason
                    break
            
            # Atualizar posição
            position["close_price"] = current_price
            position["pnl"] = total_pnl
            position["status"] = "CLOSED"
            position["close_timestamp"] = datetime.now().isoformat()
            position["close_reason"] = reason
            
            # Remover da posição ativa
            del self.positions[symbol]
            
            # Salvar estado IMEDIATAMENTE (CRÍTICO)
            self._save_state()
            
            # Log de fechamento
            logger.warning(f"[FECHADO] {reason}: {symbol} fechado a ${current_price:.2f} | P&L Total: ${total_pnl:.2f}")
            self._log_trade_close(symbol, current_price, total_pnl, reason)

        except KeyError as e:
            logger.error(f"Posição não encontrada: {symbol} - {e}")
        except Exception as e:
            logger.exception(f"❌ Erro ao fechar posição {symbol}: {e}")
    
    def _log_monitoring(self, symbol: str, price: float, pnl_percent: float, pnl_amount: float):
        """Log de monitoramento"""
        logger.debug(f"📊 {symbol}: ${price:.2f} | P&L: {pnl_percent:+.2f}% (${pnl_amount:+.2f})")
    
    def _log_trade_close(self, symbol: str, price: float, pnl: float, reason: str):
        """Log de fechamento de trade"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "close_price": price,
            "pnl": pnl,
            "reason": reason
        }
        
        # Salvar log
        log_file = f"simulation_logs/trade_close_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"⚠️ Erro ao salvar log: {e}")
        except Exception as e:
            logger.exception(f"Erro inesperado ao salvar log: {e}")
    
    def _save_trade(self, trade: Dict[str, Any]):
        """Salva trade individual"""
        try:
            filename = f"paper_trades/trade_{trade['trade_id']}.json"
            with open(filename, "w") as f:
                json.dump(trade, f, indent=2)
            logger.debug(f"Trade {trade['trade_id']} salvo em {filename}")
        except (IOError, KeyError) as e:
            logger.error(f"⚠️ Erro ao salvar trade: {e}")
        except Exception as e:
            logger.exception(f"Erro inesperado ao salvar trade: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Retorna resumo do portfólio com dados REAIS"""
        try:
            # Calcular valor total das posições abertas
            open_positions_value = 0
            for position in self.positions.values():
                open_positions_value += position["position_value"]
            
            # Calcular P&L total
            total_pnl = 0
            winning_trades = 0
            losing_trades = 0
            
            for trade in self.trade_history:
                if trade.get("status") == "CLOSED":
                    pnl = trade.get("pnl", 0)
                    total_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
                    elif pnl < 0:
                        losing_trades += 1
            
            # Calcular performance
            total_return = (self.current_balance + open_positions_value - self.initial_balance) / self.initial_balance * 100
            
            # Calcular win rate
            total_closed_trades = winning_trades + losing_trades
            win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
            
            return {
                "initial_balance": self.initial_balance,
                "current_balance": self.current_balance,
                "open_positions_value": open_positions_value,
                "total_portfolio_value": self.current_balance + open_positions_value,
                "total_pnl": total_pnl,
                "total_return_percent": total_return,
                "open_positions_count": len(self.positions),
                "total_trades": len(self.trade_history),
                "closed_trades": total_closed_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate_percent": win_rate,
                "is_monitoring": self.is_monitoring
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
        self.stop_monitoring()
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trade_history = []
        self._save_state()
        logger.info("🔄 Portfólio resetado para estado inicial")
    
    def export_performance_report(self) -> str:
        """Exporta relatório de performance REAL"""
        try:
            summary = self.get_portfolio_summary()
            history = self.get_trade_history()
            
            report = {
                "report_date": datetime.now().isoformat(),
                "summary": summary,
                "trade_history": history,
                "open_positions": self.get_open_positions(),
                "simulation_type": "REAL_PAPER_TRADING"
            }
            
            filename = f"portfolio/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)
            
            return filename
            
        except Exception as e:
            return f"Erro ao exportar relatório: {str(e)}"

# Instância global do sistema REAL
real_paper_trading = RealPaperTradingSystem()
