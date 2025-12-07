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
        self._save_lock = threading.Lock()  # Lock para salvamento de estado
        self._save_lock = threading.Lock()  # Lock para salvamento de estado
        
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
                    # MODIFICADO: Não carregar current_balance (sistema em modo P&L)
                    self.positions = state.get("positions", {})
                    self.trade_history = state.get("trade_history", [])
                    logger.info(f"Estado carregado: {len(self.positions)} posições abertas (modo P&L)")
                    
                    # MIGRAÇÃO: Adicionar campo 'source' para posições antigas que não têm
                    migration_needed = False
                    for pos_key, pos in self.positions.items():
                        if "source" not in pos:
                            # Tentar inferir da chave
                            if "_DEEPSEEK" in pos_key:
                                pos["source"] = "DEEPSEEK"
                            elif "_AGNO" in pos_key:
                                pos["source"] = "AGNO"
                            else:
                                pos["source"] = "LEGACY"  # Posições antigas sem identificação
                            migration_needed = True
                            logger.info(f"[MIGRACAO] Adicionado campo 'source' para posição {pos_key}: {pos['source']}")

                    # MIGRAÇÃO: Converter pnl para pnl_percent em trades antigos
                    for trade in self.trade_history:
                        if trade.get("pnl") is not None and trade.get("pnl_percent") is None:
                            entry = trade.get("entry_price", 1)
                            size = trade.get("position_size", 1)
                            if entry > 0 and size > 0:
                                trade["pnl_percent"] = (trade["pnl"] / (entry * size)) * 100
                                migration_needed = True
                                logger.info(f"[MIGRACAO] Trade {trade.get('trade_id')}: pnl=${trade['pnl']:.2f} -> pnl_percent={trade['pnl_percent']:.2f}%")

                    # Salvar estado após migração se necessário
                    if migration_needed:
                        self._save_state()
                        logger.info("[MIGRACAO] Estado salvo após migração de campos 'source' e pnl_percent")
                    
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
        """Salva estado atual do sistema usando escrita atômica com retry e lock"""
        # Usar lock para evitar salvamentos simultâneos
        with self._save_lock:
            max_retries = 3
            retry_delay = 0.1  # 100ms
            
            for attempt in range(max_retries):
                try:
                    # MODIFICADO: Remover current_balance (sistema em modo P&L)
                    state = {
                        "positions": self.positions,
                        "trade_history": self.trade_history,
                        "last_update": datetime.now().isoformat()
                    }

                    # Atomic write to prevent corruption
                    state_file = Path("portfolio/state.json")
                    
                    # Criar arquivo temporário no mesmo diretório com nome único
                    import time
                    temp_path = state_file.parent / f".state_{os.getpid()}_{int(time.time() * 1000000)}.tmp"
                    
                    try:
                        # Escrever para arquivo temporário
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            json.dump(state, f, indent=2)
                            f.flush()
                            os.fsync(f.fileno())  # Forçar escrita no disco

                        # Tentar substituir o arquivo original
                        # No Windows, pode falhar se o arquivo estiver aberto (ex: Streamlit)
                        try:
                            os.replace(temp_path, state_file)
                        except (PermissionError, OSError) as e:
                            # Se falhar por permissão/arquivo em uso, tentar remover o arquivo antigo primeiro
                            if state_file.exists():
                                try:
                                    # Tentar remover arquivo antigo
                                    os.remove(state_file)
                                    os.replace(temp_path, state_file)
                                except Exception as e2:
                                    logger.warning(f"Tentativa {attempt + 1}: Erro ao substituir state.json: {e2}")
                                    if temp_path.exists():
                                        try:
                                            temp_path.unlink()
                                        except:
                                            pass
                                    if attempt < max_retries - 1:
                                        time.sleep(retry_delay * (attempt + 1))
                                        continue
                                    raise
                            else:
                                # Arquivo não existe, apenas renomear
                                os.replace(temp_path, state_file)
                        
                        logger.debug(f"Estado salvo atomicamente: {len(self.positions)} posições (modo P&L)")
                        return  # Sucesso, sair da função
                        
                    except Exception as e:
                        # Cleanup temp file on error
                        if temp_path.exists():
                            try:
                                temp_path.unlink()
                            except:
                                pass
                        
                        if attempt < max_retries - 1:
                            logger.warning(f"Tentativa {attempt + 1}/{max_retries} falhou ao salvar estado: {e}. Tentando novamente...")
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            raise
                            
                except IOError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Erro de I/O ao salvar estado (tentativa {attempt + 1}): {e}. Tentando novamente...")
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Erro de I/O ao salvar estado após {max_retries} tentativas: {e}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Erro inesperado ao salvar estado (tentativa {attempt + 1}): {e}. Tentando novamente...")
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.exception(f"Erro inesperado ao salvar estado após {max_retries} tentativas: {e}")
    
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
            
            # MODIFICADO: Tamanho de posição fixo ou usar o fornecido
            # Sistema agora foca apenas em P&L, não em capital
            if position_size is None:
                # Tamanho padrão baseado em risco de $100
                if stop_loss:
                    risk_per_unit = abs(entry_price - stop_loss)
                    if risk_per_unit > 0:
                        position_size = 100.0 / risk_per_unit  # $100 de risco por trade
                    else:
                        position_size = 1.0
                else:
                    position_size = 1.0  # 1 unidade padrão
            
            # Calcular valor da posição (apenas para tracking, não deduz do saldo)
            position_value = position_size * entry_price
            
            # MODIFICADO: Verificar se já existe posição aberta para este símbolo E FONTE
            # Permite duas posições do mesmo símbolo se forem de fontes diferentes (DEEPSEEK vs AGNO)
            signal_source = signal.get("source", "UNKNOWN")  # DEEPSEEK ou AGNO
            
            # Determinar chave da posição baseada em símbolo, fonte e tipo de sinal
            if signal_type == "BUY":
                position_key = f"{symbol}_{signal_source}"
            elif signal_type == "SELL":
                position_key = f"{symbol}_{signal_source}_SHORT"
            else:
                position_key = None
            
            if position_key and position_key in self.positions:
                existing_position = self.positions[position_key]
                if existing_position.get("status") == "OPEN":
                    existing_signal = existing_position.get("signal", "UNKNOWN")
                    return {
                        "success": False,
                        "error": f"Ja existe uma posicao {existing_signal} {signal_source} aberta para {symbol}. Feche a posicao existente antes de abrir uma nova."
                    }
            
            # REMOVIDO: Verificação de saldo - sistema agora foca apenas em P&L
            
            # Criar trade
            trade_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trade = {
                "trade_id": trade_id,
                "symbol": symbol,
                "source": signal_source,  # DEEPSEEK ou AGNO
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
            
            # MODIFICADO: Não deduzir do saldo - sistema foca apenas em P&L
            # Apenas registrar a posição para tracking de P&L
            self.positions[position_key] = trade
            
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
                "message": f"Trade executado (P&L Mode): {signal_type} {position_size:.6f} unidades a ${entry_price:.2f}",
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
                for position_key, position in list(self.positions.items()):
                    # Extrair símbolo limpo da chave (pode ser SYMBOL_DEEPSEEK, SYMBOL_AGNO, SYMBOL_DEEPSEEK_SHORT, etc.)
                    clean_symbol = position_key.replace("_DEEPSEEK", "").replace("_AGNO", "").replace("_SHORT", "")
                    source = position.get("source", "UNKNOWN")
                    
                    current_price = await self.get_current_price(clean_symbol)
                    
                    if current_price is None:
                        logger.warning(f"[MONITOR] Nao foi possivel obter preco para {clean_symbol}")
                        continue
                    
                    # Log detalhado a cada verificação (INFO para visibilidade)
                    entry_price = position.get("entry_price", 0)
                    tp1 = position.get("take_profit_1", 0)
                    signal_type = position.get("signal", "UNKNOWN")
                    # Se source for UNKNOWN, tentar inferir da chave ou atualizar posição
                    if source == "UNKNOWN":
                        if "_DEEPSEEK" in position_key:
                            source = "DEEPSEEK"
                        elif "_AGNO" in position_key:
                            source = "AGNO"
                        else:
                            # Posições antigas sem source - tentar inferir do trade_id ou usar padrão
                            # Se não conseguir inferir, usar "LEGACY" e atualizar a posição
                            source = "LEGACY"
                            # Atualizar a posição com o source inferido
                            position["source"] = source
                            self._save_state()
                    logger.info(f"[MONITOR] {clean_symbol} ({signal_type} {source}): Preco ${current_price:.2f} | Entry ${entry_price:.2f} | TP1 ${tp1:.2f}")
                    
                    # Calcular P&L atual (apenas em %)
                    entry_price = position["entry_price"]
                    signal_type = position["signal"]
                    
                    if signal_type == "BUY":
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    else:  # SELL
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    
                    # Atualizar máximo lucro/prejuízo em % e SALVAR estado
                    if pnl_percent > position.get("max_profit_reached_percent", -999):
                        position["max_profit_reached_percent"] = pnl_percent
                        # Salvar estado quando atinge novo máximo
                        self._save_state()
                    if pnl_percent < position.get("max_loss_reached_percent", 999):
                        position["max_loss_reached_percent"] = pnl_percent
                        # Salvar estado quando atinge novo mínimo
                        self._save_state()
                    
                    # Verificar stop loss
                    if position.get("stop_loss"):
                        sl = position["stop_loss"]
                        if signal_type == "BUY" and current_price <= sl:
                            logger.warning(f"[STOP LOSS] Stop Loss atingido para {clean_symbol} {source}: ${current_price:.2f} (SL: ${sl:.2f})")
                            await self._close_position_auto(position_key, current_price, "STOP_LOSS")
                            continue
                        elif signal_type == "SELL" and current_price >= sl:
                            logger.warning(f"[STOP LOSS] Stop Loss atingido para {clean_symbol} {source}: ${current_price:.2f} (SL: ${sl:.2f})")
                            await self._close_position_auto(position_key, current_price, "STOP_LOSS")
                            continue

                    # Verificar take profit 1 (CORRIGIDO: fechar apenas 50% da posição)
                    if position.get("take_profit_1") and not position.get("tp1_partial_closed", False):
                        tp1 = position["take_profit_1"]
                        if signal_type == "BUY" and current_price >= tp1:
                            # Verificar se o preço REAL atingiu o take profit
                            logger.warning(f"[TAKE PROFIT 1] Take Profit 1 atingido para {clean_symbol} {source}: ${current_price:.2f} (TP: ${tp1:.2f})")
                            await self._close_position_partial(position_key, current_price, "TAKE_PROFIT_1", partial_percent=0.5)
                            continue
                        elif signal_type == "SELL" and current_price <= tp1:
                            logger.warning(f"[TAKE PROFIT 1] Take Profit 1 atingido para {clean_symbol} {source}: ${current_price:.2f} (TP: ${tp1:.2f})")
                            await self._close_position_partial(position_key, current_price, "TAKE_PROFIT_1", partial_percent=0.5)
                            continue

                    # Verificar take profit 2 (fechar 50% restante quando TP2 for atingido)
                    if position.get("take_profit_2") and position.get("tp1_partial_closed", False):
                        tp2 = position["take_profit_2"]
                        if signal_type == "BUY" and current_price >= tp2:
                            logger.warning(f"[TAKE PROFIT 2] Take Profit 2 atingido para {clean_symbol} {source}: ${current_price:.2f} (TP: ${tp2:.2f}) - Fechando 50% restante")
                            await self._close_position_auto(position_key, current_price, "TAKE_PROFIT_2")
                            continue
                        elif signal_type == "SELL" and current_price <= tp2:
                            logger.warning(f"[TAKE PROFIT 2] Take Profit 2 atingido para {clean_symbol} {source}: ${current_price:.2f} (TP: ${tp2:.2f}) - Fechando 50% restante")
                            await self._close_position_auto(position_key, current_price, "TAKE_PROFIT_2")
                            continue
                    
                    # Log de monitoramento (apenas %)
                    self._log_monitoring(clean_symbol, current_price, pnl_percent)
                
                # Pausa entre verificações
                await asyncio.sleep(5)  # Verifica a cada 5 segundos

            except asyncio.CancelledError:
                logger.info("Monitoramento cancelado pelo usuário")
                break
            except Exception as e:
                logger.exception(f"❌ Erro no monitoramento: {e}")
                await asyncio.sleep(10)
    
    async def _close_position_partial(self, position_key: str, current_price: float, reason: str, partial_percent: float = 0.5):
        """
        Fecha parcialmente uma posição (ex: 50% no TP1, 50% restante no TP2)
        
        Args:
            position_key: Chave da posição (ex: BTCUSDT_DEEPSEEK, BTCUSDT_AGNO_SHORT)
            current_price: Preço atual para fechamento
            reason: Motivo do fechamento (TAKE_PROFIT_1, TAKE_PROFIT_2, STOP_LOSS)
            partial_percent: Porcentagem da posição a fechar (0.5 = 50%)
        """
        try:
            position = self.positions[position_key]
            symbol = position.get("symbol", position_key.split("_")[0])  # Extrair símbolo da posição ou da chave
            entry_price = position["entry_price"]
            original_position_size = position.get("original_position_size", position["position_size"])
            current_position_size = position["position_size"]
            signal_type = position["signal"]
            
            # Calcular quantidade a fechar
            size_to_close = current_position_size * partial_percent
            size_remaining = current_position_size - size_to_close

            # Calcular P&L da parte fechada em %
            if signal_type == "BUY":
                pnl_percent_this_part = ((current_price - entry_price) / entry_price) * 100
            else:  # SELL (SHORT)
                pnl_percent_this_part = ((entry_price - current_price) / entry_price) * 100

            # P&L proporcional: multiplicar pelo peso da parte fechada
            # Se fechou 50%, o impacto no P&L total é 50% do pnl_percent
            weighted_pnl_percent = pnl_percent_this_part * partial_percent

            # Registrar fechamento parcial no histórico
            partial_close_entry = {
                "trade_id": f"{position.get('trade_id')}_partial_{reason}",
                "symbol": symbol,
                "source": position.get("source", "UNKNOWN"),
                "signal": signal_type,
                "entry_price": entry_price,
                "close_price": current_price,
                "position_size_closed": size_to_close,
                "partial_percent": partial_percent * 100,
                "pnl_percent": weighted_pnl_percent,  # P&L ponderado
                "pnl_percent_raw": pnl_percent_this_part,  # P&L bruto para referência
                "status": "CLOSED_PARTIAL",
                "close_timestamp": datetime.now().isoformat(),
                "close_reason": reason
            }
            self.trade_history.append(partial_close_entry)

            # Atualizar posição: reduzir tamanho e marcar que TP1 foi parcialmente fechado
            position["position_size"] = size_remaining
            position["tp1_partial_closed"] = True
            position["partial_close_price"] = current_price
            position["partial_close_pnl_percent"] = weighted_pnl_percent  # Guardar P&L ponderado
            
            # Salvar estado
            self._save_state()

            # Log de fechamento parcial (apenas %)
            logger.warning(f"[FECHADO PARCIAL] {reason}: {symbol} - {partial_percent*100:.0f}% fechado a ${current_price:.2f} | P&L parte: {pnl_percent_this_part:+.2f}% | P&L ponderado: {weighted_pnl_percent:+.2f}%")
            self._log_trade_close(f"{symbol}_partial", current_price, weighted_pnl_percent, f"{reason}_PARTIAL")

        except KeyError as e:
            logger.error(f"Posição não encontrada: {symbol} - {e}")
        except Exception as e:
            logger.exception(f"❌ Erro ao fechar parcialmente posição {symbol}: {e}")
    
    async def _close_position_auto(self, position_key: str, current_price: float, reason: str):
        """Fecha posição automaticamente com cálculo de P&L total incluindo partes anteriores"""
        try:
            position = self.positions[position_key]
            symbol = position.get("symbol", position_key.split("_")[0])  # Extrair símbolo da posição ou da chave
            entry_price = position["entry_price"]
            signal_type = position["signal"]

            # P&L da parte atual
            if signal_type == "BUY":
                current_pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:
                current_pnl_percent = ((entry_price - current_price) / entry_price) * 100

            # Se teve fechamento parcial anterior, calcular P&L ponderado
            if position.get("tp1_partial_closed") and position.get("partial_close_pnl_percent") is not None:
                # P&L da primeira parte (já ponderado por 50%)
                first_part_pnl = position["partial_close_pnl_percent"]
                # P&L da segunda parte (ponderado por 50% restante)
                second_part_pnl = current_pnl_percent * 0.5  # 50% restante
                # P&L total
                total_pnl_percent = first_part_pnl + second_part_pnl
            else:
                # Fechamento total (sem parcial anterior)
                total_pnl_percent = current_pnl_percent
            
            # Atualizar trade no histórico ANTES de remover
            for trade in self.trade_history:
                if trade.get("trade_id") == position.get("trade_id") and trade.get("status") != "CLOSED":
                    trade["close_price"] = current_price
                    trade["pnl_percent"] = total_pnl_percent  # P&L em %
                    trade["status"] = "CLOSED"
                    trade["close_timestamp"] = datetime.now().isoformat()
                    trade["close_reason"] = reason
                    break
            
            # Atualizar posição
            position["close_price"] = current_price
            position["pnl_percent"] = total_pnl_percent
            position["status"] = "CLOSED"
            position["close_timestamp"] = datetime.now().isoformat()
            position["close_reason"] = reason
            
            # Remover da posição ativa (usar position_key, não symbol)
            del self.positions[position_key]
            
            # Salvar estado IMEDIATAMENTE (CRÍTICO)
            self._save_state()

            # Log de fechamento (apenas %)
            source = position.get("source", "UNKNOWN")
            logger.warning(f"[FECHADO] {reason}: {symbol} {source} fechado a ${current_price:.2f} | P&L Total: {total_pnl_percent:+.2f}%")
            self._log_trade_close(symbol, current_price, total_pnl_percent, reason)

        except KeyError as e:
            logger.error(f"Posição não encontrada: {position_key} - {e}")
        except Exception as e:
            logger.exception(f"❌ Erro ao fechar posição {position_key}: {e}")

    async def close_position_manual(self, position_key: str, current_price: float) -> Dict[str, Any]:
        """
        Fecha uma posição manualmente pelo dashboard.

        Args:
            position_key: Chave da posição (ex: "BTCUSDT_AGNO")
            current_price: Preço atual de mercado

        Returns:
            Dict com status e mensagem
        """
        try:
            # Validar se posição existe
            if position_key not in self.positions:
                return {
                    "success": False,
                    "error": f"Posição {position_key} não encontrada"
                }

            position = self.positions[position_key]

            # Validar se está aberta
            if position.get("status") == "CLOSED":
                return {
                    "success": False,
                    "error": f"Posição {position_key} já está fechada"
                }

            # Fechar usando a lógica existente
            await self._close_position_auto(position_key, current_price, "MANUAL")

            return {
                "success": True,
                "message": f"Posição {position_key} fechada com sucesso a ${current_price:.2f}"
            }

        except Exception as e:
            logger.exception(f"❌ Erro ao fechar posição manualmente {position_key}: {e}")
            return {
                "success": False,
                "error": f"Erro ao fechar posição: {str(e)}"
            }

    def _log_monitoring(self, symbol: str, price: float, pnl_percent: float):
        """Log de monitoramento (apenas %)"""
        logger.debug(f"📊 {symbol}: ${price:.2f} | P&L: {pnl_percent:+.2f}%")
    
    def _log_trade_close(self, symbol: str, price: float, pnl_percent: float, reason: str):
        """Log de fechamento de trade (apenas %)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "close_price": price,
            "pnl_percent": pnl_percent,
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
        """Retorna resumo do portfólio focado em P&L em PORCENTAGEM"""
        try:
            # Calcular P&L de posições abertas (unrealized P&L em %)
            open_positions_pnl = []
            
            # Obter preços atuais para calcular P&L não realizado
            # MODIFICADO: Simplificado - usar apenas se não houver loop rodando
            # Se houver loop, pular cálculo de preços (será calculado no monitoramento)
            import asyncio
            open_positions_pnl = []
            
            # Tentar calcular P&L apenas se não houver loop assíncrono rodando
            try:
                asyncio.get_running_loop()
                # Há loop rodando - pular cálculo aqui (será feito no monitoramento)
                logger.debug("Loop assíncrono ativo, pulando cálculo de preços em get_portfolio_summary")
            except RuntimeError:
                # Sem loop rodando - pode calcular preços
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def _fetch_prices():
                        results = []
                        for position in self.positions.values():
                            symbol = position.get("symbol")
                            entry_price = position.get("entry_price", 0)
                            signal_type = position.get("signal", "BUY")
                            
                            try:
                                current_price = await self.get_current_price(symbol)
                                if current_price and entry_price > 0:
                                    if signal_type == "BUY":
                                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                                    else:  # SELL
                                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                                    results.append({
                                        "symbol": symbol,
                                        "pnl_percent": pnl_percent
                                    })
                            except:
                                pass
                        return results
                    
                    open_positions_pnl = loop.run_until_complete(_fetch_prices())
                    loop.close()
                except Exception as e:
                    logger.debug(f"Erro ao buscar preços para P&L: {e}")
                    open_positions_pnl = []
            
            # Calcular P&L acumulado de trades fechados (realized P&L em %)
            realized_pnl_percent = 0.0
            winning_trades = 0
            losing_trades = 0
            
            for trade in self.trade_history:
                if trade.get("status") in ["CLOSED", "CLOSED_PARTIAL"]:
                    pnl_percent = trade.get("pnl_percent", 0)
                    realized_pnl_percent += pnl_percent
                    if pnl_percent > 0:
                        winning_trades += 1
                    elif pnl_percent < 0:
                        losing_trades += 1
            
            # Calcular P&L não realizado médio (média das posições abertas)
            unrealized_pnl_percent = 0.0
            if open_positions_pnl:
                unrealized_pnl_percent = sum([p["pnl_percent"] for p in open_positions_pnl]) / len(open_positions_pnl)
            
            # P&L total acumulado (soma de todos os trades fechados)
            total_pnl_percent = realized_pnl_percent
            
            # Calcular win rate
            total_closed_trades = winning_trades + losing_trades
            win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
            
            return {
                "realized_pnl_percent": realized_pnl_percent,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "total_pnl_percent": total_pnl_percent,
                "open_positions_count": len(self.positions),
                "open_positions_pnl": open_positions_pnl,
                "total_trades": len(self.trade_history),
                "closed_trades": total_closed_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate_percent": win_rate,
                "is_monitoring": self.is_monitoring
            }
            
        except KeyError as ke:
            # Erro específico de chave faltando - pode ser current_balance ou outra chave antiga
            logger.error(f"Chave faltando ao calcular resumo do portfólio: {ke}")
            # Retornar resumo básico mesmo com erro
            return {
                "realized_pnl_percent": 0.0,
                "unrealized_pnl_percent": 0.0,
                "total_pnl_percent": 0.0,
                "open_positions_count": len(self.positions),
                "open_positions_pnl": [],
                "total_trades": len(self.trade_history),
                "closed_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_percent": 0.0,
                "is_monitoring": self.is_monitoring,
                "error": f"Chave faltando: {str(ke)}"
            }
        except Exception as e:
            logger.exception(f"Erro ao calcular resumo do portfólio: {e}")
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
        # MODIFICADO: Não resetar current_balance (sistema em modo P&L)
        self.positions = {}
        self.trade_history = []
        self._save_state()
        logger.info("Portfolio resetado para estado inicial (modo P&L)")
    
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
