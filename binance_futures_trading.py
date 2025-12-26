"""
Sistema de Trading Real na Binance Futures
Executa trades reais usando a API da Binance Futures (Testnet ou Produção)
"""

import asyncio
import aiohttp
import hashlib
import hmac
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from urllib.parse import urlencode

from config import settings
from logger import get_logger

# Setup logger
logger = get_logger(__name__)


class BinanceFuturesClient:
    """
    Cliente assíncrono para a API da Binance Futures.
    Suporta Testnet e Produção.
    """

    def __init__(self):
        """Inicializa o cliente com as configurações do .env"""
        self.api_key = settings.binance_real_api_key
        self.api_secret = settings.binance_real_api_secret
        self.testnet = settings.binance_testnet

        # URLs da API
        if self.testnet:
            self.base_url = "https://testnet.binancefuture.com"
            logger.info("[BINANCE] Usando TESTNET (ambiente de teste)")
        else:
            self.base_url = "https://fapi.binance.com"
            logger.warning("[BINANCE] Usando PRODUÇÃO (dinheiro real!)")

        # Verificar se as chaves estão configuradas
        if not self.api_key or not self.api_secret:
            logger.error("[BINANCE] API Key ou Secret não configurados!")
            raise ValueError("BINANCE_REAL_API_KEY e BINANCE_REAL_API_SECRET devem estar configurados no .env")

        # Criar diretório para logs de trades reais
        Path("real_trades").mkdir(exist_ok=True)

        # Cache de informações de símbolos
        self._symbol_info_cache: Dict[str, Any] = {}

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Gera assinatura HMAC-SHA256 para autenticação"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_timestamp(self) -> int:
        """Retorna timestamp em milissegundos"""
        return int(time.time() * 1000)

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """
        Faz requisição à API da Binance.

        Args:
            method: GET, POST, DELETE
            endpoint: Endpoint da API
            params: Parâmetros da requisição
            signed: Se True, adiciona assinatura

        Returns:
            Resposta da API em dict
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        headers = {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        if signed:
            params["timestamp"] = self._get_timestamp()
            params["recvWindow"] = 5000
            params["signature"] = self._generate_signature(params)

        try:
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        data = await response.json()
                        if response.status != 200:
                            logger.error(f"[BINANCE] Erro {response.status}: {data}")
                            return {"error": True, "code": response.status, "msg": data.get("msg", str(data))}
                        return data

                elif method == "POST":
                    async with session.post(url, data=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        data = await response.json()
                        if response.status != 200:
                            logger.error(f"[BINANCE] Erro {response.status}: {data}")
                            return {"error": True, "code": response.status, "msg": data.get("msg", str(data))}
                        return data

                elif method == "DELETE":
                    async with session.delete(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        data = await response.json()
                        if response.status != 200:
                            logger.error(f"[BINANCE] Erro {response.status}: {data}")
                            return {"error": True, "code": response.status, "msg": data.get("msg", str(data))}
                        return data

        except aiohttp.ClientError as e:
            logger.error(f"[BINANCE] Erro de conexão: {e}")
            return {"error": True, "msg": f"Erro de conexão: {str(e)}"}
        except asyncio.TimeoutError:
            logger.error("[BINANCE] Timeout na requisição")
            return {"error": True, "msg": "Timeout na requisição"}
        except Exception as e:
            logger.exception(f"[BINANCE] Erro inesperado: {e}")
            return {"error": True, "msg": f"Erro inesperado: {str(e)}"}

    # ========================================
    # MÉTODOS DE CONTA
    # ========================================

    async def get_account_info(self) -> Dict[str, Any]:
        """Obtém informações da conta Futures"""
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def get_balance(self) -> Dict[str, Any]:
        """Obtém saldo da conta"""
        result = await self._request("GET", "/fapi/v2/balance", signed=True)
        if isinstance(result, list):
            # Filtrar apenas USDT
            usdt_balance = next((b for b in result if b.get("asset") == "USDT"), None)
            return usdt_balance or {"asset": "USDT", "balance": "0", "availableBalance": "0"}
        return result

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Obtém posições abertas"""
        result = await self._request("GET", "/fapi/v2/positionRisk", signed=True)
        if isinstance(result, list):
            # Filtrar apenas posições com quantidade != 0
            return [p for p in result if float(p.get("positionAmt", 0)) != 0]
        return []

    # ========================================
    # MÉTODOS DE MERCADO
    # ========================================

    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Obtém informações do símbolo (precisão, limites, etc.)"""
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        result = await self._request("GET", "/fapi/v1/exchangeInfo")
        if result.get("error"):
            return result

        for s in result.get("symbols", []):
            if s.get("symbol") == symbol:
                self._symbol_info_cache[symbol] = s
                return s

        return {"error": True, "msg": f"Símbolo {symbol} não encontrado"}

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual do símbolo"""
        result = await self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol})
        if result.get("error"):
            return None
        return float(result.get("price", 0))

    def _round_quantity(self, quantity: float, step_size: str) -> float:
        """Arredonda quantidade de acordo com o step size do símbolo"""
        step = float(step_size)
        if step == 0:
            return quantity
        precision = len(step_size.rstrip('0').split('.')[-1]) if '.' in step_size else 0
        return round(quantity - (quantity % step), precision)

    def _round_price(self, price: float, tick_size: str) -> float:
        """Arredonda preço de acordo com o tick size do símbolo"""
        tick = float(tick_size)
        if tick == 0:
            return price
        precision = len(tick_size.rstrip('0').split('.')[-1]) if '.' in tick_size else 0
        return round(price - (price % tick), precision)

    # ========================================
    # MÉTODOS DE ORDENS
    # ========================================

    async def create_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        Cria ordem a mercado.

        Args:
            symbol: Par de trading (ex: BTCUSDT)
            side: BUY ou SELL
            quantity: Quantidade em unidades da moeda base
            reduce_only: Se True, só reduz posição existente

        Returns:
            Resposta da ordem
        """
        # Obter informações do símbolo para arredondar corretamente
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info.get("error"):
            return symbol_info

        # Encontrar step size para quantidade
        step_size = "0.001"  # Default
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step_size = f.get("stepSize", "0.001")
                break

        quantity = self._round_quantity(quantity, step_size)

        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        logger.info(f"[BINANCE] Criando ordem MARKET: {side} {quantity} {symbol}")
        result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)

        if not result.get("error"):
            logger.info(f"[BINANCE] Ordem executada: OrderID={result.get('orderId')}, AvgPrice={result.get('avgPrice')}")
            self._save_order_log(result, "MARKET")

        return result

    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Cria ordem limitada.

        Args:
            symbol: Par de trading
            side: BUY ou SELL
            quantity: Quantidade
            price: Preço limite
            time_in_force: GTC, IOC, FOK

        Returns:
            Resposta da ordem
        """
        # Obter informações do símbolo
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info.get("error"):
            return symbol_info

        # Encontrar precisões
        step_size = "0.001"
        tick_size = "0.01"
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step_size = f.get("stepSize", "0.001")
            elif f.get("filterType") == "PRICE_FILTER":
                tick_size = f.get("tickSize", "0.01")

        quantity = self._round_quantity(quantity, step_size)
        price = self._round_price(price, tick_size)

        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "quantity": quantity,
            "price": price,
            "timeInForce": time_in_force
        }

        logger.info(f"[BINANCE] Criando ordem LIMIT: {side} {quantity} {symbol} @ {price}")
        result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)

        if not result.get("error"):
            logger.info(f"[BINANCE] Ordem criada: OrderID={result.get('orderId')}")
            self._save_order_log(result, "LIMIT")

        return result

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        close_position: bool = True
    ) -> Dict[str, Any]:
        """
        Cria ordem de Stop Loss.

        Args:
            symbol: Par de trading
            side: BUY (para posição short) ou SELL (para posição long)
            quantity: Quantidade
            stop_price: Preço de ativação do stop
            close_position: Se True, fecha toda a posição

        Returns:
            Resposta da ordem
        """
        # Obter informações do símbolo
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info.get("error"):
            return symbol_info

        # Encontrar precisões
        step_size = "0.001"
        tick_size = "0.01"
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step_size = f.get("stepSize", "0.001")
            elif f.get("filterType") == "PRICE_FILTER":
                tick_size = f.get("tickSize", "0.01")

        quantity = self._round_quantity(quantity, step_size)
        stop_price = self._round_price(stop_price, tick_size)

        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "STOP_MARKET",
            "stopPrice": stop_price,
            "workingType": "MARK_PRICE"  # Usar mark price para evitar liquidações
        }

        if close_position:
            params["closePosition"] = "true"
        else:
            params["quantity"] = quantity

        logger.info(f"[BINANCE] Criando STOP LOSS: {side} {symbol} @ {stop_price}")
        result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)

        if not result.get("error"):
            logger.info(f"[BINANCE] Stop Loss criado: OrderID={result.get('orderId')}")
            self._save_order_log(result, "STOP_LOSS")

        return result

    async def create_take_profit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        close_position: bool = False
    ) -> Dict[str, Any]:
        """
        Cria ordem de Take Profit.

        Args:
            symbol: Par de trading
            side: BUY (para posição short) ou SELL (para posição long)
            quantity: Quantidade
            take_profit_price: Preço de ativação do take profit
            close_position: Se True, fecha toda a posição

        Returns:
            Resposta da ordem
        """
        # Obter informações do símbolo
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info.get("error"):
            return symbol_info

        # Encontrar precisões
        step_size = "0.001"
        tick_size = "0.01"
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step_size = f.get("stepSize", "0.001")
            elif f.get("filterType") == "PRICE_FILTER":
                tick_size = f.get("tickSize", "0.01")

        quantity = self._round_quantity(quantity, step_size)
        take_profit_price = self._round_price(take_profit_price, tick_size)

        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": take_profit_price,
            "workingType": "MARK_PRICE"
        }

        if close_position:
            params["closePosition"] = "true"
        else:
            params["quantity"] = quantity

        logger.info(f"[BINANCE] Criando TAKE PROFIT: {side} {symbol} @ {take_profit_price}")
        result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)

        if not result.get("error"):
            logger.info(f"[BINANCE] Take Profit criado: OrderID={result.get('orderId')}")
            self._save_order_log(result, "TAKE_PROFIT")

        return result

    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancela uma ordem específica"""
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        return await self._request("DELETE", "/fapi/v1/order", params=params, signed=True)

    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancela todas as ordens abertas de um símbolo"""
        params = {"symbol": symbol}
        return await self._request("DELETE", "/fapi/v1/allOpenOrders", params=params, signed=True)

    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Obtém ordens abertas"""
        params = {}
        if symbol:
            params["symbol"] = symbol
        result = await self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)
        return result if isinstance(result, list) else []

    # ========================================
    # MÉTODOS DE ALAVANCAGEM
    # ========================================

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Define alavancagem para um símbolo.

        Args:
            symbol: Par de trading
            leverage: Alavancagem (1-125)

        Returns:
            Resposta da API
        """
        params = {
            "symbol": symbol,
            "leverage": min(max(leverage, 1), 125)  # Limitar entre 1 e 125
        }
        logger.info(f"[BINANCE] Configurando alavancagem: {symbol} = {leverage}x")
        return await self._request("POST", "/fapi/v1/leverage", params=params, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str = "CROSSED") -> Dict[str, Any]:
        """
        Define tipo de margem (CROSSED ou ISOLATED).

        Args:
            symbol: Par de trading
            margin_type: CROSSED ou ISOLATED

        Returns:
            Resposta da API
        """
        params = {
            "symbol": symbol,
            "marginType": margin_type.upper()
        }
        logger.info(f"[BINANCE] Configurando margem: {symbol} = {margin_type}")
        return await self._request("POST", "/fapi/v1/marginType", params=params, signed=True)

    # ========================================
    # MÉTODOS AUXILIARES
    # ========================================

    def _save_order_log(self, order: Dict[str, Any], order_type: str):
        """Salva log de ordem executada"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(f"real_trades/order_{timestamp}_{order.get('orderId', 'unknown')}.json")

            log_data = {
                "timestamp": datetime.now().isoformat(),
                "order_type": order_type,
                "testnet": self.testnet,
                "order": order
            }

            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2)

            logger.debug(f"[BINANCE] Log de ordem salvo: {log_file}")

        except Exception as e:
            logger.error(f"[BINANCE] Erro ao salvar log de ordem: {e}")


class RealTradingSystem:
    """
    Sistema de Trading Real que executa trades na Binance Futures.
    Integra com o sistema de sinais existente.
    """

    def __init__(self):
        """Inicializa o sistema de trading real"""
        self.client = BinanceFuturesClient()
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.is_monitoring = False
        self.monitor_task = None

        # Criar diretório para estado
        Path("real_trades").mkdir(exist_ok=True)

        # Carregar estado se existir
        self._load_state()

    def _load_state(self):
        """Carrega estado anterior do sistema"""
        try:
            state_file = Path("real_trades/state.json")
            if state_file.exists():
                with open(state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.positions = state.get("positions", {})
                    logger.info(f"[REAL TRADING] Estado carregado: {len(self.positions)} posições")
        except Exception as e:
            logger.error(f"[REAL TRADING] Erro ao carregar estado: {e}")

    def _save_state(self):
        """Salva estado atual do sistema"""
        try:
            state = {
                "positions": self.positions,
                "last_update": datetime.now().isoformat()
            }

            state_file = Path("real_trades/state.json")
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            logger.debug(f"[REAL TRADING] Estado salvo: {len(self.positions)} posições")

        except Exception as e:
            logger.error(f"[REAL TRADING] Erro ao salvar estado: {e}")

    async def execute_trade(self, signal: Dict[str, Any], position_size: float) -> Dict[str, Any]:
        """
        Executa um trade real baseado no sinal.

        Args:
            signal: Sinal de trading do AGNO
            position_size: Tamanho da posição em unidades

        Returns:
            Resultado da execução
        """
        try:
            symbol = signal.get("symbol")
            signal_type = signal.get("signal")  # BUY ou SELL
            entry_price = signal.get("entry_price")
            stop_loss = signal.get("stop_loss")
            take_profit_1 = signal.get("take_profit_1")
            take_profit_2 = signal.get("take_profit_2")
            source = signal.get("source", "AGNO")

            # Validar dados obrigatórios
            if not all([symbol, signal_type, entry_price]):
                return {
                    "success": False,
                    "error": "Dados obrigatórios ausentes (symbol, signal, entry_price)"
                }

            if signal_type not in ["BUY", "SELL"]:
                return {
                    "success": False,
                    "error": f"Sinal {signal_type} não é executável"
                }

            logger.info(f"[REAL TRADING] Executando {signal_type} {symbol}: {position_size} unidades @ ${entry_price:.2f}")

            # 1. Configurar alavancagem (padrão: 5x para gestão de risco)
            leverage = 5
            await self.client.set_leverage(symbol, leverage)

            # 2. Executar ordem de entrada (MARKET)
            entry_order = await self.client.create_market_order(
                symbol=symbol,
                side=signal_type,
                quantity=position_size
            )

            if entry_order.get("error"):
                return {
                    "success": False,
                    "error": f"Erro na ordem de entrada: {entry_order.get('msg')}"
                }

            order_id = entry_order.get("orderId")
            executed_qty = float(entry_order.get("executedQty", position_size))
            avg_price = float(entry_order.get("avgPrice", entry_price))

            logger.info(f"[REAL TRADING] Ordem de entrada executada: ID={order_id}, Qty={executed_qty}, AvgPrice={avg_price}")

            # 3. Configurar Stop Loss
            sl_order = None
            if stop_loss:
                # Para BUY, stop loss é SELL. Para SELL (short), stop loss é BUY.
                sl_side = "SELL" if signal_type == "BUY" else "BUY"
                sl_order = await self.client.create_stop_loss_order(
                    symbol=symbol,
                    side=sl_side,
                    quantity=executed_qty,
                    stop_price=stop_loss,
                    close_position=True
                )

                if sl_order.get("error"):
                    logger.warning(f"[REAL TRADING] Erro ao criar Stop Loss: {sl_order.get('msg')}")
                else:
                    logger.info(f"[REAL TRADING] Stop Loss criado: {sl_order.get('orderId')} @ ${stop_loss:.2f}")

            # 4. Configurar Take Profit 1 (50% da posição)
            tp1_order = None
            if take_profit_1:
                tp_side = "SELL" if signal_type == "BUY" else "BUY"
                tp1_qty = executed_qty * 0.5  # 50% da posição
                tp1_order = await self.client.create_take_profit_order(
                    symbol=symbol,
                    side=tp_side,
                    quantity=tp1_qty,
                    take_profit_price=take_profit_1,
                    close_position=False
                )

                if tp1_order.get("error"):
                    logger.warning(f"[REAL TRADING] Erro ao criar TP1: {tp1_order.get('msg')}")
                else:
                    logger.info(f"[REAL TRADING] Take Profit 1 criado: {tp1_order.get('orderId')} @ ${take_profit_1:.2f} ({tp1_qty} unidades)")

            # 5. Configurar Take Profit 2 (50% restante)
            tp2_order = None
            if take_profit_2:
                tp_side = "SELL" if signal_type == "BUY" else "BUY"
                tp2_qty = executed_qty * 0.5  # 50% restante
                tp2_order = await self.client.create_take_profit_order(
                    symbol=symbol,
                    side=tp_side,
                    quantity=tp2_qty,
                    take_profit_price=take_profit_2,
                    close_position=False
                )

                if tp2_order.get("error"):
                    logger.warning(f"[REAL TRADING] Erro ao criar TP2: {tp2_order.get('msg')}")
                else:
                    logger.info(f"[REAL TRADING] Take Profit 2 criado: {tp2_order.get('orderId')} @ ${take_profit_2:.2f} ({tp2_qty} unidades)")

            # 6. Registrar posição
            position_key = f"{symbol}_{source}"
            if signal_type == "SELL":
                position_key += "_SHORT"

            trade_record = {
                "trade_id": f"REAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "symbol": symbol,
                "source": source,
                "signal": signal_type,
                "entry_order_id": order_id,
                "entry_price": avg_price,
                "position_size": executed_qty,
                "stop_loss": stop_loss,
                "take_profit_1": take_profit_1,
                "take_profit_2": take_profit_2,
                "sl_order_id": sl_order.get("orderId") if sl_order and not sl_order.get("error") else None,
                "tp1_order_id": tp1_order.get("orderId") if tp1_order and not tp1_order.get("error") else None,
                "tp2_order_id": tp2_order.get("orderId") if tp2_order and not tp2_order.get("error") else None,
                "timestamp": datetime.now().isoformat(),
                "status": "OPEN",
                "leverage": leverage,
                "testnet": self.client.testnet
            }

            self.positions[position_key] = trade_record
            self._save_state()

            # 7. Salvar trade individual
            self._save_trade(trade_record)

            return {
                "success": True,
                "order_id": order_id,
                "trade_id": trade_record["trade_id"],
                "message": f"Trade REAL executado: {signal_type} {executed_qty} {symbol} @ ${avg_price:.2f}",
                "entry_price": avg_price,
                "position_size": executed_qty,
                "stop_loss_order": sl_order.get("orderId") if sl_order and not sl_order.get("error") else None,
                "take_profit_1_order": tp1_order.get("orderId") if tp1_order and not tp1_order.get("error") else None,
                "take_profit_2_order": tp2_order.get("orderId") if tp2_order and not tp2_order.get("error") else None,
                "testnet": self.client.testnet
            }

        except Exception as e:
            logger.exception(f"[REAL TRADING] Erro ao executar trade: {e}")
            return {
                "success": False,
                "error": f"Erro ao executar trade: {str(e)}"
            }

    def _save_trade(self, trade: Dict[str, Any]):
        """Salva trade individual"""
        try:
            filename = Path(f"real_trades/trade_{trade['trade_id']}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(trade, f, indent=2)
            logger.debug(f"[REAL TRADING] Trade salvo: {filename}")
        except Exception as e:
            logger.error(f"[REAL TRADING] Erro ao salvar trade: {e}")

    async def close_position(self, symbol: str, position_key: str = None) -> Dict[str, Any]:
        """
        Fecha uma posição aberta.

        Args:
            symbol: Par de trading
            position_key: Chave da posição (opcional)

        Returns:
            Resultado do fechamento
        """
        try:
            # Cancelar todas as ordens pendentes
            await self.client.cancel_all_orders(symbol)

            # Obter posição atual
            positions = await self.client.get_positions()
            position = next((p for p in positions if p.get("symbol") == symbol), None)

            if not position or float(position.get("positionAmt", 0)) == 0:
                return {
                    "success": False,
                    "error": f"Nenhuma posição aberta para {symbol}"
                }

            position_amt = float(position.get("positionAmt", 0))
            side = "SELL" if position_amt > 0 else "BUY"
            quantity = abs(position_amt)

            # Fechar posição
            close_order = await self.client.create_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                reduce_only=True
            )

            if close_order.get("error"):
                return {
                    "success": False,
                    "error": f"Erro ao fechar posição: {close_order.get('msg')}"
                }

            # Atualizar registro local
            if position_key and position_key in self.positions:
                self.positions[position_key]["status"] = "CLOSED"
                self.positions[position_key]["close_timestamp"] = datetime.now().isoformat()
                self.positions[position_key]["close_order_id"] = close_order.get("orderId")
                self._save_state()

            return {
                "success": True,
                "message": f"Posição {symbol} fechada: {side} {quantity} unidades",
                "order_id": close_order.get("orderId"),
                "avg_price": close_order.get("avgPrice")
            }

        except Exception as e:
            logger.exception(f"[REAL TRADING] Erro ao fechar posição: {e}")
            return {
                "success": False,
                "error": f"Erro ao fechar posição: {str(e)}"
            }

    async def get_account_summary(self) -> Dict[str, Any]:
        """Obtém resumo da conta"""
        try:
            balance = await self.client.get_balance()
            positions = await self.client.get_positions()

            return {
                "balance": balance,
                "open_positions": positions,
                "local_positions": list(self.positions.values()),
                "testnet": self.client.testnet
            }

        except Exception as e:
            logger.error(f"[REAL TRADING] Erro ao obter resumo: {e}")
            return {"error": str(e)}


# ========================================
# FUNÇÕES DE INTERFACE
# ========================================

# Instância global do sistema
_real_trading_system: Optional[RealTradingSystem] = None


def get_real_trading_system() -> RealTradingSystem:
    """Obtém instância global do sistema de trading real"""
    global _real_trading_system
    if _real_trading_system is None:
        _real_trading_system = RealTradingSystem()
    return _real_trading_system


async def execute_real_trade(signal: Dict[str, Any], position_size: float) -> Dict[str, Any]:
    """
    Função de interface para executar trade real.
    Chamada pelo trading_agent_agno.py quando em modo LIVE.

    Args:
        signal: Sinal de trading
        position_size: Tamanho da posição

    Returns:
        Resultado da execução
    """
    system = get_real_trading_system()
    return await system.execute_trade(signal, position_size)


async def close_real_position(symbol: str, position_key: str = None) -> Dict[str, Any]:
    """
    Fecha uma posição real.

    Args:
        symbol: Par de trading
        position_key: Chave da posição

    Returns:
        Resultado do fechamento
    """
    system = get_real_trading_system()
    return await system.close_position(symbol, position_key)


async def get_real_account_summary() -> Dict[str, Any]:
    """Obtém resumo da conta real"""
    system = get_real_trading_system()
    return await system.get_account_summary()


# ========================================
# TESTE DO MÓDULO
# ========================================

async def _test_connection():
    """Testa conexão com a API da Binance"""
    try:
        client = BinanceFuturesClient()
        print(f"[TEST] Conectado à Binance {'TESTNET' if client.testnet else 'PRODUÇÃO'}")

        # Testar obtenção de saldo
        balance = await client.get_balance()
        print(f"[TEST] Saldo USDT: {balance}")

        # Testar obtenção de preço
        price = await client.get_current_price("BTCUSDT")
        print(f"[TEST] Preço BTC: ${price:.2f}")

        # Testar posições
        positions = await client.get_positions()
        print(f"[TEST] Posições abertas: {len(positions)}")

        print("[TEST] Conexão OK!")
        return True

    except Exception as e:
        print(f"[TEST] Erro: {e}")
        return False


if __name__ == "__main__":
    # Executar teste de conexão
    asyncio.run(_test_connection())
