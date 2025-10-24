"""
Trading Agent usando AGNO para orquestração
"""
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Importar todas as ferramentas
from agno_tools import (
    get_market_data,
    analyze_technical_indicators,
    analyze_market_sentiment,
    get_deepseek_analysis,
    validate_risk_and_position,
    execute_paper_trade,
    analyze_multiple_timeframes,
    analyze_order_flow,
    backtest_strategy
)

class AgnoTradingAgent:
    """
    Agent de trading que usa AGNO para orquestrar análises
    """
    
    def __init__(self, paper_trading: bool = True):
        """
        Inicializa o agent de trading.
        
        Args:
            paper_trading: Se True, apenas simula trades
        """
        self.paper_trading = paper_trading
        
        # Obter API key
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("⚠️ DEEPSEEK_API_KEY não encontrada. Executando em modo de demonstração.")
            api_key = "demo_key"  # Chave demo para testes
        
        # Aplicar decorator @tool nas ferramentas
        from agno.tools import tool
        
        # Configurar o Agent AGNO com otimizações de velocidade
        if api_key != "demo_key":
            # Modo real com DeepSeek
            self.agent = Agent(
                model=DeepSeek(id="deepseek-chat", api_key=api_key, temperature=0.3, max_tokens=1000),
                tools=[
                    tool(get_market_data),
                    tool(analyze_technical_indicators),
                    tool(analyze_market_sentiment),
                    tool(get_deepseek_analysis),
                    tool(validate_risk_and_position),
                    tool(execute_paper_trade),
                    tool(analyze_multiple_timeframes),
                    tool(analyze_order_flow),
                    tool(backtest_strategy)
                ],
                instructions=self._get_instructions()
            )
        else:
            # Modo demo - análise local sem DeepSeek
            self.agent = None
            self.demo_mode = True
        
        # Criar pastas necessárias
        Path("signals").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("paper_trades").mkdir(exist_ok=True)
    
    def _get_instructions(self) -> str:
        """Retorna as instruções para o agent"""
        return """
        Você é um trader profissional especializado em análise técnica e gestão de risco.
        
        PROCESSO DE ANÁLISE:
        1. Colete dados de mercado usando get_market_data()
        2. Analise indicadores técnicos com analyze_technical_indicators()
        3. Capture sentimento com analyze_market_sentiment()
        4. Analise multi-timeframe com analyze_multiple_timeframes()
        5. Analise order flow com analyze_order_flow()
        6. Processe análise DeepSeek com get_deepseek_analysis()
        7. Valide risco com validate_risk_and_position()
        8. Execute paper trade se apropriado com execute_paper_trade()
        9. Para backtesting, use backtest_strategy() com datas específicas
        
        REGRAS DE TRADING:
        - SEMPRE forneça um sinal: BUY ou SELL (seja decisivo)
        - NÃO use HOLD ou NÃO OPERAR
        - Para BUY/SELL, defina OBRIGATORIAMENTE:
          * Entrada: preço específico
          * Stop Loss: preço específico
          * Take Profit 1: preço específico
          * Take Profit 2: preço específico
          * Confiança: 1-10
        
        GESTÃO DE RISCO:
        - Confiança mínima 5 para executar
        - Respeite circuit breakers automáticos
        - Analise estrutura de mercado (suporte/resistência)
        - Considere múltiplos timeframes
        
        Seja detalhado na análise mas objetivo na decisão.
        """
    
    async def analyze(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Executa análise completa usando o AGNO Agent.
        
        Args:
            symbol: Símbolo para analisar
            
        Returns:
            Sinal de trading estruturado
        """
        print(f"\n🤖 AGNO Agent iniciando análise de {symbol}")
        print("="*60)
        
        # Prompt para o agent
        prompt = f"""
        Execute uma análise completa para {symbol} seguindo o processo definido:
        
        1. Colete dados de mercado usando get_market_data("{symbol}")
        2. Analise indicadores técnicos com analyze_technical_indicators("{symbol}")
        3. Capture sentimento com analyze_market_sentiment("{symbol}")
        4. Obtenha análise DeepSeek usando get_deepseek_analysis() com os dados coletados
        5. Valide o risco com validate_risk_and_position()
        6. Se apropriado, execute paper trade com execute_paper_trade()
        
        Forneça:
        - Análise detalhada de cada componente
        - Sinal final com justificativa
        - Níveis de entrada, stop loss e take profit
        - Avisos e considerações de risco
        
        Seja detalhado mas objetivo.
        """
        
        try:
            if hasattr(self, 'demo_mode') and self.demo_mode:
                # Modo demo - análise local
                signal = self._demo_analysis(symbol)
            else:
                # Executar agent - ELE VAI ORQUESTRAR TUDO!
                response = self.agent.run(prompt)
                
                # Processar resposta
                signal = self._process_agent_response(response, symbol)
            
            # Salvar sinal
            self._save_signal(signal)
            
            # Imprimir resumo
            self._print_summary(signal)
            
            return signal
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return self._create_error_signal(symbol, str(e))
    
    def _process_agent_response(self, response: Any, symbol: str) -> Dict[str, Any]:
        """Processa resposta do agent em formato estruturado"""
        
        # Extrair informações da resposta
        signal = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "agent_response": str(response),
        }
        
        # Tentar extrair sinal estruturado
        response_text = str(response)
        
        # Identificar tipo de sinal - APENAS BUY/SELL
        if "BUY" in response_text.upper():
            signal["signal"] = "BUY"
        elif "SELL" in response_text.upper():
            signal["signal"] = "SELL"
        else:
            # Se não conseguiu identificar, não executar
            signal["signal"] = "NO_SIGNAL"
        
        # Extrair números usando regex
        import re
        
        # Para NO_SIGNAL, não deve ter entrada, stop ou targets
        if signal["signal"] == "NO_SIGNAL":
            # NO_SIGNAL = não executar
            signal["entry_price"] = None
            signal["stop_loss"] = None
            signal["take_profit_1"] = None
            signal["take_profit_2"] = None
        else:
            # Para BUY/SELL, OBRIGATÓRIO ter entrada, stop e targets
            entry_patterns = [
                r"entrada[^0-9]*([0-9,]+\.?[0-9]*)",
                r"entry[^0-9]*([0-9,]+\.?[0-9]*)",
                r"preço[^0-9]*([0-9,]+\.?[0-9]*)"
            ]
            
            signal["entry_price"] = None
            for pattern in entry_patterns:
                entry_match = re.search(pattern, response_text, re.IGNORECASE)
                if entry_match:
                    try:
                        price = float(entry_match.group(1).replace(",", ""))
                        # Validar se o preço é realista (entre 1.000 e 1.000.000)
                        if 1000 <= price <= 1000000:
                            signal["entry_price"] = price
                            break
                    except ValueError:
                        continue
            
            # OBRIGATÓRIO: Stop Loss
            stop_patterns = [
                r"stop[^0-9]*loss[^0-9]*([0-9,]+\.?[0-9]*)",
                r"stop[^0-9]*([0-9,]+\.?[0-9]*)"
            ]
            
            signal["stop_loss"] = None
            for pattern in stop_patterns:
                stop_match = re.search(pattern, response_text, re.IGNORECASE)
                if stop_match:
                    try:
                        price = float(stop_match.group(1).replace(",", ""))
                        if 1000 <= price <= 1000000:
                            signal["stop_loss"] = price
                            break
                    except ValueError:
                        continue
            
            # OBRIGATÓRIO: Take Profit 1
            tp1_patterns = [
                r"take[^0-9]*profit[^0-9]*1[^0-9]*([0-9,]+\.?[0-9]*)",
                r"alvo[^0-9]*1[^0-9]*([0-9,]+\.?[0-9]*)",
                r"target[^0-9]*1[^0-9]*([0-9,]+\.?[0-9]*)"
            ]
            
            signal["take_profit_1"] = None
            for pattern in tp1_patterns:
                tp1_match = re.search(pattern, response_text, re.IGNORECASE)
                if tp1_match:
                    try:
                        price = float(tp1_match.group(1).replace(",", ""))
                        if 1000 <= price <= 1000000:
                            signal["take_profit_1"] = price
                            break
                    except ValueError:
                        continue
            
            # OBRIGATÓRIO: Take Profit 2
            tp2_patterns = [
                r"take[^0-9]*profit[^0-9]*2[^0-9]*([0-9,]+\.?[0-9]*)",
                r"alvo[^0-9]*2[^0-9]*([0-9,]+\.?[0-9]*)",
                r"target[^0-9]*2[^0-9]*([0-9,]+\.?[0-9]*)"
            ]
            
            signal["take_profit_2"] = None
            for pattern in tp2_patterns:
                tp2_match = re.search(pattern, response_text, re.IGNORECASE)
                if tp2_match:
                    try:
                        price = float(tp2_match.group(1).replace(",", ""))
                        if 1000 <= price <= 1000000:
                            signal["take_profit_2"] = price
                            break
                    except ValueError:
                        continue

        # Para BUY: stop loss ABAIXO da entrada, take profit ACIMA
        # Para SELL: stop loss ACIMA da entrada, take profit ABAIXO
        if signal["signal"] == "BUY":
            # Stop loss deve ser ABAIXO da entrada
            stop_pattern = r"stop loss[^0-9]*([0-9,]+\.?[0-9]*)"
            stop_match = re.search(stop_pattern, response_text, re.IGNORECASE)
            if stop_match:
                try:
                    stop_price = float(stop_match.group(1).replace(",", ""))
                    # Validar se stop loss é menor que entrada para BUY
                    if signal["entry_price"] and stop_price < signal["entry_price"]:
                        signal["stop_loss"] = stop_price
                except ValueError:
                    pass
        elif signal["signal"] == "SELL":
            # Stop loss deve ser ACIMA da entrada
            stop_pattern = r"stop loss[^0-9]*([0-9,]+\.?[0-9]*)"
            stop_match = re.search(stop_pattern, response_text, re.IGNORECASE)
            if stop_match:
                try:
                    stop_price = float(stop_match.group(1).replace(",", ""))
                    # Validar se stop loss é maior que entrada para SELL
                    if signal["entry_price"] and stop_price > signal["entry_price"]:
                        signal["stop_loss"] = stop_price
                except ValueError:
                    pass
        else:
            # Para HOLD, não definir stop loss (não executa)
            signal["stop_loss"] = None
        
        # Validação adicional: se não conseguiu extrair preço realista, usar preço atual
        if signal["entry_price"] is None or signal["entry_price"] < 1000:
            # Tentar extrair preço atual do texto
            current_price_pattern = r"preço[^0-9]*([0-9,]+\.?[0-9]*)"
            current_match = re.search(current_price_pattern, response_text, re.IGNORECASE)
            if current_match:
                try:
                    current_price = float(current_match.group(1).replace(",", ""))
                    if 1000 <= current_price <= 1000000:  # Preço realista para BTC
                        signal["entry_price"] = current_price
                except ValueError:
                    pass
        
        # Extrair confiança - corrigir regex para capturar corretamente
        conf_patterns = [
            r"confiança[^0-9]*([0-9]+)/10",
            r"confiança[^0-9]*([0-9]+)",
            r"confidence[^0-9]*([0-9]+)/10",
            r"confidence[^0-9]*([0-9]+)"
        ]
        
        signal["confidence"] = 5  # Default
        for pattern in conf_patterns:
            conf_match = re.search(pattern, response_text, re.IGNORECASE)
            if conf_match:
                signal["confidence"] = int(conf_match.group(1))
                break
        
        return signal
    
    def _save_signal(self, signal: Dict[str, Any]):
        """Salva sinal em arquivo JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"signals/agno_{signal['symbol']}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(signal, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Sinal salvo: {filename}")
    
    def _print_summary(self, signal: Dict[str, Any]):
        """Imprime resumo do sinal"""
        print("\n" + "="*60)
        print("📊 RESULTADO DA ANÁLISE")
        print("="*60)
        print(f"🎯 Sinal: {signal.get('signal', 'N/A')}")
        print(f"💪 Confiança: {signal.get('confidence', 0)}/10")
        if signal.get('entry_price'):
            print(f"💰 Entrada: ${signal['entry_price']:,.2f}")
        if signal.get('stop_loss'):
            print(f"🛑 Stop Loss: ${signal['stop_loss']:,.2f}")
        print("="*60)
    
    def _demo_analysis(self, symbol: str) -> Dict[str, Any]:
        """Análise demo local sem DeepSeek"""
        print(f"🔍 Executando análise demo local para {symbol}...")
        
        try:
            # Coletar dados
            market_data = get_market_data(symbol)
            technical_indicators = analyze_technical_indicators(symbol)
            sentiment = analyze_market_sentiment(symbol)
            
            print(f"📊 Dados coletados:")
            print(f"   Preço: ${market_data.get('current_price', 0):,.2f}")
            print(f"   Variação 24h: {market_data.get('price_change_24h', 0):.2f}%")
            print(f"   RSI: {technical_indicators.get('indicators', {}).get('rsi', 50):.2f}")
            print(f"   Tendência: {technical_indicators.get('trend', 'neutral')}")
            print(f"   Sentimento: {sentiment.get('sentiment', 'neutral')}")
            
            # Análise simples baseada em regras
            current_price = market_data.get('current_price', 0)
            price_change = market_data.get('price_change_24h', 0)
            rsi = technical_indicators.get('indicators', {}).get('rsi', 50)
            trend = technical_indicators.get('trend', 'neutral')
            
            # Lógica de decisão simples
            if rsi < 30 and trend == 'bearish' and price_change < -2:
                signal_type = "BUY"
                confidence = 7
                entry_price = current_price * 0.995  # Entrada 0.5% abaixo
                stop_loss = current_price * 0.97     # Stop 3% abaixo
                take_profit_1 = current_price * 1.02 # TP1 2% acima
                take_profit_2 = current_price * 1.05 # TP2 5% acima
            elif rsi > 70 and trend == 'bullish' and price_change > 2:
                signal_type = "SELL"
                confidence = 7
                entry_price = current_price * 1.005  # Entrada 0.5% acima
                stop_loss = current_price * 1.03     # Stop 3% acima
                take_profit_1 = current_price * 0.98 # TP1 2% abaixo
                take_profit_2 = current_price * 0.95 # TP2 5% abaixo
            else:
                signal_type = "BUY"  # Default para demo
                confidence = 5
                entry_price = current_price
                stop_loss = current_price * 0.97
                take_profit_1 = current_price * 1.02
                take_profit_2 = current_price * 1.05
            
            signal = {
                "symbol": symbol,
                "signal": signal_type,
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit_1": take_profit_1,
                "take_profit_2": take_profit_2,
                "timestamp": datetime.now().isoformat(),
                "demo_mode": True,
                "analysis": {
                    "market_data": market_data,
                    "technical_indicators": technical_indicators,
                    "sentiment": sentiment
                }
            }
            
            print(f"🎯 Sinal gerado: {signal_type} com confiança {confidence}/10")
            return signal
            
        except Exception as e:
            return self._create_error_signal(symbol, f"Erro na análise demo: {str(e)}")
    
    def _create_error_signal(self, symbol: str, error: str) -> Dict[str, Any]:
        """Cria sinal de erro"""
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    
    async def monitor_continuous(self, symbols: List[str], interval: int = 300):
        """
        Monitora múltiplos símbolos continuamente.
        
        Args:
            symbols: Lista de símbolos
            interval: Intervalo em segundos
        """
        print(f"🔄 Monitoramento contínuo de {symbols}")
        print(f"⏰ Intervalo: {interval}s")
        
        while True:
            for symbol in symbols:
                try:
                    await self.analyze(symbol)
                except Exception as e:
                    print(f"❌ Erro em {symbol}: {e}")
                
                await asyncio.sleep(10)  # Pausa entre símbolos
            
            print(f"💤 Aguardando {interval}s...")
            await asyncio.sleep(interval)