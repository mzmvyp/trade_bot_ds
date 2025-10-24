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
    execute_paper_trade
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
            raise ValueError("DEEPSEEK_API_KEY não encontrada. Configure no arquivo .env")
        
        # Aplicar decorator @tool nas ferramentas
        from agno.tools import tool
        
        # Configurar o Agent AGNO com otimizações de velocidade
        self.agent = Agent(
            model=DeepSeek(id="deepseek-chat", api_key=api_key, temperature=0.3, max_tokens=1000),
            tools=[
                tool(get_market_data),
                tool(analyze_technical_indicators),
                tool(analyze_market_sentiment),
                tool(get_deepseek_analysis),
                tool(validate_risk_and_position),
                tool(execute_paper_trade)
            ],
            instructions=self._get_instructions()
        )
        
        # Criar pastas necessárias
        Path("signals").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("paper_trades").mkdir(exist_ok=True)
    
    def _get_instructions(self) -> str:
        """Retorna as instruções para o agent"""
        return """
        Você é um trader profissional com acesso a dados reais de mercado.
        
        PROCESSO:
        1. get_market_data() - dados atuais
        2. analyze_technical_indicators() - indicadores técnicos
        3. analyze_market_sentiment() - sentimento de mercado
        4. get_deepseek_analysis() - análise final
        5. validate_risk_and_position() - validação
        
        SUA MISSÃO:
        Analise TODOS os dados fornecidos e decida:
        - BUY: Se os dados indicam oportunidade de compra
        - SELL: Se os dados indicam oportunidade de venda  
        - HOLD: Se os dados são inconclusivos ou negativos
        
        IMPORTANTE:
        - VOCÊ decide o sinal baseado nos dados reais
        - Confiança deve refletir sua certeza (1-10)
        - Para BUY/SELL: defina entrada, stop loss e take profits
        - Para HOLD: não defina níveis (aguardar)
        
        Seja decisivo e baseie-se nos dados fornecidos.
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
        
        # Identificar tipo de sinal - corrigir lógica
        if "**SINAL**: **HOLD**" in response_text or "SINAL: HOLD" in response_text or "SINAL FINAL: **HOLD**" in response_text:
            signal["signal"] = "HOLD"
        elif "**SINAL**: **BUY**" in response_text or "SINAL: BUY" in response_text or "SINAL FINAL: **BUY**" in response_text:
            signal["signal"] = "BUY"
        elif "**SINAL**: **SELL**" in response_text or "SINAL: SELL" in response_text or "SINAL FINAL: **SELL**" in response_text:
            signal["signal"] = "SELL"
        else:
            signal["signal"] = "HOLD"  # Default seguro
        
        # Extrair números usando regex
        import re
        
        # Para HOLD, não deve ter entrada, stop ou targets
        if signal["signal"] == "HOLD":
            # HOLD = aguardar, não executar
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