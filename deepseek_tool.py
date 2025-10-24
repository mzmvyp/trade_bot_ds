"""
Sistema aprimorado de integração com DeepSeek
"""
import aiohttp
import json
import asyncio
import jsonschema
from typing import Dict, Any, Optional
from datetime import datetime
from config import settings
from logger import log_api_call, log_error

class EnhancedDeepSeekTool:
    """Ferramenta aprimorada para integração com DeepSeek"""
    
    def __init__(self):
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url
        self.retry_attempts = 3
        self.timeout = 30
        
        # Schema de validação para resposta
        self.response_schema = {
            "type": "object",
            "properties": {
                "signal": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
                "entry_price": {"type": "number", "minimum": 0},
                "stop_loss": {"type": "number", "minimum": 0},
                "take_profit_1": {"type": "number", "minimum": 0},
                "take_profit_2": {"type": "number", "minimum": 0},
                "risk_reward_ratio": {"type": "number", "minimum": 0},
                "reasoning": {
                    "type": "object",
                    "properties": {
                        "technical": {"type": "string"},
                        "sentiment": {"type": "string"},
                        "market_structure": {"type": "string"}
                    },
                    "required": ["technical", "sentiment", "market_structure"]
                },
                "warnings": {"type": "array", "items": {"type": "string"}},
                "timeframe": {"type": "string", "enum": ["curto", "médio", "longo prazo"]}
            },
            "required": ["signal", "confidence", "entry_price", "reasoning"]
        }
    
    async def analyze_with_structured_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa dados com saída estruturada do DeepSeek
        
        Args:
            data: Dados de mercado, técnicos e sentimento
            
        Returns:
            Dicionário com sinal estruturado
        """
        start_time = datetime.now()
        
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(data)
        
        for attempt in range(self.retry_attempts):
            try:
                log_api_call("DeepSeek", "chat/completions", "attempting", None)
                
                response = await self._call_api_with_timeout(system_prompt, user_prompt)
                
                # Validar e parsear resposta
                parsed_response = self._validate_and_parse_response(response)
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                log_api_call("DeepSeek", "chat/completions", "success", response_time)
                
                return parsed_response
                
            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                log_error(e, {"attempt": attempt + 1, "response": response[:200]})
                
                if attempt == self.retry_attempts - 1:
                    raise ValueError(f"Falha ao validar resposta do DeepSeek após {self.retry_attempts} tentativas: {e}")
                
                # Backoff exponencial
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                log_error(e, {"attempt": attempt + 1})
                
                if attempt == self.retry_attempts - 1:
                    raise
                
                await asyncio.sleep(2 ** attempt)
    
    def _create_system_prompt(self) -> str:
        """Cria prompt do sistema para DeepSeek"""
        return """
        Você é um trader profissional especializado em criptomoedas com 10+ anos de experiência.
        
        SUAS RESPONSABILIDADES:
        1. Analisar dados técnicos, fundamentais e de sentimento
        2. Identificar padrões de mercado e oportunidades
        3. Calcular níveis de entrada, stop loss e take profit
        4. Avaliar risco vs recompensa
        5. Fornecer justificativa clara para suas decisões
        
        FORMATO DE RESPOSTA OBRIGATÓRIO:
        Responda APENAS com JSON válido no formato exato abaixo:
        
        {
            "signal": "BUY|SELL|HOLD",
            "confidence": 1-10,
            "entry_price": número,
            "stop_loss": número,
            "take_profit_1": número,
            "take_profit_2": número,
            "risk_reward_ratio": número,
            "reasoning": {
                "technical": "análise técnica detalhada",
                "sentiment": "análise de sentimento",
                "market_structure": "estrutura de mercado"
            },
            "warnings": ["lista de avisos importantes"],
            "timeframe": "curto|médio|longo prazo"
        }
        
        REGRAS IMPORTANTES:
        - SEMPRE responda em JSON válido
        - NUNCA inclua texto fora do JSON
        - Use números decimais para preços
        - Seja conservador com confiança (1-10)
        - Justifique claramente suas decisões
        - Identifique riscos e avisos
        """
    
    def _create_user_prompt(self, data: Dict[str, Any]) -> str:
        """Cria prompt do usuário com dados de mercado"""
        market_data = data.get('market_data', {})
        technical_signals = data.get('technical_signals', {})
        sentiment_data = data.get('sentiment_data', {})
        
        return f"""
        Analise os seguintes dados de mercado e gere um sinal de trading:
        
        DADOS DO MERCADO:
        - Símbolo: {market_data.get('symbol', 'N/A')}
        - Preço atual: ${market_data.get('current_price', 0):,.2f}
        - Variação 24h: {market_data.get('price_change_24h', 0):.2f}%
        - Volume 24h: {market_data.get('volume_24h', 0):,.0f}
        - Taxa de funding: {market_data.get('funding_rate', 0):.4f}
        - Interesse aberto: {market_data.get('open_interest', 0):,.0f}
        
        SINAIS TÉCNICOS:
        {json.dumps(technical_signals, indent=2, default=str)}
        
        SENTIMENTO DO MERCADO:
        {json.dumps(sentiment_data, indent=2, default=str)}
        
        INSTRUÇÕES:
        1. Analise todos os dados fornecidos
        2. Identifique padrões e tendências
        3. Calcule níveis de entrada, stop loss e take profit
        4. Avalie o risco vs recompensa
        5. Determine o nível de confiança (1-10)
        6. Identifique avisos importantes
        7. Responda APENAS com JSON válido no formato especificado
        """
    
    async def _call_api_with_timeout(self, system_prompt: str, user_prompt: str) -> str:
        """Chama API do DeepSeek com timeout"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 2000,
            'response_format': {"type": "json_object"}
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    raise Exception(f"Erro na API DeepSeek: {response.status} - {error_text}")
    
    def _validate_and_parse_response(self, response: str) -> Dict[str, Any]:
        """Valida e parseia resposta do DeepSeek"""
        try:
            # Limpar resposta (remover markdown se presente)
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            # Parsear JSON
            parsed = json.loads(cleaned_response)
            
            # Validar com schema
            jsonschema.validate(parsed, self.response_schema)
            
            # Validações adicionais
            self._validate_signal_logic(parsed)
            
            return parsed
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Resposta não é JSON válido: {e}")
        except jsonschema.ValidationError as e:
            raise ValueError(f"Resposta não atende ao schema: {e}")
    
    def _validate_signal_logic(self, signal: Dict[str, Any]) -> None:
        """Valida lógica do sinal"""
        signal_type = signal.get('signal')
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        take_profit_1 = signal.get('take_profit_1', 0)
        take_profit_2 = signal.get('take_profit_2', 0)
        
        if signal_type == 'BUY':
            # Para BUY: stop_loss < entry_price < take_profit_1 < take_profit_2
            if not (stop_loss < entry_price < take_profit_1 < take_profit_2):
                raise ValueError("Lógica inválida para sinal BUY")
        elif signal_type == 'SELL':
            # Para SELL: take_profit_2 < take_profit_1 < entry_price < stop_loss
            if not (take_profit_2 < take_profit_1 < entry_price < stop_loss):
                raise ValueError("Lógica inválida para sinal SELL")
        elif signal_type == 'HOLD':
            # Para HOLD, não precisa validar preços
            pass
        else:
            raise ValueError(f"Tipo de sinal inválido: {signal_type}")
    
    async def get_analysis_summary(self, symbol: str) -> Dict[str, Any]:
        """Obtém resumo da análise para um símbolo"""
        return {
            'symbol': symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'api_status': 'operational',
            'retry_attempts': self.retry_attempts,
            'timeout': self.timeout
        }

# Instância global
enhanced_deepseek = EnhancedDeepSeekTool()
