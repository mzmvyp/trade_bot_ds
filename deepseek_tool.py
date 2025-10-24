"""
Ferramenta para o Agno chamar o DeepSeek
"""
import asyncio
import aiohttp
import json
from typing import Dict, Any
from config import settings

class DeepSeekTool:
    """
    Ferramenta para o Agno chamar o DeepSeek
    """
    
    def __init__(self):
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url
    
    async def analyze_trading_data(self, market_data: Dict, technical_signals: Dict, sentiment_data: Dict) -> str:
        """
        Analisa dados de trading usando DeepSeek
        """
        prompt = f"""
        Analise os seguintes dados de mercado e gere um sinal de trading:
        
        DADOS DO MERCADO:
        - Preço atual: {market_data.get('current_price', 'N/A')}
        - Variação 24h: {market_data.get('price_change_24h', 'N/A')}%
        - Volume 24h: {market_data.get('volume_24h', 'N/A')}
        - Taxa de funding: {market_data.get('funding_rate', 'N/A')}
        - Interesse aberto: {market_data.get('open_interest', 'N/A')}
        
        SINAIS TÉCNICOS:
        {json.dumps(technical_signals, indent=2, default=str)}
        
        SENTIMENTO DO MERCADO:
        {json.dumps(sentiment_data, indent=2, default=str)}
        
        Com base nesses dados, gere um sinal de trading com:
        1. Tipo de sinal (BUY/SELL/HOLD)
        2. Preço de entrada
        3. Stop loss técnico
        4. Dois alvos técnicos
        5. Justificativa baseada na análise técnica e sentimento
        6. Nível de confiança (1-10)
        
        Formate a resposta em JSON estruturado.
        """
        
        return await self._call_deepseek(prompt)
    
    async def _call_deepseek(self, prompt: str) -> str:
        """
        Chama a API do DeepSeek
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'deepseek-chat',
            'messages': [
                {'role': 'system', 'content': 'Você é um especialista em trading de criptomoedas com foco em Bitcoin futuros.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 2000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        print(f"Erro na API DeepSeek: {response.status} - {error_text}")
                        return None
        except Exception as e:
            print(f"Erro ao conectar com DeepSeek: {e}")
            return None

# Instância global da ferramenta
deepseek_tool = DeepSeekTool()
