"""
Ferramenta personalizada do Agno para chamar DeepSeek
"""
from agno.agent import Function
from deepseek_tool import deepseek_tool

def create_deepseek_function():
    """
    Cria uma função que o Agno pode usar para chamar o DeepSeek
    """
    async def analyze_with_deepseek(market_data: dict, technical_signals: dict, sentiment_data: dict) -> str:
        """
        Analisa dados de trading usando DeepSeek
        
        Args:
            market_data: Dados do mercado (preço, volume, etc.)
            technical_signals: Sinais técnicos (RSI, MACD, etc.)
            sentiment_data: Dados de sentimento do mercado
            
        Returns:
            Análise do DeepSeek em formato JSON
        """
        try:
            result = await deepseek_tool.analyze_trading_data(market_data, technical_signals, sentiment_data)
            return result if result else "Erro ao analisar com DeepSeek"
        except Exception as e:
            return f"Erro na análise: {str(e)}"
    
    return Function(
        name="analyze_with_deepseek",
        description="Analisa dados de trading usando DeepSeek AI",
        func=analyze_with_deepseek
    )

# Criar a ferramenta
deepseek_function = create_deepseek_function()
