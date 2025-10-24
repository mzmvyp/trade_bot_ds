# 🚀 Sistema de Trading de Criptomoedas

Sistema inteligente de trading de criptomoedas que utiliza análise técnica, sentimento do mercado e IA para gerar sinais de trading.

## 📋 Funcionalidades

- **Análise Técnica**: Indicadores como RSI, MACD, Bollinger Bands, ATR, ADX, etc.
- **Análise de Sentimento**: Monitoramento de Twitter/X para capturar sentimento do mercado
- **IA DeepSeek**: Análise avançada com modelo de linguagem
- **Geração de Sinais**: Sinais estruturados com entrada, stop loss e alvos
- **Integração Binance**: Dados em tempo real da API pública da Binance

## 🛠️ Instalação

1. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

2. **Configurar variáveis de ambiente:**
```bash
# Criar arquivo .env
DEEPSEEK_API_KEY=sk-05da405f34ff423ea4e7f5a2b5631adb
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
TRADING_SYMBOL=BTCUSDT
```

## 🚀 Uso

### **Análise Única**
```bash
python main.py --mode single
```

### **Monitoramento Contínuo**
```bash
python main.py --mode monitor --interval 300
```

### **Top 10 Criptomoedas**
```bash
python main.py --mode top10
```

### **Parâmetros**
- `--symbol`: Símbolo para trading (padrão: BTCUSDT)
- `--mode`: Modo de execução (single/monitor/top10)
- `--interval`: Intervalo em segundos para modo monitor (padrão: 300)

## 📊 Sinais de Trading

O sistema gera sinais completos com:
- **Tipo**: BUY/SELL/HOLD
- **Preço de Entrada**: Baseado no preço atual
- **Stop Loss**: Calculado usando ATR
- **Alvo 1**: Primeiro alvo de lucro
- **Alvo 2**: Segundo alvo de lucro
- **Confiança**: Nível de 1-10
- **Justificativa**: Explicação baseada na análise

## 🔝 Top 10 Criptomoedas Suportadas

O sistema analisa automaticamente as 10 principais criptomoedas por market cap:

1. **BTCUSDT** - Bitcoin
2. **ETHUSDT** - Ethereum
3. **BNBUSDT** - BNB
4. **SOLUSDT** - Solana
5. **XRPUSDT** - XRP
6. **ADAUSDT** - Cardano
7. **DOGEUSDT** - Dogecoin
8. **AVAXUSDT** - Avalanche
9. **DOTUSDT** - Polkadot
10. **LINKUSDT** - Chainlink

## 🏗️ Arquitetura

```
📁 agent_trade/
├── main.py                 # Sistema principal
├── trading_agent.py        # Agent de trading aprimorado
├── binance_client.py       # Cliente da API Binance
├── technical_analysis.py     # Análise técnica avançada
├── sentiment_analysis.py    # Análise de sentimento
├── deepseek_tool.py         # Ferramenta DeepSeek aprimorada
├── risk_management.py       # Sistema de gestão de risco
├── backtesting_engine.py    # Motor de backtesting
├── logger.py               # Sistema de logging
├── config.py               # Configurações
├── requirements.txt        # Dependências
├── install.py             # Instalador
├── signals/               # Pasta com sinais gerados
│   ├── signal_*.json     # Sinais individuais
│   └── top10_summary_*.json # Resumos top 10
├── logs/                  # Pasta com logs do sistema
└── README.md              # Documentação
```

## 🔧 Componentes

### **1. TradingAgent**
- Orquestra todo o sistema
- Coleta dados da Binance
- Calcula indicadores técnicos
- Analisa sentimento
- Gera sinais de trading

### **2. BinanceClient**
- Integração com API pública da Binance
- Dados de candlesticks
- Estatísticas de 24h
- Volume, funding rate, interesse aberto

### **3. TechnicalAnalyzer**
- Cálculo de indicadores técnicos
- Análise de tendências
- Identificação de suporte e resistência
- Geração de sinais técnicos

### **4. SentimentAnalyzer**
- Análise de tweets sobre Bitcoin
- Cálculo de sentimento do mercado
- Identificação de tendências sociais

### **5. DeepSeekTool**
- Integração com API do DeepSeek
- Análise avançada com IA
- Geração de sinais inteligentes

## ⚠️ Avisos Importantes

- **Educacional**: Sistema apenas para fins educacionais
- **Riscos**: Trading envolve riscos significativos
- **Responsabilidade**: Sempre faça sua própria pesquisa
- **Não Automático**: Sistema não executa trades automaticamente

## 📝 Licença

Este projeto é apenas para fins educacionais. Use por sua conta e risco.