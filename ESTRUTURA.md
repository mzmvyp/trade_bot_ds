# 📁 Estrutura do Sistema Limpo

## 🎯 **Arquivos Principais:**

### **Sistema Core:**
- `main.py` - Sistema principal de execução
- `trading_agent.py` - Agent de trading (Agno + DeepSeek)
- `config.py` - Configurações do sistema

### **Integrações:**
- `binance_client.py` - Cliente da API Binance
- `deepseek_tool.py` - Ferramenta DeepSeek
- `agno_deepseek_tool.py` - Integração Agno + DeepSeek

### **Análises:**
- `technical_analysis.py` - Análise técnica (indicadores)
- `sentiment_analysis.py` - Análise de sentimento (Twitter)

### **Configuração:**
- `requirements.txt` - Dependências Python
- `config.env.example` - Exemplo de configuração
- `install.py` - Script de instalação
- `test.py` - Teste do sistema

### **Sinais Gerados:**
- `signals/` - Pasta com sinais gerados
  - `signal_*.json` - Sinais individuais
  - `top10_summary_*.json` - Resumos top 10

### **Documentação:**
- `README.md` - Documentação principal
- `ESTRUTURA.md` - Este arquivo

## 🚀 **Comandos Disponíveis:**

```bash
# Instalação
python install.py

# Teste do sistema
python test.py

# Análise única
python main.py --mode single

# Monitoramento contínuo
python main.py --mode monitor
```

## ✅ **Sistema Limpo e Organizado!**

- ❌ Removidos arquivos duplicados
- ❌ Removidos arquivos de teste desnecessários
- ❌ Removidos markdowns repetitivos
- ❌ Removidos arquivos temporários
- ✅ Mantidos apenas arquivos essenciais
- ✅ Sistema funcionando perfeitamente
