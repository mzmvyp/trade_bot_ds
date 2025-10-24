# Configuração do Twitter/X para Análise de Sentimento

## ⚠️ IMPORTANTE: Status Atual

**O sistema está usando FALLBACK (dados de mercado) porque o token do Twitter não está configurado.**

## 🔧 Como Configurar o Twitter Real

### 1. Obter Token do Twitter/X

1. Acesse [Twitter Developer Portal](https://developer.twitter.com/)
2. Crie uma conta de desenvolvedor
3. Crie um novo projeto/app
4. Gere um **Bearer Token**

### 2. Configurar no Sistema

Adicione ao arquivo `.env`:

```bash
TWITTER_BEARER_TOKEN=seu_token_aqui
```

### 3. Verificar se Funcionou

Execute o teste:

```bash
python test_twitter_debug.py
```

Se configurado corretamente, você verá:
```
🐦 Usando análise REAL do Twitter para BTCUSDT
```

## 🔍 Status Atual do Sistema

- ✅ **Bibliotecas**: `tweepy` e `vaderSentiment` instaladas
- ❌ **Token**: Não configurado
- 🔄 **Fallback**: Usando dados de mercado (não é Twitter real)

## 📊 O que o Fallback Faz

Quando o Twitter não está disponível, o sistema usa:
- Mudança de preço 24h
- Volume de negociação
- Taxa de funding
- Open interest
- Pressão de compra/venda

**Isso NÃO é análise real do Twitter, apenas dados de mercado.**
