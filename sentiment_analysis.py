"""
Módulo de análise de sentimento via Twitter/X
"""
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
from config import settings

class SentimentAnalyzer:
    def __init__(self):
        self.bearer_token = settings.twitter_bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Busca tweets relacionados ao Bitcoin
        """
        if not self.bearer_token:
            return []
            
        url = f"{self.base_url}/tweets/search/recent"
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        
        params = {
            'query': query,
            'max_results': max_results,
            'tweet.fields': 'created_at,public_metrics,context_annotations,lang',
            'user.fields': 'verified,public_metrics'
        }
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                else:
                    print(f"Erro na API do Twitter: {response.status}")
                    return []
        except Exception as e:
            print(f"Erro ao buscar tweets: {e}")
            return []
    
    def analyze_tweet_sentiment(self, tweet_text: str) -> Dict:
        """
        Analisa o sentimento de um tweet individual
        """
        # Palavras positivas relacionadas ao Bitcoin
        positive_words = [
            'bull', 'bullish', 'moon', 'pump', 'buy', 'long', 'hodl', 'hold',
            'breakout', 'rally', 'surge', 'gain', 'profit', 'green', 'up',
            'strong', 'support', 'resistance', 'break', 'target', 'diamond hands'
        ]
        
        # Palavras negativas relacionadas ao Bitcoin
        negative_words = [
            'bear', 'bearish', 'dump', 'crash', 'sell', 'short', 'panic',
            'fear', 'red', 'down', 'weak', 'support broken', 'resistance',
            'correction', 'pullback', 'dip', 'crash', 'bubble', 'fud'
        ]
        
        # Palavras de volume/atividade
        volume_words = [
            'volume', 'liquidity', 'trading', 'active', 'busy', 'quiet',
            'slow', 'fast', 'momentum', 'volatility'
        ]
        
        text_lower = tweet_text.lower()
        
        # Contar palavras positivas e negativas
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        volume_count = sum(1 for word in volume_words if word in text_lower)
        
        # Calcular sentimento
        sentiment_score = positive_count - negative_count
        
        # Determinar sentimento
        if sentiment_score > 0:
            sentiment = "positive"
        elif sentiment_score < 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Calcular confiança baseada no número de palavras encontradas
        total_words = positive_count + negative_count
        confidence = min(total_words / 5, 1.0) if total_words > 0 else 0.0
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'confidence': confidence,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'volume_words': volume_count,
            'text': tweet_text
        }
    
    async def analyze_bitcoin_sentiment(self) -> Dict:
        """
        Analisa o sentimento geral do Bitcoin no Twitter
        """
        # Queries para buscar tweets sobre Bitcoin
        queries = [
            "bitcoin OR btc OR #bitcoin OR #btc -is:retweet lang:en",
            "bitcoin price OR btc price -is:retweet lang:en",
            "bitcoin trading OR btc trading -is:retweet lang:en"
        ]
        
        all_tweets = []
        
        # Buscar tweets para cada query
        for query in queries:
            tweets = await self.search_tweets(query, max_results=50)
            all_tweets.extend(tweets)
        
        # Remover duplicatas
        unique_tweets = []
        seen_ids = set()
        for tweet in all_tweets:
            if tweet['id'] not in seen_ids:
                unique_tweets.append(tweet)
                seen_ids.add(tweet['id'])
        
        # Analisar sentimento de cada tweet
        sentiments = []
        for tweet in unique_tweets:
            if 'text' in tweet:
                sentiment = self.analyze_tweet_sentiment(tweet['text'])
                sentiments.append(sentiment)
        
        if not sentiments:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0,
                'confidence': 0,
                'total_tweets': 0,
                'positive_tweets': 0,
                'negative_tweets': 0,
                'neutral_tweets': 0
            }
        
        # Calcular estatísticas gerais
        total_tweets = len(sentiments)
        positive_tweets = sum(1 for s in sentiments if s['sentiment'] == 'positive')
        negative_tweets = sum(1 for s in sentiments if s['sentiment'] == 'negative')
        neutral_tweets = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
        
        # Calcular score médio
        avg_score = sum(s['score'] for s in sentiments) / total_tweets
        
        # Calcular confiança média
        avg_confidence = sum(s['confidence'] for s in sentiments) / total_tweets
        
        # Determinar sentimento geral
        if avg_score > 0.5:
            overall_sentiment = 'positive'
        elif avg_score < -0.5:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': avg_score,
            'confidence': avg_confidence,
            'total_tweets': total_tweets,
            'positive_tweets': positive_tweets,
            'negative_tweets': negative_tweets,
            'neutral_tweets': neutral_tweets,
            'tweets_analyzed': sentiments[:10]  # Primeiros 10 tweets para referência
        }
    
    async def get_trending_topics(self) -> List[str]:
        """
        Obtém tópicos em tendência relacionados ao Bitcoin
        """
        # Esta é uma implementação simplificada
        # Em uma implementação real, você usaria a API de trending topics do Twitter
        trending_queries = [
            "bitcoin",
            "btc",
            "crypto",
            "cryptocurrency",
            "blockchain",
            "defi",
            "nft"
        ]
        
        return trending_queries
