import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from textblob import TextBlob
from newsapi import NewsApiClient
from config.config import TradingConfig

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """
    News sentiment analyzer for trading decisions
    """
    
    def __init__(self):
        self.news_api_key = TradingConfig.NEWS_API_KEY
        self.news_client = None
        
        if self.news_api_key:
            self.news_client = NewsApiClient(api_key=self.news_api_key)
            
        self.sentiment_threshold = TradingConfig.NEWS_SENTIMENT_THRESHOLD
        self.news_sources = TradingConfig.NEWS_SOURCES
        
    def get_news_sentiment(self, symbol: str, days_back: int = 7) -> Dict[str, float]:
        """
        Get overall sentiment for a stock symbol based on recent news
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back for news
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            # Get news articles
            articles = self.fetch_news_articles(symbol, days_back)
            
            if not articles:
                return {
                    'sentiment_score': 0.0,
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'article_count': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0
                }
                
            # Analyze sentiment for each article
            sentiment_scores = []
            polarity_scores = []
            subjectivity_scores = []
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for article in articles:
                # Combine title and description for analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Analyze sentiment
                sentiment = self.analyze_text_sentiment(text)
                
                sentiment_scores.append(sentiment['sentiment_score'])
                polarity_scores.append(sentiment['polarity'])
                subjectivity_scores.append(sentiment['subjectivity'])
                
                # Count sentiment categories
                if sentiment['sentiment_score'] > self.sentiment_threshold:
                    positive_count += 1
                elif sentiment['sentiment_score'] < -self.sentiment_threshold:
                    negative_count += 1
                else:
                    neutral_count += 1
                    
            # Calculate overall metrics
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            overall_polarity = sum(polarity_scores) / len(polarity_scores) if polarity_scores else 0.0
            overall_subjectivity = sum(subjectivity_scores) / len(subjectivity_scores) if subjectivity_scores else 0.0
            
            return {
                'sentiment_score': overall_sentiment,
                'polarity': overall_polarity,
                'subjectivity': overall_subjectivity,
                'article_count': len(articles),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0.0,
                'polarity': 0.0,
                'subjectivity': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
            
    def fetch_news_articles(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """
        Fetch news articles for a specific symbol
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            List of news articles
        """
        try:
            if not self.news_client:
                logger.warning("News API client not initialized")
                return []
                
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Search for news articles
            articles = []
            
            # Search with company symbol
            response = self.news_client.get_everything(
                q=symbol,
                sources=','.join(self.news_sources),
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                sort_by='relevancy',
                language='en'
            )
            
            if response and response.get('articles'):
                articles.extend(response['articles'])
                
            # Also search for company name if available
            company_name = self.get_company_name(symbol)
            if company_name:
                response = self.news_client.get_everything(
                    q=company_name,
                    sources=','.join(self.news_sources),
                    from_param=from_date.strftime('%Y-%m-%d'),
                    to=to_date.strftime('%Y-%m-%d'),
                    sort_by='relevancy',
                    language='en'
                )
                
                if response and response.get('articles'):
                    articles.extend(response['articles'])
                    
            # Remove duplicates
            seen_urls = set()
            unique_articles = []
            
            for article in articles:
                url = article.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_articles.append(article)
                    
            return unique_articles[:50]  # Limit to 50 articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
            
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            blob = TextBlob(text)
            
            # Get polarity (-1 to 1) and subjectivity (0 to 1)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert polarity to sentiment score (-1 to 1)
            sentiment_score = polarity
            
            return {
                'sentiment_score': sentiment_score,
                'polarity': polarity,
                'subjectivity': subjectivity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'polarity': 0.0,
                'subjectivity': 0.0
            }
            
    def get_company_name(self, symbol: str) -> Optional[str]:
        """
        Get company name from symbol (simple mapping)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company name if known, None otherwise
        """
        # Simple mapping - in production, this would use a comprehensive database
        symbol_to_name = {
            'AAPL': 'Apple',
            'GOOGL': 'Google',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'NVIDIA',
            'NFLX': 'Netflix',
            'BABA': 'Alibaba',
            'DIS': 'Disney',
            'PYPL': 'PayPal',
            'INTC': 'Intel',
            'AMD': 'AMD',
            'CRM': 'Salesforce',
            'ORCL': 'Oracle'
        }
        
        return symbol_to_name.get(symbol.upper())
        
    def get_market_sentiment(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get overall market sentiment based on multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with market sentiment metrics
        """
        try:
            all_sentiments = []
            valid_symbols = []
            
            for symbol in symbols:
                sentiment = self.get_news_sentiment(symbol)
                if sentiment['article_count'] > 0:
                    all_sentiments.append(sentiment)
                    valid_symbols.append(symbol)
                    
            if not all_sentiments:
                return {
                    'market_sentiment': 0.0,
                    'market_polarity': 0.0,
                    'market_subjectivity': 0.0,
                    'analyzed_symbols': 0
                }
                
            # Calculate overall market metrics
            market_sentiment = sum(s['sentiment_score'] for s in all_sentiments) / len(all_sentiments)
            market_polarity = sum(s['polarity'] for s in all_sentiments) / len(all_sentiments)
            market_subjectivity = sum(s['subjectivity'] for s in all_sentiments) / len(all_sentiments)
            
            return {
                'market_sentiment': market_sentiment,
                'market_polarity': market_polarity,
                'market_subjectivity': market_subjectivity,
                'analyzed_symbols': len(valid_symbols),
                'symbols': valid_symbols
            }
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {
                'market_sentiment': 0.0,
                'market_polarity': 0.0,
                'market_subjectivity': 0.0,
                'analyzed_symbols': 0
            }
            
    def get_sentiment_signal(self, symbol: str) -> str:
        """
        Get trading signal based on sentiment analysis
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            sentiment = self.get_news_sentiment(symbol)
            
            if sentiment['article_count'] < 5:
                return 'HOLD'  # Not enough data
                
            sentiment_score = sentiment['sentiment_score']
            positive_ratio = sentiment['positive_count'] / sentiment['article_count']
            negative_ratio = sentiment['negative_count'] / sentiment['article_count']
            
            # Strong positive sentiment
            if sentiment_score > 0.3 and positive_ratio > 0.6:
                return 'BUY'
            # Strong negative sentiment
            elif sentiment_score < -0.3 and negative_ratio > 0.6:
                return 'SELL'
            # Neutral sentiment
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error getting sentiment signal for {symbol}: {e}")
            return 'HOLD'
            
    def get_news_headlines(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get recent news headlines for a symbol
        
        Args:
            symbol: Stock symbol
            limit: Number of headlines to return
            
        Returns:
            List of news headlines with sentiment
        """
        try:
            articles = self.fetch_news_articles(symbol, days_back=3)
            
            headlines = []
            for article in articles[:limit]:
                sentiment = self.analyze_text_sentiment(article.get('title', ''))
                
                headlines.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'sentiment_score': sentiment['sentiment_score'],
                    'polarity': sentiment['polarity']
                })
                
            return headlines
            
        except Exception as e:
            logger.error(f"Error getting news headlines for {symbol}: {e}")
            return []