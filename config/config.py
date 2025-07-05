import os
from dotenv import load_dotenv

load_dotenv()

class TradingConfig:
    """Configuration settings for the trading bot"""
    
    # API Configuration
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    # Trading Configuration
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
    RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '0.02'))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '10'))
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')
    
    # Trading Hours
    TRADING_START_HOUR = int(os.getenv('TRADING_START_HOUR', '9'))
    TRADING_END_HOUR = int(os.getenv('TRADING_END_HOUR', '16'))
    
    # News Sentiment Configuration
    NEWS_SENTIMENT_THRESHOLD = float(os.getenv('NEWS_SENTIMENT_THRESHOLD', '0.1'))
    NEWS_SOURCES = os.getenv('NEWS_SOURCES', 'reuters,bloomberg,cnbc,marketwatch').split(',')
    
    # Technical Indicators Configuration
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
    MACD_FAST = int(os.getenv('MACD_FAST', '12'))
    MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
    MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))
    BOLLINGER_PERIOD = int(os.getenv('BOLLINGER_PERIOD', '20'))
    BOLLINGER_STD = float(os.getenv('BOLLINGER_STD', '2'))
    
    # API Base URLs
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
    TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
    NEWS_API_BASE_URL = "https://newsapi.org/v2"
    
    # Trading Strategies
    STRATEGIES = {
        'RSI_OVERSOLD': 30,
        'RSI_OVERBOUGHT': 70,
        'MACD_BULLISH': 'macd_cross_above',
        'MACD_BEARISH': 'macd_cross_below',
        'BOLLINGER_LOWER': 'price_touch_lower',
        'BOLLINGER_UPPER': 'price_touch_upper'
    }
    
    # Risk Management
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.05'))
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '0.10'))
    
    # Position Sizing
    POSITION_SIZE_METHOD = os.getenv('POSITION_SIZE_METHOD', 'fixed_percentage')
    FIXED_POSITION_SIZE = float(os.getenv('FIXED_POSITION_SIZE', '0.1'))
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_keys = ['ALPHA_VANTAGE_API_KEY', 'TWELVEDATA_API_KEY']
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")
        
        return True