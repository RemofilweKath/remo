import requests
import time
import logging
from typing import Dict, Any, Optional, Union
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from twelvedata import TDClient
from config.config import TradingConfig

logger = logging.getLogger(__name__)

class APIClient:
    """
    Unified API client for Alpha Vantage and TwelveData APIs
    """
    
    def __init__(self, preferred_provider: str = 'alpha_vantage'):
        self.preferred_provider = preferred_provider
        self.alpha_vantage_key = TradingConfig.ALPHA_VANTAGE_API_KEY
        self.twelvedata_key = TradingConfig.TWELVEDATA_API_KEY
        
        # Initialize API clients
        self.alpha_vantage_ts = None
        self.alpha_vantage_ti = None
        self.twelvedata_client = None
        
        if self.alpha_vantage_key:
            self.alpha_vantage_ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            self.alpha_vantage_ti = TechIndicators(key=self.alpha_vantage_key, output_format='pandas')
            
        if self.twelvedata_key:
            self.twelvedata_client = TDClient(apikey=self.twelvedata_key)
            
        self.rate_limit_delay = 1.0  # seconds between API calls
        
    def get_stock_data(self, symbol: str, interval: str = 'daily', 
                      period: str = '1month') -> Optional[Dict[str, Any]]:
        """
        Get stock price data from the preferred API provider
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Data interval (daily, weekly, monthly, 1min, 5min, etc.)
            period: Data period (1month, 3month, 6month, 1year, etc.)
            
        Returns:
            Dictionary containing stock data or None if failed
        """
        try:
            if self.preferred_provider == 'alpha_vantage' and self.alpha_vantage_ts:
                return self._get_alpha_vantage_data(symbol, interval)
            elif self.preferred_provider == 'twelvedata' and self.twelvedata_client:
                return self._get_twelvedata_data(symbol, interval, period)
            else:
                # Fallback to alternative provider
                if self.alpha_vantage_ts:
                    return self._get_alpha_vantage_data(symbol, interval)
                elif self.twelvedata_client:
                    return self._get_twelvedata_data(symbol, interval, period)
                    
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
            
    def _get_alpha_vantage_data(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """Get data from Alpha Vantage API"""
        try:
            if interval == 'daily':
                data, meta_data = self.alpha_vantage_ts.get_daily_adjusted(symbol=symbol, outputsize='full')
            elif interval == 'weekly':
                data, meta_data = self.alpha_vantage_ts.get_weekly_adjusted(symbol=symbol)
            elif interval == 'monthly':
                data, meta_data = self.alpha_vantage_ts.get_monthly_adjusted(symbol=symbol)
            elif interval in ['1min', '5min', '15min', '30min', '60min']:
                data, meta_data = self.alpha_vantage_ts.get_intraday(symbol=symbol, interval=interval, outputsize='full')
            else:
                raise ValueError(f"Unsupported interval: {interval}")
                
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            return {
                'data': data,
                'meta_data': meta_data,
                'provider': 'alpha_vantage'
            }
            
        except Exception as e:
            logger.error(f"Alpha Vantage API error for {symbol}: {e}")
            return None
            
    def _get_twelvedata_data(self, symbol: str, interval: str, period: str) -> Optional[Dict[str, Any]]:
        """Get data from TwelveData API"""
        try:
            # Map interval formats
            interval_map = {
                'daily': '1day',
                'weekly': '1week',
                'monthly': '1month',
                '1min': '1min',
                '5min': '5min',
                '15min': '15min',
                '30min': '30min',
                '60min': '1h'
            }
            
            mapped_interval = interval_map.get(interval, interval)
            
            # Get time series data
            ts = self.twelvedata_client.time_series(
                symbol=symbol,
                interval=mapped_interval,
                outputsize=5000  # Maximum allowed
            )
            
            data = ts.as_pandas()
            
            time.sleep(self.rate_limit_delay)  # Rate limiting
            
            return {
                'data': data,
                'provider': 'twelvedata'
            }
            
        except Exception as e:
            logger.error(f"TwelveData API error for {symbol}: {e}")
            return None
            
    def get_technical_indicators(self, symbol: str, indicator: str, 
                               period: int = 14, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get technical indicators for a symbol
        
        Args:
            symbol: Stock symbol
            indicator: Indicator name (RSI, MACD, SMA, EMA, etc.)
            period: Period for calculation
            **kwargs: Additional parameters for the indicator
            
        Returns:
            Dictionary containing indicator data or None if failed
        """
        try:
            if self.preferred_provider == 'alpha_vantage' and self.alpha_vantage_ti:
                return self._get_alpha_vantage_indicator(symbol, indicator, period, **kwargs)
            elif self.preferred_provider == 'twelvedata' and self.twelvedata_client:
                return self._get_twelvedata_indicator(symbol, indicator, period, **kwargs)
            else:
                # Fallback
                if self.alpha_vantage_ti:
                    return self._get_alpha_vantage_indicator(symbol, indicator, period, **kwargs)
                elif self.twelvedata_client:
                    return self._get_twelvedata_indicator(symbol, indicator, period, **kwargs)
                    
        except Exception as e:
            logger.error(f"Error fetching {indicator} for {symbol}: {e}")
            return None
            
    def _get_alpha_vantage_indicator(self, symbol: str, indicator: str, 
                                   period: int, **kwargs) -> Optional[Dict[str, Any]]:
        """Get technical indicator from Alpha Vantage"""
        try:
            indicator_method = getattr(self.alpha_vantage_ti, f'get_{indicator.lower()}')
            data, meta_data = indicator_method(symbol=symbol, time_period=period, **kwargs)
            
            time.sleep(self.rate_limit_delay)
            
            return {
                'data': data,
                'meta_data': meta_data,
                'provider': 'alpha_vantage'
            }
            
        except Exception as e:
            logger.error(f"Alpha Vantage indicator error for {symbol}: {e}")
            return None
            
    def _get_twelvedata_indicator(self, symbol: str, indicator: str, 
                                period: int, **kwargs) -> Optional[Dict[str, Any]]:
        """Get technical indicator from TwelveData"""
        try:
            # Map indicator names
            indicator_map = {
                'RSI': 'rsi',
                'MACD': 'macd',
                'SMA': 'sma',
                'EMA': 'ema',
                'BBANDS': 'bbands',
                'STOCH': 'stoch'
            }
            
            mapped_indicator = indicator_map.get(indicator.upper(), indicator.lower())
            
            # Get the indicator method
            indicator_method = getattr(self.twelvedata_client, mapped_indicator)
            data = indicator_method(symbol=symbol, time_period=period, **kwargs)
            
            time.sleep(self.rate_limit_delay)
            
            return {
                'data': data.as_pandas(),
                'provider': 'twelvedata'
            }
            
        except Exception as e:
            logger.error(f"TwelveData indicator error for {symbol}: {e}")
            return None
            
    def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote for a symbol"""
        try:
            if self.preferred_provider == 'alpha_vantage' and self.alpha_vantage_key:
                return self._get_alpha_vantage_quote(symbol)
            elif self.preferred_provider == 'twelvedata' and self.twelvedata_client:
                return self._get_twelvedata_quote(symbol)
            else:
                # Fallback
                if self.alpha_vantage_key:
                    return self._get_alpha_vantage_quote(symbol)
                elif self.twelvedata_client:
                    return self._get_twelvedata_quote(symbol)
                    
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            return None
            
    def _get_alpha_vantage_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote from Alpha Vantage"""
        try:
            url = f"{TradingConfig.ALPHA_VANTAGE_BASE_URL}"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            time.sleep(self.rate_limit_delay)
            
            return {
                'data': data,
                'provider': 'alpha_vantage'
            }
            
        except Exception as e:
            logger.error(f"Alpha Vantage quote error for {symbol}: {e}")
            return None
            
    def _get_twelvedata_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time quote from TwelveData"""
        try:
            quote = self.twelvedata_client.quote(symbol=symbol)
            
            time.sleep(self.rate_limit_delay)
            
            return {
                'data': quote.as_json(),
                'provider': 'twelvedata'
            }
            
        except Exception as e:
            logger.error(f"TwelveData quote error for {symbol}: {e}")
            return None
            
    def switch_provider(self, provider: str):
        """Switch between API providers"""
        if provider in ['alpha_vantage', 'twelvedata']:
            self.preferred_provider = provider
            logger.info(f"Switched to {provider} as preferred provider")
        else:
            logger.warning(f"Unknown provider: {provider}")
            
    def get_available_providers(self) -> list:
        """Get list of available providers"""
        providers = []
        if self.alpha_vantage_key:
            providers.append('alpha_vantage')
        if self.twelvedata_key:
            providers.append('twelvedata')
        return providers