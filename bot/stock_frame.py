import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
from bot.indicators import Indicators
from utils.api_client import APIClient

logger = logging.getLogger(__name__)

class StockFrame:
    """
    Stock data frame class for storing and managing price data
    Handles data operations, indicators, and data organization
    """
    
    def __init__(self, api_client: APIClient = None):
        """
        Initialize StockFrame
        
        Args:
            api_client: API client for fetching data
        """
        self.api_client = api_client or APIClient()
        self.indicators = Indicators()
        
        # Main data storage
        self.data = {}  # Dictionary of symbol -> DataFrame
        self.symbols = set()
        
        # Data management
        self.last_updated = {}  # Track when each symbol was last updated
        self.data_intervals = {}  # Track interval for each symbol
        
        # Column mappings for different data sources
        self.column_mappings = {
            'alpha_vantage': {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adj_close',
                '6. volume': 'volume'
            },
            'twelvedata': {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            },
            'yfinance': {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
        }
        
        logger.info("StockFrame initialized")
        
    def add_symbol(self, symbol: str, interval: str = 'daily') -> bool:
        """
        Add a symbol to track
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            
        Returns:
            True if symbol added successfully
        """
        try:
            symbol = symbol.upper()
            self.symbols.add(symbol)
            
            if symbol not in self.data:
                self.data[symbol] = pd.DataFrame()
                self.data_intervals[symbol] = interval
                
            logger.info(f"Added symbol {symbol} to StockFrame")
            return True
            
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            return False
            
    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from tracking
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if symbol removed successfully
        """
        try:
            symbol = symbol.upper()
            
            if symbol in self.symbols:
                self.symbols.remove(symbol)
                
            if symbol in self.data:
                del self.data[symbol]
                
            if symbol in self.last_updated:
                del self.last_updated[symbol]
                
            if symbol in self.data_intervals:
                del self.data_intervals[symbol]
                
            logger.info(f"Removed symbol {symbol} from StockFrame")
            return True
            
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
            return False
            
    def fetch_data(self, symbol: str, interval: str = 'daily', 
                  period: str = '1month', force_refresh: bool = False) -> bool:
        """
        Fetch data for a symbol
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            period: Data period
            force_refresh: Force refresh even if data exists
            
        Returns:
            True if data fetched successfully
        """
        try:
            symbol = symbol.upper()
            
            # Check if we need to refresh data
            if not force_refresh and symbol in self.last_updated:
                last_update = self.last_updated[symbol]
                if datetime.now() - last_update < timedelta(minutes=15):  # Don't refresh too frequently
                    logger.info(f"Using cached data for {symbol}")
                    return True
                    
            # Fetch data from API
            data_response = self.api_client.get_stock_data(symbol, interval, period)
            
            if not data_response or 'data' not in data_response:
                logger.error(f"No data received for {symbol}")
                return False
                
            raw_data = data_response['data']
            provider = data_response.get('provider', 'unknown')
            
            # Convert to standardized format
            df = self._standardize_data(raw_data, provider)
            
            if df.empty:
                logger.error(f"No valid data for {symbol}")
                return False
                
            # Add symbol to tracking
            self.add_symbol(symbol, interval)
            
            # Store data
            self.data[symbol] = df
            self.last_updated[symbol] = datetime.now()
            
            logger.info(f"Fetched {len(df)} rows of data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return False
            
    def _standardize_data(self, raw_data: pd.DataFrame, provider: str) -> pd.DataFrame:
        """
        Standardize data format from different providers
        
        Args:
            raw_data: Raw data from API
            provider: Data provider name
            
        Returns:
            Standardized DataFrame
        """
        try:
            df = raw_data.copy()
            
            # Apply column mappings
            if provider in self.column_mappings:
                mapping = self.column_mappings[provider]
                df = df.rename(columns=mapping)
                
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'volume':
                        df[col] = 0  # Set volume to 0 if not available
                    else:
                        logger.warning(f"Missing required column: {col}")
                        
            # Convert to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            # Handle adjusted close
            if 'adj_close' not in df.columns:
                df['adj_close'] = df['close']
                
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # Sort by date
            df = df.sort_index()
            
            # Remove any rows with NaN values in essential columns
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error standardizing data: {e}")
            return pd.DataFrame()
            
    def append_data(self, symbol: str, new_data: pd.DataFrame) -> bool:
        """
        Append new data to existing data
        
        Args:
            symbol: Stock symbol
            new_data: New data to append
            
        Returns:
            True if data appended successfully
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data:
                self.data[symbol] = new_data
                self.add_symbol(symbol)
                logger.info(f"Added new data for {symbol}")
                return True
                
            # Combine data
            combined_data = pd.concat([self.data[symbol], new_data])
            
            # Remove duplicates
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            
            # Sort by date
            combined_data = combined_data.sort_index()
            
            self.data[symbol] = combined_data
            logger.info(f"Appended {len(new_data)} rows to {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error appending data for {symbol}: {e}")
            return False
            
    def get_data(self, symbol: str, start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get data for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with stock data
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
                
            df = self.data[symbol].copy()
            
            # Filter by date range
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
            
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest price or None if not available
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data or self.data[symbol].empty:
                return None
                
            return float(self.data[symbol]['close'].iloc[-1])
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
            
    def get_price_at_time(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """
        Get price at specific time
        
        Args:
            symbol: Stock symbol
            timestamp: Target timestamp
            
        Returns:
            Price at specified time or None if not available
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data or self.data[symbol].empty:
                return None
                
            df = self.data[symbol]
            
            # Find closest timestamp
            closest_idx = df.index.get_indexer([timestamp], method='nearest')[0]
            
            if closest_idx >= 0:
                return float(df['close'].iloc[closest_idx])
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting price at time for {symbol}: {e}")
            return None
            
    def add_indicators(self, symbol: str, force_recalculate: bool = False) -> bool:
        """
        Add technical indicators to symbol data
        
        Args:
            symbol: Stock symbol
            force_recalculate: Force recalculation of indicators
            
        Returns:
            True if indicators added successfully
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data or self.data[symbol].empty:
                logger.warning(f"No data available for {symbol}")
                return False
                
            df = self.data[symbol]
            
            # Check if indicators already exist
            indicator_columns = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'sma_50']
            has_indicators = any(col in df.columns for col in indicator_columns)
            
            if has_indicators and not force_recalculate:
                logger.info(f"Indicators already exist for {symbol}")
                return True
                
            # Add all indicators
            df_with_indicators = self.indicators.add_all_indicators(df)
            self.data[symbol] = df_with_indicators
            
            logger.info(f"Added indicators to {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding indicators for {symbol}: {e}")
            return False
            
    def get_signals(self, symbol: str) -> Dict[str, str]:
        """
        Get trading signals for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with trading signals
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data or self.data[symbol].empty:
                logger.warning(f"No data available for {symbol}")
                return {'composite': 'HOLD'}
                
            df = self.data[symbol]
            
            # Ensure indicators are calculated
            self.add_indicators(symbol)
            
            # Get signals
            signals = self.indicators.get_composite_signal(
                df['close'], df['high'], df['low'], df['volume']
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting signals for {symbol}: {e}")
            return {'composite': 'HOLD'}
            
    def calculate_returns(self, symbol: str, period: int = 1) -> pd.Series:
        """
        Calculate returns for a symbol
        
        Args:
            symbol: Stock symbol
            period: Period for return calculation
            
        Returns:
            Series of returns
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data or self.data[symbol].empty:
                logger.warning(f"No data available for {symbol}")
                return pd.Series()
                
            df = self.data[symbol]
            returns = df['close'].pct_change(periods=period)
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns for {symbol}: {e}")
            return pd.Series()
            
    def calculate_volatility(self, symbol: str, window: int = 20) -> float:
        """
        Calculate volatility for a symbol
        
        Args:
            symbol: Stock symbol
            window: Rolling window for volatility calculation
            
        Returns:
            Volatility value
        """
        try:
            symbol = symbol.upper()
            
            returns = self.calculate_returns(symbol)
            if returns.empty:
                return 0.0
                
            volatility = returns.rolling(window=window).std().iloc[-1]
            
            # Annualize volatility (assuming daily data)
            volatility_annualized = volatility * np.sqrt(252)
            
            return float(volatility_annualized) if not np.isnan(volatility_annualized) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.0
            
    def get_ohlcv(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol
        
        Args:
            symbol: Stock symbol
            periods: Number of periods to return
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data or self.data[symbol].empty:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
                
            df = self.data[symbol]
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Get the last N periods
            result = df[ohlcv_columns].tail(periods)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting OHLCV for {symbol}: {e}")
            return pd.DataFrame()
            
    def clean_data(self, symbol: str) -> bool:
        """
        Clean data for a symbol (remove outliers, handle missing values)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if data cleaned successfully
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data or self.data[symbol].empty:
                logger.warning(f"No data available for {symbol}")
                return False
                
            df = self.data[symbol]
            
            # Remove outliers using IQR method
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define outlier bounds
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    
            # Forward fill missing values
            df = df.fillna(method='ffill')
            
            # Drop any remaining NaN values
            df = df.dropna()
            
            self.data[symbol] = df
            logger.info(f"Cleaned data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data for {symbol}: {e}")
            return False
            
    def resample_data(self, symbol: str, new_frequency: str) -> bool:
        """
        Resample data to different frequency
        
        Args:
            symbol: Stock symbol
            new_frequency: New frequency ('1D', '1H', '5min', etc.)
            
        Returns:
            True if resampling successful
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data or self.data[symbol].empty:
                logger.warning(f"No data available for {symbol}")
                return False
                
            df = self.data[symbol]
            
            # Resample OHLCV data
            resampled = df.resample(new_frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Handle adjusted close
            if 'adj_close' in df.columns:
                resampled['adj_close'] = df['adj_close'].resample(new_frequency).last()
                
            # Drop NaN values
            resampled = resampled.dropna()
            
            self.data[symbol] = resampled
            logger.info(f"Resampled data for {symbol} to {new_frequency}")
            return True
            
        except Exception as e:
            logger.error(f"Error resampling data for {symbol}: {e}")
            return False
            
    def get_data_summary(self, symbol: str) -> Dict:
        """
        Get summary statistics for symbol data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            symbol = symbol.upper()
            
            if symbol not in self.data or self.data[symbol].empty:
                return {}
                
            df = self.data[symbol]
            
            summary = {
                'symbol': symbol,
                'start_date': df.index.min().strftime('%Y-%m-%d'),
                'end_date': df.index.max().strftime('%Y-%m-%d'),
                'total_rows': len(df),
                'latest_price': float(df['close'].iloc[-1]),
                'price_change': float(df['close'].iloc[-1] - df['close'].iloc[-2]) if len(df) > 1 else 0.0,
                'price_change_percent': float(((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100) if len(df) > 1 else 0.0,
                'volume': float(df['volume'].iloc[-1]),
                'avg_volume': float(df['volume'].mean()),
                'high_52w': float(df['high'].tail(252).max()) if len(df) >= 252 else float(df['high'].max()),
                'low_52w': float(df['low'].tail(252).min()) if len(df) >= 252 else float(df['low'].min()),
                'volatility': self.calculate_volatility(symbol),
                'last_updated': self.last_updated.get(symbol, datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                'has_indicators': any(col in df.columns for col in ['rsi', 'macd', 'bb_upper'])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary for {symbol}: {e}")
            return {}
            
    def get_all_symbols(self) -> List[str]:
        """
        Get all tracked symbols
        
        Returns:
            List of symbols
        """
        return list(self.symbols)
        
    def get_data_info(self) -> Dict:
        """
        Get information about all stored data
        
        Returns:
            Dictionary with data information
        """
        info = {
            'total_symbols': len(self.symbols),
            'symbols': list(self.symbols),
            'memory_usage': {},
            'last_updated': {},
            'data_intervals': self.data_intervals.copy()
        }
        
        for symbol in self.symbols:
            if symbol in self.data:
                info['memory_usage'][symbol] = f"{self.data[symbol].memory_usage(deep=True).sum() / 1024:.2f} KB"
                
            if symbol in self.last_updated:
                info['last_updated'][symbol] = self.last_updated[symbol].strftime('%Y-%m-%d %H:%M:%S')
                
        return info
        
    def clear_data(self, symbol: Optional[str] = None) -> bool:
        """
        Clear data for a symbol or all symbols
        
        Args:
            symbol: Symbol to clear (None for all)
            
        Returns:
            True if data cleared successfully
        """
        try:
            if symbol:
                symbol = symbol.upper()
                if symbol in self.data:
                    del self.data[symbol]
                if symbol in self.last_updated:
                    del self.last_updated[symbol]
                if symbol in self.symbols:
                    self.symbols.remove(symbol)
                logger.info(f"Cleared data for {symbol}")
            else:
                self.data.clear()
                self.last_updated.clear()
                self.symbols.clear()
                self.data_intervals.clear()
                logger.info("Cleared all data")
                
            return True
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            return False