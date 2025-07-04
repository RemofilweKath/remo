import numpy as np
import pandas as pd
import ta
from typing import Dict, List, Optional, Tuple, Union
import logging
from config.config import TradingConfig

logger = logging.getLogger(__name__)

class Indicators:
    """
    Technical indicators class for trading analysis
    Provides calculation methods and buy/sell signals
    """
    
    def __init__(self):
        self.config = TradingConfig()
        self.indicators_data = {}
        self.signals = {}
        
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            prices: Series of closing prices
            period: Period for calculation (default from config)
            
        Returns:
            Series of RSI values
        """
        try:
            if period is None:
                period = self.config.RSI_PERIOD
                
            rsi = ta.momentum.RSIIndicator(close=prices, window=period).rsi()
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series()
            
    def calculate_macd(self, prices: pd.Series, fast: int = None, 
                      slow: int = None, signal: int = None) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Series of closing prices
            fast: Fast period (default from config)
            slow: Slow period (default from config)
            signal: Signal period (default from config)
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        try:
            if fast is None:
                fast = self.config.MACD_FAST
            if slow is None:
                slow = self.config.MACD_SLOW
            if signal is None:
                signal = self.config.MACD_SIGNAL
                
            macd_indicator = ta.trend.MACD(close=prices, window_slow=slow, 
                                         window_fast=fast, window_sign=signal)
            
            return {
                'macd': macd_indicator.macd(),
                'signal': macd_indicator.macd_signal(),
                'histogram': macd_indicator.macd_diff()
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}
            
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = None, 
                                 std_dev: float = None) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Series of closing prices
            period: Period for calculation (default from config)
            std_dev: Standard deviation multiplier (default from config)
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        try:
            if period is None:
                period = self.config.BOLLINGER_PERIOD
            if std_dev is None:
                std_dev = self.config.BOLLINGER_STD
                
            bollinger = ta.volatility.BollingerBands(close=prices, window=period, 
                                                   window_dev=std_dev)
            
            return {
                'upper': bollinger.bollinger_hband(),
                'middle': bollinger.bollinger_mavg(),
                'lower': bollinger.bollinger_lband()
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {'upper': pd.Series(), 'middle': pd.Series(), 'lower': pd.Series()}
            
    def calculate_moving_averages(self, prices: pd.Series, 
                                periods: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Calculate Simple Moving Averages
        
        Args:
            prices: Series of closing prices
            periods: List of periods to calculate
            
        Returns:
            Dictionary with moving averages
        """
        try:
            moving_averages = {}
            
            for period in periods:
                ma = ta.trend.SMAIndicator(close=prices, window=period).sma_indicator()
                moving_averages[f'sma_{period}'] = ma
                
            return moving_averages
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {}
            
    def calculate_ema(self, prices: pd.Series, 
                     periods: List[int] = [12, 26, 50]) -> Dict[str, pd.Series]:
        """
        Calculate Exponential Moving Averages
        
        Args:
            prices: Series of closing prices
            periods: List of periods to calculate
            
        Returns:
            Dictionary with exponential moving averages
        """
        try:
            emas = {}
            
            for period in periods:
                ema = ta.trend.EMAIndicator(close=prices, window=period).ema_indicator()
                emas[f'ema_{period}'] = ema
                
            return emas
            
        except Exception as e:
            logger.error(f"Error calculating EMAs: {e}")
            return {}
            
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, 
                           close: pd.Series, k_period: int = 14, 
                           d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with %K and %D values
        """
        try:
            stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close,
                                                   window=k_period, smooth_window=d_period)
            
            return {
                'k_percent': stoch.stoch(),
                'd_percent': stoch.stoch_signal()
            }
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {'k_percent': pd.Series(), 'd_percent': pd.Series()}
            
    def calculate_atr(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Period for calculation
            
        Returns:
            Series of ATR values
        """
        try:
            atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, 
                                               window=period).average_true_range()
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series()
            
    def calculate_volume_indicators(self, close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators
        
        Args:
            close: Series of closing prices
            volume: Series of volume data
            
        Returns:
            Dictionary with volume indicators
        """
        try:
            volume_indicators = {}
            
            # Volume Moving Average
            volume_indicators['volume_sma'] = ta.volume.VolumeSMAIndicator(
                close=close, volume=volume, window=20).volume_sma()
            
            # On-Balance Volume
            volume_indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=close, volume=volume).on_balance_volume()
            
            # Volume Price Trend
            volume_indicators['vpt'] = ta.volume.VolumePriceTrendIndicator(
                close=close, volume=volume).volume_price_trend()
            
            return volume_indicators
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {}
            
    def get_rsi_signal(self, rsi_values: pd.Series) -> str:
        """
        Get trading signal based on RSI
        
        Args:
            rsi_values: Series of RSI values
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            if len(rsi_values) < 2:
                return 'HOLD'
                
            current_rsi = rsi_values.iloc[-1]
            previous_rsi = rsi_values.iloc[-2]
            
            # Oversold condition - potential buy signal
            if current_rsi < self.config.STRATEGIES['RSI_OVERSOLD']:
                return 'BUY'
            # Overbought condition - potential sell signal
            elif current_rsi > self.config.STRATEGIES['RSI_OVERBOUGHT']:
                return 'SELL'
            # RSI divergence signals
            elif previous_rsi <= self.config.STRATEGIES['RSI_OVERSOLD'] and current_rsi > self.config.STRATEGIES['RSI_OVERSOLD']:
                return 'BUY'
            elif previous_rsi >= self.config.STRATEGIES['RSI_OVERBOUGHT'] and current_rsi < self.config.STRATEGIES['RSI_OVERBOUGHT']:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error getting RSI signal: {e}")
            return 'HOLD'
            
    def get_macd_signal(self, macd_data: Dict[str, pd.Series]) -> str:
        """
        Get trading signal based on MACD
        
        Args:
            macd_data: Dictionary with MACD, signal, and histogram
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            if len(macd_data['macd']) < 2 or len(macd_data['signal']) < 2:
                return 'HOLD'
                
            current_macd = macd_data['macd'].iloc[-1]
            current_signal = macd_data['signal'].iloc[-1]
            previous_macd = macd_data['macd'].iloc[-2]
            previous_signal = macd_data['signal'].iloc[-2]
            
            # MACD crossover signals
            if previous_macd <= previous_signal and current_macd > current_signal:
                return 'BUY'  # Bullish crossover
            elif previous_macd >= previous_signal and current_macd < current_signal:
                return 'SELL'  # Bearish crossover
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error getting MACD signal: {e}")
            return 'HOLD'
            
    def get_bollinger_signal(self, prices: pd.Series, bollinger_data: Dict[str, pd.Series]) -> str:
        """
        Get trading signal based on Bollinger Bands
        
        Args:
            prices: Series of closing prices
            bollinger_data: Dictionary with upper, middle, and lower bands
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            if len(prices) < 2:
                return 'HOLD'
                
            current_price = prices.iloc[-1]
            current_upper = bollinger_data['upper'].iloc[-1]
            current_lower = bollinger_data['lower'].iloc[-1]
            current_middle = bollinger_data['middle'].iloc[-1]
            
            # Price touching lower band - potential buy signal
            if current_price <= current_lower:
                return 'BUY'
            # Price touching upper band - potential sell signal
            elif current_price >= current_upper:
                return 'SELL'
            # Price crossing middle band
            elif len(prices) > 1:
                previous_price = prices.iloc[-2]
                previous_middle = bollinger_data['middle'].iloc[-2]
                
                if previous_price <= previous_middle and current_price > current_middle:
                    return 'BUY'
                elif previous_price >= previous_middle and current_price < current_middle:
                    return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error getting Bollinger signal: {e}")
            return 'HOLD'
            
    def get_moving_average_signal(self, prices: pd.Series, ma_data: Dict[str, pd.Series]) -> str:
        """
        Get trading signal based on Moving Averages
        
        Args:
            prices: Series of closing prices
            ma_data: Dictionary with moving averages
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            if len(prices) < 2:
                return 'HOLD'
                
            current_price = prices.iloc[-1]
            
            # Golden Cross / Death Cross strategy
            if 'sma_50' in ma_data and 'sma_200' in ma_data:
                if len(ma_data['sma_50']) >= 2 and len(ma_data['sma_200']) >= 2:
                    current_sma50 = ma_data['sma_50'].iloc[-1]
                    current_sma200 = ma_data['sma_200'].iloc[-1]
                    previous_sma50 = ma_data['sma_50'].iloc[-2]
                    previous_sma200 = ma_data['sma_200'].iloc[-2]
                    
                    # Golden Cross - bullish signal
                    if previous_sma50 <= previous_sma200 and current_sma50 > current_sma200:
                        return 'BUY'
                    # Death Cross - bearish signal
                    elif previous_sma50 >= previous_sma200 and current_sma50 < current_sma200:
                        return 'SELL'
            
            # Price vs Moving Average
            if 'sma_20' in ma_data and len(ma_data['sma_20']) >= 1:
                current_sma20 = ma_data['sma_20'].iloc[-1]
                if current_price > current_sma20 * 1.02:  # 2% above MA
                    return 'BUY'
                elif current_price < current_sma20 * 0.98:  # 2% below MA
                    return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error getting Moving Average signal: {e}")
            return 'HOLD'
            
    def get_composite_signal(self, prices: pd.Series, high: pd.Series, 
                           low: pd.Series, volume: pd.Series) -> Dict[str, str]:
        """
        Get composite trading signal based on multiple indicators
        
        Args:
            prices: Series of closing prices
            high: Series of high prices
            low: Series of low prices
            volume: Series of volume data
            
        Returns:
            Dictionary with individual and composite signals
        """
        try:
            signals = {}
            
            # Calculate indicators
            rsi = self.calculate_rsi(prices)
            macd_data = self.calculate_macd(prices)
            bollinger_data = self.calculate_bollinger_bands(prices)
            ma_data = self.calculate_moving_averages(prices)
            
            # Get individual signals
            signals['rsi'] = self.get_rsi_signal(rsi)
            signals['macd'] = self.get_macd_signal(macd_data)
            signals['bollinger'] = self.get_bollinger_signal(prices, bollinger_data)
            signals['moving_average'] = self.get_moving_average_signal(prices, ma_data)
            
            # Calculate composite signal
            buy_votes = sum(1 for signal in signals.values() if signal == 'BUY')
            sell_votes = sum(1 for signal in signals.values() if signal == 'SELL')
            
            if buy_votes >= 3:
                signals['composite'] = 'BUY'
            elif sell_votes >= 3:
                signals['composite'] = 'SELL'
            elif buy_votes > sell_votes:
                signals['composite'] = 'WEAK_BUY'
            elif sell_votes > buy_votes:
                signals['composite'] = 'WEAK_SELL'
            else:
                signals['composite'] = 'HOLD'
                
            return signals
            
        except Exception as e:
            logger.error(f"Error getting composite signal: {e}")
            return {'composite': 'HOLD'}
            
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        try:
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error("DataFrame missing required OHLCV columns")
                return df
                
            # RSI
            df['rsi'] = self.calculate_rsi(df['close'])
            
            # MACD
            macd_data = self.calculate_macd(df['close'])
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bollinger_data = self.calculate_bollinger_bands(df['close'])
            df['bb_upper'] = bollinger_data['upper']
            df['bb_middle'] = bollinger_data['middle']
            df['bb_lower'] = bollinger_data['lower']
            
            # Moving Averages
            ma_data = self.calculate_moving_averages(df['close'])
            for ma_name, ma_values in ma_data.items():
                df[ma_name] = ma_values
                
            # EMAs
            ema_data = self.calculate_ema(df['close'])
            for ema_name, ema_values in ema_data.items():
                df[ema_name] = ema_values
                
            # Stochastic
            stoch_data = self.calculate_stochastic(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_data['k_percent']
            df['stoch_d'] = stoch_data['d_percent']
            
            # ATR
            df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
            
            # Volume indicators
            volume_data = self.calculate_volume_indicators(df['close'], df['volume'])
            for vol_name, vol_values in volume_data.items():
                df[vol_name] = vol_values
                
            return df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
            
    def get_signal_strength(self, signals: Dict[str, str]) -> float:
        """
        Calculate signal strength from 0 to 1
        
        Args:
            signals: Dictionary of trading signals
            
        Returns:
            Signal strength (0 = weak, 1 = strong)
        """
        try:
            signal_values = {'BUY': 1, 'WEAK_BUY': 0.5, 'HOLD': 0, 'WEAK_SELL': -0.5, 'SELL': -1}
            
            total_strength = 0
            count = 0
            
            for signal_name, signal_value in signals.items():
                if signal_name != 'composite' and signal_value in signal_values:
                    total_strength += abs(signal_values[signal_value])
                    count += 1
                    
            if count == 0:
                return 0.0
                
            return total_strength / count
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.0