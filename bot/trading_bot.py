import logging
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np

from bot.stock_frame import StockFrame
from bot.portfolio import Portfolio
from bot.trade import Trade, OrderType, OrderSide, OrderStatus
from bot.indicators import Indicators
from utils.api_client import APIClient
from utils.news_sentiment import NewsSentimentAnalyzer
from config.config import TradingConfig

logger = logging.getLogger(__name__)

class TradingBot:
    """
    Main trading bot class - highest level of hierarchy
    Handles API interactions, orchestrates trading operations, and manages the overall system
    """
    
    def __init__(self, name: str = "TradingBot", config: TradingConfig = None):
        """
        Initialize the trading bot
        
        Args:
            name: Name of the bot
            config: Configuration object
        """
        self.name = name
        self.config = config or TradingConfig()
        self.is_running = False
        self.is_trading_hours = False
        
        # Initialize core components
        self.api_client = APIClient()
        self.stock_frame = StockFrame(self.api_client)
        self.portfolio = Portfolio(self.config.INITIAL_CAPITAL, f"{name} Portfolio")
        self.indicators = Indicators()
        self.news_analyzer = NewsSentimentAnalyzer()
        
        # Trading state
        self.watchlist = set()
        self.active_orders = {}  # trade_id -> Trade
        self.current_prices = {}
        self.signals = {}
        self.news_sentiment = {}
        
        # Trading strategies
        self.strategies = []
        self.strategy_weights = {}
        
        # Performance tracking
        self.start_time = None
        self.last_update = None
        self.update_interval = 60  # seconds
        
        # Threading
        self.main_thread = None
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.on_trade_signal = None
        self.on_order_filled = None
        self.on_error = None
        
        # Risk management
        self.max_daily_trades = 20
        self.max_position_size = 0.2  # 20% of portfolio
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        logger.info(f"TradingBot '{name}' initialized")
        
    def add_to_watchlist(self, symbols: Union[str, List[str]]) -> bool:
        """
        Add symbols to watchlist
        
        Args:
            symbols: Symbol or list of symbols to add
            
        Returns:
            True if successful
        """
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
                
            for symbol in symbols:
                symbol = symbol.upper()
                self.watchlist.add(symbol)
                self.stock_frame.add_symbol(symbol)
                
            logger.info(f"Added {len(symbols)} symbols to watchlist: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding symbols to watchlist: {e}")
            return False
            
    def remove_from_watchlist(self, symbols: Union[str, List[str]]) -> bool:
        """
        Remove symbols from watchlist
        
        Args:
            symbols: Symbol or list of symbols to remove
            
        Returns:
            True if successful
        """
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
                
            for symbol in symbols:
                symbol = symbol.upper()
                if symbol in self.watchlist:
                    self.watchlist.remove(symbol)
                    self.stock_frame.remove_symbol(symbol)
                    
            logger.info(f"Removed {len(symbols)} symbols from watchlist: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing symbols from watchlist: {e}")
            return False
            
    def fetch_market_data(self, force_refresh: bool = False) -> bool:
        """
        Fetch market data for all watchlist symbols
        
        Args:
            force_refresh: Force refresh of data
            
        Returns:
            True if successful
        """
        try:
            logger.info("Fetching market data...")
            
            for symbol in self.watchlist:
                success = self.stock_frame.fetch_data(symbol, force_refresh=force_refresh)
                if success:
                    # Update current prices
                    latest_price = self.stock_frame.get_latest_price(symbol)
                    if latest_price:
                        self.current_prices[symbol] = latest_price
                        
                    # Add indicators
                    self.stock_frame.add_indicators(symbol)
                    
                    # Get signals
                    signals = self.stock_frame.get_signals(symbol)
                    self.signals[symbol] = signals
                    
                    # Get news sentiment
                    sentiment = self.news_analyzer.get_news_sentiment(symbol)
                    self.news_sentiment[symbol] = sentiment
                    
            self.last_update = datetime.now()
            logger.info(f"Market data updated for {len(self.watchlist)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return False
            
    def analyze_market(self) -> Dict[str, Dict]:
        """
        Analyze market conditions and generate trading signals
        
        Returns:
            Dictionary of analysis results
        """
        try:
            analysis = {}
            
            for symbol in self.watchlist:
                if symbol in self.current_prices:
                    symbol_analysis = {
                        'symbol': symbol,
                        'current_price': self.current_prices[symbol],
                        'signals': self.signals.get(symbol, {}),
                        'sentiment': self.news_sentiment.get(symbol, {}),
                        'recommendation': 'HOLD',
                        'confidence': 0.0,
                        'reasons': []
                    }
                    
                    # Analyze technical signals
                    signals = self.signals.get(symbol, {})
                    composite_signal = signals.get('composite', 'HOLD')
                    
                    # Analyze sentiment
                    sentiment = self.news_sentiment.get(symbol, {})
                    sentiment_signal = self.news_analyzer.get_sentiment_signal(symbol)
                    
                    # Combine signals
                    buy_votes = 0
                    sell_votes = 0
                    
                    if composite_signal in ['BUY', 'WEAK_BUY']:
                        buy_votes += 2 if composite_signal == 'BUY' else 1
                        symbol_analysis['reasons'].append(f"Technical: {composite_signal}")
                        
                    if composite_signal in ['SELL', 'WEAK_SELL']:
                        sell_votes += 2 if composite_signal == 'SELL' else 1
                        symbol_analysis['reasons'].append(f"Technical: {composite_signal}")
                        
                    if sentiment_signal == 'BUY':
                        buy_votes += 1
                        symbol_analysis['reasons'].append("Positive sentiment")
                    elif sentiment_signal == 'SELL':
                        sell_votes += 1
                        symbol_analysis['reasons'].append("Negative sentiment")
                        
                    # Determine final recommendation
                    if buy_votes > sell_votes and buy_votes >= 2:
                        symbol_analysis['recommendation'] = 'BUY'
                        symbol_analysis['confidence'] = min(buy_votes / 4.0, 1.0)
                    elif sell_votes > buy_votes and sell_votes >= 2:
                        symbol_analysis['recommendation'] = 'SELL'
                        symbol_analysis['confidence'] = min(sell_votes / 4.0, 1.0)
                    else:
                        symbol_analysis['recommendation'] = 'HOLD'
                        symbol_analysis['confidence'] = 0.1
                        
                    analysis[symbol] = symbol_analysis
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return {}
            
    def create_trade(self, symbol: str, side: OrderSide, order_type: OrderType, 
                    quantity: float, price: Optional[float] = None, 
                    stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Optional[Trade]:
        """
        Create a new trade
        
        Args:
            symbol: Stock symbol
            side: Order side (BUY/SELL)
            order_type: Order type
            quantity: Quantity to trade
            price: Limit price (if applicable)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Trade object or None if failed
        """
        try:
            # Risk checks
            if not self.can_trade():
                logger.warning("Cannot trade at this time")
                return None
                
            if not self.risk_check_passed(symbol, side, quantity):
                logger.warning(f"Risk check failed for {symbol}")
                return None
                
            # Calculate position size based on risk
            current_price = self.current_prices.get(symbol)
            if current_price and not price:
                price = current_price
                
            # Create trade
            trade = Trade(symbol, side, order_type, quantity, price)
            
            # Add risk management
            if stop_loss:
                trade.add_stop_loss(stop_loss)
            if take_profit:
                trade.add_take_profit(take_profit)
                
            # Add metadata
            trade.add_metadata('created_by', self.name)
            trade.add_metadata('strategy', 'multi_signal')
            
            logger.info(f"Created trade: {trade}")
            return trade
            
        except Exception as e:
            logger.error(f"Error creating trade: {e}")
            return None
            
    def submit_trade(self, trade: Trade) -> bool:
        """
        Submit a trade for execution
        
        Args:
            trade: Trade to submit
            
        Returns:
            True if successful
        """
        try:
            # Validate trade
            if not trade.validate_order():
                logger.error(f"Trade validation failed: {trade.trade_id}")
                return False
                
            # Submit order
            if trade.submit_order():
                self.active_orders[trade.trade_id] = trade
                
                # Simulate order filling for paper trading
                if self.config.PAPER_TRADING:
                    self._simulate_order_fill(trade)
                    
                logger.info(f"Submitted trade: {trade.trade_id}")
                return True
            else:
                logger.error(f"Failed to submit trade: {trade.trade_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting trade: {e}")
            return False
            
    def _simulate_order_fill(self, trade: Trade):
        """Simulate order filling for paper trading"""
        try:
            # Get current price
            current_price = self.current_prices.get(trade.symbol)
            if not current_price:
                return
                
            # Simulate fill based on order type
            fill_price = current_price
            
            if trade.order_type == OrderType.MARKET:
                # Market order fills immediately
                fill_price = current_price
            elif trade.order_type == OrderType.LIMIT:
                # Limit order fills if price is favorable
                if trade.side == OrderSide.BUY and current_price <= trade.price:
                    fill_price = trade.price
                elif trade.side == OrderSide.SELL and current_price >= trade.price:
                    fill_price = trade.price
                else:
                    return  # Order not filled
                    
            # Add fill
            commission = 0.0  # No commission for paper trading
            if trade.add_fill(fill_price, trade.quantity, commission):
                # Add to portfolio
                self.portfolio.add_trade(trade)
                
                # Update daily trade count
                today = datetime.now().date()
                if self.last_trade_date != today:
                    self.daily_trade_count = 0
                    self.last_trade_date = today
                self.daily_trade_count += 1
                
                # Remove from active orders
                if trade.trade_id in self.active_orders:
                    del self.active_orders[trade.trade_id]
                    
                # Trigger callback
                if self.on_order_filled:
                    self.on_order_filled(trade)
                    
                logger.info(f"Trade filled: {trade.trade_id} at ${fill_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error simulating order fill: {e}")
            
    def execute_strategy(self) -> bool:
        """
        Execute trading strategy
        
        Returns:
            True if successful
        """
        try:
            if not self.can_trade():
                return False
                
            # Analyze market
            analysis = self.analyze_market()
            
            # Generate trades based on analysis
            trades_created = 0
            for symbol, analysis_data in analysis.items():
                recommendation = analysis_data['recommendation']
                confidence = analysis_data['confidence']
                
                if recommendation in ['BUY', 'SELL'] and confidence > 0.7:
                    # Check if we already have a position
                    existing_position = self.portfolio.get_position(symbol)
                    
                    if recommendation == 'BUY' and (not existing_position or existing_position.side == OrderSide.SELL):
                        # Create buy order
                        quantity = self.calculate_position_size(symbol, OrderSide.BUY)
                        if quantity > 0:
                            trade = self.create_trade(symbol, OrderSide.BUY, OrderType.MARKET, quantity)
                            if trade and self.submit_trade(trade):
                                trades_created += 1
                                
                    elif recommendation == 'SELL' and (not existing_position or existing_position.side == OrderSide.BUY):
                        # Create sell order
                        if existing_position:
                            quantity = abs(existing_position.quantity)
                        else:
                            quantity = self.calculate_position_size(symbol, OrderSide.SELL)
                            
                        if quantity > 0:
                            trade = self.create_trade(symbol, OrderSide.SELL, OrderType.MARKET, quantity)
                            if trade and self.submit_trade(trade):
                                trades_created += 1
                                
            logger.info(f"Created {trades_created} trades")
            return True
            
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return False
            
    def calculate_position_size(self, symbol: str, side: OrderSide) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            symbol: Stock symbol
            side: Order side
            
        Returns:
            Position size in shares
        """
        try:
            current_price = self.current_prices.get(symbol)
            if not current_price:
                return 0.0
                
            # Calculate maximum position value
            portfolio_value = self.portfolio.get_portfolio_value(self.current_prices)
            max_position_value = portfolio_value * self.max_position_size
            
            # Calculate risk-based position size
            risk_amount = portfolio_value * self.config.RISK_PERCENTAGE
            
            # Use ATR for stop loss calculation
            df = self.stock_frame.get_data(symbol)
            if not df.empty and 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                stop_loss_distance = atr * 2  # 2x ATR stop loss
                
                if stop_loss_distance > 0:
                    risk_based_size = risk_amount / stop_loss_distance
                    max_size = max_position_value / current_price
                    
                    position_size = min(risk_based_size, max_size)
                    return max(0, int(position_size))
                    
            # Fallback to fixed percentage
            position_size = max_position_value / current_price
            return max(0, int(position_size))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
            
    def risk_check_passed(self, symbol: str, side: OrderSide, quantity: float) -> bool:
        """
        Check if trade passes risk management rules
        
        Args:
            symbol: Stock symbol
            side: Order side
            quantity: Quantity to trade
            
        Returns:
            True if risk check passes
        """
        try:
            # Check daily trade limit
            today = datetime.now().date()
            if self.last_trade_date == today and self.daily_trade_count >= self.max_daily_trades:
                logger.warning("Daily trade limit reached")
                return False
                
            # Check position size limit
            current_price = self.current_prices.get(symbol)
            if current_price:
                position_value = current_price * quantity
                portfolio_value = self.portfolio.get_portfolio_value(self.current_prices)
                
                if position_value > portfolio_value * self.max_position_size:
                    logger.warning(f"Position size too large: {position_value} > {portfolio_value * self.max_position_size}")
                    return False
                    
            # Check cash availability for buys
            if side == OrderSide.BUY:
                if current_price:
                    required_cash = current_price * quantity
                    if required_cash > self.portfolio.cash:
                        logger.warning(f"Insufficient cash: {required_cash} > {self.portfolio.cash}")
                        return False
                        
            # Check portfolio risk
            risk_check = self.portfolio.risk_check(self.current_prices)
            if risk_check['warnings']:
                logger.warning(f"Portfolio risk warnings: {risk_check['warnings']}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return False
            
    def can_trade(self) -> bool:
        """
        Check if trading is allowed
        
        Returns:
            True if trading is allowed
        """
        try:
            # Check if bot is running
            if not self.is_running:
                return False
                
            # Check trading hours
            if not self.is_trading_hours:
                return False
                
            # Check if we have current data
            if not self.current_prices:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking if can trade: {e}")
            return False
            
    def update_trading_hours(self):
        """Update trading hours status"""
        try:
            now = datetime.now()
            current_hour = now.hour
            
            # Simple trading hours check (9 AM to 4 PM)
            if self.config.TRADING_START_HOUR <= current_hour < self.config.TRADING_END_HOUR:
                if now.weekday() < 5:  # Monday to Friday
                    self.is_trading_hours = True
                else:
                    self.is_trading_hours = False
            else:
                self.is_trading_hours = False
                
        except Exception as e:
            logger.error(f"Error updating trading hours: {e}")
            
    def start(self) -> bool:
        """
        Start the trading bot
        
        Returns:
            True if started successfully
        """
        try:
            if self.is_running:
                logger.warning("Bot is already running")
                return False
                
            # Validate configuration
            self.config.validate_config()
            
            # Initialize watchlist with some default symbols if empty
            if not self.watchlist:
                default_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
                self.add_to_watchlist(default_symbols)
                
            # Fetch initial data
            self.fetch_market_data(force_refresh=True)
            
            # Start scheduler
            self.setup_scheduler()
            
            # Start main loop
            self.is_running = True
            self.start_time = datetime.now()
            
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            logger.info(f"TradingBot '{self.name}' started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            return False
            
    def stop(self):
        """Stop the trading bot"""
        try:
            self.is_running = False
            self.stop_event.set()
            
            # Wait for threads to finish
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=5)
                
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
                
            logger.info(f"TradingBot '{self.name}' stopped")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            
    def setup_scheduler(self):
        """Setup scheduled tasks"""
        try:
            # Schedule market data updates
            schedule.every(1).minutes.do(self.fetch_market_data)
            
            # Schedule strategy execution
            schedule.every(5).minutes.do(self.execute_strategy)
            
            # Schedule portfolio updates
            schedule.every(15).minutes.do(self.update_portfolio)
            
            # Schedule trading hours check
            schedule.every(1).minutes.do(self.update_trading_hours)
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
        except Exception as e:
            logger.error(f"Error setting up scheduler: {e}")
            
    def _run_scheduler(self):
        """Run scheduled tasks"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")
                
    def _main_loop(self):
        """Main bot loop"""
        logger.info("Starting main bot loop")
        
        while self.is_running:
            try:
                # Update trading hours
                self.update_trading_hours()
                
                # Update portfolio with current prices
                self.portfolio.update_positions(self.current_prices)
                
                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                if self.on_error:
                    self.on_error(e)
                time.sleep(60)  # Wait before retrying
                
    def update_portfolio(self):
        """Update portfolio with current prices"""
        try:
            self.portfolio.update_positions(self.current_prices)
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            
    def get_status(self) -> Dict:
        """
        Get bot status
        
        Returns:
            Dictionary with bot status
        """
        try:
            portfolio_metrics = self.portfolio.get_portfolio_metrics(self.current_prices)
            
            status = {
                'name': self.name,
                'is_running': self.is_running,
                'is_trading_hours': self.is_trading_hours,
                'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None,
                'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else None,
                'watchlist_size': len(self.watchlist),
                'watchlist': list(self.watchlist),
                'active_orders': len(self.active_orders),
                'current_prices': self.current_prices,
                'portfolio_metrics': portfolio_metrics,
                'daily_trade_count': self.daily_trade_count,
                'uptime': str(datetime.now() - self.start_time) if self.start_time else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}
            
    def get_performance_report(self) -> Dict:
        """
        Get performance report
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            portfolio_metrics = self.portfolio.get_portfolio_metrics(self.current_prices)
            
            # Get positions summary
            positions_df = self.portfolio.get_positions_summary(self.current_prices)
            
            # Get trades summary
            trades_df = self.portfolio.get_trades_summary()
            
            report = {
                'bot_name': self.name,
                'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'portfolio_metrics': portfolio_metrics,
                'positions': positions_df.to_dict('records') if not positions_df.empty else [],
                'recent_trades': trades_df.tail(10).to_dict('records') if not trades_df.empty else [],
                'watchlist': list(self.watchlist),
                'signals': self.signals,
                'news_sentiment': self.news_sentiment
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
            
    def set_callbacks(self, on_trade_signal: Callable = None, 
                     on_order_filled: Callable = None, on_error: Callable = None):
        """
        Set callback functions
        
        Args:
            on_trade_signal: Callback for trade signals
            on_order_filled: Callback for order fills
            on_error: Callback for errors
        """
        self.on_trade_signal = on_trade_signal
        self.on_order_filled = on_order_filled
        self.on_error = on_error
        
    def __str__(self) -> str:
        """String representation of the bot"""
        return f"TradingBot(name={self.name}, running={self.is_running}, watchlist={len(self.watchlist)})"
        
    def __repr__(self) -> str:
        """Detailed string representation of the bot"""
        return f"TradingBot(name={self.name}, running={self.is_running}, watchlist={len(self.watchlist)}, portfolio_value={self.portfolio.cash})"