#!/usr/bin/env python3
"""
Trading Bot Main Entry Point

This script demonstrates how to use the trading bot system with all its components:
1. TradingBot - Main orchestrator and API handler
2. StockFrame - Data storage and management
3. Portfolio - Portfolio and position management  
4. Trade - Order management
5. Indicators - Technical analysis

Usage:
    python main.py

The bot will:
- Fetch market data from Alpha Vantage/TwelveData APIs
- Analyze stocks using technical indicators
- Perform news sentiment analysis
- Generate trading signals
- Execute trades based on strategies
- Track portfolio performance
- Provide risk management
"""

import os
import sys
import logging
import signal
import time
from datetime import datetime
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import bot components
from bot.trading_bot import TradingBot
from bot.trade import Trade, OrderSide, OrderType
from config.config import TradingConfig

class TradingBotRunner:
    """
    Main runner class for the trading bot
    """
    
    def __init__(self):
        self.bot = None
        self.running = False
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.shutdown()
        
    def on_trade_signal(self, symbol: str, signal: str, confidence: float):
        """Callback for trade signals"""
        logger.info(f"Trade signal: {symbol} - {signal} (confidence: {confidence:.2f})")
        
    def on_order_filled(self, trade: Trade):
        """Callback for order fills"""
        logger.info(f"Order filled: {trade.symbol} - {trade.side.value} {trade.filled_quantity} @ ${trade.average_fill_price:.2f}")
        
        # Log portfolio metrics
        if self.bot:
            metrics = self.bot.portfolio.get_portfolio_metrics(self.bot.current_prices)
            logger.info(f"Portfolio value: ${metrics.get('portfolio_value', 0):,.2f}")
            logger.info(f"Total return: {metrics.get('total_return', 0):.2f}%")
            
    def on_error(self, error: Exception):
        """Callback for errors"""
        logger.error(f"Bot error: {error}")
        
    def run(self):
        """Run the trading bot"""
        try:
            logger.info("Starting Trading Bot System")
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Create and configure bot
            self.bot = TradingBot(name="AlgoTrader Pro", config=TradingConfig())
            
            # Set callbacks
            self.bot.set_callbacks(
                on_trade_signal=self.on_trade_signal,
                on_order_filled=self.on_order_filled,
                on_error=self.on_error
            )
            
            # Add symbols to watchlist
            symbols = [
                'AAPL',  # Apple
                'GOOGL', # Alphabet
                'MSFT',  # Microsoft
                'AMZN',  # Amazon
                'TSLA',  # Tesla
                'META',  # Meta
                'NVDA',  # NVIDIA
                'NFLX',  # Netflix
                'AMD',   # AMD
                'INTC'   # Intel
            ]
            
            logger.info(f"Adding {len(symbols)} symbols to watchlist: {symbols}")
            self.bot.add_to_watchlist(symbols)
            
            # Start the bot
            if self.bot.start():
                self.running = True
                logger.info("Trading bot started successfully!")
                
                # Print initial status
                self.print_status()
                
                # Keep the main thread alive and periodically print status
                while self.running:
                    time.sleep(300)  # Wait 5 minutes
                    
                    if self.bot.is_running:
                        self.print_status()
                        self.print_performance()
                    else:
                        logger.warning("Bot stopped unexpectedly")
                        break
                        
            else:
                logger.error("Failed to start trading bot")
                return False
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self.shutdown()
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            self.shutdown()
            return False
            
        return True
        
    def print_status(self):
        """Print bot status"""
        try:
            if not self.bot:
                return
                
            status = self.bot.get_status()
            
            print("\n" + "="*80)
            print(f"TRADING BOT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            print(f"Bot Name: {status.get('name')}")
            print(f"Running: {status.get('is_running')}")
            print(f"Trading Hours: {status.get('is_trading_hours')}")
            print(f"Uptime: {status.get('uptime')}")
            print(f"Watchlist: {status.get('watchlist_size')} symbols")
            print(f"Active Orders: {status.get('active_orders')}")
            print(f"Daily Trades: {status.get('daily_trade_count')}")
            
            # Portfolio metrics
            portfolio_metrics = status.get('portfolio_metrics', {})
            print(f"\nPortfolio Value: ${portfolio_metrics.get('portfolio_value', 0):,.2f}")
            print(f"Cash: ${portfolio_metrics.get('cash', 0):,.2f}")
            print(f"Total Return: {portfolio_metrics.get('total_return', 0):.2f}%")
            print(f"Active Positions: {portfolio_metrics.get('active_positions', 0)}")
            
            # Current prices
            current_prices = status.get('current_prices', {})
            if current_prices:
                print(f"\nCurrent Prices:")
                for symbol, price in list(current_prices.items())[:5]:  # Show first 5
                    print(f"  {symbol}: ${price:.2f}")
                if len(current_prices) > 5:
                    print(f"  ... and {len(current_prices) - 5} more")
                    
        except Exception as e:
            logger.error(f"Error printing status: {e}")
            
    def print_performance(self):
        """Print performance report"""
        try:
            if not self.bot:
                return
                
            report = self.bot.get_performance_report()
            
            print("\n" + "-"*60)
            print("PERFORMANCE REPORT")
            print("-"*60)
            
            portfolio_metrics = report.get('portfolio_metrics', {})
            
            # Performance metrics
            print(f"Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {portfolio_metrics.get('max_drawdown', 0):.2f}%")
            print(f"Win Rate: {portfolio_metrics.get('win_rate', 0):.2f}%")
            print(f"Profit Factor: {portfolio_metrics.get('profit_factor', 0):.2f}")
            print(f"Total Trades: {portfolio_metrics.get('total_trades', 0)}")
            
            # Recent positions
            positions = report.get('positions', [])
            if positions:
                print(f"\nActive Positions ({len(positions)}):")
                for pos in positions[:3]:  # Show first 3
                    print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['average_price']:.2f} "
                          f"(P&L: ${pos['unrealized_pnl']:.2f})")
                    
            # Recent trades
            recent_trades = report.get('recent_trades', [])
            if recent_trades:
                print(f"\nRecent Trades ({len(recent_trades)}):")
                for trade in recent_trades[-3:]:  # Show last 3
                    print(f"  {trade['symbol']}: {trade['side']} {trade['filled_quantity']} @ ${trade['average_price']:.2f}")
                    
        except Exception as e:
            logger.error(f"Error printing performance: {e}")
            
    def shutdown(self):
        """Shutdown the bot gracefully"""
        try:
            logger.info("Shutting down trading bot...")
            self.running = False
            
            if self.bot:
                self.bot.stop()
                
                # Print final performance report
                print("\n" + "="*80)
                print("FINAL PERFORMANCE REPORT")
                print("="*80)
                self.print_performance()
                
                # Export portfolio data
                portfolio_data = self.bot.portfolio.export_to_dict()
                logger.info(f"Final portfolio data: {len(portfolio_data.get('trades', []))} trades")
                
            logger.info("Trading bot shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("ALGORITHMIC TRADING BOT")
    print("="*80)
    print("A comprehensive trading system with:")
    print("• Multi-API data integration (Alpha Vantage, TwelveData)")
    print("• Advanced technical analysis (RSI, MACD, Bollinger Bands, etc.)")
    print("• News sentiment analysis")
    print("• Risk management and position sizing")
    print("• Portfolio tracking and performance metrics")
    print("• Paper trading simulation")
    print("="*80)
    
    # Check for API keys
    config = TradingConfig()
    
    print("\nChecking configuration...")
    if not config.ALPHA_VANTAGE_API_KEY and not config.TWELVEDATA_API_KEY:
        print("⚠️  WARNING: No API keys found!")
        print("Please set your API keys in a .env file:")
        print("ALPHA_VANTAGE_API_KEY=your_key_here")
        print("TWELVEDATA_API_KEY=your_key_here")
        print("NEWS_API_KEY=your_key_here")
        print("\nThe bot will run in demo mode with limited functionality.")
        
    print(f"📊 Paper Trading: {'Enabled' if config.PAPER_TRADING else 'Disabled'}")
    print(f"💰 Initial Capital: ${config.INITIAL_CAPITAL:,}")
    print(f"📈 Risk Per Trade: {config.RISK_PERCENTAGE*100}%")
    print(f"🕒 Trading Hours: {config.TRADING_START_HOUR}:00 - {config.TRADING_END_HOUR}:00")
    
    # Start the bot
    runner = TradingBotRunner()
    
    try:
        success = runner.run()
        if success:
            logger.info("Trading bot completed successfully")
        else:
            logger.error("Trading bot failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()