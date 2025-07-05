# Algorithmic Trading Bot

A comprehensive Python-based algorithmic trading bot that integrates multiple data sources, advanced technical analysis, news sentiment analysis, and risk management for automated trading decisions.

## 🏗️ Architecture

The trading bot follows a modular architecture with 5 core components as specified:

### 1. **TradingBot** (Main Orchestrator)
- **Highest level of hierarchy**
- Handles API interactions with Alpha Vantage and TwelveData
- Orchestrates all trading operations
- Manages threading and scheduling
- Provides real-time monitoring and callbacks

### 2. **StockFrame** (Data Management)
- Stores all price data in pandas DataFrames
- Handles data fetching, cleaning, and organization
- Manages data from multiple sources (Alpha Vantage, TwelveData)
- Provides data querying and manipulation capabilities
- Handles indicator calculations and signal generation

### 3. **Portfolio** (Portfolio Management)
- Represents a trading portfolio with multiple positions
- Calculates comprehensive portfolio metrics (Sharpe ratio, max drawdown, etc.)
- Tracks performance, P&L, and risk metrics
- Handles position sizing and risk management
- Provides detailed portfolio analytics

### 4. **Trade** (Order Management)
- Represents individual orders to be placed
- Handles order modifications (price, quantity, type)
- Manages order lifecycle (pending → submitted → filled)
- Supports complex order types (market, limit, stop, trailing stop)
- Includes risk management features (stop loss, take profit)

### 5. **Indicators** (Technical Analysis)
- Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Provides buy/sell signal generation
- Implements composite signal methodology
- Supports custom indicator combinations
- Includes signal strength calculations

## ✨ Features

### 🔌 **Multi-API Integration**
- **Alpha Vantage API**: Real-time and historical stock data
- **TwelveData API**: Alternative data source with high-frequency data
- **News API**: Financial news for sentiment analysis
- Automatic failover between data providers

### 📊 **Advanced Technical Analysis**
- **Moving Averages**: SMA, EMA with golden/death cross detection
- **Momentum Indicators**: RSI, Stochastic Oscillator
- **Trend Indicators**: MACD with signal line crossovers
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, Volume Price Trend
- **Composite Signals**: Multi-indicator consensus

### 📰 **News Sentiment Analysis**
- Real-time news sentiment analysis using TextBlob
- Multi-source news aggregation
- Sentiment-based trading signals
- Market sentiment tracking

### 🛡️ **Risk Management**
- **Position Sizing**: Risk-based position calculation
- **Stop Loss/Take Profit**: Automatic risk management
- **Portfolio Risk Limits**: Maximum position sizes and concentrations
- **Daily Trade Limits**: Prevents overtrading
- **Cash Management**: Maintains adequate liquidity

### 📈 **Performance Tracking**
- **Real-time P&L**: Unrealized and realized profits/losses
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate
- **Portfolio Analytics**: Position summaries, trade history
- **Risk Analytics**: Concentration analysis, leverage monitoring

### 🤖 **Automation Features**
- **Scheduled Execution**: Automated data updates and strategy execution
- **Paper Trading**: Safe testing environment
- **Multi-threading**: Concurrent operations for efficiency
- **Graceful Shutdown**: Clean bot termination with data preservation

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd trading-bot
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup API Keys** (Create a `.env` file):
```bash
# Required API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
TWELVEDATA_API_KEY=your_twelvedata_key
NEWS_API_KEY=your_news_api_key

# Trading Configuration
PAPER_TRADING=True
INITIAL_CAPITAL=100000
RISK_PERCENTAGE=0.02
MAX_POSITIONS=10

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN=your_telegram_token
```

4. **Run the bot**:
```bash
python main.py
```

### Getting API Keys

- **Alpha Vantage**: Get free API key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)
- **TwelveData**: Get free API key at [twelvedata.com](https://twelvedata.com/pricing)
- **News API**: Get free API key at [newsapi.org](https://newsapi.org/register)

## 📋 Usage Examples

### Basic Bot Setup

```python
from bot.trading_bot import TradingBot
from config.config import TradingConfig

# Create and configure bot
bot = TradingBot(name="MyBot", config=TradingConfig())

# Add symbols to watchlist
bot.add_to_watchlist(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'])

# Set up callbacks
def on_trade_filled(trade):
    print(f"Trade filled: {trade.symbol} - {trade.side.value}")

bot.set_callbacks(on_order_filled=on_trade_filled)

# Start the bot
bot.start()
```

### Manual Trading

```python
from bot.trade import Trade, OrderSide, OrderType

# Create a trade
trade = bot.create_trade(
    symbol='AAPL',
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=100,
    stop_loss=150.0,
    take_profit=200.0
)

# Submit the trade
if trade:
    bot.submit_trade(trade)
```

### Portfolio Analysis

```python
# Get portfolio metrics
metrics = bot.portfolio.get_portfolio_metrics(bot.current_prices)
print(f"Portfolio Value: ${metrics['portfolio_value']:,.2f}")
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Get positions summary
positions = bot.portfolio.get_positions_summary(bot.current_prices)
print(positions)
```

### Technical Analysis

```python
# Get signals for a symbol
signals = bot.stock_frame.get_signals('AAPL')
print(f"Composite Signal: {signals['composite']}")
print(f"RSI Signal: {signals['rsi']}")
print(f"MACD Signal: {signals['macd']}")

# Get raw indicator data
data = bot.stock_frame.get_data('AAPL')
print(f"Current RSI: {data['rsi'].iloc[-1]:.2f}")
print(f"Current MACD: {data['macd'].iloc[-1]:.4f}")
```

## ⚙️ Configuration

The bot is highly configurable through environment variables and the `TradingConfig` class:

### Trading Parameters
- `INITIAL_CAPITAL`: Starting capital amount (default: $100,000)
- `RISK_PERCENTAGE`: Risk per trade as percentage of portfolio (default: 2%)
- `MAX_POSITIONS`: Maximum number of concurrent positions (default: 10)
- `PAPER_TRADING`: Enable/disable paper trading mode (default: True)

### API Configuration
- `ALPHA_VANTAGE_API_KEY`: Alpha Vantage API key
- `TWELVEDATA_API_KEY`: TwelveData API key
- `NEWS_API_KEY`: News API key for sentiment analysis

### Trading Hours
- `TRADING_START_HOUR`: Start of trading day (default: 9 AM)
- `TRADING_END_HOUR`: End of trading day (default: 4 PM)

### Technical Indicators
- `RSI_PERIOD`: RSI calculation period (default: 14)
- `MACD_FAST`: MACD fast period (default: 12)
- `MACD_SLOW`: MACD slow period (default: 26)
- `BOLLINGER_PERIOD`: Bollinger Bands period (default: 20)

## 🧪 Testing

The bot includes comprehensive testing capabilities:

### Paper Trading Mode
- Safe testing environment with simulated trades
- No real money at risk
- Full feature functionality
- Performance tracking and analysis

### Backtesting Support
```python
# Example backtesting setup
from datetime import datetime, timedelta

start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()

# Fetch historical data
bot.stock_frame.fetch_data('AAPL', period='1year', force_refresh=True)

# Run strategy on historical data
# (Implementation depends on specific strategy)
```

## 📊 Performance Monitoring

### Real-time Status
The bot provides comprehensive real-time monitoring:

```bash
================================================================================
TRADING BOT STATUS - 2024-01-15 14:30:00
================================================================================
Bot Name: AlgoTrader Pro
Running: True
Trading Hours: True
Uptime: 2:15:30
Watchlist: 10 symbols
Active Orders: 2
Daily Trades: 5

Portfolio Value: $105,230.50
Cash: $45,230.50
Total Return: 5.23%
Active Positions: 3
```

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest portfolio decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Portfolio Composition**: Position sizes and allocations

## 🔧 Advanced Features

### Custom Strategies
Extend the bot with custom trading strategies:

```python
class CustomStrategy:
    def analyze(self, data):
        # Custom analysis logic
        return signal
    
    def execute(self, signal):
        # Custom execution logic
        pass

# Register custom strategy
bot.add_strategy(CustomStrategy())
```

### Webhook Integration
Set up webhooks for external signals:

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    signal_data = request.json
    # Process external signal
    bot.process_external_signal(signal_data)
    return 'OK'
```

### Database Integration
Store trading data in a database:

```python
import sqlite3

# Example database setup
conn = sqlite3.connect('trading_data.db')
bot.setup_database(conn)
```

## 🚨 Risk Warnings

⚠️ **Important**: This is educational software for learning algorithmic trading concepts.

- **Use paper trading mode** for testing and learning
- **Never risk money you cannot afford to lose**
- **Past performance does not guarantee future results**
- **Market conditions can change rapidly**
- **Always implement proper risk management**

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 .

# Run type checking
mypy .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Alpha Vantage** for providing financial data APIs
- **TwelveData** for high-quality market data
- **NewsAPI** for news data and sentiment analysis
- **TA-Lib** for technical analysis indicators
- **Pandas** for data manipulation and analysis

## 📞 Support

For questions and support:

- 📧 Email: support@tradingbot.com
- 💬 Discord: [Trading Bot Community](https://discord.gg/trading-bot)
- 📚 Documentation: [Wiki](https://github.com/trading-bot/wiki)
- 🐛 Issues: [GitHub Issues](https://github.com/trading-bot/issues)

---

**Happy Trading! 📈🤖**