import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging
from bot.trade import Trade, OrderSide, OrderStatus
from config.config import TradingConfig

logger = logging.getLogger(__name__)

class Position:
    """
    Represents a single position in the portfolio
    """
    
    def __init__(self, symbol: str, quantity: float, average_price: float, 
                 side: OrderSide, opened_at: datetime = None):
        self.symbol = symbol.upper()
        self.quantity = quantity
        self.average_price = average_price
        self.side = side
        self.opened_at = opened_at or datetime.now()
        self.trades = []  # List of trades that created this position
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
    def add_trade(self, trade: Trade):
        """Add a trade to this position"""
        self.trades.append(trade)
        
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L based on current price"""
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (current_price - self.average_price) * self.quantity
        else:
            self.unrealized_pnl = (self.average_price - current_price) * self.quantity
            
    def get_market_value(self, current_price: float) -> float:
        """Get current market value of position"""
        return current_price * abs(self.quantity)
        
    def get_cost_basis(self) -> float:
        """Get cost basis of position"""
        return self.average_price * abs(self.quantity)

class Portfolio:
    """
    Portfolio class representing a trading portfolio with multiple positions
    Calculates portfolio metrics and manages positions
    """
    
    def __init__(self, initial_capital: float = None, name: str = "Main Portfolio"):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Starting capital amount
            name: Portfolio name
        """
        self.name = name
        self.config = TradingConfig()
        self.initial_capital = initial_capital or self.config.INITIAL_CAPITAL
        self.cash = self.initial_capital
        self.positions = {}  # symbol -> Position
        self.trades = []  # All trades
        self.created_at = datetime.now()
        
        # Performance tracking
        self.performance_history = []
        self.daily_returns = []
        self.equity_curve = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.max_drawdown_date = None
        self.peak_equity = self.initial_capital
        
        # Portfolio statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        
        logger.info(f"Portfolio '{name}' initialized with ${initial_capital:,.2f}")
        
    def add_trade(self, trade: Trade) -> bool:
        """
        Add a trade to the portfolio
        
        Args:
            trade: Trade object to add
            
        Returns:
            True if trade added successfully
        """
        try:
            if trade.status != OrderStatus.FILLED:
                logger.warning(f"Trade {trade.trade_id} is not filled, cannot add to portfolio")
                return False
                
            self.trades.append(trade)
            self.total_trades += 1
            self.total_commission += trade.commission
            
            # Update cash
            trade_value = trade.average_fill_price * trade.filled_quantity
            if trade.side == OrderSide.BUY:
                self.cash -= (trade_value + trade.commission)
            else:
                self.cash += (trade_value - trade.commission)
                
            # Update or create position
            self._update_position(trade)
            
            # Update performance tracking
            self._update_performance()
            
            logger.info(f"Added trade {trade.trade_id} to portfolio")
            return True
            
        except Exception as e:
            logger.error(f"Error adding trade to portfolio: {e}")
            return False
            
    def _update_position(self, trade: Trade):
        """Update position based on trade"""
        try:
            symbol = trade.symbol
            
            if symbol not in self.positions:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=trade.filled_quantity if trade.side == OrderSide.BUY else -trade.filled_quantity,
                    average_price=trade.average_fill_price,
                    side=trade.side,
                    opened_at=trade.filled_at
                )
            else:
                # Update existing position
                position = self.positions[symbol]
                
                if trade.side == OrderSide.BUY:
                    new_quantity = position.quantity + trade.filled_quantity
                else:
                    new_quantity = position.quantity - trade.filled_quantity
                    
                # Calculate new average price
                if new_quantity != 0:
                    total_cost = (position.average_price * abs(position.quantity) + 
                                trade.average_fill_price * trade.filled_quantity)
                    position.average_price = total_cost / (abs(position.quantity) + trade.filled_quantity)
                    position.quantity = new_quantity
                else:
                    # Position closed
                    realized_pnl = self._calculate_realized_pnl(position, trade)
                    position.realized_pnl += realized_pnl
                    if realized_pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    del self.positions[symbol]
                    
            # Add trade to position
            if symbol in self.positions:
                self.positions[symbol].add_trade(trade)
                
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            
    def _calculate_realized_pnl(self, position: Position, closing_trade: Trade) -> float:
        """Calculate realized P&L when closing a position"""
        try:
            if position.side == OrderSide.BUY:
                # Long position closed with sell
                return (closing_trade.average_fill_price - position.average_price) * closing_trade.filled_quantity
            else:
                # Short position closed with buy
                return (position.average_price - closing_trade.average_fill_price) * closing_trade.filled_quantity
                
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {e}")
            return 0.0
            
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position object or None if not found
        """
        return self.positions.get(symbol.upper())
        
    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all positions
        
        Returns:
            Dictionary of all positions
        """
        return self.positions.copy()
        
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update all positions with current prices
        
        Args:
            current_prices: Dictionary of symbol -> current price
        """
        try:
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    position.update_unrealized_pnl(current_prices[symbol])
                    
            # Update performance metrics
            self._update_performance()
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Get total portfolio value
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Total portfolio value
        """
        try:
            total_value = self.cash
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    market_value = position.get_market_value(current_prices[symbol])
                    total_value += market_value
                    
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return self.cash
            
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """
        Get total unrealized P&L
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Total unrealized P&L
        """
        try:
            total_unrealized = 0.0
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    position.update_unrealized_pnl(current_prices[symbol])
                    total_unrealized += position.unrealized_pnl
                    
            return total_unrealized
            
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {e}")
            return 0.0
            
    def get_realized_pnl(self) -> float:
        """
        Get total realized P&L
        
        Returns:
            Total realized P&L
        """
        try:
            total_realized = 0.0
            
            for position in self.positions.values():
                total_realized += position.realized_pnl
                
            return total_realized
            
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {e}")
            return 0.0
            
    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """
        Get total P&L (realized + unrealized)
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Total P&L
        """
        return self.get_realized_pnl() + self.get_unrealized_pnl(current_prices)
        
    def get_portfolio_return(self, current_prices: Dict[str, float]) -> float:
        """
        Get portfolio return percentage
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Portfolio return as percentage
        """
        try:
            current_value = self.get_portfolio_value(current_prices)
            return ((current_value / self.initial_capital) - 1) * 100
            
        except Exception as e:
            logger.error(f"Error calculating portfolio return: {e}")
            return 0.0
            
    def get_position_sizes(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Get position sizes as percentage of portfolio
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Dictionary of symbol -> position size percentage
        """
        try:
            portfolio_value = self.get_portfolio_value(current_prices)
            position_sizes = {}
            
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    market_value = position.get_market_value(current_prices[symbol])
                    position_sizes[symbol] = (market_value / portfolio_value) * 100
                    
            return position_sizes
            
        except Exception as e:
            logger.error(f"Error calculating position sizes: {e}")
            return {}
            
    def get_cash_percentage(self, current_prices: Dict[str, float]) -> float:
        """
        Get cash as percentage of portfolio
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Cash percentage
        """
        try:
            portfolio_value = self.get_portfolio_value(current_prices)
            return (self.cash / portfolio_value) * 100
            
        except Exception as e:
            logger.error(f"Error calculating cash percentage: {e}")
            return 0.0
            
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        try:
            if not self.daily_returns:
                return 0.0
                
            returns = np.array(self.daily_returns)
            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
            
            if len(excess_returns) < 2:
                return 0.0
                
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
            
    def calculate_max_drawdown(self) -> Tuple[float, Optional[datetime]]:
        """
        Calculate maximum drawdown
        
        Returns:
            Tuple of (max_drawdown_percentage, date_of_max_drawdown)
        """
        try:
            if not self.equity_curve:
                return 0.0, None
                
            equity_values = [point['value'] for point in self.equity_curve]
            peak = equity_values[0]
            max_drawdown = 0.0
            max_drawdown_date = None
            
            for i, value in enumerate(equity_values):
                if value > peak:
                    peak = value
                    
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_date = self.equity_curve[i]['date']
                    
            return max_drawdown * 100, max_drawdown_date
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0, None
            
    def calculate_win_rate(self) -> float:
        """
        Calculate win rate
        
        Returns:
            Win rate as percentage
        """
        try:
            if self.total_trades == 0:
                return 0.0
                
            return (self.winning_trades / self.total_trades) * 100
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
            
    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor
        
        Returns:
            Profit factor
        """
        try:
            total_wins = sum(trade.get_profit_loss(0)['total_pnl'] for trade in self.trades 
                           if trade.get_profit_loss(0)['total_pnl'] > 0)
            total_losses = abs(sum(trade.get_profit_loss(0)['total_pnl'] for trade in self.trades 
                                 if trade.get_profit_loss(0)['total_pnl'] < 0))
            
            if total_losses == 0:
                return float('inf') if total_wins > 0 else 0.0
                
            return total_wins / total_losses
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0
            
    def get_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict[str, Union[float, int, str]]:
        """
        Get comprehensive portfolio metrics
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Dictionary of portfolio metrics
        """
        try:
            # Update positions first
            self.update_positions(current_prices)
            
            # Calculate drawdown
            max_drawdown, max_drawdown_date = self.calculate_max_drawdown()
            
            metrics = {
                # Basic metrics
                'portfolio_value': self.get_portfolio_value(current_prices),
                'initial_capital': self.initial_capital,
                'cash': self.cash,
                'cash_percentage': self.get_cash_percentage(current_prices),
                'total_return': self.get_portfolio_return(current_prices),
                'total_pnl': self.get_total_pnl(current_prices),
                'unrealized_pnl': self.get_unrealized_pnl(current_prices),
                'realized_pnl': self.get_realized_pnl(),
                
                # Performance metrics
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'max_drawdown': max_drawdown,
                'max_drawdown_date': max_drawdown_date.strftime('%Y-%m-%d') if max_drawdown_date else None,
                'win_rate': self.calculate_win_rate(),
                'profit_factor': self.calculate_profit_factor(),
                
                # Trading metrics
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'total_commission': self.total_commission,
                'active_positions': len(self.positions),
                
                # Risk metrics
                'position_sizes': self.get_position_sizes(current_prices),
                'largest_position': max(self.get_position_sizes(current_prices).values()) if self.positions else 0.0,
                
                # Portfolio info
                'portfolio_name': self.name,
                'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'days_active': (datetime.now() - self.created_at).days
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
            
    def _update_performance(self):
        """Update performance tracking"""
        try:
            # This would typically be called with current prices
            # For now, we'll just record the timestamp
            current_time = datetime.now()
            
            # Add to performance history
            self.performance_history.append({
                'timestamp': current_time,
                'cash': self.cash,
                'positions_count': len(self.positions),
                'total_trades': self.total_trades
            })
            
            # Limit history size
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
            
    def get_positions_summary(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """
        Get summary of all positions
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            DataFrame with position summaries
        """
        try:
            if not self.positions:
                return pd.DataFrame()
                
            data = []
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    position.update_unrealized_pnl(current_price)
                    
                    data.append({
                        'symbol': symbol,
                        'quantity': position.quantity,
                        'side': position.side.value,
                        'average_price': position.average_price,
                        'current_price': current_price,
                        'market_value': position.get_market_value(current_price),
                        'cost_basis': position.get_cost_basis(),
                        'unrealized_pnl': position.unrealized_pnl,
                        'unrealized_pnl_percent': (position.unrealized_pnl / position.get_cost_basis()) * 100,
                        'opened_at': position.opened_at.strftime('%Y-%m-%d %H:%M:%S'),
                        'days_held': (datetime.now() - position.opened_at).days
                    })
                    
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error getting positions summary: {e}")
            return pd.DataFrame()
            
    def get_trades_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades
        
        Returns:
            DataFrame with trade summaries
        """
        try:
            if not self.trades:
                return pd.DataFrame()
                
            data = []
            for trade in self.trades:
                data.append({
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': trade.quantity,
                    'filled_quantity': trade.filled_quantity,
                    'average_price': trade.average_fill_price,
                    'order_type': trade.order_type.value,
                    'status': trade.status.value,
                    'commission': trade.commission,
                    'created_at': trade.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'filled_at': trade.filled_at.strftime('%Y-%m-%d %H:%M:%S') if trade.filled_at else None
                })
                
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error getting trades summary: {e}")
            return pd.DataFrame()
            
    def export_to_dict(self) -> Dict:
        """
        Export portfolio to dictionary
        
        Returns:
            Dictionary representation of portfolio
        """
        try:
            return {
                'name': self.name,
                'initial_capital': self.initial_capital,
                'cash': self.cash,
                'created_at': self.created_at.isoformat(),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'total_commission': self.total_commission,
                'positions': {symbol: {
                    'quantity': pos.quantity,
                    'average_price': pos.average_price,
                    'side': pos.side.value,
                    'opened_at': pos.opened_at.isoformat(),
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                } for symbol, pos in self.positions.items()},
                'trades': [trade.to_dict() for trade in self.trades]
            }
            
        except Exception as e:
            logger.error(f"Error exporting portfolio: {e}")
            return {}
            
    def risk_check(self, current_prices: Dict[str, float]) -> Dict[str, Union[bool, str, float]]:
        """
        Perform risk checks on portfolio
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Dictionary with risk check results
        """
        try:
            position_sizes = self.get_position_sizes(current_prices)
            portfolio_value = self.get_portfolio_value(current_prices)
            
            risks = {
                'max_position_exceeded': False,
                'concentration_risk': False,
                'leverage_risk': False,
                'cash_risk': False,
                'max_position_size': 0.0,
                'concentration_count': 0,
                'leverage_ratio': 0.0,
                'cash_percentage': self.get_cash_percentage(current_prices),
                'warnings': []
            }
            
            # Check maximum position size
            max_position = max(position_sizes.values()) if position_sizes else 0.0
            risks['max_position_size'] = max_position
            if max_position > 20:  # 20% concentration limit
                risks['max_position_exceeded'] = True
                risks['warnings'].append(f"Position concentration exceeds 20%: {max_position:.1f}%")
                
            # Check concentration risk (multiple large positions)
            large_positions = [size for size in position_sizes.values() if size > 10]
            risks['concentration_count'] = len(large_positions)
            if len(large_positions) > 3:
                risks['concentration_risk'] = True
                risks['warnings'].append(f"Too many large positions: {len(large_positions)}")
                
            # Check cash levels
            cash_pct = self.get_cash_percentage(current_prices)
            if cash_pct < 5:
                risks['cash_risk'] = True
                risks['warnings'].append(f"Low cash reserves: {cash_pct:.1f}%")
                
            # Check for over-leveraging (if applicable)
            total_position_value = sum(position_sizes.values())
            if total_position_value > 100:  # Over 100% invested
                risks['leverage_risk'] = True
                risks['warnings'].append(f"Over-leveraged: {total_position_value:.1f}%")
                
            return risks
            
        except Exception as e:
            logger.error(f"Error performing risk check: {e}")
            return {'warnings': ['Error performing risk check']}
            
    def __str__(self) -> str:
        """String representation of portfolio"""
        return f"Portfolio(name={self.name}, value=${self.cash:,.2f}, positions={len(self.positions)}, trades={self.total_trades})"
        
    def __repr__(self) -> str:
        """Detailed string representation of portfolio"""
        return f"Portfolio(name={self.name}, initial_capital={self.initial_capital}, cash={self.cash}, positions={len(self.positions)}, trades={self.total_trades})"