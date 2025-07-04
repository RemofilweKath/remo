from datetime import datetime, timedelta
from typing import Dict, Optional, Union, List
from enum import Enum
import logging
import uuid
from config.config import TradingConfig

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(Enum):
    """Time in force enumeration"""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

class Trade:
    """
    Trade class representing an order to be placed
    Handles order creation, modification, and management
    """
    
    def __init__(self, symbol: str, side: OrderSide, order_type: OrderType, 
                 quantity: float, price: Optional[float] = None, 
                 stop_price: Optional[float] = None, time_in_force: TimeInForce = TimeInForce.DAY):
        """
        Initialize a new trade
        
        Args:
            symbol: Stock symbol
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/STOP/etc.)
            quantity: Quantity to trade
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
        """
        self.trade_id = str(uuid.uuid4())
        self.symbol = symbol.upper()
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        
        # Order tracking
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.remaining_quantity = quantity
        self.average_fill_price = 0.0
        self.commission = 0.0
        
        # Timestamps
        self.created_at = datetime.now()
        self.submitted_at = None
        self.filled_at = None
        self.cancelled_at = None
        
        # Risk management
        self.stop_loss_price = None
        self.take_profit_price = None
        self.trailing_stop_amount = None
        self.trailing_stop_percent = None
        
        # Fills tracking
        self.fills = []
        
        # Order legs (for complex orders)
        self.legs = []
        
        # Tags and metadata
        self.tags = []
        self.metadata = {}
        
        # Configuration
        self.config = TradingConfig()
        
        logger.info(f"Created trade {self.trade_id}: {side.value} {quantity} {symbol} @ {order_type.value}")
        
    def add_stop_loss(self, stop_loss_price: float) -> 'Trade':
        """
        Add stop loss to the trade
        
        Args:
            stop_loss_price: Stop loss price
            
        Returns:
            Self for method chaining
        """
        self.stop_loss_price = stop_loss_price
        logger.info(f"Added stop loss at {stop_loss_price} to trade {self.trade_id}")
        return self
        
    def add_take_profit(self, take_profit_price: float) -> 'Trade':
        """
        Add take profit to the trade
        
        Args:
            take_profit_price: Take profit price
            
        Returns:
            Self for method chaining
        """
        self.take_profit_price = take_profit_price
        logger.info(f"Added take profit at {take_profit_price} to trade {self.trade_id}")
        return self
        
    def add_trailing_stop(self, trailing_amount: Optional[float] = None, 
                         trailing_percent: Optional[float] = None) -> 'Trade':
        """
        Add trailing stop to the trade
        
        Args:
            trailing_amount: Trailing stop amount
            trailing_percent: Trailing stop percentage
            
        Returns:
            Self for method chaining
        """
        if trailing_amount:
            self.trailing_stop_amount = trailing_amount
        if trailing_percent:
            self.trailing_stop_percent = trailing_percent
            
        logger.info(f"Added trailing stop to trade {self.trade_id}")
        return self
        
    def modify_price(self, new_price: float) -> bool:
        """
        Modify the order price
        
        Args:
            new_price: New price
            
        Returns:
            True if modification successful
        """
        try:
            if self.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                logger.warning(f"Cannot modify price for trade {self.trade_id} with status {self.status}")
                return False
                
            old_price = self.price
            self.price = new_price
            logger.info(f"Modified price for trade {self.trade_id} from {old_price} to {new_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying price for trade {self.trade_id}: {e}")
            return False
            
    def modify_quantity(self, new_quantity: float) -> bool:
        """
        Modify the order quantity
        
        Args:
            new_quantity: New quantity
            
        Returns:
            True if modification successful
        """
        try:
            if self.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                logger.warning(f"Cannot modify quantity for trade {self.trade_id} with status {self.status}")
                return False
                
            if new_quantity <= self.filled_quantity:
                logger.warning(f"New quantity {new_quantity} must be greater than filled quantity {self.filled_quantity}")
                return False
                
            old_quantity = self.quantity
            self.quantity = new_quantity
            self.remaining_quantity = new_quantity - self.filled_quantity
            logger.info(f"Modified quantity for trade {self.trade_id} from {old_quantity} to {new_quantity}")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying quantity for trade {self.trade_id}: {e}")
            return False
            
    def modify_order_type(self, new_order_type: OrderType, new_price: Optional[float] = None) -> bool:
        """
        Modify the order type
        
        Args:
            new_order_type: New order type
            new_price: New price (if required for new order type)
            
        Returns:
            True if modification successful
        """
        try:
            if self.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                logger.warning(f"Cannot modify order type for trade {self.trade_id} with status {self.status}")
                return False
                
            old_order_type = self.order_type
            self.order_type = new_order_type
            
            if new_price is not None:
                self.price = new_price
                
            logger.info(f"Modified order type for trade {self.trade_id} from {old_order_type} to {new_order_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying order type for trade {self.trade_id}: {e}")
            return False
            
    def cancel_order(self) -> bool:
        """
        Cancel the order
        
        Returns:
            True if cancellation successful
        """
        try:
            if self.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                logger.warning(f"Cannot cancel trade {self.trade_id} with status {self.status}")
                return False
                
            self.status = OrderStatus.CANCELLED
            self.cancelled_at = datetime.now()
            logger.info(f"Cancelled trade {self.trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling trade {self.trade_id}: {e}")
            return False
            
    def submit_order(self) -> bool:
        """
        Submit the order
        
        Returns:
            True if submission successful
        """
        try:
            if self.status != OrderStatus.PENDING:
                logger.warning(f"Cannot submit trade {self.trade_id} with status {self.status}")
                return False
                
            # Validate order
            if not self.validate_order():
                return False
                
            self.status = OrderStatus.SUBMITTED
            self.submitted_at = datetime.now()
            logger.info(f"Submitted trade {self.trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting trade {self.trade_id}: {e}")
            return False
            
    def add_fill(self, fill_price: float, fill_quantity: float, commission: float = 0.0) -> bool:
        """
        Add a fill to the order
        
        Args:
            fill_price: Fill price
            fill_quantity: Fill quantity
            commission: Commission for this fill
            
        Returns:
            True if fill added successfully
        """
        try:
            if fill_quantity <= 0:
                logger.warning(f"Invalid fill quantity {fill_quantity} for trade {self.trade_id}")
                return False
                
            if self.filled_quantity + fill_quantity > self.quantity:
                logger.warning(f"Fill quantity {fill_quantity} exceeds remaining quantity for trade {self.trade_id}")
                return False
                
            # Add fill record
            fill_record = {
                'timestamp': datetime.now(),
                'price': fill_price,
                'quantity': fill_quantity,
                'commission': commission
            }
            self.fills.append(fill_record)
            
            # Update order status
            self.filled_quantity += fill_quantity
            self.remaining_quantity = self.quantity - self.filled_quantity
            self.commission += commission
            
            # Calculate average fill price
            total_value = sum(fill['price'] * fill['quantity'] for fill in self.fills)
            self.average_fill_price = total_value / self.filled_quantity
            
            # Update order status
            if self.filled_quantity >= self.quantity:
                self.status = OrderStatus.FILLED
                self.filled_at = datetime.now()
                logger.info(f"Trade {self.trade_id} fully filled at average price {self.average_fill_price}")
            else:
                self.status = OrderStatus.PARTIALLY_FILLED
                logger.info(f"Trade {self.trade_id} partially filled: {self.filled_quantity}/{self.quantity}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding fill to trade {self.trade_id}: {e}")
            return False
            
    def validate_order(self) -> bool:
        """
        Validate the order parameters
        
        Returns:
            True if order is valid
        """
        try:
            # Basic validation
            if self.quantity <= 0:
                logger.error(f"Invalid quantity {self.quantity} for trade {self.trade_id}")
                return False
                
            if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
                logger.error(f"Price required for {self.order_type} order {self.trade_id}")
                return False
                
            if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
                logger.error(f"Stop price required for {self.order_type} order {self.trade_id}")
                return False
                
            # Risk management validation
            if self.stop_loss_price and self.side == OrderSide.BUY:
                if self.stop_loss_price >= self.price:
                    logger.error(f"Stop loss price {self.stop_loss_price} must be below entry price {self.price}")
                    return False
                    
            if self.stop_loss_price and self.side == OrderSide.SELL:
                if self.stop_loss_price <= self.price:
                    logger.error(f"Stop loss price {self.stop_loss_price} must be above entry price {self.price}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade {self.trade_id}: {e}")
            return False
            
    def calculate_position_size(self, account_balance: float, risk_amount: float) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            account_balance: Current account balance
            risk_amount: Maximum risk amount
            
        Returns:
            Calculated position size
        """
        try:
            if not self.stop_loss_price or not self.price:
                return self.quantity
                
            # Calculate risk per share
            risk_per_share = abs(self.price - self.stop_loss_price)
            
            if risk_per_share <= 0:
                return self.quantity
                
            # Calculate position size
            position_size = risk_amount / risk_per_share
            
            # Ensure position size doesn't exceed account balance
            max_position_value = account_balance * self.config.FIXED_POSITION_SIZE
            max_shares = max_position_value / self.price
            
            position_size = min(position_size, max_shares)
            
            logger.info(f"Calculated position size for trade {self.trade_id}: {position_size}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for trade {self.trade_id}: {e}")
            return self.quantity
            
    def get_order_value(self) -> float:
        """
        Get the total order value
        
        Returns:
            Total order value
        """
        try:
            if self.price:
                return self.price * self.quantity
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating order value for trade {self.trade_id}: {e}")
            return 0.0
            
    def get_profit_loss(self, current_price: float) -> Dict[str, float]:
        """
        Calculate profit/loss for the trade
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with P&L metrics
        """
        try:
            if self.filled_quantity == 0:
                return {
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'total_pnl': 0.0,
                    'pnl_percent': 0.0
                }
                
            # Calculate unrealized P&L
            if self.side == OrderSide.BUY:
                unrealized_pnl = (current_price - self.average_fill_price) * self.filled_quantity
            else:
                unrealized_pnl = (self.average_fill_price - current_price) * self.filled_quantity
                
            # Include commission
            unrealized_pnl -= self.commission
            
            # Calculate percentage
            total_cost = self.average_fill_price * self.filled_quantity
            pnl_percent = (unrealized_pnl / total_cost) * 100 if total_cost > 0 else 0.0
            
            return {
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': 0.0,  # Only applicable when position is closed
                'total_pnl': unrealized_pnl,
                'pnl_percent': pnl_percent
            }
            
        except Exception as e:
            logger.error(f"Error calculating P&L for trade {self.trade_id}: {e}")
            return {
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_pnl': 0.0,
                'pnl_percent': 0.0
            }
            
    def add_tag(self, tag: str) -> 'Trade':
        """
        Add a tag to the trade
        
        Args:
            tag: Tag to add
            
        Returns:
            Self for method chaining
        """
        if tag not in self.tags:
            self.tags.append(tag)
        return self
        
    def add_metadata(self, key: str, value: any) -> 'Trade':
        """
        Add metadata to the trade
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            Self for method chaining
        """
        self.metadata[key] = value
        return self
        
    def to_dict(self) -> Dict:
        """
        Convert trade to dictionary
        
        Returns:
            Dictionary representation of the trade
        """
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_fill_price': self.average_fill_price,
            'commission': self.commission,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'trailing_stop_amount': self.trailing_stop_amount,
            'trailing_stop_percent': self.trailing_stop_percent,
            'fills': self.fills,
            'tags': self.tags,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Trade':
        """
        Create trade from dictionary
        
        Args:
            data: Dictionary representation
            
        Returns:
            Trade object
        """
        trade = cls(
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            order_type=OrderType(data['order_type']),
            quantity=data['quantity'],
            price=data.get('price'),
            stop_price=data.get('stop_price'),
            time_in_force=TimeInForce(data['time_in_force'])
        )
        
        # Restore state
        trade.trade_id = data['trade_id']
        trade.status = OrderStatus(data['status'])
        trade.filled_quantity = data['filled_quantity']
        trade.remaining_quantity = data['remaining_quantity']
        trade.average_fill_price = data['average_fill_price']
        trade.commission = data['commission']
        
        # Restore timestamps
        trade.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('submitted_at'):
            trade.submitted_at = datetime.fromisoformat(data['submitted_at'])
        if data.get('filled_at'):
            trade.filled_at = datetime.fromisoformat(data['filled_at'])
        if data.get('cancelled_at'):
            trade.cancelled_at = datetime.fromisoformat(data['cancelled_at'])
            
        # Restore risk management
        trade.stop_loss_price = data.get('stop_loss_price')
        trade.take_profit_price = data.get('take_profit_price')
        trade.trailing_stop_amount = data.get('trailing_stop_amount')
        trade.trailing_stop_percent = data.get('trailing_stop_percent')
        
        # Restore fills and metadata
        trade.fills = data.get('fills', [])
        trade.tags = data.get('tags', [])
        trade.metadata = data.get('metadata', {})
        
        return trade
        
    def __str__(self) -> str:
        """String representation of the trade"""
        return f"Trade({self.trade_id}): {self.side.value} {self.quantity} {self.symbol} @ {self.order_type.value} - {self.status.value}"
        
    def __repr__(self) -> str:
        """Detailed string representation of the trade"""
        return f"Trade(id={self.trade_id}, symbol={self.symbol}, side={self.side.value}, type={self.order_type.value}, qty={self.quantity}, price={self.price}, status={self.status.value})"