"""Execution Engine - Binance API integration with paper trading mode."""

import asyncio
import hmac
import hashlib
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import aiohttp
from models import Signal, SignalAction, ExecutionResult, OrderSide, OrderType, OrderStatus
import yaml

logger = logging.getLogger(__name__)


class OrderRequest:
    """Order request parameters."""
    def __init__(self, symbol: str, side: OrderSide, order_type: OrderType,
                 quantity: float, price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 time_in_force: str = "GTC",
                 reduce_only: bool = False,
                 post_only: bool = True):
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.reduce_only = reduce_only
        self.post_only = post_only


@dataclass
class OrderResponse:
    """Order execution response."""
    order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: float
    status: OrderStatus
    filled_qty: float = 0.0
    avg_price: float = 0.0
    timestamp: int = 0
    client_order_id: str = ""


class BinanceClient:
    """Async Binance REST API client."""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._listen_key: Optional[str] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"X-MBX-APIKEY": self.api_key},
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _sign(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature."""
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
    
    async def _request(self, method: str, endpoint: str, 
                       params: Dict = None, signed: bool = False) -> Dict:
        """Make authenticated request."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)
        
        try:
            async with self.session.request(method, url, params=params) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise Exception(f"API Error {resp.status}: {data}")
                return data
        except Exception as e:
            logger.error(f"Request failed: {method} {endpoint} - {e}")
            raise
    
    async def get_account(self) -> Dict:
        """Get account balance and positions."""
        return await self._request("GET", "/fapi/v2/account", signed=True)
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions."""
        account = await self.get_account()
        positions = []
        for pos in account.get("positions", []):
            if float(pos.get("positionAmt", 0)) != 0:
                positions.append({
                    "symbol": pos["symbol"],
                    "side": "LONG" if float(pos["positionAmt"]) > 0 else "SHORT",
                    "size": abs(float(pos["positionAmt"])),
                    "entry_price": float(pos["entryPrice"]),
                    "mark_price": float(pos["markPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"]),
                    "leverage": int(pos["leverage"]),
                })
        return positions
    
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place new order."""
        params = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": f"{order.quantity:.4f}",
            "timeInForce": order.time_in_force,
            "reduceOnly": "true" if order.reduce_only else "false",
        }
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP]:
            params["price"] = f"{order.price:.2f}"
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_MARKET]:
            params["stopPrice"] = f"{order.stop_price:.2f}"
        
        if order.post_only and order.order_type == OrderType.LIMIT:
            params["priceProtect"] = "true"
        
        result = await self._request("POST", "/fapi/v1/order", params, signed=True)
        
        return OrderResponse(
            order_id=str(result["orderId"]),
            symbol=result["symbol"],
            side=OrderSide(result["side"]),
            type=OrderType(result["type"]),
            quantity=float(result["origQty"]),
            price=float(result["price"]),
            status=OrderStatus(result["status"]),
            timestamp=result["updateTime"],
            client_order_id=result["clientOrderId"]
        )
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel order."""
        params = {"symbol": symbol, "orderId": order_id}
        return await self._request("DELETE", "/fapi/v1/order", params, signed=True)
    
    async def get_order(self, symbol: str, order_id: str) -> Dict:
        """Query order status."""
        params = {"symbol": symbol, "orderId": order_id}
        return await self._request("GET", "/fapi/v1/order", params, signed=True)
    
    async def change_leverage(self, symbol: str, leverage: int) -> Dict:
        """Change leverage."""
        params = {"symbol": symbol, "leverage": leverage}
        return await self._request("POST", "/fapi/v1/leverage", params, signed=True)
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker."""
        return await self._request("GET", "/fapi/v1/ticker/24hr", {"symbol": symbol})
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[List]:
        """Get kline/candlestick data."""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return await self._request("GET", "/fapi/v1/klines", params)


class PaperClient:
    """Paper trading client for simulation."""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[str, OrderResponse] = {}
        self.order_counter = 0
        self.fills: List[Dict] = []
        
        # Simulated market data
        self.current_prices: Dict[str, float] = {}
    
    def update_price(self, symbol: str, price: float):
        """Update current price for paper fills."""
        self.current_prices[symbol] = price
        self._check_fills(symbol, price)
    
    def _check_fills(self, symbol: str, price: float):
        """Check if any pending orders should fill."""
        to_remove = []
        for order_id, order in self.orders.items():
            if order.symbol != symbol or order.status != OrderStatus.NEW:
                continue
            
            filled = False
            fill_price = 0
            
            if order.type == OrderType.MARKET:
                filled = True
                fill_price = price
            elif order.type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.price:
                    filled = True
                    fill_price = order.price
                elif order.side == OrderSide.SELL and price >= order.price:
                    filled = True
                    fill_price = order.price
            elif order.type in [OrderType.STOP, OrderType.STOP_MARKET]:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    filled = True
                    fill_price = price if order.type == OrderType.STOP_MARKET else order.price
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    filled = True
                    fill_price = price if order.type == OrderType.STOP_MARKET else order.price
            
            if filled:
                order.status = OrderStatus.FILLED
                order.filled_qty = order.quantity
                order.avg_price = fill_price
                order.timestamp = int(time.time() * 1000)
                
                # Update position
                self._update_position(order, fill_price)
                
                # Record fill
                self.fills.append({
                    "order_id": order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "price": fill_price,
                    "qty": order.quantity,
                    "time": order.timestamp
                })
                
                to_remove.append(order_id)
        
        for oid in to_remove:
            del self.orders[oid]
    
    def _update_position(self, order: OrderResponse, fill_price: float):
        """Update position after fill."""
        symbol = order.symbol
        qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                "size": 0,
                "entry_price": 0,
                "unrealized_pnl": 0
            }
        
        pos = self.positions[symbol]
        old_size = pos["size"]
        old_entry = pos["entry_price"]
        
        new_size = old_size + qty
        
        if new_size == 0:
            # Position closed
            pnl = (fill_price - old_entry) * abs(old_size)
            self.balance += pnl
            del self.positions[symbol]
        elif (old_size >= 0 and qty > 0) or (old_size <= 0 and qty < 0):
            # Adding to position
            total_cost = old_entry * abs(old_size) + fill_price * abs(qty)
            pos["entry_price"] = total_cost / abs(new_size)
            pos["size"] = new_size
        else:
            # Reducing position
            closed_qty = min(abs(old_size), abs(qty))
            pnl = (fill_price - old_entry) * closed_qty * (1 if old_size > 0 else -1)
            self.balance += pnl
            pos["size"] = new_size
            if pos["size"] == 0:
                del self.positions[symbol]
    
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Simulate order placement."""
        self.order_counter += 1
        order_id = f"paper_{self.order_counter}"
        
        response = OrderResponse(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            type=order.order_type,
            quantity=order.quantity,
            price=order.price or 0,
            status=OrderStatus.NEW,
            timestamp=int(time.time() * 1000),
            client_order_id=order_id
        )
        
        self.orders[order_id] = response
        
        # For market orders, try immediate fill
        if order.order_type == OrderType.MARKET:
            current_price = self.current_prices.get(order.symbol, order.price)
            if current_price:
                self.update_price(order.symbol, current_price)
        
        return response
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel order."""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELED
            return {"status": "CANCELED"}
        return {"error": "Order not found"}
    
    async def get_order(self, symbol: str, order_id: str) -> Dict:
        """Get order status."""
        if order_id in self.orders:
            return self.orders[order_id].__dict__
        return {"error": "Order not found"}
    
    def get_account(self) -> Dict:
        """Get account balance and positions."""
        total_unrealized = sum(
            (self.current_prices.get(s, p["entry_price"]) - p["entry_price"]) * p["size"]
            for s, p in self.positions.items()
        )
        
        return {
            "balance": self.balance,
            "equity": self.balance + total_unrealized,
            "unrealized_pnl": total_unrealized,
            "positions": [
                {"symbol": s, "side": "LONG" if p["size"] > 0 else "SHORT",
                 "size": abs(p["size"]), "entry_price": p["entry_price"],
                 "unrealized_pnl": (self.current_prices.get(s, p["entry_price"]) - p["entry_price"]) * p["size"]}
                for s, p in self.positions.items()
            ]
        }
    
    def get_positions(self) -> List[Dict]:
        """Get open positions."""
        return [
            {"symbol": s, "side": "LONG" if p["size"] > 0 else "SHORT",
             "size": abs(p["size"]), "entry_price": p["entry_price"]}
            for s, p in self.positions.items()
        ]


class ExecutionEngine:
    """
    Unified execution engine supporting both live and paper trading.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        exec_cfg = self.config.get("execution", {})
        sys_cfg = self.config.get("system", {})
        
        self.paper_mode = sys_cfg.get("paper_mode", True)
        self.maker_preference = exec_cfg.get("maker_preference", True)
        self.default_type = exec_cfg.get("default_type", "LIMIT")
        self.post_only = exec_cfg.get("post_only", True)
        self.max_slippage_bps = exec_cfg.get("max_slippage_bps", 5)
        
        self.client: Optional[BinanceClient] = None
        self.paper_client: Optional[PaperClient] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._price_callbacks: List[callable] = []
        
        if self.paper_mode:
            self.paper_client = PaperClient(
                initial_balance=self.config.get("paper", {}).get("initial_balance", 10000.0)
            )
        else:
            api_key = self.config.get("binance", {}).get("api_key", "")
            api_secret = self.config.get("binance", {}).get("api_secret", "")
            testnet = self.config.get("system", {}).get("testnet", True)
            self.client = BinanceClient(api_key, api_secret, testnet)
    
    async def start(self):
        """Initialize connections."""
        if self.client:
            await self.client.__aenter__()
        logger.info(f"Execution engine started (paper_mode={self.paper_mode})")
    
    async def stop(self):
        """Close connections."""
        if self.client:
            await self.client.__aexit__(None, None, None)
        if self._ws_task:
            self._ws_task.cancel()
        logger.info("Execution engine stopped")
    
    def add_price_callback(self, callback: callable):
        """Add callback for price updates (for paper trading)."""
        self._price_callbacks.append(callback)
    
    def _notify_price(self, symbol: str, price: float):
        """Notify all callbacks of price update."""
        for cb in self._price_callbacks:
            try:
                cb(symbol, price)
            except Exception as e:
                logger.error(f"Price callback error: {e}")
    
    async def execute(self, signal: Signal) -> ExecutionResult:
        """
        Execute signal -> place order -> return result.
        """
        if signal.action == SignalAction.HOLD:
            return ExecutionResult(
                success=False,
                reason="Signal is HOLD",
                order_id="",
                symbol=signal.symbol
            )
        
        # Determine order parameters
        side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL
        
        if self.default_type == "LIMIT" and self.maker_preference:
            order_type = OrderType.LIMIT
            price = signal.entry_price
        else:
            order_type = OrderType.MARKET
            price = None
        
        # Get account state for position sizing
        account = await self.get_account()
        equity = account.get("equity", account.get("balance", 10000))
        quantity = equity * signal.size_pct
        
        # Get current price for quantity calculation
        if self.paper_mode:
            current_price = self.paper_client.current_prices.get(signal.symbol, price or 50000)
        else:
            ticker = await self.client.get_ticker(signal.symbol)
            current_price = float(ticker["lastPrice"])
        
        if signal.entry_price:
            current_price = signal.entry_price
        
        qty = quantity / current_price
        
        # Round quantity to exchange precision
        qty = round(qty, 4)  # BTCUSDT precision
        
        if qty <= 0:
            return ExecutionResult(
                success=False,
                reason="Invalid quantity",
                order_id="",
                symbol=signal.symbol
            )
        
        # Create order
        order = OrderRequest(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType(order_type),
            quantity=qty,
            price=price,
            post_only=self.post_only,
            reduce_only=False
        )
        
        # Execute
        try:
            if self.paper_mode:
                response = await self.paper_client.place_order(order)
                # Update price for immediate fill check
                self.paper_client.update_price(signal.symbol, current_price)
            else:
                async with self.client as c:
                    response = await c.place_order(order)
            
            if response.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                return ExecutionResult(
                    success=True,
                    reason="Order filled",
                    order_id=response.order_id,
                    symbol=response.symbol,
                    filled_qty=response.filled_qty,
                    avg_price=response.avg_price
                )
            elif response.status == OrderStatus.NEW:
                return ExecutionResult(
                    success=True,
                    reason="Order placed (pending)",
                    order_id=response.order_id,
                    symbol=response.symbol
                )
            else:
                return ExecutionResult(
                    success=False,
                    reason=f"Order status: {response.status}",
                    order_id=response.order_id,
                    symbol=response.symbol
                )
                
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                reason=f"Execution error: {str(e)}",
                order_id="",
                symbol=signal.symbol
            )
    
    async def get_account(self) -> Dict:
        """Get account state."""
        if self.paper_mode:
            return self.paper_client.get_account()
        else:
            async with self.client as c:
                return await c.get_account()
    
    async def get_positions(self) -> List[Dict]:
        """Get open positions."""
        if self.paper_mode:
            return self.paper_client.get_positions()
        else:
            async with self.client as c:
                return await c.get_positions()
    
    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for symbol."""
        try:
            if self.paper_mode:
                to_cancel = [
                    oid for oid, order in self.paper_client.orders.items()
                    if order.symbol == symbol and order.status == OrderStatus.NEW
                ]
                for oid in to_cancel:
                    self.paper_client.orders[oid].status = OrderStatus.CANCELED
                return True
            else:
                async with self.client as c:
                    pass
            return True
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return False


async def create_execution_engine(config_path: str = "config.yaml") -> ExecutionEngine:
    engine = ExecutionEngine(config_path)
    await engine.start()
    return engine