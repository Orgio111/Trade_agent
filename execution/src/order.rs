use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderStatus {
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub exchange_order_id: Option<String>,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    pub leverage: u8,
    pub timestamp: u64,
    pub status: OrderStatus,
    pub reduce_only: bool,
    pub post_only: bool,
}

impl Order {
    pub fn new(
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: f64,
        price: Option<f64>,
        leverage: u8,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            exchange_order_id: None,
            symbol,
            side,
            order_type,
            quantity,
            filled_quantity: 0.0,
            price,
            stop_price: None,
            leverage,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            status: OrderStatus::Pending,
            reduce_only: false,
            post_only: false,
        }
    }

    pub fn notional_value(&self) -> f64 {
        let price = self.price.unwrap_or(0.0);
        self.quantity * price * self.leverage as f64
    }

    pub fn is_filled(&self) -> bool {
        matches!(self.status, OrderStatus::Filled)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub entry_price: f64,
    pub current_price: f64,
    pub quantity: f64,
    pub leverage: u8,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub stop_loss: Option<f64>,
    pub take_profits: Vec<TakeProfitLevel>,
    pub opened_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeProfitLevel {
    pub level: u8,
    pub price: f64,
    pub qty_pct: f64,  // 0.0 - 1.0
    pub filled: bool,
    pub trail: bool,
    pub trail_distance: Option<f64>,
}

impl Position {
    pub fn new(
        symbol: String,
        side: OrderSide,
        entry_price: f64,
        quantity: f64,
        leverage: u8,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            symbol,
            side,
            entry_price,
            current_price: entry_price,
            quantity,
            leverage,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            stop_loss: None,
            take_profits: Vec::new(),
            opened_at: chrono::Utc::now().timestamp_millis() as u64,
        }
    }

    pub fn pnl_pct(&self) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }
        match self.side {
            OrderSide::Buy => (self.current_price - self.entry_price) / self.entry_price,
            OrderSide::Sell => (self.entry_price - self.current_price) / self.entry_price,
        }
    }

    pub fn update_price(&mut self, price: f64) {
        self.current_price = price;
        let raw_pnl = self.entry_price * self.quantity * self.pnl_pct() * self.leverage as f64;
        self.unrealized_pnl = raw_pnl;
    }
}
