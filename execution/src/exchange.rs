use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;

use crate::order::{Order, OrderSide, OrderType, Position};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    pub balance: f64,
    pub equity: f64,
    pub free_collateral: f64,
    pub unrealized_pnl: f64,
    pub margin_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub last: f64,
    pub volume_24h: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub close_time: u64,
    pub trades: u64,
}

#[async_trait]
pub trait ExchangeConnector: Send + Sync {
    /// Get account information
    async fn get_account(&self) -> Result<AccountInfo>;

    /// Place an order
    async fn place_order(&self, order: &Order) -> Result<String>;

    /// Cancel an order
    async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<bool>;

    /// Get order status
    async fn get_order(&self, symbol: &str, order_id: &str) -> Result<Order>;

    /// Get open orders
    async fn get_open_orders(&self, symbol: &str) -> Result<Vec<Order>>;

    /// Get position by symbol
    async fn get_position(&self, symbol: &str) -> Result<Option<Position>>;

    /// Get all positions
    async fn get_positions(&self) -> Result<Vec<Position>>;

    /// Get current market data
    async fn get_ticker(&self, symbol: &str) -> Result<MarketData>;

    /// Get order book
    async fn get_order_book(&self, symbol: &str, limit: u32) -> Result<OrderBook>;

    /// Get recent klines
    async fn get_klines(&self, symbol: &str, interval: &str, limit: u32) -> Result<Vec<Kline>>;

    /// Get exchange info (trading pairs, min notional, etc.)
    async fn get_exchange_info(&self) -> Result<HashMap<String, ExchangeSymbol>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeSymbol {
    pub symbol: String,
    pub base_asset: String,
    pub quote_asset: String,
    pub min_notional: f64,
    pub min_qty: f64,
    pub tick_size: f64,
    pub step_size: f64,
    pub leverage_max: u8,
}

/// Binance testnet connector (mock for Phase 1)
pub struct BinanceConnector {
    pub api_key: String,
    pub secret: String,
    pub testnet: bool,
    pub base_url: String,
}

impl BinanceConnector {
    pub fn new(api_key: String, secret: String) -> Self {
        Self {
            api_key,
            secret,
            testnet: true,
            base_url: "https://testnet.binance.vision/api".to_string(),
        }
    }

    fn build_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert("X-MBX-APIKEY".to_string(), self.api_key.clone());
        headers
    }

    fn signature(&self, query: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        let mut mac = Hmac::<Sha256>::new_from_slice(self.secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }
}

#[async_trait]
impl ExchangeConnector for BinanceConnector {
    async fn get_account(&self) -> Result<AccountInfo> {
        // Mock for Phase 1 — real HTTP calls with HMAC signing in Phase 2
        Ok(AccountInfo {
            balance: 100.0,
            equity: 100.0,
            free_collateral: 100.0,
            unrealized_pnl: 0.0,
            margin_ratio: 0.0,
        })
    }

    async fn place_order(&self, order: &Order) -> Result<String> {
        tracing::info!(
            "📤 Order: {} {} {} qty={} price={:?} lev={}x",
            order.symbol, 
            if order.side == OrderSide::Buy { "BUY" } else { "SELL" },
            if order.order_type == OrderType::Market { "MARKET" } else { "LIMIT" },
            order.quantity,
            order.price,
            order.leverage,
        );
        // Mock order ID
        Ok(format!("mock_{}", uuid::Uuid::new_v4()))
    }

    async fn cancel_order(&self, _symbol: &str, _order_id: &str) -> Result<bool> {
        Ok(true)
    }

    async fn get_order(&self, _symbol: &str, _order_id: &str) -> Result<Order> {
        anyhow::bail!("Not implemented in mock")
    }

    async fn get_open_orders(&self, _symbol: &str) -> Result<Vec<Order>> {
        Ok(Vec::new())
    }

    async fn get_position(&self, _symbol: &str) -> Result<Option<Position>> {
        Ok(None)
    }

    async fn get_positions(&self) -> Result<Vec<Position>> {
        Ok(Vec::new())
    }

    async fn get_ticker(&self, symbol: &str) -> Result<MarketData> {
        Ok(MarketData {
            symbol: symbol.to_string(),
            bid: 50000.0,
            ask: 50001.0,
            last: 50000.5,
            volume_24h: 10000.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        })
    }

    async fn get_order_book(&self, _symbol: &str, _limit: u32) -> Result<OrderBook> {
        Ok(OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel { price: 50000.0, quantity: 1.0 },
                OrderBookLevel { price: 49900.0, quantity: 2.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 50100.0, quantity: 1.5 },
                OrderBookLevel { price: 50200.0, quantity: 2.5 },
            ],
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        })
    }

    async fn get_klines(&self, _symbol: &str, _interval: &str, _limit: u32) -> Result<Vec<Kline>> {
        Ok(Vec::new())
    }

    async fn get_exchange_info(&self) -> Result<HashMap<String, ExchangeSymbol>> {
        let mut map = HashMap::new();
        map.insert("BTCUSDT".to_string(), ExchangeSymbol {
            symbol: "BTCUSDT".to_string(),
            base_asset: "BTC".to_string(),
            quote_asset: "USDT".to_string(),
            min_notional: 10.0,
            min_qty: 0.001,
            tick_size: 0.01,
            step_size: 0.001,
            leverage_max: 10,
        });
        Ok(map)
    }
}
