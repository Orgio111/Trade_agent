#![allow(dead_code)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::{Result, Context};
use reqwest::Client;

use crate::order::{Order, OrderSide, OrderType, Position};

/// Format f64 with fixed decimal precision (replaces Python's {:.8f}).
fn format_f64_prec(val: f64, prec: usize) -> String {
    format!("{:.prec$e}", val, prec = prec).parse::<f64>().unwrap_or(0.0).to_string()
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    pub balance: f64,
    pub equity: f64,
    pub free_collateral: f64,
    pub unrealized_pnl: f64,
    pub margin_ratio: f64,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: u64,
}

#[allow(dead_code)]
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
#[allow(dead_code)]
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

    /// Place OCO (One-Cancels-Other) bracket order with stop-loss and take-profit.
    async fn place_oco_order(
        &self,
        symbol: &str,
        stop_loss_price: f64,
        take_profit_price: f64,
        quantity: f64,
    ) -> Result<String>;
}

#[allow(dead_code)]
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

// ── Binance Testnet API response types ─────────────────────

#[derive(Deserialize, Debug)]
struct BinanceAccount {
    balances: Vec<BinanceBalance>,
}

#[derive(Deserialize, Debug)]
struct BinanceBalance {
    asset: String,
    #[serde(rename = "free")]
    free: String,
    #[serde(rename = "locked")]
    locked: String,
}

#[derive(Deserialize, Debug)]
struct BinanceTicker {
    symbol: String,
    bid_price: String,
    ask_price: String,
    last_price: String,
    volume: String,
    high_price: String,
    low_price: String,
    #[serde(rename = "closeTime")]
    close_time: u64,
}

#[derive(Deserialize, Debug)]
struct BinanceOrderResult {
    #[serde(rename = "orderId")]
    order_id: Option<i64>,
    #[serde(rename = "clientOrderId")]
    client_order_id: Option<String>,
    status: Option<String>,
    #[serde(rename = "executedQty")]
    executed_qty: Option<String>,
    #[serde(rename = "cummulativeQuoteQty")]
    cumm_quote_qty: Option<String>,
    side: Option<String>,
}

#[derive(Deserialize, Debug)]
struct BinanceFill {
    price: String,
    qty: String,
    commission: String,
}

#[derive(Deserialize, Debug)]
struct BinanceDepth {
    bids: Vec<[String; 2]>,
    asks: Vec<[String; 2]>,
}

// Binance klines API returns arrays, not objects.
// We parse as Vec<Vec<Value>> and extract by position index.
#[derive(Deserialize, Debug)]
struct BinanceKlineRaw(
    u64,    // 0: open_time
    String, // 1: open
    String, // 2: high
    String, // 3: low
    String, // 4: close
    String, // 5: volume
    u64,    // 6: close_time
    String, // 7: quote_vol
    u64,    // 8: count
    String, // 9: taker_buy_vol
    String, // 10: taker_buy_quote_vol
    String, // 11: ignore
);

#[derive(Deserialize, Debug)]
struct BinanceExchangeInfo {
    symbols: Vec<BinanceSymbolInfo>,
}

#[derive(Deserialize, Debug)]
struct BinanceSymbolInfo {
    symbol: String,
    #[serde(rename = "baseAsset")]
    base_asset: String,
    #[serde(rename = "quoteAsset")]
    quote_asset: String,
    filters: Vec<serde_json::Value>,
    #[serde(rename = "isSpotTradingAllowed")]
    is_spot_trading_allowed: Option<bool>,
}

// ── Binance Testnet Connector ──────────────────────────────

pub struct BinanceConnector {
    pub api_key: String,
    pub secret: String,
    pub testnet: bool,
    pub base_url: String,
    pub ws_url: String,
    client: Client,
}

impl BinanceConnector {
    pub fn new(api_key: String, secret: String) -> Self {
        Self {
            api_key,
            secret,
            testnet: true,
            base_url: "https://demo-api.binance.com/api".to_string(),
            ws_url: "wss://demo-stream.binance.com/ws".to_string(),
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "X-MBX-APIKEY",
            reqwest::header::HeaderValue::from_str(&self.api_key).unwrap(),
        );
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

    fn signed_query(&self, params: &[(&str, &str)]) -> String {
        let timestamp = chrono::Utc::now().timestamp_millis();
        let ts_str = timestamp.to_string();
        let mut pairs = params.to_vec();
        pairs.push(("timestamp", &ts_str));
        pairs.push(("recvWindow", "5000"));

        let query_string: String = pairs
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&");

        let sig = self.signature(&query_string);
        format!("{}&signature={}", query_string, sig)
    }

    /// Parse a string price into f64, defaulting to 0.0 on error
    fn parse_f64(s: &str) -> f64 {
        s.parse::<f64>().unwrap_or(0.0)
    }
}

// ── Binance Account/REST response helpers ──────────────────

fn find_min_notional(filters: &[serde_json::Value]) -> f64 {
    for f in filters {
        if let Some(filter_type) = f.get("filterType").and_then(|v| v.as_str()) {
            if filter_type == "MIN_NOTIONAL" {
                if let Some(v) = f.get("minNotional").and_then(|v| v.as_str()) {
                    return v.parse().unwrap_or(10.0);
                }
            }
        }
    }
    10.0
}

fn find_filter_value(filters: &[serde_json::Value], filter_type: &str, key: &str) -> f64 {
    for f in filters {
        if let Some(ft) = f.get("filterType").and_then(|v| v.as_str()) {
            if ft == filter_type {
                if let Some(v) = f.get(key).and_then(|v| v.as_str()) {
                    return v.parse().unwrap_or(0.0);
                }
            }
        }
    }
    0.0
}

#[async_trait]
impl ExchangeConnector for BinanceConnector {
    async fn get_account(&self) -> Result<AccountInfo> {
        let query = self.signed_query(&[]);
        let url = format!("{}/v3/account?{}", self.base_url, query);

        let resp = self
            .client
            .get(&url)
            .headers(self.build_headers())
            .send()
            .await
            .context("Failed to call Binance account endpoint")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Binance account API error {}: {}", status, text);
        }

        let account: BinanceAccount = resp.json().await?;

        // Find USDT balance
        let usdt_balance: f64 = account
            .balances
            .iter()
            .find(|b| b.asset == "USDT")
            .map(|b| Self::parse_f64(&b.free) + Self::parse_f64(&b.locked))
            .unwrap_or(0.0);

        tracing::info!("💰 Account: ${:.2} USDT balance", usdt_balance);

        Ok(AccountInfo {
            balance: usdt_balance,
            equity: usdt_balance,
            free_collateral: Self::parse_f64(
                &account
                    .balances
                    .iter()
                    .find(|b| b.asset == "USDT")
                    .map(|b| &b.free)
                    .unwrap_or(&"0".to_string()),
            ),
            unrealized_pnl: 0.0,
            margin_ratio: 0.0,
        })
    }

    async fn place_order(&self, order: &Order) -> Result<String> {
        let side = match order.side {
            OrderSide::Buy => "BUY",
            OrderSide::Sell => "SELL",
        };
        let order_type = match order.order_type {
            OrderType::Market => "MARKET",
            OrderType::Limit => "LIMIT",
            OrderType::Stop | OrderType::StopLimit => "STOP_LOSS_LIMIT",
        };

        let qty_str = order.quantity.to_string();
        let _qty_str = order.quantity.to_string();
        let price_opt = order.price.map(|p| p.to_string());
        let mut params: Vec<(&str, &str)> = vec![
            ("symbol", order.symbol.as_str()),
            ("side", side),
            ("type", order_type),
            ("quantity", _qty_str.as_str()),
        ];

        if let Some(ref price_str) = price_opt {
            params.push(("price", price_str.as_str()));
            params.push(("timeInForce", "GTC"));
        }

        let query = self.signed_query(&params);
        let url = format!("{}/v3/order?{}", self.base_url, query);

        tracing::info!(
            "📤 Order: {} {} {} qty={} lev={}x",
            order.symbol, side, order_type, order.quantity, order.leverage,
        );

        let resp = self
            .client
            .post(&url)
            .headers(self.build_headers())
            .send()
            .await
            .context("Failed to place order on Binance")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Binance order error {}: {}", status, text);
        }

        let result: BinanceOrderResult = resp.json().await?;
        let order_id = result
            .order_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        tracing::info!("✅ Order placed: ID={}", order_id);
        Ok(order_id)
    }

    async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<bool> {
        let query = self.signed_query(&[
            ("symbol", symbol),
            ("orderId", order_id),
        ]);
        let url = format!("{}/v3/order?{}", self.base_url, query);

        let resp = self
            .client
            .delete(&url)
            .headers(self.build_headers())
            .send()
            .await
            .context("Failed to cancel order")?;

        Ok(resp.status().is_success())
    }

    async fn get_order(&self, symbol: &str, order_id: &str) -> Result<Order> {
        let query = self.signed_query(&[
            ("symbol", symbol),
            ("orderId", order_id),
        ]);
        let url = format!("{}/v3/order?{}", self.base_url, query);

        let resp = self
            .client
            .get(&url)
            .headers(self.build_headers())
            .send()
            .await
            .context("Failed to get order status")?;

        if !resp.status().is_success() {
            anyhow::bail!("Order not found: {}", order_id);
        }

        let result: BinanceOrderResult = resp.json().await?;

        let status = match result.status.as_deref() {
            Some("NEW") | Some("PARTIALLY_FILLED") => crate::order::OrderStatus::Open,
            Some("FILLED") => crate::order::OrderStatus::Filled,
            Some("CANCELED") | Some("EXPIRED") | Some("REJECTED") => crate::order::OrderStatus::Rejected,
            _ => crate::order::OrderStatus::Open,
        };

        let side = match result.side.as_deref() {
            Some("BUY") => OrderSide::Buy,
            _ => OrderSide::Sell,
        };

        Ok(Order {
            id: order_id.to_string(),
            symbol: symbol.to_string(),
            side,
            order_type: OrderType::Market,
            quantity: result.executed_qty.as_deref().and_then(|q| q.parse().ok()).unwrap_or(0.0),
            filled_quantity: result.executed_qty.as_deref().and_then(|q| q.parse().ok()).unwrap_or(0.0),
            price: result.cumm_quote_qty.as_deref().and_then(|q| q.parse().ok()),
            stop_price: None,
            leverage: 1,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            status,
            exchange_order_id: Some(order_id.to_string()),
            reduce_only: false,
            post_only: false,
            stop_loss: None,
            take_profit: None,
        })
    }

    async fn get_open_orders(&self, _symbol: &str) -> Result<Vec<Order>> {
        let query = self.signed_query(&[]);
        let url = format!("{}/v3/openOrders?{}", self.base_url, query);

        let resp = self
            .client
            .get(&url)
            .headers(self.build_headers())
            .send()
            .await
            .context("Failed to get open orders")?;

        let orders: Vec<serde_json::Value> = resp.json().await?;
        Ok(orders
            .into_iter()
            .map(|o| {
                let order_id = o.get("orderId").and_then(|v| v.as_i64()).map(|i| i.to_string());
                let qty = o.get("origQty").and_then(|v| v.as_str()).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let price = o.get("price").and_then(|v| v.as_str()).and_then(|s| s.parse().ok());
                Order {
                    id: order_id.clone().unwrap_or_default(),
                    symbol: o.get("symbol").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    side: if o.get("side").and_then(|v| v.as_str()) == Some("BUY") { OrderSide::Buy } else { OrderSide::Sell },
                    order_type: OrderType::Market,
                    quantity: qty,
                    filled_quantity: o.get("executedQty").and_then(|v| v.as_str()).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                    price,
                    stop_price: None,
                    leverage: 1,
                    timestamp: o.get("time").and_then(|v| v.as_i64()).unwrap_or(0) as u64,
                    status: crate::order::OrderStatus::Open,
                    exchange_order_id: order_id,
                    reduce_only: false,
                    post_only: false,
                    stop_loss: None,
                    take_profit: None,
                }
            })
            .collect())
    }

    async fn get_position(&self, _symbol: &str) -> Result<Option<Position>> {
        // Binance testnet spot doesn't have positions in futures sense
        // For futures testnet, we'd query /fapi/v2/positionRisk
        // For now, return None (positions managed by Python paper account)
        Ok(None)
    }

    async fn get_positions(&self) -> Result<Vec<Position>> {
        Ok(Vec::new())
    }

    async fn get_ticker(&self, symbol: &str) -> Result<MarketData> {
        let url = format!("{}/v3/ticker/24hr?symbol={}", self.base_url, symbol);

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch ticker from Binance")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Binance ticker error {}: {}", status, text);
        }

        let ticker: BinanceTicker = resp.json().await?;

        let bid = Self::parse_f64(&ticker.bid_price);
        let ask = Self::parse_f64(&ticker.ask_price);
        let last = Self::parse_f64(&ticker.last_price);

        tracing::info!("📊 {}: bid={:.2} ask={:.2} last={:.2}", ticker.symbol, bid, ask, last);

        Ok(MarketData {
            symbol: ticker.symbol,
            bid,
            ask,
            last,
            volume_24h: Self::parse_f64(&ticker.volume),
            high_24h: Self::parse_f64(&ticker.high_price),
            low_24h: Self::parse_f64(&ticker.low_price),
            timestamp: ticker.close_time,
        })
    }

    async fn get_order_book(&self, symbol: &str, limit: u32) -> Result<OrderBook> {
        let url = format!(
            "{}/v3/depth?symbol={}&limit={}",
            self.base_url, symbol, limit.min(100)
        );

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch order book")?;

        let depth: BinanceDepth = resp.json().await?;

        Ok(OrderBook {
            symbol: symbol.to_string(),
            bids: depth
                .bids
                .into_iter()
                .map(|b| OrderBookLevel {
                    price: Self::parse_f64(&b[0]),
                    quantity: Self::parse_f64(&b[1]),
                })
                .collect(),
            asks: depth
                .asks
                .into_iter()
                .map(|a| OrderBookLevel {
                    price: Self::parse_f64(&a[0]),
                    quantity: Self::parse_f64(&a[1]),
                })
                .collect(),
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        })
    }

    async fn get_klines(&self, symbol: &str, interval: &str, limit: u32) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v3/klines?symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit.min(1500)
        );

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch klines")?;

        let raw: Vec<BinanceKlineRaw> = resp.json().await?;

        Ok(raw
            .into_iter()
            .map(|k| Kline {
                symbol: symbol.to_string(),
                open: Self::parse_f64(&k.1),
                high: Self::parse_f64(&k.2),
                low: Self::parse_f64(&k.3),
                close: Self::parse_f64(&k.4),
                volume: Self::parse_f64(&k.5),
                close_time: k.6,
                trades: k.8,
            })
            .collect())
    }

    async fn get_exchange_info(&self) -> Result<HashMap<String, ExchangeSymbol>> {
        let url = format!("{}/v3/exchangeInfo", self.base_url);

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch exchange info")?;

        let info: BinanceExchangeInfo = resp.json().await?;

        let mut map = HashMap::new();
        for s in info.symbols {
            if s.is_spot_trading_allowed.unwrap_or(false) {
                map.insert(
                    s.symbol.clone(),
                    ExchangeSymbol {
                        symbol: s.symbol,
                        base_asset: s.base_asset,
                        quote_asset: s.quote_asset,
                        min_notional: find_min_notional(&s.filters),
                        min_qty: find_filter_value(&s.filters, "LOT_SIZE", "minQty"),
                        tick_size: find_filter_value(&s.filters, "PRICE_FILTER", "tickSize"),
                        step_size: find_filter_value(&s.filters, "LOT_SIZE", "stepSize"),
                        leverage_max: 10,
                    },
                );
            }
        }

        tracing::info!("📋 Exchange info: {} trading pairs loaded", map.len());
        Ok(map)
    }

    /// Place OCO (One-Cancels-Other) bracket order on Binance Testnet.
    /// Uses the /v3/order/oco endpoint with STOP_LOSS_LIMIT + LIMIT_MAKER.
    async fn place_oco_order(
        &self,
        symbol: &str,
        stop_loss_price: f64,
        take_profit_price: f64,
        quantity: f64,
    ) -> Result<String> {
        let url = format!("{}/v3/order/oco", self.base_url);

        let params = format!(
            "symbol={}&side=SELL&quantity={}&price={}&stopPrice={}&stopLimitPrice={}&stopLimitTimeInForce=GTC&type=OCO&timestamp={}",
            symbol.replace("/", ""),
            format_f64_prec(quantity, 8),
            format_f64_prec(take_profit_price, 8),
            format_f64_prec(stop_loss_price, 8),
            format_f64_prec(stop_loss_price, 8),
            chrono::Utc::now().timestamp_millis()
        );

        let signature = self.signature(&params);
        let body = format!("{}&signature={}", params, signature);

        let resp = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .context("OCO order request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("OCO order rejected ({}): {}", status, text);
        }

        let result: serde_json::Value = resp.json().await?;
        let order_list_id = result["orderListId"]
            .as_i64()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        tracing::info!(
            "🎯 OCO order placed: {} SL={:.2} TP={:.2} listId={}",
            symbol, stop_loss_price, take_profit_price, order_list_id
        );

        Ok(order_list_id)
    }
}

impl std::fmt::Debug for BinanceConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinanceConnector")
            .field("testnet", &self.testnet)
            .field("base_url", &self.base_url)
            .finish()
    }
}
