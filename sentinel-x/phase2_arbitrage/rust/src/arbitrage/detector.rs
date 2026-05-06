//! Real-time arbitrage opportunity detector.
//! Processes order book updates from all exchanges in < 1μs per check.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, register_counter,
                 register_gauge, register_histogram};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use tracing::{info, warn};

use crate::exchange::interface::{BestQuote, ExchangeId, OrderBook};

/// Minimum net profit after ALL fees to consider an arb trade.
const MIN_NET_PROFIT_BPS: f64 = 2.0;  // 0.02% minimum edge
/// Maximum age of a quote before it's considered stale.
const QUOTE_STALE_NS: u64 = 200_000_000;  // 200ms

#[derive(Debug, Clone)]
pub struct ArbOpportunity {
    pub id:          String,
    pub arb_type:    ArbType,
    pub buy_exchange:  ExchangeId,
    pub sell_exchange: ExchangeId,
    pub symbol:      String,
    pub buy_price:   f64,
    pub sell_price:  f64,
    pub gross_spread_bps: f64,
    pub estimated_fees_bps: f64,
    pub net_profit_bps:  f64,
    pub max_qty:     f64,       // limited by order book depth
    pub estimated_pnl_usd: f64,
    pub detected_ns: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArbType {
    Spatial,      // price difference across exchanges
    Triangular,   // A→B→C→A cycle on one exchange
}

pub struct ArbDetector {
    /// Live best quotes: symbol → (exchange → BestQuote)
    quotes: Arc<DashMap<String, DashMap<ExchangeId, BestQuote>>>,
    /// Order books for slippage-aware sizing: exchange+symbol → OrderBook
    books:  Arc<DashMap<(ExchangeId, String), OrderBook>>,

    // Metrics
    opps_detected: Counter,
    opps_executed: Counter,
    profit_gauge:  Gauge,
    detect_latency: Histogram,
}

impl ArbDetector {
    pub fn new() -> Self {
        let opps_detected = register_counter!(
            "arb_opportunities_detected_total", "Arbitrage opportunities detected"
        ).unwrap();
        let opps_executed = register_counter!(
            "arb_opportunities_executed_total", "Arbitrage opportunities executed"
        ).unwrap();
        let profit_gauge = register_gauge!(
            "arb_cumulative_pnl_usd", "Cumulative arbitrage PnL in USD"
        ).unwrap();
        let detect_latency = register_histogram!(
            HistogramOpts::new("arb_detection_latency_ns", "Detection latency in nanoseconds")
                .buckets(vec![100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0])
        ).unwrap();

        Self {
            quotes: Arc::new(DashMap::new()),
            books:  Arc::new(DashMap::new()),
            opps_detected,
            opps_executed,
            profit_gauge,
            detect_latency,
        }
    }

    /// Update the best quote for a symbol on an exchange.
    /// Called from the WebSocket feed handler — must be O(1).
    #[inline(always)]
    pub fn update_quote(&self, quote: BestQuote) {
        let sym = std::str::from_utf8(&quote.symbol)
            .unwrap_or("")
            .trim_end_matches('\0')
            .to_string();

        self.quotes
            .entry(sym)
            .or_insert_with(DashMap::new)
            .insert(quote.exchange, quote);
    }

    /// Check all symbols for spatial arbitrage. O(n_symbols × n_exchange_pairs).
    /// Returns all profitable opportunities above MIN_NET_PROFIT_BPS.
    pub fn scan_spatial(&self) -> Vec<ArbOpportunity> {
        let t0 = nanos_now();
        let mut opportunities = Vec::new();
        let exchange_pairs = [
            (ExchangeId::Binance, ExchangeId::Bybit),
            (ExchangeId::Binance, ExchangeId::OKX),
            (ExchangeId::Bybit,   ExchangeId::OKX),
        ];

        for entry in self.quotes.iter() {
            let symbol = entry.key().clone();
            let exchange_quotes = entry.value();

            for (ex_a, ex_b) in &exchange_pairs {
                if let (Some(q_a), Some(q_b)) = (
                    exchange_quotes.get(ex_a),
                    exchange_quotes.get(ex_b),
                ) {
                    let now = nanos_now();
                    // Skip stale quotes
                    if now - q_a.ts_ns > QUOTE_STALE_NS || now - q_b.ts_ns > QUOTE_STALE_NS {
                        continue;
                    }

                    // Buy on A, sell on B
                    if let Some(opp) = self.check_spatial_pair(
                        &symbol, *ex_a, &q_a, *ex_b, &q_b
                    ) {
                        opportunities.push(opp);
                    }
                    // Buy on B, sell on A
                    if let Some(opp) = self.check_spatial_pair(
                        &symbol, *ex_b, &q_b, *ex_a, &q_a
                    ) {
                        opportunities.push(opp);
                    }
                }
            }
        }

        let latency_ns = nanos_now() - t0;
        self.detect_latency.observe(latency_ns as f64);

        if !opportunities.is_empty() {
            self.opps_detected.inc_by(opportunities.len() as f64);
        }
        opportunities
    }

    fn check_spatial_pair(
        &self,
        symbol: &str,
        buy_ex: ExchangeId,
        buy_q: &BestQuote,
        sell_ex: ExchangeId,
        sell_q: &BestQuote,
    ) -> Option<ArbOpportunity> {
        let buy_price  = buy_q.ask.to_f64()?;
        let sell_price = sell_q.bid.to_f64()?;
        let buy_qty    = buy_q.ask_qty.to_f64()?;
        let sell_qty   = sell_q.bid_qty.to_f64()?;

        if sell_price <= buy_price { return None; }

        let gross_spread_bps = (sell_price - buy_price) / buy_price * 10_000.0;
        let fees_bps = (buy_ex.taker_fee() + sell_ex.taker_fee()).to_f64()? * 10_000.0;
        let net_profit_bps = gross_spread_bps - fees_bps;

        if net_profit_bps < MIN_NET_PROFIT_BPS { return None; }

        let max_qty = buy_qty.min(sell_qty);
        let pnl_usd = max_qty * buy_price * (net_profit_bps / 10_000.0);

        Some(ArbOpportunity {
            id:            uuid::Uuid::new_v4().to_string(),
            arb_type:      ArbType::Spatial,
            buy_exchange:  buy_ex,
            sell_exchange: sell_ex,
            symbol:        symbol.to_string(),
            buy_price,
            sell_price,
            gross_spread_bps,
            estimated_fees_bps: fees_bps,
            net_profit_bps,
            max_qty,
            estimated_pnl_usd: pnl_usd,
            detected_ns:   nanos_now(),
        })
    }
}

#[inline(always)]
pub fn nanos_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}
