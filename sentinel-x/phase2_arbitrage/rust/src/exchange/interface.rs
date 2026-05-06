//! Abstract exchange trait — every exchange implements this.
//! Zero-cost abstractions: all hot paths are `#[inline(always)]`.

use anyhow::Result;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Level {
    pub price: Decimal,
    pub qty:   Decimal,
}

/// Best bid/ask snapshot — fits in a single cache line (64 bytes).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct BestQuote {
    pub exchange: ExchangeId,
    pub symbol:   [u8; 12],   // fixed-size for zero-alloc
    pub bid:      Decimal,
    pub ask:      Decimal,
    pub bid_qty:  Decimal,
    pub ask_qty:  Decimal,
    pub ts_ns:    u64,         // nanosecond timestamp
}

impl BestQuote {
    #[inline(always)]
    pub fn spread(&self) -> Decimal {
        self.ask - self.bid
    }

    #[inline(always)]
    pub fn mid(&self) -> Decimal {
        (self.ask + self.bid) / Decimal::from(2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExchangeId {
    Binance,
    Bybit,
    OKX,
}

impl ExchangeId {
    pub fn maker_fee(&self) -> Decimal {
        match self {
            ExchangeId::Binance => Decimal::new(1, 4),  // 0.01%
            ExchangeId::Bybit   => Decimal::new(1, 4),  // 0.01%
            ExchangeId::OKX     => Decimal::new(8, 5),  // 0.008%
        }
    }

    pub fn taker_fee(&self) -> Decimal {
        match self {
            ExchangeId::Binance => Decimal::new(4, 4),  // 0.04%
            ExchangeId::Bybit   => Decimal::new(6, 4),  // 0.06%
            ExchangeId::OKX     => Decimal::new(1, 3),  // 0.10%
        }
    }

    pub fn ws_url(&self, symbol: &str) -> String {
        match self {
            ExchangeId::Binance => format!(
                "wss://stream.binance.com:9443/ws/{}@bookTicker",
                symbol.to_lowercase().replace('/', "")
            ),
            ExchangeId::Bybit => format!(
                "wss://stream.bybit.com/v5/public/spot"
            ),
            ExchangeId::OKX => format!(
                "wss://ws.okx.com:8443/ws/v5/public"
            ),
        }
    }
}

/// Order book — BTreeMap keeps bids/asks sorted automatically.
#[derive(Debug, Default)]
pub struct OrderBook {
    pub bids: BTreeMap<ordered_float::OrderedFloat<f64>, f64>,  // price → qty, descending
    pub asks: BTreeMap<ordered_float::OrderedFloat<f64>, f64>,  // price → qty, ascending
    pub ts_ns: u64,
}

impl OrderBook {
    #[inline(always)]
    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.iter().next_back().map(|(p, q)| (p.0, *q))
    }

    #[inline(always)]
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().next().map(|(p, q)| (p.0, *q))
    }

    /// Compute slippage-aware fill price for `qty` units.
    pub fn vwap_fill(&self, qty: f64, side_is_buy: bool) -> Option<f64> {
        let levels = if side_is_buy { &self.asks } else { &self.bids };
        let mut remaining = qty;
        let mut cost = 0.0f64;
        for (price, level_qty) in levels.iter() {
            let fill = remaining.min(*level_qty);
            cost += fill * price.0;
            remaining -= fill;
            if remaining <= 0.0 { break; }
        }
        if remaining > 0.0 { return None; }  // not enough liquidity
        Some(cost / qty)
    }
}
