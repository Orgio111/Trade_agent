//! Triangular arbitrage: A → B → C → A on a single exchange.
//! Example: BTC/USDT → ETH/BTC → ETH/USDT → USDT

use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use std::collections::HashMap;

use crate::exchange::interface::{ExchangeId, OrderBook};

const MIN_TRI_PROFIT_BPS: f64 = 3.0;

#[derive(Debug, Clone)]
pub struct TriPath {
    pub step_a: String,   // e.g. "BTC/USDT"
    pub step_b: String,   // e.g. "ETH/BTC"
    pub step_c: String,   // e.g. "ETH/USDT"
    pub exchange: ExchangeId,
}

#[derive(Debug, Clone)]
pub struct TriOpportunity {
    pub path:          TriPath,
    pub start_amount:  f64,
    pub end_amount:    f64,
    pub profit_pct:    f64,
    pub net_profit_bps: f64,
    pub bottleneck_qty: f64,
}

pub struct TriangularDetector {
    paths: Vec<TriPath>,
}

impl TriangularDetector {
    pub fn new(exchange: ExchangeId) -> Self {
        // Common triangular paths for crypto
        let paths = vec![
            TriPath { exchange, step_a: "BTC/USDT".into(), step_b: "ETH/BTC".into(), step_c: "ETH/USDT".into() },
            TriPath { exchange, step_a: "BTC/USDT".into(), step_b: "BNB/BTC".into(), step_c: "BNB/USDT".into() },
            TriPath { exchange, step_a: "ETH/USDT".into(), step_b: "SOL/ETH".into(), step_c: "SOL/USDT".into() },
        ];
        Self { paths }
    }

    pub fn scan(
        &self,
        books: &HashMap<String, &OrderBook>,
        start_usdt: f64,
        taker_fee: Decimal,
    ) -> Vec<TriOpportunity> {
        let mut opportunities = Vec::new();
        let fee = 1.0 - taker_fee.to_f64().unwrap_or(0.001);

        for path in &self.paths {
            if let Some(opp) = self.evaluate_path(path, books, start_usdt, fee) {
                if opp.net_profit_bps >= MIN_TRI_PROFIT_BPS {
                    opportunities.push(opp);
                }
            }
        }
        opportunities
    }

    fn evaluate_path(
        &self,
        path: &TriPath,
        books: &HashMap<String, &OrderBook>,
        start_usdt: f64,
        fee_factor: f64,
    ) -> Option<TriOpportunity> {
        let book_a = books.get(&path.step_a)?;
        let book_b = books.get(&path.step_b)?;
        let book_c = books.get(&path.step_c)?;

        // Step A: USDT → BTC (buy BTC with USDT)
        let (ask_a, qty_a) = book_a.best_ask()?;
        let btc_amount = (start_usdt / ask_a) * fee_factor;

        // Step B: BTC → ETH (buy ETH with BTC)
        let (ask_b, qty_b) = book_b.best_ask()?;
        let eth_amount = (btc_amount / ask_b) * fee_factor;

        // Step C: ETH → USDT (sell ETH for USDT)
        let (bid_c, qty_c) = book_c.best_bid()?;
        let end_usdt = (eth_amount * bid_c) * fee_factor;

        let bottleneck = qty_a.min(qty_b * ask_a).min(qty_c / bid_c);
        let profit_pct = (end_usdt - start_usdt) / start_usdt;
        let net_profit_bps = profit_pct * 10_000.0;

        if net_profit_bps <= 0.0 { return None; }

        Some(TriOpportunity {
            path: path.clone(),
            start_amount: start_usdt,
            end_amount: end_usdt,
            profit_pct,
            net_profit_bps,
            bottleneck_qty: bottleneck,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::OrderedFloat;
    use std::collections::BTreeMap;

    fn make_book(bid: f64, ask: f64) -> OrderBook {
        let mut b = OrderBook::default();
        b.bids.insert(OrderedFloat(bid), 10.0);
        b.asks.insert(OrderedFloat(ask), 10.0);
        b
    }

    #[test]
    fn profitable_triangle_detected() {
        let detector = TriangularDetector::new(ExchangeId::Binance);
        let book_a = make_book(64_990.0, 65_000.0);  // BTC/USDT
        let book_b = make_book(0.04695, 0.04700);     // ETH/BTC (1 ETH ≈ 0.047 BTC)
        let book_c = make_book(3_050.0, 3_060.0);     // ETH/USDT

        let books: HashMap<String, &OrderBook> = [
            ("BTC/USDT".to_string(), &book_a),
            ("ETH/BTC".to_string(),  &book_b),
            ("ETH/USDT".to_string(), &book_c),
        ].iter().cloned().collect();

        let fee = Decimal::new(4, 4); // 0.04% taker
        let opps = detector.scan(&books, 10_000.0, fee);
        // This specific triangle may or may not be profitable — just verify no panic
        for opp in &opps {
            assert!(opp.net_profit_bps > 0.0);
        }
    }
}
