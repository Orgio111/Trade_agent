use std::sync::Arc;
use dashmap::DashMap;
use anyhow::Result;

use crate::order::{Order, OrderStatus, Position};
use crate::risk::{RiskEngine, RiskParams};
use crate::exchange::{ExchangeConnector, BinanceConnector};

pub struct TradingEngine {
    pub positions: Arc<DashMap<String, Position>>,
    pub orders: Arc<DashMap<String, Order>>,
    pub risk: RiskEngine,
    pub exchange: Arc<dyn ExchangeConnector>,
    pub balance: Arc<tokio::sync::RwLock<f64>>,
    pub consecutive_losses: Arc<tokio::sync::RwLock<u32>>,
}

impl TradingEngine {
    pub async fn new() -> Result<Self> {
        let exchange = Arc::new(BinanceConnector::new(
            std::env::var("BINANCE_API_KEY").unwrap_or_default(),
            std::env::var("BINANCE_SECRET").unwrap_or_default(),
        ));

        Ok(Self {
            positions: Arc::new(DashMap::new()),
            orders: Arc::new(DashMap::new()),
            risk: RiskEngine::new(RiskParams::default()),
            exchange,
            balance: Arc::new(tokio::sync::RwLock::new(10.0)),
            consecutive_losses: Arc::new(tokio::sync::RwLock::new(0)),
        })
    }

    pub async fn start_event_loop(&self) {
        tracing::info!("📡 Event loop started");
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            self.update_positions().await;
        }
    }

    async fn update_positions(&self) {
        for mut pos in self.positions.iter_mut() {
            // In Phase 1, use mock price updates
            let entry = pos.entry_price;
            pos.update_price(entry * (1.0 + 0.001)); // Simulate slight movement
        }
    }

    pub async fn execute_order(&self, order: Order, atr: f64, drawdown: f64) -> Result<OrderStatus> {
        let balance = *self.balance.read().await;
        let all_positions: Vec<Position> = self.positions.iter().map(|p| p.clone()).collect();
        let consec_losses = *self.consecutive_losses.read().await;

        // Risk check
        let risk_check = self.risk.check_trade(
            &order,
            &self.exchange.get_account().await?,
            &all_positions,
            consec_losses,
            drawdown,
        );

        if !risk_check.approved {
            tracing::warn!("⛔ Risk veto: {}", risk_check.reason);
            return Ok(OrderStatus::Rejected);
        }

        // Position sizing with risk multiplier
        let size_mult = risk_check.size_multiplier.unwrap_or(1.0);
        let final_size = self.risk.calculate_position_size(balance, atr, order.price.unwrap_or(0.0), order.leverage) * size_mult;

        let final_order = Order {
            quantity: final_size,
            ..order
        };

        // Submit to exchange
        let exchange_id = self.exchange.place_order(&final_order).await?;
        tracing::info!("✅ Order executed: {} (ID: {})", final_order.id, exchange_id);

        // Track position
        if let Some(price) = final_order.price {
            let position = Position::new(
                final_order.symbol.clone(),
                final_order.side.clone(),
                price,
                final_order.quantity,
                final_order.leverage,
            );
            self.positions.insert(position.id.clone(), position);
        }

        let mut order = final_order;
        order.status = OrderStatus::Filled;
        order.exchange_order_id = Some(exchange_id);
        self.orders.insert(order.id.clone(), order);

        Ok(OrderStatus::Filled)
    }

    pub async fn close_position(&self, position_id: &str) -> Result<()> {
        if let Some((_, pos)) = self.positions.remove(position_id) {
            let pnl = pos.unrealized_pnl;
            let mut balance = self.balance.write().await;
            *balance += pnl;
            tracing::info!("🔒 Position closed: {} PnL: ${:.4}", position_id, pnl);

            if pnl < 0.0 {
                let mut losses = self.consecutive_losses.write().await;
                *losses += 1;
            } else {
                let mut losses = self.consecutive_losses.write().await;
                *losses = 0;
            }
        }
        Ok(())
    }

    pub fn get_position(&self, id: &str) -> Option<Position> {
        self.positions.get(id).map(|p| p.clone())
    }

    pub async fn get_balance(&self) -> f64 {
        *self.balance.read().await
    }
}

impl Clone for TradingEngine {
    fn clone(&self) -> Self {
        Self {
            positions: self.positions.clone(),
            orders: self.orders.clone(),
            risk: RiskEngine::new(RiskParams::default()),
            exchange: self.exchange.clone(),
            balance: self.balance.clone(),
            consecutive_losses: self.consecutive_losses.clone(),
        }
    }
}
