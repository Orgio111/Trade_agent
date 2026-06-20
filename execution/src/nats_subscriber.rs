//! NATS JetStream subscriber + Dead Man's Switch for the Execution Engine.
//!
//! Subscribes to `signals.aggregated` subject (published by Go Orchestrator).
//! Implements Dead Man's Switch: if no heartbeat within DMS_TIMEOUT_SECS,
//! all new entry orders are halted until heartbeat resumes.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::Deserialize;
use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::engine::TradingEngine;

// ── Configuration ─────────────────────────────────────────────

const NATS_URL: &str = "nats://localhost:4222";
const NATS_SUBJECT: &str = "signals.aggregated";
const DMS_TIMEOUT_SECS: u64 = 3;

// ── Aggregated Signal from Go Orchestrator ─────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct AggregatedSignal {
    pub symbol: String,
    pub action: String,       // "BUY" | "SELL" | "HOLD"
    pub final_score: f64,     // aggregated weighted score
    pub confidence: f64,      // combined confidence
    pub position_size_pct: f64, // recommended position size as % of equity
    pub stop_loss: f64,       // ATR-based stop loss price
    pub take_profit: f64,     // ATR-based take profit price
    pub timestamp_ms: u64,
    pub brain_count: u32,     // how many brains contributed
}

// ── Dead Man's Switch ─────────────────────────────────────────

pub struct DeadMansSwitch {
    /// True = heartbeat received within timeout; entries allowed.
    alive: AtomicBool,
    /// Timestamp (ms) of last heartbeat.
    last_heartbeat_ms: AtomicU64,
    /// Timeout in seconds.
    timeout_secs: u64,
    /// Channel to notify engine when DMS trips/recedes.
    halt_tx: watch::Sender<bool>,
    halt_rx: watch::Receiver<bool>,
}

impl DeadMansSwitch {
    pub fn new(timeout_secs: u64) -> Self {
        let (halt_tx, halt_rx) = watch::channel(false);
        Self {
            alive: AtomicBool::new(true),
            last_heartbeat_ms: AtomicU64::new(0),
            timeout_secs,
            halt_tx,
            halt_rx,
        }
    }

    /// Called when a heartbeat message arrives.
    pub fn heartbeat(&self) {
        let now = chrono_now_ms();
        self.last_heartbeat_ms.store(now, Ordering::SeqCst);
        let was_dead = !self.alive.load(Ordering::SeqCst);
        self.alive.store(true, Ordering::SeqCst);
        if was_dead {
            info!("🟢 Dead Man's Switch: heartbeat resumed — entries re-enabled");
            let _ = self.halt_tx.send(false);
        }
    }

    /// Check if the switch has tripped (no heartbeat within timeout).
    pub fn check(&self) -> bool {
        let now = chrono_now_ms();
        let last = self.last_heartbeat_ms.load(Ordering::SeqCst);
        let elapsed_ms = now.saturating_sub(last);

        if last == 0 {
            // First check — no heartbeat yet, allow a grace period
            return true;
        }

        let tripped = elapsed_ms > self.timeout_secs * 1000;
        if tripped && self.alive.load(Ordering::SeqCst) {
            warn!(
                "🔴 Dead Man's Switch: NO heartbeat for {}ms > {}s — HALTING new entries",
                elapsed_ms,
                self.timeout_secs
            );
            self.alive.store(false, Ordering::SeqCst);
            let _ = self.halt_tx.send(true);
        }
        tripped
    }

    /// Whether new entry orders are allowed right now.
    pub fn entries_allowed(&self) -> bool {
        !self.check()
    }

    /// Get a receiver for halt state changes.
    pub fn halt_rx(&self) -> watch::Receiver<bool> {
        self.halt_rx.clone()
    }
}

fn chrono_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ── NATS Subscriber ───────────────────────────────────────────

pub struct NATSSubscriber {
    nats_url: String,
    subject: String,
    dms: Arc<DeadMansSwitch>,
    engine: TradingEngine,
}

impl NATSSubscriber {
    pub fn new(engine: TradingEngine) -> Self {
        let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| NATS_URL.to_string());
        let subject = std::env::var("NATS_SUBJECT_AGGREGATED")
            .unwrap_or_else(|_| NATS_SUBJECT.to_string());
        let dms_timeout = std::env::var("DMS_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DMS_TIMEOUT_SECS);

        Self {
            nats_url,
            subject,
            dms: Arc::new(DeadMansSwitch::new(dms_timeout)),
            engine,
        }
    }

    /// Main loop: connect to NATS, subscribe, process signals.
    pub async fn run(&self) -> Result<()> {
        info!("📡 Connecting to NATS at {}...", self.nats_url);

        let nc = async_nats::connect(self.nats_url.clone())
            .await
            .context("Failed to connect to NATS")?;

        info!("✅ Connected to NATS, subscribing to {}", self.subject);

        let mut subscriber = nc.subscribe(self.subject.clone()).await?;

        // Spawn DMS checker task
        let dms_checker = self.dms.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                dms_checker.check();
            }
        });

        // Main message loop
        while let Some(msg) = subscriber.next().await {
            self.handle_message(&msg).await;
        }

        Ok(())
    }

    async fn handle_message(&self, msg: &async_nats::Message) {
        // Parse the aggregated signal
        let signal: AggregatedSignal = match serde_json::from_slice(&msg.payload) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to parse aggregated signal: {}", e);
                return;
            }
        };

        // Heartbeat: any message counts as a heartbeat
        self.dms.heartbeat();

        // Check DMS before processing entry signals
        if !self.dms.entries_allowed() {
            warn!(
                "⛔ DMS tripped — ignoring {} signal for {}",
                signal.action, signal.symbol
            );
            return;
        }

        match signal.action.as_str() {
            "BUY" => {
                info!(
                    "📈 BUY signal: {} score={:.4} confidence={:.2} SL={:.2} TP={:.2}",
                    signal.symbol,
                    signal.final_score,
                    signal.confidence,
                    signal.stop_loss,
                    signal.take_profit
                );
                self.execute_buy(&signal).await;
            }
            "SELL" => {
                info!(
                    "📉 SELL signal: {} score={:.4} confidence={:.2} SL={:.2} TP={:.2}",
                    signal.symbol,
                    signal.final_score,
                    signal.confidence,
                    signal.stop_loss,
                    signal.take_profit
                );
                self.execute_sell(&signal).await;
            }
            "HOLD" => {
                tracing::debug!(
                    "⏸️  HOLD signal: {} score={:.4}",
                    signal.symbol,
                    signal.final_score
                );
            }
            other => {
                warn!("Unknown action '{}' for {}", other, signal.symbol);
            }
        }
    }

    async fn execute_buy(&self, signal: &AggregatedSignal) {
        use crate::order::{Order, OrderSide, OrderType};

        let order = Order {
            id: uuid::Uuid::new_v4().to_string(),
            symbol: signal.symbol.clone(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: signal.position_size_pct,
            filled_quantity: 0.0,
            price: Some(signal.stop_loss),
            stop_price: None,
            leverage: 1,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            status: crate::order::OrderStatus::Pending,
            exchange_order_id: None,
            reduce_only: false,
            post_only: false,
            stop_loss: Some(signal.stop_loss),
            take_profit: Some(signal.take_profit),
        };

        match self.engine.execute_order(order, 0.0, 0.0).await {
            Ok(status) => {
                info!("✅ BUY order status: {:?}", status);
                // Place OCO bracket order (SL + TP)
                self.place_bracket_order(signal).await;
            }
            Err(e) => error!("❌ BUY order failed: {}", e),
        }
    }

    async fn execute_sell(&self, signal: &AggregatedSignal) {
        use crate::order::{Order, OrderSide, OrderType};

        let order = Order {
            id: uuid::Uuid::new_v4().to_string(),
            symbol: signal.symbol.clone(),
            side: OrderSide::Sell,
            order_type: OrderType::Market,
            quantity: signal.position_size_pct,
            filled_quantity: 0.0,
            price: Some(signal.take_profit),
            stop_price: None,
            leverage: 1,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            status: crate::order::OrderStatus::Pending,
            exchange_order_id: None,
            reduce_only: false,
            post_only: false,
            stop_loss: Some(signal.stop_loss),
            take_profit: Some(signal.take_profit),
        };

        match self.engine.execute_order(order, 0.0, 0.0).await {
            Ok(status) => {
                info!("✅ SELL order status: {:?}", status);
                self.place_bracket_order(signal).await;
            }
            Err(e) => error!("❌ SELL order failed: {}", e),
        }
    }

    /// Place OCO (One-Cancels-Other) bracket order with SL and TP.
    async fn place_bracket_order(&self, signal: &AggregatedSignal) {
        match self
            .engine
            .exchange
            .place_oco_order(
                &signal.symbol,
                signal.stop_loss,
                signal.take_profit,
                signal.position_size_pct,
            )
            .await
        {
            Ok(order_id) => {
                info!(
                    "🎯 Bracket order placed: {} SL={:.2} TP={:.2} (ID: {})",
                    signal.symbol, signal.stop_loss, signal.take_profit, order_id
                );
            }
            Err(e) => {
                warn!("⚠️  Bracket order failed (may not support OCO): {}", e);
            }
        }
    }
}
