use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod order;
mod risk;
mod exchange;
mod engine;
mod ws;
mod nats_subscriber;

use engine::TradingEngine;
use nats_subscriber::NATSSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("quantex_execution=info".parse()?))
        .init();

    tracing::info!("🚀 QUANTEX Execution Engine (Trinity Layer C) starting...");

    let engine = TradingEngine::new().await?;

    // Start NATS subscriber (signals.aggregated → execute trades)
    let subscriber = NATSSubscriber::new(engine.clone());
    tokio::spawn(async move {
        if let Err(e) = subscriber.run().await {
            tracing::error!("NATS subscriber error: {}", e);
        }
    });

    // Start WebSocket order book feed
    let symbols: Vec<String> = std::env::var("TRADE_SYMBOLS")
        .unwrap_or_else(|_| "BTC/USDT".to_string())
        .split(',')
        .map(|s| s.trim().replace("/", "").to_lowercase())
        .collect();

    let ws_url = std::env::var("BINANCE_WS_URL")
        .unwrap_or_else(|_| "wss://testnet.binance.vision/ws".to_string());

    let ws_engine = ws::WebSocketEngine::new(ws_url, symbols);
    tokio::spawn(async move {
        ws_engine.run().await;
    });

    tracing::info!("✅ QUANTEX Execution Engine ready — awaiting signals on NATS");
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down...");
    Ok(())
}
