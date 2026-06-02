use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod order;
mod risk;
mod exchange;
mod engine;
mod ws;

use engine::TradingEngine;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("quantex_execution=info".parse()?))
        .init();

    tracing::info!("🚀 QUANTEX Execution Engine starting...");

    let engine = TradingEngine::new().await?;

    // Start NATS event listener
    let engine_clone = engine.clone();
    tokio::spawn(async move {
        engine_clone.start_event_loop().await;
    });

    // Start order book feed
    let ws_engine = ws::WebSocketEngine::new(
        "wss://testnet.binance.vision/ws".to_string(),
        vec!["btcusdt".to_string()],
    );
    tokio::spawn(async move {
        ws_engine.run().await;
    });

    tracing::info!("✅ QUANTEX Execution Engine ready");
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down...");
    Ok(())
}
