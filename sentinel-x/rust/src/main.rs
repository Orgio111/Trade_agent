//! Sentinel-X Risk Engine — gRPC server with CLI backtest mode.
//!
//! Normal mode: gRPC server (default port 50051).
//! Backtest mode: ``sentinel-risk --backtest < json_input.json`` reads
//! OHLCV data from stdin (JSON) and outputs backtest results as JSON.
//!
//! Backtest JSON input format:
//! .. code-block:: json
//!
//!     {
//!       "closes": [65000, 65100, ...],
//!       "highs":  [65200, 65250, ...],
//!       "lows":   [64800, 64900, ...],
//!       "volumes": [1.2, 1.5, ...],
//!       "initial_equity": 100000.0,
//!       "kelly_fraction": 0.25,
//!       "atr_multiplier": 2.0,
//!       "atr_period": 14
//!     }

#![allow(unused_imports)]

mod backtest;
mod risk;
mod sbe;

use std::io::Read;

use anyhow::Result;
use prometheus::{Encoder, TextEncoder};
use serde::Deserialize;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use backtest::engine::{BacktestEngine, BacktestResult};
use risk::engine::RiskEngineService;

pub mod proto {
    tonic::include_proto!("sentinelx.risk");
}

#[derive(Deserialize)]
struct BacktestInput {
    closes:         Vec<f64>,
    highs:          Vec<f64>,
    lows:           Vec<f64>,
    volumes:        Vec<f64>,
    #[serde(default = "default_equity")]
    initial_equity: f64,
    #[serde(default = "default_kelly")]
    kelly_fraction: f64,
    #[serde(default = "default_atr_mult")]
    atr_multiplier: f64,
    #[serde(default = "default_atr_period")]
    atr_period:     usize,
}

fn default_equity() -> f64 { 100_000.0 }
fn default_kelly() -> f64 { 0.25 }
fn default_atr_mult() -> f64 { 2.0 }
fn default_atr_period() -> usize { 14 }

fn run_cli_backtest() -> Result<()> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let input: BacktestInput = serde_json::from_str(&buf)?;

    let n = input.closes.len()
        .min(input.highs.len())
        .min(input.lows.len())
        .min(input.volumes.len());
    if n < 2 {
        let empty = BacktestResult::default();
        println!("{}", serde_json::to_string_pretty(&empty)?);
        return Ok(());
    }

    let bars: Vec<backtest::engine::Bar> = (0..n)
        .map(|i| backtest::engine::Bar {
            close:  input.closes[i],
            high:   input.highs[i],
            low:    input.lows[i],
            volume: input.volumes[i],
        })
        .collect();

    let engine = BacktestEngine::new(
        input.initial_equity.max(1_000.0),
        input.atr_period.max(1),
        input.atr_multiplier.max(0.5),
        input.kelly_fraction.max(0.01).min(1.0),
    );

    let result = engine.run(&bars);
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── CLI backtest mode ──────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--backtest" {
        return run_cli_backtest();
    }
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let tcp_addr = std::env::var("RISK_TCP_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50051".into());

    let tcp_listener = tokio::net::TcpListener::bind(&tcp_addr).await?;
    let tcp_incoming = TcpListenerStream::new(tcp_listener);

    info!("Sentinel-X Risk Engine listening on TCP: {}", tcp_addr);

    // ── Start Prometheus metrics HTTP server ──────────────────────────────
    start_metrics_server();

    let svc = RiskEngineService::new();

    Server::builder()
        .add_service(proto::risk_engine_server::RiskEngineServer::new(svc))
        .serve_with_incoming(tcp_incoming)
        .await?;

    Ok(())
}

/// Spawn a minimal HTTP server for Prometheus metrics scraping.
/// Binds to the port specified by RISK_METRICS_ADDR (default :9180).
fn start_metrics_server() {
    let metrics_addr = std::env::var("RISK_METRICS_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9180".into());
    let metrics_addr_clone = metrics_addr.clone();
    tokio::spawn(async move {
        let registry = prometheus::default_registry();
        let encoder = TextEncoder::new();
        let metrics_listener = tokio::net::TcpListener::bind(&metrics_addr_clone)
            .await
            .expect(&format!("Failed to bind metrics on {}", metrics_addr_clone));
        info!("Risk engine metrics on {}/metrics", metrics_addr_clone);
        loop {
            if let Ok((mut stream, _)) = metrics_listener.accept().await {
                let mut buffer = vec![];
                encoder.encode(&registry.gather(), &mut buffer).unwrap();
                let body = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n",
                    buffer.len()
                );
                use tokio::io::AsyncWriteExt;
                let _ = stream.write_all(body.as_bytes()).await;
                let _ = stream.write_all(&buffer).await;
            }
        }
    });
}
