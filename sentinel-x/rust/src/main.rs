//! Sentinel-X Risk Engine — gRPC server
//!
//! TCP (default port 50051) for the Python bridge and any other client.
//! Sets RISK_UDS_PATH to skip UDS binding on platforms that don't support it.

#![allow(unused_imports)]

mod backtest;
mod risk;
mod sbe;

use anyhow::Result;
use prometheus::{Encoder, TextEncoder};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use risk::engine::RiskEngineService;

pub mod proto {
    tonic::include_proto!("sentinelx.risk");
}

#[tokio::main]
async fn main() -> Result<()> {
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
fn start_metrics_server() {
    tokio::spawn(async {
        let registry = prometheus::default_registry();
        let encoder = TextEncoder::new();
        let metrics_listener = tokio::net::TcpListener::bind("0.0.0.0:9091")
            .await
            .unwrap();
        info!("Risk engine metrics on :9091/metrics");
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
