//! Sentinel-X Risk Engine — gRPC server over Unix Domain Socket
mod risk;
mod sbe;
mod backtest;

use anyhow::Result;
use prometheus::{Encoder, TextEncoder};
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
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

    let uds_path = std::env::var("RISK_UDS_PATH")
        .unwrap_or_else(|_| "/tmp/sentinel-risk.sock".into());

    // Clean up stale socket
    let _ = std::fs::remove_file(&uds_path);

    let uds = UnixListener::bind(&uds_path)?;
    let stream = UnixListenerStream::new(uds);

    info!("Sentinel-X Risk Engine listening on UDS: {}", uds_path);

    let svc = RiskEngineService::new();

    // Prometheus metrics endpoint (TCP for scraping)
    tokio::spawn(async {
        let registry = prometheus::default_registry();
        let encoder  = TextEncoder::new();
        let listener = tokio::net::TcpListener::bind("0.0.0.0:9091").await.unwrap();
        info!("Risk engine metrics on :9091/metrics");
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
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
    });

    Server::builder()
        .add_service(proto::risk_engine_server::RiskEngineServer::new(svc))
        .serve_with_incoming(stream)
        .await?;

    Ok(())
}
