use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};
pub struct WebSocketEngine {
    pub url: String,
    pub symbols: Vec<String>,
}

impl WebSocketEngine {
    pub fn new(base_url: String, symbols: Vec<String>) -> Self {
        // Build combined stream URL
        let streams: Vec<String> = symbols.iter()
            .flat_map(|s| {
                vec![
                    format!("{}@kline_1m", s.to_lowercase()),
                    format!("{}@depth20@100ms", s.to_lowercase()),
                    format!("{}@aggTrade", s.to_lowercase()),
                ]
            })
            .collect();
        let stream_str = streams.join("/");
        let url = format!("{}/stream?streams={}", base_url, stream_str);

        Self { url, symbols }
    }

    pub async fn run(&self) {
        tracing::info!("🔌 Connecting to WebSocket: {}", self.url);
        
        loop {
            match connect_async(&self.url).await {
                Ok((ws_stream, _)) => {
                    tracing::info!("✅ WebSocket connected");
                    let (mut write, mut read) = ws_stream.split();

                    // Subscribe message
                    let subscribe = serde_json::json!({
                        "method": "SUBSCRIBE",
                        "params": self.symbols.iter().map(|s| format!("{}@ticker", s.to_lowercase())).collect::<Vec<_>>(),
                        "id": 1
                    });

                    if let Ok(msg) = serde_json::to_string(&subscribe) {
                        let _ = write.send(Message::Text(msg.into())).await;
                    }

                    // Process incoming messages
                    while let Some(msg) = read.next().await {
                        match msg {
                            Ok(Message::Text(text)) => {
                                if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                                    self.process_message(data).await;
                                }
                            }
                            Ok(Message::Close(_)) => {
                                tracing::warn!("WebSocket closed, reconnecting...");
                                break;
                            }
                            Err(e) => {
                                tracing::error!("WebSocket error: {}", e);
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("WebSocket connection failed: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        }
    }

    async fn process_message(&self, data: serde_json::Value) {
        // Route to NATS for other services
        if let Some(stream) = data.get("stream").and_then(|s| s.as_str()) {
            if stream.ends_with("@ticker") {
                if let Some(data) = data.get("data") {
                    tracing::debug!("📊 Ticker: {:?}", data);
                    // Publish to NATS in Phase 2
                }
            }
        }
    }
}
