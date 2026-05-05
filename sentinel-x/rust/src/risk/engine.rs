//! gRPC service implementation — wires all risk sub-modules together.
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use prometheus::{Counter, Histogram, HistogramOpts, register_counter, register_histogram};
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::proto::{
    risk_engine_server::RiskEngine,
    KillSwitchEvent, KillSwitchRequest,
    RiskRequest, RiskResponse,
};
use crate::risk::{
    correlation::{compute_heat, PortfolioHeatMap},
    kelly::{atr_stop, atr_take_profit, kelly_quarter, position_size_usd},
    kill_switch::KillSwitch,
    var::blended_var,
};
use crate::sbe::encoder::encode_risk_snapshot;

pub struct RiskEngineService {
    kill_switch:        Arc<KillSwitch>,
    validation_counter: Counter,
    rejection_counter:  Counter,
    latency_hist:       Histogram,
    // subscribers for kill-switch events
    ks_subscribers:     Arc<RwLock<Vec<tokio::sync::mpsc::Sender<Result<KillSwitchEvent, Status>>>>>,
}

impl RiskEngineService {
    pub fn new() -> Self {
        let validation_counter = register_counter!(
            "risk_validations_total", "Total risk validation requests"
        ).unwrap();
        let rejection_counter = register_counter!(
            "risk_rejections_total", "Total risk validation rejections"
        ).unwrap();
        let latency_hist = register_histogram!(
            HistogramOpts::new("risk_validation_latency_seconds", "Risk validation latency")
                .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])
        ).unwrap();

        Self {
            kill_switch: Arc::new(KillSwitch::new(100_000.0, 0.05)),
            validation_counter,
            rejection_counter,
            latency_hist,
            ks_subscribers: Arc::new(RwLock::new(vec![])),
        }
    }

    async fn broadcast_kill_switch(&self, drawdown: f64) {
        let event = KillSwitchEvent {
            active: true,
            reason: format!("Daily drawdown {:.2}% exceeded 5% threshold", drawdown * 100.0),
            drawdown_pct: drawdown,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH).unwrap().as_nanos() as i64,
        };
        let mut subs = self.ks_subscribers.write().await;
        subs.retain(|tx| tx.try_send(Ok(event.clone())).is_ok());
    }
}

#[tonic::async_trait]
impl RiskEngine for RiskEngineService {
    type GetPortfolioHeatStream = ReceiverStream<Result<RiskResponse, Status>>;
    type SubscribeKillSwitchStream = ReceiverStream<Result<KillSwitchEvent, Status>>;

    async fn validate(
        &self,
        request: Request<RiskRequest>,
    ) -> Result<Response<RiskResponse>, Status> {
        let timer = self.latency_hist.start_timer();
        let req = request.into_inner();
        self.validation_counter.inc();

        // ── Kill switch passthrough ────────────────────────────────────────
        if self.kill_switch.is_active() {
            self.rejection_counter.inc();
            return Ok(Response::new(RiskResponse {
                approved: false,
                rejection_reason: "Kill switch is active — all trading halted".into(),
                ..Default::default()
            }));
        }

        let returns = &req.return_series;
        let equity  = req.portfolio_equity;
        let price   = req.current_price;
        let atr     = req.atr_14;

        // ── VaR / CVaR ────────────────────────────────────────────────────
        let (var_99, cvar_99, mc_var_99) = blended_var(returns, 0.99)
            .map_err(|e| Status::internal(e.to_string()))?;
        let (var_95, _, _) = blended_var(returns, 0.95)
            .map_err(|e| Status::internal(e.to_string()))?;

        // ── Kelly sizing ──────────────────────────────────────────────────
        let kelly_raw  = kelly_quarter(0.55, 0.015, 0.010) * 4.0; // un-fractioned
        let kelly_frac = kelly_quarter(0.55, 0.015, 0.010);
        let size_usd   = position_size_usd(equity, kelly_frac, 0.10);
        let size_units = if price > 0.0 { size_usd / price } else { 0.0 };

        // ── ATR stops ─────────────────────────────────────────────────────
        let side_sign = if req.side == "BUY" { 1.0 } else { -1.0 };
        let stop_price = atr_stop(price, atr, 2.0, side_sign);
        let tp_price   = atr_take_profit(price, atr, 2.0, side_sign);

        // ── Correlation heat ──────────────────────────────────────────────
        let positions: HashMap<String, (f64, Vec<f64>)> = req.open_positions
            .iter()
            .map(|p| (p.symbol.clone(), (p.notional, returns.clone())))
            .collect();
        let heat = compute_heat(&positions, &req.symbol, size_usd, returns);

        // ── Approval logic ────────────────────────────────────────────────
        let mut rejection: Option<String> = None;

        if req.consensus_score < 0.65 {
            rejection = Some(format!(
                "Consensus score {:.2} < 0.65 threshold", req.consensus_score
            ));
        } else if var_99 > 0.15 {
            rejection = Some(format!("VaR99 {:.1}% exceeds 15% limit", var_99 * 100.0));
        } else if heat.heat_score > 0.75 {
            rejection = Some(format!(
                "Portfolio heat {:.2} too high — correlated exposure", heat.heat_score
            ));
        } else if size_units < 1e-6 {
            rejection = Some("Kelly sizing too small — insufficient edge".into());
        }

        let approved = rejection.is_none();
        if !approved { self.rejection_counter.inc(); }

        // ── SBE snapshot ──────────────────────────────────────────────────
        let sbe_bytes = encode_risk_snapshot(var_99, kelly_frac, size_usd, heat.heat_score);
        let sbe_b64   = BASE64.encode(&sbe_bytes);

        // ── Kill switch check ─────────────────────────────────────────────
        let drawdown = self.kill_switch.current_drawdown_pct(equity);
        if self.kill_switch.check_and_trip(equity) {
            self.broadcast_kill_switch(drawdown).await;
        }

        timer.observe_duration();

        Ok(Response::new(RiskResponse {
            approved,
            rejection_reason: rejection.unwrap_or_default(),
            var_95,
            var_99,
            cvar_99,
            monte_carlo_var_99: mc_var_99,
            kelly_raw,
            kelly_fractional: kelly_frac,
            position_size_usd: size_usd,
            position_size_units: size_units,
            stop_loss_price: stop_price,
            take_profit_price: tp_price,
            portfolio_heat: heat.heat_score,
            sbe_payload: sbe_b64,
        }))
    }

    async fn get_portfolio_heat(
        &self,
        request: Request<tonic::Streaming<RiskRequest>>,
    ) -> Result<Response<Self::GetPortfolioHeatStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let service = RiskEngineService::new();
        tokio::spawn(async move {
            let mut stream = request.into_inner();
            while let Ok(Some(req)) = stream.message().await {
                let resp = service.validate(Request::new(req)).await;
                match resp {
                    Ok(r) => { let _ = tx.send(Ok(r.into_inner())).await; }
                    Err(e) => { let _ = tx.send(Err(e)).await; }
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn subscribe_kill_switch(
        &self,
        _request: Request<KillSwitchRequest>,
    ) -> Result<Response<Self::SubscribeKillSwitchStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        self.ks_subscribers.write().await.push(tx);
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
