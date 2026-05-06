//! Hard kill-switch: atomically trips on daily drawdown > threshold.
use prometheus::{Counter, Gauge, register_counter, register_gauge};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct KillSwitch {
    active:           Arc<AtomicBool>,
    day_open_equity:  Arc<AtomicI64>,  // stored as fixed-point (×1_000_000)
    peak_equity:      Arc<AtomicI64>,
    threshold_pct:    f64,
    trigger_count:    Counter,
    active_gauge:     Gauge,
}

impl KillSwitch {
    pub fn new(initial_equity: f64, threshold_pct: f64) -> Self {
        let trigger_count = register_counter!(
            "kill_switch_triggers_total",
            "Total number of kill switch activations"
        )
        .unwrap();
        let active_gauge = register_gauge!(
            "kill_switch_active",
            "1 when the kill switch is engaged, 0 otherwise"
        )
        .unwrap();

        let fp = (initial_equity * 1_000_000.0) as i64;
        Self {
            active:          Arc::new(AtomicBool::new(false)),
            day_open_equity: Arc::new(AtomicI64::new(fp)),
            peak_equity:     Arc::new(AtomicI64::new(fp)),
            threshold_pct,
            trigger_count,
            active_gauge,
        }
    }

    /// Returns true if the kill switch just triggered (newly activated).
    /// Uses a compare-and-swap to prevent double-trips.
    pub fn check_and_trip(&self, current_equity: f64) -> bool {
        if self.active.load(Ordering::Acquire) {
            return false;  // already tripped
        }
        let day_open = self.day_open_equity.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let drawdown = (day_open - current_equity) / day_open;

        if drawdown >= self.threshold_pct {
            let was_inactive = self.active
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok();
            if was_inactive {
                self.trigger_count.inc();
                self.active_gauge.set(1.0);
                tracing::error!(
                    drawdown_pct = drawdown * 100.0,
                    threshold_pct = self.threshold_pct * 100.0,
                    "KILL SWITCH ACTIVATED — halting all trading"
                );
                return true;
            }
        }
        false
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    pub fn manual_reset(&self, new_equity: f64) {
        let fp = (new_equity * 1_000_000.0) as i64;
        self.day_open_equity.store(fp, Ordering::Release);
        self.peak_equity.store(fp, Ordering::Release);
        self.active.store(false, Ordering::Release);
        self.active_gauge.set(0.0);
        tracing::warn!("Kill switch manually reset — new baseline equity={:.2}", new_equity);
    }

    pub fn current_drawdown_pct(&self, current_equity: f64) -> f64 {
        let day_open = self.day_open_equity.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        ((day_open - current_equity) / day_open).max(0.0)
    }
}
