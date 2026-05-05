//! Walk-Forward Backtest Engine — runs in Rust for maximum throughput.
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct Bar {
    pub close:  f64,
    pub high:   f64,
    pub low:    f64,
    pub volume: f64,
}

#[derive(Debug, Default)]
pub struct BacktestResult {
    pub total_trades:   usize,
    pub winning_trades: usize,
    pub total_pnl:      f64,
    pub sharpe_ratio:   f64,
    pub max_drawdown:   f64,
    pub win_rate:       f64,
    pub returns:        Vec<f64>,
}

pub struct BacktestEngine {
    initial_equity: f64,
    atr_period:     usize,
    atr_multiplier: f64,
    kelly_fraction: f64,
}

impl BacktestEngine {
    pub fn new(
        initial_equity: f64,
        atr_period: usize,
        atr_multiplier: f64,
        kelly_fraction: f64,
    ) -> Self {
        Self { initial_equity, atr_period, atr_multiplier, kelly_fraction }
    }

    /// Run a simple ATR-breakout strategy backtest over `bars`.
    /// Returns period returns for VaR/Sharpe calculation.
    pub fn run(&self, bars: &[Bar]) -> BacktestResult {
        if bars.len() < self.atr_period + 2 {
            return BacktestResult::default();
        }

        let mut equity     = self.initial_equity;
        let mut peak       = equity;
        let mut max_dd     = 0.0f64;
        let mut returns    = Vec::with_capacity(bars.len());
        let mut n_trades   = 0usize;
        let mut n_wins     = 0usize;
        let mut total_pnl  = 0.0f64;

        let atrs = self.compute_atr(bars);

        let mut position: Option<(f64, f64)> = None; // (entry_price, stop)

        for i in (self.atr_period + 1)..bars.len() {
            let bar = &bars[i];
            let prev = &bars[i - 1];
            let atr  = atrs[i];

            // Exit logic
            if let Some((entry, stop)) = position {
                if bar.low <= stop {
                    let pnl = (stop - entry) * (equity * self.kelly_fraction / entry.max(1e-10));
                    equity += pnl;
                    total_pnl += pnl;
                    n_trades += 1;
                    if pnl > 0.0 { n_wins += 1; }
                    returns.push(pnl / self.initial_equity);
                    position = None;
                }
            }

            // Entry logic: breakout above 20-period high
            if position.is_none() {
                let high_20 = bars[(i.saturating_sub(20))..i]
                    .iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
                if bar.close > high_20 {
                    let stop = bar.close - self.atr_multiplier * atr;
                    position = Some((bar.close, stop));
                }
            }

            // Drawdown tracking
            if equity > peak { peak = equity; }
            let dd = (peak - equity) / peak;
            if dd > max_dd { max_dd = dd; }
        }

        let sharpe = self.sharpe(&returns);
        BacktestResult {
            total_trades: n_trades,
            winning_trades: n_wins,
            total_pnl,
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
            win_rate: if n_trades > 0 { n_wins as f64 / n_trades as f64 } else { 0.0 },
            returns,
        }
    }

    fn compute_atr(&self, bars: &[Bar]) -> Vec<f64> {
        let mut atrs = vec![0.0f64; bars.len()];
        let p = self.atr_period;
        for i in (p + 1)..bars.len() {
            let tr: f64 = bars[(i - p)..i].iter().zip(bars[(i - p - 1)..i - 1].iter())
                .map(|(b, prev)| {
                    (b.high - b.low)
                        .max((b.high - prev.close).abs())
                        .max((b.low - prev.close).abs())
                })
                .sum::<f64>() / p as f64;
            atrs[i] = tr;
        }
        atrs
    }

    fn sharpe(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 { return 0.0; }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let var  = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std  = var.sqrt();
        if std < 1e-10 { return 0.0; }
        (mean / std) * (252.0_f64).sqrt()
    }
}
