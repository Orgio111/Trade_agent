//! Fractional Kelly Criterion with Quarter-Kelly default.

/// Full Kelly fraction = (odds * win_rate - loss_rate) / odds
/// Never negative; clipped to [0, 1].
pub fn kelly_full(win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
    if avg_loss < 1e-10 { return 0.0; }
    let odds = avg_win / avg_loss;
    let k = (odds * win_rate - (1.0 - win_rate)) / odds;
    k.max(0.0).min(1.0)
}

/// Quarter-Kelly (25%) — the institutional standard for avoiding ruin.
pub fn kelly_quarter(win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
    kelly_full(win_rate, avg_win, avg_loss) * 0.25
}

/// Fractional Kelly with custom fraction.
pub fn kelly_fractional(win_rate: f64, avg_win: f64, avg_loss: f64, fraction: f64) -> f64 {
    kelly_full(win_rate, avg_win, avg_loss) * fraction.clamp(0.0, 1.0)
}

/// Position size in USD given equity and Kelly fraction.
pub fn position_size_usd(
    portfolio_equity: f64,
    kelly_frac: f64,
    max_position_pct: f64,
) -> f64 {
    let raw = portfolio_equity * kelly_frac;
    raw.min(portfolio_equity * max_position_pct)
}

/// ATR-based stop loss price.
/// Side: 1.0 = BUY (stop below), -1.0 = SELL (stop above).
pub fn atr_stop(current_price: f64, atr: f64, multiplier: f64, side: f64) -> f64 {
    current_price - side * multiplier * atr
}

/// ATR-based take profit.
pub fn atr_take_profit(current_price: f64, atr: f64, multiplier: f64, side: f64) -> f64 {
    current_price + side * multiplier * 1.5 * atr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_edge_gives_positive_kelly() {
        let k = kelly_full(0.6, 0.02, 0.01);
        assert!(k > 0.0, "Positive edge should give positive Kelly");
    }

    #[test]
    fn zero_edge_gives_zero_kelly() {
        let k = kelly_full(0.5, 0.01, 0.01);
        assert!(k <= 0.0001, "No edge should give ~zero Kelly");
    }

    #[test]
    fn quarter_kelly_is_25_percent() {
        let full = kelly_full(0.6, 0.02, 0.01);
        let quarter = kelly_quarter(0.6, 0.02, 0.01);
        let ratio = quarter / full;
        assert!((ratio - 0.25).abs() < 1e-9);
    }

    #[test]
    fn position_capped_at_max_pct() {
        let size = position_size_usd(100_000.0, 0.50, 0.10);
        assert!(size <= 10_000.0 + 1e-6);
    }
}
