use serde::{Deserialize, Serialize};
use crate::order::{Order, Position};
use crate::exchange::AccountInfo;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParams {
    pub max_risk_per_trade: f64,    // 0.01 = 1%
    pub max_risk_per_session: f64,  // 0.05 = 5%
    pub max_total_exposure: f64,    // 0.15 = 15%
    pub max_drawdown_soft: f64,     // 0.10 = 10%
    pub max_drawdown_hard: f64,     // 0.20 = 20%
    pub max_consecutive_losses: u32,// 4
    pub max_open_positions: u32,    // 3
    pub max_leverage: u8,           // 10
    pub atr_multiplier_sl: f64,     // 1.5
    pub atr_multiplier_tp: f64,     // 3.0
}

impl Default for RiskParams {
    fn default() -> Self {
        Self {
            max_risk_per_trade: 0.01,
            max_risk_per_session: 0.05,
            max_total_exposure: 0.15,
            max_drawdown_soft: 0.10,
            max_drawdown_hard: 0.20,
            max_consecutive_losses: 4,
            max_open_positions: 3,
            max_leverage: 10,
            atr_multiplier_sl: 1.5,
            atr_multiplier_tp: 3.0,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheckResult {
    pub approved: bool,
    pub severity: String,        // "hard" | "soft" | "none"
    pub reason: String,
    pub size_multiplier: Option<f64>,
    pub max_leverage: Option<u8>,
}

pub struct RiskEngine {
    #[allow(dead_code)]
    pub params: RiskParams,
}

impl RiskEngine {
    pub fn new(params: RiskParams) -> Self {
        Self { params }
    }

    #[allow(dead_code)]
    /// ATR-based position sizing
    pub fn calculate_position_size(
        &self,
        balance: f64,
        atr: f64,
        _price: f64,
        leverage: u8,
    ) -> f64 {
        let risk_amount = balance * self.params.max_risk_per_trade;
        let sl_distance = atr * self.params.atr_multiplier_sl;
        let position = (risk_amount / sl_distance) * leverage as f64;
        let max_size = balance * self.params.max_total_exposure * leverage as f64;
        position.min(max_size).max(0.0)
    }

    #[allow(dead_code)]
    /// Half-Kelly criterion position sizing
    pub fn kelly_size(
        &self,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        balance: f64,
    ) -> f64 {
        if avg_loss.abs() < f64::EPSILON {
            return 0.0;
        }
        let q = 1.0 - win_rate;
        let kelly = (win_rate / avg_loss.abs()) - (q / avg_win.abs());
        let half_kelly = (kelly * 0.5).max(0.0).min(0.25);
        balance * half_kelly
    }

    #[allow(dead_code)]
    /// Full risk gate check before trade
    pub fn check_trade(
        &self,
        order: &Order,
        _account: &AccountInfo,
        positions: &[Position],
        consecutive_losses: u32,
        drawdown: f64,
    ) -> RiskCheckResult {
        // Hard: max drawdown exceeded
        if drawdown >= self.params.max_drawdown_hard {
            return RiskCheckResult {
                approved: false,
                severity: "hard".into(),
                reason: format!("Max drawdown {:.1}% exceeded {:.1}%", 
                    drawdown * 100.0, self.params.max_drawdown_hard * 100.0),
                size_multiplier: None,
                max_leverage: None,
            };
        }

        // Hard: too many consecutive losses
        if consecutive_losses >= self.params.max_consecutive_losses {
            return RiskCheckResult {
                approved: false,
                severity: "hard".into(),
                reason: format!("{} consecutive losses — cooldown required", consecutive_losses),
                size_multiplier: None,
                max_leverage: None,
            };
        }

        // Hard: max open positions
        if positions.len() as u32 >= self.params.max_open_positions {
            return RiskCheckResult {
                approved: false,
                severity: "hard".into(),
                reason: format!("Max open positions ({}) reached", self.params.max_open_positions),
                size_multiplier: None,
                max_leverage: None,
            };
        }

        // Hard: leverage exceeds max
        if order.leverage > self.params.max_leverage {
            return RiskCheckResult {
                approved: false,
                severity: "hard".into(),
                reason: format!("Leverage {} exceeds max {}", order.leverage, self.params.max_leverage),
                size_multiplier: None,
                max_leverage: None,
            };
        }

        // Soft: drawdown warning — reduce size
        if drawdown >= self.params.max_drawdown_soft {
            let reduction = 1.0 - (drawdown - self.params.max_drawdown_soft) / 0.10;
            return RiskCheckResult {
                approved: true,
                severity: "soft".into(),
                reason: format!("Soft DD — reducing size by {:.0}%", (1.0 - reduction) * 100.0),
                size_multiplier: Some(reduction.max(0.25)),
                max_leverage: Some(5),
            };
        }

        // Soft: recent losses — reduce size
        if consecutive_losses >= 2 {
            return RiskCheckResult {
                approved: true,
                severity: "soft".into(),
                reason: "Recent losses — reducing size 50%".into(),
                size_multiplier: Some(0.5),
                max_leverage: Some(3),
            };
        }

        RiskCheckResult {
            approved: true,
            severity: "none".into(),
            reason: "All gates passed".into(),
            size_multiplier: Some(1.0),
            max_leverage: Some(self.params.max_leverage),
        }
    }

    #[allow(dead_code)]
    /// Calculate dynamic leverage based on volatility and confidence
    pub fn dynamic_leverage(
        &self,
        base_leverage: u8,
        volatility_percentile: f64,
        confidence: f64,
        regime: &str,
    ) -> u8 {
        let vol_scalar = 1.0 - (volatility_percentile / 200.0);
        let conf_scalar = 0.5 + confidence * 0.5;

        let regime_scalar = match regime {
            "strong_trend" => 1.2,
            "weak_trend" => 1.0,
            "ranging" => 0.7,
            "high_vol" => 0.5,
            "crisis" => 0.0,
            _ => 1.0,
        };

        let final_leverage = base_leverage as f64 * vol_scalar * conf_scalar * regime_scalar;
        (final_leverage as u8).clamp(1, self.params.max_leverage)
    }

    #[allow(dead_code)]
    /// Calculate maximum drawdown from equity curve
    pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }
        let mut peak = equity_curve[0];
        let mut max_dd = 0.0;
        for &value in equity_curve {
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        max_dd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_size_calculation() {
        let risk = RiskEngine::new(RiskParams::default());
        let size = risk.calculate_position_size(100.0, 100.0, 50000.0, 3);
        assert!(size > 0.0);
        assert!(size <= 100.0 * 0.15 * 3.0);
    }

    #[test]
    fn test_max_drawdown() {
        let curve = vec![100.0, 110.0, 105.0, 95.0, 90.0, 100.0];
        let dd = RiskEngine::max_drawdown(&curve);
        assert!((dd - 0.1818).abs() < 0.01);
    }

    #[test]
    fn test_dynamic_leverage_reduces_in_high_vol() {
        let risk = RiskEngine::new(RiskParams::default());
        let low_vol = risk.dynamic_leverage(10, 30.0, 0.8, "strong_trend");
        let high_vol = risk.dynamic_leverage(10, 95.0, 0.8, "high_vol");
        assert!(high_vol < low_vol);
    }
}
