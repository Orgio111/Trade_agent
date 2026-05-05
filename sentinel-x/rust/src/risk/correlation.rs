//! Portfolio correlation heat map — prevents overexposure to correlated assets.
use std::collections::HashMap;

/// Compute Pearson correlation between two return series.
pub fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n < 2.0 { return 0.0; }

    let mean_a = a.iter().take(n as usize).sum::<f64>() / n;
    let mean_b = b.iter().take(n as usize).sum::<f64>() / n;

    let cov: f64 = a.iter().zip(b.iter()).take(n as usize)
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum::<f64>() / n;

    let std_a = (a.iter().take(n as usize).map(|x| (x - mean_a).powi(2)).sum::<f64>() / n).sqrt();
    let std_b = (b.iter().take(n as usize).map(|y| (y - mean_b).powi(2)).sum::<f64>() / n).sqrt();

    if std_a < 1e-10 || std_b < 1e-10 { return 0.0; }
    cov / (std_a * std_b)
}

/// Portfolio-level correlation heat score.
/// Returns weighted sum of |correlation| × notional_fraction for all pairs.
/// Threshold > 0.7 means portfolio is dangerously concentrated.
pub struct PortfolioHeatMap {
    pub max_pair_correlation: f64,
    pub heat_score: f64,           // [0, 1] — 1 = perfectly correlated
    pub correlated_pairs: Vec<(String, String, f64)>,
}

pub fn compute_heat(
    positions: &HashMap<String, (f64, Vec<f64>)>,  // symbol → (notional, returns)
    proposed_symbol: &str,
    proposed_notional: f64,
    proposed_returns: &[f64],
) -> PortfolioHeatMap {
    let total_notional: f64 = positions.values().map(|(n, _)| n).sum::<f64>() + proposed_notional;
    if total_notional < 1e-10 {
        return PortfolioHeatMap {
            max_pair_correlation: 0.0,
            heat_score: 0.0,
            correlated_pairs: vec![],
        };
    }

    let mut max_corr: f64 = 0.0;
    let mut weighted_heat: f64 = 0.0;
    let mut pairs: Vec<(String, String, f64)> = vec![];

    for (sym, (notional, returns)) in positions.iter() {
        let corr = pearson_correlation(proposed_returns, returns).abs();
        let weight = (notional + proposed_notional) / (2.0 * total_notional);
        weighted_heat += corr * weight;

        if corr > max_corr { max_corr = corr; }
        if corr > 0.5 {
            pairs.push((proposed_symbol.to_string(), sym.clone(), corr));
        }
    }

    // Also check all existing pairs
    let syms: Vec<_> = positions.keys().collect();
    for i in 0..syms.len() {
        for j in (i + 1)..syms.len() {
            let (n_a, r_a) = &positions[syms[i]];
            let (n_b, r_b) = &positions[syms[j]];
            let corr = pearson_correlation(r_a, r_b).abs();
            let weight = (n_a + n_b) / (2.0 * total_notional);
            weighted_heat += corr * weight;
            if corr > max_corr { max_corr = corr; }
        }
    }

    PortfolioHeatMap {
        max_pair_correlation: max_corr,
        heat_score: weighted_heat.min(1.0),
        correlated_pairs: pairs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_series_correlation_one() {
        let r: Vec<f64> = (1..=100).map(|i| i as f64 * 0.001).collect();
        assert!((pearson_correlation(&r, &r) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn uncorrelated_near_zero() {
        let a: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let b: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).cos()).collect();
        let c = pearson_correlation(&a, &b).abs();
        assert!(c < 0.15, "sin/cos should be near-uncorrelated: {}", c);
    }
}
