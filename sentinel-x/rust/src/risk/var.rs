//! Historical Simulation + Monte Carlo VaR/CVaR calculations.
use anyhow::Result;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use statrs::distribution::{ContinuousCDF, Normal as StatNormal};

/// Historical simulation VaR and CVaR.
/// Returns (var, cvar) as positive loss fractions.
pub fn historical_var(returns: &[f64], confidence: f64) -> (f64, f64) {
    if returns.len() < 30 {
        return (0.05, 0.08);
    }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let idx = ((1.0 - confidence) * n as f64) as usize;
    let idx = idx.max(1).min(n - 1);

    let var  = -sorted[idx - 1];
    let cvar = -sorted[..idx].iter().sum::<f64>() / idx as f64;

    (var.max(0.001), cvar.max(var))
}

/// Parametric (Gaussian) VaR.
pub fn parametric_var(returns: &[f64], confidence: f64) -> f64 {
    let n  = returns.len() as f64;
    let mu = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / n;
    let sigma = variance.sqrt();

    let dist = StatNormal::new(0.0, 1.0).unwrap();
    let z    = dist.inverse_cdf(1.0 - confidence);
    -(mu + z * sigma)
}

/// Monte Carlo VaR using Geometric Brownian Motion.
/// `n_paths` = number of simulation paths (default 100_000).
pub fn monte_carlo_var(
    returns: &[f64],
    holding_period: usize,
    confidence: f64,
    n_paths: usize,
) -> Result<f64> {
    let n  = returns.len() as f64;
    let mu = returns.iter().sum::<f64>() / n;
    let sigma = {
        let var = returns.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / n;
        var.sqrt()
    };

    let normal = Normal::new(mu, sigma)?;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let mut terminal_returns: Vec<f64> = (0..n_paths)
        .map(|_| {
            (0..holding_period)
                .map(|_| normal.sample(&mut rng))
                .sum::<f64>()
        })
        .collect();

    terminal_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - confidence) * n_paths as f64) as usize;
    let var = -terminal_returns[idx.max(1) - 1];
    Ok(var.max(0.001))
}

/// Blend historical, parametric, and Monte Carlo VaR (conservative: max).
pub fn blended_var(returns: &[f64], confidence: f64) -> Result<(f64, f64, f64)> {
    let (hist_var, cvar) = historical_var(returns, confidence);
    let param_var        = parametric_var(returns, confidence);
    let mc_var           = monte_carlo_var(returns, 1, confidence, 100_000)?;

    let final_var = hist_var.max(param_var).max(mc_var);
    Ok((final_var, cvar, mc_var))
}
