"""
QUANTEX HMM vs ADX Regime Detection — Comparison on Real BTC Data.

Fetches 365 days of BTCUSDT 1h data from Binance, then:
  1. Fits HMMRegimeDetector on first 80% (train)
  2. Evaluates both HMM and ADX on last 20% (test)
  3. Compares regime stability, confidence, and signal quality
  4. Shows transition matrices and regime statistics

Usage:
    python test_hmm_vs_adx.py
"""

import sys, os, asyncio, json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from orchestrator.backtest import DataLoader
from orchestrator.hmm_regime import HMMRegimeDetector
from orchestrator.feature_engine import FeatureEngine


async def main():
    print("=" * 70)
    print("  HMM vs ADX Regime Detection — Real BTC Data Comparison")
    print("=" * 70)

    # ---- Step 1: Fetch 365d BTCUSDT 1h data ------------------
    print()
    print("=" * 70)
    print("  STEP 1: Fetching BTCUSDT 1h data (365 days)")
    print("=" * 70)

    end = datetime.now()
    start = end - timedelta(days=365)
    df = await DataLoader.from_binance_api(
        symbol="BTCUSDT",
        interval="1h",
        start_time=start,
        end_time=end,
    )

    if df.empty:
        print("  [ERR] No data from Binance!")
        return

    print("  [OK] {} candles: {} -> {}".format(
        len(df), df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')))
    print("  Price range: ${:.2f} - ${:.2f}".format(df.close.min(), df.close.max()))
    print("  Current price: ${:.2f}".format(df.close.iloc[-1]))

    # ---- Step 2: Split train/test ---------------------------
    split = int(len(df) * 0.80)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    print()
    print("  Train: {} candles ({} -> {})".format(
        len(train_df), train_df.index[0].strftime('%Y-%m-%d'), train_df.index[-1].strftime('%Y-%m-%d')))
    print("  Test:  {} candles ({} -> {})".format(
        len(test_df), test_df.index[0].strftime('%Y-%m-%d'), test_df.index[-1].strftime('%Y-%m-%d')))

    # Guard: test set too small
    if len(test_df) < 50:
        print("  [ERR] Test set too small ({}) — need at least 50 candles".format(len(test_df)))
        return

    # ---- Step 3: Fit HMM -----------------------------------
    print()
    print("=" * 70)
    print("  STEP 2: Fitting HMM (4 regimes, 5 features)")
    print("=" * 70)

    hmm_detector = HMMRegimeDetector(n_regimes=4, seed=42)
    hmm_detector.fit(train_df, force=True)
    print("  [OK] HMM fitted on {} candles".format(len(train_df)))

    # Print HMM parameters
    transmat = hmm_detector.get_regime_transition_matrix()
    label_map = hmm_detector._regime_labels or {}
    if transmat is not None:
        print()
        print("  Transition matrix (state -> state):")
        for i in range(transmat.shape[0]):
            label = label_map.get(i, "S{}".format(i))
            probs = "  ".join("{:.3f}".format(p) for p in transmat[i])
            print("    {:>15}: [{}]".format(label, probs))

    # Print emission means
    if hmm_detector._model is not None:
        means = hmm_detector._model.means_
        print()
        print("  Emission means (normalized features per regime):")
        print("    {:>15} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}".format(
            "Regime", "Ret_1", "Ret_5", "Vol", "VolRatio", "RSI"))
        print("    {:-^15}-+-{:-^8}-+-{:-^8}-+-{:-^8}-+-{:-^8}-+-{:-^8}".format(
            "", "", "", "", "", ""))
        for i in range(means.shape[0]):
            label = label_map.get(i, "S{}".format(i))
            vals = " | ".join("{:>8.4f}".format(means[i, j]) for j in range(means.shape[1]))
            print("    {:>15} | {}".format(label, vals))

    # ---- Step 4: Evaluate both methods on test set ---------
    print()
    print("=" * 70)
    print("  STEP 3: Evaluating on test data ({} candles)".format(len(test_df)))
    print("=" * 70)

    adx_regimes = []
    hmm_regimes = []
    hmm_confidences = []
    test_prices = test_df["close"].values
    test_dates = test_df.index

    print()
    print("  Processing {} candles...".format(len(test_df)))

    fe = FeatureEngine()
    for i in range(50, len(test_df)):
        window = test_df.iloc[:i + 1]

        # ADX regime
        adx_r = fe.detect_regime(window)
        adx_regimes.append(adx_r)

        # HMM regime
        hmm_result = hmm_detector.predict(window)
        hmm_regimes.append(hmm_result["regime"])
        hmm_confidences.append(hmm_result["confidence"])

    # ---- Step 5: Comparison Stats -------------------------
    print()
    print("=" * 70)
    print("  STEP 4: Comparison Statistics")
    print("=" * 70)

    # Count regime frequencies
    adx_counts = {}
    hmm_counts = {}
    for r in adx_regimes:
        adx_counts[r] = adx_counts.get(r, 0) + 1
    for r in hmm_regimes:
        hmm_counts[r] = hmm_counts.get(r, 0) + 1

    total_adx = len(adx_regimes)
    total_hmm = len(hmm_regimes)

    print()
    print("  Regime distribution:")
    print("  {:20} {:>10} {:>10} {:>10}".format("Regime", "ADX", "HMM", "HMM Conf"))
    print("  {:-^20} {:-^10} {:-^10} {:-^10}".format("", "", "", ""))

    all_regimes = sorted(set(list(adx_counts.keys()) + list(hmm_counts.keys())))
    for r in all_regimes:
        adx_pct = adx_counts.get(r, 0) / total_adx * 100 if total_adx > 0 else 0
        hmm_pct = hmm_counts.get(r, 0) / total_hmm * 100 if total_hmm > 0 else 0
        confs = [hmm_confidences[j] for j, hr in enumerate(hmm_regimes) if hr == r]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        print("  {:20} {:>9.1f}% {:>9.1f}% {:>9.3f}".format(r, adx_pct, hmm_pct, avg_conf))

    # Agreement rate: exact matches only (exclude ADX's weak_trend)
    adx_to_hmm = {
        "strong_uptrend": "bull",
        "strong_downtrend": "bear",
        "ranging": "ranging",
        "high_volatility": "high_volatility",
    }
    total_compare = sum(1 for a in adx_regimes if a in adx_to_hmm)
    agreements = sum(1 for a, h in zip(adx_regimes, hmm_regimes)
                     if a in adx_to_hmm and adx_to_hmm[a] == h)
    agreement_rate = agreements / total_compare * 100 if total_compare > 0 else 0

    print()
    print("  Agreement rate (exact matches, weak_trend excluded): {:.1f}%".format(agreement_rate))

    # Stability
    adx_changes = sum(1 for i in range(1, len(adx_regimes)) if adx_regimes[i] != adx_regimes[i-1])
    hmm_changes = sum(1 for i in range(1, len(hmm_regimes)) if hmm_regimes[i] != hmm_regimes[i-1])
    print()
    print("  Regime change frequency (stability):")
    print("    ADX: {} changes in {} steps ({:.1f}%)".format(
        adx_changes, len(adx_regimes), adx_changes / len(adx_regimes) * 100))
    print("    HMM: {} changes in {} steps ({:.1f}%)".format(
        hmm_changes, len(hmm_regimes), hmm_changes / len(hmm_regimes) * 100))

    # HMM confidence
    conf_array = np.array(hmm_confidences)
    print()
    print("  HMM confidence stats:")
    print("    Mean:     {:.4f}".format(conf_array.mean()))
    print("    Std:      {:.4f}".format(conf_array.std()))
    print("    Min:      {:.4f}".format(conf_array.min()))
    print("    Max:      {:.4f}".format(conf_array.max()))
    print("    > 0.7:    {}/{} ({:.1f}%)".format(
        int((conf_array > 0.7).sum()), len(conf_array), (conf_array > 0.7).mean() * 100))
    print("    > 0.9:    {}/{} ({:.1f}%)".format(
        int((conf_array > 0.9).sum()), len(conf_array), (conf_array > 0.9).mean() * 100))

    # ---- Step 6: Price performance per regime ------------
    print()
    print("=" * 70)
    print("  STEP 5: Price performance per regime (next 24h return)")
    print("=" * 70)

    print()
    print("  {:20} {:>10} {:>10} {:>12} {:>10}".format(
        "HMM Regime", "Count", "Avg Conf", "Avg 24h Ret", "24h WR"))
    print("  {:-^20} {:-^10} {:-^10} {:-^12} {:-^10}".format("", "", "", "", ""))

    for hr in sorted(set(hmm_regimes)):
        confs = [hmm_confidences[j] for j, r2 in enumerate(hmm_regimes) if r2 == hr]
        idxs = [j for j, r2 in enumerate(hmm_regimes) if r2 == hr]
        returns = []
        for j in idxs:
            ci = j + 50
            if ci + 24 < len(test_df):
                ret = (test_prices[ci + 24] - test_prices[ci]) / test_prices[ci]
                returns.append(ret)
        count = len(returns)
        avg_ret = float(np.mean(returns)) * 100 if returns else 0.0
        wr = sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else 0.0
        avg_c = float(np.mean(confs)) if confs else 0.0
        print("  {:20} {:>10} {:>10.4f} {:>+11.2f}% {:>9.1f}%".format(
            hr, count, avg_c, avg_ret, wr))

    # ---- Step 7: Trading signal comparison ---------------
    print()
    print("=" * 70)
    print("  STEP 6: Trading signal comparison")
    print("=" * 70)

    hmm_signal_map = {
        "bull": "long",
        "bear": "short",
        "ranging": "hold",
        "high_volatility": "hold",
    }

    print()
    print("  {:20} {:10} {:10} {:>10} {:>8}".format(
        "HMM Regime", "Signal", "Avg Conf", "24h Ret", "24h WR"))
    print("  {:-^20} {:-^10} {:-^10} {:-^10} {:-^8}".format("", "", "", "", ""))

    for hr in sorted(set(hmm_regimes)):
        signal = hmm_signal_map.get(hr, "hold")
        confs = [hmm_confidences[j] for j, r2 in enumerate(hmm_regimes) if r2 == hr]
        idxs = [j for j, r2 in enumerate(hmm_regimes) if r2 == hr]
        returns = []
        for j in idxs:
            ci = j + 50
            if ci + 24 < len(test_df):
                ret = (test_prices[ci + 24] - test_prices[ci]) / test_prices[ci]
                returns.append(ret)
        avg_ret = float(np.mean(returns)) * 100 if returns else 0.0
        wr = sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else 0.0
        avg_c = float(np.mean(confs)) if confs else 0.0
        print("  {:20} {:10} {:10.4f} {:>+9.2f}% {:>7.1f}%".format(
            hr, signal, avg_c, avg_ret, wr))

    # ---- Summary -----------------------------------------
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print()
    print("  HMM Advantages:")
    print("  1. Probabilistic - gives confidence scores (avg: {:.3f})".format(conf_array.mean()))
    print("  2. Multi-dimensional - uses 5 features instead of ADX alone")
    print("  3. Stable transitions - transition matrix prevents flickering")
    print("  4. Self-calibrating - adapts to market structure automatically")
    print()
    print("  ADX Advantages:")
    print("  1. Simpler - no training needed, works with 50 candles")
    print("  2. Faster - O(n) vs O(n * iterations * states^2)")
    print("  3. Deterministic - same result every time")
    print("  4. No dependency on hmmlearn")
    print()
    print("  Recommendation:")
    print("  - Use HMM as the primary regime detector (once fitted)")
    print("  - Fall back to ADX when < 50 candles available")
    print("  - HMM confidence > 0.7 = high conviction signal")
    print("  - Monitor HMM transition matrix for regime shift early warning")
    print()

    # Save results
    results = {
        "data": {"candles": len(df), "train": len(train_df), "test": len(test_df)},
        "hmm": {
            "regime_counts": {k: hmm_counts.get(k, 0) for k in all_regimes},
            "avg_confidence": round(float(conf_array.mean()), 4),
            "regime_changes": hmm_changes,
            "agreement_with_adx_pct": round(agreement_rate, 1),
        },
        "adx": {
            "regime_counts": {k: adx_counts.get(k, 0) for k in all_regimes},
            "regime_changes": adx_changes,
        },
    }
    out_path = "models/hmm_vs_adx_comparison.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results saved to: {}".format(out_path))
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
