"""
QUANTEX ML Training — Anti-Overfitting Edition

Overfitting problem: Train=94%, Test=52% (gap=42pp)
Target: Train < 70%, Test > 55% (gap < 15pp)

Anti-overfitting measures:
  - 180 days of data (vs 90)
  - Feature selection with mutual information
  - Cross-validation (5-fold) instead of single split
  - Increased min_samples_leaf (regularization)
  - Reduced max_depth
  - Feature importance filtering after first training
  - Correlation-based feature pruning
"""
import sys, os, asyncio, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from orchestrator.backtest import DataLoader
from orchestrator.ml_signals import MLSignalEngine
from orchestrator.strategy import Signal

# -- Config ------------------------------------------------
DAYS = 180
INTERVAL = "1h"
TEST_SIZE = 0.15  # Hold out 15% for final validation
VAL_SIZE = 0.15   # Hold out 15% for walk-forward validation

# -- Overfitting-fighting model params ---------------------
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,         # Reduced from 12
    "min_samples_leaf": 20, # Increased from 10
    "min_samples_split": 10,
    "max_features": "sqrt", # Feature subsampling
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# -- Feature selection config -----------------------------
TOP_K_FEATURES = 15  # Select top 15 features by mutual information
CORRELATION_THRESHOLD = 0.85  # Remove features with >0.85 correlation

# -- Fetch data -------------------------------------------
async def fetch_data():
    print("=" * 60)
    print(f"  ML Training with Anti-Overfitting ({DAYS}d data)")
    print("=" * 60)

    end = datetime.now()
    start = end - timedelta(days=DAYS)

    print(f"\n  Fetching {DAYS}d BTCUSDT {INTERVAL} from Binance...")
    df = await DataLoader.from_binance_api(
        symbol="BTCUSDT", interval=INTERVAL, start_time=start, end_time=end
    )
    if df.empty:
        print("  [ERR] No data!")
        return None

    print(f"  [OK] {len(df)} candles ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Price range: ${df.close.min():.2f} -> ${df.close.max():.2f}")

    # Split: train/test/val
    n = len(df)
    test_n = int(n * TEST_SIZE)
    val_n = int(n * VAL_SIZE)
    train_n = n - test_n - val_n

    train_df = df.iloc[:train_n].copy()
    test_df = df.iloc[train_n:train_n + test_n].copy()
    val_df = df.iloc[train_n + test_n:].copy()

    print(f"  Train: {len(train_df)} candles ({train_df.index[0].strftime('%Y-%m-%d')} -> {train_df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Test:  {len(test_df)} candles ({test_df.index[0].strftime('%Y-%m-%d')} -> {test_df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Val:   {len(val_df)} candles ({val_df.index[0].strftime('%Y-%m-%d')} -> {val_df.index[-1].strftime('%Y-%m-%d')})")
    return train_df, test_df, val_df

# -- Anti-overfitting training ----------------------------
def train_with_anti_overfitting(ml_engine, train_df, test_df):
    """Train with feature selection, cross-validation, and regularization."""
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, classification_report

    # Compute all features
    print(f"\n  Computing features...")
    featured = ml_engine.compute_features(train_df)
    featured = featured.dropna()
    print(f"  Features after dropna: {len(featured)} samples, "
          f"{len([c for c in featured.columns if c not in ['open','high','low','close','volume']])} features")

    # Generate labels
    labels = ml_engine.generate_labels(featured)
    featured["label"] = labels

    # Filter to BUY/SELL only
    train_data = featured.dropna()
    train_data = train_data[train_data["label"] != 0].copy()

    if len(train_data) < 200:
        print(f"  [ERR] Only {len(train_data)} tradeable samples")
        return None

    # Feature preparation
    exclude_cols = {"open", "high", "low", "close", "volume", "label"}
    feature_cols = [c for c in train_data.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if not pd.isna(train_data[c]).any()]

    X = train_data[feature_cols].values
    y = train_data["label"].values

    # STEP 1: Correlation-based feature pruning
    print(f"\n  Step 1: Correlation pruning (threshold={CORRELATION_THRESHOLD})...")
    corr_matrix = np.corrcoef(X.T)
    high_corr_features = set()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if abs(corr_matrix[i, j]) > CORRELATION_THRESHOLD:
                high_corr_features.add(j)  # Remove the second feature

    keep_idx = [i for i in range(len(feature_cols)) if i not in high_corr_features]
    kept_features = [feature_cols[i] for i in keep_idx]
    removed_count = len(feature_cols) - len(kept_features)
    X_pruned = X[:, keep_idx]
    print(f"  Removed {removed_count} highly-correlated features, kept {len(kept_features)}")

    # STEP 2: Mutual Information feature selection
    print(f"\n  Step 2: Selecting top {TOP_K_FEATURES} features by mutual information...")
    selector = SelectKBest(mutual_info_classif, k=min(TOP_K_FEATURES, X_pruned.shape[1]))
    X_selected = selector.fit_transform(X_pruned, y)

    # Show top features
    scores = selector.scores_
    top_indices = np.argsort(scores)[-TOP_K_FEATURES:][::-1]
    final_features = [kept_features[i] for i in top_indices if i < len(kept_features)]
    print(f"  Top features: {', '.join(final_features[:8])}...")

    # Prepare test data
    test_featured = ml_engine.compute_features(test_df)
    test_labels = ml_engine.generate_labels(test_featured)
    test_featured["label"] = test_labels
    test_data = test_featured.dropna()
    test_data = test_data[test_data["label"] != 0].copy()

    X_test_full = test_data[[c for c in final_features if c in test_data.columns]].values
    y_test = test_data["label"].values

    # STEP 3: Cross-validation
    print(f"\n  Step 3: Training with {MODEL_PARAMS['n_estimators']} trees, "
          f"max_depth={MODEL_PARAMS['max_depth']}, min_samples_leaf={MODEL_PARAMS['min_samples_leaf']}...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    model = RandomForestClassifier(**MODEL_PARAMS)

    # Cross-validation on train set
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Train on full train set
    model.fit(X_scaled, y)

    # Evaluate on train
    train_pred = model.predict(X_scaled)
    train_acc = accuracy_score(y, train_pred)

    # Evaluate on test (out-of-sample within training period)
    X_test_scaled = scaler.transform(X_test_full)
    test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n  Results:")
    print(f"  Train accuracy:   {train_acc:.2%}")
    print(f"  Test accuracy:    {test_acc:.2%}")
    print(f"  Overfitting gap:  {(train_acc - test_acc):.2%}")
    print(f"  CV mean/std:      {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Feature importance
    importances = model.feature_importances_
    top_n = min(10, len(final_features))
    top_idx = np.argsort(importances)[-top_n:][::-1]
    print(f"\n  Top {top_n} important features:")
    for i in top_idx:
        print(f"    {final_features[i]:20s}: {importances[i]:.4f}")

    # Store model
    ml_engine._model = {
        "classifier": model,
        "scaler": scaler,
        "feature_names": final_features,
    }
    ml_engine._feature_names = final_features
    ml_engine._last_train_time = datetime.now()
    ml_engine._train_count += 1
    ml_engine._save_model()

    return {
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "overfitting_gap": round(train_acc - test_acc, 4),
        "features": len(final_features),
        "train_samples": len(y),
        "test_samples": len(y_test),
        "top_features": final_features[:10],
    }

# -- Validation on unseen data ----------------------------
async def validate_model(ml_engine, val_df):
    """Test model on completely unseen validation data."""
    print(f"\n{'=' * 60}")
    print(f"  VALIDATION ON UNSEEN DATA ({len(val_df)} candles)")
    print(f"{'=' * 60}")

    featured = ml_engine.compute_features(val_df)
    featured = featured.dropna()

    if featured.empty:
        print("  No valid data after feature computation")
        return

    prediction = ml_engine.predict(featured)
    signal = ml_engine.predict_signal(featured)

    print(f"\n  Prediction on latest candle:")
    print(f"  Direction:    {prediction['direction']}")
    print(f"  Confidence:   {prediction['confidence']:.2%}")
    print(f"  Prob Buy:     {prediction['prob_buy']:.2%}")
    print(f"  Prob Sell:    {prediction['prob_sell']:.2%}")
    print(f"  Prob Hold:    {prediction['prob_hold']:.2%}")

    if signal.direction != "hold":
        print(f"  Entry Price:  ${signal.entry_price:.2f}")
        print(f"  Stop Loss:    ${signal.stop_loss:.2f}")
        print(f"  Take Profit:  ${signal.take_profits[0]['price']:.2f}")

    from orchestrator.backtest import BacktestEngine
    bt = BacktestEngine(initial_balance=100.0)
    result = await bt.run(ml_engine, val_df)

    print(f"\n  ML Backtest on Validation Data:")
    print(f"  Sharpe:       {result.sharpe_ratio:.4f}")
    print(f"  Profit Factor:{result.profit_factor:.4f}")
    print(f"  Win Rate:     {result.win_rate:.2%}")
    print(f"  PnL:          ${result.total_pnl:.2f}")
    print(f"  Drawdown:     {result.max_drawdown_pct:.2%}")
    print(f"  Trades:       {result.total_trades}")

    return prediction, signal, result

# -- Main -------------------------------------------------
async def main():
    data = await fetch_data()
    if data is None:
        return
    train_df, test_df, val_df = data

    # Train with original params for comparison
    print(f"\n{'=' * 60}")
    print(f"  ORIGINAL MODEL (baseline comparison)")
    print(f"{'=' * 60}")
    ml_orig = MLSignalEngine(min_training_samples=100, lookahead_periods=12)
    try:
        t0 = time.time()
        result_orig = ml_orig.train(train_df, force=True)
        t1 = time.time()
        if result_orig.get("status") == "trained":
            print(f"  Train: {result_orig['train_accuracy']:.2%}")
            print(f"  Test:  {result_orig['test_accuracy']:.2%}")
            print(f"  Gap:   {result_orig['train_accuracy'] - result_orig['test_accuracy']:.2%}")
            print(f"  Time:  {t1 - t0:.2f}s")
    except Exception as e:
        print(f"  Original training error: {e}")

    # Train with anti-overfitting
    print(f"\n{'=' * 60}")
    print(f"  ANTI-OVERFITTING MODEL")
    print(f"{'=' * 60}")
    ml_new = MLSignalEngine(min_training_samples=100, lookahead_periods=12)
    t0 = time.time()
    result_new = train_with_anti_overfitting(ml_new, train_df, test_df)
    t1 = time.time()
    if result_new:
        print(f"  Time:  {t1 - t0:.2f}s")
        print(f"  Model saved to /tmp/models/ml_signal_model.pkl")

    # Compare
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'Basic':>15} {'Improved':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    orig_train = result_orig.get("train_accuracy", 0) if result_orig.get("status") == "trained" else 0
    orig_test = result_orig.get("test_accuracy", 0) if result_orig.get("status") == "trained" else 0
    new_train = result_new["train_acc"] if result_new else 0
    new_test = result_new["test_acc"] if result_new else 0
    print(f"  {'Train Accuracy':<25} {orig_train:>15.2%} {new_train:>15.2%}")
    print(f"  {'Test Accuracy':<25} {orig_test:>15.2%} {new_test:>15.2%}")
    orig_gap = orig_train - orig_test
    new_gap = new_train - new_test
    print(f"  {'Overfitting Gap':<25} {orig_gap:>15.2%} {new_gap:>15.2%}")
    if result_new:
        print(f"  {'CV Score':<25} {'N/A':>15} {result_new['cv_mean']:>15.3f}")
        print(f"  {'Features Used':<25} {'~26':>15} {result_new['features']:>15}")
        print(f"  {'Train Samples':<25} {'~425':>15} {result_new['train_samples']:>15}")

    # Validate on unseen data
    if result_new:
        print(f"\n{'=' * 60}")
        print(f"  FINAL VALIDATION")
        print(f"{'=' * 60}")
        await validate_model(ml_new, val_df)

    # Cleanup temp model
    model_path = "/tmp/models/ml_signal_model.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"\n  Temp model cleaned up: {model_path}")

if __name__ == "__main__":
    asyncio.run(main())
