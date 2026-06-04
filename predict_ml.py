"""
QUANTEX ML Predict — Load saved model and predict on latest data
"""
import sys, os, asyncio, json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from orchestrator.backtest import DataLoader
from orchestrator.ml_signals import MLSignalEngine

async def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    print("=" * 60)
    print(f"  ML SIGNAL PREDICTION - Latest {days} Days")
    print("=" * 60)

    end = datetime.now()
    start = end - timedelta(days=days)
    df = await DataLoader.from_binance_api(
        symbol="BTCUSDT", interval="1h", start_time=start, end_time=end
    )
    if df.empty:
        print("[ERR] No data from Binance")
        return

    print(f"  Data:    {len(df)} candles ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Price:   ${df.close.iloc[-1]:.2f}")

    ml = MLSignalEngine()
    loaded = ml.load_model()
    if not loaded:
        print("[INFO] No saved model found - training on-the-fly...")
        result = ml.train(df, force=True)
        if result.get("status") != "trained":
            print(f"  Training failed: {result}")
            return
        print(f"  Trained: acc={result.get('test_accuracy', 0):.2%}")

    print(f"  Model:   trained={ml._train_count}x, last={ml._last_train_time}")

    pred = ml.predict(df)
    signal = ml.predict_signal(df)

    print(f"\n  {'='*60}")
    print(f"  PREDICTION RESULT")
    print(f"  {'='*60}")
    print(f"  Direction:      {pred['direction']}")
    print(f"  Confidence:     {pred['confidence']:.2%}")
    print(f"  Prob Buy:       {pred['prob_buy']:.2%}")
    print(f"  Prob Sell:      {pred['prob_sell']:.2%}")
    print(f"  Prob Hold:      {pred['prob_hold']:.2%}")

    if signal.direction != "hold":
        sl_pct = (signal.stop_loss / signal.entry_price - 1) * 100
        print(f"  Entry Price:    ${signal.entry_price:.2f}")
        print(f"  Stop Loss:      ${signal.stop_loss:.2f} ({sl_pct:+.2f}%)")
        print(f"  TP1:            ${signal.take_profits[0]['price']:.2f}")
        print(f"  TP2:            ${signal.take_profits[1]['price']:.2f}")

    print(f"  Reason:         {pred.get('reason', 'N/A')}")

    # JSON output
    print(f"\n  JSON OUTPUT:")
    output = {
        "symbol": "BTCUSDT",
        "candles": len(df),
        "source": "Binance API",
        "signal": pred["direction"],
        "confidence": round(pred["confidence"], 4),
        "prob_buy": round(pred["prob_buy"], 4),
        "prob_sell": round(pred["prob_sell"], 4),
        "prob_hold": round(pred["prob_hold"], 4),
        "entry_price": round(signal.entry_price, 2) if signal.entry_price else None,
        "stop_loss": round(signal.stop_loss, 2) if signal.stop_loss else None,
        "take_profits": signal.take_profits,
        "reason": pred.get("reason", ""),
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
