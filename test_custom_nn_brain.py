"""Test script for Custom NN Brain.

Usage:
    python test_custom_nn_brain.py

Runs the brain in standalone mode to verify it works.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.brains.custom_nn_brain import CustomNNBrain
from orchestrator.brains.base_brain import BrainSignal


async def test_brain():
    """Test the Custom NN Brain."""
    print("=" * 60)
    print("  Custom NN Brain - Standalone Test")
    print("=" * 60)
    
    # Initialize brain
    brain = CustomNNBrain()
    print(f"\n  Brain ID:      {brain.brain_id}")
    print(f"  Model dir:     {brain._model_dir}")
    print(f"  Lookback:      {brain._lookback}")
    print(f"  Max bars:      {brain._max_bars}")
    
    # Warmup (load model if exists)
    print("\n  Running warmup...")
    await brain.warmup()
    model_loaded = brain._model.is_ready
    print(f"  Model loaded:  {model_loaded}")
    
    # Compute score
    print("\n  Computing score for BTCUSDT...")
    signal: BrainSignal = await brain.compute_score("BTCUSDT")
    
    print(f"\n  -- Signal --")
    print(f"  Brain ID:    {signal.brain_id}")
    print(f"  Symbol:      {signal.symbol}")
    print(f"  Score:       {signal.score:+.4f}")
    print(f"  Confidence:  {signal.confidence:.2%}")
    direction = "BUY" if signal.score > 0 else "SELL" if signal.score < 0 else "HOLD"
    print(f"  Direction:   {direction}")
    print(f"  Metadata:    {signal.metadata}")
    
    # Verify signal is valid
    assert -1.0 <= signal.score <= 1.0, f"Score out of range: {signal.score}"
    assert 0.0 <= signal.confidence <= 1.0, f"Confidence out of range: {signal.confidence}"
    assert signal.brain_id == "custom_nn", f"Wrong brain_id: {signal.brain_id}"
    
    print("\n  [OK] All assertions passed!")
    
    # If model not loaded, try auto-training
    if not model_loaded:
        print("\n  Model not loaded. Attempting auto-train...")
        print("  (Requires 100+ OHLCV bars from Binance)")
        trained = brain.auto_train()
        if trained:
            print("  [OK] Auto-training succeeded!")
            # Re-compute with trained model
            signal2 = await brain.compute_score("BTCUSDT")
            print(f"\n  -- Post-training Signal --")
            print(f"  Score:       {signal2.score:+.4f}")
            print(f"  Confidence:  {signal2.confidence:.2%}")
            print(f"  Metadata:    {signal2.metadata}")
        else:
            print("  [WARN] Auto-training skipped (need more data or PyTorch)")
    
    print("\n" + "=" * 60)
    print("  Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_brain())
