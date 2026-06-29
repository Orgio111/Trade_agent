"""
1-minute trade simulation test.
Full pipeline: Python Layer A → NATS → Layer B (aggregate) → NATS → Layer C (execute)
"""
import asyncio
import json
import random
import time
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, ".")

from orchestrator.brains import BRAIN_REGISTRY, BaseBrain, BrainSignal


async def run_brains_publish(nc, js):
    """Layer A: All 9 brains compute scores → publish to NATS signals.raw."""
    print("=" * 60)
    print("LAYER A: All AI Brains → signals.raw")
    print("=" * 60)

    brain_weights = {
        "timesfm": 0.20, "freqai": 0.15, "llm_regime": 0.12,
        "microstructure": 0.10, "finbert_nlp": 0.07, "finrl_kelly": 0.07,
        "onchain_whale": 0.05, "statarb_funding": 0.05, "orderflow_nautilus": 0.07,
        "custom_nn": 0.05, "polymarket_alpha": 0.07,
    }

    results = {}
    for name, BrainCls in BRAIN_REGISTRY.items():
        try:
            brain = BrainCls()
            score = brain.compute_score()
            if not isinstance(score, (int, float)):
                score = float(score)
            direction = 1 if score > 0.2 else (-1 if score < -0.2 else 0)
            conf = min(1.0, abs(score) * 1.5 + random.uniform(0.3, 0.5))
            sig = BrainSignal(
                brain_id=name,
                symbol="BTC/USDT",
                score=score,
                confidence=conf,
                weight=brain_weights.get(name, 0.05),
                direction=direction,
            )
        except Exception:
            # Brain needs model/API deps — synthetic signal
            score = round(random.uniform(-0.5, 0.8), 3)
            direction = 1 if score > 0.2 else (-1 if score < -0.2 else 0)
            conf = round(min(1.0, abs(score) * 1.5 + random.uniform(0.3, 0.5)), 4)
            sig = BrainSignal(
                brain_id=name,
                symbol="BTC/USDT",
                score=score,
                confidence=conf,
                weight=brain_weights.get(name, 0.05),
                direction=direction,
            )
            name_label = f"{name:20s} [SYNTH]"
        else:
            name_label = f"{name:20s} [LIVE]"

        results[name] = sig
        payload = sig.to_json()
        ack = await js.publish("signals.raw", payload)

        act = "BUY" if direction == 1 else ("SELL" if direction == -1 else "HOLD")
        print(f"  {name_label} {act:5s} score={score:+.3f} conf={conf:.2f}  → seq={ack.seq}")

    return results


async def run_go_aggregation(nc, js):
    """Layer B: Go-style signal aggregation."""
    print()
    print("=" * 60)
    print("LAYER B: Go Aggregator → signals.aggregated")
    print("=" * 60)

    # Consume raw signals
    sub = await js.subscribe("signals.raw", durable="test-aggregator-v2")
    raw_signals = []
    for _ in range(len(BRAIN_REGISTRY)):
        try:
            msg = await asyncio.wait_for(sub.next_msg(), timeout=3.0)
            raw_signals.append(json.loads(msg.data.decode()))
            await msg.ack()
        except asyncio.TimeoutError:
            break
    await sub.unsubscribe()

    if not raw_signals:
        print("  No raw signals received!")
        return None

    # Weighted aggregation (Go orchestrator logic)
    w_score = 0.0
    w_conf = 0.0
    w_total = 0.0
    buy = sell = hold = 0

    for s in raw_signals:
        w = s.get("weight", 0.05)
        sc = s.get("score", 0.0)
        cf = s.get("confidence", 0.5)
        d = s.get("direction", 0)
        w_score += sc * w * cf
        w_conf += cf * w
        w_total += w
        if d == 1:  buy += 1
        elif d == -1: sell += 1
        else: hold += 1

    if w_total > 0:
        w_score /= w_total
        w_conf /= w_total

    direction = 1 if w_score > 0.15 else (-1 if w_score < -0.15 else 0)
    action = "BUY" if direction == 1 else ("SELL" if direction == -1 else "HOLD")

    aggregated = {
        "action": action,
        "direction": direction,
        "score": round(w_score, 4),
        "confidence": round(w_conf, 4),
        "symbol": "BTC/USDT",
        "buy_votes": buy,
        "sell_votes": sell,
        "hold_votes": hold,
        "total_brains": len(raw_signals),
        "ts": time.time(),
    }

    ack = await js.publish("signals.aggregated", json.dumps(aggregated).encode())
    print(f"  ▸ {action}  weighted_score={w_score:+.4f}  confidence={w_conf:.4f}")
    print(f"  ▸ Votes: BUY={buy} SELL={sell} HOLD={hold}  (total={len(raw_signals)})")
    print(f"  ▸ Published to signals.aggregated  seq={ack.seq}")

    return aggregated


async def run_rust_execution(nc, js):
    """Layer C: Rust execution engine (paper mode simulation)."""
    print()
    print("=" * 60)
    print("LAYER C: Rust Execution Engine → signals.executed")
    print("=" * 60)

    sub = await js.subscribe("signals.aggregated", durable="test-executor-v2")
    try:
        msg = await asyncio.wait_for(sub.next_msg(), timeout=5.0)
        signal = json.loads(msg.data.decode())
        await msg.ack()
    except asyncio.TimeoutError:
        print("  No aggregated signal received!")
        await sub.unsubscribe()
        return None
    await sub.unsubscribe()

    action = signal.get("action", "HOLD")
    if action == "HOLD":
        print(f"  ▸ Signal: HOLD — no order placed")
        executed = {
            "order_id": "NONE",
            "side": "HOLD",
            "symbol": "BTC/USDT",
            "status": "SKIPPED",
            "reason": "aggregated action=HOLD",
            "ts": time.time(),
        }
        ack = await js.publish("signals.executed", json.dumps(executed).encode())
        print(f"  ▸ Published HOLD to signals.executed  seq={ack.seq}")
        return executed

    # Simulate Rust execution engine fill
    price = round(random.uniform(107000, 109000), 2)  # BTC/USDT reasonable price
    conf = signal.get("confidence", 0.5)
    qty = round(0.001 * conf, 6)

    if action == "BUY":
        sl = round(price * 0.975, 2)   # -2.5%
        tp = round(price * 1.045, 2)   # +4.5%
    else:
        sl = round(price * 1.025, 2)   # +2.5%
        tp = round(price * 0.955, 2)   # -4.5%

    executed = {
        "order_id": f"QTX-{int(time.time()*1000)}",
        "side": action,
        "symbol": "BTC/USDT",
        "price": price,
        "qty": qty,
        "sl": sl,
        "tp": tp,
        "status": "FILLED",
        "engine": "quantex-execution (Rust)",
        "score": signal.get("score"),
        "confidence": conf,
        "ts": time.time(),
    }

    ack = await js.publish("signals.executed", json.dumps(executed).encode())
    print(f"  ▸ {action} BTC/USDT @ ${price:,.2f}  qty={qty:.6f}")
    print(f"  ▸ SL=${sl:,.2f}  TP=${tp:,.2f}")
    print(f"  ▸ order_id={executed['order_id']}  status=FILLED")
    print(f"  ▸ Published to signals.executed  seq={ack.seq}")

    return executed


async def main():
    print()
    print("█" * 60)
    print("  TRINITY TRADE AGENT — 1-MINUTE LIVE SIMULATION")
    print("█" * 60)
    print()

    import nats
    nc = await nats.connect("nats://127.0.0.1:4222", name="quantex-trade-test")
    js = nc.jetstream()

    # Ensure stream exists
    try:
        await js.add_stream(name="SIGNALS", subjects=["signals.raw", "signals.aggregated", "signals.executed"])
        print("Stream SIGNALS created")
    except Exception:
        print("Stream SIGNALS exists")

    # Purge old messages for clean test
    try:
        await js.purge_stream("SIGNALS")
        print("Stream purged (clean slate)")
    except Exception:
        pass

    print()

    t0 = time.time()

    # ── Layer A ──
    brain_results = await run_brains_publish(nc, js)

    # ── Layer B ──
    aggregated = await run_go_aggregation(nc, js)

    # ── Layer C ──
    executed = await run_rust_execution(nc, js)

    elapsed = time.time() - t0

    # ── SUMMARY ──
    print()
    print("█" * 60)
    print("  SIMULATION RESULT")
    print("█" * 60)

    print()
    print("Layer A — Brain Signals:")
    for name, sig in brain_results.items():
        act = "BUY" if sig.direction == 1 else ("SELL" if sig.direction == -1 else "HOLD")
        print(f"  {sig.brain_id:20s} {act:5s}  score={sig.score:+.3f}  conf={sig.confidence:.2f}  w={sig.weight:.2f}")

    if aggregated:
        print()
        print("Layer B — Aggregated Signal:")
        print(f"  action={aggregated['action']}  score={aggregated['score']:+.4f}")
        print(f"  confidence={aggregated['confidence']:.4f}")
        print(f"  votes: BUY={aggregated['buy_votes']} SELL={aggregated['sell_votes']} HOLD={aggregated['hold_votes']}")

    if executed:
        print()
        print("Layer C — Execution Result:")
        if executed.get("status") == "SKIPPED":
            print(f"  {executed['side']} — {executed['reason']}")
        else:
            print(f"  {executed['side']} {executed['symbol']} @ ${executed['price']:,.2f}")
            print(f"  qty={executed['qty']:.6f}  SL=${executed['sl']:,.2f}  TP=${executed['tp']:,.2f}")
            print(f"  order_id={executed['order_id']}  status={executed['status']}")

    print()
    print(f"Total pipeline latency: {elapsed*1000:.0f}ms")
    print(f"Pipeline: Brains(A) → NATS → Aggregator(B) → NATS → Executor(C) → NATS")
    print()

    await nc.close()


if __name__ == "__main__":
    asyncio.run(main())
