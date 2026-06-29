"""Test client for Local Trading AI System."""

import asyncio
import json
import time
import websockets
from models import Candle


async def test_health():
    """Test health endpoint."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/health") as resp:
            print(f"Health: {await resp.json()}")


async def test_state():
    """Test state endpoint."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/state") as resp:
            print(f"State: {await resp.json()}")


async def test_websocket():
    """Test WebSocket with simulated candles."""
    uri = "ws://localhost:8000/candle"
    
    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket")
        
        # Simulate 1-minute candles
        base_price = 50000.0
        base_time = int(time.time() * 1000)
        
        for i in range(10):
            # Simulate price movement
            change = (i % 3 - 1) * 10  # -10, 0, +10
            base_price += change
            
            candle = Candle(
                timestamp=base_time + i * 60000,
                open=base_price - 5,
                high=base_price + 15,
                low=base_price - 15,
                close=base_price,
                volume=100.0 + i * 10,
                symbol="BTCUSDT"
            )
            
            # Send candle
            await ws.send(json.dumps(candle.to_dict()))
            print(f"Sent candle {i+1}: close={base_price:.2f}")
            
            # Receive response
            response = await ws.recv()
            data = json.loads(response)
            
            signal = data.get("signal")
            risk = data.get("risk_decision", {})
            latency = data.get("latency_ms", 0)
            
            if signal:
                print(f"  Signal: {signal['action']} conf={signal['confidence']:.2f} size={signal['size_pct']:.2%} model={signal['model_used']}")
            print(f"  Risk: {'ALLOW' if risk.get('allow') else 'BLOCK: ' + risk.get('reason', 'unknown')}")
            print(f"  Latency: {latency:.1f}ms")
            
            # Small delay
            await asyncio.sleep(0.1)
        
        print("Test complete")


async def test_manual_signal():
    """Test manual signal endpoint."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        signal = {
            "action": "BUY",
            "confidence": 0.85,
            "size_pct": 0.5,
            "entry_price": 50000,
            "stop_loss": 49500,
            "take_profit": 51000,
            "reasoning": "Strong bullish breakout above resistance",
            "regime": "trend_up"
        }
        
        async with session.post("http://localhost:8000/signal", json=signal) as resp:
            result = await resp.json()
            print(f"Manual signal result: {json.dumps(result, indent=2)}")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Local Trading AI System - Test Suite")
    print("=" * 50)
    
    # Wait for server to be ready
    await asyncio.sleep(2)
    
    # Test health
    print("\n1. Testing health endpoint...")
    await test_health()
    
    # Test state
    print("\n2. Testing state endpoint...")
    await test_state()
    
    # Test manual signal
    print("\n3. Testing manual signal...")
    await test_manual_signal()
    
    # Test WebSocket
    print("\n4. Testing WebSocket with simulated candles...")
    await test_websocket()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())