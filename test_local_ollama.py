"""
QUANTEX Local Ollama Provider — Test & Setup Script.

Checks if Ollama is installed, tests LocalOllamaProvider,
and verifies inference router integration.

Usage:
    python test_local_ollama.py          # Check status only
    python test_local_ollama.py --full   # Full test (requires Ollama running)
"""

import os
import sys
import json
import time
import asyncio
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

async def check_ollama_installed():
    """Check if Ollama is installed."""
    import shutil
    if shutil.which("ollama"):
        return True
    common = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Ollama\ollama.exe"),
        os.path.expandvars(r"%PROGRAMFILES(X86)%\Ollama\ollama.exe"),
    ]
    return any(os.path.isfile(p) for p in common)

async def check_ollama_running():
    """Check if Ollama server is running on port 11434."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=3.0) as resp:
                return resp.status == 200
    except Exception:
        return False

async def get_ollama_models():
    """Get list of available Ollama models."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=3.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []

async def pull_model(model="phi-3.5:mini"):
    """Pull an Ollama model."""
    import aiohttp
    print(f"  Pulling {model} (this may take a few minutes)...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/api/pull",
            json={"name": model, "stream": False},
            timeout=600,
        ) as resp:
            if resp.status == 200:
                print(f"  [OK] Model {model} pulled!")
                return True
            print(f"  [ERR] Pull failed: {await resp.text()}")
            return False

async def test_inference(model="phi-3.5:mini"):
    """Test a real inference call."""
    from inference.providers.local_ollama import LocalOllamaProvider
    provider = LocalOllamaProvider()
    print(f"\n  Testing inference with {model}...")
    t0 = time.time()
    try:
        response = await provider.infer(
            model=model,
            messages=[
                {"role": "system", "content": "You are a market analyst. Respond in 1-2 sentences."},
                {"role": "user", "content": "What is the current sentiment for BTC?"},
            ],
            temperature=0.1, max_tokens=100,
        )
        elapsed = time.time() - t0
        print(f"  [OK] Inference in {elapsed:.1f}s")
        print(f"  Response: {response.content[:200]}")
        print(f"  Model: {response.model}")
        print(f"  Latency: {response.latency_ms:.0f}ms")
        print(f"  Cost: ${response.cost_usd:.6f} (FREE)")
        return True
    except Exception as e:
        print(f"  [ERR] Inference failed: {e}")
        return False
    finally:
        await provider.close()

async def test_router_fallback():
    """Test that the inference router includes LocalOllamaProvider."""
    from inference.router import InferenceRouter
    router = InferenceRouter()
    health = await router.get_provider_health()
    if "local_ollama" in health:
        h = health["local_ollama"]
        status = "Available" if h.get("available") else "Not available"
        print(f"  [OK] LocalOllamaProvider registered in router ({status})")
    else:
        print(f"  [ERR] LocalOllamaProvider NOT registered")

async def main():
    parser = argparse.ArgumentParser(description="Ollama Local LLM Test")
    parser.add_argument("--full", action="store_true", help="Run full test with inference")
    parser.add_argument("--model", default="phi-3.5:mini", help="Model to test")
    args = parser.parse_args()

    print("=" * 60)
    print("  QUANTEX Local LLM — Ollama Test & Setup")
    print("=" * 60)

    # Step 1: Installed?
    print(f"\n{'=' * 60}")
    print(f"  STEP 1: Ollama installation check")
    print(f"{'=' * 60}")
    installed = await check_ollama_installed()
    if installed:
        print(f"  [OK] Ollama is installed!")
    else:
        print(f"  [--] Ollama is NOT installed")
        print(f"\n  INSTALLATION:")
        print(f"  1. Download: https://ollama.com/download")
        print(f"  2. Run OllamaSetup.exe")
        print(f"  3. Open terminal: ollama pull phi-3.5:mini")
        print(f"  4. Verify: ollama list")
        print(f"\n  Docker alternative:")
        print(f"  docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama")
        print(f"  docker exec ollama ollama pull phi-3.5:mini")

    # Step 2: Running?
    print(f"\n{'=' * 60}")
    print(f"  STEP 2: Ollama server status")
    print(f"{'=' * 60}")
    running = await check_ollama_running()
    models = []
    if running:
        print(f"  [OK] Ollama server running at localhost:11434")
        models = await get_ollama_models()
        if models:
            print(f"  Models: {', '.join(models)}")
        else:
            print(f"  [--] No models pulled yet")
    else:
        print(f"  [--] Ollama server not running")
        print(f"  Start: ollama serve")

    # Step 3: Router integration
    print(f"\n{'=' * 60}")
    print(f"  STEP 3: Router integration test")
    print(f"{'=' * 60}")
    await test_router_fallback()

    # Step 4-5: Full test
    if args.full and running:
        print(f"\n{'=' * 60}")
        print(f"  STEP 4: Pull model")
        print(f"{'=' * 60}")
        if args.model not in models:
            await pull_model(args.model)
        else:
            print(f"  [OK] Model {args.model} already available")
        print(f"\n{'=' * 60}")
        print(f"  STEP 5: Inference test")
        print(f"{'=' * 60}")
        await test_inference(args.model)

    # Summary
    print(f"\n{'=' * 60}")
    if running:
        print(f"  RESULT: Ollama READY ({len(models)} models)")
    else:
        print(f"  RESULT: Install Ollama first (see instructions above)")
    print(f"  GPU: RTX 4050 6GB VRAM")
    print(f"  phi-3.5:mini -> 2.5GB VRAM, ~45 t/s")
    print(f"  qwen2.5:7b   -> 4.5GB VRAM, ~25 t/s")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    asyncio.run(main())
