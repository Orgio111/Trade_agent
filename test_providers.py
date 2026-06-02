"""
QUANTEX AI Provider Test -- Tests all inference providers and routing.
Windows cp1252-safe: no emoji, no fancy Unicode characters.
"""
import os
import sys
import asyncio
import json

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# Load .env file
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

sys.path.insert(0, os.path.dirname(__file__))

results = {"providers": {}, "router": {}, "adaptive_routing": {}, "integration": {}, "errors": []}


def print_banner(title):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def truncate(s, n=30):
    return str(s)[:n]


# ── Check API Keys ──────────────────────────────────────────
print_banner("QUANTEX AI PROVIDER TEST")

nim_key = os.getenv("NVIDIA_API_KEY", "")
or_key = os.getenv("OPENROUTER_API_KEY", "")
groq_key = os.getenv("GROQ_API_KEY", "")

print("API Key Status:")
print(f"  NVIDIA:     {'SET (' + truncate(nim_key) + '...)' if nim_key else 'MISSING'}")
print(f"  OpenRouter: {'SET (' + truncate(or_key) + '...)' if or_key else 'MISSING'}")
print(f"  Groq:       {'SET (' + truncate(groq_key) + '...)' if groq_key else 'MISSING'}")

if not any([nim_key, or_key, groq_key]):
    print("\nERROR: No API keys found! Cannot test providers.")
    sys.exit(1)

# ── Provider Tests ──────────────────────────────────────────

from inference.providers import GroqProvider, NvidiaNIMProvider, OpenRouterProvider
from inference.router import InferenceRouter, InferenceTask, RouterConfig


async def test_provider(name, provider_cls, key_name):
    print(f"\n--- Testing: {name} ---")
    provider = provider_cls()
    available = await provider.is_available()
    health = await provider.health()

    models = list(health.get("models", []))[:3]
    print(f"  Available:      {available}")
    print(f"  Free tier:      {health.get('free_tier', '?')}")
    print(f"  Models:         {models}")
    print(f"  Base URL:       {provider.config.base_url}")

    results["providers"][name] = {
        "available": available,
        "health": {k: v for k, v in health.items() if k != "models"},
        "models": models,
    }

    if available:
        print(f"  Test inference...")
        try:
            model = list(provider.config.models.values())[0]
            t0 = asyncio.get_event_loop().time()
            resp = await provider.infer(
                model=model,
                messages=[{"role": "user", "content": "Reply with just the word: OK"}],
                temperature=0.0,
                max_tokens=10,
            )
            latency = (asyncio.get_event_loop().time() - t0) * 1000
            content = resp.content.strip()
            print(f"  Response:       '{content}' ({latency:.0f}ms)")
            results["providers"][name]["inference_ok"] = True
            results["providers"][name]["latency_ms"] = round(latency, 1)
        except Exception as e:
            print(f"  Inference FAILED: {type(e).__name__}: {str(e)[:150]}")
            results["providers"][name]["inference_ok"] = False
            results["providers"][name]["error"] = str(e)[:200]
    else:
        # Check endpoint directly
        print(f"  Checking endpoint...")
        try:
            import httpx
            key = os.getenv(key_name, "")
            async with httpx.AsyncClient(timeout=5) as client:
                base = provider.config.base_url.rstrip("/")
                hdrs = {"Authorization": f"Bearer {key}"}
                if "groq" in name.lower():
                    r = await client.get(f"{base}/models", headers=hdrs)
                elif "nim" in name.lower():
                    r = await client.get(f"{base}/models", headers=hdrs)
                elif "openrouter" in name.lower():
                    r = await client.get("https://openrouter.ai/api/v1/models", headers=hdrs)
                print(f"  Endpoint: HTTP {r.status_code}")
                results["providers"][name]["endpoint_status"] = r.status_code
        except Exception as e:
            print(f"  Endpoint error: {type(e).__name__}")
            results["providers"][name]["endpoint_error"] = str(e)[:100]


# ── Router Tests ────────────────────────────────────────────

async def test_router():
    print(f"\n--- Testing: InferenceRouter ---")

    config = RouterConfig(budget_tier="free", enable_cache=False)
    router = InferenceRouter(config=config)

    # Provider health
    print(f"\n  Provider Health:")
    health = await router.get_provider_health()
    for name, h in health.items():
        status = "AVAILABLE" if h.get("available") else "UNAVAILABLE"
        cb = " | CIRCUIT OPEN" if h.get("circuit_open") else ""
        print(f"    {name}: {status}{cb}")
    results["router"]["health"] = {n: {"available": h.get("available"), "circuit_open": h.get("circuit_open")} for n, h in health.items()}

    # Router inference
    print(f"\n  Router inference test...")
    try:
        task = InferenceTask(
            task_type="fast",
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            temperature=0.0,
            max_tokens=10,
        )
        t0 = asyncio.get_event_loop().time()
        resp = await router.infer(task)
        latency = (asyncio.get_event_loop().time() - t0) * 1000
        print(f"  Response from '{resp.provider}': '{resp.content.strip()}' ({latency:.0f}ms)")
        results["router"]["inference_ok"] = True
        results["router"]["used_provider"] = resp.provider
        results["router"]["latency_ms"] = round(latency, 1)
    except Exception as e:
        print(f"  Router inference FAILED: {type(e).__name__}: {str(e)[:200]}")
        results["router"]["inference_ok"] = False
        results["router"]["error"] = str(e)[:300]

    # Agent -> Provider chains display
    from orchestrator.agent_routing import AGENT_PROVIDER_CHAINS, AGENT_TASK_TYPE_OVERRIDES
    print(f"\n  Agent -> Provider Chains ({len(AGENT_PROVIDER_CHAINS)} agents):")
    for agent, chain in sorted(AGENT_PROVIDER_CHAINS.items()):
        task_ovr = AGENT_TASK_TYPE_OVERRIDES.get(agent, "-")
        print(f"    {agent:22s} -> {', '.join(chain):45s} [{task_ovr}]")

    results["router"]["agent_chains_count"] = len(AGENT_PROVIDER_CHAINS)

    # Cost tracker
    today = router.get_todays_usage()
    print(f"\n  Cost Tracker:")
    print(f"    Budget tier:   {router.config.budget_tier}")
    print(f"    Today's cost:  ${today['total_cost']:.6f}")
    print(f"    Today's reqs:  {today['requests']}")
    results["router"]["cost_today"] = today


# ── Adaptive Routing Tests ──────────────────────────────────

async def test_adaptive_routing():
    print(f"\n--- Testing: Adaptive Routing ---")

    from orchestrator.agent_routing import AgentModelRouter, AGENT_PROVIDER_CHAINS
    import random

    router = AgentModelRouter()

    agents = ["market_analyst", "scalping_agent", "risk_guardian"]
    providers = ["groq", "nvidia_nim", "openrouter"]

    print(f"  Recording test latencies (12 samples per provider)...")
    for agent in agents:
        for prov in providers:
            for _ in range(12):
                latency = random.uniform(100, 2000)
                success = random.random() > 0.1
                router.record_provider_latency(agent, prov, latency, success)

    print(f"\n  Adapted Chains:")
    for agent in agents:
        base = AGENT_PROVIDER_CHAINS.get(agent, [])
        adapted = router.get_optimal_provider_chain(agent, base)
        changed = " (ADAPTED)" if adapted != base else ""
        print(f"    {agent:22s} base={base} -> adapted={adapted}{changed}")

    summary = router.get_adaptive_routing_summary()
    for agent in agents:
        if agent in summary:
            for prov in providers:
                if prov in summary[agent]:
                    d = summary[agent][prov]
                    print(f"      {prov:15s} ema={d['ema_latency_ms']}ms p50={d['p50_latency_ms']}ms rate={d['success_rate']:.0%}")

    results["adaptive_routing"]["tested"] = agents


# ── Integration Bridge Tests ────────────────────────────────

async def test_integration():
    print(f"\n--- Testing: InferenceIntegration (agent bridge) ---")

    from orchestrator.inference_integration import InferenceIntegration

    integration = InferenceIntegration(budget_tier="free")

    summary = integration.get_agent_model_summary()
    print(f"  Agent<->Model mappings: {len(summary)} agents")
    for s in summary[:6]:
        print(f"    {s['agent_name']:22s} -> {s['primary_model']:30s} ({s['primary_provider']})")
    if len(summary) > 6:
        print(f"    ... and {len(summary)-6} more")

    results["integration"]["agent_count"] = len(summary)

    print(f"\n  Route inference through Integration bridge...")
    try:
        resp = await integration.route_inference(
            task_type="fast",
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            temperature=0.0,
            max_tokens=10,
        )
        print(f"  Bridge response: '{resp.strip()}'")
        results["integration"]["bridge_ok"] = True
    except Exception as e:
        print(f"  Bridge FAILED: {type(e).__name__}: {str(e)[:200]}")
        results["integration"]["bridge_ok"] = False
        results["integration"]["error"] = str(e)[:200]


# ── Main ────────────────────────────────────────────────────

async def main():
    print(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")

    # 1. Individual provider tests
    await test_provider("groq", GroqProvider, "GROQ_API_KEY")
    await test_provider("nvidia_nim", NvidiaNIMProvider, "NVIDIA_API_KEY")
    await test_provider("openrouter", OpenRouterProvider, "OPENROUTER_API_KEY")

    # 2. Inference router
    await test_router()

    # 3. Adaptive routing system
    await test_adaptive_routing()

    # 4. Integration bridge
    await test_integration()

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  TEST SUMMARY")
    print(f"{'=' * 60}")

    for name, data in results["providers"].items():
        if data.get("inference_ok"):
            status = "WORKING"
        elif data.get("available"):
            status = "AVAILABLE (no test)"
        else:
            status = "UNAVAILABLE"
        lat = data.get("latency_ms", "-")
        print(f"  {name:15s}: {status:20s} (latency: {lat}ms)")

    r = results["router"]
    router_ok = r.get("inference_ok", False)
    print(f"  {'router':15s}: {'WORKING' if router_ok else 'FAILED':20s} (used: {r.get('used_provider', '-')}, latency: {r.get('latency_ms', '-')}ms)")

    i = results["integration"]
    bridge_ok = i.get("bridge_ok", False)
    print(f"  {'integration':15s}: {'WORKING' if bridge_ok else 'FAILED':20s}")

    # Save results
    with open("test_providers_result.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to test_providers_result.json")


if __name__ == "__main__":
    asyncio.run(main())
