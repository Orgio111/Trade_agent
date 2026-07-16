"""Regression tests for quarantined legacy mutation and cloud paths."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint_name", "args"),
    [
        ("reset_account", (1000.0,)),
        ("update_price", ({"symbol": "BTCUSDT", "price": 50_000},)),
        ("execute_trade", ({"symbol": "BTCUSDT", "side": "buy"},)),
        (
            "open_position",
            (
                {
                    "symbol": "BTCUSDT",
                    "direction": "long",
                    "entry_price": 50_000,
                    "stop_loss": 49_000,
                },
            ),
        ),
        ("close_position", ("position-id",)),
    ],
)
async def test_legacy_fastapi_mutations_are_gone(endpoint_name, args):
    from orchestrator import main

    endpoint = getattr(main, endpoint_name)
    with pytest.raises(HTTPException) as exc_info:
        await endpoint(*args)

    assert exc_info.value.status_code == 410
    assert exc_info.value.detail["code"] == "legacy_mutation_endpoint_disabled"


@pytest.mark.asyncio
async def test_legacy_execution_denies_order_mutation_by_default(monkeypatch):
    from services.execution import (
        ExecutionMode,
        ExecutionOrchestrator,
        Order,
        OrderSide,
        OrderType,
    )

    monkeypatch.delenv("QUANTEX_ENABLE_LEGACY_PAPER_MUTATIONS", raising=False)
    orchestrator = ExecutionOrchestrator({"mode": "paper"})
    broker = orchestrator.get_broker(ExecutionMode.PAPER)
    broker.place_order = AsyncMock()

    order = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=0.001,
    )
    with pytest.raises(PermissionError, match="Legacy order mutation is disabled"):
        await orchestrator.place_order(order)

    broker.place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_execution_opt_in_is_paper_only(monkeypatch):
    from services.execution import (
        ExecutionMode,
        ExecutionOrchestrator,
        Order,
        OrderSide,
        OrderType,
    )

    monkeypatch.setenv("QUANTEX_ENABLE_LEGACY_PAPER_MUTATIONS", "true")
    orchestrator = ExecutionOrchestrator({"mode": "paper"})
    broker = orchestrator.get_broker(ExecutionMode.PAPER)
    expected = object()
    broker.place_order = AsyncMock(return_value=expected)
    order = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=0.001,
    )

    assert await orchestrator.place_order(order) is expected
    with pytest.raises(PermissionError, match="Legacy live execution"):
        orchestrator.set_mode(ExecutionMode.LIVE)


def test_legacy_live_execution_cannot_start(monkeypatch):
    from services.execution import ExecutionOrchestrator

    monkeypatch.setenv("QUANTEX_ENABLE_LEGACY_PAPER_MUTATIONS", "true")
    with pytest.raises(RuntimeError, match="Legacy live execution is quarantined"):
        ExecutionOrchestrator({"mode": "live"})


@pytest.mark.asyncio
async def test_legacy_risk_alert_never_places_liquidation_orders():
    from services.risk import RiskConfig, RiskEngine

    execution = SimpleNamespace(
        place_order=AsyncMock(),
        get_positions=AsyncMock(return_value=[]),
    )
    engine = RiskEngine()
    engine.set_execution(execution)

    assert RiskConfig().auto_liquidate_on_breach is False
    await engine._emergency_liquidate()
    execution.place_order.assert_not_awaited()


def test_inference_router_does_not_construct_cloud_providers_by_default(monkeypatch):
    from inference import router
    from inference.providers import local_ollama

    local_provider = SimpleNamespace(name="local_ollama")
    monkeypatch.setattr(local_ollama, "LocalOllamaProvider", lambda: local_provider)

    def unexpected_cloud_provider():
        pytest.fail("cloud provider was constructed without the explicit legacy gate")

    monkeypatch.setattr(router, "GroqProvider", unexpected_cloud_provider)
    monkeypatch.setattr(router, "NvidiaNIMProvider", unexpected_cloud_provider)
    monkeypatch.setattr(router, "OpenRouterProvider", unexpected_cloud_provider)

    providers = router._create_providers(allow_legacy_cloud=False)
    assert providers == {"local_ollama": local_provider}


def test_cloud_provider_override_is_filtered_when_quarantined():
    from inference.router import InferenceRouter, InferenceTask, RouterConfig

    router = object.__new__(InferenceRouter)
    router.config = RouterConfig()
    router._providers = {
        "local_ollama": SimpleNamespace(
            config=SimpleNamespace(models={"fast": "phi3:3.8b"})
        )
    }

    task = InferenceTask(provider_override=["openrouter", "nvidia_nim", "local_ollama"])
    assert router._get_provider_chain(task) == ["local_ollama"]
