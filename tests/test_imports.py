"""Smoke tests for the orchestrator import graph.

These guard against regressions like the previously-broken
``from .data.pipeline import MarketDataOrchestrator`` import that prevented the
FastAPI app from starting.
"""
import importlib


def test_data_pipeline_exposes_datapipeline():
    mod = importlib.import_module("orchestrator.data_pipeline")
    assert hasattr(mod, "DataPipeline")


def test_orchestrator_main_imports_and_builds_app():
    main = importlib.import_module("orchestrator.main")
    # FastAPI app is constructed at import time.
    assert main.app.__class__.__name__ == "FastAPI"
    # The forecaster symbol is wired in.
    assert hasattr(main, "TimesFMForecaster")


def test_timesfm_routes_registered():
    main = importlib.import_module("orchestrator.main")
    paths = {route.path for route in main.app.routes}
    assert "/api/v2/forecast/timesfm" in paths
    assert "/api/v2/forecast/timesfm/status" in paths
