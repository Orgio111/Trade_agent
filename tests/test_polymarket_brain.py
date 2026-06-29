"""Tests for the Polymarket Alpha Brain — 5-Formula Mathematical Pipeline.

Verifies:
  1. Import registration (brain in BRAIN_REGISTRY, __all__)
  2. Each formula independently
  3. Full pipeline end-to-end (4-stage demo)
  4. Config override path
  5. Edge cases (no data, zero price, extreme evidence)
"""
from __future__ import annotations

import importlib
from typing import Any

import pytest


# ── Import / Registration ───────────────────────────────────────

def test_polymarket_brain_importable():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    assert hasattr(mod, "PolymarketBrain")
    assert hasattr(mod, "AlphaResult")
    assert hasattr(mod, "BayesianEvidenceTracker")
    assert hasattr(mod, "LongshotBiasCorrector")
    assert hasattr(mod, "EVROIFilter")
    assert hasattr(mod, "QuarterKellySizer")
    assert hasattr(mod, "NashRouter")


def test_polymarket_in_brain_registry():
    brains = importlib.import_module("orchestrator.brains")
    assert "polymarket_alpha" in brains.BRAIN_REGISTRY
    assert brains.BRAIN_REGISTRY["polymarket_alpha"] is brains.PolymarketBrain


def test_polymarket_in_all():
    brains = importlib.import_module("orchestrator.brains")
    assert "PolymarketBrain" in brains.__all__


# ── Formula 1: Bayesian Evidence Tracking ───────────────────────

def test_bayesian_no_evidence_returns_prior():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults()
    tracker = mod.BayesianEvidenceTracker(cfg)
    posterior, meta = tracker.update(prior=0.35, evidence=[])
    assert posterior == pytest.approx(0.35, abs=0.001)
    assert meta["evidence_count"] == 0
    assert meta["likelihood_ratio"] == pytest.approx(1.0, abs=0.001)


def test_bayesian_single_evidence_updates_prior():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults()
    tracker = mod.BayesianEvidenceTracker(cfg)
    ev = [mod.EvidenceSnapshot(source="test", headline="h", impact=0.60, reliability=0.80)]
    posterior, meta = tracker.update(prior=0.35, evidence=ev)
    # LR = 0.80 * (1 + 0.60) = 1.28
    # posterior = 0.35*1.28 / (0.35*1.28 + 0.65) = 0.448/1.098 ≈ 0.408
    assert posterior == pytest.approx(0.408, abs=0.002)
    assert meta["likelihood_ratio"] == pytest.approx(1.28, abs=0.01)
    assert meta["jump_triggered"] is False  # 0.058 < 0.20


def test_bayesian_jump_detection():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults(bayesian_jump_threshold=0.05)
    tracker = mod.BayesianEvidenceTracker(cfg)
    ev = [mod.EvidenceSnapshot(source="test", headline="h", impact=0.90, reliability=0.95)]
    posterior, meta = tracker.update(prior=0.30, evidence=ev)
    # LR = 0.95 * (1 + 0.90) = 1.805
    # posterior = 0.30*1.805 / (0.30*1.805 + 0.70) = 0.5415/1.2815 ≈ 0.423
    assert meta["jump"] > 0.05
    assert meta["jump_triggered"] is True


# ── Formula 2: Longshot Bias Correction ────────────────────────

def test_longshot_low_price_penalty():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults(longshot_low_threshold=0.10, longshot_low_penalty=0.57)
    corrector = mod.LongshotBiasCorrector(cfg)
    adjusted, meta = corrector.correct(raw_prob=0.05)
    assert adjusted == pytest.approx(0.05 * 0.57, abs=0.001)
    assert "longshot_penalty" in meta["correction"]


def test_longshot_high_price_boost():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults(longshot_high_threshold=0.80, longshot_high_boost=0.02)
    corrector = mod.LongshotBiasCorrector(cfg)
    adjusted, meta = corrector.correct(raw_prob=0.85)
    assert adjusted == pytest.approx(0.87, abs=0.001)
    assert "favorite_boost" in meta["correction"]


def test_longshot_midrange_no_correction():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults()
    corrector = mod.LongshotBiasCorrector(cfg)
    adjusted, meta = corrector.correct(raw_prob=0.50)
    assert adjusted == pytest.approx(0.50, abs=0.001)
    assert meta["correction"] == "none"


# ── Formula 3: EV & ROI Filter ─────────────────────────────────

def test_ev_positive_when_adjusted_above_cost():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    ev_filter = mod.EVROIFilter()
    ev, roi, meta = ev_filter.evaluate(adjusted_prob=0.60, yes_price=0.35)
    assert ev == pytest.approx(0.25, abs=0.001)  # 0.60*1 - 0.35
    assert roi == pytest.approx(0.714, abs=0.01)  # 0.25/0.35
    assert meta["tradeable"] is True


def test_ev_negative_when_adjusted_below_cost():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    ev_filter = mod.EVROIFilter()
    ev, roi, meta = ev_filter.evaluate(adjusted_prob=0.20, yes_price=0.50)
    assert ev < 0
    assert meta["tradeable"] is False


def test_ev_zero_price_edge_case():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    ev_filter = mod.EVROIFilter()
    ev, roi, meta = ev_filter.evaluate(adjusted_prob=0.50, yes_price=0.001)
    assert ev > 0
    assert meta["tradeable"] is True


# ── Formula 4: Quarter-Kelly Sizing ─────────────────────────────

def test_kelly_basic():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults(kelly_quarter_factor=0.25, kelly_hard_cap=0.05)
    sizer = mod.QuarterKellySizer(cfg)
    f_raw, f_capped, meta = sizer.size(prob=0.60, yes_price=0.35)
    # b = 0.65/0.35 ≈ 1.857; p=0.6; q=0.4
    # f_kelly = (1.857*0.6 - 0.4)/1.857 ≈ 0.3846
    # f_safe = 0.25 * 0.3846 ≈ 0.0962 → capped at 0.05
    assert f_capped == pytest.approx(0.05, abs=0.001)
    assert meta["kelly_capped"] == pytest.approx(0.05, abs=0.001)


def test_kelly_no_edge_zero_kelly():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults()
    sizer = mod.QuarterKellySizer(cfg)
    f_raw, f_capped, meta = sizer.size(prob=0.20, yes_price=0.50)
    assert f_raw == pytest.approx(0.0, abs=0.001)
    assert f_capped == pytest.approx(0.0, abs=0.001)


# ── Formula 5: Nash Router ──────────────────────────────────────

def test_nash_router_maker_by_default():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults(maker_default_probability=0.70, taker_threshold_pct=0.20)
    router = mod.NashRouter(cfg)
    route, maker_prob, meta = router.route(ev=0.05, roi=0.10)
    assert route == "MAKER"
    assert maker_prob == pytest.approx(0.70, abs=0.01)


def test_nash_router_taker_on_high_edge():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    cfg = mod._PolymarketDefaults(maker_default_probability=0.70, taker_threshold_pct=0.20)
    router = mod.NashRouter(cfg)
    route, maker_prob, meta = router.route(ev=0.30, roi=0.50)
    assert route == "TAKER"
    assert maker_prob == pytest.approx(0.30, abs=0.01)


# ── Full Pipeline ────────────────────────────────────────────────

def test_pipeline_baseline_no_evidence():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    brain = mod.PolymarketBrain()
    brain.push_market("Test Market", yes_price=0.35)
    result = brain._run_pipeline(brain._market_buffer[-1])
    # No evidence → posterior = prior = 0.35, midrange → no longshot correction
    assert result.adjusted_prob == pytest.approx(0.35, abs=0.01)
    assert result.expected_value == pytest.approx(0.0, abs=0.01)
    assert result.kelly_capped == pytest.approx(0.0, abs=0.001)  # EV=0
    assert result.route == "MAKER"


def test_pipeline_with_evidence():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    brain = mod.PolymarketBrain()
    brain.push_market("Test Market", yes_price=0.35)
    brain.push_evidence("BLS", "Weak jobs report", impact=0.60, reliability=0.80)
    result = brain._run_pipeline(brain._market_buffer[-1])
    # With evidence → posterior > 0.35 → positive EV
    assert result.adjusted_prob > 0.35
    assert result.expected_value > 0
    assert result.should_trade is True


def test_pipeline_demo_4_stages():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    brain = mod.PolymarketBrain()
    results = brain.run_demo()
    assert len(results) == 4
    # Stage 1: baseline, no evidence
    assert results[0].posterior_yes == pytest.approx(0.35, abs=0.01)
    # Stage 4: strongest evidence
    assert results[3].posterior_yes > results[0].posterior_yes
    assert results[3].adjusted_prob > 0.50  # crosses 50% YES
    assert results[3].route == "TAKER"  # high edge → TAKER


@pytest.mark.asyncio
async def test_brain_compute_score_no_data():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    brain = mod.PolymarketBrain()
    signal = await brain.compute_score("BTC")
    assert signal.brain_id == "polymarket_alpha"
    assert signal.score == 0.0
    assert signal.confidence == 0.1
    assert signal.metadata["status"] == "no_market_data"


@pytest.mark.asyncio
async def test_brain_compute_score_with_data():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    brain = mod.PolymarketBrain()
    brain.push_market("Fed Rate Cut", yes_price=0.35)
    brain.push_evidence("BLS", "Weak jobs", impact=0.60, reliability=0.80)
    signal = await brain.compute_score("BTC")
    assert signal.brain_id == "polymarket_alpha"
    # adjusted_prob=0.408 < 0.5, but EV=0.058 > 0 → YES edge → score > 0
    assert signal.score > 0
    # adjusted_prob < 0.5 so direction = -1 (below-midpoint signal)
    assert signal.direction == -1
    assert signal.confidence > 0.3
    assert "posterior_yes" in signal.metadata
    assert "route" in signal.metadata


# ── Config Override ─────────────────────────────────────────────

def test_config_override_changes_behavior():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    # Default: longshot threshold 0.10 → prob=0.08 gets penalized
    cfg_default = mod._PolymarketDefaults()
    corrector_default = mod.LongshotBiasCorrector(cfg_default)
    adj1, _ = corrector_default.correct(0.08)
    # Override: threshold 0.05 → prob=0.08 no longer low end
    cfg_custom = mod._PolymarketDefaults(longshot_low_threshold=0.05)
    corrector_custom = mod.LongshotBiasCorrector(cfg_custom)
    adj2, _ = corrector_custom.correct(0.08)
    # Custom threshold means 0.08 is no longer "longshot" → no penalty
    assert adj1 < adj2


def test_brain_signal_serializable():
    mod = importlib.import_module("orchestrator.brains.polymarket_brain")
    from orchestrator.brains.base_brain import BrainSignal
    import time
    signal = BrainSignal(
        brain_id="polymarket_alpha",
        symbol="BTC",
        score=0.45,
        confidence=0.60,
        weight=0.05,
        direction=1,
        metadata={"ev": 0.25, "route": "MAKER"},
    )
    raw = signal.to_json()
    assert isinstance(raw, bytes)
    data = __import__("json").loads(raw)
    assert data["brain_id"] == "polymarket_alpha"
    assert data["score"] == 0.45
