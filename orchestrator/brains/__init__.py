"""Trinity Architecture — Layer A: 9 Parallel AI Brains.

Each brain publishes its signal score to NATS JetStream subject `signals.raw`.
The Go Orchestrator (Layer B) aggregates these signals with weighted scoring.

Brain 9 (orderflow_nautilus) uses NautilusTrader for L2/L3 orderbook depth.
"""

from .base_brain import BaseBrain, BrainSignal
from .timesfm_brain import TimesFMBrain
from .freqai_brain import FreqAIBrain
from .llm_regime_brain import LLMRegimeBrain
from .microstructure_brain import MicrostructureBrain
from .finbert_brain import FinBERTBrain
from .finrl_brain import FinRLBrain
from .onchain_brain import OnChainBrain
from .statarb_brain import StatArbBrain
from .orderflow_nautilus_brain import OrderFlowNautilusBrain
from .polymarket_brain import PolymarketBrain
from .custom_nn_brain import CustomNNBrain

# ── BRAIN_REGISTRY: name → class mapping ──────────────────────────────────
BRAIN_REGISTRY: dict[str, type[BaseBrain]] = {
    "timesfm":              TimesFMBrain,
    "freqai":               FreqAIBrain,
    "llm_regime":           LLMRegimeBrain,
    "microstructure":       MicrostructureBrain,
    "finbert":              FinBERTBrain,
    "finrl":                FinRLBrain,
    "onchain":             OnChainBrain,
    "statarb":             StatArbBrain,
    "orderflow_nautilus":  OrderFlowNautilusBrain,
    "polymarket_alpha":    PolymarketBrain,
    "custom_nn":           CustomNNBrain,
}

__all__ = [
    "BaseBrain",
    "BrainSignal",
    "BRAIN_REGISTRY",
    "TimesFMBrain",
    "FreqAIBrain",
    "LLMRegimeBrain",
    "MicrostructureBrain",
    "FinBERTBrain",
    "FinRLBrain",
    "OnChainBrain",
    "StatArbBrain",
    "OrderFlowNautilusBrain",
    "PolymarketBrain",
    "CustomNNBrain",
]
