"""Trinity Architecture — Layer A: 8 Parallel AI Brains.

Each brain publishes its signal score to NATS JetStream subject `signals.raw`.
The Go Orchestrator (Layer B) aggregates these signals with weighted scoring.
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

__all__ = [
    "BaseBrain",
    "BrainSignal",
    "TimesFMBrain",
    "FreqAIBrain",
    "LLMRegimeBrain",
    "MicrostructureBrain",
    "FinBERTBrain",
    "FinRLBrain",
    "OnChainBrain",
    "StatArbBrain",
]
