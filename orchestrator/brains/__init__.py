"""Trinity Architecture Brains Registry.

Aggregates all quantitative brains:
  - FreqAI / TimesFM forecasting
  - LLM regime classification
  - FinBERT sentiment
  - FinRL RLlib PPO engine
  - Polymarket Bayesian alpha
  - OrderFlow (Nautilus.Backtest)
  - Microstructure (orderbook imbalance)

Integrates:
  - MOSS Signal Factory (moss_engine/)
    SkillRegistry: dynamic hot-swappable brains
    MossCompositeEngine: 5-pillar composite signal
    KryptCryptoCore: whale tracker, momentum scanner, contrarian fade
    ReconciliationEngine: boot reconciliation
    ReflectiveEvolutionLoop: self-optimizing parameters

Each brain publishes raw signals to NATS JetStream subject `signals.raw`
-> Nats Aggregator -> Go orchestrator -> execution.

MOSS Engine brains:
  moss_composite — 5-pillar composite (Trend, Momentum, Mean Reversion,
                  Volume, Volatility)
  krypt_core — micro-structure events (whales, 15m volume anomalies,
              contrarian fade decisions)
  moss_reflection — MOSS 7 Reflection Principles + adaptive parameter
                  optimization (±30% bounds)
"""

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
from .odoo_brain import OdooBrain

# MOSS Signal Factory integration (deferred — uncomment when moss_engine/ is production-ready)

# -- BRAIN_REGISTRY: name -> class mapping -------------------------------------------------------
BRAIN_REGISTRY: dict[str, type] = {
    "timesfm": TimesFMBrain,
    "freqai": FreqAIBrain,
    "llm_regime": LLMRegimeBrain,
    "microstructure": MicrostructureBrain,
    "finbert_nlp": FinBERTBrain,
    "finrl_kelly": FinRLBrain,
    "onchain_whale": OnChainBrain,
    "statarb_funding": StatArbBrain,
    "orderflow_nautilus": OrderFlowNautilusBrain,
    "polymarket_alpha": PolymarketBrain,
    "custom_nn": CustomNNBrain,
    "odoo_erp": OdooBrain,
    # MOSS SIGNAL FACTORY (deferred)
    # "moss_composite": MossCompositeEngine,      # 5-pillar composite
    # "krypt_core": KryptCryptoCore,              # microstructure
    # "moss_reflection": ReflectiveEvolutionLoop  # self-optimizing
}

__all__ = [
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
    "OdooBrain",
    "BRAIN_REGISTRY",
    # MOSS Engine (deferred)
    # "SkillRegistry",
    # "MossCompositeEngine",
    # "KryptCryptoCore",
    # "ReconciliationEngine",
    # "ReflectiveEvolutionLoop",
]