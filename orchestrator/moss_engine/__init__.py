"""Moss Signal Engine & Dynamic Skills Registry.

Unified production-grade module merging 3 repository patterns:
  1. MOSS Trading Skills Factory v1.0.28 — 5-pillar composite signal + reflection
  2. Skills_Registry — dynamic hot-swappable brain registration + state tracking
  3. Krypt-Trader — crypto microstructure (whale/OFI, momentum scanner, contrarian fade,
     book reconciliation, guardrails/kill-switch)

Repository Integration Contract references:
  - https://github.com/moss-site/moss-trade-bot-skills.git  (v1.0.28)
  - https://github.com/smith6jt-cop/Skills_Registry.git    (Skills_Registry)
  - https://github.com/scripflipped/Krypt-Trader.git      (Krypt-Trader)

All path references are dynamically resolved via platformdirs / Path.home().
No hardcoded paths.  Secrets loaded exclusively from .env via python-dotenv.
"""

from .schemas import (
    # Data archetypes
    PillarOutput,
    CompositeSignal,
    SkillMeta,
    SkillState,
    ReconciliationReport,
    GuardrailStatus,
    EvolutionReport,
    ReflectionVerdict,
    WhaleEvent,
    MomentumScanResult,
    FadeSignal,
    # Version
    MOSS_FACTORY_VERSION,
)
from .binance_ingestion import BinanceIngestionEngine
from .composite_engine import MossCompositeEngine
from .krypt_core import KryptCryptoCore
from .reflection_engine import ReflectiveEvolutionLoop
from .skill_registry import MossSkill, SkillRegistry

__all__ = [
    "PillarOutput",
    "CompositeSignal",
    "SkillMeta",
    "SkillState",
    "ReconciliationReport",
    "GuardrailStatus",
    "EvolutionReport",
    "ReflectionVerdict",
    "WhaleEvent",
    "MomentumScanResult",
    "FadeSignal",
    "MossCompositeEngine",
    "BinanceIngestionEngine",
    "KryptCryptoCore",
    "SkillRegistry",
    "ReflectiveEvolutionLoop",
    "MossSkill",
    "MOSS_FACTORY_VERSION",
]
