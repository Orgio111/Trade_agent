"""QUANTEX AI Orchestrator — Scalping engine, chart segmentation, GPU inference, multi-agent pipeline, PPO portfolio manager."""

from .ppo_portfolio_manager import PPOPortfolioManager, PortfolioManagerResult

__all__ = [
    "PPOPortfolioManager",
    "PortfolioManagerResult",
    # Lazy-loaded (see __getattr__ below)
    "CandleBuffer",
    "VLMAgent",
    "VLMResult",
    "RAGAgent",
    "RAGContext",
    "NATSLangGraphBridge",
    "ScalpingEngine",
    "ScalpDecision",
    "scalp_decision",
    "ChartSegmenter",
    "ChartFeatures",
    "segment_chart",
    "GPUOptimizer",
    "LocalInferenceClient",
]


def __getattr__(name: str):
    """Lazy imports — avoids breaking the package when optional deps are missing."""
    _lazy = {
        "CandleBuffer": ".candle_buffer",
        "VLMAgent": ".vlm_agent",
        "VLMResult": ".vlm_agent",
        "RAGAgent": ".rag_agent",
        "RAGContext": ".rag_agent",
        "NATSLangGraphBridge": ".nats_langgraph_bridge",
        "ScalpingEngine": ".scalping_engine",
        "ScalpDecision": ".scalping_engine",
        "scalp_decision": ".scalping_engine",
        "ChartSegmenter": ".chart_segmentation",
        "ChartFeatures": ".chart_segmentation",
        "segment_chart": ".chart_segmentation",
        "GPUOptimizer": ".gpu_optimizer",
        "LocalInferenceClient": ".local_inference_server",
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
