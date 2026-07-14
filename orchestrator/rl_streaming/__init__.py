"""
Real-Time RL Streaming Training + Meta-Learning System

Core components:
- streaming_reward.py: ms-level reward computation from trade execution
- online_policy.py: light PPO / bandit style online policy updates
- meta_learner.py: LLM + RL hybrid meta-learning for strategy improvement
- seg_generator.py: Self-Evolving Strategy Generator (SEG v1) with DSL
- backtest_engine.py: Historical replay for strategy evaluation
- evolution_engine.py: Genetic algorithm + RL selection for strategy evolution
- memory_integration.py: Unified memory store integration (Redis/Postgres/Qdrant)
- langgraph_nodes.py: LangGraph integration nodes
"""

from .streaming_reward import (
    StreamingRewardEngine,
    RewardEvent,
    RewardEventType,
    compute_streaming_reward,
    StreamingRewardConfig,
)
from .online_policy import (
    OnlinePolicy,
    OnlineBanditPolicy,
    OnlinePolicyConfig,
    create_online_policy,
    create_bandit_policy,
)
from .meta_learner import MetaLearner, MetaLearningConfig
from .memory_integration import StreamingMemoryManager, StreamingMemoryConfig
from .langgraph_nodes import (
    create_rl_streaming_node,
    create_meta_learning_node,
    create_seg_evolution_node,
    add_streaming_rl_to_graph,
    create_rl_enhanced_pipeline,
)

# CLI entry point
from .cli_commands import cli as rl_streaming_cli

__all__ = [
    # Streaming Reward
    "StreamingRewardEngine",
    "RewardEvent",
    "RewardEventType",
    "compute_streaming_reward",
    "StreamingRewardConfig",
    # Online Policy
    "OnlinePolicy",
    "OnlineBanditPolicy",
    "OnlinePolicyConfig",
    "create_online_policy",
    "create_bandit_policy",
    # Meta Learning
    "MetaLearner",
    "MetaLearningConfig",
    # Memory Integration
    "StreamingMemoryManager",
    "StreamingMemoryConfig",
    # LangGraph Nodes
    "create_rl_streaming_node",
    "create_meta_learning_node",
    "create_seg_evolution_node",
    "add_streaming_rl_to_graph",
    "create_rl_enhanced_pipeline",
    # CLI
    "rl_streaming_cli",
]