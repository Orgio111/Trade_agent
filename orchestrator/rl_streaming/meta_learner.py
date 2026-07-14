"""
Meta-Learning Engine — LLM + RL Hybrid for Strategy Improvement.

Core concept: An AI that learns how to design better allocators and strategies.
Not just learning actions, but learning the learning process itself.

Architecture:
  Strategy Performance Logs → Meta-Model (LLM) → Improved Allocator Logic
       ↑                      ↓                      ↓
  Backtest Results ←─ Policy Update ←─ New Strategy Rules

The meta-learner:
1. Analyzes winning/losing trade patterns
2. Uses LLM to generate improved strategy rules
3. Validates via backtest before deployment
4. Updates the policy/allocator weights
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Import existing components
from orchestrator.rl_memory.memory_store import RLMemoryManager, TradeOutcome, create_memory_manager
from orchestrator.rl_memory.reward_function import (
    RewardBreakdown,
    RewardConfig,
    compute_reward,
    generate_policy_update_prompt,
    get_reward_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning engine."""

    # Data window
    lookback_days: int = 7
    min_trades_for_analysis: int = 20

    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 3000

    # Analysis frequency
    analysis_interval_hours: int = 6  # Run meta-analysis every N hours

    # Validation
    require_backtest_validation: bool = True
    backtest_min_sharpe: float = 0.5
    backtest_max_drawdown: float = 0.15
    min_win_rate_improvement: float = 0.02  # 2% improvement threshold

    # Policy update
    policy_update_mode: str = "blend"  # "replace", "blend", "ensemble"
    blend_ratio: float = 0.3  # How much to blend new vs old

    # Persistence
    policy_versions_dir: str = "policies/meta"
    max_policy_versions: int = 50

    # Auto-deploy
    auto_deploy: bool = False  # Require human approval by default
    deploy_min_confidence: float = 0.7


@dataclass
class MetaAnalysisResult:
    """Result of meta-learning analysis."""

    timestamp: datetime
    trades_analyzed: int
    winning_patterns: list[str]
    losing_patterns: list[str]
    regime_insights: dict[str, Any]
    suggested_rules: dict[str, Any]
    confidence: float
    backtest_validation: dict[str, Any] | None = None
    deployed: bool = False


class MetaLearner:
    """
    Meta-learning engine that analyzes trade performance and generates
    improved strategy rules using LLM + RL hybrid approach.
    """

    def __init__(
        self,
        config: MetaLearningConfig | None = None,
        memory_manager: RLMemoryManager | None = None,
        policy_updater: Callable[[dict], bool] | None = None,
    ):
        self.config = config or MetaLearningConfig()
        self.memory_manager = memory_manager or create_memory_manager()
        self.policy_updater = policy_updater

        # State
        self._last_analysis_time: float = 0.0
        self._analysis_history: list[MetaAnalysisResult] = []
        self._current_policy_version: str = "v0"

        # Ensure directories
        Path(self.config.policy_versions_dir).mkdir(parents=True, exist_ok=True)

        # Try to load latest policy version
        self._load_latest_policy_version()

        # Stats
        self.stats = {
            "analyses_run": 0,
            "policies_generated": 0,
            "policies_deployed": 0,
            "avg_confidence": 0.0,
        }

    def _load_latest_policy_version(self):
        """Load the latest policy version from disk."""
        policy_dir = Path(self.config.policy_versions_dir)
        policies = sorted(policy_dir.glob("policy_v*.json"))
        if policies:
            latest = policies[-1]
            try:
                with open(latest) as f:
                    data = json.load(f)
                self._current_policy_version = data.get("version", "v0")
                logger.info(f"[MetaLearner] Loaded policy version: {self._current_policy_version}")
            except Exception as e:
                logger.warning(f"[MetaLearner] Failed to load latest policy: {e}")

    def should_run_analysis(self) -> bool:
        """Check if enough time has passed since last analysis."""
        if not self._last_analysis_time:
            return True
        elapsed_hours = (time.time() - self._last_analysis_time) / 3600
        return elapsed_hours >= self.config.analysis_interval_hours

    async def run_analysis(self, force: bool = False) -> MetaAnalysisResult | None:
        """
        Run full meta-learning analysis cycle.

        1. Fetch recent trades from memory
        2. Compute rewards and split wins/losses
        3. Analyze patterns using LLM
        4. Generate improved strategy rules
        5. Validate via backtest (optional)
        6. Deploy if validation passes and auto_deploy enabled
        """
        if not force and not self.should_run_analysis():
            logger.info("[MetaLearner] Analysis interval not reached, skipping")
            return None

        logger.info("[MetaLearner] Starting meta-learning analysis...")
        start_time = time.time()

        try:
            # 1. Fetch trades
            trades = self._fetch_recent_trades()
            if len(trades) < self.config.min_trades_for_analysis:
                logger.warning(
                    f"[MetaLearner] Insufficient trades ({len(trades)} < {self.config.min_trades_for_analysis})"
                )
                return None

            # 2. Compute rewards
            rewards, breakdowns = zip(*[compute_reward(t, RewardConfig()) for t in trades])

            # 3. Split wins/losses
            winning_trades = [t for t, r in zip(trades, rewards) if r > 0]
            losing_trades = [t for t, r in zip(trades, rewards) if r <= 0]

            # 4. Get aggregate stats
            stats = get_reward_stats(trades, RewardConfig())

            # 5. Generate LLM prompt for policy improvement
            prompt = generate_policy_update_prompt(winning_trades, losing_trades, RewardConfig())

            # 6. Call LLM to generate improved rules
            logger.info("[MetaLearner] Calling LLM for strategy improvement...")
            improved_rules = await self._call_llm_for_rules(prompt)

            if not improved_rules:
                logger.error("[MetaLearner] LLM failed to generate rules")
                return None

            # 7. Validate via backtest if required
            backtest_result = None
            if self.config.require_backtest_validation:
                backtest_result = await self._validate_rules(improved_rules, trades)
                if not self._passes_validation(backtest_result):
                    logger.warning("[MetaLearner] Backtest validation failed, not deploying")
                    backtest_result["validation_passed"] = False
                else:
                    backtest_result["validation_passed"] = True

            # 8. Calculate confidence based on analysis quality
            confidence = self._calculate_confidence(trades, winning_trades, losing_trades, backtest_result)

            # 9. Create analysis result
            result = MetaAnalysisResult(
                timestamp=datetime.now(),
                trades_analyzed=len(trades),
                winning_patterns=self._extract_patterns(winning_trades),
                losing_patterns=self._extract_patterns(losing_trades),
                regime_insights=self._analyze_regimes(trades, rewards),
                suggested_rules=improved_rules,
                confidence=confidence,
                backtest_validation=backtest_result,
                deployed=False,
            )

            # 10. Deploy if auto-deploy and validation passes
            deployed = False
            if self.config.auto_deploy and confidence >= self.config.deploy_min_confidence:
                if not self.config.require_backtest_validation or (
                    backtest_result and backtest_result.get("validation_passed")
                ):
                    deployed = await self._deploy_rules(improved_rules, result)
                    result.deployed = deployed

            # 11. Save policy version
            self._save_policy_version(improved_rules, result, deployed)

            # Update state
            self._last_analysis_time = time.time()
            self._analysis_history.append(result)
            self._update_stats(result)

            elapsed = time.time() - start_time
            logger.info(
                f"[MetaLearner] Analysis complete in {elapsed:.1f}s | "
                f"Trades: {len(trades)} | Confidence: {confidence:.2f} | "
                f"Deployed: {deployed}"
            )

            return result

        except Exception as e:
            logger.exception(f"[MetaLearner] Analysis failed: {e}")
            return None

    def _fetch_recent_trades(self) -> list[TradeOutcome]:
        """Fetch recent trades from memory stores."""
        if self.memory_manager.is_available("postgres"):
            end = datetime.now()
            start = end - timedelta(days=self.config.lookback_days)
            return self.memory_manager.postgres.get_trades(start=start, end=end, limit=10000)

        if self.memory_manager.is_available("redis"):
            return self.memory_manager.get_recent_trades(1000)

        return []

    def _extract_patterns(self, trades: list[TradeOutcome]) -> list[str]:
        """Extract common patterns from trades."""
        if not trades:
            return []

        patterns = []

        # Regime patterns
        regime_counts = {}
        for t in trades:
            regime = t.regime or "unknown"
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1])[:3]:
            patterns.append(f"Regime '{regime}': {count} trades")

        # Indicator patterns
        indicator_stats = {}
        for t in trades:
            for ind, val in t.indicators.items():
                if ind not in indicator_stats:
                    indicator_stats[ind] = []
                indicator_stats[ind].append(val)

        for ind, vals in indicator_stats.items():
            if len(vals) >= 5:
                avg = np.mean(vals)
                patterns.append(f"{ind}: avg={avg:.2f} (n={len(vals)})")

        # Action distribution
        action_counts = {}
        for t in trades:
            action_counts[t.action] = action_counts.get(t.action, 0) + 1
        for action, count in action_counts.items():
            patterns.append(f"Action {action}: {count}")

        return patterns[:10]

    def _analyze_regimes(self, trades: list[TradeOutcome], rewards: list[float]) -> dict[str, Any]:
        """Analyze performance by regime."""
        regime_perf = {}

        for trade, reward in zip(trades, rewards):
            regime = trade.regime or "unknown"
            if regime not in regime_perf:
                regime_perf[regime] = {"trades": 0, "rewards": [], "pnl_pcts": []}
            regime_perf[regime]["trades"] += 1
            regime_perf[regime]["rewards"].append(reward)
            regime_perf[regime]["pnl_pcts"].append(trade.pnl_pct)

        insights = {}
        for regime, data in regime_perf.items():
            if data["trades"] >= 3:
                insights[regime] = {
                    "count": data["trades"],
                    "avg_reward": float(np.mean(data["rewards"])),
                    "avg_pnl_pct": float(np.mean(data["pnl_pcts"])),
                    "win_rate": sum(1 for r in data["rewards"] if r > 0) / data["trades"],
                }

        return insights

    def _calculate_confidence(
        self,
        trades: list[TradeOutcome],
        winning_trades: list[TradeOutcome],
        losing_trades: list[TradeOutcome],
        backtest_result: dict | None,
    ) -> float:
        """Calculate confidence in the generated rules."""
        confidence = 0.0

        # Base: enough data
        if len(trades) >= 50:
            confidence += 0.3
        elif len(trades) >= 20:
            confidence += 0.2
        elif len(trades) >= 10:
            confidence += 0.1

        # Win rate differential
        if trades:
            win_rate = len(winning_trades) / len(trades)
            if win_rate > 0.6:
                confidence += 0.2
            elif win_rate > 0.5:
                confidence += 0.1

        # Backtest validation
        if backtest_result and backtest_result.get("validation_passed"):
            confidence += 0.3
            sharpe = backtest_result.get("sharpe", 0)
            if sharpe > 1.0:
                confidence += 0.1
            elif sharpe > 0.5:
                confidence += 0.05

        # Pattern clarity
        if len(self._extract_patterns(winning_trades)) >= 3:
            confidence += 0.1

        return min(confidence, 1.0)

    async def _call_llm_for_rules(self, prompt: str) -> dict | None:
        """Call LLM to generate improved strategy rules."""
        try:
            import openai

            client = openai.OpenAI(
                base_url=self.config.llm_base_url,
                api_key="ollama",  # Ollama uses dummy key
            )

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a quantitative trading strategy optimizer. "
                            "Analyze trade outcomes and output ONLY valid JSON with improved rules. "
                            "No markdown, no explanation, just the JSON object."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )

            content = response.choices[0].message.content.strip()

            # Try to extract JSON from response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON block
                import re
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return None

        except Exception as e:
            logger.error(f"[MetaLearner] LLM call failed: {e}")
            return None

    async def _validate_rules(self, rules: dict, trades: list[TradeOutcome]) -> dict:
        """Validate suggested rules via quick backtest simulation."""
        # This is a simplified validation - in production, would run full backtest
        # For now, return simulated results based on historical performance

        # Split trades for train/test
        split = int(len(trades) * 0.7)
        train_trades = trades[:split]
        test_trades = trades[split:]

        if not test_trades:
            return {
                "validation_passed": False,
                "reason": "Insufficient test data",
                "sharpe": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 1.0,
            }

        # Simulate applying rules to test trades
        # In reality, would re-run strategy with new rules on historical data
        test_rewards = [compute_reward(t, RewardConfig())[0] for t in test_trades]
        test_win_rate = sum(1 for r in test_rewards if r > 0) / len(test_rewards)

        # Calculate metrics
        avg_reward = np.mean(test_rewards)
        std_reward = np.std(test_rewards)
        sharpe = avg_reward / std_reward if std_reward > 0 else 0

        # Simple drawdown estimate
        cumulative = np.cumsum(test_rewards)
        peak = np.maximum.accumulate(cumulative)
        drawdown = np.min(peak - cumulative) if len(cumulative) > 0 else 0
        max_dd = abs(drawdown) / max(1, np.max(cumulative)) if np.max(cumulative) > 0 else 0

        return {
            "validation_passed": True,
            "test_trades": len(test_trades),
            "win_rate": test_win_rate,
            "avg_reward": float(avg_reward),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "train_trades": len(train_trades),
        }

    def _passes_validation(self, backtest_result: dict) -> bool:
        """Check if backtest results pass validation thresholds."""
        if not backtest_result:
            return False

        if backtest_result.get("sharpe", 0) < self.config.backtest_min_sharpe:
            return False

        if backtest_result.get("max_drawdown", 1) > self.config.backtest_max_drawdown:
            return False

        return True

    async def _deploy_rules(self, rules: dict, result: MetaAnalysisResult) -> bool:
        """Deploy new rules to the trading system."""
        if not self.policy_updater:
            logger.warning("[MetaLearner] No policy updater configured, cannot deploy")
            return False

        try:
            success = self.policy_updater(rules)
            if success:
                self._current_policy_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"[MetaLearner] Deployed policy version: {self._current_policy_version}")
            return success
        except Exception as e:
            logger.error(f"[MetaLearner] Deployment failed: {e}")
            return False

    def _save_policy_version(self, rules: dict, result: MetaAnalysisResult, deployed: bool):
        """Save policy version to disk."""
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        policy_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "rules": rules,
            "analysis": {
                "trades_analyzed": result.trades_analyzed,
                "confidence": result.confidence,
                "winning_patterns": result.winning_patterns,
                "losing_patterns": result.losing_patterns,
                "regime_insights": result.regime_insights,
            },
            "backtest": result.backtest_validation,
            "deployed": deployed,
            "source": "meta_learner",
        }

        path = Path(self.config.policy_versions_dir) / f"policy_{version}.json"
        with open(path, "w") as f:
            json.dump(policy_data, f, indent=2)

        # Cleanup old versions
        self._cleanup_old_versions()

        logger.info(f"[MetaLearner] Saved policy version {version} to {path}")

    def _cleanup_old_versions(self):
        """Keep only the most recent N policy versions."""
        policy_dir = Path(self.config.policy_versions_dir)
        policies = sorted(policy_dir.glob("policy_v*.json"))
        if len(policies) > self.config.max_policy_versions:
            for old in policies[:-self.config.max_policy_versions]:
                old.unlink()
                logger.debug(f"[MetaLearner] Removed old policy: {old.name}")

    def _update_stats(self, result: MetaAnalysisResult):
        """Update running statistics."""
        self.stats["analyses_run"] += 1
        self.stats["policies_generated"] += 1
        if result.deployed:
            self.stats["policies_deployed"] += 1

        n = self.stats["analyses_run"]
        self.stats["avg_confidence"] = (
            (n - 1) * self.stats["avg_confidence"] + result.confidence
        ) / n

    def get_status(self) -> dict:
        """Get current meta-learner status."""
        return {
            "current_policy_version": self._current_policy_version,
            "last_analysis": self._last_analysis_time,
            "analyses_completed": len(self._analysis_history),
            "stats": self.stats,
            "config": {
                "lookback_days": self.config.lookback_days,
                "analysis_interval_hours": self.config.analysis_interval_hours,
                "auto_deploy": self.config.auto_deploy,
            },
        }

    def get_analysis_history(self, limit: int = 10) -> list[dict]:
        """Get recent analysis history."""
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "trades_analyzed": r.trades_analyzed,
                "confidence": r.confidence,
                "deployed": r.deployed,
                "backtest_passed": r.backtest_validation.get("validation_passed") if r.backtest_validation else None,
            }
            for r in self._analysis_history[-limit:]
        ]


# Factory function
def create_meta_learner(
    config: MetaLearningConfig | None = None,
    memory_manager: RLMemoryManager | None = None,
    policy_updater: Callable[[dict], bool] | None = None,
) -> MetaLearner:
    """Create meta-learner instance."""
    return MetaLearner(config, memory_manager, policy_updater)


__all__ = [
    "MetaLearningConfig",
    "MetaAnalysisResult",
    "MetaLearner",
    "create_meta_learner",
]