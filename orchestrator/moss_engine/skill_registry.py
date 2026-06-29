"""Dynamic Skills Registry — hot-swappable brain/skill manager.

Architectural pattern from Skills_Registry.git:
  - Register skills/brains with full metadata (version, status, metrics, flags)
  - Resolve by name at runtime — late-binding
  - Hot-swap: atomically replace a skill implementation without losing state
  - Track execution metrics (invocations, errors, latency, PnL)
  - State memory fix: _state_buffers keyed by (asset, interval) — never reset

Security: API keys loaded from .env via dotenv.  No hardcoded secrets.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .schemas import (
    MOSS_FACTORY_VERSION,
    CompositeSignal,
    SkillMeta,
    SkillState,
)

logger = logging.getLogger(__name__)

# ── .env auto-load ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parents[2]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════
# Abstract Skill Interface (MOSS execution contract)
# ══════════════════════════════════════════════════════════════════════

class MossSkill(ABC):
    """Abstract base for all MOSS-registered skills/brains.

    Every skill must implement:
      - skill_name: str property
      - execute(symbol, interval, ohlcv) -> CompositeSignal | dict
    Optional overrides:
      - warmup() — async initialisation
      - cooldown() — async cleanup
      - get_state_key(symbol, interval) -> str
    """

    @property
    @abstractmethod
    def skill_name(self) -> str:
        """Unique skill identifier used as registry key."""
        ...

    @abstractmethod
    def execute(self, symbol: str, interval: str,
                ohlcv: List[Dict[str, Any]]) -> Any:
        """Execute the skill and return a signal dict or CompositeSignal."""
        ...

    def get_state_key(self, symbol: str, interval: str) -> str:
        """Produce a partition key for _state_buffers persistence."""
        return f"{symbol}_{interval}"

    async def warmup(self) -> None:
        """Optional async initialisation hook."""
        return None

    async def cooldown(self) -> None:
        """Optional async cleanup hook."""
        return None


# ══════════════════════════════════════════════════════════════════════
# SECTION A:  SkillRegistry
# ══════════════════════════════════════════════════════════════════════

class SkillRegistry:
    """Dynamic hot-swappable skill/brain registry.

    Provides:
      - register()      — register a skill class or instance with metadata
      - resolve()       — look up an active skill by name
      - hot_swap()      — atomically replace skill implementation preserving state
      - deregister()    — remove a skill and its buffers
      - list_skills()   — enumerate all registered skills with status
      - execute()       — run a named skill with full metrics tracking
      - get_state()     — read mutable runtime state
      - update_state()  — patch runtime state fields
      - arm_kill_switch() — flip master kill-switch (zero-initializes all outputs)

    State Memory Fix:
      _state_buffers is a multi-level dict: _state_buffers[state_key][field_name] -> value
      Partitioned by asset + interval.  Sliding windows and indicator matrices
      persist across ticks and are NEVER reset to defaults during live execution.
    """

    def __init__(self, version: str = MOSS_FACTORY_VERSION) -> None:
        self._version: str = version
        # name -> MossSkill instance
        self._instances: Dict[str, MossSkill] = {}
        # name -> SkillMeta (immutable metadata snapshot)
        self._metas: Dict[str, SkillMeta] = {}
        # name -> SkillState (mutable runtime state)
        self._states: Dict[str, SkillState] = {}
        # State Memory Fix: persistent multi-level buffer keyed by (symbol_interval)
        self._state_buffers: Dict[str, Dict[str, Any]] = {}
        # name -> original class for hot-swap reference
        self._classes: Dict[str, Type[MossSkill]] = {}
        # Master kill-switch
        self._master_kill_switch: bool = False

    # ── Registration ────────────────────────────────────────────

    def register(
        self,
        name: str,
        skill_cls_or_instance: Type[MossSkill] | MossSkill,
        *,
        version: str = MOSS_FACTORY_VERSION,
        description: str = "",
        tags: Tuple[str, ...] = (),
        hot_swap_enabled: bool = True,
        kill_switch_armed: bool = False,
        max_daily_loss_pct: float = 5.0,
        max_take_profit_pct: float = 50.0,
    ) -> None:
        """Register a skill/brain under the given name.

        Args:
            name: unique registry key
            skill_cls_or_instance: class (instantiated here) or ready instance
            version: semantic version string
            description: human-readable purpose
            tags: categorisation labels for filtering
            hot_swap_enabled: allow hot_swap() on this skill
            kill_switch_armed: whether kill-switch applies to this skill
            max_daily_loss_pct: hard daily trailing stop-loss threshold
            max_take_profit_pct: rigid max take-profit level
        """
        if name in self._instances:
            logger.warning("[SkillRegistry] '%s' already registered — overwriting", name)

        # Instantiate if a class was passed
        if isinstance(skill_cls_or_instance, type):
            instance = skill_cls_or_instance()
            cls_ref = skill_cls_or_instance
        else:
            instance = skill_cls_or_instance
            cls_ref = type(skill_cls_or_instance)

        self._instances[name] = instance
        self._classes[name] = cls_ref

        meta = SkillMeta(
            name=name,
            version=version,
            description=description,
            tags=tags,
            hot_swap_enabled=hot_swap_enabled,
            kill_switch_armed=kill_switch_armed,
            max_daily_loss_pct=max_daily_loss_pct,
            max_take_profit_pct=max_take_profit_pct,
        )
        self._metas[name] = meta
        self._states[name] = SkillState()

        # Ensure initial state buffer entry via the skill's key function
        # (empty for now — populated on first execute)
        logger.info(
            "[SkillRegistry] Registered '%s' v%s (hot_swap=%s)",
            name, version, hot_swap_enabled,
        )

    # ── Resolution ──────────────────────────────────────────────

    def resolve(self, name: str) -> Optional[MossSkill]:
        """Look up an active skill by name.  Returns None if not found or killed."""
        if self._master_kill_switch:
            logger.error("[SkillRegistry] Master kill-switch ACTIVE — resolve returning None")
            return None
        state = self._states.get(name)
        if state is not None and state.kill_switch_active:
            logger.warning("[SkillRegistry] '%s' individual kill-switch active — returning None", name)
            return None
        instance = self._instances.get(name)
        if instance is None:
            logger.warning("[SkillRegistry] '%s' not found in registry", name)
        return instance

    # ── Hot-Swap ────────────────────────────────────────────────

    def hot_swap(self, name: str, new_skill_cls_or_instance: Type[MossSkill] | MossSkill) -> bool:
        """Atomically replace skill implementation, preserving state buffers.

        Returns True if swap succeeded, False if not allowed or not found.
        """
        meta = self._metas.get(name)
        if meta is None:
            logger.error("[SkillRegistry] hot_swap: '%s' not registered", name)
            return False
        if not meta.hot_swap_enabled:
            logger.error("[SkillRegistry] hot_swap: '%s' has hot_swap_enabled=False", name)
            return False
        # Preserve state and buffers
        saved_state = self._states.get(name, SkillState())
        saved_buffers = dict(self._state_buffers)  # shallow copy of buffer keys

        # Instantiate new
        if isinstance(new_skill_cls_or_instance, type):
            new_instance = new_skill_cls_or_instance()
            cls_ref = new_skill_cls_or_instance
        else:
            new_instance = new_skill_cls_or_instance
            cls_ref = type(new_skill_cls_or_instance)

        # Atomic swap
        self._instances[name] = new_instance
        self._classes[name] = cls_ref
        self._states[name] = saved_state
        self._state_buffers.update(saved_buffers)

        logger.info("[SkillRegistry] Hot-swapped '%s' → %s (state preserved)", name, cls_ref.__name__)
        return True

    # ── Deregistration ───────────────────────────────────────────

    def deregister(self, name: str) -> bool:
        """Remove a skill and its associated state buffers."""
        if name not in self._instances:
            logger.warning("[SkillRegistry] deregister: '%s' not found", name)
            return False
        del self._instances[name]
        self._metas.pop(name, None)
        self._states.pop(name, None)
        self._classes.pop(name, None)
        # Note: state_buffers entries keyed by skill's state_key, not registry name
        logger.info("[SkillRegistry] Deregistered '%s' (buffers may persist for other skills)", name)
        return True

    # ── Listing ─────────────────────────────────────────────────

    def list_skills(self) -> List[Dict[str, Any]]:
        """Return a summary list of all registered skills with metadata."""
        result: List[Dict[str, Any]] = []
        for name in sorted(self._instances.keys()):
            meta = self._metas.get(name)
            state = self._states.get(name)
            entry: Dict[str, Any] = {
                "name": name,
                "version": meta.version if meta else "unknown",
                "status": state.status if state else "unknown",
                "invocations": state.total_invocations if state else 0,
                "errors": state.total_errors if state else 0,
                "avg_latency_ms": state.avg_latency_ms if state else 0.0,
                "kill_switch_active": state.kill_switch_active if state else False,
            }
            result.append(entry)
        return result

    # ── Execution with Metrics ──────────────────────────────────

    def execute(self, name: str, symbol: str, interval: str,
                ohlcv: List[Dict[str, Any]]) -> Any:
        """Execute a named skill with full metrics tracking.

        Records latency, invocation count, and error count.
        If master kill-switch is active, returns None (zero-initialized output).
        """
        if self._master_kill_switch:
            logger.error("[SkillRegistry] Master kill-switch ACTIVE — execute returning None for '%s'", name)
            return None

        state = self._states.get(name)
        if state is None:
            logger.error("[SkillRegistry] execute: '%s' not registered", name)
            return None
        if state.kill_switch_active:
            logger.warning("[SkillRegistry] '%s' individual kill-switch active", name)
            return None

        instance = self._instances.get(name)
        if instance is None:
            logger.error("[SkillRegistry] execute: '%s' instance missing", name)
            return None

        # Guardrail pre-check: daily loss limit
        meta = self._metas.get(name)
        if meta is not None and state.daily_drawdown_pct >= meta.max_daily_loss_pct:
            logger.warning(
                "[SkillRegistry] '%s' daily drawdown %.2f%% >= max %.2f%% — skipping execution",
                name, state.daily_drawdown_pct, meta.max_daily_loss_pct,
            )
            state.kill_switch_active = True
            return None

        t_start = time.monotonic()
        try:
            result = instance.execute(symbol, interval, ohlcv)
            elapsed_ms = (time.monotonic() - t_start) * 1000.0
            state.total_invocations += 1
            state.last_latency_ms = elapsed_ms
            state.avg_latency_ms = (
                (state.avg_latency_ms * (state.total_invocations - 1) + elapsed_ms)
                / state.total_invocations
            )
            state.last_execution_ts = int(time.time() * 1000)
            state.status = "active"
            return result
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t_start) * 1000.0
            state.total_errors += 1
            state.total_invocations += 1
            state.last_latency_ms = elapsed_ms
            state.status = "error"
            logger.error("[SkillRegistry] '%s' execution failed: %s", name, exc, exc_info=True)
            return None

    # ── State Access & Mutation ─────────────────────────────────

    def get_state(self, name: str) -> Optional[SkillState]:
        """Retrieve mutable runtime state for a named skill."""
        return self._states.get(name)

    def update_state(self, name: str, **fields: Any) -> bool:
        """Patch runtime state fields for a named skill.

        Example: registry.update_state("krypt_core", daily_pnl=-150.0)
        Returns True if patched, False if skill not found.
        """
        state = self._states.get(name)
        if state is None:
            return False
        for key, value in fields.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                logger.warning("[SkillRegistry] update_state: '%s' has no field '%s'", name, key)
        return True

    # ── State Memory Fix: Persistent Buffers ─────────────────────

    def get_state_buffer(self, state_key: str) -> Dict[str, Any]:
        """Retrieve the persistent state buffer for a given (symbol, interval) key.

        Creates an empty dict on first access — never returns a stale default.
        This is the state memory fix: sliding windows, alpha matrices, and
        indicator histories survive across ticks and hot-swaps.
        """
        if state_key not in self._state_buffers:
            self._state_buffers[state_key] = {}
        return self._state_buffers[state_key]

    def set_state_buffer_field(self, state_key: str, field_name: str, value: Any) -> None:
        """Set a single field in the persistent state buffer."""
        buf = self.get_state_buffer(state_key)
        buf[field_name] = value

    def get_state_buffer_field(self, state_key: str, field_name: str,
                               default: Any = None) -> Any:
        """Get a single field from the persistent state buffer."""
        buf = self.get_state_buffer(state_key)
        return buf.get(field_name, default)

    # ── Kill-Switch ─────────────────────────────────────────────

    def arm_kill_switch(self, active: bool) -> None:
        """Flip the master kill-switch.

        When active=True, all resolve() and execute() calls return None.
        All output signals are zero-initialized.
        """
        self._master_kill_switch = active
        if active:
            logger.critical("[SkillRegistry] MASTER KILL-SWITCH ARMED — all outputs zeroed")
            # Zero-initialize all skill states
            for name, state in self._states.items():
                state.kill_switch_active = True
                state.status = "killed"
        else:
            logger.info("[SkillRegistry] Master kill-switch DISARMED — resuming normal operation")
            for name, state in self._states.items():
                state.kill_switch_active = False
                if state.status == "killed":
                    state.status = "idle"

    @property
    def master_kill_switch(self) -> bool:
        return self._master_kill_switch

    # ── Utility ─────────────────────────────────────────────────

    @property
    def version(self) -> str:
        return self._version

    @property
    def registered_count(self) -> int:
        return len(self._instances)

    def get_meta(self, name: str) -> Optional[SkillMeta]:
        return self._metas.get(name)

    def clear_daily_state(self) -> None:
        """Reset daily PnL tracking counters (call at UTC midnight)."""
        for state in self._states.values():
            state.daily_pnl = 0.0
            state.daily_trailing_high = 0.0
            state.daily_drawdown_pct = 0.0
            state.consecutive_losses = 0
            if state.status == "killed" and not self._master_kill_switch:
                state.status = "idle"
                state.kill_switch_active = False
        logger.info("[SkillRegistry] Daily state counters cleared")
