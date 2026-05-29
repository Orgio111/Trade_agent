"""
Intelligent compute scheduler — GPU circuit breaker with automatic CPU fallback.

Port of the Sentinel-X scheduler (sentinel-x/python/core/scheduler.py).
Routes LLM / embed / RL-inference tasks to GPU (NIM) when healthy; fails over
to lightweight CPU heuristics when GPU p95 latency exceeds 200ms for N
consecutive calls.

Usage
-----
    result = await route(
        TaskType.LLM,
        gpu_fn=lambda: llm_chat([...]),
        cpu_fn=lambda: _cpu_fallback(...),
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from prometheus_client import Counter, Gauge, Histogram

log = logging.getLogger(__name__)

T = TypeVar("T")

# ── Prometheus metrics ────────────────────────────────────────────────────────
GPU_LATENCY = Histogram(
    "scheduler_gpu_latency_seconds",
    "GPU (NIM) inference latency",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)
CPU_LATENCY = Histogram(
    "scheduler_cpu_latency_seconds",
    "CPU inference latency",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
)
ROUTE_COUNTER = Counter(
    "scheduler_routes_total",
    "Routing decisions by task type and backend",
    ["task_type", "backend"],
)
CB_STATE = Gauge(
    "scheduler_circuit_breaker_open",
    "1 = GPU circuit breaker open, 0.5 = half-open, 0 = closed",
)

# ── Task types ────────────────────────────────────────────────────────────────


class TaskType(str, Enum):
    LLM = "llm"  # → GPU (NIM)
    EMBED = "embed"  # → GPU (NIM)
    INDICATOR = "indicator"  # → CPU (NumPy)
    FEATURE = "feature"  # → CPU
    RL_INFER = "rl_infer"  # → GPU (Triton/Ray Serve)


# ── Circuit breaker ───────────────────────────────────────────────────────────


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class GPUCircuitBreaker:
    """Trips when GPU latency spikes above 200ms for N consecutive calls.

    - CLOSED: normal GPU operation
    - OPEN: GPU degraded — all calls routed to CPU
    - HALF_OPEN: after `reset_timeout_s` seconds, probes with one GPU call
    """

    SPIKE_THRESHOLD_S = 0.200
    CONSECUTIVE_SPIKES = 3
    RESET_TIMEOUT_S = 10.0

    def __init__(self) -> None:
        self._state = CircuitState.CLOSED
        self._spike_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0.0)
            if elapsed > self.RESET_TIMEOUT_S:
                self._state = CircuitState.HALF_OPEN
                CB_STATE.set(0.5)
        return self._state

    def record_latency(self, latency_s: float) -> None:
        if self.state == CircuitState.HALF_OPEN:
            if latency_s < self.SPIKE_THRESHOLD_S:
                self._state = CircuitState.CLOSED
                self._spike_count = 0
                CB_STATE.set(0)
                log.info("GPU circuit breaker CLOSED — latency recovered")
            else:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                CB_STATE.set(1)
            return

        if latency_s > self.SPIKE_THRESHOLD_S:
            self._spike_count += 1
            if self._spike_count >= self.CONSECUTIVE_SPIKES and self._state == CircuitState.CLOSED:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                CB_STATE.set(1)
                log.warning(
                    "GPU circuit breaker OPENED — p95 latency %.0fms > 200ms threshold",
                    latency_s * 1000,
                )
        else:
            self._spike_count = 0

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN


# ── Singleton breaker ─────────────────────────────────────────────────────────
_gpu_cb = GPUCircuitBreaker()


async def route(
    task_type: TaskType,
    gpu_fn: Callable[..., Awaitable[T]],
    cpu_fn: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Route *task_type* to the appropriate backend.

    CPU-native tasks (INDICATOR, FEATURE) always go CPU.
    GPU tasks (LLM, EMBED, RL_INFER) go GPU unless the circuit breaker is open,
    in which case the CPU fallback is used transparently.

    Parameters
    ----------
    task_type:
        Type of computation to route.
    gpu_fn:
        Async callable for GPU-backed execution (NIM / Triton / Ray Serve).
    cpu_fn:
        Async callable for CPU-based fallback (local stats / heuristic).
    """
    cpu_native = task_type in (TaskType.INDICATOR, TaskType.FEATURE)

    if cpu_native or _gpu_cb.is_open:
        backend = "cpu"
        ROUTE_COUNTER.labels(task_type=task_type.value, backend=backend).inc()
        t0 = time.monotonic()
        try:
            result = await cpu_fn(*args, **kwargs)
            return result
        finally:
            CPU_LATENCY.observe(time.monotonic() - t0)

    backend = "gpu"
    ROUTE_COUNTER.labels(task_type=task_type.value, backend=backend).inc()
    t0 = time.monotonic()
    try:
        result = await gpu_fn(*args, **kwargs)
        latency = time.monotonic() - t0
        GPU_LATENCY.observe(latency)
        _gpu_cb.record_latency(latency)
        return result
    except Exception as exc:
        latency = time.monotonic() - t0
        # Treat error as a spike so the breaker opens faster under load
        _gpu_cb.record_latency(max(latency, GPUCircuitBreaker.SPIKE_THRESHOLD_S * 2))
        log.warning("GPU task %s failed — falling back to CPU: %s", task_type.value, exc)
        try:
            return await cpu_fn(*args, **kwargs)
        finally:
            CPU_LATENCY.observe(time.monotonic() - t0)


def get_circuit_breaker_state() -> str:
    """Return the current circuit breaker state (for dashboard / health checks)."""
    return _gpu_cb.state.value
