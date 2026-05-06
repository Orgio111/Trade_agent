"""
Intelligent compute scheduler: routes tasks to GPU (NIM/Triton) or CPU.
Implements a circuit breaker that falls back to lightweight CPU models
when GPU latency spikes above 200ms.
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from prometheus_client import Counter, Histogram, Gauge

log = logging.getLogger(__name__)

T = TypeVar("T")

GPU_LATENCY = Histogram(
    "scheduler_gpu_latency_seconds",
    "GPU inference latency",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)
CPU_LATENCY = Histogram(
    "scheduler_cpu_latency_seconds",
    "CPU inference latency",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
)
ROUTE_COUNTER = Counter(
    "scheduler_routes_total",
    "Routing decisions",
    ["task_type", "backend"],
)
CB_STATE = Gauge("scheduler_circuit_breaker_open", "1 = GPU circuit breaker open")


class TaskType(str, Enum):
    LLM        = "llm"          # → GPU (NIM)
    EMBED      = "embed"        # → GPU (NIM)
    INDICATOR  = "indicator"    # → CPU (NumPy/Polars)
    FEATURE    = "feature"      # → CPU
    RL_INFER   = "rl_infer"     # → GPU (Triton)


class CircuitState(str, Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class GPUCircuitBreaker:
    """
    Trips when GPU p95 latency > 200ms for N consecutive calls.
    Attempts half-open probe every `reset_timeout_s` seconds.
    """

    SPIKE_THRESHOLD_S = 0.200
    CONSECUTIVE_SPIKES = 3
    RESET_TIMEOUT_S    = 10.0

    def __init__(self) -> None:
        self._state         = CircuitState.CLOSED
        self._spike_count   = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if (time.monotonic() - (self._opened_at or 0)) > self.RESET_TIMEOUT_S:
                self._state = CircuitState.HALF_OPEN
                CB_STATE.set(0.5)
        return self._state

    def record_latency(self, latency_s: float) -> None:
        if self.state == CircuitState.HALF_OPEN and latency_s < self.SPIKE_THRESHOLD_S:
            self._state = CircuitState.CLOSED
            self._spike_count = 0
            CB_STATE.set(0)
            log.info("GPU circuit breaker CLOSED — latency recovered")
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


_gpu_cb = GPUCircuitBreaker()


async def route(
    task_type: TaskType,
    gpu_fn:    Callable[..., Awaitable[T]],
    cpu_fn:    Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Route `task_type` to the appropriate backend.
    CPU-native tasks always go CPU. LLM/embed/RL tasks go GPU unless breaker is open.
    """
    cpu_native = task_type in (TaskType.INDICATOR, TaskType.FEATURE)

    if cpu_native or _gpu_cb.is_open:
        backend = "cpu"
        ROUTE_COUNTER.labels(task_type=task_type.value, backend=backend).inc()
        t0 = time.monotonic()
        result = await cpu_fn(*args, **kwargs)
        CPU_LATENCY.observe(time.monotonic() - t0)
        return result

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
        _gpu_cb.record_latency(self.SPIKE_THRESHOLD_S * 2)  # treat error as spike
        log.warning("GPU task failed (%s) — falling back to CPU: %s", task_type.value, exc)
        return await cpu_fn(*args, **kwargs)
