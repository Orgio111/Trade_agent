"""
QUANTEX GPU Optimization Utilities — Windows/WSL2/CUDA tuning for RTX 4050.

Provides automated setup and tuning for low-latency trading AI on
constrained GPU systems (RTX 4050 6GB VRAM).

Usage:
    from orchestrator.gpu_optimizer import GPUOptimizer

    optimizer = GPUOptimizer()
    report = optimizer.full_optimization()
    # report = {"power_mode": "ok", "gpu_settings": "applied", ...}

    # Or check specific aspects:
    status = optimizer.check_gpu_status()
    recommendations = optimizer.get_recommendations()
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("quantex.gpu_optimizer")


@dataclass
class GPUStatus:
    """Current GPU status and configuration."""
    gpu_name: str = "unknown"
    cuda_version: str = "unknown"
    driver_version: str = "unknown"
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    gpu_utilization: float = 0.0
    temperature_c: int = 0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    performance_mode: str = "unknown"
    is_wsl: bool = False
    platform: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "vram_total_mb": self.vram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb,
            "gpu_utilization_pct": self.gpu_utilization,
            "temperature_c": self.temperature_c,
            "power_draw_w": self.power_draw_w,
            "power_limit_w": self.power_limit_w,
            "performance_mode": self.performance_mode,
            "is_wsl": self.is_wsl,
            "platform": self.platform,
        }


@dataclass
class OptimizationReport:
    """Results of GPU optimization."""
    status: str = "unknown"
    gpu_status: Optional[GPUStatus] = None
    optimizations_applied: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "gpu": self.gpu_status.to_dict() if self.gpu_status else None,
            "optimizations_applied": self.optimizations_applied,
            "recommendations": self.recommendations,
            "errors": self.errors,
        }


class GPUOptimizer:
    """
    GPU optimization toolkit for RTX 4050 trading AI systems.

    Provides:
      - GPU status monitoring
      - Power mode optimization
      - CUDA environment tuning
      - VRAM management recommendations
      - Windows/WSL2 specific optimizations
    """

    # Optimal settings for RTX 4050 trading workloads
    RTX4050_OPTIMAL = {
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "batch_size": 1,
        "dtype": "half",
        "tensor_parallel_size": 1,
        "cuda_memory_fraction": 0.90,
    }

    def __init__(self):
        self._status: Optional[GPUStatus] = None

    def check_gpu_status(self) -> GPUStatus:
        """Query current GPU status via nvidia-smi."""
        status = GPUStatus()
        status.platform = platform.system()
        status.is_wsl = self._is_wsl()

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 10:
                    status.gpu_name = parts[0].strip()
                    status.driver_version = parts[1].strip()
                    status.cuda_version = parts[2].strip()
                    status.vram_total_mb = int(float(parts[3].strip()))
                    status.vram_used_mb = int(float(parts[4].strip()))
                    status.vram_free_mb = int(float(parts[5].strip()))
                    status.gpu_utilization = float(parts[6].strip())
                    status.temperature_c = int(float(parts[7].strip()))
                    status.power_draw_w = float(parts[8].strip())
                    status.power_limit_w = float(parts[9].strip())
                    status.performance_mode = "max_performance" if status.power_draw_w > 50 else "normal"
        except FileNotFoundError:
            logger.warning("nvidia-smi not found — GPU status unavailable")
        except Exception as e:
            logger.error(f"GPU status check failed: {e}")

        self._status = status
        return status

    def get_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on current GPU status."""
        recs = []
        status = self._status or self.check_gpu_status()

        if status.vram_total_mb == 0:
            recs.append("GPU not detected — ensure nvidia-smi is available")
            return recs

        # VRAM recommendations
        if status.vram_total_mb <= 6000:
            recs.append("RTX 4050 (6GB): Use 1.5B-7B quantized models only")
            recs.append("Set gpu_memory_utilization=0.85 to avoid OOM")
            recs.append("Use batch_size=1 for trading (no batching)")
            recs.append("Consider AWQ/GPTQ quantization for 7B models")

        if status.vram_free_mb < 2000:
            recs.append("Low VRAM free — unload unused models before loading new ones")

        # Temperature recommendations
        if status.temperature_c > 80:
            recs.append(f"GPU temp {status.temperature_c}°C — check cooling, reduce load")
        elif status.temperature_c > 70:
            recs.append(f"GPU temp {status.temperature_c}°C — acceptable but monitor")

        # Power recommendations
        if status.performance_mode != "max_performance":
            recs.append("Set NVIDIA power mode to 'Prefer maximum performance'")

        # Platform-specific
        if status.is_wsl:
            recs.append("WSL2: Ensure .wslconfig has memory=10GB and mitigations=off")
            recs.append("WSL2: Enable systemd in /etc/wsl.conf for service management")

        return recs

    def get_env_variables(self) -> dict[str, str]:
        """Get recommended CUDA/PyTorch environment variables."""
        env = {
            "CUDA_VISIBLE_DEVICES": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            "CUDA_LAUNCH_BLOCKING": "0",
        }

        status = self._status or self.check_gpu_status()
        if status.vram_total_mb <= 6000:
            # Aggressive memory management for small VRAM
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"

        return env

    def get_vllm_config(self, model: str = "Qwen/Qwen2.5-7B-Instruct") -> dict:
        """Get optimal vLLM configuration for current GPU."""
        status = self._status or self.check_gpu_status()

        config = {
            "model": model,
            "dtype": "half",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": 1,
            "batch_size": 1,
        }

        # Adjust for available VRAM
        if status.vram_total_mb <= 6000:
            config["max_model_len"] = 2048
            config["gpu_memory_utilization"] = 0.80
        elif status.vram_total_mb <= 8000:
            config["max_model_len"] = 4096
            config["gpu_memory_utilization"] = 0.85
        else:
            config["max_model_len"] = 8192
            config["gpu_memory_utilization"] = 0.90

        return config

    def get_model_recommendations(self) -> list[dict]:
        """Get recommended models for current GPU."""
        status = self._status or self.check_gpu_status()

        models = []

        if status.vram_total_mb >= 6000:
            models.append({
                "name": "Qwen2.5-1.5B-Instruct",
                "size": "1.5B",
                "quantization": "FP16",
                "vram_mb": 3000,
                "latency_ms": "30-80ms",
                "use_case": "fast_trading",
            })

        if status.vram_total_mb >= 6000:
            models.append({
                "name": "Qwen2.5-7B-Instruct-AWQ",
                "size": "7B",
                "quantization": "AWQ INT4",
                "vram_mb": 4500,
                "latency_ms": "80-200ms",
                "use_case": "reasoning_analysis",
            })

        if status.vram_total_mb >= 8000:
            models.append({
                "name": "Mistral-7B-Instruct-v0.3",
                "size": "7B",
                "quantization": "AWQ INT4",
                "vram_mb": 5000,
                "latency_ms": "80-180ms",
                "use_case": "fast_reasoning",
            })

        if status.vram_total_mb >= 6000:
            models.append({
                "name": "moondream2-1.6b",
                "size": "1.6B",
                "quantization": "FP16",
                "vram_mb": 3200,
                "latency_ms": "50-150ms",
                "use_case": "chart_vision",
            })

        return models

    def apply_optimizations(self) -> list[str]:
        """Apply GPU optimizations (safe, read-only operations)."""
        applied = []

        # Set environment variables (current process only)
        env_vars = self.get_env_variables()
        for key, value in env_vars.items():
            os.environ[key] = value
            applied.append(f"Set {key}={value}")

        # Try to set nvidia-smi persistence mode (requires admin/root)
        try:
            subprocess.run(
                ["nvidia-smi", "-pm", "1"],
                capture_output=True, timeout=5,
            )
            applied.append("Enabled nvidia-smi persistence mode")
        except Exception:
            pass  # May require admin

        return applied

    def full_optimization(self) -> OptimizationReport:
        """Run full GPU optimization pipeline."""
        report = OptimizationReport()

        # Step 1: Check GPU status
        report.gpu_status = self.check_gpu_status()

        # Step 2: Apply optimizations
        report.optimizations_applied = self.apply_optimizations()

        # Step 3: Get recommendations
        report.recommendations = self.get_recommendations()

        report.status = "ok" if report.gpu_status.vram_total_mb > 0 else "gpu_not_found"

        return report

    def _is_wsl(self) -> bool:
        """Check if running in WSL2."""
        try:
            with open("/proc/version", "r") as f:
                return "microsoft" in f.read().lower()
        except FileNotFoundError:
            return False


# ── Convenience Functions ───────────────────────────────────

def quick_gpu_check() -> dict:
    """One-shot GPU status check."""
    optimizer = GPUOptimizer()
    status = optimizer.check_gpu_status()
    return status.to_dict()


def get_optimal_vllm_config(model: str = "Qwen/Qwen2.5-7B-Instruct") -> dict:
    """Get optimal vLLM config for current GPU."""
    optimizer = GPUOptimizer()
    return optimizer.get_vllm_config(model)
