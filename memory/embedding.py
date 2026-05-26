"""
Embedding Pipeline — converts market states and events into vectors.

Two-tier fallback:
  1. NVIDIA NV-Embed via NIM (primary) — high-quality semantic embeddings
  2. Feature-based embedding (fallback) — deterministic, no network needed

Design
──────
  Market State (prices, indicators, signals)
       │
       ▼
  State Serialiser ───→ text representation
       │
       ▼
  NIM Embed API ───→ 768-dim vector
  (or feature fallback)
       │
       ▼
  TurboVec Engine
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np

from core.config import get_settings
from core.nim_client import nim_embed

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_FEATURE_DIM = 64  # fallback deterministic embedding dimension


class EmbeddingPipeline:
    """Converts market states and events into fixed-dimension vectors.

    Usage:
        pipe = EmbeddingPipeline()
        vec = await pipe.embed_market_state(symbol="BTC/USDT", closes=..., ...)
    """

    def __init__(self, use_nim: bool = True, dimension: int | None = None) -> None:
        self.use_nim = use_nim
        self._cache: dict[str, np.ndarray] = {}
        self._max_cache = 1000
        # Read expected dimension from config, falling back to nv-embed default (768)
        cfg = get_settings()
        self.dimension = dimension or cfg.turbovec_dimension
        if self.dimension != 768:
            logger.info(
                "EmbeddingPipeline using custom dimension=%d (default is 768)",
                self.dimension,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def embed_text(self, text: str) -> np.ndarray:
        """Embed arbitrary text via NIM (or fallback hash).

        Validates that the returned vector dimension matches the
        configured ``turbovec_dimension``.
        """
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.use_nim:
            try:
                result = await nim_embed([text])
                vec = np.array(result[0], dtype=np.float32)
                if len(vec) != self.dimension:
                    logger.warning(
                        "NIM returned dim=%d, expected dim=%d — trimming/padding",
                        len(vec),
                        self.dimension,
                    )
                    if len(vec) > self.dimension:
                        vec = vec[:self.dimension]
                    else:
                        vec = np.pad(vec, (0, self.dimension - len(vec)), mode="constant")
                self._cache[cache_key] = vec
                self._trim_cache()
                return vec
            except Exception as exc:
                logger.warning("NIM embedding failed (%s) — using fallback", exc)

        vec = self._fallback_embed(text)
        self._cache[cache_key] = vec
        self._trim_cache()
        return vec

    async def embed_market_state(
        self,
        symbol: str,
        closes: np.ndarray,
        volumes: np.ndarray | None = None,
        rsi: float | None = None,
        ema_fast: float | None = None,
        ema_slow: float | None = None,
        atr: float | None = None,
        macd: float | None = None,
        macd_signal: float | None = None,
        bb_upper: float | None = None,
        bb_lower: float | None = None,
        volume_ratio: float | None = None,
        trend: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Build a text representation of market state and embed it.

        The text representation is designed to capture both quantitative
        features and qualitative context for semantic similarity matching.
        """
        state_text = self._build_state_text(
            symbol=symbol,
            closes=closes,
            volumes=volumes,
            rsi=rsi,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            atr=atr,
            macd=macd,
            macd_signal=macd_signal,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            volume_ratio=volume_ratio,
            trend=trend,
            extra=extra_metadata,
        )
        return await self.embed_text(state_text)

    async def embed_trade_outcome(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float | None,
        pnl_pct: float | None,
        win: bool | None,
        rationale: str = "",
        market_state_text: str = "",
    ) -> np.ndarray:
        """Embed a trade outcome for similarity learning."""
        parts = [
            f"Trade: {symbol} {side}",
        ]
        if entry_price:
            parts.append(f"Entry: ${entry_price:.2f}")
        if exit_price:
            parts.append(f"Exit: ${exit_price:.2f}")
        if pnl_pct is not None:
            parts.append(f"PnL: {pnl_pct:+.4f}")
        if win is not None:
            parts.append("Result: WIN" if win else "Result: LOSS")
        if rationale:
            parts.append(f"Rationale: {rationale}")
        if market_state_text:
            parts.append(f"Market: {market_state_text[:200]}")

        return await self.embed_text(" | ".join(parts))

    async def embed_risk_event(
        self,
        symbol: str,
        event_type: str,
        var_95: float,
        var_99: float,
        drawdown_pct: float,
        extra: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Embed a risk event (drawdown spike, VaR breach, kill-switch)."""
        text = (
            f"Risk Event: {symbol} type={event_type} "
            f"VaR95={var_95:.4f} VaR99={var_99:.4f} DD={drawdown_pct:.2%}"
        )
        if extra:
            text += f" {json.dumps(extra, default=str)[:200]}"
        return await self.embed_text(text)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_state_text(
        self,
        symbol: str,
        closes: np.ndarray,
        volumes: np.ndarray | None = None,
        rsi: float | None = None,
        ema_fast: float | None = None,
        ema_slow: float | None = None,
        atr: float | None = None,
        macd: float | None = None,
        macd_signal: float | None = None,
        bb_upper: float | None = None,
        bb_lower: float | None = None,
        volume_ratio: float | None = None,
        trend: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Serialize market state to a text string for embedding.

        The text is designed so that semantically similar market conditions
        produce similar embeddings (e.g., "BTC volatility spike with bearish
        momentum" will be close to other high-volatility bearish BTC states).
        """
        parts = [f"Symbol: {symbol}"]

        # Price action summary (last 20 bars)
        if len(closes) >= 20:
            recent = closes[-20:]
            ret = (recent[-1] - recent[0]) / recent[0]
            hi = recent.max()
            lo = recent.min()
            volatility = (hi - lo) / lo
            parts.append(
                f"Price: last=${recent[-1]:.2f} "
                f"return_20={ret:+.4f} "
                f"volatility_20={volatility:.4f} "
                f"high_20=${hi:.2f} low_20=${lo:.2f}"
            )
        elif len(closes) > 0:
            parts.append(f"Price: last=${closes[-1]:.2f}")

        # Volume
        if volumes is not None and len(volumes) > 0:
            avg_v = volumes[-20:].mean() if len(volumes) >= 20 else volumes.mean()
            last_v = volumes[-1]
            v_ratio = last_v / (avg_v + 1e-10)
            parts.append(f"Volume: last={last_v:.0f} avg_20={avg_v:.0f} ratio={v_ratio:.2f}")

        # Technical indicators
        if rsi is not None:
            regime = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
            parts.append(f"RSI: {rsi:.1f} ({regime})")
        if ema_fast is not None and ema_slow is not None:
            delta = (ema_fast - ema_slow) / (ema_slow + 1e-10)
            parts.append(f"EMA: fast={ema_fast:.2f} slow={ema_slow:.2f} spread={delta:+.4f}")
        if atr is not None:
            parts.append(f"ATR: {atr:.4f}")
        if macd is not None and macd_signal is not None:
            parts.append(f"MACD: {macd:.4f} signal={macd_signal:.4f}")
        if bb_upper is not None and bb_lower is not None:
            width = (bb_upper - bb_lower) / (bb_lower + 1e-10)
            parts.append(f"BB: width={width:.4f}")
        if volume_ratio is not None:
            parts.append(f"VolRatio: {volume_ratio:.2f}")
        if trend is not None:
            parts.append(f"Trend: {trend}")

        # Additional context
        if extra:
            for k, v in extra.items():
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.4f}")
                else:
                    parts.append(f"{k}: {v}")

        return " | ".join(parts)

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Deterministic fallback embedding using text hash features.

        Produces a stable ``self.dimension``-dim vector that captures the
        character-level structure of the input text. Not semantically
        meaningful, but provides consistent similarity for identical/
        near-identical texts.
        """
        d = self.dimension
        vec = np.zeros(d, dtype=np.float32)

        # Character n-gram features (unigrams and bigrams)
        for i, ch in enumerate(text):
            vec[hash(ch) % d] += 1.0
            if i > 0:
                bigram = text[i - 1 : i + 1]
                vec[hash(bigram) % d] += 0.5

        # Length feature
        vec[0] = len(text) / 1000.0

        # Normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def _trim_cache(self) -> None:
        if len(self._cache) > self._max_cache:
            # Remove oldest entries
            keys = list(self._cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._cache[k]


# ── Global singleton ──────────────────────────────────────────────────────────
_embedder: EmbeddingPipeline | None = None


def get_embedder() -> EmbeddingPipeline:
    """Return the global EmbeddingPipeline singleton."""
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingPipeline()
    return _embedder
