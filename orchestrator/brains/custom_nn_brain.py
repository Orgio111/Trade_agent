"""Brain #10: Custom Neural Network (LSTM + Transformer) for temporal pattern recognition.

Uses PyTorch LSTM or Transformer to learn sequential patterns from OHLCV data and
technical indicators. Complements XGBoost-based freqai_brain with
deep learning temporal features.

Architecture selection: set CUSTOM_NN_ARCHITECTURE env var to "lstm" (default) or "transformer".

Weight in Go orchestrator: 0.05.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)

# ── .env auto-load ─────────────────────────────────────
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parents[2]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass


# ── Positional Encoding (for Transformer) ──────────────

class PositionalEncoding:
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500):
        self.d_model = d_model
        self.max_len = max_len
        self._pe = None

    def _build(self):
        try:
            import torch
            pe = torch.zeros(self.max_len, self.d_model)
            position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32) *
                (-math.log(10000.0) / self.d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self._pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        except ImportError:
            pass

    def __call__(self, x):
        """Add positional encoding to input tensor."""
        try:
            import torch
            if self._pe is None:
                self._build()
            if self._pe is None:
                return x
            seq_len = x.size(1)
            return x + self._pe[:, :seq_len, :].to(x.device)
        except Exception:
            return x


# ── LSTM Model ──────────────────────────────────────────

class LSTMPredictor:
    """LSTM model for price direction prediction.

    Input: sequence of feature vectors (lookback steps × feature_dim)
    Output: score in [-1, 1] (buy/sell direction + strength)
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self._model = None
        self._device = "cpu"

    def _build_model(self):
        """Build PyTorch LSTM model."""
        try:
            import torch
            import torch.nn as nn

            class LSTMNet(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_dim, hidden_dim, num_layers,
                        batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1),
                    )

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    out = self.fc(lstm_out[:, -1, :])
                    return torch.tanh(out)

            self._model = LSTMNet(
                self.input_dim, self.hidden_dim, self.num_layers, self.dropout,
            )
            self._device = "cpu"
            logger.info("[custom_nn] Built LSTM model: input=%d, hidden=%d, layers=%d",
                        self.input_dim, self.hidden_dim, self.num_layers)
        except ImportError:
            logger.warning("[custom_nn] PyTorch not installed, LSTM disabled")

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model is not None

    def predict(self, features: np.ndarray) -> float:
        """Predict score from feature sequence.

        Args:
            features: (seq_len, input_dim) array of features

        Returns:
            score in [-1, 1]
        """
        if not self.is_ready:
            self._build_model()
        if not self.is_ready:
            return 0.0

        try:
            import torch

            self._model.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self._device)
                score = self._model(x).item()
                return float(np.clip(score, -1.0, 1.0))
        except Exception as e:
            logger.warning("[custom_nn] Prediction failed: %s", e)
            return 0.0

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self._model is None:
            return
        try:
            import torch
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), path)
            logger.info("[custom_nn] LSTM model saved to %s", path)
        except Exception as e:
            logger.warning("[custom_nn] Save failed: %s", e)

    def load(self, path: str) -> bool:
        """Load model from disk."""
        if not os.path.exists(path):
            return False
        try:
            import torch
            self._build_model()
            if self._model is None:
                return False
            self._model.load_state_dict(torch.load(path, map_location=self._device))
            self._model.eval()
            logger.info("[custom_nn] LSTM model loaded from %s", path)
            return True
        except Exception as e:
            logger.warning("[custom_nn] Load failed: %s", e)
            return False


# ── Transformer Model ───────────────────────────────────

class TransformerPredictor:
    """Transformer model for price direction prediction.

    Uses multi-head self-attention to capture temporal dependencies.
    Better at long-range patterns than LSTM, attention weights show
    which timeframes matter most.

    Input: sequence of feature vectors (lookback steps × feature_dim)
    Output: score in [-1, 1] (buy/sell direction + strength)
    """

    def __init__(
        self,
        input_dim: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 500,
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self._model = None
        self._device = "cpu"

    def _build_model(self):
        """Build PyTorch Transformer model."""
        try:
            import torch
            import torch.nn as nn

            class TransformerNet(nn.Module):
                def __init__(self, input_dim, d_model, nhead, num_layers,
                             dim_feedforward, dropout, max_seq_len):
                    super().__init__()
                    self.input_projection = nn.Linear(input_dim, d_model)
                    self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        batch_first=True,
                    )
                    self.transformer_encoder = nn.TransformerEncoder(
                        encoder_layer, num_layers=num_layers
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(d_model, d_model // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model // 2, 1),
                    )

                def forward(self, x):
                    # Project input to d_model dimensions
                    x = self.input_projection(x)
                    # Add positional encoding
                    x = self.pos_encoder(x)
                    # Transformer encoding
                    x = self.transformer_encoder(x)
                    # Use the last timestep's output
                    out = self.fc(x[:, -1, :])
                    return torch.tanh(out)

                def forward_with_attention(self, x):
                    """Forward pass that also returns last layer attention weights."""
                    x = self.input_projection(x)
                    x = self.pos_encoder(x)
                    # Run all layers except the last normally
                    for layer in self.transformer_encoder.layers[:-1]:
                        x = layer(x)
                    # Run last layer manually to capture attention
                    last_layer = self.transformer_encoder.layers[-1]
                    x2 = last_layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
                    attn_output, attn_weights = x2
                    # Apply the rest of the last layer (norm + FFN)
                    x = last_layer.norm1(attn_output + x)
                    x2 = last_layer.linear2(last_layer.dropout(last_layer.activation(last_layer.linear1(x))))
                    x = last_layer.norm2(x2 + x)
                    out = self.fc(x[:, -1, :])
                    return torch.tanh(out), attn_weights

            self._model = TransformerNet(
                self.input_dim, self.d_model, self.nhead, self.num_layers,
                self.dim_feedforward, self.dropout, self.max_seq_len,
            )
            self._device = "cpu"
            logger.info("[custom_nn] Built Transformer model: input=%d, d_model=%d, heads=%d, layers=%d",
                        self.input_dim, self.d_model, self.nhead, self.num_layers)
        except ImportError:
            logger.warning("[custom_nn] PyTorch not installed, Transformer disabled")

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model is not None

    def predict(self, features: np.ndarray) -> float:
        """Predict score from feature sequence.

        Args:
            features: (seq_len, input_dim) array of features

        Returns:
            score in [-1, 1]
        """
        if not self.is_ready:
            self._build_model()
        if not self.is_ready:
            return 0.0

        try:
            import torch

            self._model.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self._device)
                score = self._model(x).item()
                return float(np.clip(score, -1.0, 1.0))
        except Exception as e:
            logger.warning("[custom_nn] Transformer prediction failed: %s", e)
            return 0.0

    def get_attention_weights(self, features: np.ndarray) -> np.ndarray | None:
        """Get attention weights from the last Transformer layer.

        Uses forward_with_attention to capture attention weight matrices.

        Returns:
            (nhead, seq_len, seq_len) attention weight matrix, or None if unavailable
        """
        if not self.is_ready:
            return None

        try:
            import torch

            self._model.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self._device)
                _, attn_weights = self._model.forward_with_attention(x)
                if attn_weights is not None:
                    return attn_weights.detach().cpu().numpy()
                return None
        except Exception as e:
            logger.debug("[custom_nn] Could not get attention weights: %s", e)
            return None

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self._model is None:
            return
        try:
            import torch
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Save with config for rebuilding
            checkpoint = {
                'state_dict': self._model.state_dict(),
                'config': {
                    'input_dim': self.input_dim,
                    'd_model': self.d_model,
                    'nhead': self.nhead,
                    'num_layers': self.num_layers,
                    'dim_feedforward': self.dim_feedforward,
                    'dropout': self.dropout,
                    'max_seq_len': self.max_seq_len,
                }
            }
            torch.save(checkpoint, path)
            logger.info("[custom_nn] Transformer model saved to %s", path)
        except Exception as e:
            logger.warning("[custom_nn] Transformer save failed: %s", e)

    def load(self, path: str) -> bool:
        """Load model from disk."""
        if not os.path.exists(path):
            return False
        try:
            import torch
            checkpoint = torch.load(path, map_location="cpu")

            # Support both checkpoint format (dict with config) and raw state_dict
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
                self.input_dim = config.get('input_dim', self.input_dim)
                self.d_model = config.get('d_model', self.d_model)
                self.nhead = config.get('nhead', self.nhead)
                self.num_layers = config.get('num_layers', self.num_layers)
                self.dim_feedforward = config.get('dim_feedforward', self.dim_feedforward)
                self.dropout = config.get('dropout', self.dropout)
                self.max_seq_len = config.get('max_seq_len', self.max_seq_len)
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self._build_model()
            if self._model is None:
                return False
            self._model.load_state_dict(state_dict)
            self._model.eval()
            logger.info("[custom_nn] Transformer model loaded from %s", path)
            return True
        except Exception as e:
            logger.warning("[custom_nn] Transformer load failed: %s", e)
            return False


# ── Brain ───────────────────────────────────────────────

class CustomNNBrain(BaseBrain):
    """Brain #10: Custom Neural Network (LSTM or Transformer) for temporal pattern recognition.

    Computes RSI, MACD, Bollinger Band, ATR, and volume features from
    recent OHLCV data, then feeds them through an LSTM or Transformer model
    to predict price direction.

    Architecture selection:
      - Set CUSTOM_NN_ARCHITECTURE env var to "lstm" (default) or "transformer"
      - Transformer provides attention weights for interpretability

    Features:
      - Auto-fetches OHLCV from Binance via ccxt
      - Auto-trains model from historical data if enough bars available
      - Persists trained model to disk for reuse across restarts
      - Falls back to rule-based scoring when model unavailable
    """

    @property
    def brain_id(self) -> str:
        return "custom_nn"

    def __init__(self) -> None:
        # Architecture selection via env var
        self._architecture = os.getenv("CUSTOM_NN_ARCHITECTURE", "lstm").lower()

        # Create the appropriate predictor
        if self._architecture == "transformer":
            self._model: LSTMPredictor | TransformerPredictor = TransformerPredictor(
                input_dim=12,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                max_seq_len=500,
            )
        else:
            self._model = LSTMPredictor(
                input_dim=12,
                hidden_dim=64,
                num_layers=2,
                dropout=0.2,
            )

        self._ohlcv_buffer: list[dict] = []
        self._max_bars = 200
        self._lookback = 50  # sequence length
        self._model_dir = os.getenv(
            "CUSTOM_NN_MODEL_DIR",
            str(Path(__file__).resolve().parents[2] / "models" / "nn_brain"),
        )
        self._model_path = os.getenv("CUSTOM_NN_MODEL_PATH", "")
        self._binance_fetched: dict[str, float] = {}
        self._train_count = 0

        logger.info("[custom_nn] Initialized with architecture=%s", self._architecture)

    async def warmup(self) -> None:
        """Load pre-trained model if available."""
        # Explicit path
        if self._model_path and os.path.exists(self._model_path):
            if self._model.load(self._model_path):
                return

        # Auto-discover from model dir
        model_dir = Path(self._model_dir)
        model_file = model_dir / f"custom_nn_latest_{self._architecture}.pt"
        if model_file.exists():
            self._model.load(str(model_file))

    async def compute_score(self, symbol: str) -> BrainSignal:
        """Compute trading signal using neural network prediction."""
        self._auto_fetch_ohlcv(symbol)

        bars = self._ohlcv_buffer[-100:]
        if len(bars) < self._lookback:
            return BrainSignal(
                brain_id=self.brain_id,
                symbol=symbol,
                score=0.0,
                confidence=0.1,
                metadata={"reason": "insufficient_ohlcv", "bars": len(bars)},
            )

        closes = np.array([b["close"] for b in bars])
        highs = np.array([b["high"] for b in bars])
        lows = np.array([b["low"] for b in bars])
        volumes = np.array([b["volume"] for b in bars])

        # Rule-based scoring (fallback when no model)
        rsi = self._compute_rsi(closes, 14)
        macd_hist = self._compute_macd(closes)
        bb_pos = self._compute_bb_position(closes, 20)

        score = 0.0
        confidence = 0.0

        # RSI signal
        if rsi < 30:
            score += 0.3
            confidence += 0.2
        elif rsi > 70:
            score -= 0.3
            confidence += 0.2
        else:
            score += (50 - rsi) / 100
            confidence += 0.05

        # MACD histogram signal
        if macd_hist > 0:
            score += min(0.2, macd_hist / closes[-1] * 1000)
        else:
            score -= min(0.2, abs(macd_hist) / closes[-1] * 1000)

        # Bollinger Band position
        if bb_pos < 0.2:
            score += 0.15
        elif bb_pos > 0.8:
            score -= 0.15

        confidence = float(np.clip(confidence, 0.1, 0.95))

        # Try neural network model if available
        model_used = False
        attention_weights = None

        if self._model.is_ready:
            try:
                features = self._extract_features_sequence(closes, highs, lows, volumes)
                model_score = self._model.predict(features)

                # Get attention weights if Transformer
                if self._architecture == "transformer" and isinstance(self._model, TransformerPredictor):
                    attention_weights = self._model.get_attention_weights(features)

                # Blend: 60% model, 40% rules
                score = 0.6 * model_score + 0.4 * score
                confidence = min(0.95, confidence + 0.15)
                model_used = True
            except Exception as e:
                logger.warning("[custom_nn] Model prediction failed: %s", e)
        elif len(self._ohlcv_buffer) >= 100:
            # Auto-train if we have enough data
            self.auto_train()

        score = float(np.clip(score, -1.0, 1.0))

        metadata = {
            "architecture": self._architecture,
            "rsi": round(rsi, 2),
            "macd_hist": round(float(macd_hist), 4),
            "bb_pos": round(bb_pos, 3),
            "model_used": model_used,
        }

        # Add attention summary if available
        if attention_weights is not None:
            # Average across heads, then use last timestep
            avg_attn = np.mean(attention_weights, axis=0)  # (seq, seq)
            last_row = avg_attn[-1]
            top_indices = np.argsort(last_row)[-5:][::-1]
            metadata["attention_top_positions"] = top_indices.tolist()
            metadata["attention_entropy"] = float(-np.sum(last_row * np.log(last_row + 1e-10)))

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=score,
            confidence=confidence,
            metadata=metadata,
        )

    # ── Data fetching ────────────────────────────────────

    def _auto_fetch_ohlcv(self, symbol: str) -> None:
        """Fetch fresh OHLCV from Binance via ccxt if buffer is stale."""
        now = time.time()
        last_fetch = self._binance_fetched.get(symbol, 0)
        if now - last_fetch < 300 and len(self._ohlcv_buffer) >= 50:
            return

        try:
            import ccxt
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=200)
            if ohlcv:
                self._ohlcv_buffer.clear()
                for row in ohlcv:
                    self._ohlcv_buffer.append({
                        "timestamp": row[0],
                        "open": row[1],
                        "high": row[2],
                        "low": row[3],
                        "close": row[4],
                        "volume": row[5],
                    })
                self._binance_fetched[symbol] = now
                logger.info("[custom_nn] Fetched %d OHLCV bars for %s", len(ohlcv), symbol)
        except Exception as e:
            logger.warning("[custom_nn] Binance fetch failed: %s", e)

    # ── Feature engineering ──────────────────────────────

    def _extract_features_sequence(
        self, closes: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, volumes: np.ndarray,
    ) -> np.ndarray:
        """Extract feature sequence for model input.

        Returns:
            (lookback, input_dim) array
        """
        seq_len = min(self._lookback, len(closes))
        features = []

        for i in range(len(closes) - seq_len, len(closes)):
            window_closes = closes[:i + 1]
            window_highs = highs[:i + 1]
            window_lows = lows[:i + 1]
            window_volumes = volumes[:i + 1]

            feat = self._extract_single_features(
                window_closes, window_highs, window_lows, window_volumes,
            )
            features.append(feat)

        return np.array(features, dtype=np.float32)

    def _extract_single_features(
        self, closes: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, volumes: np.ndarray,
    ) -> list[float]:
        """Extract single feature vector (12-dim)."""
        rsi = self._compute_rsi(closes, 14) / 100.0
        macd = self._compute_macd(closes)
        bb = self._compute_bb_position(closes, 20)
        vol_score = self._compute_volume_score(volumes, 20)
        atr = self._compute_atr(closes, highs, lows, 14)
        roc5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 else 0.0
        roc10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 10 else 0.0
        roc20 = (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 20 else 0.0
        hl_spread = (highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0.0
        vol_change = (volumes[-1] - np.mean(volumes[-20:])) / (np.mean(volumes[-20:]) + 1e-10) if len(volumes) > 20 else 0.0
        price_ma_ratio = closes[-1] / np.mean(closes[-20:]) if len(closes) > 20 else 1.0
        volatility = float(np.std(np.diff(closes[-20:]) / closes[-20:-1])) if len(closes) > 20 else 0.0

        return [
            rsi, macd / closes[-1] * 100, bb, vol_score,
            atr / closes[-1], roc5, roc10, roc20,
            hl_spread, vol_change, price_ma_ratio - 1.0, volatility,
        ]

    # ── Technical indicators ─────────────────────────────

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def _compute_macd(self, closes: np.ndarray) -> float:
        if len(closes) < 26:
            return 0.0
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = macd_line * 0.2
        return float(macd_line - signal_line)

    def _compute_bb_position(self, closes: np.ndarray, period: int = 20) -> float:
        if len(closes) < period:
            return 0.5
        window = closes[-period:]
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            return 0.5
        return float((closes[-1] - (mean - 2 * std)) / (4 * std))

    def _compute_volume_score(self, volumes: np.ndarray, period: int = 20) -> float:
        if len(volumes) < period:
            return 0.3
        avg_vol = np.mean(volumes[-period:])
        recent_vol = volumes[-1]
        if avg_vol == 0:
            return 0.3
        ratio = recent_vol / avg_vol
        return float(min(1.0, ratio / 3))

    def _compute_atr(self, closes: np.ndarray, highs: np.ndarray,
                     lows: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0
        tr = np.maximum(
            highs[-period:] - lows[-period:],
            np.maximum(
                np.abs(highs[-period:] - closes[-period - 1:-1]),
                np.abs(lows[-period:] - closes[-period - 1:-1]),
            ),
        )
        return float(np.mean(tr))

    def _ema(self, data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0.0
        multiplier = 2 / (period + 1)
        ema = float(np.mean(data[:period]))
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    # ── Training ─────────────────────────────────────────

    def push_ohlcv(self, bar: dict) -> None:
        """Push a new OHLCV bar into the buffer."""
        self._ohlcv_buffer.append(bar)
        if len(self._ohlcv_buffer) > self._max_bars:
            self._ohlcv_buffer = self._ohlcv_buffer[-self._max_bars:]

    def auto_train(self) -> bool:
        """Auto-train model from accumulated OHLCV data."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.debug("[custom_nn] PyTorch not installed, skipping auto-train")
            return False

        bars = self._ohlcv_buffer[-200:]
        if len(bars) < 100:
            return False

        closes = np.array([b["close"] for b in bars])
        highs = np.array([b["high"] for b in bars])
        lows = np.array([b["low"] for b in bars])
        volumes = np.array([b["volume"] for b in bars])

        try:
            # Build training data: sequences of (lookback, input_dim)
            X, y = [], []
            for i in range(self._lookback, len(closes) - 1):
                seq_features = []
                for j in range(i - self._lookback + 1, i + 1):
                    feat = self._extract_single_features(
                        closes[:j + 1], highs[:j + 1], lows[:j + 1], volumes[:j + 1],
                    )
                    seq_features.append(feat)
                X.append(seq_features)  # shape: (lookback, 12)

                # Label: next bar return direction
                next_return = (closes[i + 1] - closes[i]) / closes[i]
                label = 1.0 if next_return > 0 else -1.0
                y.append(label)

            X = np.array(X, dtype=np.float32)  # (n_samples, lookback, 12)
            y = np.array(y, dtype=np.float32).reshape(-1, 1)  # (n_samples, 1)

            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            # Build and train model
            self._model._build_model()
            if self._model._model is None:
                return False

            model = self._model._model
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(20):
                optimizer.zero_grad()
                pred = model(X_tensor)
                loss = criterion(pred, y_tensor)
                loss.backward()
                optimizer.step()

            model.eval()
            self._train_count += 1

            # Save model
            model_dir = Path(self._model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file = model_dir / f"custom_nn_latest_{self._architecture}.pt"
            self._model.save(str(model_file))

            logger.info("[custom_nn] Auto-trained %s on %d samples (epoch loss=%.4f), saved to %s",
                        self._architecture, len(X), loss.item(), model_file)
            return True

        except Exception as e:
            logger.warning("[custom_nn] Auto-train failed: %s", e)
            return False
