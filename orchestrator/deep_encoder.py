"""
QUANTEX Deep Market Encoder — GPU-accelerated LSTM + Transformer market representation.

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │  Input: (lookback=50, features=8)                    │
  │  [OHLCV(5) + returns(1) + volume_ratio(1) + hl_range(1)] │
  └──────────────────────┬───────────────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │  LayerNorm + Linear                                  │
  │  (8 → 64, projects raw features to model dim)       │
  └──────────────────────┬───────────────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │  LSTM(64 → 128, bidirectional)                       │  ← GPU
  │  Captures temporal patterns across lookback window    │
  └──────────────────────┬───────────────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────┐
  │  TransformerEncoder(128, 4 heads, 2 layers)          │  ← GPU
  │  Self-attention over sequence positions               │
  └──────────────────────┬───────────────────────────────┘
                         ▼ (mean pool)
  ┌──────────────────────────────────────────────────────┐
  │  Linear(128 → 32)                                    │
  │  Final market encoding → obs[96:128]                 │
  └──────────────────────────────────────────────────────┘

Usage (training):
    encoder = DeepMarketEncoder(device="cuda")
    seq = encoder.build_sequence(df, step_index=100)  # numpy (50, 8)
    encoding = encoder.encode(seq)                     # numpy (32,)

Integration:
    In GymTradingEnv: pass deep_encoder, it runs .encode() at each step
    and injects the result into obs[96:128].
    PPO's MlpPolicy stays on CPU — only the encoder uses GPU.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── Sequence Builder ────────────────────────────────────────

INPUT_FEATURES = 8  # OHLCV(5) + returns(1) + vol_ratio(1) + hl_range(1)


def build_market_sequence(
    df,
    step_index: int,
    lookback: int = 50,
) -> np.ndarray:
    """
    Build a normalized market sequence from OHLCV data.

    Args:
        df: OHLCV DataFrame with [open, high, low, close, volume]
        step_index: Current step index
        lookback: How many candles to include

    Returns:
        np.ndarray of shape (lookback, 8) normalized to ~[-1, 1]
    """
    start = max(0, step_index - lookback + 1)
    end = min(len(df), step_index + 1)
    window = df.iloc[start:end].copy()

    # Pad if we don't have enough data at the beginning
    if len(window) < lookback:
        pad_len = lookback - len(window)
        pad = pd.DataFrame(
            np.zeros((pad_len, len(window.columns))),
            columns=window.columns,
        )
        window = pd.concat([pad, window], ignore_index=True)

    close = window["close"].values
    base_price = close[-1] if close[-1] > 0 else 1.0

    # Build feature matrix (lookback, 8)
    seq = np.zeros((lookback, INPUT_FEATURES), dtype=np.float32)

    # 0-4: Normalized OHLCV
    seq[:, 0] = window["open"].values / base_price - 1.0     # open offset
    seq[:, 1] = window["high"].values / base_price - 1.0     # high offset
    seq[:, 2] = window["low"].values / base_price - 1.0      # low offset
    seq[:, 3] = window["close"].values / base_price - 1.0    # close offset
    seq[:, 4] = window["volume"].values / (window["volume"].mean() + 1e-10) - 1.0  # vol ratio - 1

    # 5: Returns (1-period)
    seq[:, 5] = np.diff(close, prepend=close[0]) / (close + 1e-10)
    seq[:, 5] = np.clip(seq[:, 5], -0.1, 0.1) * 10  # scale to [-1, 1]

    # 6: Volume ratio (vs rolling mean)
    vol_mean = window["volume"].rolling(20).mean().fillna(window["volume"].mean()).values
    seq[:, 6] = window["volume"].values / (vol_mean + 1e-10) - 1.0
    seq[:, 6] = np.clip(seq[:, 6], -1, 1)

    # 7: High-Low range normalized
    candle_range = (window["high"] - window["low"]).values
    seq[:, 7] = candle_range / (close + 1e-10)
    seq[:, 7] = np.clip(seq[:, 7], 0, 0.1) * 10  # scale to [0, 1]

    return seq


# ── Positional Encoding ─────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


# ── Deep Market Encoder ─────────────────────────────────────

class DeepMarketEncoder(nn.Module):
    """
    GPU-accelerated deep market encoder.

    Architecture:
      Linear(8 → 64) → LayerNorm → LSTM(64→128, bidir) →
      PositionalEncoding → TransformerEncoder(128, 4 heads, 2 layers) →
      MeanPool → Linear(128 → 32) → output encoding

    Input:  (batch, seq_len=50, features=8)
    Output: (batch, encoding_dim=32)
    """

    def __init__(
        self,
        input_features: int = INPUT_FEATURES,
        seq_len: int = 50,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        transformer_d_model: int = 128,
        transformer_nheads: int = 4,
        transformer_layers: int = 2,
        encoding_dim: int = 32,
        device: str = "auto",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.encoding_dim = encoding_dim

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Input projection: raw features → model dimension
        self.input_proj = nn.Linear(input_features, lstm_hidden)

        # LayerNorm after projection
        self.norm_in = nn.LayerNorm(lstm_hidden)

        # LSTM: captures temporal dependencies
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0,
        )

        # LSTM outputs 2 * lstm_hidden (bidirectional), project to d_model
        self.lstm_proj = nn.Linear(lstm_hidden * 2, transformer_d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(transformer_d_model, max_len=seq_len)

        # TransformerEncoder: self-attention over sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nheads,
            dim_feedforward=transformer_d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        # Output projection: pooled → encoding
        self.output_proj = nn.Sequential(
            nn.Linear(transformer_d_model, encoding_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Move to target device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: (batch, seq_len, input_features) — raw market sequence

        Returns:
            (batch, encoding_dim) — market encoding
        """
        # Input projection
        x = self.input_proj(x)       # (batch, seq, lstm_hidden)
        x = self.norm_in(x)

        # LSTM
        lstm_out, _ = self.lstm(x)   # (batch, seq, lstm_hidden * 2)

        # Project to transformer dim
        x = self.lstm_proj(lstm_out)  # (batch, seq, d_model)

        # Positional encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)       # (batch, seq, d_model)

        # Mean pool over sequence dimension
        x = x.mean(dim=1)             # (batch, d_model)

        # Final projection
        encoding = self.output_proj(x)  # (batch, encoding_dim)

        return encoding

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        """
        Encode a single market sequence (numpy in, numpy out).

        Args:
            sequence: np.ndarray of shape (seq_len, input_features)

        Returns:
            np.ndarray of shape (encoding_dim,)
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
            # (1, seq_len, features)
            encoding = self.forward(tensor)
            return encoding.cpu().numpy().flatten()

    def encode_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Encode multiple sequences in a batch.

        Args:
            sequences: np.ndarray of shape (batch, seq_len, input_features)

        Returns:
            np.ndarray of shape (batch, encoding_dim)
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(sequences).float().to(self.device)
            encodings = self.forward(tensor)
            return encodings.cpu().numpy()


# ── Market SequenceBuffer (for environment integration) ─────

class MarketSequenceBuffer:
    """
    Maintains a rolling buffer of the last N market candles
    and builds normalized sequences for the encoder.

    Used inside the environment to feed the DeepMarketEncoder.
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self._buffer: list[np.ndarray] = []  # Each element: (8,) feature vector

    def reset(self, df, start_index: int):
        """Initialize buffer from a starting position in the DataFrame."""
        self._buffer = []
        start = max(0, start_index - self.lookback + 1)
        for i in range(start, start_index + 1):
            self._append_row(df, i)

    def update(self, df, step_index: int):
        """Add a new row to the buffer."""
        self._append_row(df, step_index)
        # Trim to lookback
        if len(self._buffer) > self.lookback:
            self._buffer = self._buffer[-self.lookback:]

    def _append_row(self, df, idx):
        """Compute normalized feature vector for one candle."""
        if idx >= len(df) or idx < 0:
            self._buffer.append(np.zeros(INPUT_FEATURES, dtype=np.float32))
            return

        row = df.iloc[idx]
        close = float(row["close"]) if not pd.isna(row["close"]) else 1.0
        base = close if close > 0 else 1.0

        vec = np.zeros(INPUT_FEATURES, dtype=np.float32)
        vec[0] = float(row["open"]) / base - 1.0
        vec[1] = float(row["high"]) / base - 1.0
        vec[2] = float(row["low"]) / base - 1.0
        vec[3] = 0.0  # close is reference

        vol = float(row["volume"]) if not pd.isna(row["volume"]) else 0
        vol_mean = np.mean([float(df.iloc[j]["volume"]) for j in range(max(0, idx - 20), idx + 1)
                           if not pd.isna(df.iloc[j]["volume"])]) if idx >= 0 else vol
        vec[4] = vol / (vol_mean + 1e-10) - 1.0

        prev_close = float(df.iloc[max(0, idx - 1)]["close"]) if idx > 0 else close
        ret = (close - prev_close) / (prev_close + 1e-10)
        vec[5] = np.clip(ret, -0.1, 0.1) * 10

        # 6: Volume ratio (separate from the raw vol/mean above)
        # vec[4] is vol/mean - 1, vec[6] is same but clipped to [-1, 1]
        vec[6] = np.clip(vec[4], -1, 1)

        hl_range = (float(row["high"]) - float(row["low"])) / base
        vec[7] = np.clip(hl_range, 0, 0.1) * 10

        self._buffer.append(vec)

    def get_sequence(self) -> np.ndarray:
        """Get the current sequence as a numpy array."""
        if len(self._buffer) < self.lookback:
            pad = self.lookback - len(self._buffer)
            return np.pad(
                np.array(self._buffer[-len(self._buffer):], dtype=np.float32),
                ((pad, 0), (0, 0)),
                mode="constant",
            )
        return np.array(self._buffer[-self.lookback:], dtype=np.float32)

    @property
    def is_ready(self) -> bool:
        """Buffer has enough data for a valid sequence."""
        return len(self._buffer) >= self.lookback



