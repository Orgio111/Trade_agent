---
title: "NN Brain Development Guide"
type: playbook
tags: [neural-network, brain, development, tutorial, workflow]
created: 2026-06-26
updated: 2026-06-26
sources: [[codecrafters-build-your-own-x]]
status: draft
---

# NN Brain Development Guide

## TL;DR

Step-by-step playbook for building, training, and deploying a custom neural network brain in the Trade_agent orchestrator. Based on research from [[codecrafters-build-your-own-x]] and Andrej Karpathy's "Neural Networks: Zero to Hero" course.

## Phase 1: Foundation (Day 1-2)

### Step 1.1: Environment Setup
```bash
# Verify PyTorch is installed
python -c "import torch; print(torch.__version__)"

# If not installed:
pip install torch torchvision torchaudio
```

### Step 1.2: Quick Prototype — LSTM Brain
Create `orchestrator/brains/custom_nn_brain.py` following the pattern from [[freqai-brain]]:

```python
class CustomNNBrain(BaseBrain):
    """Brain #10: Custom Neural Network (LSTM) for temporal pattern recognition."""
    
    @property
    def brain_id(self) -> str:
        return "custom_nn"
    
    def __init__(self):
        self._model = None
        self._ohlcv_buffer = []
        self._max_bars = 200
        self._model_path = "models/nn_brain/custom_nn_latest.pt"
    
    async def warmup(self) -> None:
        """Load pre-trained LSTM model if available."""
        # Load from disk or create new
        ...
    
    async def compute_score(self, symbol: str) -> BrainSignal:
        """Compute trading signal using LSTM prediction."""
        # 1. Fetch OHLCV (same pattern as freqai_brain)
        # 2. Compute technical indicators (features)
        # 3. Run LSTM model
        # 4. Return BrainSignal(score, confidence)
        ...
```

### Step 1.3: Feature Engineering
Use the same features as [[freqai-brain]]:
- RSI (14-period)
- MACD (12/26/9)
- Bollinger Bands (20-period)
- ATR (14-period)
- Volume profile (20-period)
- Price returns (1h, 5h, 24h)

Plus new sequence features:
- OHLCV normalized to [0, 1]
- Rolling volatility (20-period)
- Price position relative to 50-period MA

## Phase 2: Training (Day 3-5)

### Step 2.1: Data Collection
```python
# Fetch 180 days of BTCUSDT 1h data
from orchestrator.backtest import DataLoader
from datetime import timedelta

end = datetime.now()
start = end - timedelta(days=180)
df = await DataLoader.from_binance_api(
    symbol="BTCUSDT", interval="1h",
    start_time=start, end_time=end
)
```

### Step 2.2: Train-Test Split
```python
n = len(df)
train_n = int(n * 0.70)  # 126 days
test_n = int(n * 0.15)   # 27 days
val_n = n - train_n - test_n  # 27 days

train_df = df.iloc[:train_n]
test_df = df.iloc[train_n:train_n + test_n]
val_df = df.iloc[train_n + test_n:]
```

### Step 2.3: Train LSTM Model
```python
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # predict next return direction
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return torch.tanh(out)  # score in [-1, 1]

# Training loop
model = LSTMPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()

for epoch in range(50):
    for batch in train_loader:
        pred = model(batch.features)
        loss = criterion(pred, batch.labels)
        loss.backward()
        optimizer.step()
```

### Step 2.4: Validate
- Test accuracy > 52%
- Overfitting gap < 15pp
- Sharpe ratio > 0.5

## Phase 3: Integration (Day 6-7)

### Step 3.1: Register in brain_registry.json
```json
{
  "brain_id": "custom_nn",
  "class": "CustomNNBrain",
  "description": "Custom LSTM neural network for temporal pattern recognition",
  "weight": 0.05,
  "source_file": "custom_nn_brain.py"
}
```

### Step 3.2: Update Brain Weights
Current total: 0.90 → New total: 0.95

### Step 3.3: Run Brain Backtest
```bash
python -m orchestrator.brain_backtest --symbol BTCUSDT --days 90
```

Compare with existing brains to ensure positive contribution.

### Step 3.4: Paper Trade Test
Run for 24-48 hours in paper trading mode before live deployment.

## Phase 4: Migration to Transformer (Week 2-3)

After LSTM validates, migrate to Transformer architecture:

```python
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=50, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return torch.tanh(out)
```

Key advantages:
- Attention weights show which timeframes matter
- Better long-range pattern capture
- More robust to regime changes

## Success Criteria

| Criterion | Target | Measurement |
|---|---|---|
| Direction accuracy | >52% | Validation set |
| Sharpe ratio | >0.5 | Brain backtest |
| Profit factor | >1.2 | Brain backtest |
| Inference latency | <200ms | Profiling |
| Model size | <10MB | File size |

## Related

- [[custom-trading-brain-architecture]] — full design document
- [[neural-network-brain]] — entity page
- [[freqai-brain]] — pattern reference for brain implementation
- [[brain-backtest-infrastructure]] — evaluation framework
- [[codecrafters-build-your-own-x]] — learning resources
