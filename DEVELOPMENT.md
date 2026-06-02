# QUANTEX Development Guide — WSL2 + CUDA Setup

> **Target Hardware:** Windows 11 laptop with RTX 4050 (6GB VRAM) or any NVIDIA GPU  
> **Target OS:** Windows 11 with WSL2 + Ubuntu 24.04  
> **Last Updated:** June 2, 2026

---

## Table of Contents

1. [Why WSL2?](#1-why-wsl2)
2. [Prerequisites](#2-prerequisites)
3. [WSL2 Installation](#3-wsl2-installation)
4. [NVIDIA CUDA in WSL2](#4-nvidia-cuda-in-wsl2)
5. [Docker Setup](#5-docker-setup)
6. [Project Setup](#6-project-setup)
7. [Running the Stack](#7-running-the-stack)
8. [GPU-Accelerated Development](#8-gpu-accelerated-development)
9. [Docker GPU Configuration](#9-docker-gpu-configuration)
10. [Working with the Codebase](#10-working-with-the-codebase)
11. [Troubleshooting](#11-troubleshooting)
12. [Performance Tuning](#12-performance-tuning)

---

## 1. Why WSL2?

All QUANTEX services are designed to run in Linux containers. WSL2 provides:

- **Native Linux kernel** inside Windows — full system call compatibility
- **CUDA passthrough** — GPU in WSL2 uses your host NVIDIA driver directly (no driver duplication)
- **Docker Desktop integration** — runs Docker Engine inside WSL2 for near-native I/O performance
- **File system performance** — keep project files inside the WSL2 filesystem (`~/quantex`), **not** on `/mnt/c/` (Windows drives are ~10× slower for container mounts and git operations)

### Performance Comparison

| Operation | WSL2 filesystem (`~/project`) | Windows mount (`/mnt/c/...`) |
|-----------|-------------------------------|------------------------------|
| `git status` | 0.3s | 3-5s |
| `npm install` | 15s | 60s+ |
| Docker build | 30s | 120s+ |
| File read IOPS | ~200k | ~15k |

> **Golden rule:** Clone the repo into WSL2's native filesystem at `~/quantex`. Never work from `/mnt/c/`.

---

## 2. Prerequisites

Before starting, verify you have:

| Component | Minimum Version | Check Command |
|-----------|----------------|---------------|
| Windows 11 | Build 22621+ | `winver` |
| NVIDIA Driver | 545+ (Game Ready or Studio) | `nvidia-smi` in PowerShell |
| WSL2 | 2.0+ | `wsl --version` in PowerShell |
| Docker Desktop | 4.30+ | `docker --version` in WSL2 |

### Required Accounts

- [Docker Hub](https://hub.docker.com/) account
- [NVIDIA NGC](https://ngc.nvidia.com/) account (for NVIDIA_API_KEY)
- [Groq Console](https://console.groq.com/) (for GROQ_API_KEY — free)
- [OpenRouter](https://openrouter.ai/) (for OPENROUTER_API_KEY — free)

---

## 3. WSL2 Installation

### Step 1: Enable WSL2 (PowerShell as Admin)

```powershell
# Enable WSL
wsl --install -d Ubuntu-24.04

# Set WSL2 as default
wsl --set-default-version 2

# Verify
wsl --list --verbose
# Should show: Ubuntu-24.04  Running  2
```

### Step 2: Configure WSL2 Resources

Create/edit `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=16GB          # Allocate 16GB RAM to WSL2 (adjust for your system)
processors=8         # Use 8 CPU cores
localhostForwarding=true
kernelCommandLine=vsyscall=emulate

# GPU: CUDA works automatically with WSL2 — no special config needed
# Networking: WSL2 shares Windows IP — services are accessible at localhost
```

Then restart WSL2:

```powershell
wsl --shutdown
wsl
```

### Step 3: Update Ubuntu Packages

```bash
# Inside WSL2 (Ubuntu terminal)
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release \
    neovim \
    htop \
    nvtop \
    net-tools
```

---

## 4. NVIDIA CUDA in WSL2

WSL2 uses **Windows' native NVIDIA driver** — you do NOT install a separate driver or CUDA toolkit inside WSL2. The GPU driver runs on Windows, and WSL2 forwards CUDA calls through the WSL2 GPU paravirtualization layer.

### Step 1: Verify GPU Passthrough

```bash
# Inside WSL2
nvidia-smi
```

Expected output:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 545.xx    Driver Version: 545.xx      CUDA Version: 12.3         |
|-------------------------------+----------------------+----------------------+
| GPU  Name              TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 4050 ...  WDDM |    00000000:01:00.0 On |                  N/A |
| 29%   52C    P0    25W /  95W |    512MiB /  6144MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

> **Important:** If `nvidia-smi` shows nothing or errors, the GPU driver on Windows needs updating. Download the latest Game Ready or Studio driver from NVIDIA's website and reboot.

### Step 2: Install CUDA Toolkit (for development tools only)

The runtime is provided by the Windows driver, but you need the toolkit for:

- `nvcc` compiler (only needed if building custom CUDA kernels)
- `cuDNN` headers (needed by some ML frameworks)
- `cuBLAS`, `cuFFT` development libraries

```bash
# Inside WSL2 — install CUDA 12.x toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6 cudnn-cuda-12

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### Step 3: Python with CUDA Support

```bash
# Inside WSL2
# Install Python 3.12
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Make python3.12 the default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --config python3

# Install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
```

---

## 5. Docker Setup

### Step 1: Install Docker (Recommended: Docker Desktop)

### Option A: Docker Desktop (Recommended)

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Install with WSL2 backend selected
3. Go to **Settings → Resources → WSL Integration**
4. Enable integration for your Ubuntu-24.04 distro
5. Apply & Restart

### Option B: Docker Engine Directly in WSL2

If you prefer not using Docker Desktop, install Docker Engine directly inside WSL2.
You must enable systemd in WSL2 for this to work:

```bash
# Inside WSL2 — edit /etc/wsl.conf
sudo tee /etc/wsl.conf << 'EOF'
[boot]
systemd=true
EOF

# Exit WSL2 and restart
# In PowerShell: wsl --shutdown && wsl

# Then install Docker Engine
# https://docs.docker.com/engine/install/ubuntu/
```

> **Note:** GPU support (`--gpus all`) requires installing `nvidia-container-toolkit` separately.

### Step 2: Verify Docker in WSL2

```bash
# Inside WSL2
docker --version
docker compose version

# Test GPU access in containers (match tag to your CUDA version from nvidia-smi)
docker run --rm --gpus all nvidia/cuda:12.6.0-runtime-ubuntu24.04 nvidia-smi
```

Expected output — the container sees the same GPU:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 545.xx    Driver Version: 545.xx      CUDA Version: 12.3         |
+-----------------------------------------------------------------------------+
```

> **Note:** If the exact tag doesn't exist, check [NVIDIA CUDA tags on Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags) and use the closest matching release.

### Step 3: Configure Docker Daemon for GPU

Docker Desktop automatically configures the `nvidia-container-runtime`. Verify:

```bash
docker info | grep -i runtime
# Should show: Runtimes: nvidia runc
```

If `nvidia` runtime is missing, create/edit `%USERPROFILE%\.docker\daemon.json`:

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

Then restart Docker Desktop.

---

## 6. Project Setup

### Step 1: Clone (to WSL2 filesystem!)

```bash
# CRITICAL: Clone inside WSL2, NOT into /mnt/c/
cd ~
git clone https://github.com/your/quantex.git
cd ~/quantex
```

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required for inference routing
NVIDIA_API_KEY=nvapi-your-key-here
OPENROUTER_API_KEY=sk-or-v1-your-key-here
GROQ_API_KEY=gsk-your-key-here

# Budget tier: free | low | medium
INFERENCE_BUDGET_TIER=free

# Optional: adjust paper trading balance
INITIAL_BALANCE=1000.0
```

> At least one API key is needed for cloud inference. The router will fall through providers. With all three configured, you get full failover.

### Step 3: Install Python Dependencies

```bash
cd ~/quantex/orchestrator

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with CUDA-aware ML packages
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install GPU-accelerated ML packages (optional)
# Check https://pytorch.org/get-started/locally/ for the latest CUDA version URL
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install unsloth bitsandbytes flash-attn --no-build-isolation
pip install stable-baselines3[extra]
```

### Step 4: Rust Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build the execution engine
cd ~/quantex/execution
cargo build --release
```

### Step 5: Go Setup

```bash
# Install Go 1.26+
wget https://go.dev/dl/go1.26.0.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.26.0.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Build realtime service
cd ~/quantex/realtime
go build -o realtime .
```

### Step 6: Frontend Setup

```bash
cd ~/quantex/frontend

# Install Node.js 22+
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# Install dependencies
npm install
```

---

## 7. Running the Stack

### Option A: Full Stack (Docker Compose)

This runs all services in containers:

```bash
cd ~/quantex

# Start infrastructure services
docker compose up -d

# Verify all services are healthy
docker compose ps
```

Services started:

| Service | Port | Health Check |
|---------|------|-------------|
| PostgreSQL + pgvector | 5432 | pg_isready |
| Redis | 6379 | redis-cli ping |
| Qdrant | 6333, 6334 | HTTP /healthz |
| NATS JetStream | 4222 | HTTP /healthz |
| InfluxDB | 8086 | HTTP /health |
| Prometheus | 9090 | HTTP /-/healthy |
| Grafana | 3001 | HTTP /api/health |
| Nginx | 80, 443 | HTTP /health |
| Orchestrator | — (via nginx) | HTTP /health |
| Realtime | — (via nginx) | HTTP /health |
| Frontend | — (via nginx) | pgrep + wget |

### Option B: Development Mode (Hot Reload)

For active development, run services outside Docker for faster iteration:

```bash
# Terminal 1: Infrastructure (Docker)
cd ~/quantex
docker compose up -d postgres redis qdrant nats influxdb

# Terminal 2: Orchestrator (hot reload)
cd ~/quantex/orchestrator
source .venv/bin/activate
uvicorn orchestrator.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 3: Frontend (hot reload)
cd ~/quantex/frontend
npm run dev

# Terminal 4: Realtime (if needed)
cd ~/quantex/realtime
./realtime
```

### Verify Everything Works

```bash
# Health check
curl http://localhost:8001/health

# Expected response:
# {"status":"ok","service":"quantex-orchestrator","version":"1.0.0",...}

# Portable data
curl http://localhost:8001/api/v1/portfolio

# Inference routing status
curl http://localhost:8001/api/v2/inference/routing
```

---

## 8. GPU-Accelerated Development

### 8.1 GPU-Accelerated ML Training

```bash
cd ~/quantex/orchestrator
source .venv/bin/activate

# Train ML model (uses CPU — scikit-learn doesn't need GPU)
python -c "
from ml_signals import MLSignalEngine
from backtest import DataLoader

engine = MLSignalEngine()
data = DataLoader.generate_mock_data(periods=5000)
result = engine.train(data, force=True)
print(result)
"
```

### 8.2 Reinforcement Learning with GPU

```bash
python -c "
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Create environment
from rl.trading_env import TradingEnvironment
from backtest import DataLoader

data = DataLoader.generate_mock_data(periods=10000)
env = TradingEnvironment(data)

# Train PPO on GPU
model = PPO(
    'MlpPolicy', env,
    n_steps=128,           # Reduced for 6GB VRAM
    batch_size=64,         # Reduced for 6GB VRAM
    ent_coef=0.01,
    policy_kwargs={'net_arch': [128, 64]},
    device='cuda',         # Uses your RTX 4050
    verbose=1,
)

model.learn(total_timesteps=10000)
print('PPO training complete on GPU!')
"
```

### 8.3 Memory-Efficient QLoRA Fine-Tuning

For fine-tuning small models (up to 7B) on 6GB VRAM using Unsloth:

```python
# save as unsloth_demo.py
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing="unsloth",
)

print(f"Model loaded on: {model.device}")
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
```

### 8.4 VRAM Budget on RTX 4050 (6GB)

```
Available VRAM:    ~5,444 MB (after OS/CUDA overhead)

Unsloth QLoRA (Qwen2.5-7B):
  Model:            4,500 MB  (Q4_K_M quantized)
  KV cache (4k):     300 MB
  Optimizer states:   400 MB
  Gradients:          400 MB
  ─────────────      ──────
  Total:            5,600 MB  (tight — reduce r=8 if OOM)

PPO RL Training:
  Policy network:     150 MB
  Replay buffer:      200 MB
  Rollout storage:    100 MB
  ─────────────      ──────
  Total:              450 MB  (comfortable)

Ollama Inference:
  Qwen2.5-7B Q4:    4,500 MB
  Phi-3.5-mini Q4:  2,500 MB
  Gemma-2-2B Q4:    1,500 MB
```

### 8.5 Monitoring GPU

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# For GPU temperature and fan (text UI)
nvtop

# Programmatic check
python3 -c "
import torch
print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'Cached:    {torch.cuda.memory_reserved()/1e9:.2f} GB')
"
```

---

## 9. Docker GPU Configuration

### 9.1 Using GPU in Containers

The orchestrator container doesn't need GPU (all inference is cloud API calls).  
But for local fine-tuning or Ollama:

```bash
# Run Ollama with GPU access (requires nvidia-container-runtime)
docker run -d --gpus all \
    -p 11434:11434 \
    -v ollama_data:/root/.ollama \
    ollama/ollama

# Verify GPU in container
docker exec -it <container_id> nvidia-smi
```

### 9.2 Docker Compose GPU Resources

Add this to `docker-compose.yml` when adding GPU-dependent services:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    profiles: ["local-inference"]  # Only starts when explicitly enabled
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ollama_data:/root/.ollama
```

Start with GPU profile:

```bash
docker compose --profile local-inference up -d ollama
```

### 9.3 Docker Desktop GPU Settings

In **Docker Desktop → Settings → Resources → Advanced**:

- **CPUs:** At least 4 (recommended: 8 for development)
- **Memory:** At least 8GB (recommended: 16GB)
- **Swap:** 2GB
- **Disk image location:** Keep on SSD, not HDD

---

## 10. Working with the Codebase

### 10.1 Project Layout Quick Reference

```
~/quantex/
├── inference/              # Multi-provider LLM router (Groq, NIM, OpenRouter)
│   ├── router.py           #   Main router with fallback chains
│   ├── cache.py            #   Semantic caching via Qdrant
│   ├── cost_tracker.py     #   Token/cost monitoring
│   └── providers/          #   Individual API clients
│
├── orchestrator/           # FastAPI app — all AI/ML logic
│   ├── main.py             #   50+ REST endpoints + WebSocket
│   ├── agents.py           #   7 AI agent definitions
│   ├── agent_routing.py    #   Per-agent provider chains + adaptive routing
│   ├── inference_integration.py  # Bridge: agents → InferenceRouter
│   ├── main.py             #   FastAPI entry point
│   ├── metrics.py          #   Prometheus metrics
│   └── ...                 #   Strategy, ML, backtest, risk, etc.
│
├── execution/              # Rust — ultra-low latency order execution
├── realtime/               # Go — WebSocket broadcast + NATS bridge
├── frontend/               # TypeScript — Next.js dashboard
├── deployment/             # Nginx, K8s, Terraform configs
├── monitoring/             # Prometheus rules + Grafana dashboards
├── db/                     # PostgreSQL migration SQL
└── docker-compose.yml      # 12+ service stack
```

### 10.2 Common Development Workflows

**Adding a new agent:**
1. Define the agent class in `orchestrator/agents.py`
2. Add provider chain in `orchestrator/agent_routing.py` (`AGENT_PROVIDER_CHAINS`)
3. Add task type override in `orchestrator/agent_routing.py` (`AGENT_TASK_TYPE_OVERRIDES`)
4. Instantiate in `orchestrator/main.py` lifespan
5. Add REST endpoint in `orchestrator/main.py` if needed

**Adding a new inference provider:**
1. Create `inference/providers/new_provider.py` implementing `BaseProvider`
2. Register in `inference/router.py` `_create_providers()`
3. Add timeout config in `RouterConfig`
4. Add API key to `.env.example`

**Running backtests:**
```bash
curl -X POST http://localhost:8001/api/v1/backtest/run \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "interval": "1h", "days": 30}'
```

**Triggering swarm debate:**
```bash
curl -X POST http://localhost:8001/api/v1/swarm/debate \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "price": 50000, "regime": "unknown"}'
```

### 10.3 Testing

```bash
# Python — run with pytest (add tests to orchestrator/tests/ first)
cd ~/quantex/orchestrator
source .venv/bin/activate
pip install pytest  # if not already installed
python -m pytest tests/ -v

# TypeScript types check
cd ~/quantex/frontend
npx tsc --noEmit

# Rust tests
cd ~/quantex/execution
cargo test

# Go tests
cd ~/quantex/realtime
go test ./...
```

> **Note:** Python unit tests under `orchestrator/tests/` are not yet written. The test command above assumes you've added them. For ad-hoc testing, import and use the modules directly from a Python shell or script.

---

## 11. Troubleshooting

### 11.1 GPU / CUDA Issues

| Problem | Diagnosis | Fix |
|---------|-----------|-----|
| `nvidia-smi` not found | Driver not installed | Install NVIDIA driver on Windows, reboot |
| `nvidia-smi` shows no GPU | WSL2 not using GPU | Run `wsl --shutdown`, restart WSL2 |
| Docker `--gpus all` fails | nvidia-container-runtime missing | Enable WSL2 GPU in Docker Desktop settings |
| `torch.cuda.is_available()` = False | PyTorch built without CUDA | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| OOM in PPO training | Batch size too large | Reduce `n_steps` and `batch_size` by half |
| CUDA out of memory in Unsloth | 7B model + r=16 too large | Use `r=8`, reduce `max_seq_length` to 1024 |

### 11.2 Docker Issues

| Problem | Diagnosis | Fix |
|---------|-----------|-----|
| Container can't connect to postgres | Service not ready | Wait for health check, check `docker compose logs postgres` |
| Port already in use | Another service on same port | `netstat -tulpn \| grep <port>`, stop conflicting service |
| `wsl: detecting WSL2...` hangs | Docker Desktop not running | Start Docker Desktop from Windows Start Menu |
| Docker build slow | Files on /mnt/c/ | Move project to `~/quantex` inside WSL2 filesystem |

### 11.3 WSL2 Networking

```bash
# Check WSL2 IP
ip addr show eth0 | grep inet

# If ports aren't forwarding from Windows localhost
wsl --shutdown
# Wait 10 seconds, then restart WSL2

# For DNS issues in WSL2
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 1.1.1.1" >> /etc/resolv.conf'
sudo chattr +i /etc/resolv.conf  # Prevent auto-overwrite
```

### 11.4 API Connection Issues

```bash
# Test orchestrator directly
curl http://localhost:8001/health

# Test through nginx
curl http://localhost/health

# Check nginx logs
docker compose logs nginx

# Check orchestrator logs
docker compose logs orchestrator

# In development mode (outside Docker), check uvicorn output in terminal
```

### 11.5 Performance Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Git operations slow | Repo on /mnt/c/ | Move to `~/quantex` |
| Docker build slow | Repo on /mnt/c/ | Move to `~/quantex` |
| LLM inference slow | Circuit breaker open | Check `GET /api/v2/inference/adaptive-routing` |
| High GPU temps (>85°C) | Sustained load | Set power limit: `nvidia-smi -pl 75` |
| WebSocket drops | NATS not running | `docker compose up -d nats` |

---

## 12. Performance Tuning

### 12.1 GPU Power & Thermal Management

```bash
# Set power limit to reduce thermal throttling (RTX 4050 default: 95W)
sudo nvidia-smi -pl 75

# Enable persistence mode (faster GPU wake from idle)
sudo nvidia-smi --persistence-mode=1

# Monitor temperatures
watch -n 1 nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory --format=csv
```

### 12.2 WSL2 Resource Allocation

Edit `%USERPROFILE%\.wslconfig` for optimal performance:

```ini
[wsl2]
memory=16GB
processors=8
swap=2GB
localhostForwarding=true
```

After editing, run `wsl --shutdown` and restart.

### 12.3 Docker Resource Limits

The `docker-compose.yml` already has resource limits for each service:

| Service | CPU Limit | Memory Limit |
|---------|-----------|-------------|
| postgres | 1.0 core | 1GB |
| redis | 0.5 core | 512MB |
| qdrant | 1.0 core | 1GB |
| nats | 0.5 core | 256MB |
| orchestrator | 2.0 cores | 4GB |
| realtime | 1.0 core | 1GB |
| frontend | 0.5 core | 1GB |
| nginx | 0.5 core | 256MB |

Adjust these in `docker-compose.yml` if you have more or less RAM available.

### 12.4 LLM Inference Latency

Typical cloud inference latencies from the project's measured data:

| Provider | Model | P50 Latency | P99 Latency |
|----------|-------|-------------|-------------|
| Groq | Llama-3.1-8B | 180ms | 450ms |
| Groq | Mixtral-8x7B | 350ms | 800ms |
| NVIDIA NIM | Nemotron-8B | 420ms | 950ms |
| OpenRouter | DeepSeek V4 Flash | 650ms | 1.5s |

The adaptive router automatically reorders providers based on real observed latencies per agent. Check current performance at:

```bash
curl http://localhost:8001/api/v2/inference/adaptive-routing
```

### 12.5 Local Model Fallback (Offline Mode)

To run local models as fallback when cloud APIs are unavailable:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download fallback models
ollama pull qwen2.5:7b-q4_K_M     # 4.5 GB VRAM — primary fallback
ollama pull phi-3.5:3.8b-mini-q4  # 2.5 GB VRAM — low VRAM fallback

# Test local inference
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen2.5:7b", "prompt": "Analyze BTC trend", "stream": false}'
```

---

## Appendix: Useful Commands Cheat Sheet

```bash
# ── WSL2 ──
wsl --list --verbose           # List WSL2 distros
wsl --shutdown                 # Restart WSL2 kernel
wsl -d Ubuntu-24.04            # Start specific distro

# ── GPU ──
nvidia-smi                     # GPU status
nvidia-smi -pl 75              # Set power limit to 75W
nvtop                          # GPU process monitor (text UI)
watch -n 1 nvidia-smi          # Real-time GPU monitor

# ── Docker ──
docker compose up -d           # Start all services
docker compose up -d postgres redis  # Start specific services
docker compose logs -f orchestrator  # Follow service logs
docker compose down            # Stop all services
docker compose --profile local-inference up -d  # Start with GPU profile

# ── Project ──
cd ~/quantex && python -m orchestrator.main     # Start orchestrator
cd ~/quantex/frontend && npm run dev            # Start frontend
cd ~/quantex/execution && cargo build --release # Build Rust engine

# ── API ──
curl http://localhost:8001/health               # Health check
curl http://localhost:8001/api/v1/portfolio     # Portfolio status
curl http://localhost:8001/metrics              # Prometheus metrics
```
