# Sentinel-X: Institutional-Grade Multi-Agent AI Trading Fund

**Tri-language architecture** — Python brain · Go heart · Rust shield  
**NVIDIA NIM + Triton** · **LangGraph** · **gRPC/UDS** · **SBE** · **FAISS R-Mem** · **H100 K8s cluster**

---

## System Architecture — Mermaid Diagram

```mermaid
flowchart TD
    subgraph INGEST["Data Ingestion"]
        WS["WebSocket Feed\nccxt.pro"]
        NEWS["News Scraper\nMCP-based"]
        ONCHAIN["On-Chain API\nCryptoCompare"]
    end

    subgraph PYTHON["Python Brain — GPU Node"]
        subgraph RMEM["R-Mem Layer"]
            FAISS["FAISS Vector Store\nStrategy Memory"]
            RMEM_AGENT["R-Mem Agent\nRetrieves similar setups"]
        end

        subgraph COUNCIL["5-Agent Council (parallel)"]
            BULL["Bull Agent\n(NVIDIA NIM LLM)"]
            BEAR["Bear Agent\n(NVIDIA NIM LLM)"]
            FUND["Fundamental Agent\n(NVIDIA NIM LLM)"]
            SENT["Sentiment Agent\n(NIM + MCP scraping)"]
            QUANT["Quant Agent\nHurst · GK-Vol · Regime\nCPU/NumPy"]
        end

        SCHED["Intelligent Scheduler\nGPU→NIM | CPU→NumPy/Polars\nCircuit Breaker 200ms"]
        STL["STL Protocol\nCalibrated Confidence Score\n75% gate"]
        SUPER["Supervisor Agent\n(NVIDIA NIM LLM)\nFinal rationale + dispatch"]
    end

    subgraph GO["Go Heart — Low-Latency"]
        GW["API Gateway\nHTTP + gRPC router"]
        OMS["Order Management System\nAsync execution\nExponential backoff retry"]
        CB["Circuit Breaker\n(gobreaker)\nFallback on risk timeout"]
    end

    subgraph RUST["Rust Shield — Risk Engine"]
        VAR["Monte Carlo VaR\n+ Historical Simulation\n+ Parametric blend"]
        KELLY["Quarter-Kelly Criterion\nmax(Kelly×0.25, 10% equity)"]
        CORR["Portfolio Heat Map\nCorrelation check\n0.7 threshold"]
        SBE_ENC["SBE Encoder\n40-byte zero-copy frame"]
        KS["Kill Switch\nAtomic CAS\nDD > 5% → HALT"]
    end

    subgraph BACKTEST["Rust Backtest Engine"]
        WFO["Walk-Forward Optimizer\nDifferential evolution\nSharpe maximization"]
        DRIFT["PSI Drift Detector\nPSI > 0.2 → retrain"]
    end

    subgraph INFRA["Infrastructure"]
        TRITON["Triton Inference Server\nDynamic batching\nH100 GPU cluster"]
        REDIS["Redis Streams\nSBE-framed messages"]
        PG["PostgreSQL\nTrade log + prompt history"]
        PROM["Prometheus\nFour Golden Signals"]
        GRAFANA["Grafana\nPnL · Sharpe · Agent Confidence"]
    end

    %% Data flow
    WS & NEWS & ONCHAIN -->|raw data| RMEM_AGENT
    RMEM_AGENT -->|retrieves k-NN context| FAISS
    FAISS -->|historical reasoning| RMEM_AGENT
    RMEM_AGENT -->|R-Mem context| COUNCIL

    SCHED -->|LLM tasks| TRITON
    SCHED -->|CPU tasks| QUANT
    BULL & BEAR & FUND & SENT & QUANT --> STL
    STL -->|confidence ≥ 75%| SUPER
    STL -->|confidence < 75%| REDIS

    %% gRPC UDS
    SUPER -->|"POST /v1/trade\n(HTTP)"| GW
    GW --> CB
    CB -->|"gRPC/UDS"| RUST
    VAR & KELLY & CORR --> SBE_ENC
    SBE_ENC -->|"40-byte frame"| KS
    KS -->|approved + sized order| OMS
    OMS -->|"retry × 4"| EXCHANGE["Exchange\n(ccxt)"]

    %% Memory loop
    EXCHANGE -->|fill confirmation| PG
    PG -->|trade outcome| RMEM_AGENT
    RMEM_AGENT -->|post-mortem NIM| FAISS
    RMEM_AGENT -->|refined prompts| COUNCIL

    %% Observability
    PROM -.->|scrape| PYTHON & GO & RUST & TRITON
    GRAFANA -.->|query| PROM

    %% Retraining
    DRIFT -->|PSI > 0.2| WFO
    WFO -->|new model| TRITON
```

---

## Project Structure

```
sentinel-x/
├── proto/                          # Language-neutral gRPC definitions
│   ├── risk.proto                  # Rust ↔ Go: VaR + Kelly + SBE
│   ├── orders.proto                # Go ↔ Exchange: OMS order lifecycle
│   └── agents.proto                # Python ↔ Go: council decisions
│
├── python/                         # The Brain (GPU, LLM, ML)
│   ├── core/
│   │   ├── nim_client.py           # NVIDIA NIM async OpenAI-compatible client
│   │   └── scheduler.py            # GPU/CPU router + 200ms circuit breaker
│   ├── agents/
│   │   └── quant.py                # Hurst, GK-Vol, regime detection
│   ├── council/
│   │   ├── supervisor.py           # LangGraph StateGraph with R-Mem
│   │   └── stl_protocol.py         # 5-agent STL calibration + 75% gate
│   └── memory/
│       ├── vector_store.py         # FAISS strategy memory (1024-dim)
│       └── r_mem.py                # R-Mem: retrieval + prompt engineering
│
├── go/                             # The Heart (low-latency routing)
│   ├── cmd/sentinel/main.go        # Service entry point
│   └── internal/
│       ├── gateway/gateway.go      # HTTP API gateway (confidence gate)
│       └── oms/executor.go         # Async OMS + circuit breaker + retry
│
├── rust/                           # The Shield (risk, backtest, SBE)
│   └── src/
│       ├── risk/
│       │   ├── var.rs              # Historical + MC + Parametric VaR blend
│       │   ├── kelly.rs            # Quarter-Kelly + ATR stops (unit tested)
│       │   ├── correlation.rs      # Pearson + portfolio heat map
│       │   ├── kill_switch.rs      # Atomic CAS kill switch
│       │   └── engine.rs           # gRPC service — wires everything
│       ├── sbe/encoder.rs          # 40-byte zero-copy SBE frame
│       └── backtest/engine.rs      # ATR-breakout WFO backtest
│
└── infra/
    ├── k8s/
    │   ├── deployment.yaml         # Python/Go/Rust/Triton deployments
    │   └── gpu-node-pool.yaml      # H100 node pool + NVIDIA device plugin
    ├── docker/
    │   ├── Dockerfile.python       # CUDA 12.4 base
    │   ├── Dockerfile.go           # Distroless runtime
    │   └── Dockerfile.rust         # release LTO build
    └── prometheus/                 # Four Golden Signals config
```

---

## Trade Flow Walkthrough: News Flash → Filled Order

```
T+0ms    News flash: "Fed raises rates 50bps unexpectedly"
         ↓ MCP news scraper captures headline

T+5ms    R-Mem queries FAISS for k=5 similar historical setups
         → finds 3 past "surprise rate hike" setups (2 losses, 1 win)
         → synthesizes: "Rate shocks historically trigger 2-4hr BTC sell-off"

T+10ms   Intelligent Scheduler fans out to 5 agents in parallel:
         • Bull  (GPU/NIM):   Looks for accumulation signals, DCA opportunity
         • Bear  (GPU/NIM):   Rate-shock = risk-off, leverage unwinding imminent
         • Fundamental (GPU): on_chain_score=-0.3, MVRV=3.2 (historically risky)
         • Sentiment (GPU):   news=-0.8, social=-0.7, fear_greed=28 (fear)
         • Quant (CPU/NumPy): Hurst=0.48 (near random walk), regime=HIGH_VOL
                              → tradeable=False (crisis conditions)

T+35ms   STL Protocol aggregates all 5 arguments:
         → Quant says not tradeable → caps confidence at 60%
         → 3 contradictions between Bull supporting factors and Sentiment
         → confidence_pct = 58% < 75% threshold
         → BLOCKED: "Insufficient consensus: high volatility + rate shock"

T+36ms   Trade logged to PostgreSQL as BLOCKED
         R-Mem records reasoning chain in FAISS for future retrieval
         Prometheus counter: council_blocks_total++ (symbol="BTC/USDT")

------  [Alternative scenario: normal trending market] ------

T+10ms   Bull: EMA crossover + volume 1.8× avg + RSI=52 (not overbought)
         Bear: Near 6-month resistance → quantitative_score=0.35 (weak)
         Quant: Hurst=0.63 (trending), regime=LOW_VOL, tradeable=True, adj=+0.08
         Sentiment: news=+0.4, social=+0.3, fear_greed=62 (greed)

T+35ms   STL Protocol: bull_score=0.72, bear_score=0.28
         → 1 contradiction (-5%), no shallow reasoning
         → confidence_pct = 81% ≥ 75% ✓ APPROVED

T+36ms   Supervisor (NIM) generates rationale:
         "BTC/USDT presents a trending setup (Hurst=0.63) with EMA crossover
          confirmed by 1.8× volume expansion. VaR99=2.1% allows ¼ Kelly sizing
          of $8,750 (8.75% equity). Primary risk: 6-month resistance at $67,200."

T+37ms   Python POSTs to Go Gateway: POST /v1/trade
         Go Gateway checks confidence_pct=81% ≥ 75% ✓

T+38ms   Go OMS calls Rust Risk Engine via gRPC over Unix Domain Socket:
         → Monte Carlo VaR99=2.1% (100K paths, 1-day holding)
         → ¼ Kelly = 0.0875 → $8,750 size (capped at 10% equity)
         → ATR stop = $64,200 (2 × ATR below entry)
         → Portfolio heat = 0.42 (BTC+ETH correlation < 0.7 threshold) ✓
         → SBE-encoded 40-byte snapshot for audit

T+42ms   Rust Kill Switch: daily drawdown = 0.8% < 5% threshold ✓
         Risk APPROVED — response returns in 4ms via UDS

T+43ms   OMS dispatches TWAP order: 5 slices × 0.0308 BTC
         Retry logic: attempt 1 succeeds, slippage = 3.5 bps ✓

T+120ms  Order FILLED @ $65,023 avg (slippage 3.5 bps < 20 bps limit)
         PostgreSQL records trade outcome
         Prometheus: oms_orders_total{side="BUY",status="FILLED"}++
         Grafana: PnL panel updates in real-time
```

---

## Risk Control Stack (7 Layers)

| Layer | Where | Constraint |
|---|---|---|
| Confidence Gate | Python STL | ≥ 75% weighted consensus required |
| Volatility Regime | Python Quant | Crisis/untradeable regime → blocked |
| VaR Gate | Rust | Blended VaR99 (hist+MC+parametric) < 15% |
| Kelly Sizing | Rust | Position ≤ min(¼ Kelly, 10% equity) |
| Correlation Heat | Rust | Portfolio heat score < 0.75 |
| ATR Stop-Loss | Rust | Hard stop 2× ATR from entry |
| Kill Switch | Rust (atomic CAS) | DD > 5% → halt all services |

---

## NVIDIA NIM + Triton Integration

| Component | NIM Model | Backend |
|---|---|---|
| Bull/Bear/Fundamental/Sentiment/Supervisor | `meta/llama-3.1-70b-instruct` | NIM Cloud |
| Embeddings (R-Mem FAISS) | `nvidia/nv-embedqa-e5-v5` | NIM Cloud |
| PPO RL execution | Custom ONNX | Triton (H100) |
| Price prediction | Custom TensorRT | Triton (H100) |

All GPU inference routed through the `scheduler.py` circuit breaker — falls back to lightweight CPU heuristics if latency spikes > 200ms.

---

## IPC Architecture: Zero-Copy UDS + SBE

```
Python ─────── HTTP ──────────────► Go Gateway
                                         │
                                     gRPC/UDS
                                         │
                                         ▼
                                    Rust Risk Engine
                                         │
                                    SBE 40-byte frame
                               [ABCD|01|01|00000020|
                                var_99|kelly|size|heat]
                                         │
                                    Back to Go OMS
                                         │
                                    Order dispatch
```

Unix Domain Sockets eliminate TCP stack overhead (~50μs → ~5μs RTT).
SBE 40-byte frames eliminate JSON parsing for the risk snapshot audit trail.

---

## Build 3 — Sentinel-X Evolution: 7-Phase Fully Autonomous Hedge Fund

Extends Sentinel-X into a fully self-evolving, tokenized, multi-exchange autonomous hedge fund. No human prompt engineering required after deployment.

```mermaid
flowchart TD
    subgraph PHASE1["Phase 1 — Self-Evolving AI Core"]
        FC["Failure Classifier\nbad_prediction|regime_mismatch\nbad_execution|model_drift"]
        PE["Prompt Evolver\nUCB1 version control\nMeta-LLM rewriting"]
        MS["Model Selector\nUCB1 bandit\nLatency-penalized scoring"]
        HM["Hierarchical Memory\nSTM (100 trades, 24h decay)\nLTM (FAISS + pattern abstraction)"]
        FC -->|diagnosis| PE
        PE -->|evolved prompts| COUNCIL
        MS -->|best model| NIM
        HM -->|STM+LTM context| COUNCIL
    end

    subgraph PHASE2["Phase 2 — Multi-Exchange Arbitrage"]
        WS_B["Binance WS\nbook_ticker"] & WS_Y["Bybit WS"] & WS_O["OKX WS"]
        QUO["Lock-free DashMap\nBestQuote cache"]
        SPATIAL["Spatial Arb\nDetector\n< 1μs scan"]
        TRI["Triangular Arb\nA→B→C→A\nFee-aware"]
        WS_B & WS_Y & WS_O -->|"BestQuote\n(64-byte cache-line)"| QUO
        QUO --> SPATIAL & TRI
    end

    subgraph PHASE3["Phase 3 — Sub-10ms Execution"]
        SQ["Validate Queue\nArrayQueue lock-free"]
        EQ["Encode Queue\nSBE 40-byte frame"]
        DQ["Dispatch Queue\nThread-pinned"]
        PIN1["CPU Core 0\nValidation stage\n< 1μs"]
        PIN2["CPU Core 1\nSBE Encoding\n< 200ns"]
        PIN3["CPU Core 2\nExchange Dispatch\n< 5ms"]
        SQ -->|hot path| EQ -->|zero-copy| DQ
        PIN1 --- SQ
        PIN2 --- EQ
        PIN3 --- DQ
    end

    subgraph PHASE4["Phase 4 — On-Chain Tokenization"]
        SHARE["SentinelShare\n(ERC-20 SNTL)"]
        ORACLE["NAVOracle\n2-of-3 reporter consensus\n5% tolerance band"]
        FUND["SentinelFund\nDeposit/Withdraw\n2% mgmt + 20% perf fees\nHigh-water mark"]
        ORACLE -->|finalized NAV| FUND
        FUND -->|mint/burn| SHARE
    end

    subgraph PHASE5["Phase 5 — $1M Simulation"]
        SIM["FundSimulator\n252 trading days\nMonte Carlo paths"]
        RPT["Report\nSharpe · Sortino · Calmar\nMax DD · Win rate\nPer-strategy breakdown"]
        SIM --> RPT
    end

    subgraph PHASE6["Phase 6 — Gradual Rollout"]
        T["Testnet 1% ($10K)"]
        L1["Mainnet 5% ($50K)"]
        L2["Mainnet 20% ($200K)"]
        L3["Full $1M"]
        T -->|"1 week\nSharpe > 0.5"| L1
        L1 -->|"2 weeks\nDD < 5%"| L2
        L2 -->|"1 month\nKS never tripped"| L3
    end

    subgraph PHASE7["Phase 7 — GPU Cluster"]
        HEAD["Ray Head Node\n4 CPU, 8GB"]
        GPU_INF["GPU Inference Workers\n2-8 nodes, 1× H100\nLLM + Embed + PPO"]
        CPU_WRK["CPU Compute Workers\n2-20 nodes\nIndicators + Backtest"]
        H100["H100 Training Workers\n0-4 nodes (on-demand)\n8× H100, 640GB"]
        HEAD --> GPU_INF & CPU_WRK
        H100 -.->|"PSI > 0.2\nretrain trigger"| HEAD
    end

    SPATIAL & TRI -->|arb signals| PHASE3
    PHASE1 -->|evolved agents| COUNCIL["5-Agent Council\n(from Sentinel-X base)"]
    COUNCIL -->|decision| PHASE3
    PHASE3 -->|filled orders| ORACLE
    ORACLE -->|NAV update| FUND
    PHASE5 -->|calibration| PHASE1
    PHASE6 -->|live telemetry| PHASE1
    PHASE7 -->|model serving| NIM["NVIDIA NIM\nllama-3.1-70b"]
```

### Phase Directory Structure

```
sentinel-x/
├── SENTINEL_X_EVOLUTION.md             # Full architecture + risk matrix
│
├── phase1_self_evolving/python/
│   ├── evolution/
│   │   ├── failure_classifier.py       # 6 failure types, fast heuristic + NIM fallback
│   │   ├── prompt_evolver.py           # UCB1 version control, 70% confidence gate
│   │   └── model_selector.py           # UCB1 bandit with latency penalty
│   └── memory/
│       └── hierarchical_memory.py      # STM (100 trades, 24h decay) + LTM (FAISS)
│
├── phase2_arbitrage/rust/src/
│   ├── arbitrage/
│   │   ├── detector.rs                 # Spatial arb < 1μs, DashMap lock-free quotes
│   │   └── triangular.rs              # A→B→C→A fee-aware cycle
│   └── exchange/
│       └── interface.rs               # BestQuote 64-byte cache-line aligned
│
├── phase3_sub10ms/rust/src/execution/
│   └── order_pipeline.rs              # HotOrder 64-byte struct, ArrayQueue SPSC
│                                       # 3-stage: validate → SBE encode → dispatch
│                                       # Thread-pinned + spin_loop CPU isolation
│
├── phase4_tokenization/contracts/
│   ├── SentinelFund.sol               # Deposit/withdraw vault, 2%+20% fees, HWM
│   ├── NAVOracle.sol                  # 2-of-3 consensus, 5% tolerance, 1h staleness
│   └── ShareToken.sol                 # ERC-20 SNTL, fund-only mint/burn
│
├── phase5_simulation/python/
│   └── simulator.py                   # $1M Monte Carlo, sqrt market impact
│                                       # 252-day, 3 strategies, per-strategy breakdown
│
├── phase6_deployment/k8s/
│   └── gradual-rollout.yaml           # Argo Rollouts: 1%→5%→20%→100%
│                                       # Prometheus AnalysisTemplates, DD>5% rollback
│
└── phase7_gpu_cluster/ray/
    ├── cluster.yaml                    # 4 node types: head, inference, compute, H100
    └── serve_config.py                # Ray Serve: /llm + /embed + /ppo endpoints
```

### Phase Implementation Summary

| Phase | Week | Key Mechanism | Risk Mitigation |
|---|---|---|---|
| 1 — Self-Evolving AI | 1–2 | UCB1 model selection + prompt evolution + STM/LTM | 70% confidence gate before applying evolved prompts |
| 2 — Multi-Exchange Arb | 2–3 | Spatial + triangular arb, DashMap lock-free quotes | 200ms staleness guard + min 2 bps net profit |
| 3 — Sub-10ms Execution | 3–4 | 64-byte HotOrder, ArrayQueue SPSC, thread pinning | Isolated CPU cores via `taskset` |
| 4 — Fund Tokenization | 4–5 | ERC-20 SNTL, 2-of-3 NAV oracle, 24h cooldown | Hardhat tests + Slither static analysis |
| 5 — $1M Simulation | 5 | 252-day Monte Carlo, sqrt market impact model | Out-of-sample WFO validation windows |
| 6 — Gradual Rollout | 6–10 | Argo Rollouts canary, Prometheus gates | Hard kill switch + automatic Argo rollback |
| 7 — GPU Cluster | 8+ | Ray cluster auto-scaling, vLLM or NIM fallback | 2-replica minimum + circuit breaker |

### $1M Simulation Results (calibrated)

```
Capital:              $1,000,000
Trading days:         252
Strategies:           Momentum (2/day) + Spatial Arb (15/day) + Tri Arb (8/day)
Slippage model:       σ × √(qty/ADV),  σ = 0.1
Fee model:            0.04% taker (Binance VIP 0)

Expected Output:
  Sharpe Ratio:    ~1.2 – 1.8
  Max Drawdown:    4 – 7%
  Annual Return:   18 – 35%
  Win Rate:        61% blended
  Profit Factor:   1.4 – 1.8
```

```bash
# Run the simulation
cd sentinel-x/phase5_simulation/python
python simulator.py
```

### Gradual Rollout Gates

| Phase | Capital | Duration | Gate Condition |
|---|---|---|---|
| Testnet | $10K (1%) | 1 week | Sharpe > 0.5 |
| Mainnet 5% | $50K | 2 weeks | DD < 5% continuously |
| Mainnet 20% | $200K | 1 month | Kill switch never tripped |
| Full | $1M | Ongoing | All metrics green |

Automatic rollback: drawdown > 5% → Argo immediately reverts to paper trading.

### Running the Full System

```bash
# Phase 5: $1M capital simulation
cd sentinel-x/phase5_simulation/python
python simulator.py

# Phase 7: Deploy Ray GPU cluster
cd sentinel-x/phase7_gpu_cluster/ray
ray up cluster.yaml --yes
python serve_config.py

# Phase 6: Deploy with gradual rollout (requires kubectl + Argo Rollouts)
kubectl apply -f sentinel-x/phase6_deployment/k8s/gradual-rollout.yaml
```

### Risk Matrix

| Phase | Critical Risk | Mitigation |
|---|---|---|
| 1 | Prompt regression | 70% confidence gate + rollback by version |
| 2 | Arb evaporation lag | 200ms staleness + 2 bps min net profit |
| 3 | Spin-loop CPU monopoly | Isolated CPU affinity via `taskset` |
| 4 | Smart contract exploit | 24h cooldown + oracle consensus + audit |
| 5 | Overfitted simulation | Out-of-sample WFO windows |
| 6 | Live capital loss | Hard kill switch + Argo rollback |
| 7 | GPU node failure | 2-replica minimum + circuit breaker |
