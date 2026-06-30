---
title: InferenceRoutingPanel
type: entity
tags:
  - frontend
  - inference
  - routing
  - latency
  - adaptive
created: 2026-06-30
updated: 2026-06-30
source_file: frontend/src/components/InferenceRoutingPanel.tsx
---

# InferenceRoutingPanel

**File:** `frontend/src/components/InferenceRoutingPanel.tsx`
**Type:** React component (largest dashboard panel — ~600 lines, 5 sub-sections)
**Purpose:** Full observability into the multi-provider inference system — provider health, agent fallback chains, adaptive routing performance, latency heatmap, and cost tracking.

## Architecture

```
InferenceRoutingData (props)
    ├── provider_health    → ProviderHealthCard × N (Groq, NIM, OpenRouter)
    ├── agent_chains       → AgentChainRow × N (provider fallback order)
    ├── adaptive_routing   → AdaptivePerfCard × N + LatencyHeatmap
    ├── agent_mappings     → Agent Model Assignments table
    └── cost_usage         → Cost & Cache panels
```

## Data Interface

```typescript
interface InferenceRoutingData {
  provider_health: Record<string, ProviderHealth>;
  agent_chains: Record<string, AgentChain>;
  adaptive_routing: Record<string, AdaptiveAgentData>;
  agent_mappings: AgentMapping[];
  cost_usage: CostUsage;
}
```

## 5 Dashboard Sections

### 1. Summary Bar (4 stat cards)

| Stat | Color Logic |
|------|-------------|
| Providers Online | Green (all), Yellow (partial), Red (none) |
| Agents Tracked | Green (count with adaptive data) |
| Adapted Chains | Yellow (chains reordered by latency) |
| Budget Status | Green (OK) / Red (OVER) |

### 2. Provider Health Cards

Each provider shows:
- **Status badge:** ONLINE (green) / OFFLINE (gray)
- **Avg latency:** Numeric ms with colored bar (0–1000ms scale)
- **Error rate:** Green (<5%), Yellow (5–10%), Red (>10%)
- **Description:** Provider-specific text (e.g., "Ultra-fast LPU inference, 800+ tok/s")

| Provider | Color | Description |
|----------|-------|-------------|
| Groq LPU | `#00ff88` | Ultra-fast LPU inference, 800+ tok/s |
| NVIDIA NIM | `#00aaff` | High-quality reasoning with DeepSeek V4 Flash |
| OpenRouter | `#aa00ff` | 200+ models, universal fallback |

### 3. Agent Provider Chains

Shows each agent's fallback chain (e.g., Groq → NIM → OpenRouter) with:
- **Task override badge:** URGENT (red), FAST (yellow), REASONING (blue), ANALYSIS (green)
- **Chain arrows:** Provider labels with → separators
- **ADAPTED badge:** Yellow when chain was reordered by latency

### 4. Adaptive Routing Performance

Per-agent × per-provider performance cards:
- **EMA latency** (exponential moving average) with colored bar
- **P50 latency** (median)
- **Success rate:** Green (>95%), Yellow (80–95%), Red (<80%)
- **Sample count**
- **Adapted chain indicator:** Shows reordered provider sequence

### 5. Latency Heatmap

Matrix visualization: agents (rows) × providers (cells):
- **Cell color:** Green (fast) → Yellow (medium) → Red (slow) based on EMA latency
- **Hover:** Shows full tooltip (EMA, P50, success rate, samples) with scale effect
- **Adapted indicator:** Yellow "A" badge for agents with reordered chains
- **Color scale legend:** Green → Yellow → Red gradient bar

## Provider Color Map

```typescript
const PROVIDER_COLORS = {
  groq: "#00ff88",       // green — fastest
  nvidia_nim: "#00aaff", // blue — quality
  openrouter: "#aa00ff", // purple — fallback
};
```

## Task Override Colors

```typescript
const TASK_COLORS = {
  urgent: "#ff0044",       // red
  fast: "#ffaa00",         // orange
  reasoning: "#00aaff",    // blue
  analysis: "#00ff88",     // green
  classification: "#aa00ff", // purple
};
```

## Performance

- **Pure React** — no external charting library
- **CSS animations** — `slideUp` entrance animation per section (0.3s–0.55s staggered)
- **Debounced hover** — mouse enter/leave transforms with z-index and box-shadow

## API Endpoint

Data sourced from: `GET /api/v2/inference/routing`

## Related

- [[inference-router]] — Multi-provider inference routing backend
- [[vllm-inference-provider]] — Self-hosted GPU inference via vLLM
- [[real-time-trading-dashboard]] — Dashboard layout context
- [[brain-ecosystem]] — Brain weights and provider assignments
