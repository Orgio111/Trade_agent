---
title: Infrastructure Overview
type: overview
tags:
  - infrastructure
  - docker
  - kubernetes
  - terraform
  - nginx
  - deployment
created: 2026-06-30
updated: 2026-07-16
status: draft
---

# Infrastructure Overview

Single source of truth for the QUANTEX deployment architecture across three environments: Docker Compose (local/dev), Kubernetes (production), and Terraform (bare-metal provisioning).

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 7-8: MONITORING                            │
│   Prometheus (9090) ── Grafana (3001) ── NATS Exporter (7777)      │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 6: REVERSE PROXY                           │
│   Nginx (80/443) ── SSL termination ── Rate limiting ── Routing    │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 4-5: APPLICATION SERVICES                  │
│   Frontend (3000) ── Orchestrator (8001) ── Realtime (8082)        │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 2-3: STORAGE + STREAMING                   │
│   Qdrant (6333) ── NATS JetStream (4222) ── InfluxDB (8086)       │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 1: INFRASTRUCTURE                          │
│   PostgreSQL (5432) ── Redis (6379)                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Service Catalog (Docker Compose)

| Service | Image | Port(s) | CPU/Mem Limit | Health Check | Layer |
|---------|-------|:-------:|:-------------:|:------------:|:-----:|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | 1.0 CPU / 1G | `pg_isready` | 1 |
| `redis` | `redis:7.2-alpine` | 6379 | 0.5 CPU / 512M | `redis-cli ping` | 1 |
| `qdrant` | `qdrant/qdrant:latest` | 6333, 6334 | 1.0 CPU / 1G | — | 2 |
| `nats` | `nats:2.10-alpine` | 4222, 8222 | 0.5 CPU / 256M | `wget /healthz` | 2-3 |
| `influxdb` | `influxdb:2.7` | 8086 | 0.5 CPU / 512M | — | 2 |
| `orchestrator` | Custom build | 8001 | 2.0 CPU / 4G | `GET /health` | 4-5 |
| `realtime` | Custom build | 8082 | 1.0 CPU / 1G | `GET /health` | 4-5 |
| `frontend` | Custom build | 3000 | 0.5 CPU / 1G | — | 4-5 |
| `nginx` | Custom build | 80, 443 | 0.5 CPU / 256M | — | 6 |
| `certbot` | `certbot/certbot:v2.2.0` | — | — | — | 6 |
| `prometheus` | `prom/prometheus:latest` | 9090 | 0.5 CPU / 512M | — | 7-8 |
| `nats-exporter` | `natsio/prometheus-nats-exporter` | 7777 | 0.25 CPU / 128M | — | 7-8 |
| `grafana` | `grafana/grafana:latest` | 3001→3000 | 0.5 CPU / 256M | — | 7-8 |

**Total:** 13 services, 10 named volumes, ~8.75 CPU / ~11.5G memory limits.

## Docker Compose: Quick Start

```bash
# Production (all 13 services)
docker compose up -d

# Dev mode (hot-reload, no nginx)
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Logs
docker compose logs -f orchestrator realtime nats

# Health checks
curl http://localhost:8001/health
curl http://localhost:8082/health
curl http://localhost:8222/healthz
curl http://localhost:9090/-/healthy
```

### Dev Mode Overrides

| Service | Production | Dev Override |
|---------|-----------|--------------|
| `orchestrator` | Build image, no exposed port | Mount source, `uvicorn --reload`, port 8001 |
| `frontend` | Build image, no exposed port | Mount source, `npm run dev`, port 3000 |
| `nginx` | Active (80/443) | Disabled via `profiles: [production]` |

## Docker Compose: Volume Persistence

| Volume | Service | Purpose |
|--------|---------|---------|
| `postgres_data` | PostgreSQL | Database state |
| `redis_data` | Redis | Cache + session state |
| `qdrant_data` | Qdrant | Vector embeddings |
| `nats_data` | NATS | JetStream message persistence |
| `influx_data` | InfluxDB | Time-series metrics |
| `nginx_logs` | Nginx | Access/error logs |
| `certbot_www` | Certbot | ACME challenge files |
| `certbot_certs` | Certbot | Let's Encrypt certificates |
| `prometheus_data` | Prometheus | 30-day metrics retention |
| `grafana_data` | Grafana | Dashboard + config |

## Nginx Reverse Proxy

### Routing Rules

| Path | Target | Rate Limit | Notes |
|------|--------|:----------:|-------|
| `/api/*` | `orchestrator:8001` | 30 r/s, burst 20 | Strips `/api` prefix, adds `/api/v1` |
| `/api/v2/*` | `orchestrator:8001` | 30 r/s, burst 20 | Direct passthrough |
| `/ws` | `orchestrator:8001/ws` | 10 r/s, burst 5 | WebSocket upgrade, 1h timeout |
| `/ws-market` | `realtime:8082/ws` | 10 r/s, burst 5 | WebSocket upgrade, 1h timeout |
| `/metrics` | `orchestrator:8001/metrics` | 5 r/s | Internal-only (RFC1918) |
| `/*` | `frontend:3000` | — | Default, Next.js HMR support |

### SSL & Security

- **TLS 1.2/1.3** with Let's Encrypt certificates
- **Fallback:** Self-signed certs generated at startup if LE certs not yet obtained
- **Auto-renewal:** Certbot checks every 12h, nginx reloads every 12h
- **Security headers:** HSTS, X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Referrer-Policy
- **HTTP→HTTPS redirect** on port 80

## Kubernetes Deployment

### Resource Summary (25 K8s Resources)

| Resource Type | Count | Components |
|:-------------:|:-----:|------------|
| Deployment | 8 | orchestrator, realtime, frontend, redis, nats, nats-exporter, prometheus, grafana |
| StatefulSet | 1 | PostgreSQL |
| Service | 9 | All above + headless services |
| ConfigMap | 2 | quantex-brain-weights, quantex-prometheus-config |
| PersistentVolumeClaim | 2 | postgres-data, quantex-model-cache |
| RayCluster | 1 | RL training cluster |

### K8s Resource Requests/Limits

| Component | CPU Request | CPU Limit | Mem Request | Mem Limit |
|-----------|:-----------:|:---------:|:-----------:|:---------:|
| orchestrator (×2) | 1 | 2 | 2Gi | 4Gi |
| realtime (×2) | 0.5 | 1 | 512Mi | 1Gi |
| frontend (×2) | 0.5 | 1 | 512Mi | 1Gi |
| postgres (×1) | — | — | — | — |
| redis (×1) | — | — | — | — |
| nats (×1) | 0.25 | 0.5 | 256Mi | 512Mi |
| nats-exporter (×1) | 0.1 | 0.25 | 64Mi | 128Mi |
| vllm (×1) | 2 | 4 | 8Gi | 16Gi |
| **Total** | **~6.35** | **~13.75** | **~14.5Gi** | **~28.5Gi** |

### GPU Inference (vLLM)

```yaml
# GPU Node requirements
nodeSelector:
  nvidia.com/gpu: present
tolerations:
  - key: nvidia.com/gpu
    effect: NoSchedule
resources:
  nvidia.com/gpu: 1
```

| GPU | 7B Model | 13B Model | 70B Model |
|-----|:--------:|:---------:|:---------:|
| RTX 4090 (24GB) | 50-150ms | 150-400ms | N/A (OOM) |
| A100 80GB | 20-80ms | 40-120ms | 100-300ms |
| H100 80GB | 15-50ms | 30-80ms | 50-200ms |

### K8s Secrets

```bash
# Load these values from a local secret manager. Bash process substitution keeps
# the values out of the kubectl command arguments.
kubectl -n quantex create secret generic quantex-secrets \
  --from-file=nvidia-api-key=<(printf '%s' "${NVIDIA_API_KEY:?NVIDIA_API_KEY is required}") \
  --from-file=openrouter-api-key=<(printf '%s' "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}") \
  --from-file=groq-api-key=<(printf '%s' "${GROQ_API_KEY:?GROQ_API_KEY is required}") \
  --from-file=postgres-password=<(printf '%s' "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}") \
  --from-file=database-url=<(printf '%s' "${DATABASE_URL:?DATABASE_URL is required}") \
  --from-file=grafana-admin-password=<(printf '%s' "${GRAFANA_ADMIN_PASSWORD:?GRAFANA_ADMIN_PASSWORD is required}")
```

### Brain Weights ConfigMap

```bash
kubectl -n quantex edit configmap quantex-brain-weights
kubectl -n quantex rollout restart deployment quantex-realtime
```

### Ray Cluster (RL Training)

- **Head:** 4 CPU, 8Gi RAM (dashboard port 8265, GCS port 10001)
- **Workers:** 3× (2 CPU, 4Gi RAM each)
- **Total cluster:** 10 CPU, 20Gi RAM

## Terraform (Hetzner Cloud)

### Deployment Tiers

| Tier | Server Type | vCPU | RAM | Cost | Use Case |
|------|------------|:----:|:---:|:----:|----------|
| `tier0` | `cx21` | 2 | 4GB | ~$4-5/mo | MVP, testing |
| `tier1` | `cx41` | 4 | 16GB | ~$20-40/mo | Production |

### Firewall Rules

| Port | Protocol | Source | Purpose |
|:----:|:--------:|--------|---------|
| 22 | TCP | 0.0.0.0/0 | SSH |
| 80 | TCP | 0.0.0.0/0 | HTTP (redirect to HTTPS) |
| 443 | TCP | 0.0.0.0/0 | HTTPS |
| 3000 | TCP | 0.0.0.0/0 | Frontend |
| 8001 | TCP | 0.0.0.0/0 | Orchestrator API |
| 9090 | TCP | 0.0.0.0/0 | Prometheus |
| 3001 | TCP | 0.0.0.0/0 | Grafana |
| 4222 | TCP | 10.0.0.0/8, 172.16.0.0/12 | NATS (internal only) |

### Provisioning Commands

```bash
cd deployment/terraform
export TF_VAR_hcloud_token="your-token"
export TF_VAR_deployment_tier="tier0"  # or "tier1"
terraform init
terraform apply
```

Server auto-provisions Docker + Docker Compose via `remote-exec` provisioner.

## Data Flow (Trinity Architecture)

```
Layer A (Python)                Layer B (Go)                 Layer C (Next.js)
┌─────────────────┐           ┌─────────────────┐           ┌──────────────┐
│ orchestrator     │──raw────▶│  realtime       │──agg─────▶│  frontend    │
│ 12 brains via    │  NATS    │  weighted       │  NATS     │  dashboard   │
│ NATS JetStream   │  4222    │  aggregation    │  WebSocket│  components  │
└────────┬────────┘           └────────┬────────┘           └──────────────┘
         │                             │
         ▼                             ▼
   ┌──────────┐                ┌──────────────┐
   │  NATS    │                │ Prometheus   │
   │JetStream │                │ (metrics)    │
   └──────────┘                └──────────────┘
```

### NATS Subject Hierarchy

| Category | Subject Pattern | Stream | Retention |
|----------|----------------|:------:|:---------:|
| `signals` | `signals.raw.<source>.<symbol>` | file | 7 days |
| `market` | `market.<type>.<symbol>` | file | 3 days |
| `portfolio` | `portfolio.<type>` | file | 30 days |
| `rl` | `rl.<type>.<agent>` | file | 14 days |
| `ws` | `ws.<type>` | memory | — |
| `system` | `system.<type>` | memory | 7 days |

## Monitoring Stack

### Prometheus

- **Targets:** NATS exporter (connz, routez, subz, varz), vLLM metrics (`/metrics`)
- **Retention:** 30 days (`--storage.tsdb.retention.time=30d`)
- **Alerts:** `monitoring/alerts.yml`

### Grafana

- **URL:** http://localhost:3001 (Docker) / http://localhost:3000 (K8s)
- **User:** `admin`
- **Password source:** `GRAFANA_ADMIN_PASSWORD` (Docker) or `quantex-secrets/grafana-admin-password` (K8s)
- **Dashboards:** Auto-provisioned from `monitoring/grafana/dashboards/`
- **Datasources:** Auto-provisioned from `monitoring/grafana/datasources/`

## Resource Limits Comparison

| Environment | Total CPU | Total Memory | GPU |
|------------|:---------:|:------------:|:---:|
| Docker Compose (limits) | 8.75 | ~11.5G | — |
| Kubernetes (requests) | ~6.35 | ~14.5Gi | — |
| Kubernetes (limits) | ~13.75 | ~28.5Gi | 1× GPU (vllm) |
| Terraform tier0 | 2 vCPU | 4GB | — |
| Terraform tier1 | 4 vCPU | 16GB | — |

## Cleanup

```bash
# Docker Compose
docker compose down          # Preserve volumes
docker compose down -v       # Destroy everything

# Kubernetes
kubectl delete namespace quantex

# Terraform
cd deployment/terraform && terraform destroy
```

## Related

- [[trade-project-full-integration-build-plan]] — current audited target architecture and migration plan
- [[nats-event-system]] — NATS JetStream event backbone details
- [[vllm-inference-provider]] — vLLM GPU inference provider
- [[brain-ecosystem]] — Brain weights stored in ConfigMap
- [[real-time-trading-dashboard]] — Frontend dashboard architecture
- [[ppo-portfolio-manager]] — RL training (Ray Cluster)
- [[broker-abstraction-layer]] — Execution layer

## Contradictions / updates

**2026-07-14 repository audit:** this page describes an earlier deployment intent. The current Compose file defines 21 services, several new services do not yet use the NATS/PostgreSQL/Redis connections implied by the diagram, and the full topology is too large for the stated 16 GB laptop target. [[trade-project-full-integration-build-plan]] is the current decision: use profiled infrastructure and a three-worker core; keep Kubernetes, Ray, vLLM, Qdrant, InfluxDB, Go, and Rust outside the default MVP until a measured requirement exists.
