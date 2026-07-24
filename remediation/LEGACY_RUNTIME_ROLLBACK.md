# Legacy Runtime Containment and Rollback Record

Captured: 2026-07-24 (Asia/Ulaanbaatar)  
Compose project: `trade_agent`  
Action boundary: stop only; do not remove containers, networks, or volumes.

The current topology is the stale 23-container legacy deployment. All project
volumes remain intact and must be treated as read-only legacy recovery state until
the isolated restore and canonical convergence gates pass.

| Container | Image | Pre-containment state | Health signal | Persistent/bind destinations |
|---|---|---|---|---|
| `quantex-chroma` | `chromadb/chroma:1.5.9` | running | no container healthcheck | `/data` |
| `quantex-nginx` | `trade_agent-nginx` | running | no container healthcheck | `/etc/letsencrypt`, `/var/log/nginx`, `/var/www/certbot` |
| `quantex-nats` | `nats:2.10-alpine` | running | healthy | `/data` |
| `quantex-frontend` | `node:22-alpine` | running | healthy | `/app`, `/app/.next`, `/app/node_modules` |
| `quantex-orchestrator` | `trade_agent-orchestrator` | running | shallow healthy | `/app/.env`, `/app/inference`, `/app/models`, `/app/orchestrator` |
| `quantex-orderbook` | `trade_agent-orderbook` | running | none | `/app/data` |
| `quantex-moe-router` | `trade_agent-moe-router` | running | none | `/app/data` |
| `quantex-rl-learning` | `trade_agent-rl-learning` | running | none | `/app/data`, `/app/models` |
| `quantex-dashboard` | `trade_agent-dashboard` | running | none | none |
| `quantex-feature-engine` | `trade_agent-feature-engine` | running | none | `/app/data` |
| `quantex-risk` | `trade_agent-risk` | running | none | `/app/data` |
| `quantex-market-data` | `trade_agent-market-data` | running | none | `/app/data` |
| `quantex-execution` | `trade_agent-execution` | running | none | `/app/data` |
| `quantex-inference` | `trade_agent-inference` | running | false-positive HTTP health | none |
| `quantex-grafana` | `grafana/grafana:latest` | running | HTTP only | `/var/lib/grafana`, provisioning binds |
| `quantex-prometheus` | `prom/prometheus:latest` | running | HTTP only; config invalid | `/prometheus`, config/rule binds |
| `quantex-realtime` | `trade_agent-realtime` | running | healthy transport; no signals | none |
| `quantex-vllm` | `vllm/vllm-openai:latest` | running/restarting | model unavailable; restart count 79 | none |
| `quantex-influxdb` | `influxdb:2.7` | running | HTTP only | `/etc/influxdb2`, `/var/lib/influxdb2` |
| `quantex-redis` | `redis:7.2-alpine` | running | healthy | `/data` |
| `quantex-nats-exporter` | `natsio/prometheus-nats-exporter:latest` | running | none | none |
| `quantex-postgres` | `pgvector/pgvector:pg16` | running | healthy | `/var/lib/postgresql/data`, migration bind |
| `quantex-qdrant` | `qdrant/qdrant:latest` | running | HTTP only | `/qdrant/storage` |

Pre-containment safety facts:

- Paper mode was reported, but there was no canonical schema/kill switch.
- NATS had zero messages and zero consumers.
- All 34 mutation operations were anonymous.
- Numerous host bindings used `0.0.0.0`/`[::]`.
- No project container or volume is approved for deletion.

Rollback command, only after a documented need:

```powershell
docker start quantex-chroma quantex-nginx quantex-nats quantex-frontend `
  quantex-orchestrator quantex-orderbook quantex-moe-router `
  quantex-rl-learning quantex-dashboard quantex-feature-engine quantex-risk `
  quantex-market-data quantex-execution quantex-inference quantex-grafana `
  quantex-prometheus quantex-realtime quantex-vllm quantex-influxdb `
  quantex-redis quantex-nats-exporter quantex-postgres quantex-qdrant
```

Rollback restores the insecure legacy exposure and therefore requires a documented
incident/debugging reason. It does not make the stack a release candidate.

