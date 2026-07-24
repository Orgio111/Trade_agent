# Container Inventory

Snapshot date: 2026-07-24. The final inventory contained 23 running and 4 stopped
containers. An earlier discovery snapshot briefly contained 29 total/6 stopped;
two transient exited containers were no longer present later. The audit did not
remove Docker containers.

CPU and RAM are one point-in-time sample. `Mixed` volume means bind and/or named
volumes were present; exact secret-bearing mount sources are intentionally omitted.

| Container | Image | Purpose | Status | Health | Ports | Network | Volume | Restart | CPU | RAM | Main issue |
|---|---|---|---|---|---|---|---|---|---:|---:|---|
| quantex-chroma | `chromadb/chroma:1.5.9` | Knowledge/runtime vectors | Running | No container check | 127.0.0.1:8100 | default | Persistent | unless-stopped | 0.00% | 44.6 MiB | Critical advisory; no auth; runtime memory empty |
| quantex-nginx | local nginx | Edge proxy | Running | No check shown | 0.0.0.0:80,443 | default | Mixed | unless-stopped | 0.00% | 7.5 MiB | Public edge; historical key; no CSP |
| quantex-nats | `nats:2.10-alpine` | JetStream | Running | Healthy | 0.0.0.0:4222,8222 | default | Persistent | unless-stopped | 0.05% | 13.5 MiB | Legacy zero-message stream, ports public |
| quantex-frontend | `node:22-alpine` | Next.js UI | Running | Healthy | 0.0.0.0:3000 | default | Mixed | unless-stopped | 2.59% | 645.5 MiB | Experimental, E2E/build not verified |
| quantex-orchestrator | local | Legacy orchestrator | Running | Healthy | 0.0.0.0:8001 | default | Mixed | unless-stopped | 6.99% | 153.1 MiB | Unauth mutations, NATS disconnected, broad secrets |
| quantex-orderbook | local | Order book API | Running | Shallow HTTP | 0.0.0.0:8083 | default | Mixed | unless-stopped | 8.41% | 381.1 MiB | Empty books |
| quantex-moe-router | local | Mixture-of-experts | Running | Shallow HTTP | 0.0.0.0:8084 | default | Mixed | unless-stopped | 16.05% | 690.8 MiB | High idle cost; no verified outputs |
| quantex-rl-learning | local | RL service | Running | Shallow HTTP | 0.0.0.0:8088 | default | Mixed | unless-stopped | 0.26% | 236.2 MiB | Model restore failed; zero steps |
| quantex-dashboard | local | Dashboard API | Running | Misconfigured probe | 0.0.0.0:8080 | default | Mixed | unless-stopped | 0.26% | 46.5 MiB | `/health` mismatch; CORS `*`; empty portfolio |
| quantex-feature-engine | local | Legacy features | Running | Shallow HTTP | 0.0.0.0:8082 | default | Mixed | unless-stopped | 2.22% | 419.3 MiB | No symbols/data |
| quantex-risk | local | Legacy risk API | Running | Shallow HTTP | 0.0.0.0:8087 | default | Mixed | unless-stopped | 0.17% | 50.6 MiB | Not canonical authoritative gate |
| quantex-market-data | local | Legacy market feed | Running | Shallow HTTP | 0.0.0.0:8081 | default | Mixed | unless-stopped | 0.17% | 344.9 MiB | WebSocket 404 loop; status 500 |
| quantex-execution | local | Legacy paper/testnet API | Running | Shallow HTTP | 0.0.0.0:8096 | default | Mixed | unless-stopped | 0.17% | 45.7 MiB | Broker secrets supplied; user stream broken |
| quantex-inference | local | Provider router | Running | False-positive | 0.0.0.0:8085 | default | Mixed | unless-stopped | 0.17% | 39.4 MiB | Reports OK while vLLM fails |
| quantex-grafana | `grafana/grafana:latest` | Dashboards | Running | HTTP 200 | 0.0.0.0:3001 | default | Persistent | unless-stopped | 1.88% | 142.1 MiB | Mutable image; plugin errors; public |
| quantex-prometheus | `prom/prometheus:latest` | Metrics | Running | HTTP 200 | 0.0.0.0:9090 | default | Persistent | unless-stopped | 0.38% | 107 MiB | Invalid rules reference; public |
| quantex-realtime | local | Go aggregate/WebSocket | Running | Healthy | Internal 8082 | default | None | unless-stopped | 0.04% | 16.0 MiB | No signals; NATS inactive |
| quantex-vllm | `vllm/vllm-openai:latest` | GPU inference | Restarting/running | No ready model | 0.0.0.0:8000 | default | Persistent | unless-stopped | 98.23% | 886.8 MiB | Crash loop/resource admission |
| quantex-influxdb | `influxdb:2.7` | Time-series DB | Running | HTTP 200 | 0.0.0.0:8086 | default | Persistent | unless-stopped | 0.03% | 62.5 MiB | Public port; no verified application data |
| quantex-redis | `redis:7.2-alpine` | Cache | Running | Healthy | 0.0.0.0:6379 | default | Persistent | unless-stopped | 0.16% | 8.9 MiB | Empty; public port |
| quantex-nats-exporter | `...:latest` | NATS metrics | Running | No check shown | Internal 7777 | default | None | unless-stopped | 0.05% | 15.0 MiB | Mutable image |
| quantex-postgres | `pgvector/pgvector:pg16` | SQL state | Running | Healthy | 0.0.0.0:5432 | default | Persistent | unless-stopped | 0.00% | 53.2 MiB | Superuser runtime, legacy schema, public port |
| quantex-qdrant | `qdrant/qdrant:latest` | Vector DB | Running | HTTP green | 0.0.0.0:6333,6334 | default | Persistent | unless-stopped | 0.03% | 32.7 MiB | Zero points; mutable image; public ports |
| heuristic_curie | `nvidia/cuda:11.8.0-base-ubuntu22.04` | Ad hoc CUDA container | Exited (0) | N/A | None | bridge | None | no | — | — | Unmanaged residue |
| silly_dijkstra | `docker/desktop-storage-provisioner:v2.0` | Desktop/K8s helper | Exited (2) | N/A | None | bridge | None | no | — | — | Non-project residue |
| focused_lumiere | `registry.k8s.io/pause:3.10.1` | K8s sandbox | Exited (255) | N/A | None | bridge | None | no | — | — | Non-project residue |
| ubuntu-server | `ubuntu:24.04` | Ad hoc container | Exited (255) | N/A | None | bridge | None | no | — | — | Unmanaged residue |

Most observed application containers use root/default users, writable root
filesystems, no explicit capability drop, and unbounded `json-file` logging.

