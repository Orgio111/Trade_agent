# Project Inventory

## Repository and Git

| Item | Observed state |
|---|---|
| Branch / commit | `fix/production-ready-paper-runtime` / `0139367` |
| Tracking | Local tracking ref matched; remote freshness not asserted because no fetch was performed |
| Tracked files | 591 |
| Python files | 357; all parsed by AST |
| Test/spec files | 80 |
| Current dirty files before audit | `frontend/next-env.d.ts`, `frontend/package-lock.json` (user-owned, preserved) |
| Submodules / LFS | None |
| Git integrity | Connectivity passed; dangling objects only |
| Packed Git size | Approximately 236.78 MB |

The remote default pointer resolves to a `claude/**` branch rather than `main`.
The audited branch is 51 commits ahead of the local `origin/main` ref. No claim is
made about server-side freshness.

## Component map

| Component | Language/runtime | Manifest status | Audit classification |
|---|---|---|---|
| `packages/events`, `packages/persistence`, `packages/risk`, `packages/execution` | Python | Active | Canonical production candidate |
| `workers/*` | Python | Active | Canonical production candidate |
| `orchestrator/control_plane.py` | Python/FastAPI | Active | Canonical read-only control plane |
| `orchestrator/`, legacy entry points | Python/FastAPI | Quarantined/legacy | Running stale topology |
| `services/*` | Python/FastAPI | Quarantined/experimental | Running legacy microservices |
| `frontend/` | TypeScript/Next.js | Experimental | Dashboard/UI |
| `realtime/` | Go | Experimental | NATS aggregation/WebSocket |
| `execution/` | Rust | Quarantined | Not toolchain-verified |
| `deployment/`, `monitoring/` | Compose/K8s/Terraform | Experimental | Drifted from canonical runtime |
| `wiki/` | Markdown/Obsidian | Active | Durable project knowledge |
| `inference/`, `agents/`, `ml/`, `rl/` | Python | Mixed | Model adapters/research/legacy |

The manifest describes 20 components. A conservative import-graph approximation
found 285 production Python files: 75 reachable from canonical seeds, 88 reachable
from legacy seeds, and 122 not reached from either. The latter are candidates for
research CLI, tooling, or dead/unwired code—not confirmed dead code.

## Entry points

- Canonical Compose: migration job, market producer, market/feature/candidate/
  decision/execution/reconciliation workers, and read-only control plane.
- Legacy Compose profiles: orchestrator, service APIs, Go realtime, frontend,
  dashboard, monitoring, inference, and databases.
- Frontend: Next.js application.
- Go: `realtime` service.
- Rust: execution crate, unverified in this environment.
- Operations: `scripts/apply_migrations.py`, `scripts/acceptance.py`,
  `scripts/replay_dlq.py`, knowledge CLI.

## Data and generated artifacts

- `models/ml_signal_model.pkl` is a tracked 3.09 MB serialized model.
- `memory/chroma.sqlite3` is a tracked 188 KB generated database under a manifest
  exception.
- Tracked SBOM/audit JSON artifacts include multi-megabyte generated output.
- Build artifacts such as `.venv`, `node_modules`, `__pycache__`, and Rust `target`
  are ignored rather than tracked.

## Governance status

The knowledge/manifest validation fails because 45 governed paths added by the
recent production-runtime work have no manifest owner. This includes acceptance
artifacts, audit artifacts, scripts, and new tests.

## Inventory risks

- Canonical and legacy implementations duplicate orchestration, risk, execution,
  and data paths.
- The deployed container set maps to legacy code, not the current manifest's
  production candidate.
- Machine-specific `.obsidian/workspace.json` remains tracked and has been changed
  historically despite repository guidance.

