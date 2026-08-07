---
title: 24/7 Paper Runtime Certification
type: playbook
tags:
  - paper-trading
  - reliability
  - chaos-engineering
  - sre
  - promotion-gates
created: 2026-07-28
updated: 2026-07-28
sources:
  - "[[production-remediation-2026-07-24]]"
  - "[[canonical-local-paper-runtime-v1]]"
status: active
---

# 24/7 Paper Runtime Certification

## Purpose

Certify the canonical local paper runtime with actual elapsed evidence. The
controller never shortens or simulates the required duration: it runs a
24-hour fault-injection phase, then a seven-day public-feed paper phase. It
cannot approve testnet, live, or mainnet execution.

## Authority boundary

- The only admitted Compose namespace is `trade_agent_canonical` or an
  explicitly suffixed canonical test namespace.
- Configuration preflight must admit `paper_live`; provider and broker secrets
  remain outside the canonical graph.
- The controller may stop/start canonical NATS and PostgreSQL and crash the
  canonical candidate worker at scheduled points. It never touches legacy
  containers, volumes, broker credentials, or external infrastructure.
- A failed or interrupted controller is resumable from local signed state.
  Changing Git HEAD or thresholds invalidates resume.

## Default sequence

1. Validate the canonical env and Compose graph.
2. Start or verify the canonical paper runtime and capture baseline database
   and container restart counters.
3. Sample readiness, dependency health, data-integrity counters, outbox/DLQ,
   and restart counts every 60 seconds.
4. During the 24-hour phase inject:
   - a bounded NATS outage at 10% elapsed;
   - a candidate-worker crash at 40% elapsed;
   - a bounded PostgreSQL outage at 70% elapsed.
5. Advance to the seven-day phase only when elapsed time, sample coverage,
   availability, recovery, duplicate, mismatch, DLQ, outbox, and restart gates
   pass.
6. Promote only after the seven-day core gates plus acknowledged alert
   delivery and fresh off-host restore evidence within RPO/RTO thresholds pass.

## Start

For a foreground operator session:

```powershell
uv run --frozen python scripts/certify_24x7.py run `
  --start-runtime `
  --project trade_agent_canonical `
  --env-file .local/canonical.env `
  --alert-webhook-url-file .local/certification/alert-webhook-url
```

For an eight-day run that resumes after a controller failure, Docker Desktop
delay, or user logon:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/install_certification_task.ps1
```

The scheduled task starts immediately, ignores overlapping invocations, retries
every five minutes for ten days, wakes the host when Windows permits, and keeps
running across AC/battery transitions. The controller also holds a Windows
system-required execution state to prevent automatic sleep while evidence is
being collected. Its own PID lock is a second overlap boundary.

The webhook file is optional at launch but mandatory for final promotion. It
must contain an approved HTTPS receiver URL or a loopback HTTP URL. The URL is
never written to evidence; only a destination hash, HTTP status, latency, and
response hash are retained.

## Observe and verify

```powershell
uv run --frozen python scripts/certify_24x7.py status
uv run --frozen python scripts/certify_24x7.py verify
```

Local state is under `.local/certification/`:

- `state.json` — resumable aggregate state;
- `samples.jsonl` — append-only sample/fault journal;
- `signed-status.json` — HMAC-SHA256 authenticated status;
- `certification-signing.key` — ignored local key;
- `controller.lock` — single-run process guard.

Only a fully promoted run replaces
`artifacts/24x7-certification-latest.json`. A pending or failed run cannot
overwrite the durable promotion artifact.

## Hard gates

| Gate | Default requirement |
|---|---:|
| Chaos duration | 24 actual hours |
| Paper duration | 7 actual days after chaos |
| Sample coverage | at least 90% |
| Readiness availability | at least 99.5% |
| Paper-execution availability | at least 99.5% |
| Duplicate inbox / fills | exactly 0 |
| Reconciliation discrepancies | exactly 0 |
| New DLQ events | exactly 0 |
| Final pending outbox | exactly 0 |
| Per-service restart delta | at most 3 |
| Fault recovery | every scheduled fault, at most 300 seconds |
| Alert delivery | acknowledged HTTP 2xx |
| RPO | at most 1 hour |
| RTO | at most 30 minutes |
| Restore evidence | off-host verified, after run start, at most 24 hours old |

## Failure recovery

- Controller process loss: rerun the exact command. The same Git commit and
  thresholds resume; a second concurrent process is rejected.
- Host restart: the scheduled task starts the same runner when its next
  five-minute trigger becomes available.
- Canonical dependency does not recover: the fault is recorded failed and the
  runtime is left in a recovered/start attempt state. Investigate before
  starting a new certification run.
- Git or threshold drift: preserve the old local evidence, choose a new state
  directory, and restart the full elapsed certification.
- Missing alert or restore evidence: the completed seven-day run remains
  `awaiting_external_evidence`; it is not promoted.

## Related

- [[production-remediation-2026-07-24]]
- [[canonical-local-paper-runtime-v1]]
- [[deterministic-paper-core-v1]]
- [[nats-event-system]]
