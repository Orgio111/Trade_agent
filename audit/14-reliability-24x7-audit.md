# 24/7 Reliability Audit

Reliability score: **30/100**.  
`24/7 AUTONOMOUS OPERATION: NO`.

## Positive design evidence

- Canonical inbox/outbox, durable consumers, deduplication, worker leases, DLQ,
  replay tooling, deterministic risk, reconciliation, and kill-switch state.
- Committed isolated acceptance artifact shows NATS outage makes the control plane
  NotReady and disables execution, followed by recovery after dependency return.
- Paper execution is the only canonical execution implementation.

## Blocking evidence

- The recovery-tested topology is not the running topology.
- Acceptance produced no risk decision, order intent, or fill.
- Real market ingestion, local inference, legacy NATS connectivity, and RL restore
  are broken.
- Prometheus config is invalid; alert delivery is unverified.
- Container logs are unbounded.
- PostgreSQL backup/restore and NATS disaster recovery are absent.
- No long soak, clock-skew, network partition, disk-full, corrupt-message,
  duplicate-event, slow-consumer, or DB failover artifact exists.
- All canonical workers share one broad DB runtime role rather than per-service
  privileges.

## Failure-mode matrix

| Failure | Current behavior/evidence | Required safe behavior |
|---|---|---|
| NATS outage | Canonical artifact: NotReady then recovery | Preserve inbox/outbox invariants; no execution |
| Market feed 404 | Infinite-looking retry storm; shallow health | NotReady after freshness budget; bounded circuit |
| vLLM OOM/admission | Restart loop; aggregate health green | Candidate unavailable/fallback explicit; alert |
| DB schema drift | Legacy DB accepted by legacy services | Canonical workers refuse start |
| Worker death | Canonical leases designed | Alert and execution disabled before lease TTL |
| Duplicate event | Canonical inbox designed | Exactly one domain effect and auditable duplicate |
| Reconciliation mismatch | Canonical kill-switch design | Fail closed; operator review |
| Disk full | No verified control | Alert before limit; preserve DB/NATS; bounded logs |
| Host loss | No restore proof | Restore within RPO/RTO |
| Credential compromise | Broad shared secrets | Scoped revocation with minimal blast radius |

## Paper autonomy criteria

Before 24/7 paper operation:

1. Canonical-only deployment from immutable images.
2. Green migration/manifest/test/security/config gates.
3. Real public-feed lineage through reconciled paper fills.
4. Explicit supervised session enablement and tested manual kill switch.
5. 24-hour fault-injection soak and seven-day paper soak.
6. Restore drill and on-call runbook.
7. SLO dashboards/alerts for freshness, lag, rejection, fill, reconciliation,
   restart, resource, and security events.

Live/mainnet operation remains out of scope and must not inherit paper approval.

