# Production Readiness

## Weighted score

Scores evaluate the actual observed deployment while crediting source controls that
are demonstrably implemented and tested. They do not equate a design artifact with
runtime evidence.

| Category | Weight | Score | Weighted points | Evidence |
|---|---:|---:|---:|---|
| Architecture | 10 | 60/100 | 6.00 | Coherent canonical event/inbox/outbox/risk design; deployment drift |
| Code quality | 10 | 45/100 | 4.50 | Canonical gates pass; deployed legacy code has 533 Ruff/477 MyPy findings |
| Runtime correctness | 15 | 15/100 | 2.25 | Legacy stack running; no signals; market/model paths broken |
| Docker/infrastructure | 10 | 25/100 | 2.50 | Resource limits exist; root/writable/latest/public exposure and topology drift |
| Database/data integrity | 10 | 20/100 | 2.00 | Canonical schema absent; superuser runtime; no DR |
| Security | 15 | 20/100 | 3.00 | Critical unauth API and Chroma issue; secret/hardening gaps |
| Testing | 10 | 50/100 | 5.00 | 604 passes, but release gate red and cross-language/E2E gaps |
| Observability | 5 | 25/100 | 1.25 | Four targets up; invalid config and false-positive health |
| Reliability/recovery | 10 | 30/100 | 3.00 | NATS recovery artifact; no E2E, soak, backup, or deployed equivalence |
| Documentation/operations | 5 | 70/100 | 3.50 | Strong wiki/runbooks; environment contract and actual topology drift |
| **Total** | **100** |  | **33.00/100** | **Not production ready** |

## Hard gates

| Gate | Result | Blocking evidence |
|---|---|---|
| Single authoritative deployable architecture | Fail | Canonical source vs legacy runtime |
| Clean tests and governance | Fail | One pytest failure; 45 unowned paths |
| Real data lineage | Fail | Market feed broken; no NATS messages |
| Deterministic risk enforced at runtime | Unverified | Canonical worker/schema absent |
| Paper order/fill/reconciliation | Fail | Acceptance counts all zero |
| Authentication/authorization | Fail | Legacy mutations unauthenticated |
| Critical dependency policy | Fail | Chroma 1.5.9 advisory |
| Secrets least privilege | Fail | Broad shared writable environment |
| Monitoring and alerting | Fail | Invalid Prometheus config; alerts unverified |
| Backup/restore | Fail | No drill/RPO/RTO |
| 24-hour soak | Fail | Not performed |
| Immutable release provenance | Fail | Latest/mutable tags |

## Release stages

| Stage | Decision |
|---|---|
| Developer unit/replay work | Conditional; fix red gates first |
| Isolated canonical paper acceptance | Conditional after P0 config/security fixes |
| Supervised public-feed paper session | No, until market-to-fill acceptance passes |
| 24/7 autonomous paper | No |
| Testnet broker | No |
| Live/mainnet | No |

## Final verdict

```text
PRODUCTION READY: NO
24/7 AUTONOMOUS OPERATION: NO
REAL DATA VERIFIED: PARTIAL
SECURITY APPROVED: NO
```

The fastest credible route is not to repair every legacy container. It is to make
the canonical paper topology the only deployable profile, prove it end-to-end,
then delete/quarantine the old network surface.

