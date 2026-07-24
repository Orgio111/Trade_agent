# Safe Validation Commands

These commands are intended for post-remediation validation. They are read-only or
test-only and must be run in paper/testnet mode. Do not paste secret values into
the shell history. Do not run live order endpoints.

## Repository and governance

```powershell
git status --short
git fsck --connectivity-only
uv run python -m packages.knowledge.cli validate
uv run python scripts/knowledge.py check
```

## Static and unit tests

```powershell
uv run --frozen pytest -q
uv run --frozen ruff check packages workers
uv run --frozen mypy packages workers
Push-Location frontend
npm run lint
npm run typecheck
npm audit
Pop-Location
Push-Location realtime
go test -count=1 ./...
go vet ./...
Pop-Location
Push-Location execution
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
Pop-Location
```

Run frontend build and Playwright in a clean disposable checkout to avoid modifying
user-owned generated files:

```powershell
npm run build
npx playwright test
```

## Compose and images

The required secrets must already be provided by an approved secret manager. These
commands print resolved configuration, so first confirm the output redaction policy.

```powershell
docker compose config --services
docker compose config --profiles
docker compose ps --all
docker compose images
docker compose logs --tail 200
docker system df
```

Check non-secret container controls:

```powershell
docker inspect quantex-market-producer --format '{{json .Config.User}}'
docker inspect quantex-market-producer --format '{{json .HostConfig.LogConfig}}'
docker inspect quantex-market-producer --format '{{json .HostConfig.SecurityOpt}}'
docker inspect quantex-market-producer --format '{{json .HostConfig.CapDrop}}'
```

## Health and application readiness

```powershell
curl.exe -fsS http://127.0.0.1:8001/health
curl.exe -fsS http://127.0.0.1:8001/api/v1/readiness
curl.exe -fsS http://127.0.0.1:8222/healthz
curl.exe -fsS "http://127.0.0.1:8222/jsz?streams=true&consumers=true"
curl.exe -fsS http://127.0.0.1:9090/-/ready
curl.exe -fsS http://127.0.0.1:3001/api/health
```

Do not treat a 200 as sufficient. Assert fresh worker leases, required schema
version, canonical subjects/consumers, recent validated market timestamps,
reconciliation state, and kill-switch/execution state in the response.

## Prometheus

```powershell
docker exec quantex-prometheus promtool check config /etc/prometheus/prometheus.yml
curl.exe -fsS http://127.0.0.1:9090/api/v1/targets
curl.exe -fsS http://127.0.0.1:9090/api/v1/rules
curl.exe -fsS http://127.0.0.1:9090/api/v1/alerts
```

## Data-store read checks

Use approved identities already injected into containers; never echo passwords.

```powershell
docker exec quantex-redis redis-cli PING
docker exec quantex-redis redis-cli DBSIZE
curl.exe -fsS http://127.0.0.1:6333/collections
curl.exe -fsS http://127.0.0.1:8100/api/v2/heartbeat
```

PostgreSQL validation queries:

```sql
SELECT version, checksum, applied_at FROM schema_migrations ORDER BY version;
SELECT rolname, rolsuper, rolcreatedb, rolcreaterole, rolreplication
FROM pg_roles WHERE rolcanlogin ORDER BY rolname;
SELECT COUNT(*) FROM risk_decisions;
SELECT COUNT(*) FROM order_intents;
SELECT COUNT(*) FROM fills;
SELECT COUNT(*) FROM outbox_events WHERE published_at IS NULL;
SELECT worker_name, expires_at > now() AS fresh FROM worker_leases ORDER BY worker_name;
```

## Security and dependency validation

```powershell
uv run --frozen --with pip-audit pip-audit
Push-Location frontend
npm audit
Pop-Location
rg -n ":latest|uses: .*@v[0-9]" docker-compose.yml deployment .github
```

Use the organization-approved secret scanner and image scanner in CI. Configure
them to report locations and advisory IDs while masking values.

## Paper-only behavioral gates

- Deterministic replay: duplicate, out-of-order, stale, gap, high-spread, kill
  switch, stale portfolio, policy missing, and broker-constraint failures.
- Supervised public feed: closed candle to validated event, feature, candidate,
  deterministic risk decision, paper intent, simulated fill, reconciliation.
- Fault injection: NATS unavailable, DB unavailable, model timeout, worker death,
  disk-pressure threshold. Every failure must disable execution and alert.
- Restore drill: restore into isolated infrastructure and compare event/order/fill
  invariants. Never restore over the active environment.

