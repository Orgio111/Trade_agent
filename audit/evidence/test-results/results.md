# Test and Validation Results

Audit date: 2026-07-24  
All commands were non-destructive.

| Validation | Result | Interpretation |
|---|---:|---|
| `uv run --frozen pytest -q` | **FAIL** — 1 failed, 604 passed, 7 warnings | Migration contract expects only migrations 001–003; repository contains 001–007 |
| Pytest collection | 605 tests | 385 unit, 149 root tests, 50 contract, 18 integration, 3 replay |
| Canonical Ruff (`packages`, `workers`) | PASS | Current canonical code meets selected lint gate |
| Full-repository Ruff | FAIL — 533 findings | Legacy and experimental surfaces are not lint-clean |
| Canonical MyPy (`packages`, `workers`) | PASS | Selected canonical paths meet type gate |
| Legacy MyPy | FAIL — 477 errors / 87 files | Quarantined and experimental code has substantial type debt |
| Python AST parse | PASS — 357 files | No syntax parse failures |
| Frontend lint | PASS with 7 warnings | Warnings remain |
| Frontend TypeScript check | PASS | Current source type-checks |
| Frontend `npm audit` | PASS — 0 advisories | Dependency audit clean at audit time |
| Playwright discovery | 20 tests in 3 files | Test definitions exist |
| Playwright execution | TIMEOUT | No E2E pass claim is made |
| Go `go test -count=1 ./...` | PASS, no test files | Compilation signal only |
| Go `go vet ./...` | PASS | Static vet clean |
| Rust | UNVERIFIED | `cargo`/`rustc` unavailable |
| Knowledge validation | FAIL | 45 governed paths are unowned by `project.manifest.toml` |
| Prometheus `promtool check config` | FAIL | Referenced `recording_rules.yml` missing |

## Existing acceptance artifact

The committed `artifacts/acceptance-latest.json` reports a passing isolated
NATS-outage/recovery run dated 2026-07-23. It verifies readiness loss and recovery,
seven migrations, policy activation, worker readiness, and kill-switch behavior.
It records zero risk decisions, zero order intents, and zero fills. It therefore
does not prove a market-to-fill path.

## Coverage

No current coverage result was produced because the coverage tooling was not
available in the locked environment. Test quantity is not treated as coverage.

