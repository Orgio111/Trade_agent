# Testing and Quality Audit

Test readiness score: **50/100**.

## Results

The locked Python suite produced **1 failed, 604 passed, 7 warnings**. Canonical
Ruff/MyPy and frontend lint/typecheck passed. npm audit was clean. Go compiled and
vetted but contained no tests. Rust could not be verified. Twenty Playwright tests
were discoverable, but execution timed out and no pass result is claimed.

## [AUD-012] Release gates are red and the audited branch can bypass push CI

- Severity: High
- Category: CI / governance
- Component: Migration contract, manifest ownership, GitHub Actions
- File: `tests/contract/test_migration_runner.py`, `project.manifest.toml`, `.github/workflows/ci.yml`
- Line: Migration list expectation; path ownership; branch filters
- Runtime service: CI
- Status: Confirmed
- Evidence: Pytest expects migrations 001–003 while 004–007 exist; knowledge validation reports 45 unowned governed paths; workflow push filters do not include the current `fix/**` branch pattern.
- Impact: A commit can appear locally production-focused while its contract/governance gates fail or do not run on push.
- Root cause: Recent runtime work added migrations/files without updating contractual expected state and CI branch policy.
- Reproduction: `uv run --frozen pytest -q`; run both knowledge validation commands; inspect workflow branch filters.
- Recommended fix: Update the migration contract to derive/verify all seven immutable checksums; assign owners/lifecycle to all 45 paths; require PR checks for every branch and protected release branch.
- Validation after fix: Pytest and knowledge checks pass from a clean clone; required CI checks run and are branch-protected.
- Estimated effort: S
- Priority: P0

## [AUD-022] Coverage, E2E, Go behavior, and Rust behavior are unproven

- Severity: Medium
- Category: Test completeness
- Component: Repository-wide
- File: `tests/`, `frontend/e2e/`, `realtime/`, `execution/`
- Line: N/A
- Runtime service: CI and release validation
- Status: Confirmed
- Evidence: No coverage tool/result in the locked environment; Playwright timed out; Go reports no test files; Rust toolchain is absent; CI does not run Go, Rust, E2E, Compose acceptance, security scans, or image policy.
- Impact: Concurrency, serialization, UI, broker, and cross-language failures can reach a release undetected.
- Root cause: Test count is broad in Python but release validation is not a multi-runtime test pyramid.
- Reproduction: Run the commands in [`17-validation-commands.md`](17-validation-commands.md).
- Recommended fix: Add coverage by risk, deterministic contract fixtures across Python/Go/Rust/TypeScript, hermetic E2E, and paper-only Compose acceptance with artifacts.
- Validation after fix: All language gates and E2E pass in CI; coverage thresholds protect critical risk/execution/reconciliation branches.
- Estimated effort: L
- Priority: P1

## [AUD-023] Frontend build/release artifact is not verified on the audited worktree

- Severity: Low
- Category: Frontend release quality
- Component: Next.js frontend
- File: `frontend/next-env.d.ts`, `frontend/package-lock.json`
- Line: Current user-owned modifications
- Runtime service: Frontend
- Status: Unverified
- Evidence: Lint/typecheck passed with seven warnings, but the build was not rerun because it could rewrite existing user-owned tracked generated files; E2E timed out.
- Impact: The exact worktree cannot be claimed as a reproducible frontend production build.
- Root cause: Generated framework metadata and platform lockfile changes are tracked/dirty.
- Reproduction: Inspect Git status and run build only in a clean disposable checkout.
- Recommended fix: Define generated-file ownership, stabilize lockfile platform behavior, and validate build/E2E in an isolated clean checkout.
- Validation after fix: Clean-clone build and E2E pass without modifying tracked files.
- Estimated effort: S
- Priority: P2

## Test quality interpretation

The Python suite provides valuable unit/contract coverage for the canonical core,
including inbox/outbox, leases, risk, market producer, reconciliation, and
acceptance logic. It is not evidence that the running legacy deployment works. The
single migration failure is a release blocker because schema order/checksum is a
safety contract.

