---
title: Alpha Certification Pipeline
type: playbook
tags:
  - alpha-research
  - backtesting
  - walk-forward
  - model-risk
  - paper-trading
  - promotion-gates
created: 2026-07-30
updated: 2026-07-30
sources:
  - "[[deterministic-paper-core-v1]]"
  - "[[24x7-paper-certification]]"
  - "[[production-remediation-2026-07-24]]"
status: active
---

# Alpha Certification Pipeline

## Purpose

Keep research models outside the running model surface until two independent
certifications pass:

1. alpha certification proves immutable data, purged multi-regime OOS behavior,
   conservative canonical replay, cost stress, and integrity;
2. reliability certification plus model-specific paper shadow proves the exact
   artifact in the canonical runtime.

This pipeline cannot create live authority. Candidate workers still admit only
`replay` and `paper_live`, while deterministic risk and paper execution remain
authoritative under [[deterministic-paper-core-v1]].

## Promotion sequence

```text
immutable data
  -> purged/embargoed walk-forward OOS
  -> realistic-cost next-bar canonical replay
  -> signed alpha model card + immutable staged artifact
  -> promoted [[24x7-paper-certification]] evidence
  -> seven-day exact-artifact paper shadow
  -> signed promotion report + immutable model registry
```

No step may overwrite a running artifact. Offline training writes only into a
temporary candidate directory. A failed candidate is deleted with that
directory; an alpha-passing candidate is copied by digest into `registry/staged`
but not `registry/models`.

## Immutable historical data

`download` reads fixed-interval Binance klines into canonical JSONL:

- request windows are UTC, interval-aligned, and half-open `[start, end)`;
- pagination remains bounded and continues through short pages;
- duplicate rows must be byte-equivalent after normalization;
- off-grid timestamps, malformed OHLCV, and interval-close drift fail closed;
- `reject`, `record`, and exact `allowlist` gap policies are explicit;
- an interval-close anomaly can be excluded only when that exact candle open is
  allowlisted; it is then sealed as a gap instead of being normalized silently;
- data and manifest names include the content digest and are create-only;
- loading re-verifies the data hash, row count, chronology, exact grid, gaps,
  and non-symlink data path.

Example:

```powershell
uv run --frozen python scripts/certify_alpha.py download `
  --symbol BTCUSDT --interval 1h `
  --start 2022-01-01T00:00:00Z --end 2026-01-01T00:00:00Z
```

## OOS and replay contract

- Walk-forward folds are chronological and use explicit purge and embargo
  durations. Purge must cover at least the prediction horizon.
- The default uses a bounded rolling training window and binary-searched fold
  boundaries. Expanding history is opt-in with `--expanding` because repeated
  fitting otherwise becomes superlinear as the dataset grows.
- Every fold trains a fresh temporary artifact and evaluates only its forward
  OOS window.
- Before each OOS window, the canonical feature engine is warmed with at most
  200 already-closed candles preceding the test boundary. Warm-up candles
  cannot produce a signal, risk decision, order, fill, metric, or training
  label; they only reconstruct the feature state that live runtime would have
  observed.
- Causal canonical feature snapshots label trend/volatility regimes without
  future observations. Promotion requires at least three OOS regimes overall
  and at least two in every fold by default.
- Replay reuses the canonical incremental feature engine, validated candidate
  envelope, and deterministic `RiskEngine`.
- A signal created at candle close can fill only at the next candle open.
  Adverse gaps that exceed the approved stop-risk amount are rejected.
- Spread, slippage, taker fees, stop, target, and time exits are explicit.
  When stop and target are both touched in one candle, stop wins.
- Every fold is replayed again with fee, spread, and slippage multiplied by
  `1.5` by default. The stressed replay must independently pass the same
  profitability, baseline, stability, drawdown, and integrity gates.
- Duplicate signal/fill identities, reconciliation mismatch, and next-bar
  violations are release-blocking.

## Offline gates

| Gate | Default |
|---|---:|
| Valid folds | at least 3 |
| OOS trades | at least 30 per fold |
| Profit factor | at least 1.10 in every fold |
| Maximum drawdown | at most 15% in every fold |
| Baseline excess return | at least 0 in every fold |
| Positive-fold ratio | at least 80% |
| Worst fold loss | no worse than -2% |
| OOS regime coverage | at least 3 overall / 2 per fold |
| Dataset gaps | zero or exact explicit allowlist |
| Cost stress | every gate passes at 1.5× costs |
| Duplicate signals/fills | exactly 0 |
| Reconciliation mismatch | exactly 0 |
| Next-bar violation | exactly 0 |

Run:

```powershell
uv run --frozen python scripts/certify_alpha.py certify `
  --dataset-manifest <btc-manifest> `
  --dataset-manifest <eth-manifest>
```

The signed model card binds Git commit, code digest, dataset manifests and
content hashes, project manifest-lock digest, feature schema, split geometry,
per-fold artifact and trace digests, feature warm-up boundaries, regime counts,
OOS metrics, thresholds, and gate decisions. A reliability report is acceptable
only when its signed Git commit and source-lock digest match this card.

## Historical snapshot evidence

The downloader sealed a real half-open `2022-01-01T00:00:00Z` through
`2026-01-01T00:00:00Z` hourly snapshot for both symbols:

| Symbol | Canonical rows | Data SHA-256 |
|---|---:|---|
| BTCUSDT | 35,062 | `cfba38961a401633a4546662a8150655dc833821b9197596b4de1f2a492a6079` |
| ETHUSDT | 35,062 | `e6e061d7c91ce262dd0fd3b23ef458aa0f7d053e2d8021bef9bb0f5293b2c334` |

Both snapshots require the same exact Binance maintenance allowlist:
`2023-03-24T12:00:00Z` is a truncated noncanonical candle and
`2023-03-24T13:00:00Z` is absent. The earlier research RF audit failed alpha
promotion: three of four OOS folds lost heavily after costs, balanced accuracy
was near random, and the only positive ETH fold was unstable. These immutable
snapshots are therefore test inputs, not evidence that a model is deployable.

## Exact-artifact paper shadow

Shadow activation is explicit. Configure the canonical ignored env with the
staged digest:

```dotenv
CANDIDATE_PROVIDER=alpha_shadow
ALPHA_CANDIDATE_DIRECTORY=<host registry/staged/<sha256> directory>
ALPHA_CANDIDATE_ARTIFACT=/run/alpha-candidate/candidate.joblib
ALPHA_CANDIDATE_SHA256=<sha256>
```

The staged directory is mounted read-only. Worker startup verifies the file
hash before deserialization; every canonical candidate/risk envelope preserves
provider `alpha_shadow`, model identity, and artifact digest. The shared
preflight rejects incomplete bindings or artifact settings attached to another
provider.

Start a fresh canonical paper runtime built from the same code digest, then:

```powershell
uv run --frozen python scripts/certify_alpha.py shadow-run `
  --model-card <signed-alpha-model-card> `
  --reliability-document artifacts/24x7-certification-latest.json `
  --reliability-key <reliability-signing-key> `
  --env-file .local/canonical.env
```

The resumable controller samples readiness, paper execution, container image
IDs, the artifact hash inside `candidate-worker`, candidate-event model
digests, fills, duplicates, reconciliation, and portfolio drawdown. Defaults
require seven actual days, at least 90% sample coverage, at least 99.5%
availability, exact artifact/provider continuity, at least one candidate and
fill, drawdown at most 15%, and zero integrity mismatch.

## Signed promotion

Three signed documents remain separate:

- reliability report: exact canonical image/source runtime behavior;
- alpha model card: immutable data and OOS replay evidence;
- shadow report: the exact staged artifact under paper runtime.

The shadow report binds the hashes of the first two documents. Final promotion
does not retrain:

```powershell
uv run --frozen python scripts/certify_alpha.py promote `
  --model-card <signed-alpha-model-card> `
  --reliability-document <signed-reliability-report> `
  --reliability-key <reliability-signing-key> `
  --shadow-document <signed-shadow-report> `
  --shadow-key <shadow-signing-key>
```

Only then is the exact staged file copied by digest into `registry/models`.
Signature, evidence, code, artifact, elapsed, or integrity mismatch fails
closed. This is alpha promotion inside the paper/replay boundary, not live
approval.

## Current release classification

The machinery is implemented, but no trainable artifact is currently approved.
The active reliability run predates this alpha code, so it cannot certify the
new image/source state. After that run completes, this version requires a fresh
[[24x7-paper-certification]] followed by a fresh seven-day alpha shadow.
Python validation currently passes 679 tests. Knowledge manifest validation is
valid, while lock regeneration is intentionally deferred until the pre-alpha
soak finishes so its original source-lock evidence is not mutated mid-run.

## Scaling and next upgrade

Feature state is bounded and incremental; binary-searched splits plus the
default fixed rolling window avoid the previous quadratic full-history scan and
fit pattern. Explicit expanding mode still repeats model fitting as history
grows. The next scaling boundary is a content-addressed feature/fold cache and
incremental estimator contract that preserves identical split and artifact
evidence. Public verification should also migrate HMAC evidence to asymmetric,
KMS-backed signatures and off-host immutable artifact storage.

## Related

- [[deterministic-paper-core-v1]]
- [[24x7-paper-certification]]
- [[production-remediation-2026-07-24]]
- [[canonical-local-paper-runtime-v1]]
