# Codebase Audit

## Scope and method

All 591 tracked paths were inventoried. Python source was parsed, selected canonical
and full-repository lint/type checks were run, entry points were mapped, production
seeds were approximated through static imports, and mock/fallback/deserialization
patterns were searched. File-by-file classification is in
[`20-file-classification.md`](20-file-classification.md).

## Code health

| Surface | Result |
|---|---|
| Canonical `packages/` and `workers/` | Ruff pass; MyPy pass |
| All Python | 357/357 AST parse |
| Full repository Ruff | 533 findings |
| Legacy MyPy | 477 errors across 87/157 checked files |
| Broad exception handlers | 411 matches before qualitative filtering |
| Duplicate implementations | Risk, execution, data, orchestration in multiple layers |

Pattern search counts are signals, not automatic defects: `mock` 46 matches,
`synthetic` 38, `random` 313, `fallback` 254, `NotImplemented` 8, `TODO` 3, and
`pass` 126. For example, canonical random use includes retry jitter. These require
path-level classification before removal.

## [AUD-013] Legacy and experimental code fails repository-wide quality gates

- Severity: Medium
- Category: Code quality
- Component: Legacy orchestrator, services, agents, ML/RL paths
- File: `orchestrator/`, `services/`, `agents/`, `ml/`, `rl/`
- Line: Multiple
- Runtime service: Legacy 23-container stack
- Status: Confirmed
- Evidence: Full Ruff reported 533 findings; legacy MyPy reported 477 errors in 87 files. Canonical paths passed both.
- Impact: Defects can survive in the exact legacy code currently deployed, while CI's narrow canonical gate remains green.
- Root cause: A new canonical core was added without either removing or bringing the deployed legacy surface under equivalent gates.
- Reproduction: `uv run --frozen ruff check .`; run the repository's legacy MyPy target.
- Recommended fix: Define manifest-owned lint/type scopes for every deployable profile; quarantine or delete non-deployable code; make the deployed scope a required CI gate.
- Validation after fix: Full deployable profile passes Ruff/MyPy with no ignored undefined-name or type-safety defects.
- Estimated effort: L
- Priority: P1

## [AUD-018] Unsafe serialized model formats are tracked and loaded

- Severity: Medium
- Category: Supply chain / code execution
- Component: ML/RL model loading
- File: `models/ml_signal_model.pkl`, `services/rl/`, `ml/`, `agents/`
- Line: Multiple `pickle`, `joblib`, and `torch.load` call sites
- Runtime service: RL, ML signal, FreqAI, FinRL paths
- Status: Confirmed
- Evidence: A 3.09 MB pickle model is tracked; source contains pickle/joblib/torch deserialization; runtime RL restore already fails compatibility checks.
- Impact: A tampered model artifact can execute code at load time or fail nondeterministically across library versions.
- Root cause: Model provenance, signature verification, safe format policy, and compatibility metadata are not enforced centrally.
- Reproduction: `rg -n "pickle\\.load|joblib\\.load|torch\\.load" --glob "*.py"`.
- Recommended fix: Prefer `safetensors` or schema-bound numeric formats; require artifact digest/signature and model-card compatibility; reject untrusted pickle/joblib files.
- Validation after fix: Model loaders refuse altered digests and no production entry point loads executable serialization.
- Estimated effort: M
- Priority: P1

## [AUD-024] Large unwired surface and duplicate implementation layers remain

- Severity: Low
- Category: Maintainability / architecture drift
- Component: Repository-wide Python
- File: Multiple
- Line: N/A
- Runtime service: Multiple
- Status: Probable
- Evidence: Conservative seed analysis found 75 canonical-reachable, 88 legacy-reachable, and 122 unclassified/unreached production Python files with no overlap between canonical and legacy seeds.
- Impact: Ownership is unclear, dead paths accumulate, and a deploy can select the wrong implementation.
- Root cause: Migration-by-addition without a completed decommission plan and without exhaustive manifest ownership.
- Reproduction: Re-run the static import/entry-point inventory and compare with `project.manifest.toml`.
- Recommended fix: Assign every file an owner and lifecycle state; delete proven dead code; convert research tools into explicit CLI packages; block unowned deployable paths.
- Validation after fix: Knowledge validation passes and every deployable entry point maps only to active components.
- Estimated effort: L
- Priority: P2

## [AUD-025] Generated databases and large audit artifacts increase repository risk

- Severity: Low
- Category: Repository hygiene
- Component: Models, memory, audit artifacts
- File: `memory/chroma.sqlite3`, `models/ml_signal_model.pkl`, `artifacts/*.json`
- Line: N/A
- Runtime service: Knowledge/model workflows
- Status: Confirmed
- Evidence: Generated SQLite/model/SBOM artifacts are tracked and Git pack size is approximately 236.78 MB.
- Impact: Review noise, accidental data retention, slow clones, and artifact provenance ambiguity.
- Root cause: Generated-state exceptions were added instead of using an artifact registry and reproducible build metadata.
- Reproduction: `git ls-files` plus size inventory; `git count-objects -vH`.
- Recommended fix: Move models/SBOMs/runtime databases to signed release artifacts or object storage; retain only checksums and generation recipes.
- Validation after fix: Repository policy rejects generated runtime databases and large binary models unless explicitly release-managed.
- Estimated effort: M
- Priority: P2

## Call-graph conclusion

The canonical pipeline is internally coherent and type/lint clean, but the
repository does not yet enforce a single authoritative deployable graph. No file
was deleted or reclassified during this audit.

