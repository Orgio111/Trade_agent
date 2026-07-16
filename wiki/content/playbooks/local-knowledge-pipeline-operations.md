---
title: Local Knowledge Pipeline Operations
type: playbook
tags:
  - knowledge-management
  - operations
  - manifest
  - chromadb
  - ollama
  - embeddings
  - ci
created: 2026-07-15
updated: 2026-07-16
sources:
  - "[[local-knowledge-pipeline-v1]]"
status: stable
---

# Local Knowledge Pipeline Operations

## Purpose

Operate the fully local code-documentation-ownership-embedding pipeline defined by [[local-knowledge-pipeline-v1]]. The workflow keeps the Obsidian wiki authoritative, makes the vector index rebuildable, and detects repository drift without requiring a model or database in CI.

Run every command from the repository root with the project virtual environment active.

## Authoritative artifacts

| Artifact | Role | Committed |
|---|---|:---:|
| `project.manifest.toml` | Human-authored ownership, responsibility, source, chunking, and local-service contract | Yes |
| `project.manifest.lock.json` | Generated deterministic ownership/content evidence plus embedding-source chunk inventory checked by CI | Yes |
| `.local/knowledge/sync-receipt.json` | Machine-local proof of the last successful Chroma synchronization | No |
| Obsidian `wiki/` | Permanent project knowledge and architecture source of truth | Yes |
| ChromaDB collection | Rebuildable semantic-retrieval projection | No |

## Prerequisites

- Python dependencies are installed from the project configuration.
- The pinned `chromadb/chroma:1.5.9` Docker service is running at `127.0.0.1:8100` for live operations; container port 8000 is not exposed beyond loopback.
- Ollama is running on the configured loopback endpoint.
- `nomic-embed-text` is installed locally: `ollama pull nomic-embed-text`.
- The manifest and lock are reviewed like source code.

No cloud API key is required or permitted for this pipeline.

## Command summary

| Command | Runtime need | Mutates state | Use |
|---|---|---:|---|
| `python scripts/knowledge.py lock` | None | `project.manifest.lock.json` | Rebuild governed-file hashes and the expected embedding-source/chunk inventory |
| `python scripts/knowledge.py check` | None | No | Run the same fail-closed drift checks used by CI |
| `python scripts/knowledge.py sync` | Local Ollama + ChromaDB | Chroma collection and local receipt | Embed and reconcile the live local index |
| `python scripts/knowledge.py verify` | Local Ollama + ChromaDB | No | Compare the live collection, model identity, and receipt with the current lock |
| `python scripts/knowledge.py query "<question>"` | Local Ollama + ChromaDB | No | Retrieve verified project knowledge with source metadata |

Use `python scripts/knowledge.py <command> --help` for command-specific options. Paths and service endpoints come from `project.manifest.toml`, never hard-coded user directories.

## Standard change workflow

### 1. Update ownership and documentation

When adding or moving production code:

1. Assign it to exactly one manifest component.
2. Keep that component's primary responsibility singular and concrete.
3. Add or update the corresponding durable wiki page.
4. Add only permanent code/documentation/research inputs to the embedding set.
5. Never select `wiki/raw/`, `.env`, generated output, model files, caches, logs, or local database directories.

The tracked-file policy discovers relevant code, configuration, and durable documentation even under a previously unknown root. A new path such as `new_service/app.py` therefore fails CI until it has one explicit owner. Add a `tracked_exclude` rule only for a narrow generated, raw, machine-local, or documented legacy path; never use it to hide production ownership drift.

### 2. Rebuild the deterministic lock

```powershell
python scripts/knowledge.py lock
python scripts/knowledge.py check
```

Review the lock diff. Every governed owned text file appears under `governed_files`, even when it is not an embedding input; quarantined paths remain here while staying absent from `sources`. Expected changes should identify only the owner/component hashes, source hashes, and chunks affected by the change. The configured generated lock is the sole self-reference exception because `check` compares its complete canonical content. A large unrelated diff indicates an ownership glob, normalization, or chunk-policy error.

`lock` is offline and does not prove that embeddings were generated.

### 3. Synchronize the local index

Start the pinned local ChromaDB Docker service, then verify Ollama and synchronize:

```powershell
docker compose up -d chroma
ollama list
python scripts/knowledge.py sync
```

`sync` uses local Ollama `nomic-embed-text`, upserts changed chunks, preserves unchanged chunks, removes only stale records owned by this manifest/collection, and writes `.local/knowledge/sync-receipt.json` only after success.

Before it writes, `sync` acquires one machine-global lease for the configured local Chroma endpoint and collection. The lease is stored under the host temporary directory and is shared by clones and worktrees on that machine. If another writer already holds it, the second `sync` fails closed; do not bypass the lease or run concurrent writers against the same collection.

### 4. Verify live state

```powershell
python scripts/knowledge.py verify
```

A pass requires the live collection and local receipt to match the lock's project, collection/schema version, embedding model, content identities, and chunk inventory. Run this after every `sync`, model update, collection restore, or Chroma container/volume move.

`verify` is strictly read-only: it opens the configured collection without creating or modifying it and fails closed when that collection is absent. Only `sync` may initialize the manifest-owned collection.

### 5. Test retrieval

```powershell
python scripts/knowledge.py query "Where is paper execution ownership documented?"
```

Each result should expose the normalized source path, component ID, owner, source hash, and chunk identity. Treat results as engineering context, not as trading instructions. Query must refuse a drifting or unverified collection.

## CI contract

CI runs:

```powershell
python scripts/knowledge.py check
```

It deliberately does not start Ollama or ChromaDB and does not read the machine-local receipt. The job fails on manifest schema errors, unowned/duplicate ownership, missing documentation, prohibited inputs, governed-file drift, or embedding source/chunk/lock drift. This makes the check deterministic and offline, but it verifies expected state only. Local `verify` supplies the separate live-index guarantee.

Do not weaken the CI command with `continue-on-error`, shell fallbacks, warning-only modes, or automatic lock regeneration.

## Security exclusions

The following must never enter the embedding input set:

- `.env`, `.env.*`, credential files, private keys, tokens, cookies, or local shell history;
- `wiki/raw/` immutable source drops until they are synthesized into approved durable wiki pages;
- `.git/`, `.obsidian/`, editor state, caches, logs, coverage, build output, virtual environments, `node_modules/`, Rust targets, and Python bytecode;
- `.local/knowledge/`, Chroma persistence volumes, model weights, datasets, screenshots, archives, binaries, and temporary files;
- files reached through a symlink that resolves outside the repository;
- any path or file type rejected by the executable repository-path and text-type admission rules.

If a high-confidence secret detector fires, remove and rotate the secret before refreshing the lock. Do not add a broad exclusion merely to make CI green.

YAML, TOML, and env-style configuration must reference injected secrets. Plain `${VAR}`, required `${VAR:?message}` / `${VAR?message}`, GitHub Actions secret expressions, and equivalent secret-manager references are accepted. Literal values and default-bearing `${VAR:-literal}` expressions are rejected so placeholder credentials cannot enter the lock or vector store.

## Failure handling

### `check` reports content or lock drift

1. Inspect the named path and owning component.
2. Confirm the code change has matching durable documentation.
3. Correct accidental ownership overlap or unsafe glob expansion.
4. Run `lock`, review the diff, then rerun `check`.
5. Run `sync` and `verify` before relying on local retrieval.

### `check` reports an unowned or multiply owned file

- Assign the file to one existing component if its responsibility matches.
- Otherwise create a component with one primary responsibility and a durable architecture/operations page.
- Narrow overlapping globs; never resolve ambiguity by allowing multiple primary owners.

### `sync` cannot reach Ollama

```powershell
ollama list
curl http://127.0.0.1:11434/api/tags
```

Start Ollama, confirm `nomic-embed-text` exists, and retry. Do not substitute a cloud model, zero vector, random vector, or deterministic-hash pseudo-embedding.

### `sync` cannot reach ChromaDB

- Confirm the `chroma` service uses `chromadb/chroma:1.5.9`, is healthy, and maps `127.0.0.1:8100` to container port 8000.
- Confirm the client/server versions and collection name match `project.manifest.toml`.
- Do not point the pipeline at a remote or shared Chroma service as a shortcut.

### `sync` reports dimension or model mismatch

- Stop and verify the manifest model and collection version.
- Do not mix vectors from different models in one collection.
- If the model/chunk policy intentionally changed, review a collection-version bump, rebuild via `sync`, then run `verify`.

### `sync` fails partway through

- The command must not write a success receipt for the failed run.
- Rerun `sync`; stable chunk IDs make upserts idempotent.
- Run `verify` to identify missing, stale, or mismatched records.
- Never delete unrelated ChromaDB collections as a recovery shortcut.

### `sync` reports another writer is running

- Let the active `sync` finish, then rerun `check`, `sync`, and `verify`.
- If no process is active, confirm that endpoint and collection settings are correct before investigating the host temporary-directory lease.
- Never bypass the collection-scoped lease to run two writers against one local collection.

### `verify` reports stale or missing chunks

Run `check` first. If repository and lock are valid, rerun `sync` and then `verify`. If drift persists, inspect the Chroma endpoint/volume, collection version, embedding-model metadata, and manifest scope before rebuilding the manifest-owned collection.

### `verify` reports a missing or stale receipt

Run `sync` against the intended local Chroma instance. Never copy another machine's receipt or commit `.local/knowledge/sync-receipt.json`.

### `query` refuses to run

Run `check`, `sync`, and `verify` in that order. A refusal is the intended fail-closed behavior when retrieval provenance cannot be established.

## Backup and rebuild

The checked-in manifest, lock, source code, and wiki are the recovery assets. ChromaDB and the local receipt are disposable:

1. Restore the repository at the desired revision.
2. Run `check`.
3. Start pinned ChromaDB 1.5.9 and local Ollama with `nomic-embed-text`.
4. Run `sync` into the manifest-scoped collection.
5. Run `verify` and one source-attributed `query` smoke test.

Do not treat a copied Chroma volume or receipt as authoritative without verification.

## Release checklist

- [ ] Every changed production path has exactly one component owner.
- [ ] Primary responsibility and durable documentation agree.
- [ ] No prohibited or temporary inputs are selected.
- [ ] `project.manifest.lock.json` diff is reviewed and scoped.
- [ ] `check` passes offline.
- [ ] `sync` completes with local `nomic-embed-text` and ChromaDB 1.5.9.
- [ ] `verify` proves the live collection and local receipt match the lock.
- [ ] A source-attributed `query` smoke test passes.
- [ ] Changelog, architecture decision, index, and wiki log are synchronized.

## Related

- [[local-knowledge-pipeline-v1]]
- [[trade-project-full-integration-build-plan]]
- [[rag-agent]]
- [[deterministic-paper-core-v1]]
