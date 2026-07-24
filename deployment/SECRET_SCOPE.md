# Secret Scope Contract

The canonical Compose graph consumes exactly two operator-supplied secrets. Values
are sourced by Compose and mounted as read-only files; they are never copied into
tracked configuration.

| Secret file | Runtime recipients | Purpose |
|---|---|---|
| `POSTGRES_PASSWORD_FILE_PATH` | `postgres`, one-shot `migrate` | Database owner/bootstrap only |
| `DB_RUNTIME_PASSWORD_FILE_PATH` | `migrate`, control plane, canonical workers | Non-owner canonical runtime role |

Canonical paper services must not receive any `BINANCE_*`, `OPENAI_*`, `GROQ_*`,
`OPENROUTER_*`, `NVIDIA_*`, or `TELEGRAM_*` variable or secret.

Provider compatibility:

- `NVIDIA_NIM_API_KEY` is the only supported NIM name in quarantined research
  configuration.
- `NVIDIA_API_KEY` is a legacy alias. Do not define both. Migrate the consumer
  first, then remove the alias from the local legacy secret source.
- Provider credentials are not admitted by the canonical paper Compose graph.

Local operator sequence:

1. Run `uv run python scripts/init_local_runtime.py`. It creates the ignored
   `.local/canonical.env` exactly once and never overwrites credentials.
2. Generate independent owner/runtime database passwords; do not reuse provider or
   broker credentials. The initialization command does this with the operating
   system CSPRNG.
3. Restrict the file ACL to the owner, Administrators, and SYSTEM.
4. Run `uv run python scripts/config_preflight.py --env-file .local/canonical.env`.
5. Pass the file explicitly with
   `docker compose --env-file .local/canonical.env -p trade_agent_canonical up -d --build`.

The preflight reports variable names and rule failures only. It never emits values.
