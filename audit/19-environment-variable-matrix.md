# Environment Variable Matrix

No values are shown. `Present` is only `Yes`, `No`, `Empty`, or `Placeholder`.
“Used in” is based on source/Compose references and observed container mapping, not
proof that an external credential is valid.

| Variable | Defined in | Used in | Required | Present | Default | Secret | Runtime status | Issue |
|---|---|---|---:|---|---|---:|---|---|
| DB_RUNTIME_PASSWORD | Compose only | migrate + canonical workers/control plane | Yes | No | None | Yes | Blocks config render | Missing from example/docs |
| POSTGRES_PASSWORD | `.env`, example, Compose | PostgreSQL/migrate | Yes | Yes | None | Yes | Legacy DB running | Shared secret scope |
| POSTGRES_USER | Compose | PostgreSQL/migrate | No | No | `quantex` | No | Default used | Runtime role drift |
| POSTGRES_DB | Compose | PostgreSQL/migrate | No | No | `quantex` | No | Default used | Two DBs observed |
| POSTGRES_PORT | Compose | Host publish | No | No | `5432` | No | Public all interfaces | Should be loopback/private |
| DATABASE_URL | `.env`, example, services | Python DB clients | Contextual | Placeholder | None | Yes | Not valid in local file | Competing with composed URL |
| BINANCE_API_KEY | `.env`, example | Legacy broker/execution | No for paper | Yes | None | Yes | Injected to legacy path | Must be absent from paper |
| BINANCE_SECRET_KEY | `.env`, example | Legacy broker/execution | No for paper | Yes | None | Yes | Injected to legacy path | Must be absent from paper |
| BINANCE_TESTNET | `.env`, example | Broker adapters | Yes for broker tests | Yes | Code-specific | No | Testnet indicated | Does not make exposed API safe |
| BINANCE_REST_BASE | `.env` | Legacy broker | No | Yes | Provider default | No | Integration errors | Endpoint drift possible |
| BINANCE_WS_BASE | `.env` | Market feed | Yes for live feed | Yes | Provider default | No | HTTP 404 loop | Broken runtime URL/protocol |
| TRADE_SYMBOLS | `.env`, example | Producers/strategies | Yes | Yes | Project-specific | No | Configured | Actual downstream empty |
| TRADE_TIMEFRAMES | `.env` | Market/features | No | Yes | Project-specific | No | Configured | Runtime path broken |
| PAPER_TRADING | `.env`, example | Execution mode | Yes | Yes | Safety default varies | No | Duplicated | Duplicate definitions |
| RUNTIME_MODE | Compose | Canonical workers | No | No | `paper` | No | Source default | Must fail closed outside paper |
| INITIAL_BALANCE | `.env`, Compose | Paper account | No | Yes | `10000` in Compose | No | Inconsistent services | 10k vs 100k observed |
| PAPER_ACCOUNT_ID | Compose | Canonical paper execution | No | No | `paper-primary` | No | Canonical absent | Must be stable/idempotent |
| ACCOUNT_EQUITY_USDT | `.env`, example | Risk sizing | Contextual | Yes | Project-specific | No | Duplicated | Conflicting source of truth |
| RISK_PCT_PER_TRADE | `.env`, example | Risk sizing | Contextual | Yes | Project-specific | No | Duplicated | Canonical policy should dominate |
| MAX_NOTIONAL_PCT | `.env` | Legacy risk | No | Yes | Code-specific | No | Unverified | Duplicate policy layer |
| MAX_OPEN_POSITIONS | `.env` | Risk | No | Yes | Code-specific | No | Unverified | Canonical DB policy differs |
| MAX_CONSECUTIVE_LOSSES | `.env` | Risk | No | Yes | Code-specific | No | Unverified | Canonical DB policy differs |
| ATR_PERIOD | `.env` | Feature/strategy | No | Yes | Code-specific | No | Unverified | Feature version not runtime-proven |
| SL_ATR_MULT | `.env` | Strategy/risk | No | Yes | Code-specific | No | Unverified | Exit logic not proven |
| TP_ATR_MULT | `.env` | Strategy/risk | No | Yes | Code-specific | No | Unverified | Exit logic not proven |
| NATS_URL | `.env`, example | Event clients | Yes | Yes | Service-specific | No | Legacy orchestrator disconnected | Host/container URL drift |
| NATS_STREAM_SIGNALS | `.env` | Legacy stream | No | Yes | Legacy | No | Stream exists, empty | Not canonical stream |
| NATS_SUBJECT_RAW | `.env`, example | Legacy producers | No | Yes | Legacy | No | No messages | Subject model superseded |
| NATS_SUBJECT_AGGREGATED | `.env`, example | Go/orchestrator | No | Yes | Legacy | No | No messages | Subject model superseded |
| NATS_SUBJECT_EXECUTED | `.env`, example | Execution | No | Yes | Legacy | No | No messages | Subject model superseded |
| NATS_PORT | Compose | Host publish | No | No | `4222` | No | Public all interfaces | Should be private/loopback |
| NATS_MONITOR_PORT | Compose | Monitoring | No | No | `8222` | No | Public all interfaces | Monitoring unauthenticated |
| REDIS_PORT | Compose | Host publish | No | No | `6379` | No | Public all interfaces | Empty but exposed |
| CONTROL_PLANE_PORT | Compose | Canonical API | No | No | `8001` | No | Legacy API on port | Identity/topology ambiguity |
| NVIDIA_API_KEY | `.env` | Legacy cloud inference | No | Yes | None | Yes | Present | Example uses different name |
| NVIDIA_NIM_API_KEY | `.env.example` | NIM adapter | No | Placeholder | None | Yes | Name mismatch | Contract drift |
| OPENROUTER_API_KEY | `.env`, example | Cloud inference | No | Yes | None | Yes | Present, not validated | Over-broad service scope |
| GROQ_API_KEY | `.env` | Cloud inference | No | Yes | None | Yes | Present, not validated | Missing from example |
| OPENAI_API_KEY | `.env`, example | Cloud inference | No | Yes | Empty in example | Yes | Present, not validated | Over-broad service scope |
| INFERENCE_BUDGET_TIER | `.env`, Compose | Inference routing | No | Yes | `free` | No | Configured | Cost SLO unverified |
| OLLAMA_BASE_URL | `.env`, example | Canonical/legacy local LLM | Yes for candidate | Yes | Local service | No | Canonical absent | Runtime provider unverified |
| OLLAMA_REGIME_MODEL | `.env` | Legacy model selection | No | Yes | Code-specific | No | Unverified | No model registry |
| OLLAMA_SENTIMENT_MODEL | `.env` | Legacy model selection | No | Yes | Code-specific | No | Unverified | No model registry |
| CLOUD_LATENCY_BUDGET_MS | `.env` | Provider router | No | Yes | Code-specific | No | Unverified | No measured SLO |
| TELEGRAM_BOT_TOKEN | `.env`, example | Alerts | No | Yes | None | Yes | Delivery unverified | Shared environment |
| TELEGRAM_CHAT_ID | `.env`, example | Alerts | No | Yes | None | Sensitive | Delivery unverified | Shared environment |
| GRAFANA_ADMIN_PASSWORD | `.env`, example | Grafana | Yes | Yes | None | Yes | Grafana running | Must use managed secret |
| CERTBOT_DOMAIN | Compose | Certbot profile | Profile-only | No | None | No | Profile inactive/unverified | TLS automation unverified |
| CERTBOT_EMAIL | Compose | Certbot profile | Profile-only | No | None | Sensitive | Profile inactive/unverified | TLS automation unverified |
| DMS_HEARTBEAT_TIMEOUT_SECS | `.env` | Legacy distributed state | No | Yes | Code-specific | No | Unverified | Missing from example |
| DMS_TIMEOUT_SECS | `.env`, example | Legacy distributed state | No | Yes | Code-specific | No | Configured | Overlapping timeout knobs |

## Matrix conclusions

- The canonical required runtime DB secret is the only Compose variable that
  prevents a clean render and is absent from the example.
- Legacy variables dominate the local file, while canonical policy/state is intended
  to live in PostgreSQL.
- Paper mode should have no broker or unnecessary cloud-provider credentials.
- Duplicates must be rejected by a preflight parser rather than resolved by
  last-value-wins semantics.

