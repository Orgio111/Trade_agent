# Log

> Append-only timeline of every operation on the wiki. Newest at top. Reverse chronological so the current state is the first thing you read.
>
> Entry format: `## [YYYY-MM-DD] <op> | <subject>` where `<op>` is `ingest` · `query` · `lint` · `note`.
> Quick last-5: `grep "^## \[" log.md | tail -5`

---

## [2026-06-25] note | Wiki initialized

Scaffolded the second brain inside `Trade_agent/wiki/` per the LLM Wiki pattern. Created `ZCODE.md` (schema), `index.md` (catalog), this `log.md`, `README.md`, and the full `raw/` + `content/` directory tree. No sources ingested yet. Awaiting first ingest.
