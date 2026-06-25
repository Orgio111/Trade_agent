# Second Brain — Trading & Quant Wiki

A persistent, compounding knowledge base for trading and quantitative research. The LLM (ZCode) reads sources, extracts knowledge, and **integrates** it into this wiki so that every question benefits from everything ever read — instead of re-deriving answers from scratch each time.

## What's where

| Path | What | Who writes it |
|------|------|---------------|
| [`ZCODE.md`](ZCODE.md) | The schema — rules, conventions, operations. The constitution. | ZCode (co-evolved with you) |
| [`index.md`](index.md) | Content catalog, by category. First stop for any question. | ZCode |
| [`log.md`](log.md) | Append-only timeline of every ingest / query / lint. | ZCode |
| `raw/` | Immutable sources: papers, articles, notes, data, assets. **Read-only.** | You |
| `content/` | The wiki itself: concepts, strategies, indicators, entities, etc. | ZCode |

## How to use it (for the human)

1. **Add a source** → drop a file into `raw/` (e.g. a PDF in `raw/papers/`, a clipped article in `raw/articles/`) and tell ZCode: *"ingest `<file>`"*.
2. **Ask a question** → *"how does the wiki think about order-flow imbalance?"* or *"compare mean-reversion vs momentum here."* ZCode reads the wiki and answers with citations.
3. **Health-check** → *"lint the wiki"* to catch contradictions, orphans, and gaps.

In practice: keep ZCode open on one side and Obsidian on the other. ZCode edits; you browse the graph and the pages in real time.

## Conventions (short version — full rules in `ZCODE.md`)

- Pages are `kebab-case.md`, linked as `[[page-slug]]`.
- Every page has YAML frontmatter (`title`, `type`, `tags`, `created`, `updated`, `sources`, `status`).
- One fact lives in one place; everything else links to it.
- Raw sources are never modified. The wiki is synthesized, not copy-pasted.

---
*This wiki lives inside the `Trade_agent` repo as a knowledge layer alongside the trading system code. Code and wiki are separate concerns that share a home.*
