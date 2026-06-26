---
title: "CodeCrafters"
type: entity
tags: [platform, education, tutorials, open-source]
created: 2026-06-26
updated: 2026-06-26
sources: [[codecrafters-build-your-own-x]]
status: stable
---

# CodeCrafters

## Definition

Developer education platform and open-source maintainer. Best known for the "Build your own X" repository — a curated index of 200+ step-by-step tutorials for re-creating core computer-science technologies from scratch.

## Intuition

CodeCrafters occupies the "learn by building" niche in developer education. Their Build your own X repo is the most comprehensive open-source index of implementation-level CS tutorials, covering 30+ categories (neural networks, databases, compilers, operating systems, web servers, etc.) in 15+ programming languages.

## How we use it

- **Reference curriculum** — when a Trade_agent subsystem needs deep work, we check this index for relevant tutorials before modifying code.
- **Architecture validation** — our polyglot stack (Python/Rust/Go/TypeScript) mirrors the languages most heavily represented in the collection.
- **Gap identification** — categories with no trading-specific tutorials highlight areas where we need to build our own expertise.

## Strengths & weaknesses

- **Strengths**: Comprehensive coverage; community-validated; free; multi-language; directly applicable to our stack
- **Weaknesses**: General-purpose (no quant/trading tutorials); depth varies; some entries dated

## Related

- [[build-your-own-x]] — the meta-learning concept derived from this source
- [[codecrafters-build-your-own-x]] — the source summary page
- [[nautilustrader]] — execution platform that could benefit from "database internals" tutorials
- [[finrl-brain]] — RL brain that could benefit from "neural network from scratch" understanding

## Sources

- [[codecrafters-build-your-own-x]] — the full Build your own X index
