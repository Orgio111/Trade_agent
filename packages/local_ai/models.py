"""One immutable registry for every approved local Ollama model role."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType


class LocalModelRole(str, Enum):
    """Stable roles resolved only to the operator-approved local models."""

    REASONING = "reasoning"
    FAST = "fast"
    RESEARCH = "research"
    VISION = "vision"
    TOOL_FORMATTING = "tool_formatting"
    EMBEDDING = "embedding"


LOCAL_MODEL_BY_ROLE: Mapping[LocalModelRole, str] = MappingProxyType(
    {
        LocalModelRole.REASONING: "qwen3:8b",
        LocalModelRole.FAST: "phi3:3.8b",
        LocalModelRole.RESEARCH: "deepseek-r1:8b",
        LocalModelRole.VISION: "moondream",
        LocalModelRole.TOOL_FORMATTING: "mistral",
        LocalModelRole.EMBEDDING: "nomic-embed-text",
    }
)


def model_name_matches(actual: str, expected: str) -> bool:
    """Accept Ollama's implicit ``:latest`` only for untagged model names."""

    return actual == expected or (
        ":" not in expected and actual == f"{expected}:latest"
    )
