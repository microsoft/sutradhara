"""Abstract interface for building orchestrator KV-cache hints.

Prompt formats differ per workload — production traces are JSON message arrays,
BFCL traces are ChatML — so the parsing is pluggable. The engine-facing contract
is identical across formats: a list of ``{"start", "end", "type"}`` dicts with
token-level offsets. Only the parsing differs.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable


class KVHintBuilder(ABC):
    """Builds the semantic hints the serving layer uses to tag KV blocks."""

    @abstractmethod
    def build_semantic_hints(
        self,
        prompt: str,
        tokenize_fn: Callable[[str], tuple[list[int], list[tuple[int, int]]]],
    ) -> list[dict[str, Any]]:
        """Return ``[{"start": int, "end": int, "type": str}, ...]`` token hints."""

    @abstractmethod
    def build_split_a_hints(self, prompt: str) -> list[dict[str, Any]]:
        """Single ``split_a`` hint pinning the whole prefix at top priority."""
