"""Abstract interface for locating a prompt's Split A / Split B boundary.

Finding the split point means finding the assistant→tool boundary, which is
format-specific, so it is pluggable per workload. Returning -1 disables the
split for that prompt and the request runs as an ordinary single prefill.
"""

from abc import ABC, abstractmethod


class PrefillSplitter(ABC):
    @abstractmethod
    def get_split_a_end_token(self, prompt: str) -> int:
        """Char index where Split A ends, or -1 if there is no valid split point."""
