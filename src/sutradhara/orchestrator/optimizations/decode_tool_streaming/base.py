"""Base class for tool-trigger detection during decode streaming."""

from abc import ABC, abstractmethod
from typing import Dict, List


class ToolTriggerDetector(ABC):
    """Decides which tools to launch at each decode step.

    Implementations are called once per decoded token (``on_token``) and
    once after the stream ends (``on_stream_end``).  The returned list of
    tool-info dicts is used by the orchestrator to kick off tool execution.
    """

    @abstractmethod
    def on_token(self, token_idx: int, token_text: str) -> List[Dict]:
        """Called once per decoded token.

        Returns a (possibly empty) list of tool-info dicts whose execution
        should be kicked off *now*.
        """

    @abstractmethod
    def on_stream_end(self) -> List[Dict]:
        """Called after the last token has been consumed.

        Returns any tools that were deferred to stream end.
        """
