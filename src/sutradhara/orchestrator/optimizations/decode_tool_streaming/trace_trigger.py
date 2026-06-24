"""Trace-driven tool triggering – fire tools at pre-recorded token indices."""

from typing import Dict, List

from .base import ToolTriggerDetector


def _trigger_index(tool_info: Dict) -> int:
    """Recorded trigger token index for a tool, or -1 if absent/non-integer."""
    idx = tool_info.get("trigger_token_index", tool_info.get("trigger_idx", -1))
    return idx if isinstance(idx, int) else -1


def has_trigger_indices(tool_infos: List[Dict]) -> bool:
    """True if any tool carries a usable (>= 0) pre-recorded trigger position."""
    return any(_trigger_index(t) >= 0 for t in tool_infos)


class TraceTriggerDetector(ToolTriggerDetector):
    """Fire tools at the ``trigger_token_index`` positions recorded in the trace."""

    def __init__(self, tool_infos: List[Dict], max_tokens: int) -> None:
        self._triggers: Dict[int, List[Dict]] = {}
        self._deferred: List[Dict] = []

        for t in tool_infos:
            idx = _trigger_index(t)
            if 0 <= idx < max_tokens:
                self._triggers.setdefault(idx, []).append(t)
            else:
                self._deferred.append(t)

    def on_token(self, token_idx: int, token_text: str) -> List[Dict]:
        return self._triggers.get(token_idx, [])

    def on_stream_end(self) -> List[Dict]:
        return self._deferred
