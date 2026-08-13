"""Deferred (baseline) tool execution – fire all tools after decode ends."""

from typing import Dict, List

from .base import ToolTriggerDetector


class DeferredToolExec(ToolTriggerDetector):
    """Baseline mode (``decode_tool_streaming=False``)."""

    def __init__(self, tool_infos: List[Dict]) -> None:
        self._tools = list(tool_infos)

    def on_token(self, token_idx: int, token_text: str) -> List[Dict]:
        return []

    def on_stream_end(self) -> List[Dict]:
        return self._tools
