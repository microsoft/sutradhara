"""Decode-tool-streaming optimization: *when* tool calls launch vs. decode.

- ``decode_tool_streaming=False`` (baseline, ``DeferredToolExec``): fire all
  tools after the decode stream finishes.
- ``decode_tool_streaming=True``: fire each tool as its trigger point is reached,
  overlapping tool execution with the rest of decode. Two detectors —
  ``TraceTriggerDetector`` replays pre-recorded ``trigger_token_index`` positions
  (when ``has_trigger_indices``); ``StreamingToolWatcher`` watches the online decode
  stream and fires each tool as its JSON object completes.

All implement the ``ToolTriggerDetector`` ABC consumed by the replayer.
"""

from sutradhara.orchestrator.optimizations.decode_tool_streaming.base import (
    ToolTriggerDetector,
)
from sutradhara.orchestrator.optimizations.decode_tool_streaming.deferred import (
    DeferredToolExec,
)
from sutradhara.orchestrator.optimizations.decode_tool_streaming.streaming_watcher import (
    StreamingToolWatcher,
)
from sutradhara.orchestrator.optimizations.decode_tool_streaming.trace_trigger import (
    TraceTriggerDetector,
    has_trigger_indices,
)

__all__ = [
    "ToolTriggerDetector",
    "DeferredToolExec",
    "TraceTriggerDetector",
    "StreamingToolWatcher",
    "has_trigger_indices",
]
