"""Online tool-call detection: fire each tool as its JSON object closes."""

from typing import Dict, List, Optional

from .base import ToolTriggerDetector

_TOOL_CALLS_MARKER = '"tool_calls"'


class StreamingToolWatcher(ToolTriggerDetector):
    """Fire the i-th pending tool when the i-th object in the streamed
    ``tool_calls`` array closes; defer any unmatched tools to stream end."""

    def __init__(
        self, tool_infos: List[Dict], max_tokens: Optional[int] = None
    ) -> None:
        self._pending: List[Dict] = list(tool_infos)
        self._done = False
        # Marker/array progress, carried across tokens.
        self._marker_len = 0      # chars of _TOOL_CALLS_MARKER matched so far
        self._in_array = False    # marker + opening '[' both seen
        self._depth = 0
        self._in_string = False
        self._escape = False

    def on_token(self, token_idx: int, token_text: str) -> List[Dict]:
        fired: List[Dict] = []
        if self._done or not self._pending:
            return fired

        for ch in token_text:
            if not self._in_array:
                # Locate the start of the tool_calls array.
                if self._marker_len < len(_TOOL_CALLS_MARKER):
                    if ch == _TOOL_CALLS_MARKER[self._marker_len]:
                        self._marker_len += 1
                    else:
                        self._marker_len = 1 if ch == _TOOL_CALLS_MARKER[0] else 0
                elif ch == "[":
                    self._in_array = True
                continue

            # Scan the array, respecting JSON strings/escapes.
            if self._escape:
                self._escape = False
            elif ch == "\\":
                self._escape = True
            elif ch == '"':
                self._in_string = not self._in_string
            elif self._in_string:
                continue
            elif ch == "{":
                self._depth += 1
            elif ch == "}":
                self._depth -= 1
                if self._depth == 0:
                    fired.append(self._pending.pop(0))
                    if not self._pending:
                        self._done = True
                        break
            elif ch == "]" and self._depth == 0:
                self._done = True
                break

        return fired

    def on_stream_end(self) -> List[Dict]:
        remaining = self._pending
        self._pending = []
        return remaining
