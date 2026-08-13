"""KV-cache hints for BFCL v4 traces (ChatML prompts).

BFCL prompts are ChatML (``<|im_start|>role\\n...<|im_end|>``) rather than a JSON
message array, so message boundaries come from the ``<|im_start|>`` markers.
Unlike the production builder, assistant turns after the user query are tagged
``response`` rather than lumped in with ``tool_output`` — BFCL interleaves many
assistant/tool steps per request, and the two have different reuse behaviour.
"""

import re
from bisect import bisect_left
from typing import Any

from sutradhara.orchestrator.optimizations.kv_hints.base import KVHintBuilder

# Hint type strings (must match KVBlockType names on the serving layer).
SYSTEM_PROMPT = "system_prompt"
USER_QUERY = "user_query"
TOOL_OUTPUT = "tool_output"
RESPONSE = "response"
SPLIT_A = "split_a"

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_IM_START_RE = re.compile(re.escape(_IM_START) + r"(\w+)")


def _parse_chatml_messages(prompt: str) -> list[dict[str, Any]]:
    """Parse ChatML into ``{"role", "content"}`` dicts ([] if no markers)."""
    messages: list[dict[str, Any]] = []
    for m in _IM_START_RE.finditer(prompt):
        role = m.group(1)
        content_start = m.end()
        end_pos = prompt.find(_IM_END, content_start)
        content = (
            prompt[content_start:] if end_pos == -1 else prompt[content_start:end_pos]
        )
        if content.startswith("\n"):
            content = content[1:]
        messages.append({"role": role, "content": content})
    return messages


def _find_chatml_char_boundaries(
    prompt: str,
    messages: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Return ``(char_start, char_end)`` per message, spanning marker to marker.

    Falls back to an even split if the marker count and message count disagree,
    so a malformed prompt still yields usable (if approximate) hints.
    """
    starts = [m.start() for m in _IM_START_RE.finditer(prompt)]
    if len(starts) != len(messages):
        total, n = len(prompt), len(messages)
        return [(i * total // n, (i + 1) * total // n) for i in range(n)]

    boundaries: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(prompt)
        boundaries.append((start, end))
    return boundaries


def _find_last_user_index(messages: list[dict[str, Any]]) -> int:
    """Return the index of the last message with ``role == 'user'``, or -1."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _assign_semantic_types(messages: list[dict[str, Any]]) -> list[str]:
    """Tag each message: prefix before the query, the query, then per-step
    assistant ``response`` vs ``tool_output``."""
    last_user_idx = _find_last_user_index(messages)
    msg_types: list[str] = []
    for i, msg in enumerate(messages):
        if last_user_idx < 0 or i < last_user_idx:
            msg_types.append(SYSTEM_PROMPT)
        elif i == last_user_idx:
            msg_types.append(USER_QUERY)
        else:
            msg_types.append(
                RESPONSE if msg.get("role") == "assistant" else TOOL_OUTPUT
            )
    return msg_types


def _char_boundaries_to_token_boundaries(
    char_boundaries: list[tuple[int, int]],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map char boundaries to token boundaries via binary search on token
    start offsets."""
    if not offsets or not char_boundaries:
        return [(0, 0)] * len(char_boundaries)

    token_starts = [s for s, _ in offsets]
    n = len(token_starts)

    token_boundaries: list[tuple[int, int]] = []
    for char_start, char_end in char_boundaries:
        tok_start = min(bisect_left(token_starts, char_start), n)
        tok_end = min(bisect_left(token_starts, char_end), n)
        tok_end = max(tok_start, tok_end)
        token_boundaries.append((tok_start, tok_end))

    return token_boundaries


def _merge_hints(
    token_boundaries: list[tuple[int, int]],
    msg_types: list[str],
) -> list[dict[str, Any]]:
    """Collapse adjacent same-type segments, dropping empty ones."""
    hints: list[dict[str, Any]] = []
    for (tok_start, tok_end), seg_type in zip(token_boundaries, msg_types):
        if tok_start >= tok_end:
            continue
        if hints and hints[-1]["type"] == seg_type:
            hints[-1]["end"] = tok_end
        else:
            hints.append({"start": tok_start, "end": tok_end, "type": seg_type})
    return hints


class BFCLKVHintBuilder(KVHintBuilder):
    """BFCL trace prompt format: ChatML (``<|im_start|>role\\n...<|im_end|>``)."""

    def build_semantic_hints(self, prompt, tokenize_fn):
        messages = _parse_chatml_messages(prompt)
        if not messages:
            return []
        char_boundaries = _find_chatml_char_boundaries(prompt, messages)
        msg_types = _assign_semantic_types(messages)
        _, offsets = tokenize_fn(prompt)
        token_boundaries = _char_boundaries_to_token_boundaries(
            char_boundaries, offsets
        )
        return _merge_hints(token_boundaries, msg_types)

    def build_split_a_hints(self, prompt):
        if not prompt:
            return []
        return [{"start": 0, "end": len(prompt), "type": SPLIT_A}]
