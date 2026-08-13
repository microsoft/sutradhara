"""KV-cache hints for production traces (JSON array of chat-message dicts).

Builds the semantic hints that let the engine manage KV-cache eviction by role:
everything before the last ``user`` message is ``system_prompt``, that message
is ``user_query``, and everything after it is ``tool_output``. Each hint carries
token-level offsets so the serving layer can tag the corresponding KV blocks.
"""

import json
from bisect import bisect_left
from typing import Any, Callable

from sutradhara.orchestrator.optimizations.kv_hints.base import KVHintBuilder

# Hint type strings (must match KVBlockType names on the serving layer).
SYSTEM_PROMPT = "system_prompt"
USER_QUERY = "user_query"
TOOL_OUTPUT = "tool_output"
SPLIT_A = "split_a"

_JSON_DECODER = json.JSONDecoder()


def _parse_prompt_json(prompt_json: str) -> list[dict[str, Any]]:
    """Parse the messages array into dicts ([] on failure).

    ``raw_decode`` tolerates a trailing tool-definitions array after it.
    """
    try:
        messages, _ = _JSON_DECODER.raw_decode(prompt_json)
        return messages
    except (json.JSONDecodeError, ValueError):
        return []


def _find_last_user_index(messages: list[dict[str, Any]]) -> int:
    """Return the index of the last message with ``role == 'user'``, or -1."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def _find_message_char_boundaries(
    prompt_json: str,
    messages: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Return ``(char_start, char_end)`` for each message.

    Each message spans from its opening brace to the next message's brace (the
    last extends to the end of the string). Brace positions come from a
    ``raw_decode`` walk, so braces inside string content can't mislead it.
    """
    starts: list[int] = []
    idx = prompt_json.index("{")
    for _ in messages:
        starts.append(idx)
        _, end = _JSON_DECODER.raw_decode(prompt_json, idx)
        nxt = prompt_json.find("{", end)
        idx = nxt if nxt != -1 else len(prompt_json)

    return [
        (start, starts[i + 1] if i + 1 < len(starts) else len(prompt_json))
        for i, start in enumerate(starts)
    ]


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


def _partition_prompt(
    prompt_json: str,
    tokenize_fn: Callable[[str], tuple[list[int], list[tuple[int, int]]]],
) -> list[dict[str, Any]]:
    """Partition ``prompt_json`` into merged semantic segments with token-level
    offsets: ``[{"start": int, "end": int, "type": str}, ...]``.

    ``tokenize_fn(text) -> (token_ids, offset_mapping)`` supplies the per-token
    ``(char_start, char_end)`` mapping used to convert char spans to tokens.
    """
    messages = _parse_prompt_json(prompt_json)
    if not messages:
        return []

    last_user_idx = _find_last_user_index(messages)

    # Everything up to the last user message is the reusable prefix
    # (system_prompt); that message is the query; everything after is
    # per-request tool_output.
    msg_types: list[str] = []
    for i in range(len(messages)):
        if i == last_user_idx:
            msg_types.append(USER_QUERY)
        elif last_user_idx < 0 or i < last_user_idx:
            msg_types.append(SYSTEM_PROMPT)
        else:
            msg_types.append(TOOL_OUTPUT)

    char_boundaries = _find_message_char_boundaries(prompt_json, messages)
    _, offsets = tokenize_fn(prompt_json)
    token_boundaries = _char_boundaries_to_token_boundaries(
        char_boundaries, offsets
    )

    # Merge adjacent segments of the same type.
    hints: list[dict[str, Any]] = []
    for (tok_start, tok_end), seg_type in zip(token_boundaries, msg_types):
        if tok_start >= tok_end:
            continue
        if hints and hints[-1]["type"] == seg_type:
            hints[-1]["end"] = tok_end
        else:
            hints.append({"start": tok_start, "end": tok_end, "type": seg_type})

    return hints


class ProductionKVHintBuilder(KVHintBuilder):
    """Production trace prompt format: JSON array of chat-message dicts."""

    def build_semantic_hints(self, prompt, tokenize_fn):
        """Semantic KV hints for a normal or split-B request."""
        return _partition_prompt(prompt, tokenize_fn)

    def build_split_a_hints(self, prompt):
        """Single ``split_a`` hint pinning the whole prefix at top priority for
        the tool-overlap window.
        """
        if not prompt:
            return []
        return [{"start": 0, "end": len(prompt), "type": SPLIT_A}]
