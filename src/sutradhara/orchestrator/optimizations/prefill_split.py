import json

_JSON_DECODER = json.JSONDecoder()


def get_split_a_end_token(prompt: str) -> int:
    """Char index where Split A ends: the start of the first trailing tool
    message. ``prompt[:index]`` is Split A; ``prompt`` is the full Split B.

    Returns -1 if there is no assistant→tool boundary to split on.
    """
    try:
        messages, _ = _JSON_DECODER.raw_decode(prompt)
    except (json.JSONDecodeError, ValueError):
        return -1

    if not isinstance(messages, list) or len(messages) < 2:
        return -1

    # The split is before the trailing run of tool messages.
    n_tools = 0
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            n_tools += 1
        else:
            break
    if n_tools == 0 or n_tools >= len(messages):
        return -1

    # They must be preceded by the assistant that issued matching tool_calls.
    assistant = messages[-n_tools - 1]
    tool_calls = assistant.get("tool_calls")
    if (
        assistant.get("role") != "assistant"
        or not isinstance(tool_calls, list)
        or len(tool_calls) != n_tools
    ):
        return -1

    # Walk the raw string to the (len-n_tools)-th object — the first tool
    # message — reusing the decoder so content braces/keys can't mislead us.
    split_idx = len(messages) - n_tools
    idx = prompt.index("{")
    for _ in range(split_idx):
        _, end = _JSON_DECODER.raw_decode(prompt, idx)
        idx = prompt.index("{", end)
    return idx
