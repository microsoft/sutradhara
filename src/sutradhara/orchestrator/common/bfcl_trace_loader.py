"""Trace loader for BFCL v4: JSONL, ChatML prompts, XML tool calls.

One JSON object per line, each one request. Outer list = turn, inner = step:

    "id":                       str
    "result":                   List[List[str]]    # model output text
    "input_token_count":        List[List[int]]
    "output_token_count":       List[List[int]]
    "cached_token_count":       List[List[int]]
    "kv_cache_hit_percentage":  List[List[float]]
    "latency":                  List[List[float]]  # ms
    "tool_latency":             List[List[...]]    # seconds, compressed
    "inference_log":            List[Dict]
    "reasoning_content":        List[Optional[str]]
"""

import json
import re
from typing import Callable, Dict, List, Any, Optional, Tuple

from sutradhara.orchestrator.common.request_classes import AgenticRequest


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_bfcl_trace(filepath: str) -> List[Dict[str, Any]]:
    """Load BFCL JSONL trace file (one JSON object per line)."""
    requests = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))
    return requests


# ---------------------------------------------------------------------------
# Request-level accessors
# ---------------------------------------------------------------------------

def get_request_id(request: Dict[str, Any]) -> str:
    return request["id"]


def get_num_turns(request: Dict[str, Any]) -> int:
    return len(request["result"])


def get_num_steps(request: Dict[str, Any], turn_idx: int) -> int:
    return len(request["result"][turn_idx])


def get_response_length(request, turn_idx, step_idx):
    return request["output_token_count"][turn_idx][step_idx]


def get_input_token_count(request, turn_idx, step_idx):
    return request["input_token_count"][turn_idx][step_idx]


def get_cached_token_count(request, turn_idx, step_idx):
    return request["cached_token_count"][turn_idx][step_idx]


def get_kv_cache_hit_percentage(request, turn_idx, step_idx):
    return request["kv_cache_hit_percentage"][turn_idx][step_idx]


def get_latency_ms(request, turn_idx, step_idx):
    return request["latency"][turn_idx][step_idx]


def get_reasoning_content(request, turn_idx):
    rc = request.get("reasoning_content")
    if rc is None or turn_idx >= len(rc):
        return None
    return rc[turn_idx]


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

def get_prompt_json(request, turn_idx, step_idx):
    """Extract the formatted prompt from the inference log."""
    log = request["inference_log"][turn_idx]
    step_key = f"step_{step_idx}"
    step_msgs = log[step_key]
    for msg in step_msgs:
        if isinstance(msg, dict) and msg.get("role") == "inference_input":
            content = msg.get("content", {})
            if isinstance(content, dict):
                return content.get("formatted_prompt", "")
    return ""


def get_begin_of_turn_query(request, turn_idx):
    log = request["inference_log"][turn_idx]
    return log.get("begin_of_turn_query", [])


# ---------------------------------------------------------------------------
# Tool call parsing (XML-based: <tool_call>JSON</tool_call>)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TokenizeFn = Callable[[str], Tuple[List[int], List[Tuple[int, int]]]]


def parse_tool_calls(result_str: str) -> List[Dict]:
    """Parse tool calls from <tool_call>JSON</tool_call> markers in text."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(result_str):
        try:
            parsed = json.loads(m.group(1))
            calls.append(parsed)
        except json.JSONDecodeError:
            continue
    return calls


def get_tool_names(request, turn_idx, step_idx) -> List[str]:
    result_str = request["result"][turn_idx][step_idx]
    return [c.get("name", "unknown") for c in parse_tool_calls(result_str)]


def get_n_tool_calls(request, turn_idx, step_idx) -> int:
    return len(parse_tool_calls(request["result"][turn_idx][step_idx]))


def get_tool_latencies_ms(request, turn_idx, step_idx) -> List[float]:
    """Get per-tool latencies from the compressed tool_latency array."""
    tl = request.get("tool_latency")
    if tl is None or turn_idx >= len(tl):
        return []
    # Compute offset into the turn's tool_latency array
    tool_step_offset = 0
    for s in range(step_idx):
        if "<tool_call>" in request["result"][turn_idx][s]:
            tool_step_offset += 1
    if "<tool_call>" not in request["result"][turn_idx][step_idx]:
        return []
    turn_tl = tl[turn_idx]
    if tool_step_offset >= len(turn_tl):
        return []
    return [v * 1000.0 for v in turn_tl[tool_step_offset]]


# ---------------------------------------------------------------------------
# Decode streaming: map each </tool_call> char offset to a token index,
# aligned to response_length (last tool trigger = response_length - 1).
# ---------------------------------------------------------------------------

def compute_trigger_token_indices(
    result_str: str,
    response_length: int,
    tokenize_fn: TokenizeFn,
) -> List[int]:
    """Compute per-tool trigger token indices from result text.

    Each trigger index marks where a tool call is fully known in the decode
    stream. The alignment ensures the last trigger matches response_length - 1
    (the last generated token).
    """
    char_ends = [m.end(1) for m in _TOOL_CALL_RE.finditer(result_str)]
    if not char_ends:
        return []
    _, offset_mapping = tokenize_fn(result_str)
    token_indices = []
    for char_end in char_ends:
        for tok_idx, (_s, end) in enumerate(offset_mapping):
            if end >= char_end:
                token_indices.append(tok_idx)
                break
        else:
            token_indices.append(len(offset_mapping) - 1)
    # Align: force last trigger to response_length - 1
    if token_indices and response_length > 0:
        diff = (response_length - 1) - token_indices[-1]
        if diff != 0:
            token_indices = [max(0, idx + diff) for idx in token_indices]
    return token_indices


# ---------------------------------------------------------------------------
# Tool-info construction (plain dicts — see module docstring)
# ---------------------------------------------------------------------------

def get_tool_info(
    request, turn_idx, step_idx,
    default_latency_ms: float = 0.0,
    default_response_length: int = 0,
    tokenize_fn: Optional[TokenizeFn] = None,
) -> List[Dict[str, Any]]:
    """Build tool_info dict list for a given step."""
    result_str = request["result"][turn_idx][step_idx]
    calls = parse_tool_calls(result_str)
    response_length = default_response_length

    per_call_latencies = []
    if default_latency_ms == 0.0:
        per_call_latencies = get_tool_latencies_ms(request, turn_idx, step_idx)

    trigger_indices = []
    if tokenize_fn is not None and calls:
        response_length = get_response_length(request, turn_idx, step_idx)
        trigger_indices = compute_trigger_token_indices(
            result_str, response_length, tokenize_fn
        )

    return [
        {
            "function": c.get("name", "unknown"),
            "latency": (
                per_call_latencies[i]
                if i < len(per_call_latencies)
                else default_latency_ms
            ),
            "response_length": response_length,
            "trigger_token_index": (
                trigger_indices[i] if i < len(trigger_indices) else -1
            ),
        }
        for i, c in enumerate(calls)
    ]


# ---------------------------------------------------------------------------
# Iteration type & helpers
# ---------------------------------------------------------------------------

def get_iter_type(request, turn_idx, step_idx) -> str:
    if "<tool_call>" in request["result"][turn_idx][step_idx]:
        return "TOOL_DECODE"
    return "RESPONSE_DECODE"


def is_new_turn(step_idx) -> bool:
    return step_idx == 0


def generate_iteration_ids(request) -> List[str]:
    ids, counter = [], 0
    for turn_idx in range(get_num_turns(request)):
        for _ in range(get_num_steps(request, turn_idx)):
            ids.append(str(counter))
            counter += 1
    return ids


# ---------------------------------------------------------------------------
# Main extraction: request → List[iter_info dict]
# ---------------------------------------------------------------------------

def extract_iterations_info(
    request: Dict[str, Any],
    default_tool_latency_ms: float = 0.0,
    default_tool_response_length: int = 0,
    tokenize_fn: Optional[TokenizeFn] = None,
) -> List[Dict[str, Any]]:
    """Convert one BFCL request into a list of iter_info dicts."""
    request_id = get_request_id(request)
    iter_ids = generate_iteration_ids(request)
    iterations, idx = [], 0

    for turn_idx in range(get_num_turns(request)):
        for step_idx in range(get_num_steps(request, turn_idx)):
            tool_info = get_tool_info(
                request, turn_idx, step_idx,
                default_latency_ms=default_tool_latency_ms,
                default_response_length=default_tool_response_length,
                tokenize_fn=tokenize_fn,
            )
            iterations.append({
                "iteration_id": iter_ids[idx],
                "request_id": request_id,
                "tool_info": tool_info,
                "n_tool_calls": get_n_tool_calls(request, turn_idx, step_idx),
                "prompt_json": get_prompt_json(request, turn_idx, step_idx),
                "iter_type": get_iter_type(request, turn_idx, step_idx),
                "response_length": get_response_length(request, turn_idx, step_idx),
                "is_new_turn": is_new_turn(step_idx),
            })
            idx += 1

    return iterations


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def load_bfcl_agentic_requests(
    trace_path: str,
    default_tool_latency_ms: float = 0.0,
    default_tool_response_length: int = 0,
    tokenize_fn: Optional[TokenizeFn] = None,
) -> List[AgenticRequest]:
    """Load BFCL JSONL trace and return AgenticRequest objects."""
    raw_requests = load_bfcl_trace(trace_path)
    results = []
    for req in raw_requests:
        request_id = get_request_id(req)
        iters = extract_iterations_info(
            req,
            default_tool_latency_ms=default_tool_latency_ms,
            default_tool_response_length=default_tool_response_length,
            tokenize_fn=tokenize_fn,
        )
        if not iters:
            continue
        results.append(AgenticRequest(request_id=request_id, iterations=iters))
    return results
