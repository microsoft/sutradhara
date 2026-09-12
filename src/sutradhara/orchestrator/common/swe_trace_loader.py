"""Trace loader for SWE-agent traces (terminus / mini-swe).

Supports a single JSON file with one of two layouts:
  (A) Dict-of-instances: {"instance_id_1": {...}, "instance_id_2": {...}}
  (B) Single trace: {"instance_id": str, "messages"|"iterations": [...], ...}

Also accepts a directory of ``.json`` files.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from sutradhara.orchestrator.common.request_classes import AgenticRequest
from sutradhara.orchestrator.logger.logger import setup_logging

logger = setup_logging()

TokenizeFn = Callable[[str], Tuple[List[int], List[Tuple[int, int]]]]

# Match a full <tool_call>…</tool_call> block; group 1 is the JSON body.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_INFO_KEEP = ("model", "dataset", "agent", "model_provider", "task", "date")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _normalize_trace(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize so the trace always has ``messages`` and ``instance_id``."""
    if "iterations" in raw and "messages" not in raw:
        raw["messages"] = raw.pop("iterations")
    if "instance_id" not in raw and "request_id" in raw:
        raw["instance_id"] = raw["request_id"]
    return raw


def load_swe_traces(filepath: str) -> List[Dict[str, Any]]:
    """Load SWE-agent trace file(s) into a list of raw instance dicts."""
    if os.path.isdir(filepath):
        traces = []
        for path in sorted(glob.glob(os.path.join(filepath, "*.json"))):
            with open(path) as f:
                traces.append(_normalize_trace(json.load(f)))
        return traces

    with open(filepath) as f:
        data = json.load(f)

    # Dict-of-instances: no top-level messages/iterations.
    if (
        isinstance(data, dict)
        and "messages" not in data
        and "iterations" not in data
    ):
        traces = []
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            if "instance_id" not in entry:
                entry["instance_id"] = key
            traces.append(_normalize_trace(entry))
        return traces

    return [_normalize_trace(data)]


def get_request_id(request: Dict[str, Any]) -> str:
    return request.get("request_id") or request["instance_id"]


# ---------------------------------------------------------------------------
# Step segmentation
# ---------------------------------------------------------------------------

def _get_step_groups(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One step per assistant message (+ following tool/user result msgs)."""
    messages = request["messages"]
    assistant_indices = [
        i for i, m in enumerate(messages) if m.get("role") == "assistant"
    ]
    steps: List[Dict[str, Any]] = []

    for pos, msg_idx in enumerate(assistant_indices):
        assistant_msg = messages[msg_idx]
        end_idx = (
            assistant_indices[pos + 1]
            if pos + 1 < len(assistant_indices)
            else len(messages)
        )

        structured_ids = {
            tc.get("id")
            for tc in assistant_msg.get("tool_calls") or []
            if isinstance(tc, dict) and tc.get("id")
        }

        tool_responses: List[Dict[str, Any]] = []
        followed_by_exit = False
        for j in range(msg_idx + 1, end_idx):
            m = messages[j]
            role = m.get("role")
            if role == "tool" and m.get("tool_call_id") in structured_ids:
                tool_responses.append(m)
            elif role == "user":
                # Text-tool dialect: terminal dump / format-error as user.
                tool_responses.append(m)
            elif role == "exit":
                followed_by_exit = True

        steps.append(
            {
                "step_idx": pos,
                "assistant_msg_idx": msg_idx,
                "assistant": assistant_msg,
                "tool_responses": tool_responses,
                "is_last_step": pos == len(assistant_indices) - 1,
                "followed_by_exit": followed_by_exit,
            }
        )

    return steps


# ---------------------------------------------------------------------------
# Tool parsing
# ---------------------------------------------------------------------------

def parse_tool_calls_from_content(content: str) -> List[Dict[str, Any]]:
    """Parse ``<tool_call>JSON</tool_call>`` markers from assistant text."""
    calls: List[Dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(content or ""):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            calls.append(parsed)
    return calls


def _parse_structured_tool_calls(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for tc in step["assistant"].get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        name = func.get("name", "unknown")
        raw_args = func.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            arguments = {"raw": raw_args}
        calls.append(
            {
                "id": tc.get("id", ""),
                "name": name,
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
        )
    return calls


def _parse_tool_calls(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prefer structured tool_calls; fall back to text markers in content."""
    structured = _parse_structured_tool_calls(step)
    if structured:
        return structured
    content = step["assistant"].get("content") or ""
    return [
        {
            "id": "",
            "name": c.get("name", "unknown"),
            "arguments": c.get("arguments")
            if isinstance(c.get("arguments"), dict)
            else {},
        }
        for c in parse_tool_calls_from_content(content)
    ]


def _assistant_output_text(step: Dict[str, Any]) -> str:
    """Text used for response_length / trigger indices.

    Text-tool traces already store the generated markers in ``content``.
    Structured traces may only have ``tool_calls`` — reconstruct markers so
    trigger positions can still be computed.
    """
    content = step["assistant"].get("content") or ""
    if "<tool_call>" in content:
        return content

    structured = step["assistant"].get("tool_calls") or []
    if not structured:
        return content

    parts = [content]
    for call in _parse_structured_tool_calls(step):
        tool_json = json.dumps(
            {"name": call["name"], "arguments": call["arguments"]},
            ensure_ascii=False,
        )
        parts.append(f"\n<tool_call>\n{tool_json}\n</tool_call>")
    return "".join(parts)


def _latency_from_duration_args(call: Dict[str, Any]) -> float:
    """bash_command ``duration`` (seconds) → ms; else 0."""
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return 0.0
    duration = args.get("duration")
    if duration is None:
        return 0.0
    try:
        return float(duration) * 1000.0
    except (TypeError, ValueError):
        return 0.0


def _get_tool_latencies_ms(
    step: Dict[str, Any], calls: List[Dict[str, Any]]
) -> List[float]:
    """Per-tool latency in ms.

    Prefer structured ``extra.tool_latency`` (seconds) on tool messages;
    otherwise use ``arguments.duration`` from text/structured call args.
    """
    latency_by_id: Dict[str, float] = {}
    for tm in step.get("tool_responses") or []:
        if tm.get("role") != "tool":
            continue
        tc_id = tm.get("tool_call_id", "")
        lat_s = (tm.get("extra") or {}).get("tool_latency")
        if lat_s is None:
            continue
        try:
            latency_by_id[tc_id] = float(lat_s) * 1000.0
        except (TypeError, ValueError):
            continue

    latencies: List[float] = []
    for call in calls:
        tc_id = call.get("id") or ""
        if tc_id and tc_id in latency_by_id and latency_by_id[tc_id] > 0:
            latencies.append(latency_by_id[tc_id])
        else:
            latencies.append(_latency_from_duration_args(call))
    return latencies


# ---------------------------------------------------------------------------
# Prompt / lengths / triggers
# ---------------------------------------------------------------------------

def _clean_message_for_prompt(msg: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {"role": msg["role"]}
    if msg.get("content") is not None:
        clean["content"] = msg["content"]
    if msg.get("tool_calls"):
        clean["tool_calls"] = msg["tool_calls"]
    if msg.get("tool_call_id"):
        clean["tool_call_id"] = msg["tool_call_id"]
    return clean


def _get_accumulated_prompt(request: Dict[str, Any], step: Dict[str, Any]) -> str:
    messages = request["messages"]
    msg_idx = step["assistant_msg_idx"]
    cleaned = [_clean_message_for_prompt(m) for m in messages[:msg_idx]]
    return json.dumps(cleaned, ensure_ascii=False)


def _get_response_length_from_usage(step: Dict[str, Any]) -> int:
    return int(
        (step["assistant"].get("extra") or {})
        .get("response", {})
        .get("usage", {})
        .get("completion_tokens", 0)
        or 0
    )


def _token_count(text: str, tokenize_fn: TokenizeFn) -> int:
    token_ids, _ = tokenize_fn(text)
    return len(token_ids)


def compute_trigger_token_indices(
    result_str: str,
    response_length: int,
    tokenize_fn: TokenizeFn,
) -> List[int]:
    """Map each ``</tool_call>`` end to a token index; align last to response_length-1."""
    # Use end of the full match (includes </tool_call>) — tool is fully known.
    char_ends = [m.end() for m in _TOOL_CALL_RE.finditer(result_str)]
    if not char_ends:
        return []
    _, offset_mapping = tokenize_fn(result_str)
    if not offset_mapping:
        return []

    token_indices: List[int] = []
    for char_end in char_ends:
        for tok_idx, (_s, end) in enumerate(offset_mapping):
            if end >= char_end:
                token_indices.append(tok_idx)
                break
        else:
            token_indices.append(len(offset_mapping) - 1)

    if token_indices and response_length > 0:
        diff = (response_length - 1) - token_indices[-1]
        if diff != 0:
            token_indices = [max(0, idx + diff) for idx in token_indices]
    return token_indices


def _get_tool_info(
    step: Dict[str, Any],
    response_length: int,
    default_latency_ms: float = 0.0,
    tokenize_fn: Optional[TokenizeFn] = None,
) -> List[Dict[str, Any]]:
    calls = _parse_tool_calls(step)
    latencies = _get_tool_latencies_ms(step, calls)
    output_text = _assistant_output_text(step)

    trigger_indices: List[int] = []
    if tokenize_fn is not None and calls and output_text:
        trigger_indices = compute_trigger_token_indices(
            output_text, response_length, tokenize_fn
        )

    return [
        {
            "function": c.get("name", "unknown"),
            "latency": (
                latencies[i]
                if i < len(latencies) and latencies[i] > 0.0
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
# Extraction
# ---------------------------------------------------------------------------

def extract_iterations_info(
    request: Dict[str, Any],
    default_tool_latency_ms: float = 0.0,
    tokenize_fn: Optional[TokenizeFn] = None,
) -> List[Dict[str, Any]]:
    """Convert one SWE instance into a list of iter-info dicts."""
    request_id = get_request_id(request)
    steps = _get_step_groups(request)
    iterations: List[Dict[str, Any]] = []

    for step in steps:
        output_text = _assistant_output_text(step)
        response_length = _get_response_length_from_usage(step)
        if response_length <= 0 and tokenize_fn is not None and output_text:
            response_length = _token_count(output_text, tokenize_fn)

        calls = _parse_tool_calls(step)
        n_tools = len(calls)
        iter_type = "TOOL_DECODE" if n_tools > 0 else "RESPONSE_DECODE"

        tool_info = _get_tool_info(
            step,
            response_length=response_length,
            default_latency_ms=default_tool_latency_ms,
            tokenize_fn=tokenize_fn,
        )

        iterations.append(
            {
                "iteration_id": str(step["step_idx"]),
                "request_id": request_id,
                "tool_info": tool_info,
                "n_tool_calls": n_tools,
                "prompt_json": _get_accumulated_prompt(request, step),
                "iter_type": iter_type,
                "response_length": response_length,
                "is_new_turn": step["step_idx"] == 0,
            }
        )

    return iterations


def load_swe_agentic_requests(
    trace_path: str,
    default_tool_latency_ms: float = 0.0,
    tokenize_fn: Optional[TokenizeFn] = None,
) -> List[AgenticRequest]:
    """Load SWE-agent trace(s) and return AgenticRequest objects."""
    raw_traces = load_swe_traces(trace_path)
    results: List[AgenticRequest] = []
    for trace in raw_traces:
        request_id = get_request_id(trace)
        iters = extract_iterations_info(
            trace,
            default_tool_latency_ms=default_tool_latency_ms,
            tokenize_fn=tokenize_fn,
        )
        if not iters:
            logger.warning("No iterations for SWE request %s, skipping", request_id)
            continue
        results.append(AgenticRequest(request_id=request_id, iterations=iters))
    logger.info(
        "Prepared %d SWE agentic requests from %s", len(results), trace_path
    )
    return results


# ---------------------------------------------------------------------------
# Offline cleanup helper
# ---------------------------------------------------------------------------

def clean_swe_trace_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Strip selection/verifier ballast; rename iterations → messages."""
    out: Dict[str, Any] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        messages = entry.get("messages")
        if messages is None:
            messages = entry.get("iterations")
        info_in = entry.get("info") or {}
        info_out = {k: info_in[k] for k in _INFO_KEEP if k in info_in}
        cleaned = {
            "instance_id": entry.get("instance_id", key),
            "request_id": entry.get("request_id", key),
            "messages": messages or [],
        }
        if info_out:
            cleaned["info"] = info_out
        out[key] = cleaned
    return out
