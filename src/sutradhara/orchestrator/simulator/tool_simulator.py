from typing import Any, Dict, Optional
import asyncio


class ToolSimulator:
    def __init__(self, tool_latencies: Optional[Dict[str, float]] = None):
        self.tool_latencies = tool_latencies or {}
        self.default_latency = 100.0  # Default 100ms

    async def execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Use provided latency from trace, or fall back to configured/default
        execution_time_ms = (
            latency_ms
            if latency_ms is not None
            else self.tool_latencies.get(tool_name, self.default_latency)
        )

        # Sleep to emulate tool execution time
        await asyncio.sleep(execution_time_ms / 1000.0)

        result = {
            "tool_name": tool_name,
            "args": tool_args,
            "result": f"Mock result from {tool_name}",
            "tool_call_time_ms": execution_time_ms,
        }

        return result
