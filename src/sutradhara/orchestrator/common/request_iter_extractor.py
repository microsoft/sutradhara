import json
from typing import List, Dict, Any

from sutradhara.orchestrator.logger.logger import setup_logging
from sutradhara.orchestrator.common.request_classes import AgenticRequest

logger = setup_logging()

_ALLOWED_ITER_TYPES = {"TOOL_DECODE", "RESPONSE_DECODE"}


def load_agentic_requests(trace_path: str) -> List[AgenticRequest]:
    # Load all agentic requests from a trace file.
    try:
        with open(trace_path, "r") as f:
            trace_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error("Failed to load trace %s: %s", trace_path, e)
        return []

    raw_requests = trace_data.get("requests", [])
    logger.info("Loaded %d requests from %s", len(raw_requests), trace_path)

    results: List[AgenticRequest] = []
    for req in raw_requests:
        request_id = req.get("request_id")
        if request_id is None:
            continue

        iterations = [
            it
            for it in req.get("iter_info", [])
            if it.get("iter_type") in _ALLOWED_ITER_TYPES
        ]
        if not iterations:
            logger.warning(
                "No relevant iterations for request %s, skipping.", request_id
            )
            continue

        results.append(AgenticRequest(request_id=request_id, iterations=iterations))

    logger.info("Prepared %d agentic requests from %s", len(results), trace_path)
    return results
