from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
import orjson
from transformers import AutoTokenizer

from sutradhara.orchestrator.logger.logger import setup_logging
from sutradhara.orchestrator.optimizations.kv_cache_hints import (
    build_semantic_hints,
    build_split_a_hints,
)

logger = setup_logging()


class OrchestratorEngineAPI(ABC):
    """Orchestrator engine co-design API"""

    @abstractmethod
    def tokenize(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Return ``(token_ids, offset_mapping)`` for *text*."""

    @abstractmethod
    def generate(
        self, prompt: str, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Submit a completion request, yielding response-chunk dicts."""

    def tag_kv_blocks(self, prompt_json: str) -> List[Dict[str, Any]]:
        """Tag KV blocks with semantic roles → ``orchestrator_hints``."""
        return build_semantic_hints(prompt_json, self.tokenize)

    def set_reuse_priority(self, prompt_json: str) -> List[Dict[str, Any]]:
        """Pin the prompt region at top reuse priority (split-A ``split_a`` hint)."""
        return build_split_a_hints(prompt_json)

    def submit_partial_prefill(
        self, prompt: str, priority: int, request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Eagerly prime the tool-independent prefix (split-A), pinned via split_a hints."""
        return self.generate(
            prompt,
            max_tokens=1,
            min_tokens=1,
            ignore_eos=True,
            priority=priority,
            request_id=request_id,
            stream=False,
            orchestrator_hints=self.set_reuse_priority(prompt),
        )

    def extend_prefill(
        self,
        prompt: str,
        max_tokens: int,
        priority: int,
        request_id: str,
        orchestrator_hints: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Extend the (optionally pinned) prefix with full context and decode."""
        return self.generate(
            prompt,
            max_tokens=max_tokens,
            min_tokens=max_tokens,
            ignore_eos=True,
            stream=True,
            priority=priority,
            request_id=request_id,
            orchestrator_hints=orchestrator_hints,
        )


class VLLMClient(OrchestratorEngineAPI):
    """Async client for interacting with vLLM server using httpx."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        max_connections: int = 100,
        max_keepalive: int = 20,
        timeout: Optional[float] = None,
        tokenizer_name: str = "Qwen/Qwen3-14B",
    ):
        self.base_url: str = base_url
        self.generate_url: str = f"{base_url}/v1/completions"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        logger.info(
            "Loaded tokenizer: %s (vocab=%d)", tokenizer_name, len(self.tokenizer)
        )

        limits = httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_keepalive
        )
        self.client = httpx.AsyncClient(
            timeout=None if timeout is None else httpx.Timeout(timeout), limits=limits
        )

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Tokenize *text* and return (token_ids, offset_mapping).

        offset_mapping is a list of (char_start, char_end) per token,
        enabling exact character→token boundary mapping.
        """
        enc = self.tokenizer(text, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        priority: Optional[int] = 0,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "priority": priority,
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        if stream:
            payload["stream_options"] = {"include_usage": True}

        if request_id is not None:
            payload["request_id"] = request_id

        _JSON_HEADERS = {"Content-Type": "application/json"}
        serialized = orjson.dumps(payload)

        try:
            if stream:
                async with self.client.stream(
                    "POST", self.generate_url, content=serialized, headers=_JSON_HEADERS
                ) as resp:
                    resp.raise_for_status()

                    async for raw_line in resp.aiter_lines():
                        if not raw_line:
                            continue

                        if raw_line.startswith("data: "):
                            data_str = raw_line[6:]
                        else:
                            data_str = raw_line

                        if data_str.startswith("[DONE]"):
                            break

                        try:
                            data = orjson.loads(data_str)

                            # Safe text extraction (handle empty choices in final chunk)
                            text = ""
                            if "choices" in data and len(data["choices"]) > 0:
                                text = data["choices"][0].get("text", "")

                            yield {
                                "text": text,
                                "usage": data.get("usage"),
                                "metrics": data.get("metrics"),
                                "finished": False,
                            }

                        except orjson.JSONDecodeError:
                            logger.error(
                                f"Failed to decode JSON chunk for {request_id}: {data_str}"
                            )
                            continue
                        except Exception as e:
                            logger.error(f"Error parsing chunk for {request_id}: {e}")
                            continue
            else:
                # Non-streaming fallback
                resp = await self.client.post(
                    self.generate_url, content=serialized, headers=_JSON_HEADERS
                )
                resp.raise_for_status()
                data = orjson.loads(resp.content)

                text = ""
                if "choices" in data and len(data["choices"]) > 0:
                    text = data["choices"][0].get("text", "")

                yield {
                    "text": text,
                    "usage": data.get("usage"),
                    "metrics": data.get("metrics"),
                    "finished": True,
                }

        except httpx.HTTPError as e:
            error_details = ""
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                error_details = f" | Response body: {e.response.text}"
            logger.error(
                "Failed at request id %s with error: %s%s",
                request_id,
                str(e),
                error_details,
            )
            # Yield empty structure on error
            yield {"text": "", "metrics": {}, "usage": {}, "finished": True}

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        await self.client.aclose()
