"""Thin OpenAI-compatible chat client with wall-clock timing.

Deliberately uses raw httpx (not the openai SDK) so we keep the full server
response — including llama.cpp's non-standard `timings` block — for analysis.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str = "local",
        api_key: str = "-",
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.0,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if extra_body:
            payload.update(extra_body)

        started = time.perf_counter()
        resp = self._client.post("/v1/chat/completions", json=payload)
        latency = time.perf_counter() - started
        resp.raise_for_status()
        data = resp.json()
        data["_latency_s"] = latency
        return data

    def close(self) -> None:
        self._client.close()
