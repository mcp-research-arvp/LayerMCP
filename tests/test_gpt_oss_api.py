import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from models.architectures.gpt_oss_pytorch import api


class _Tokenizer:
    def __init__(self, decoded: str) -> None:
        self.decoded = decoded

    def encode(self, _prompt, allowed_special=None):
        return [1, 2]

    def decode(self, _token_ids):
        return self.decoded


class _Generator:
    call_token = 99
    return_token = 100
    eot_token = 101

    def __init__(self, decoded: str, tokens: list[int] | None = None, error=None) -> None:
        self.tokenizer = _Tokenizer(decoded)
        self.tokens = tokens or [7]
        self.error = error

    def generate(self, **_kwargs):
        if self.error:
            raise self.error
        yield from self.tokens


class GptOssApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    def test_health_and_models_are_openai_compatible(self) -> None:
        health = self.client.get("/health")
        models = self.client.get("/v1/models")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(models.status_code, 200)
        self.assertEqual(models.json()["object"], "list")
        self.assertEqual(models.json()["data"][0]["id"], api.MODEL_ID)

    def test_non_streaming_text_completion_uses_openai_response_shape(self) -> None:
        generator = _Generator("Hello from GPT-OSS.")
        with patch.object(api, "get_generator", return_value=generator):
            response = self.client.post("/v1/chat/completions", json={
                "model": api.MODEL_ID,
                "messages": [{"role": "user", "content": "Hello"}],
            })

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"], {
            "role": "assistant", "content": "Hello from GPT-OSS.",
        })
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["usage"]["total_tokens"], 3)

    def test_non_streaming_tool_call_uses_openai_tool_call_shape(self) -> None:
        generator = _Generator(
            'to=calculator to=functions<|message|>{"expression":"2+2"}<|call|>',
            [99],
        )
        with patch.object(api, "get_generator", return_value=generator):
            response = self.client.post("/v1/chat/completions", json={
                "model": api.MODEL_ID,
                "messages": [{"role": "user", "content": "Calculate 2+2"}],
                "tools": [{"type": "function", "function": {
                    "name": "calculator", "parameters": {"type": "object"},
                }}],
            })

        choice = response.json()["choices"][0]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"], {
            "name": "calculator", "arguments": '{"expression": "2+2"}',
        })

    def test_streaming_text_completion_emits_sse_chunks_and_done_marker(self) -> None:
        generator = _Generator("Hello")
        with patch.object(api, "get_generator", return_value=generator):
            response = self.client.post("/v1/chat/completions", json={
                "model": api.MODEL_ID,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            })

        events = response.text.split("\n\n")
        first = json.loads(events[0].removeprefix("data: "))
        text_chunk = json.loads(events[1].removeprefix("data: "))
        self.assertEqual(response.headers["content-type"].split(";", 1)[0], "text/event-stream")
        self.assertEqual(first["object"], "chat.completion.chunk")
        self.assertEqual(first["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(text_chunk["choices"][0]["delta"], {"content": "Hello"})
        self.assertIn("data: [DONE]", response.text)

    def test_streaming_tool_call_emits_openai_tool_call_chunk(self) -> None:
        generator = _Generator(
            'to=calculator to=functions<|message|>{"expression":"2+2"}<|call|>',
            [99],
        )
        with patch.object(api, "get_generator", return_value=generator):
            response = self.client.post("/v1/chat/completions", json={
                "model": api.MODEL_ID,
                "messages": [{"role": "user", "content": "Calculate 2+2"}],
                "tools": [{"type": "function", "function": {
                    "name": "calculator", "parameters": {"type": "object"},
                }}],
                "stream": True,
            })

        event_payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        tool_choice = event_payloads[1]["choices"][0]
        self.assertEqual(tool_choice["finish_reason"], "tool_calls")
        self.assertEqual(tool_choice["delta"]["tool_calls"][0]["function"]["name"], "calculator")
        self.assertIn("data: [DONE]", response.text)

    def test_invalid_request_returns_client_error(self) -> None:
        response = self.client.post("/v1/chat/completions", json={"messages": []})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_generation_failures_return_openai_error_events_or_responses(self) -> None:
        generator = _Generator("", error=RuntimeError("GPU unavailable"))
        payload = {
            "model": api.MODEL_ID,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with patch.object(api, "get_generator", return_value=generator):
            response = self.client.post("/v1/chat/completions", json=payload)
            stream_response = self.client.post(
                "/v1/chat/completions", json=payload | {"stream": True}
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "generation_failed")
        self.assertIn("event: error", stream_response.text)
        self.assertIn('"code": "generation_failed"', stream_response.text)
        self.assertNotIn("[SERVER ERROR]", stream_response.text)


if __name__ == "__main__":
    unittest.main()
