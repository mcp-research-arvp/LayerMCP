import importlib
import asyncio
import json
import sys
import unittest
from unittest.mock import patch

from models.architectures.gpt_oss_pytorch import inference
from models.architectures.gpt_oss_pytorch.schemas import ChatCompletionRequest


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

    def __init__(self, decoded: str, tokens: list[int] | None = None) -> None:
        self.tokenizer = _Tokenizer(decoded)
        self.tokens = [7] if tokens is None else tokens

    def generate(self, **_kwargs):
        yield from self.tokens


_API_MODULE = "models.architectures.gpt_oss_pytorch.api"
sys.modules.pop(_API_MODULE, None)
_IMPORT_GENERATOR = _Generator("")
with patch.object(
    inference,
    "TokenGenerator",
    return_value=_IMPORT_GENERATOR,
) as _token_generator_constructor:
    api = importlib.import_module(_API_MODULE)
    _IMPORT_INITIALIZATION_COUNT = _token_generator_constructor.call_count


class _Request:
    def __init__(self, body: dict) -> None:
        self.body = body

    async def json(self) -> dict:
        return self.body


class _Response:
    def __init__(
        self,
        payload: dict | None = None,
        *,
        text: str = "",
        content_type: str = "application/json",
    ) -> None:
        self.status_code = 200
        self._payload = payload
        self.text = text
        self.headers = {"content-type": content_type}

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload


class _DirectApiClient:
    """Exercise Tony's endpoint functions without Starlette's cluster thread pool."""

    def get(self, path: str) -> _Response:
        handlers = {
            "/health": api.health,
            "/v1/models": api.models,
        }
        return _Response(handlers[path]())

    def post(self, path: str, *, json: dict) -> _Response:
        if path != "/v1/chat/completions":
            raise AssertionError(path)
        result = asyncio.run(api.chat(_Request(json)))
        if not json.get("stream"):
            return _Response(result)
        self_response = result
        req = ChatCompletionRequest(**json)
        text = "".join(api.stream_response(req))
        return _Response(
            text=text,
            content_type=f"{self_response.media_type}; charset=utf-8",
        )


class GptOssApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _DirectApiClient()

    def test_generator_is_initialized_when_api_module_is_imported(self) -> None:
        self.assertEqual(_IMPORT_INITIALIZATION_COUNT, 1)
        self.assertIs(api.generator, _IMPORT_GENERATOR)

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_models(self) -> None:
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "object": "list",
                "data": [
                    {
                        "id": "gpt-oss-20b",
                        "object": "model",
                        "created": 0,
                        "owned_by": "openai",
                    }
                ],
            },
        )

    def test_non_streaming_text_response(self) -> None:
        generator = _Generator("Hello from GPT-OSS.")
        with patch.object(api, "generator", generator):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(
            payload["choices"][0]["message"],
            {"role": "assistant", "content": "Hello from GPT-OSS."},
        )
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["usage"]["total_tokens"], 3)

    def test_non_streaming_tool_call(self) -> None:
        generator = _Generator(
            'to=calculator to=functions<|message|>{"expression":"2+2"}<|call|>',
            [99],
        )
        with patch.object(api, "generator", generator):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-oss-20b",
                    "messages": [
                        {"role": "user", "content": "Calculate 2+2"}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "description": "Evaluate arithmetic.",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                },
            )

        choice = response.json()["choices"][0]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(
            choice["message"]["tool_calls"][0]["function"],
            {
                "name": "calculator",
                "arguments": '{"expression": "2+2"}',
            },
        )

    def test_streaming_text_response(self) -> None:
        generator = _Generator("Hello")
        with patch.object(api, "generator", generator):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )

        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(
            response.headers["content-type"].split(";", 1)[0],
            "text/event-stream",
        )
        self.assertEqual(payloads[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(payloads[1]["choices"][0]["delta"], {"content": "Hello"})
        self.assertEqual(payloads[2]["choices"][0]["finish_reason"], "stop")
        self.assertIn("data: [DONE]", response.text)

    def test_streaming_tool_call(self) -> None:
        generator = _Generator(
            'to=calculator to=functions<|message|>{"expression":"2+2"}<|call|>',
            [99],
        )
        with patch.object(api, "generator", generator):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-oss-20b",
                    "messages": [
                        {"role": "user", "content": "Calculate 2+2"}
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "description": "Evaluate arithmetic.",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "stream": True,
                },
            )

        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        tool_choice = payloads[1]["choices"][0]
        self.assertEqual(tool_choice["finish_reason"], "tool_calls")
        self.assertEqual(
            tool_choice["delta"]["tool_calls"][0]["function"],
            {
                "name": "calculator",
                "arguments": '{"expression": "2+2"}',
            },
        )
        self.assertIn("data: [DONE]", response.text)

    def test_prompt_rendering_preserves_schema_and_conversation_history(self) -> None:
        request = ChatCompletionRequest(
            model="gpt-oss-20b",
            messages=[
                {"role": "system", "content": "Follow policy."},
                {"role": "user", "content": "Calculate 2+2."},
                {
                    "role": "assistant",
                    "content": "I will calculate it.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": {"expression": "2+2"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "4"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Evaluate arithmetic.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Expression to evaluate.",
                                }
                            },
                            "required": ["expression"],
                        },
                    },
                }
            ],
        )

        prompt = api.build_prompt(request.messages, request.tools)

        self.assertIn("Tool name: calculator", prompt)
        self.assertIn("- expression (string) [REQUIRED]", prompt)
        self.assertIn("<|start|>system<|message|>\nFollow policy.", prompt)
        self.assertIn("<|start|>user<|message|>\nCalculate 2+2.", prompt)
        self.assertIn(
            "to=calculator to=functions<|channel|><|constrain|>json"
            '<|message|>{"expression": "2+2"}<|call|>',
            prompt,
        )
        self.assertIn("<|start|>tool<|message|>\n4", prompt)
        self.assertTrue(
            prompt.endswith(
                "<|start|>assistant<|channel|>commentary<|message|>\n"
            )
        )

    def test_harmony_parsing_uses_direct_name_then_commentary_fallback(self) -> None:
        direct = api.parse_harmony_tool(
            'to=calculator to=functions<|message|>{"expression":"2+2"}<|call|>',
            ["calculator"],
        )
        fallback = api.parse_harmony_tool(
            "We must call calculator. "
            'to=functions<|message|>{"expression":"3+3"}<|call|>',
            ["calculator"],
        )

        self.assertEqual(
            direct,
            {"name": "calculator", "arguments": {"expression": "2+2"}},
        )
        self.assertEqual(
            fallback,
            {"name": "calculator", "arguments": {"expression": "3+3"}},
        )


if __name__ == "__main__":
    unittest.main()
