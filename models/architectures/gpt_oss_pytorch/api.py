"""OpenAI-compatible HTTP API for the local GPT-OSS PyTorch model.

Run from the repository root with::

    uvicorn models.architectures.gpt_oss_pytorch.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import re
import time
import uuid
from functools import lru_cache
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from models.architectures.gpt_oss_pytorch.inference import parse_tool_call
from models.architectures.gpt_oss_pytorch.schemas import ChatCompletionRequest


MODEL_ID = "gpt-oss-20b"
app = FastAPI(title="Local GPT-OSS API")


def _error_response(message: str, status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "code": code,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error(_request, exc: RequestValidationError) -> JSONResponse:
    return _error_response(
        "Invalid chat completion request: " + str(exc.errors()),
        422,
        "invalid_request",
    )


@lru_cache(maxsize=1)
def get_generator():
    """Load the checkpoint on the first completion request, not at import time."""
    from models.architectures.gpt_oss_pytorch.inference import TokenGenerator

    return TokenGenerator()


def clean_model_output(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.replace("```", "").replace("<|return|>", "").strip()


def safe_decode(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode tokens while tolerating Harmony control tokens."""
    decode = getattr(tokenizer, "decode_utf8", None) or getattr(tokenizer, "decode")
    try:
        return decode(token_ids)
    except Exception:
        parts = []
        for token_id in token_ids:
            try:
                parts.append(decode([token_id]))
            except Exception:
                continue
        return "".join(parts)


def _render_parameters(parameters: dict[str, Any]) -> str:
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    definitions = parameters.get("$defs", {})
    if not properties:
        return "  (no arguments required)"

    lines = []
    for name, schema in properties.items():
        resolved = schema
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            resolved = definitions.get(ref.removeprefix("#/$defs/"), schema)
        value_type = resolved.get("type", "any")
        if isinstance(value_type, list):
            value_type = " | ".join(value_type)
        enum = resolved.get("enum")
        if enum:
            value_type = f"{value_type}; one of: {', '.join(map(str, enum))}"
        marker = "required" if name in required else "optional"
        description = schema.get("description") or resolved.get("description", "")
        suffix = f": {description}" if description else ""
        lines.append(f"  - {name} ({value_type}, {marker}){suffix}")
    return "\n".join(lines)


def build_prompt(messages: list[Any], tools: list[dict[str, Any]] | None = None) -> str:
    system_lines = ["You are a helpful assistant."]
    if tools:
        system_lines.append("\nYou have access to the following tools:\n")
        for tool in tools:
            function = tool["function"]
            system_lines.append(
                f"Tool name: {function['name']}\n"
                f"Description: {function.get('description', '')}\n"
                f"Parameters:\n{_render_parameters(function.get('parameters', {}))}\n"
            )
        system_lines.append(
            "\nTo call a tool, emit exactly:\n"
            "to=TOOL_NAME to=functions<|channel|><|constrain|>json"
            "<|message|>{\"arg\": \"value\"}<|call|>\n"
            "Use only a listed tool, provide a valid JSON object, and add no text "
            "after <|call|>."
        )

    system_text = "\n".join(system_lines)
    output = [f"<|start|>system<|message|>\n{system_text}\n<|end|>\n"]
    for message in messages:
        role = message.role.lower()
        content = message.content or ""
        if role == "assistant" and message.tool_calls:
            calls = []
            for tool_call in message.tool_calls:
                function = tool_call.get("function", {})
                arguments = function.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                calls.append(
                    f"to={function.get('name', '')} to=functions"
                    f"<|channel|><|constrain|>json<|message|>{arguments}<|call|>"
                )
            content = (content + "\n" if content else "") + "\n".join(calls)
            output.append(
                f"<|start|>assistant<|channel|>final<|message|>\n{content}\n<|end|>\n"
            )
        elif role == "assistant":
            output.append(
                f"<|start|>assistant<|channel|>final<|message|>\n{content}\n<|end|>\n"
            )
        else:
            output.append(f"<|start|>{role}<|message|>\n{content}\n<|end|>\n")
    output.append("<|start|>assistant<|channel|>commentary<|message|>\n")
    return "".join(output)


def _tool_names(request: ChatCompletionRequest) -> list[str]:
    return [
        tool["function"]["name"]
        for tool in request.tools or []
        if isinstance(tool.get("function"), dict) and "name" in tool["function"]
    ]


def _base(request_id: str, created: int, model: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }


def stream_response(request: ChatCompletionRequest) -> Iterator[str]:
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    model = request.model or MODEL_ID
    base = _base(request_id, created, model)
    yield "data: " + json.dumps(
        base | {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    ) + "\n\n"

    generated: list[int] = []
    previous_clean = ""
    try:
        generator = get_generator()
        prompt = build_prompt(request.messages, request.tools)
        tokens = generator.tokenizer.encode(prompt, allowed_special="all")
        for token in generator.generate(
            prompt_tokens=tokens,
            stop_tokens=[generator.eot_token, generator.call_token],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            return_logprobs=False,
        ):
            generated.append(token)
            text = safe_decode(generator.tokenizer, generated)
            if token == generator.call_token:
                tool_call = parse_tool_call(text, _tool_names(request))
                if tool_call:
                    payload = base | {
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [{
                                "index": 0,
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": tool_call.function.model_dump(),
                            }]},
                            "finish_reason": "tool_calls",
                        }]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
                return

            clean = re.sub(r"<\|[^|]+\|>", "", text)
            piece = clean[len(previous_clean):]
            previous_clean = clean
            if piece:
                payload = base | {
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        payload = base | {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        payload = {
            "error": {
                "message": f"Model generation failed: {exc}",
                "type": "server_error",
                "code": "generation_failed",
            }
        }
        yield "event: error\n"
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "openai"}],
    }


@app.post("/v1/chat/completions")
async def chat(completion_request: ChatCompletionRequest):
    if completion_request.stream:
        return StreamingResponse(
            stream_response(completion_request), media_type="text/event-stream"
        )

    try:
        generator = get_generator()
        prompt = build_prompt(completion_request.messages, completion_request.tools)
        prompt_tokens = generator.tokenizer.encode(prompt, allowed_special="all")
        output_tokens = list(generator.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=[generator.call_token, generator.return_token, generator.eot_token],
            temperature=completion_request.temperature,
            max_tokens=completion_request.max_tokens,
            return_logprobs=False,
        ))
    except Exception as exc:
        return _error_response(
            f"Model generation failed: {exc}",
            500,
            "generation_failed",
        )
    text = safe_decode(generator.tokenizer, output_tokens)
    tool_call = parse_tool_call(text, _tool_names(completion_request))
    message: dict[str, Any]
    finish_reason = "stop"
    if tool_call:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": tool_call.function.model_dump(),
            }],
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": clean_model_output(text)}

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": completion_request.model or MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(output_tokens),
            "total_tokens": len(prompt_tokens) + len(output_tokens),
        },
    }
