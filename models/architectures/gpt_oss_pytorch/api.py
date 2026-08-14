"""OpenAI-compatible API adapted from Tony's GPT-OSS PyTorch example.

Upstream source:
https://github.com/myousefnezhad/application-templates/blob/main/Python.Examples/ModelSurgery/GPT_OSS/PyTorch/api.py

Run from the LayerMCP repository root with::

    python -m uvicorn models.architectures.gpt_oss_pytorch.api:app --host 0.0.0.0 --port 8000
"""

import re
import json
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from models.architectures.gpt_oss_pytorch.inference import TokenGenerator
from models.architectures.gpt_oss_pytorch.schemas import ChatCompletionRequest

app = FastAPI()
generator = TokenGenerator()


def clean_model_output(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = text.replace("<|return|>", "")
    return text.strip()


def safe_decode(tokenizer, token_ids: list) -> str:
    """
    Decode a list of token ids, gracefully skipping any special tokens
    that the Harmony tokenizer cannot map to plain text.
    """
    try:
        return tokenizer.decode(token_ids)
    except Exception:
        pass
    parts = []
    for tid in token_ids:
        try:
            parts.append(tokenizer.decode([tid]))
        except Exception:
            pass
    return "".join(parts)


# ------------------------------------------------------------------ #
# Native Harmony format parser                                       #
# ------------------------------------------------------------------ #
# The model's actual wire format (observed from RAW logs) is:
#
#   <commentary text mentioning the tool name>
#   <|end|><|start|>assistant<|channel|>commentary
#   to=functions<|channel|><|constrain|>json<|message|>{...}<|call|>
#
# Key observations:
#   1. "to=" is ALWAYS "to=functions" — it is a fixed dispatch keyword,
#      NOT the tool name.
#   2. The real tool name appears in the commentary reasoning text just
#      before the call, e.g. "We must call get_user_by_email".
#   3. Arguments follow <|message|> as a JSON object.
#   4. The call terminates with <|call|>.
#
# Strategy: extract tool name from the last "to=<name>" that is NOT
# "functions", or fall back to scanning the commentary for a known
# tool name mentioned just before the call.
# ------------------------------------------------------------------ #


def parse_harmony_tool(text: str, known_tools: list[str] | None = None):
    """
    Parse the model's native tool call from generated text.
    Returns {"name": str, "arguments": dict} or None.
    """
    # Extract the JSON arguments — they always follow <|message|>
    m_args = re.search(
        r"<\|message\|>\s*(\{.*?\})\s*(?:<\|call\|>|$)",
        text,
        re.DOTALL,
    )
    args: dict = {}
    if m_args:
        try:
            args = json.loads(m_args.group(1).strip())
        except json.JSONDecodeError:
            args = {}

    # ---- Determine the tool name ----
    # First: find all "to=X" occurrences; use the last non-"functions" one.
    all_to = re.findall(r"to=([a-zA-Z0-9_\-]+)", text)
    real_name = next(
        (n for n in reversed(all_to) if n.lower() != "functions"),
        None,
    )
    if real_name:
        return {"name": real_name, "arguments": args}

    # Fallback: scan the commentary text for a known tool name.
    # The model tends to say "call the tool get_user_by_email" right before
    # the dispatch token.
    if known_tools:
        # Search from the end of the text — the tool mention is usually late.
        for tool in known_tools:
            # match tool name as a whole word
            if re.search(rf"\b{re.escape(tool)}\b", text):
                return {"name": tool, "arguments": args}

    # Nothing parseable — not a tool call.
    return None


def _render_tool_call_as_harmony(tool_call: dict) -> str:
    """
    Convert an OpenAI-style tool_call back into the model's native format
    so the conversation history is rendered correctly in-prompt.
    """
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    raw_args = fn.get("arguments", "{}")
    if isinstance(raw_args, dict):
        args_str = json.dumps(raw_args)
    else:
        args_str = raw_args
    # Use the model's observed format: to=functions ... <tool_name> ... {args}<|call|>
    # Simplest faithful reconstruction that the model will recognise in history:
    return (
        f"to=functions<|channel|><|constrain|>json"
        f"<|message|>{args_str}<|call|>\n"
        f"<!-- called: {name} -->"
    )


# ------------------------------------------------------------------ #
# Health & model listing                                             #
# ------------------------------------------------------------------ #


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-oss-20b", "object": "model", "created": 0, "owned_by": "openai"}
        ],
    }


# ------------------------------------------------------------------ #
# Prompt builder                                                     #
# ------------------------------------------------------------------ #


def _resolve_ref(ref: str, defs: dict) -> dict:
    """
    Resolve a JSON Schema $ref like '#/$defs/RiskType' against the
    $defs block in the same schema. Returns the resolved sub-schema
    or an empty dict if the ref can't be resolved.
    """
    # Only handle local refs of the form #/$defs/<Name>
    if not ref.startswith("#/$defs/"):
        return {}
    key = ref[len("#/$defs/"):]
    return defs.get(key, {})


def _render_parameters(params: dict) -> str:
    """
    Render a JSON Schema parameters object as a clear human-readable block
    so the model knows exactly what arguments to supply.

    Handles:
    - Plain properties:  {"type": "string"}
    - Nullable types:    {"type": ["string", "null"]}
    - $ref to $defs:     {"$ref": "#/$defs/RiskType"} → resolves enum values
    - No properties:     tool takes no arguments
    """
    props = params.get("properties", {})
    required = set(params.get("required", []))
    defs = params.get("$defs", {})

    if not props:
        return "  (no arguments required)"

    lines = []
    for name, schema in props.items():
        req_marker = " [REQUIRED]" if name in required else " [optional]"
        desc = schema.get("description", "")

        # Resolve $ref → look up in $defs for enum/type info
        if "$ref" in schema:
            resolved = _resolve_ref(schema["$ref"], defs)
            base_type = resolved.get("type", "string")
            enum_vals = resolved.get("enum", [])
            if enum_vals:
                typ = f"string, one of: {{{', '.join(enum_vals)}}}"
            else:
                typ = base_type

        else:
            raw_type = schema.get("type", "any")
            # Normalise ["string", "null"] → "string (optional)"
            if isinstance(raw_type, list):
                non_null = [t for t in raw_type if t != "null"]
                typ = non_null[0] if non_null else "any"
                if "null" in raw_type and name not in required:
                    typ += " (nullable)"
            else:
                typ = raw_type

            enum_vals = schema.get("enum", [])
            if enum_vals:
                typ += f", one of: {{{', '.join(str(v) for v in enum_vals)}}}"

        line = f"  - {name} ({typ}){req_marker}"
        if desc:
            line += f": {desc}"
        lines.append(line)

    return "\n".join(lines)


def build_prompt(messages, tools=None):
    out = []

    # ---- system block ----
    system_lines = ["You are a helpful assistant."]

    if tools:
        system_lines.append("\nYou have access to the following tools:\n")
        for t in tools:
            f = t["function"]
            params = f.get("parameters", {})
            print(f"[TOOL SCHEMA] {f['name']}: {json.dumps(params)}")
            system_lines.append(
                f"Tool name: {f['name']}\n\n"
                f"Description:\n{f['description']}\n\n"
                f"Parameters:\n{_render_parameters(params)}\n"
            )

        # Teach the model its OWN native format using real examples from logs.
        # We do NOT invent a new format — we reflect back what the model emits.
        system_lines.append(
            "\nTo call a tool, reason briefly about which tool to use, then emit:\n"
            "to=functions<|channel|><|constrain|>json<|message|>{\"arg\": \"value\"}<|call|>\n\n"
            "The tool name MUST appear immediately before 'to=functions' like this:\n"
            "to=X to=functions<|channel|><|constrain|>json<|message|>{}<|call|>\n\n"
            "Rules:\n"
            "- The first 'to=X' before 'to=functions' is the actual tool name.\n"
            "- Use only tools from the list above.\n"
            "- Arguments must be a valid JSON object. Use {} for no arguments.\n"
            "- Do NOT add any text after <|call|>.\n"
        )

    out.append(
        "<|start|>system<|message|>\n"
        + "\n".join(system_lines)
        + "\n<|end|>\n"
    )

    # ---- conversation history ----
    for m in messages:
        role = m.role.lower()

        if role == "system":
            out.append(f"<|start|>system<|message|>\n{m.content or ''}\n<|end|>\n")

        elif role == "user":
            out.append(f"<|start|>user<|message|>\n{m.content or ''}\n<|end|>\n")

        elif role == "assistant":
            if m.tool_calls:
                # Reconstruct in native Harmony format so history is coherent.
                # We emit "to=<real_name> to=functions..." so the model sees
                # what it actually called.
                parts = []
                for tc in m.tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
                    parts.append(
                        f"to={name} to=functions<|channel|><|constrain|>json"
                        f"<|message|>{args_str}<|call|>"
                    )
                content_prefix = (m.content or "").strip()
                block = (content_prefix + "\n" if content_prefix else "") + "\n".join(parts)
                out.append(
                    f"<|start|>assistant<|channel|>final<|message|>\n{block}\n<|end|>\n"
                )
            else:
                out.append(
                    "<|start|>assistant<|channel|>final<|message|>\n"
                    f"{m.content or ''}\n<|end|>\n"
                )

        elif role == "tool":
            out.append(f"<|start|>tool<|message|>\n{m.content or ''}\n<|end|>\n")

    # ---- assistant turn begins ----
    out.append("<|start|>assistant<|channel|>commentary<|message|>\n")

    prompt = "".join(out)
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    return prompt


def _tool_names_from_req(req: ChatCompletionRequest) -> list[str]:
    return [t["function"]["name"] for t in (req.tools or []) if "function" in t]


# ------------------------------------------------------------------ #
# Streaming                                                          #
# ------------------------------------------------------------------ #


def stream_response(req: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    prompt = build_prompt(req.messages, req.tools)
    tokenizer = generator.tokenizer
    tokens = tokenizer.encode(prompt, allowed_special="all")
    known_tools = _tool_names_from_req(req)

    first_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model or "gpt-oss-20b",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

    try:
        prev_text = ""
        generated = []
        stop_tokens = [generator.eot_token, generator.call_token]

        for token in generator.generate(
            prompt_tokens=tokens,
            stop_tokens=stop_tokens,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            return_logprobs=False,
        ):
            generated.append(token)
            text = safe_decode(tokenizer, generated)

            if token == generator.call_token:
                print("RAW:", repr(text))
                tool = parse_harmony_tool(text, known_tools)
                if tool:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "id": f"call_{uuid.uuid4().hex[:8]}",
                                            "type": "function",
                                            "function": {
                                                "name": tool["name"],
                                                "arguments": json.dumps(tool["arguments"]),
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Stream only the commentary/reasoning text that precedes the call.
            # Strip out any Harmony control tokens that leaked into the text.
            clean = re.sub(r"<\|[^|]+\|>", "", text)
            piece = clean[len(re.sub(r"<\|[^|]+\|>", "", prev_text)):]
            prev_text = text
            if not piece:
                continue

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model or "gpt-oss-20b",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model or "gpt-oss-20b",
            "choices": [
                {"index": 0, "delta": {"content": f"\n[SERVER ERROR] {str(e)}"}, "finish_reason": "stop"}
            ],
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


# ------------------------------------------------------------------ #
# Chat completions                                                   #
# ------------------------------------------------------------------ #


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    print(json.dumps(body, indent=2, ensure_ascii=False))
    req = ChatCompletionRequest(**body)
    known_tools = _tool_names_from_req(req)

    if req.stream:
        return StreamingResponse(stream_response(req), media_type="text/event-stream")

    prompt = build_prompt(req.messages, req.tools)
    tokenizer = generator.tokenizer
    tokens = tokenizer.encode(prompt, allowed_special="all")

    out = []
    stop_tokens = [generator.call_token, generator.return_token, generator.eot_token]

    for token in generator.generate(
        prompt_tokens=tokens,
        stop_tokens=stop_tokens,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        return_logprobs=False,
    ):
        out.append(token)

    text = safe_decode(tokenizer, out)
    tool = parse_harmony_tool(text, known_tools)

    if tool:
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": tool["name"],
                                    "arguments": json.dumps(tool["arguments"]),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(out),
                "total_tokens": len(tokens) + len(out),
            },
        }

    clean_text = clean_model_output(text)
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-oss-20b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": clean_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(tokens),
            "completion_tokens": len(out),
            "total_tokens": len(tokens) + len(out),
        },
    }
