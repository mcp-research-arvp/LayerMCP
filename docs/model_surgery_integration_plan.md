# ModelSurgery Integration Plan

LayerMCP and Tony's ModelSurgery repo should stay cleanly separated.
LayerMCP is the benchmark/evaluation harness. ModelSurgery is the model
implementation and intervention workspace.

## LayerMCP Responsibilities

- Define benchmark samples.
- Provide available MCP tool names and schemas.
- Provide deterministic offline fixture tools for repeatable tests.
- Build routing prompts.
- Store expected tool labels.
- Score selected tool name against `expected_tool`.
- Track hallucinated or unparsable tool outputs.
- Save per-sample and aggregate results.
- Compare baseline runs against modified-layer runs.

LayerMCP should not implement DeepSeek architecture, safetensor loading,
activation patching, layer randomization, or fine-tuning.

## Tony ModelSurgery Responsibilities

- Validate tokenizer, checkpoint, and generation behavior.
- Define explicit layer-accessible PyTorch architectures.
- Load safetensors/checkpoints.
- Wrap generation.
- Expose an OpenAI-style `/v1/chat/completions` API.
- Return and normalize `tool_calls`.
- Provide direct Python access to layers for interventions.
- Implement DeepSeek architecture work, including future DeepSeek-LLM-7B support.

Tony's existing flow maps well to this split:

- `HuggingFace/batch.py` validates tokenizer/checkpoint/generation.
- `PyTorch/model.py` defines the explicit layer-accessible architecture.
- `PyTorch/weights.py` loads safetensors/checkpoints.
- `PyTorch/inference.py` wraps generation.
- `PyTorch/api.py` exposes OpenAI-style chat completions and normalizes tool calls.

## Easiest First Connection: API Integration

API integration is the easiest first connection because LayerMCP can treat
ModelSurgery as a model server. That keeps benchmark code independent from
architecture details and lets Tony iterate on model internals without changing
LayerMCP.

Expected API contract:

```text
POST /v1/chat/completions
```

Input:

```json
{
  "model": "model-surgery-model-name",
  "messages": [
    {"role": "user", "content": "User query here"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Tool description",
        "parameters": {}
      }
    }
  ],
  "tool_choice": "auto"
}
```

Expected output:

```json
{
  "choices": [
    {
      "message": {
        "content": null,
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "calculator",
              "arguments": "{}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "raw_text": "optional raw model text",
  "parse_status": "optional normalized parse status"
}
```

LayerMCP primarily needs the selected tool name:

```text
choices[0].message.tool_calls[0].function.name
```

Raw text and parse status are useful for debugging parser failures.

## Future Router Interface

LayerMCP adapters should converge on:

```python
choose_tool(user_query: str, available_tools: list[dict]) -> str
```

For compatibility with the current Qwen router, adapters may also accept a list
of tool-name strings where appropriate.

## Direct Python Integration Later

Direct Python integration may be useful after the API baseline works. It would
allow LayerMCP experiments to call ModelSurgery objects directly while Tony's
code modifies or inspects layers.

This is useful for:

- layer randomization
- activation patching
- selective fine-tuning
- direct layer-output logging
- avoiding API serialization overhead during controlled experiments

Direct integration should still keep architecture logic in ModelSurgery.

## Baseline Vs Modified-Layer Workflow

```text
benchmark sample
-> available tools
-> ModelSurgery model/API chooses tool
-> LayerMCP compares selected tool with expected_tool
-> save baseline result
-> modify/randomize/fine-tune layer in ModelSurgery
-> rerun same benchmark
-> compare accuracy and parser failures
```

The first milestone is a stable baseline run through the API. Layer-level
experiments come after the API can reliably return tool calls.
