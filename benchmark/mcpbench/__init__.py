"""mcpbench — MCP tool-selection benchmark harness for LayerMCP.

Backend-agnostic: it talks to any OpenAI-compatible /v1/chat/completions
endpoint (llama-server locally, vLLM on the cluster), so the same suite and
metrics apply across models, quantizations, and fine-tunes.
"""

__version__ = "0.1.0"
