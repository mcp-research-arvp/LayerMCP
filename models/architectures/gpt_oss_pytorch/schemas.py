from typing import Optional, Any
from pydantic import BaseModel


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: Optional[str] = None
    type: str = "function"
    function: ToolFunction


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: list[dict[str, Any]] = []
    parallel_tool_calls: bool = False
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None


class GenerationResult(BaseModel):
    text: str
    tool_call: Optional[ToolCall] = None
