import re
import time
import json
import torch
from .config import Config
from .model import Transformer
from .schemas import ToolCall, ToolFunction, GenerationResult
from typing import Optional, List, Any, Generator, Union, Tuple
from transformers import AutoTokenizer


def debug_print(*args: Any, **kwargs: Any) -> None:
    if Config.debug_mode:
        print("[DEBUG]", *args, **kwargs)


def get_tokenizer(checkpoint: str = Config.checkpoint_path):
    """
    Load the Gemma 4 tokenizer (SentencePiece + chat template) straight from
    the checkpoint directory. This replaces the GPT-OSS Harmony encoding.
    """
    return AutoTokenizer.from_pretrained(checkpoint)


# Gemma 4 tool-call control tokens (see convert_gemma4_weights._RESPONSE_TEMPLATE
# and the tokenizer special tokens). A tool call looks like:
#
#   <|tool_call>call:NAME{ key: <|"|>string val<|"|>, n: 3, flag: true }<tool_call|>
#
# i.e. keys are UNQUOTED and string values are wrapped in the escape token
# `<|"|>` (not regular quotes). The format may repeat for parallel calls.
_TC_OPEN = "<|tool_call>"
_TC_CLOSE = "<tool_call|>"
_TC_ESC = '<|"|>'  # escape_token, acts as the string delimiter inside arg blobs
# Matches the call name only; the brace body is extracted with an escape-aware
# scanner (below) because braces/commas/colons may appear inside string values.
_TC_NAME_RE = re.compile(r"call:(?P<name>\w+)")
# Quote bare identifier keys that appear right after `{` or `,` and before `:`.
_TC_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_]\w*)(\s*:)")


def _extract_braced(s: str, start_search: int = 0):
    """
    Return the balanced `{...}` substring of `s` at/after `start_search`, treating
    text inside `<|"|>` ... `<|"|>` as opaque (braces there don't count). The
    escape token is its own closer, so it simply toggles an "in string" flag.
    """
    start = s.find("{", start_search)
    if start < 0:
        return None
    depth, in_str, i, n = 0, False, start, len(s)
    while i < n:
        if s.startswith(_TC_ESC, i):
            in_str = not in_str
            i += len(_TC_ESC)
            continue
        c = s[i]
        if not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        i += 1
    return s[start:]  # unbalanced (truncated generation); return what we have


def _toolargs_to_json(body: str) -> dict:
    """
    Convert a Gemma 4 tool-call argument blob into a Python dict.

    The blob uses unquoted keys and `<|"|>`-delimited string values, e.g.
        { location: <|"|>San Francisco, CA<|"|>, unit: <|"|>celsius<|"|>, days: 3 }
    Splitting on the escape token isolates string contents (odd segments), so
    delimiters/commas/braces inside strings can't corrupt the parse.
    """
    parts = body.split(_TC_ESC)
    # Even indices are structural JSON; odd indices are literal string contents.
    rebuilt = []
    for i, seg in enumerate(parts):
        if i % 2 == 1:
            # String literal: emit a properly escaped JSON string.
            rebuilt.append(json.dumps(seg))
        else:
            # Structural: quote bare keys; numbers/true/false/null pass through.
            rebuilt.append(_TC_KEY_RE.sub(r'\1"\2"\3', seg))
    json_str = "".join(rebuilt)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Permissive fallback: treat every value as a string.
        args: dict = {}
        inner = body.strip().lstrip("{").rstrip("}")
        for pair in inner.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                args[k.strip().strip('"')] = v.replace(_TC_ESC, "").strip().strip('"')
        return args


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Parse all Gemma 4 tool calls present in `text` (supports parallel calls)."""
    calls: List[ToolCall] = []
    # Prefer content strictly inside <|tool_call> ... <tool_call|> blocks, but
    # also tolerate a bare `call:NAME{...}` if the markers were stripped upstream.
    blocks = re.findall(re.escape(_TC_OPEN) + r"(.*?)" + re.escape(_TC_CLOSE), text, re.DOTALL)
    search_spaces = blocks if blocks else [text]
    for space in search_spaces:
        for m in _TC_NAME_RE.finditer(space):
            body = _extract_braced(space, m.end())
            if body is None:
                continue
            name = m.group("name")
            args = _toolargs_to_json(body)
            calls.append(
                ToolCall(id=None, type="function",
                         function=ToolFunction(name=name, arguments=json.dumps(args)))
            )
    return calls


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """Parse the first Gemma 4 tool call in `text`, or None. See `parse_tool_calls`."""
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


class TokenGenerator:
    _model: Optional[Transformer] = None
    _tokenizer = None

    def __init__(self, checkpoint: str = Config.checkpoint_path, device: torch.device = Config.device):
        self.device = device

        if TokenGenerator._model is None:
            debug_print(f"Loading Gemma 4 weights from {checkpoint}...")
            start = time.time()
            TokenGenerator._model = Transformer.from_checkpoint(checkpoint, device=device)
            print(f"\u2713 Model weights loaded in {time.time() - start:.2f}s")
        else:
            print("Model weights already loaded. Reusing existing instance.")
        self.model: Transformer = TokenGenerator._model

        if TokenGenerator._tokenizer is None:
            print("Loading tokenizer...")
            TokenGenerator._tokenizer = get_tokenizer(checkpoint)
        self.tokenizer = TokenGenerator._tokenizer

        # Gemma 4 ends an assistant turn with `<turn|>` (eot_token), NOT the
        # legacy `<end_of_turn>`. We stop on `<turn|>` and `<eos>`. `<end_of_turn>`
        # is kept as a defensive fallback in case a build aliases it.
        candidate_stops = [
            self._tok_id("<turn|>"),
            self._tok_id("<end_of_turn>"),
            self.tokenizer.eos_token_id,
            self._tok_id("<eos>"),
        ]
        self.eot_token = candidate_stops[0]
        self.eos_token = self.tokenizer.eos_token_id or self._tok_id("<eos>", default=1)
        self.stop_tokens = sorted({t for t in candidate_stops if t is not None})
        if not self.stop_tokens:
            self.stop_tokens = [1]
        debug_print(f"stop tokens: {self.stop_tokens}")

    def _tok_id(self, piece: str, default: Optional[int] = None) -> Optional[int]:
        tid = self.tokenizer.convert_tokens_to_ids(piece)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if tid is None or tid == unk:
            return default
        return tid

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        temperature: float = Config.temperature,
        top_p: float = Config.top_p,
        max_tokens: int = Config.max_tokens,
        return_logprobs: bool = False,
    ) -> Generator[Union[int, Tuple[int, float]], None, None]:

        cfg = self.model.configs
        stop_tokens = stop_tokens if stop_tokens is not None else self.stop_tokens

        max_gen = max_tokens if max_tokens > 0 else cfg.max_position_embeddings
        cache_size = min(len(prompt_tokens) + max_gen, cfg.max_position_embeddings)

        print("Initialising KV caches...")
        print(f"  - Cache size : {cache_size} tokens")
        print(f"  - Layers     : {cfg.num_hidden_layers} (global: {sum(cfg.is_global(i) for i in range(cfg.num_hidden_layers))})")
        cache_start = time.time()
        caches = self.model.build_caches(batch_size=1, max_len=cache_size, device=self.device)
        print(f"\u2713 Caches initialised in {time.time() - cache_start:.2f}s")

        tokens = list(prompt_tokens)
        print(f"Prompt length: {len(tokens)} tokens")

        # ---- Prefill ----
        prefill_start = time.time()
        input_tensor = torch.as_tensor([tokens], dtype=torch.long, device=self.device)
        logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)
        print(f"\u2713 Prefill complete in {time.time() - prefill_start:.2f}s")

        # ---- Decode ----
        num_generated = 0
        predicted = None
        while max_tokens == 0 or num_generated < max_tokens:
            if num_generated > 0:
                input_tensor = torch.as_tensor([[predicted]], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)

            predicted = self._sample(logits, temperature, top_p)
            tokens.append(predicted)
            num_generated += 1

            if return_logprobs:
                logprob = torch.log_softmax(logits, dim=-1)[predicted].item()
                yield predicted, logprob
            else:
                yield predicted

            if predicted in stop_tokens:
                print("Stop token encountered, ending generation")
                break

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature == 0.0:
            return int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep = cumulative - sorted_probs <= top_p
            sorted_probs = sorted_probs * keep
            sorted_probs = sorted_probs / sorted_probs.sum()
            choice = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_idx[choice].item())
        return int(torch.multinomial(probs, num_samples=1).item())

    @torch.inference_mode()
    def generate_text(
        self,
        prompt_tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        temperature: float = Config.temperature,
        max_tokens: int = Config.max_tokens,
    ) -> GenerationResult:
        out = [t for t in self.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=stop_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            return_logprobs=False,
        )]
        text = self.tokenizer.decode(out, skip_special_tokens=False)
        return GenerationResult(text=text, tool_call=parse_tool_call(text))


