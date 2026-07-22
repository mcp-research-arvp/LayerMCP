import re
import json
import time
import torch
from .config import Config
from .model import Transformer, build_caches
from .schemas import ToolCall, ToolFunction, GenerationResult
from typing import Optional, List, Any, Generator, Union, Tuple


def debug_print(*args: Any, **kwargs: Any) -> None:
    if Config.debug_mode:
        print("[DEBUG]", *args, **kwargs)


def get_tokenizer(checkpoint: str):
    """
    Load the Qwen3.5 tokenizer. Prefers HF `AutoTokenizer`; falls back to the
    raw `tokenizers` library reading `tokenizer.json` from the checkpoint.
    """
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - fallback path
        debug_print(f"AutoTokenizer unavailable ({e}); falling back to tokenizers.Tokenizer")
        import os
        from tokenizers import Tokenizer
        return Tokenizer.from_file(os.path.join(checkpoint, "tokenizer.json"))


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Parse a Qwen-style tool call:
        <tool_call>{"name": "fn", "arguments": {...}}</tool_call>
    """
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if m is None:
        return None
    try:
        payload = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None

    name = payload.get("name")
    if not name:
        return None
    args = payload.get("arguments", {})
    return ToolCall(
        id=None,
        type="function",
        function=ToolFunction(name=name, arguments=json.dumps(args)),
    )


def _coerce_ids(out) -> List[int]:
    """Normalise any tokenizer/processor output to a flat list[int]."""
    # tokenizers.Encoding
    if hasattr(out, "ids"):
        out = out.ids
    # BatchEncoding / dict
    if isinstance(out, dict):
        out = out["input_ids"]
    # torch tensor / numpy
    if hasattr(out, "tolist"):
        out = out.tolist()
    # batched -> take first row
    if len(out) > 0 and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def _encode(tokenizer, text: str) -> List[int]:
    return _coerce_ids(tokenizer.encode(text))


def _decode(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids)


class TokenGenerator:
    _model: Optional[Transformer] = None
    _tokenizer = None
    _stop_tokens: Optional[List[int]] = None

    def __init__(
        self,
        checkpoint: str = Config.checkpoint_path,
        device: torch.device = Config.device,
    ):
        self.device = device

        if TokenGenerator._model is None:
            debug_print(f"Loading model weights from {checkpoint}...")
            start = time.time()
            TokenGenerator._model = Transformer.from_checkpoint(checkpoint, device=device)
            print(f"\u2713 Model weights loaded in {time.time() - start:.2f}s")
        else:
            print("Model weights already loaded. Reusing existing instance.")
        self.model: Transformer = TokenGenerator._model

        if TokenGenerator._tokenizer is None:
            print("Loading tokenizer...")
            TokenGenerator._tokenizer = get_tokenizer(checkpoint)
            tok = TokenGenerator._tokenizer

            # Resolve stop tokens (EOS + <|im_end|> if present).
            stop: List[int] = []
            eos = getattr(tok, "eos_token_id", None)
            if isinstance(eos, int):
                stop.append(eos)
            for special in ("<|im_end|>", "<|endoftext|>"):
                try:
                    tid = tok.convert_tokens_to_ids(special) if hasattr(tok, "convert_tokens_to_ids") \
                        else tok.token_to_id(special)
                    if isinstance(tid, int) and tid >= 0 and tid not in stop:
                        stop.append(tid)
                except Exception:
                    pass
            TokenGenerator._stop_tokens = stop or [0]
            debug_print(f"Stop tokens: {TokenGenerator._stop_tokens}")

        self.tokenizer = TokenGenerator._tokenizer
        self.stop_tokens = TokenGenerator._stop_tokens
        # Backwards-compatible alias used by run_batch.py
        self.eot_token = self.stop_tokens[0]

    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> List[int]:
        return _encode(self.tokenizer, text)

    def apply_chat_template(
        self,
        prompt: str,
        tools: Optional[List[dict[str, Any]]] = None,
    ) -> List[int]:
        """Wrap a user prompt with the Qwen chat template when available."""
        tok = self.tokenizer
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tok, "apply_chat_template"):
            try:
                template_kwargs = {
                    "add_generation_prompt": True,
                    "tokenize": True,
                }
                if tools:
                    template_kwargs["tools"] = tools
                out = tok.apply_chat_template(messages, **template_kwargs)
                return _coerce_ids(out)
            except Exception as e:
                debug_print(f"chat template failed ({e}); using raw encode")
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return self.encode(text)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        temperature: float = Config.temperature,
        top_p: float = Config.top_p,
        top_k: int = Config.top_k,
        max_tokens: int = Config.max_tokens,
        return_logprobs: bool = False,
    ) -> Generator[Union[int, Tuple[int, float]], None, None]:

        cfg = self.model.configs
        stop_tokens = stop_tokens if stop_tokens is not None else self.stop_tokens

        max_gen_tokens = max_tokens if max_tokens > 0 else cfg.max_position_embeddings
        cache_size = min(len(prompt_tokens) + max_gen_tokens, cfg.max_position_embeddings)

        print("Initialising hybrid caches...")
        print(f"  - Cache size      : {cache_size} tokens")
        print(f"  - Number of layers: {cfg.num_hidden_layers}")
        n_full = sum(t == "full_attention" for t in cfg.layer_types)
        print(f"  - Full-attn layers: {n_full}  |  Linear-attn layers: {cfg.num_hidden_layers - n_full}")

        cache_start = time.time()
        param_dtype = next(self.model.parameters()).dtype
        caches = build_caches(cfg, batch_size=1, n_ctx=cache_size, device=self.device, dtype=param_dtype)
        print(f"\u2713 Caches initialised in {time.time() - cache_start:.2f}s")

        tokens = list(prompt_tokens)
        print(f"Prompt length: {len(tokens)} tokens")

        # --- Prefill ---
        print(f"Starting prefill phase (processing {len(tokens)} tokens)...")
        prefill_start = time.time()
        input_tensor = torch.as_tensor([tokens], dtype=torch.long, device=self.device)
        position_ids = torch.arange(len(tokens), device=self.device)[None, :]
        logits = self.model(input_tensor, caches=caches, position_ids=position_ids)[:, -1, :].squeeze(0)
        print(f"\u2713 Prefill complete in {time.time() - prefill_start:.2f}s")

        # --- Decode ---
        print(f"Starting decoding phase (max {max_tokens} tokens)...")
        num_generated = 0
        cur_pos = len(tokens)
        predicted_token = None

        while max_tokens == 0 or num_generated < max_tokens:
            iter_start = time.time()

            if num_generated > 0:
                input_tensor = torch.as_tensor([[predicted_token]], dtype=torch.long, device=self.device)
                position_ids = torch.as_tensor([[cur_pos]], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor, caches=caches, position_ids=position_ids)[:, -1, :].squeeze(0)
                cur_pos += 1

            predicted_token = self._sample(logits, temperature, top_p, top_k)

            tokens.append(predicted_token)
            num_generated += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                selected = logprobs[predicted_token].item()
                if Config.debug_mode:
                    print(f"[DEBUG] Token {num_generated}: {predicted_token} ({time.time() - iter_start:.4f}s)")
                yield predicted_token, selected
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                print("Stop token encountered, ending generation")
                break

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> int:
        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits.float() / temperature

        if top_k and top_k > 0:
            k = min(top_k, logits.shape[-1])
            kth = torch.topk(logits, k).values[..., -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

        probs = torch.softmax(logits, dim=-1)

        if top_p and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumsum - sorted_probs > top_p
            sorted_probs[cutoff] = 0.0
            sorted_probs /= sorted_probs.sum()
            choice = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_idx[choice].item())

        return int(torch.multinomial(probs, num_samples=1).item())

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate_text(
        self,
        prompt_tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        temperature: float = Config.temperature,
        max_tokens: int = Config.max_tokens,
    ) -> GenerationResult:
        out: List[int] = []
        for token in self.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=stop_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            return_logprobs=False,
        ):
            out.append(token)
        text = _decode(self.tokenizer, out)
        return GenerationResult(text=text, tool_call=parse_tool_call(text))
