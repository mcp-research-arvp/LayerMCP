import re
import json
import time
import torch
from models.architectures.phi4_pytorch.config import Config
from models.architectures.phi4_pytorch.model import Transformer, Cache
from models.architectures.phi4_pytorch.schemas import (
    ToolCall,
    ToolFunction,
    GenerationResult,
)
from typing import Optional, List, Any, Generator, Union, Tuple
from transformers import AutoTokenizer


def debug_print(*args: Any, **kwargs: Any) -> None:
    """Prints a message only if Config.debug_mode is True."""
    if Config.debug_mode:
        print("[DEBUG]", *args, **kwargs)


def get_tokenizer(checkpoint_path: str):
    """
    Load the Phi-4-multimodal tokenizer that ships alongside the weights.

    Phi-4 uses a standard HF tokenizer (GPT-2 style BPE + a set of added
    special tokens such as <|user|>, <|assistant|>, <|end|>, <|image|>,
    <|audio|>), so we no longer need the GPT-OSS Harmony encoding.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    return tokenizer


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Best-effort tool-call extraction for Phi-4 output.

    Phi-4 emits tool calls as JSON inside the assistant turn (commonly wrapped
    in a ```json fence or a <|tool_call|> tag). We look for the first JSON
    object/array carrying a `name` field and surface it; otherwise return None.
    """
    candidates = []

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates.extend(fenced)

    tagged = re.findall(r"<\|tool_call\|>\s*(.*?)(?:<\|/tool_call\|>|$)", text, re.DOTALL)
    candidates.extend(tagged)

    # Fall back to scanning for a bare JSON object.
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        candidates.append(m.group(1))

    for raw in candidates:
        raw = raw.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else None
        if not isinstance(parsed, dict) or "name" not in parsed:
            continue

        args = parsed.get("arguments", parsed.get("parameters", {}))
        if not isinstance(args, str):
            args = json.dumps(args)
        return ToolCall(
            id=None,
            type="function",
            function=ToolFunction(name=parsed["name"], arguments=args),
        )

    return None


class TokenGenerator:
    _model: Optional[Transformer] = None
    _tokenizer = None

    def __init__(
        self,
        checkpoint: str = Config.checkpoint_path,
        device: torch.device = Config.device,
    ):
        self.device = device
        self.checkpoint = checkpoint

        if TokenGenerator._model is None:
            debug_print(f"Loading model weights from {checkpoint}...")
            start = time.time()
            TokenGenerator._model = Transformer.from_checkpoint(checkpoint, device=self.device)
            print(f"✓ Model weights loaded in {time.time() - start:.2f}s")
        else:
            print("Model weights already loaded. Reusing existing instance.")

        self.model: Transformer = TokenGenerator._model

        if TokenGenerator._tokenizer is None:
            print("Loading tokenizer...")
            TokenGenerator._tokenizer = get_tokenizer(checkpoint)

        self.tokenizer = TokenGenerator._tokenizer

        # Derive stop tokens from the tokenizer + model config rather than
        # hardcoding, so this works across Phi-4 variants (the multimodal model
        # uses <|end|>/<|endoftext|> = 200020/199999, while plain Phi-4 uses the
        # ChatML tokens <|im_end|>/<|endoftext|> = 100265/100257).
        stop_ids = set()

        def _add(value):
            if isinstance(value, int) and value >= 0:
                stop_ids.add(value)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    _add(v)

        _add(getattr(self.tokenizer, "eos_token_id", None))
        _add(getattr(self.model.configs, "eos_token_id", None))

        # Supplement with any recognised end-of-turn markers that exist in the vocab.
        unk = getattr(self.tokenizer, "unk_token_id", None)
        for piece in ("<|im_end|>", "<|end|>", "<|endoftext|>", "<|eot_id|>"):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(piece)
            except Exception:
                tid = None
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                stop_ids.add(tid)

        self.stop_tokens = sorted(stop_ids)
        # Backwards-compatible alias used elsewhere in the codebase.
        self.eot_token = self.stop_tokens[0] if self.stop_tokens else None

        debug_print(f"Stop tokens: {self.stop_tokens}")

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        temperature: float = Config.temperature,
        max_tokens: int = Config.max_tokens,
        return_logprobs: bool = False,
    ) -> Generator[Union[int, Tuple[int, float]], None, None]:

        batch_size = 1
        model_configs = self.model.configs

        # Determine max generation tokens, using RoPE context limit if max_tokens is 0
        max_gen_tokens = max_tokens if max_tokens > 0 else (
            int(model_configs.initial_context_length * int(model_configs.rope_scaling_factor))
        )
        total_tokens = len(prompt_tokens) + max_gen_tokens
        cache_size = min(
            total_tokens,
            int(model_configs.initial_context_length * int(model_configs.rope_scaling_factor)),
        )

        # --- Cache Initialisation ---
        print(f"Initialising KV caches...")
        print(f"  - Cache size      : {cache_size} tokens")
        print(f"  - Number of layers: {model_configs.num_hidden_layers}")
        print(f"  - KV heads        : {model_configs.num_key_value_heads}")
        print(f"  - Head dim        : {model_configs.head_dim}")

        cache_start = time.time()
        caches = [
            Cache(
                batch_size=batch_size,
                n_ctx=cache_size,
                n_kv_heads=model_configs.num_key_value_heads,
                d_head=model_configs.head_dim,
                device=self.device,
            )
            for _ in range(model_configs.num_hidden_layers)
        ]
        print(f"✓ Caches initialised in {time.time() - cache_start:.2f}s")

        tokens = list(prompt_tokens)
        print(f"Prompt length: {len(tokens)} tokens")

        # --- Prefill Phase ---
        print(f"Starting prefill phase (processing {len(tokens)} tokens)...")
        prefill_start = time.time()
        input_tensor = torch.as_tensor([tokens], dtype=torch.long, device=self.device)
        debug_print(f"  - Input tensor shape: {input_tensor.shape}")

        logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)
        print(f"✓ Prefill complete in {time.time() - prefill_start:.2f}s")

        # --- Generation Phase (Decoding loop) ---
        print(f"Starting decoding phase (max {max_tokens} tokens)...")
        num_generated_tokens = 0
        is_debugging = Config.debug_mode
        predicted_token = None

        while max_tokens == 0 or num_generated_tokens < max_tokens:
            iter_start = time.time()

            if num_generated_tokens > 0:
                input_tensor = torch.as_tensor(
                    [[predicted_token]], dtype=torch.long, device=self.device
                )
                logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)

            # Sample next token
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                if is_debugging:
                    print(
                        f"[DEBUG] Token {num_generated_tokens}: {predicted_token} "
                        f"({time.time() - iter_start:.4f}s)"
                    )
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                print("Stop token encountered, ending generation")
                break

            if max_tokens > 0 and num_generated_tokens >= max_tokens:
                break

    @torch.inference_mode()
    def generate_text(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        temperature: float = Config.temperature,
        max_tokens: int = Config.max_tokens,
    ) -> GenerationResult:
        out = []
        for token in self.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=stop_tokens,
            temperature=temperature,
            max_tokens=max_tokens,
            return_logprobs=False,
        ):
            out.append(token)
        text = self.tokenizer.decode(out, skip_special_tokens=True)
        tool = parse_tool_call(text)
        return GenerationResult(text=text, tool_call=tool)
