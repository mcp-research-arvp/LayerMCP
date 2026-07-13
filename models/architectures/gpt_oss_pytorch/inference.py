import re
import json
import time
import torch
from models.architectures.gpt_oss_pytorch.config import Config
from models.architectures.gpt_oss_pytorch.model import Transformer, Cache
from models.architectures.gpt_oss_pytorch.schemas import (
    GenerationResult,
    ToolCall,
    ToolFunction,
)
from typing import Optional, List, Any, Generator, Union, Tuple
from openai_harmony import load_harmony_encoding, HarmonyEncodingName

def debug_print(*args: Any, **kwargs: Any) -> None:
    """Prints a message only if Config.debug_mode is True."""
    if Config.debug_mode:
        print("[DEBUG]", *args, **kwargs)

def get_tokenizer():
    """
    Loads the official Harmony tokenizer for GPT-OSS models instantly.
    """
    tokenizer = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return tokenizer

# ------------------------------------------------------------------ #
# parse_tool_call is defined BEFORE TokenGenerator so generate_text   #
# can reference it without relying on forward-declaration timing.      #
# ------------------------------------------------------------------ #

def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Parse a Harmony-format tool call from generated text.

    Expected format emitted by the model:
        to=tool_name<|message|>{"arg": "value"}<|call|>
    """
    m = re.search(r"to=([a-zA-Z0-9_\-]+)", text)
    if m is None:
        return None
    name = m.group(1)

    m2 = re.search(r"<\|message\|>\s*(.*?)\s*<\|call\|>", text, re.DOTALL)
    args: dict = {}
    if m2:
        raw = m2.group(1).strip()
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            args = {}

    return ToolCall(
        id=None,
        type="function",
        function=ToolFunction(name=name, arguments=json.dumps(args)),
    )


class TokenGenerator:
    _model: Optional[Transformer] = None
    _tokenizer = None
    _eot_token: Optional[int] = None

    def __init__(
        self,
        checkpoint: str = Config.checkpoint_path,
        device: torch.device = Config.device,
    ):
        self.device = device

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
            TokenGenerator._tokenizer = get_tokenizer()
            tok = TokenGenerator._tokenizer

            TokenGenerator._eot_token = tok.encode("<|return|>", allowed_special="all")[0]
            TokenGenerator._call_token = tok.encode("<|call|>", allowed_special="all")[0]
            TokenGenerator._end_token = tok.encode("<|end|>", allowed_special="all")[0]
            TokenGenerator._return_token = tok.encode("<|return|>", allowed_special="all")[0]

            debug_print("Special tokens:")
            debug_print(f"call   : {TokenGenerator._call_token}")
            debug_print(f"end    : {TokenGenerator._end_token}")
            debug_print(f"return : {TokenGenerator._return_token}")

        self.tokenizer = TokenGenerator._tokenizer
        self.eot_token = TokenGenerator._eot_token
        self.call_token = TokenGenerator._call_token
        self.end_token = TokenGenerator._end_token
        self.return_token = TokenGenerator._return_token

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
            model_configs.initial_context_length * int(model_configs.rope_scaling_factor)
        )
        total_tokens = len(prompt_tokens) + max_gen_tokens
        cache_size = min(
            total_tokens,
            model_configs.initial_context_length * int(model_configs.rope_scaling_factor),
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
        predicted_token = None  # satisfies linter; set before first use below

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
        text = self.tokenizer.decode(out)
        tool = parse_tool_call(text)
        return GenerationResult(text=text, tool_call=tool)
