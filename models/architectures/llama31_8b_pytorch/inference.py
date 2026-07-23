import time
from typing import Any, Generator, Optional, Union, Tuple

import torch
from transformers import AutoTokenizer

from models.architectures.llama31_8b_pytorch.config import Config
from models.architectures.llama31_8b_pytorch.model import Cache, Transformer


def debug_print(*args: Any, **kwargs: Any) -> None:
    if Config.debug_mode:
        print("[DEBUG]", *args, **kwargs)


def get_tokenizer(checkpoint: str):
    return AutoTokenizer.from_pretrained(checkpoint)


class TokenGenerator:
    _model: Optional[Transformer] = None
    _tokenizer = None

    def __init__(
        self,
        checkpoint: str = Config.checkpoint_path,
        device: torch.device = Config.device,
    ):
        self.checkpoint = checkpoint
        self.device = device

        if TokenGenerator._model is None:
            debug_print(f"Loading model weights from {checkpoint}...")
            start = time.time()
            TokenGenerator._model = Transformer.from_checkpoint(checkpoint, device=self.device)
            print(f"Model weights loaded in {time.time() - start:.2f}s")
        else:
            print("Model weights already loaded. Reusing existing instance.")
        self.model: Transformer = TokenGenerator._model

        if TokenGenerator._tokenizer is None:
            print("Loading tokenizer...")
            TokenGenerator._tokenizer = get_tokenizer(checkpoint)
        self.tokenizer = TokenGenerator._tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_tokens = self._resolve_stop_tokens()

    def _resolve_stop_tokens(self) -> list[int]:
        stop_tokens: list[int] = []
        if isinstance(self.eos_token_id, int):
            stop_tokens.append(self.eos_token_id)
        elif isinstance(self.eos_token_id, (list, tuple)):
            stop_tokens.extend(
                token_id
                for token_id in self.eos_token_id
                if isinstance(token_id, int) and token_id not in stop_tokens
            )

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        for token in ("<|eot_id|>", "<|eom_id|>"):
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if (
                isinstance(token_id, int)
                and token_id >= 0
                and token_id != unk_token_id
                and token_id not in stop_tokens
            ):
                stop_tokens.append(token_id)
        return stop_tokens

    def encode_chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if tools:
            template_kwargs["tools"] = tools
        try:
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except (TypeError, ValueError):
            template_kwargs.pop("tools", None)
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        encoded = self.tokenizer(prompt, return_tensors=None)["input_ids"]
        if hasattr(encoded, "keys") and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return encoded

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int] | None = None,
        temperature: float = Config.temperature,
        max_tokens: int = Config.max_tokens,
        return_logprobs: bool = False,
    ) -> Generator[Union[int, Tuple[int, float]], None, None]:
        stop_tokens = stop_tokens or self.stop_tokens
        batch_size = 1
        model_configs = self.model.configs
        max_gen_tokens = max_tokens if max_tokens > 0 else model_configs.max_position_embeddings
        cache_size = min(len(prompt_tokens) + max_gen_tokens, model_configs.max_position_embeddings)

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

        input_tensor = torch.as_tensor([prompt_tokens], dtype=torch.long, device=self.device)
        logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)
        predicted_token = None

        for _ in range(max_gen_tokens):
            if predicted_token is not None:
                input_tensor = torch.as_tensor([[predicted_token]], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)

            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                yield predicted_token, logprobs[predicted_token].item()
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break

    @torch.inference_mode()
    def generate_text(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int] | None = None,
        temperature: float = Config.temperature,
        max_tokens: int = Config.max_tokens,
    ) -> str:
        out = list(
            self.generate(
                prompt_tokens=prompt_tokens,
                stop_tokens=stop_tokens,
                temperature=temperature,
                max_tokens=max_tokens,
                return_logprobs=False,
            )
        )
        return self.tokenizer.decode(out, skip_special_tokens=False)
