from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def encode_choice(tokenizer: Any, choice: str) -> tuple[int, ...]:
    """Encode a plain choice across HF, tokenizers, and Harmony tokenizers."""
    try:
        encoded = tokenizer.encode(choice, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(choice)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    return tuple(int(token) for token in encoded)


class ChoiceConstraint:
    """Tracks the valid next tokens for a finite catalog of strings."""

    def __init__(self, tokenizer: Any, choices: Sequence[str], stop_token: int):
        if not choices:
            raise ValueError("choices must not be empty.")
        self.stop_token = stop_token
        self.token_sequences = {
            choice: encode_choice(tokenizer, choice) for choice in choices
        }
        if any(not tokens for tokens in self.token_sequences.values()):
            raise ValueError("choices must encode to at least one token.")

    def allowed_tokens(self, generated: tuple[int, ...]) -> set[int]:
        allowed: set[int] = set()
        for tokens in self.token_sequences.values():
            if tokens[: len(generated)] != generated:
                continue
            if len(generated) == len(tokens):
                allowed.add(self.stop_token)
            else:
                allowed.add(tokens[len(generated)])
        return allowed

    def resolve(self, generated: Sequence[int]) -> str:
        tokens = tuple(generated)
        if tokens and tokens[-1] == self.stop_token:
            tokens = tokens[:-1]
        for choice, candidate in self.token_sequences.items():
            if tokens == candidate:
                return choice
        raise RuntimeError("Constrained decoding did not produce a catalog choice.")


def constrained_argmax(logits: torch.Tensor, allowed_tokens: set[int]) -> int:
    if not allowed_tokens:
        raise RuntimeError("Token constraint returned no valid continuation.")
    allowed = torch.as_tensor(
        sorted(allowed_tokens), dtype=torch.long, device=logits.device
    )
    index = torch.argmax(logits.index_select(0, allowed), dim=-1)
    return int(allowed[index].item())


def generate_choice(
    generator: Any,
    prompt_tokens: list[int],
    choices: Sequence[str],
    stop_token: int,
) -> str:
    constraint = ChoiceConstraint(generator.tokenizer, choices, stop_token)
    generated = list(
        generator.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=[stop_token],
            temperature=0.0,
            max_tokens=max(len(tokens) for tokens in constraint.token_sequences.values()) + 1,
            return_logprobs=False,
            allowed_tokens_fn=constraint.allowed_tokens,
        )
    )
    return constraint.resolve(generated)
