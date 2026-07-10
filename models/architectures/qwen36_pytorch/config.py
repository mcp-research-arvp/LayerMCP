from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "qwen-3.6"
CHECKPOINT_ENV_VAR = "LAYERMCP_QWEN36_CHECKPOINT"


@dataclass
class Config:
    """Runtime configuration for the local Qwen 3.6 token generator."""

    debug_mode: bool = False
    checkpoint_path: str = os.environ.get(
        CHECKPOINT_ENV_VAR,
        str(DEFAULT_CHECKPOINT_PATH),
    )
    device: torch.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    temperature: float = 0.0
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 1024
    use_chat_template: bool = True
