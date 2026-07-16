from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "gemma-4"
CHECKPOINT_ENV_VAR = "LAYERMCP_GEMMA4_CHECKPOINT"


@dataclass
class Config:
    """Runtime configuration for the local Gemma 4 token generator."""

    debug_mode: bool = False
    checkpoint_path: str = os.environ.get(
        CHECKPOINT_ENV_VAR,
        str(DEFAULT_CHECKPOINT_PATH),
    )
    model_id: str = "google/gemma-4-26b-a4b-it"
    device: torch.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 1024
