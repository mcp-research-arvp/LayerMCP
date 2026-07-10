import os
from pathlib import Path

import torch
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "gpt-oss-20b" / "original"
CHECKPOINT_ENV_VAR = "LAYERMCP_GPT_OSS_CHECKPOINT"

@dataclass()
class Config:
    """
    Centralised configuration for the token generator.
    """
    debug_mode: bool = True
    checkpoint_path: str = os.environ.get(
        CHECKPOINT_ENV_VAR,
        str(DEFAULT_CHECKPOINT_PATH),
    )
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    temperature: float = 0.1
    max_tokens: int = 4096  # if set to 0 -> full max context length
