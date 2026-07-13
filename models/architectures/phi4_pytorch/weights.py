"""
Checkpoint loader for Phi-4-multimodal.

The original GPT-OSS loader dequantized MXFP4-packed MoE weights. Phi-4 ships
as plain bf16 `.safetensors`, so this version just indexes every tensor across
the shards and returns them as-is. Name mapping (HF <-> local module names) is
handled by the caller in `model.py`.
"""

import os
import torch
from safetensors import safe_open


class Checkpoint:
    def __init__(self, path: str, device: torch.device):
        device_str = (
            device.type if device.index is None else f"{device.type}:{device.index}"
        )
        self.device_str = device_str

        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")

        # Map every tensor name to the shard file that holds it.
        tensor_name_to_file = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device=device_str) as f:
                for key in f.keys():
                    tensor_name_to_file[key] = safetensor_file

        self.tensor_name_to_file = tensor_name_to_file

    def has(self, name: str) -> bool:
        return name in self.tensor_name_to_file

    def get(self, name: str) -> torch.Tensor:
        if name not in self.tensor_name_to_file:
            raise KeyError(f"Tensor {name} not found in checkpoint.")
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)

    def keys(self):
        return self.tensor_name_to_file.keys()
