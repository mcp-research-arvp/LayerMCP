"""
Minimal safetensors loader for Qwen3.5 checkpoints.

Unlike GPT-OSS (which ships MXFP4-quantised MoE weights), Qwen3.5 dense
checkpoints store plain bf16/fp16 tensors, so this is just a thin index over
the `*.safetensors` shards that returns tensors by name.
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

        # Map every tensor name to the shard file that contains it.
        tensor_name_to_file: dict[str, str] = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_name_to_file[key] = safetensor_file

        self.tensor_name_to_file = tensor_name_to_file

    def keys(self):
        return set(self.tensor_name_to_file.keys())

    def __contains__(self, name: str) -> bool:
        return name in self.tensor_name_to_file

    def get(self, name: str) -> torch.Tensor:
        if name not in self.tensor_name_to_file:
            raise KeyError(f"Tensor {name} not found in checkpoint.")
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)


