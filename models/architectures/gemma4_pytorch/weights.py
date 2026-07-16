"""
Checkpoint loader for the official Gemma 4 bf16 safetensors checkpoint
(`google/gemma-4-26B-A4B-it`).

Unlike the original GPT-OSS loader, Gemma 4's public checkpoint is NOT in the
MXFP4 block/scale format - it is plain bf16. So this file is a thin wrapper
around `safetensors.safe_open` with no dequantisation. (If you instead grab a
quantised variant such as the NVFP4 or GGUF checkpoints, this loader will not
work as-is - those need their own dequant paths.)

Gemma 4 stores all decoder weights under the `model.language_model.` prefix,
and the MoE experts are stored as *stacked* tensors:
    model.language_model.layers.{i}.experts.gate_up_proj   # (num_experts, ...)
    model.language_model.layers.{i}.experts.down_proj      # (num_experts, ...)
"""

import os
import torch
from safetensors import safe_open


class Checkpoint:
    def __init__(self, path: str, device: torch.device):
        device_str = (
            device.type if device.index is None
            else f"{device.type}:{device.index}"
        )
        self.device_str = device_str
        self.path = path

        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]
        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {path!r}. Download the bf16 "
                f"checkpoint first, e.g. `huggingface-cli download "
                f"google/gemma-4-26B-A4B-it --local-dir {path}`."
            )

        # Map every tensor name to the shard file that contains it.
        tensor_name_to_file: dict[str, str] = {}
        for f_path in safetensor_files:
            with safe_open(f_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_name_to_file[key] = f_path
        self.tensor_name_to_file = tensor_name_to_file

    def has(self, name: str) -> bool:
        return name in self.tensor_name_to_file

    def keys(self):
        return self.tensor_name_to_file.keys()

    def get(self, name: str) -> torch.Tensor:
        """Return the named tensor (bf16) placed on the target device."""
        if name not in self.tensor_name_to_file:
            raise KeyError(f"Tensor {name!r} not found in checkpoint at {self.path!r}.")
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)


