import os

import torch
from safetensors import safe_open


def _to_hf_name(name: str) -> str:
    if name == "embed_tokens.weight":
        return "model.embed_tokens.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name == "lm_head.weight":
        return "lm_head.weight"

    if name.startswith("layers."):
        parts = name.split(".")
        layer_idx = parts[1]
        rest = ".".join(parts[2:])
        return f"model.layers.{layer_idx}.{rest}"

    return name


class Checkpoint:
    def __init__(self, path: str, device: torch.device):
        device_str = device.type if device.index is None else f"{device.type}:{device.index}"
        self.device_str = device_str
        self.tensor_name_to_file: dict[str, str] = {}

        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]
        assert safetensor_files, f"No .safetensors files found in {path}"

        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.tensor_name_to_file[key] = safetensor_file

    def get(self, name: str) -> torch.Tensor | None:
        hf_name = _to_hf_name(name)
        if hf_name not in self.tensor_name_to_file:
            return None

        with safe_open(
            self.tensor_name_to_file[hf_name],
            framework="pt",
            device=self.device_str,
        ) as f:
            return f.get_tensor(hf_name)
