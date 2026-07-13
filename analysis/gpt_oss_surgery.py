from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def freeze_all_layers(model: nn.Module) -> None:
    freeze_module(model)


def unfreeze_gpt_oss_layers(model: nn.Module, layer_ids: Iterable[int]) -> None:
    for layer_id in layer_ids:
        unfreeze_module(model.block[layer_id])


def freeze_except_gpt_oss_layers(model: nn.Module, layer_ids: Iterable[int]) -> None:
    freeze_all_layers(model)
    unfreeze_gpt_oss_layers(model, layer_ids)


def zero_gpt_oss_mlp_expert(model: nn.Module, layer_id: int, expert_id: int) -> None:
    mlp = model.block[layer_id].mlp
    with torch.no_grad():
        mlp.mlp1_weight[expert_id].zero_()
        mlp.mlp1_bias[expert_id].zero_()
        mlp.mlp2_weight[expert_id].zero_()
        mlp.mlp2_bias[expert_id].zero_()


def zero_gpt_oss_attention_layer(model: nn.Module, layer_id: int) -> None:
    attn = model.block[layer_id].attn
    with torch.no_grad():
        for parameter in attn.parameters():
            parameter.zero_()


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

