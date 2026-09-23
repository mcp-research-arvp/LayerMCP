from __future__ import annotations

import unittest

import torch
from torch import nn

from research.phase2.intervention import (
    AttentionInterventionError,
    attention_module_for_layer,
    available_attention_layers,
    temporary_attention_intervention,
)
from research.phase2.intervention_smoke import run_generation_pair


class TinyDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3, bias=False))
        self.mlp = nn.Linear(3, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.self_attn(values))


class TinyLlamaLayout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyDecoderLayer(), TinyDecoderLayer()])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return values


class TinyHuggingFaceLayout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyLlamaLayout()


class TinyGptOssBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(3, 3)
        self.mlp = nn.Linear(3, 3)


class TinyGptOssLayout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList([TinyGptOssBlock(), TinyGptOssBlock()])


class TinyQwenLayer(nn.Module):
    def __init__(self, *, full_attention: bool) -> None:
        super().__init__()
        if full_attention:
            self.self_attn = nn.Linear(3, 3)
        else:
            self.linear_attn = nn.Linear(3, 3)
        self.mlp = nn.Linear(3, 3)


class TinyQwenLayout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [TinyQwenLayer(full_attention=False), TinyQwenLayer(full_attention=True)]
        )


class TinySmokeGenerator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.stop_tokens: list[int] = []

    def generate(self, prompt_tokens, stop_tokens, temperature, max_tokens):
        self.model(torch.ones(1, 3))
        yield 1


def _parameter_values(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


class Phase2AttentionInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.model = TinyLlamaLayout()

    def test_discovers_only_the_model_supplied_layer_indices(self) -> None:
        self.assertEqual(available_attention_layers(self.model), (0, 1))
        self.assertIs(attention_module_for_layer(self.model, 1), self.model.layers[1].self_attn)

    def test_supported_repository_layouts_are_identified(self) -> None:
        self.assertEqual(available_attention_layers(TinyHuggingFaceLayout()), (0, 1))
        self.assertEqual(available_attention_layers(TinyGptOssLayout()), (0, 1))
        self.assertEqual(available_attention_layers(TinyQwenLayout()), (0, 1))

    def test_local_llama_transformer_layout_is_supported_without_a_checkpoint(self) -> None:
        from models.architectures.llama31_8b_pytorch.model import ModelConfigs, Transformer

        model = Transformer(
            ModelConfigs(
                vocab_size=8,
                num_hidden_layers=2,
                hidden_size=4,
                intermediate_size=8,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                max_position_embeddings=4,
            ),
            device=torch.device("cpu"),
        )
        self.assertEqual(available_attention_layers(model), (0, 1))
        self.assertIs(attention_module_for_layer(model, 0), model.layers[0].self_attn)

    def test_disabled_intervention_is_a_true_noop(self) -> None:
        values = torch.ones(1, 3)
        before = _parameter_values(self.model)
        baseline = self.model(values).detach().clone()

        with temporary_attention_intervention(
            self.model, 0, method="noise", seed=7, strength=3.0, enabled=False
        ) as intervention:
            actual = self.model(values)

        self.assertTrue(torch.equal(baseline, actual))
        self.assertEqual(intervention.changed_parameter_names, ())
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))

    def test_only_selected_attention_parameters_change(self) -> None:
        before = _parameter_values(self.model)

        with temporary_attention_intervention(self.model, 1, method="noise", seed=2, strength=1.0):
            changed = _parameter_values(self.model)

            for name, value in changed.items():
                if name.startswith("layers.1.self_attn."):
                    self.assertFalse(torch.equal(value, before[name]), name)
                else:
                    self.assertTrue(torch.equal(value, before[name]), name)

    def test_noise_and_replace_follow_the_documented_rules(self) -> None:
        model = TinyLlamaLayout()
        original = _parameter_values(model)
        strength = 0.25
        generator = torch.Generator(device="cpu").manual_seed(19)
        expected_noise: dict[str, torch.Tensor] = {}
        expected_replace: dict[str, torch.Tensor] = {}
        for name, parameter in model.layers[0].self_attn.named_parameters():
            z = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
            expected_noise[name] = original[f"layers.0.self_attn.{name}"] + z * strength

        with temporary_attention_intervention(model, 0, method="noise", seed=19, strength=strength):
            for name, parameter in model.layers[0].self_attn.named_parameters():
                torch.testing.assert_close(parameter, expected_noise[name])

        generator.manual_seed(19)
        for name, parameter in model.layers[0].self_attn.named_parameters():
            expected_replace[name] = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            ) * strength
        with temporary_attention_intervention(model, 0, method="replace", seed=19, strength=strength):
            for name, parameter in model.layers[0].self_attn.named_parameters():
                torch.testing.assert_close(parameter, expected_replace[name])

    def test_seeded_interventions_are_reproducible(self) -> None:
        original = _parameter_values(self.model)
        same_seed: dict[str, torch.Tensor]
        with temporary_attention_intervention(self.model, 0, method="noise", seed=11, strength=0.5):
            same_seed = _parameter_values(self.model)
        with temporary_attention_intervention(self.model, 0, method="noise", seed=11, strength=0.5):
            again = _parameter_values(self.model)
        with temporary_attention_intervention(self.model, 0, method="noise", seed=12, strength=0.5):
            different = _parameter_values(self.model)

        for name in same_seed:
            torch.testing.assert_close(same_seed[name], again[name])
        self.assertTrue(
            any(
                not torch.equal(same_seed[name], different[name])
                for name in original
                if name.startswith("layers.0.self_attn.")
            )
        )

    def test_preserves_parameter_shapes_devices_and_dtypes(self) -> None:
        model = TinyLlamaLayout().to(dtype=torch.float64)
        before = {
            name: (tuple(parameter.shape), parameter.device, parameter.dtype)
            for name, parameter in model.layers[0].self_attn.named_parameters()
        }
        with temporary_attention_intervention(model, 0, method="replace", seed=2, strength=0.5):
            for name, parameter in model.layers[0].self_attn.named_parameters():
                self.assertEqual((tuple(parameter.shape), parameter.device, parameter.dtype), before[name])

    def test_restores_exact_values_after_exception(self) -> None:
        before = _parameter_values(self.model)
        intervention = temporary_attention_intervention(
            self.model, 0, method="replace", seed=9, strength=1.0
        )

        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with intervention:
                raise RuntimeError("intentional failure")

        self.assertTrue(intervention.restored_exactly)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_invalid_layer_and_unsupported_structures_explain_the_problem(self) -> None:
        with self.assertRaisesRegex(AttentionInterventionError, "Valid layer indices for this model: 0, 1"):
            with temporary_attention_intervention(self.model, 4):
                pass

        unsupported = nn.Module()
        unsupported.layers = nn.ModuleList([nn.Linear(3, 3)])
        with self.assertRaisesRegex(AttentionInterventionError, "No identifiable attention modules"):
            with temporary_attention_intervention(unsupported, 0):
                pass

    def test_quantized_models_are_rejected_before_mutation(self) -> None:
        self.model.is_loaded_in_4bit = True
        before = _parameter_values(self.model)
        with self.assertRaisesRegex(AttentionInterventionError, "Quantized models are not supported"):
            with temporary_attention_intervention(self.model, 0):
                pass
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))

    def test_shared_attention_parameter_is_rejected_before_mutation(self) -> None:
        self.model.layers[0].mlp.weight = self.model.layers[0].self_attn[0].weight
        before = _parameter_values(self.model)
        with self.assertRaisesRegex(AttentionInterventionError, "tied attention parameter"):
            with temporary_attention_intervention(self.model, 0):
                pass
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))

    def test_smoke_pair_verifies_the_live_generation_boundary(self) -> None:
        result = run_generation_pair(
            TinySmokeGenerator(self.model),
            (1, 2, 3),
            layer_index=0,
            temperature=0.0,
            max_tokens=1,
        )
        self.assertEqual(result["disabled_token_ids"], [1])
        self.assertEqual(result["noise_token_ids"], [1])
        self.assertTrue(result["verification"]["only_selected_attention_parameters_changed"])
        self.assertTrue(result["verification"]["all_selected_attention_parameter_values_changed"])
        self.assertTrue(result["verification"]["restored_exactly"])
