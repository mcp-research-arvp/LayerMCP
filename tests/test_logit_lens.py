from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from analysis.logit_lens import (
    aggregate_summary,
    build_forced_choice_prompt,
    build_tool_label_mapping,
    compute_layer_metrics,
    get_label_token_ids,
    group_layer_margins,
    load_benchmark,
    make_output_paths,
    select_strongest_successes,
    select_weakest_cases,
    summarize_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            " A": 101,
            " B": 102,
            " C": 103,
            " D": 104,
            "A": 201,
            "B": 202,
            "C": 203,
            "D": 204,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.vocab:
            return [self.vocab[text]]
        return [999, 1000]


class LogitLensHelperTests(unittest.TestCase):
    def test_loads_phase2_seed_benchmark(self) -> None:
        samples = load_benchmark(
            PROJECT_ROOT / "benchmark" / "tool_routing_phase2_seed.json"
        )
        self.assertEqual(len(samples), 16)
        self.assertEqual(
            {sample["domain"] for sample in samples},
            {"finance", "mathematics", "coding", "enterprise_automation"},
        )

    def test_tool_to_label_mapping(self) -> None:
        mapping = build_tool_label_mapping(
            ["calculator", "unit_converter", "stock_price_api"]
        )
        self.assertEqual(
            mapping,
            {
                "calculator": "A",
                "unit_converter": "B",
                "stock_price_api": "C",
            },
        )

    def test_forced_choice_prompt_construction(self) -> None:
        sample = {
            "query": "Convert 10 kilometers to miles.",
            "available_tools": ["calculator", "unit_converter"],
            "expected_tool": "unit_converter",
        }
        mapping = build_tool_label_mapping(sample["available_tools"])
        prompt = build_forced_choice_prompt(sample, mapping)
        self.assertIn("A. calculator", prompt)
        self.assertIn("B. unit_converter", prompt)
        self.assertIn("Return only the single letter label", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_label_token_mapping_prefers_single_leading_space_tokens(self) -> None:
        token_ids = get_label_token_ids(FakeTokenizer(), ["A", "B", "C"])
        self.assertEqual(token_ids, {"A": 101, "B": 102, "C": 103})

    def test_metric_computation_from_fake_logits(self) -> None:
        row = compute_layer_metrics(
            sample={
                "id": "sample_1",
                "domain": "mathematics",
                "phase2_focus": "same_domain_confusion",
            },
            layer_index=3,
            label_logits={"A": 1.0, "B": 4.0, "C": 2.0},
            correct_label="C",
            predicted_label_at_final_layer="B",
            correct_tool="calculator",
        )
        self.assertEqual(row["predicted_label_at_layer"], "B")
        self.assertEqual(row["best_wrong_label"], "B")
        self.assertEqual(row["correct_label_rank"], 2)
        self.assertEqual(row["correct_minus_best_wrong_logit"], -2.0)

    def test_output_path_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = make_output_paths(Path(tmpdir), timestamp="fixed")
            self.assertTrue(Path(tmpdir).exists())
            self.assertEqual(paths["csv"].name, "logit_lens_fixed.csv")
            self.assertEqual(paths["summary"].name, "logit_lens_fixed_summary.json")
            self.assertEqual(paths["plot"].name, "logit_lens_fixed.png")
            self.assertEqual(
                paths["sample_summary"].name,
                "logit_lens_fixed_sample_summary.csv",
            )

    def test_summary_aggregation_from_fake_rows(self) -> None:
        rows = [
            {
                "sample_id": "one",
                "domain": "finance",
                "phase2_focus": "easy_routing",
                "layer_index": 0,
                "correct_label": "A",
                "predicted_label_at_final_layer": "A",
                "correct_minus_best_wrong_logit": 0.5,
            },
            {
                "sample_id": "one",
                "domain": "finance",
                "phase2_focus": "easy_routing",
                "layer_index": 1,
                "correct_label": "A",
                "predicted_label_at_final_layer": "A",
                "correct_minus_best_wrong_logit": 1.5,
            },
            {
                "sample_id": "two",
                "domain": "coding",
                "phase2_focus": "known_failure",
                "layer_index": 1,
                "correct_label": "B",
                "predicted_label_at_final_layer": "A",
                "correct_minus_best_wrong_logit": -1.0,
            },
        ]
        summary = aggregate_summary(rows)
        self.assertEqual(summary["total_rows"], 3)
        self.assertEqual(summary["total_samples"], 2)
        self.assertEqual(summary["final_layer_accuracy"], 0.5)
        self.assertIn("known_failure", summary["by_phase2_focus"])
        self.assertIn("finance", summary["by_domain"])

    def test_groups_layer_margins_by_domain_and_focus(self) -> None:
        rows = [
            {
                "sample_id": "one",
                "domain": "finance",
                "phase2_focus": "easy_routing",
                "layer_index": 0,
                "correct_minus_best_wrong_logit": 1.0,
            },
            {
                "sample_id": "two",
                "domain": "coding",
                "phase2_focus": "easy_routing",
                "layer_index": 0,
                "correct_minus_best_wrong_logit": -2.0,
            },
            {
                "sample_id": "three",
                "domain": "finance",
                "phase2_focus": "same_domain_confusion",
                "layer_index": 1,
                "correct_minus_best_wrong_logit": 3.5,
            },
        ]

        by_domain = group_layer_margins(rows, group_by="domain")
        by_focus = group_layer_margins(rows, group_by="phase2_focus")

        self.assertEqual(by_domain["finance"][0], [1.0])
        self.assertEqual(by_domain["finance"][1], [3.5])
        self.assertEqual(by_domain["coding"][0], [-2.0])
        self.assertEqual(by_focus["easy_routing"][0], [1.0, -2.0])
        self.assertEqual(by_focus["same_domain_confusion"][1], [3.5])

    def test_sample_summary_aggregation(self) -> None:
        rows = [
            {
                "sample_id": "one",
                "domain": "finance",
                "phase2_focus": "easy_routing",
                "layer_index": 0,
                "correct_tool": "stock_price_api",
                "correct_label": "A",
                "predicted_label_at_final_layer": "A",
                "correct_minus_best_wrong_logit": -1.0,
            },
            {
                "sample_id": "one",
                "domain": "finance",
                "phase2_focus": "easy_routing",
                "layer_index": 1,
                "correct_tool": "stock_price_api",
                "correct_label": "A",
                "predicted_label_at_final_layer": "A",
                "correct_minus_best_wrong_logit": 2.5,
            },
            {
                "sample_id": "two",
                "domain": "coding",
                "phase2_focus": "known_failure",
                "layer_index": 0,
                "correct_tool": "read_code_file",
                "correct_label": "B",
                "predicted_label_at_final_layer": "C",
                "correct_minus_best_wrong_logit": -3.0,
            },
        ]

        summaries = summarize_samples(rows)

        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["sample_id"], "one")
        self.assertTrue(summaries[0]["final_correct"])
        self.assertEqual(summaries[0]["final_correct_minus_best_wrong_logit"], 2.5)
        self.assertEqual(summaries[0]["strongest_layer"], 1)
        self.assertEqual(summaries[0]["weakest_layer"], 0)
        self.assertFalse(summaries[1]["final_correct"])

    def test_selects_strongest_and_weakest_examples(self) -> None:
        summaries = [
            {
                "sample_id": "strong",
                "final_correct": True,
                "final_correct_minus_best_wrong_logit": 5.0,
            },
            {
                "sample_id": "weak_success",
                "final_correct": True,
                "final_correct_minus_best_wrong_logit": 0.1,
            },
            {
                "sample_id": "failure",
                "final_correct": False,
                "final_correct_minus_best_wrong_logit": -2.0,
            },
        ]

        strongest = select_strongest_successes(summaries, limit=2)
        weakest = select_weakest_cases(summaries, limit=2)

        self.assertEqual([row["sample_id"] for row in strongest], ["strong", "weak_success"])
        self.assertEqual([row["sample_id"] for row in weakest], ["failure", "weak_success"])


if __name__ == "__main__":
    unittest.main()
