from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from models.model_loader import load_model_components, resolve_model_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "tool_routing_phase2_seed.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
DEFAULT_MODEL_NAME = resolve_model_name()
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PROMPT_TEMPLATE = "forced_choice_label_v1"


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Benchmark must be a JSON list.")

    return data


def build_tool_label_mapping(available_tools: Sequence[str]) -> dict[str, str]:
    if not available_tools:
        raise ValueError("available_tools must not be empty.")
    if len(available_tools) > len(LABELS):
        raise ValueError(f"At most {len(LABELS)} tools are supported.")

    return {tool: LABELS[index] for index, tool in enumerate(available_tools)}


def build_forced_choice_prompt(sample: Mapping[str, Any], tool_to_label: Mapping[str, str]) -> str:
    options = "\n".join(
        f"{label}. {tool}" for tool, label in tool_to_label.items()
    )
    return f"""
You are routing a user request to exactly one MCP tool.

Return only the single letter label for the best tool.

Options:
{options}

User request:
{sample["query"]}

Answer:
""".strip()


def _encode_without_specials(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))

    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        return list(input_ids[0])
    return list(input_ids)


def get_label_token_ids(tokenizer: Any, labels: Sequence[str]) -> dict[str, int]:
    label_token_ids: dict[str, int] = {}
    for label in labels:
        for token_text in (f" {label}", label):
            token_ids = _encode_without_specials(tokenizer, token_text)
            if len(token_ids) == 1:
                label_token_ids[label] = token_ids[0]
                break
        else:
            raise ValueError(
                f"Label {label!r} is not a single token with or without a leading space."
            )

    return label_token_ids


def compute_layer_metrics(
    *,
    sample: Mapping[str, Any],
    layer_index: int,
    label_logits: Mapping[str, float],
    correct_label: str,
    predicted_label_at_final_layer: str,
    correct_tool: str,
) -> dict[str, Any]:
    ranked = sorted(label_logits.items(), key=lambda item: item[1], reverse=True)
    rank_lookup = {label: rank + 1 for rank, (label, _) in enumerate(ranked)}
    predicted_label_at_layer = ranked[0][0]
    wrong_ranked = [(label, logit) for label, logit in ranked if label != correct_label]
    best_wrong_label, best_wrong_label_logit = wrong_ranked[0]
    correct_label_logit = float(label_logits[correct_label])

    return {
        "sample_id": sample["id"],
        "domain": sample["domain"],
        "phase2_focus": sample.get("phase2_focus", "unspecified"),
        "layer_index": layer_index,
        "correct_tool": correct_tool,
        "correct_label": correct_label,
        "predicted_label_at_layer": predicted_label_at_layer,
        "predicted_label_at_final_layer": predicted_label_at_final_layer,
        "correct_label_logit": correct_label_logit,
        "best_wrong_label": best_wrong_label,
        "best_wrong_label_logit": float(best_wrong_label_logit),
        "correct_label_rank": rank_lookup[correct_label],
        "correct_minus_best_wrong_logit": correct_label_logit - float(best_wrong_label_logit),
    }


def aggregate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "total_rows": 0,
            "total_samples": 0,
            "average_correct_minus_best_wrong_logit": 0.0,
            "final_layer_accuracy": 0.0,
            "by_phase2_focus": {},
        }

    sample_final_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if (
            sample_id not in sample_final_rows
            or int(row["layer_index"]) > int(sample_final_rows[sample_id]["layer_index"])
        ):
            sample_final_rows[sample_id] = row

    final_correct = sum(
        1
        for row in sample_final_rows.values()
        if row["predicted_label_at_final_layer"] == row["correct_label"]
    )
    focus_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        focus_groups[str(row["phase2_focus"])].append(row)

    return {
        "total_rows": len(rows),
        "total_samples": len(sample_final_rows),
        "average_correct_minus_best_wrong_logit": sum(
            float(row["correct_minus_best_wrong_logit"]) for row in rows
        )
        / len(rows),
        "final_layer_accuracy": final_correct / len(sample_final_rows),
        "by_phase2_focus": {
            focus: {
                "rows": len(group),
                "average_correct_minus_best_wrong_logit": sum(
                    float(row["correct_minus_best_wrong_logit"]) for row in group
                )
                / len(group),
            }
            for focus, group in sorted(focus_groups.items())
        },
    }


def make_output_paths(output_dir: Path, timestamp: str | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return {
        "csv": output_dir / f"logit_lens_{run_timestamp}.csv",
        "summary": output_dir / f"logit_lens_{run_timestamp}_summary.json",
        "plot": output_dir / f"logit_lens_{run_timestamp}.png",
    }


def _project_hidden_state(model: Any, hidden_vector: Any) -> Any:
    output_embeddings = model.get_output_embeddings()
    return output_embeddings(hidden_vector)


def _label_logits_from_vector(logits: Any, label_token_ids: Mapping[str, int]) -> dict[str, float]:
    return {
        label: float(logits[token_id].detach().cpu())
        for label, token_id in label_token_ids.items()
    }


def analyze_sample(model: Any, tokenizer: Any, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    import torch

    available_tools = sample["available_tools"]
    tool_to_label = build_tool_label_mapping(available_tools)
    label_to_tool = {label: tool for tool, label in tool_to_label.items()}
    correct_tool = sample["expected_tool"]
    if correct_tool not in tool_to_label:
        raise ValueError(f"Expected tool {correct_tool!r} is not in available_tools.")

    correct_label = tool_to_label[correct_tool]
    label_token_ids = get_label_token_ids(tokenizer, list(label_to_tool))
    prompt = build_forced_choice_prompt(sample, tool_to_label)
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    else:
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

    # Use no_grad rather than inference_mode because hidden states are projected
    # through the output head after the forward pass for logit-lens scoring.
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    final_logits = _project_hidden_state(model, hidden_states[-1][0, -1, :])
    final_label_logits = _label_logits_from_vector(final_logits, label_token_ids)
    predicted_label_at_final_layer = max(
        final_label_logits.items(), key=lambda item: item[1]
    )[0]

    rows = []
    for layer_index, hidden_state in enumerate(hidden_states):
        logits = _project_hidden_state(model, hidden_state[0, -1, :])
        label_logits = _label_logits_from_vector(logits, label_token_ids)
        rows.append(
            compute_layer_metrics(
                sample=sample,
                layer_index=layer_index,
                label_logits=label_logits,
                correct_label=correct_label,
                predicted_label_at_final_layer=predicted_label_at_final_layer,
                correct_tool=correct_tool,
            )
        )

    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "sample_id",
        "domain",
        "phase2_focus",
        "layer_index",
        "correct_tool",
        "correct_label",
        "predicted_label_at_layer",
        "predicted_label_at_final_layer",
        "correct_label_logit",
        "best_wrong_label",
        "best_wrong_label_logit",
        "correct_label_rank",
        "correct_minus_best_wrong_logit",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    benchmark_path: Path,
    model_name: str,
) -> None:
    summary = aggregate_summary(rows)
    summary.update(
        {
            "benchmark_path": str(benchmark_path),
            "model_name": model_name,
            "prompt_template": PROMPT_TEMPLATE,
        }
    )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)


def plot_results(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["phase2_focus"])][int(row["layer_index"])].append(
            float(row["correct_minus_best_wrong_logit"])
        )

    plt.figure(figsize=(10, 6))
    for focus, layer_values in sorted(grouped.items()):
        layers = sorted(layer_values)
        averages = [
            sum(layer_values[layer]) / len(layer_values[layer])
            for layer in layers
        ]
        plt.plot(layers, averages, marker="o", label=focus)

    plt.xlabel("Layer")
    plt.ylabel("Average correct minus best-wrong logit")
    plt.title("Layer-wise MCP tool-label separation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_analysis(args: argparse.Namespace) -> dict[str, Path]:
    import torch

    if args.plot:
        try:
            import matplotlib.pyplot  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install matplotlib before using --plot: "
                "python -m pip install -e ."
            ) from exc

    benchmark_path = args.benchmark
    samples = load_benchmark(benchmark_path)
    if args.max_examples is not None:
        samples = samples[: args.max_examples]

    components = load_model_components(args.model, output_hidden_states=True)
    tokenizer = components.tokenizer
    model = components.model
    if not torch.cuda.is_available():
        model.eval()

    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.extend(analyze_sample(model, tokenizer, sample))

    paths = make_output_paths(args.output_dir)
    write_csv(rows, paths["csv"])
    write_summary(
        rows=rows,
        path=paths["summary"],
        benchmark_path=benchmark_path,
        model_name=args.model,
    )
    if args.plot:
        plot_results(rows, paths["plot"])

    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run layer-wise logit-lens tracking for MCP tool-label choices."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--plot", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = run_analysis(args)
    print(f"CSV: {paths['csv']}")
    print(f"Summary: {paths['summary']}")
    if args.plot:
        print(f"Plot: {paths['plot']}")


if __name__ == "__main__":
    main()
