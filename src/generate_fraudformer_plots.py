"""Generate standalone FraudFormer and combined model-comparison plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
PLOT_DIR = ROOT / "results" / "plots" / "model_performance"
TABLE_PATH = ROOT / "results" / "tables" / "model_comparison_metrics.csv"

MODEL_FILES = {
    "BERT": "bert_metrics.json",
    "RoBERTa": "roberta_metrics.json",
    "DeBERTa": "deberta_metrics.json",
    "FraudFormer-RoBERTa": "fraudformer_metrics.json",
    "FraudFormer-DeBERTa": "fraudformer_experiment_deberta_frozen_metrics.json",
    "FraudFormer Fusion (ensemble)": "fraudformer_fusion_metrics.json",
}

MODEL_COLORS = {
    "BERT": "#4C78A8",
    "RoBERTa": "#F58518",
    "DeBERTa": "#54A24B",
    "FraudFormer-RoBERTa": "#E45756",
    "FraudFormer-DeBERTa": "#7A5195",
    "FraudFormer Fusion (ensemble)": "#2A9D8F",
}

OVERALL_METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("class_1_f1", "Human-fraud F1"),
    ("class_2_f1", "AI-fraud F1"),
]

CLASS_NAMES = ["Legitimate", "Human-written Fraud", "AI-generated Fraud"]


def load_results() -> dict[str, dict]:
    results = {}
    for model_name, filename in MODEL_FILES.items():
        with (METRICS_DIR / filename).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        results[model_name] = payload["test"]
    return results


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def add_value_labels(ax: plt.Axes, bars, *, fontsize: int = 8) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
        )


def plot_grouped_metrics(
    results: dict[str, dict],
    model_names: list[str],
    title: str,
    output_name: str,
    *,
    figsize: tuple[int, int],
) -> None:
    x = np.arange(len(OVERALL_METRICS))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=figsize)

    for index, model_name in enumerate(model_names):
        offset = (index - (len(model_names) - 1) / 2) * width
        values = [results[model_name][key] * 100 for key, _ in OVERALL_METRICS]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            color=MODEL_COLORS[model_name],
            edgecolor="white",
            linewidth=0.5,
        )
        add_value_labels(ax, bars, fontsize=7 if len(model_names) > 3 else 8)

    ax.set_title(title, fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Test score (%)")
    ax.set_xticks(x, [label for _, label in OVERALL_METRICS])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncol=2 if len(model_names) > 3 else 1, frameon=False, loc="lower right")
    save_figure(PLOT_DIR / output_name)


def plot_fraudformer_per_class(results: dict[str, dict]) -> None:
    model_name = "FraudFormer-DeBERTa"
    payload = results[model_name]
    x = np.arange(len(CLASS_NAMES))
    width = 0.24
    series = [
        ("Precision", [payload[f"class_{i}_precision"] * 100 for i in range(3)], "#4C78A8"),
        ("Recall", [payload[f"class_{i}_recall"] * 100 for i in range(3)], "#F58518"),
        ("F1", [payload[f"class_{i}_f1"] * 100 for i in range(3)], "#7A5195"),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    for index, (label, values, color) in enumerate(series):
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        add_value_labels(ax, bars)

    ax.set_title("FraudFormer-DeBERTa Per-Class Test Performance", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save_figure(PLOT_DIR / "fraudformer_per_class_metrics.png")


def plot_combined_class_f1(results: dict[str, dict]) -> None:
    model_names = list(MODEL_FILES)
    x = np.arange(len(model_names))
    width = 0.24
    class_colors = ["#4C78A8", "#E45756", "#2A9D8F"]

    fig, ax = plt.subplots(figsize=(16, 8))
    for class_index, (class_name, color) in enumerate(zip(CLASS_NAMES, class_colors)):
        values = [results[model][f"class_{class_index}_f1"] * 100 for model in model_names]
        bars = ax.bar(
            x + (class_index - 1) * width,
            values,
            width,
            label=class_name,
            color=color,
        )
        add_value_labels(ax, bars, fontsize=7)

    ax.set_title("Per-Class F1: Baselines and FraudFormer Systems", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Test F1 score (%)")
    ax.set_xticks(x, model_names, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    save_figure(PLOT_DIR / "model_comparison_per_class_f1.png")


def write_comparison_table(results: dict[str, dict]) -> None:
    rows = []
    for model_name in MODEL_FILES:
        payload = results[model_name]
        rows.append(
            {
                "model": model_name,
                "accuracy": payload["accuracy"] * 100,
                "macro_f1": payload["macro_f1"] * 100,
                "weighted_f1": payload["weighted_f1"] * 100,
                "class1_f1": payload["class_1_f1"] * 100,
                "class2_f1": payload["class_2_f1"] * 100,
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_PATH, index=False)


def main() -> None:
    results = load_results()
    fraudformer_models = [
        "FraudFormer-RoBERTa",
        "FraudFormer-DeBERTa",
        "FraudFormer Fusion (ensemble)",
    ]
    plot_grouped_metrics(
        results,
        fraudformer_models,
        "FraudFormer Test Results",
        "fraudformer_results.png",
        figsize=(13, 7),
    )
    plot_fraudformer_per_class(results)
    plot_grouped_metrics(
        results,
        list(MODEL_FILES),
        "Combined Test Performance: Baselines and FraudFormer Systems",
        "model_comparison_metrics.png",
        figsize=(17, 8),
    )
    plot_combined_class_f1(results)
    write_comparison_table(results)
    print(f"Saved plots to {PLOT_DIR}")
    print(f"Updated table: {TABLE_PATH}")


if __name__ == "__main__":
    main()
