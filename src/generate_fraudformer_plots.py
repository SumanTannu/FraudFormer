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
DOCUMENTATION_PLOT_DIR = ROOT / "results" / "documentation_screenshots"

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

MODEL_SLUGS = {
    "BERT": "bert",
    "RoBERTa": "roberta",
    "DeBERTa": "deberta",
    "FraudFormer-RoBERTa": "fraudformer_roberta",
    "FraudFormer-DeBERTa": "fraudformer_deberta",
    "FraudFormer Fusion (ensemble)": "fraudformer_fusion",
}

EVALUATION_METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_precision", "Macro\nPrecision"),
    ("macro_recall", "Macro\nRecall"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
]

METRIC_COLORS = ["#4C78A8", "#72A98F", "#E9A23B", "#8B78B4", "#5BA4A8"]

OVERALL_METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("class_1_f1", "Human-fraud F1"),
    ("class_2_f1", "AI-fraud F1"),
]

CLASS_NAMES = ["Legitimate", "Human-written Fraud", "AI-generated Fraud"]

TRAINING_RUNS = {
    "BERT": ROOT / "models" / "bert" / "baseline_v1",
    "RoBERTa": ROOT / "models" / "roberta" / "baseline_v1",
    "DeBERTa": ROOT / "models" / "deberta" / "baseline_v1",
    "FraudFormer-RoBERTa": ROOT / "models" / "fraudformer" / "experiment_v1",
    "FraudFormer-DeBERTa": ROOT / "models" / "fraudformer" / "experiment_deberta_frozen",
}


def load_payloads() -> dict[str, dict]:
    payloads = {}
    for model_name, filename in MODEL_FILES.items():
        with (METRICS_DIR / filename).open(encoding="utf-8") as handle:
            payloads[model_name] = json.load(handle)
    return payloads


def save_figure(path: Path, aliases: tuple[Path, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    for alias in aliases:
        alias.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(alias, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def add_value_labels(
    ax: plt.Axes,
    bars,
    *,
    fontsize: int = 8,
    rotation: int = 90,
    suffix: str = "%",
) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
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


def plot_evaluation_summary(model_name: str, payload: dict) -> None:
    values = [payload[key] * 100 for key, _ in EVALUATION_METRICS]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(
        [label for _, label in EVALUATION_METRICS],
        values,
        color=METRIC_COLORS,
        edgecolor="white",
    )
    add_value_labels(ax, bars, fontsize=9, rotation=0)
    ax.set_title(f"{model_name} Evaluation Results", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Test score (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save_figure(PLOT_DIR / f"{MODEL_SLUGS[model_name]}_test_metrics.png")


def plot_per_class_metrics(model_name: str, payload: dict) -> None:
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

    ax.set_title(f"{model_name} Per-Class Test Performance", fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    save_figure(PLOT_DIR / f"{MODEL_SLUGS[model_name]}_per_class_metrics.png")


def validation_metrics(payload: dict) -> dict | None:
    validation = payload.get("validation")
    if validation is None:
        return None
    if "raw" in validation:
        return validation["raw"]
    return {
        key.removeprefix("validation_"): value
        for key, value in validation.items()
        if key.startswith("validation_")
    }


def plot_validation_vs_test(model_name: str, payload: dict) -> None:
    validation = validation_metrics(payload)
    if validation is None:
        return
    test = payload["test"]
    keys_and_labels = [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]
    x = np.arange(len(keys_and_labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.5))
    validation_bars = ax.bar(
        x - width / 2,
        [validation[key] * 100 for key, _ in keys_and_labels],
        width,
        label="Validation",
        color="#72A98F",
    )
    test_bars = ax.bar(
        x + width / 2,
        [test[key] * 100 for key, _ in keys_and_labels],
        width,
        label="Test",
        color=MODEL_COLORS[model_name],
    )
    add_value_labels(ax, validation_bars, fontsize=8)
    add_value_labels(ax, test_bars, fontsize=8)
    title = f"{model_name} Validation vs Test"
    if "raw" in payload.get("validation", {}):
        title += " (Recorded Metrics)"
    ax.set_title(title, fontsize=17, fontweight="bold", pad=16)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, [label for _, label in keys_and_labels])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    save_figure(PLOT_DIR / f"{MODEL_SLUGS[model_name]}_validation_vs_test.png")


def plot_confusion_matrix(model_name: str, payload: dict) -> None:
    matrix = np.asarray(payload["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#111827",
                fontsize=12,
                fontweight="bold",
            )
    ax.set_title(f"{model_name} Confusion Matrix", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks(range(3), CLASS_NAMES, rotation=15, ha="right")
    ax.set_yticks(range(3), CLASS_NAMES)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(ROOT / "results" / "confusion_matrices" / f"{MODEL_SLUGS[model_name]}_confusion_matrix.png")


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


def load_training_losses(run_dir: Path) -> tuple[list[float], list[float]]:
    state_paths = list(run_dir.glob("checkpoint-*/trainer_state.json"))
    if not state_paths:
        raise FileNotFoundError(f"No trainer state found under {run_dir}")
    state_path = max(
        state_paths,
        key=lambda path: int(path.parent.name.removeprefix("checkpoint-")),
    )
    with state_path.open(encoding="utf-8") as handle:
        history = json.load(handle)["log_history"]
    loss_by_epoch = {
        float(item["epoch"]): float(item["loss"])
        for item in history
        if "epoch" in item and "loss" in item
    }
    epochs = sorted(loss_by_epoch)
    return epochs, [loss_by_epoch[epoch] for epoch in epochs]


def plot_training_progress() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for model_name, run_dir in TRAINING_RUNS.items():
        epochs, losses = load_training_losses(run_dir)
        ax.plot(
            epochs,
            losses,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=model_name,
            color=MODEL_COLORS[model_name],
        )

    ax.set_title("Training Progress", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_xticks(range(1, 6))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    save_figure(
        PLOT_DIR / "training_progress.png",
        aliases=(DOCUMENTATION_PLOT_DIR / "06_training_progress.png",),
    )

    for model_name, run_dir in TRAINING_RUNS.items():
        epochs, losses = load_training_losses(run_dir)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.plot(
            epochs,
            losses,
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=MODEL_COLORS[model_name],
        )
        del bars
        for epoch, loss in zip(epochs, losses):
            ax.annotate(f"{loss:.4f}", (epoch, loss), xytext=(0, 8), textcoords="offset points", ha="center")
        ax.set_title(f"{model_name} Training Progress", fontsize=17, fontweight="bold", pad=16)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_xticks(range(1, int(max(epochs)) + 1))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        save_figure(PLOT_DIR / f"{MODEL_SLUGS[model_name]}_training_progress.png")


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
    payloads = load_payloads()
    results = {model_name: payload["test"] for model_name, payload in payloads.items()}
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
    for model_name, payload in payloads.items():
        plot_evaluation_summary(model_name, payload["test"])
        plot_per_class_metrics(model_name, payload["test"])
        plot_validation_vs_test(model_name, payload)
        plot_confusion_matrix(model_name, payload)
    plot_grouped_metrics(
        results,
        list(MODEL_FILES),
        "Combined Test Performance: Baselines and FraudFormer Systems",
        "model_comparison_metrics.png",
        figsize=(17, 8),
    )
    plot_combined_class_f1(results)
    plot_training_progress()
    write_comparison_table(results)
    print(f"Saved plots to {PLOT_DIR}")
    print(f"Updated table: {TABLE_PATH}")


if __name__ == "__main__":
    main()
