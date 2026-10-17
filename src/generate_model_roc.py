"""Generate consistent per-model and combined ROC-AUC curves."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fraudformer import FraudFormerForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "splits" / "test.csv"
METRICS_DIR = ROOT / "results" / "metrics"
OUTPUT_DIR = ROOT / "results" / "roc_auc"
CACHE_DIR = ROOT / "results" / "logits" / "roc_auc"
LABELS = [0, 1, 2]
LABEL_NAMES = ["Legitimate", "Human-written Fraud", "AI-generated Fraud"]
CLASS_COLORS = ["#4C78A8", "#E45756", "#2A9D8F"]

MODEL_SPECS = {
    "bert": {
        "name": "BERT",
        "checkpoint": ROOT / "models" / "bert" / "baseline_v1" / "final_checkpoint",
        "metrics": "bert_metrics.json",
        "custom": False,
        "batch_size": 16,
        "color": "#4C78A8",
    },
    "roberta": {
        "name": "RoBERTa",
        "checkpoint": ROOT / "models" / "roberta" / "baseline_v1" / "final_checkpoint",
        "metrics": "roberta_metrics.json",
        "custom": False,
        "batch_size": 16,
        "color": "#F58518",
    },
    "deberta": {
        "name": "DeBERTa",
        "checkpoint": ROOT / "models" / "deberta" / "baseline_v1" / "final_checkpoint",
        "metrics": "deberta_metrics.json",
        "custom": False,
        "batch_size": 8,
        "color": "#54A24B",
    },
    "fraudformer_roberta": {
        "name": "FraudFormer-RoBERTa",
        "checkpoint": ROOT / "models" / "fraudformer" / "experiment_v1" / "final_checkpoint",
        "metrics": "fraudformer_metrics.json",
        "custom": True,
        "batch_size": 8,
        "color": "#E45756",
    },
    "fraudformer_deberta": {
        "name": "FraudFormer-DeBERTa",
        "checkpoint": ROOT / "models" / "fraudformer" / "experiment_deberta_frozen" / "final_checkpoint",
        "metrics": "fraudformer_experiment_deberta_frozen_metrics.json",
        "custom": True,
        "batch_size": 8,
        "color": "#7A5195",
    },
}

FUSION_SLUG = "fraudformer_fusion"
FUSION_NAME = "FraudFormer Fusion (ensemble)"
FUSION_COLOR = "#2A9D8F"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def load_bias(checkpoint: Path) -> np.ndarray:
    path = checkpoint / "calibration.json"
    if not path.exists():
        return np.zeros(3, dtype=np.float32)
    with path.open(encoding="utf-8") as handle:
        return np.asarray(json.load(handle)["logit_bias"], dtype=np.float32)


def collect_logits(
    slug: str,
    spec: dict,
    texts: list[str],
    device: torch.device,
    requested_batch_size: int | None,
) -> np.ndarray:
    cache_path = CACHE_DIR / f"{slug}.npz"
    if cache_path.exists():
        print(f"Loading cached logits for {spec['name']}", flush=True)
        return np.load(cache_path)["logits"]

    tokenizer = AutoTokenizer.from_pretrained(spec["checkpoint"])
    model_class = FraudFormerForSequenceClassification if spec["custom"] else AutoModelForSequenceClassification
    model = model_class.from_pretrained(spec["checkpoint"])
    model.to(device).eval()
    batch_size = requested_batch_size or spec["batch_size"]
    logits_parts = []
    print(f"Running {spec['name']} on {device} with batch size {batch_size}", flush=True)
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            logits_parts.append(model(**batch).logits.cpu().numpy())
        completed = min(start + batch_size, len(texts))
        if completed % 800 == 0 or completed == len(texts):
            print(f"  {completed}/{len(texts)}", flush=True)

    logits = np.vstack(logits_parts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, logits=logits)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def expected_accuracy(metrics_filename: str) -> float:
    with (METRICS_DIR / metrics_filename).open(encoding="utf-8") as handle:
        return json.load(handle)["test"]["accuracy"]


def validate_accuracy(name: str, labels: np.ndarray, probabilities: np.ndarray, expected: float) -> None:
    observed = accuracy_score(labels, probabilities.argmax(axis=1))
    if not np.isclose(observed, expected, atol=1e-12):
        raise RuntimeError(f"{name} accuracy mismatch: expected {expected:.12f}, observed {observed:.12f}")
    print(f"Validated {name} accuracy: {observed:.6f}", flush=True)


def curve_data(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    binary = label_binarize(labels, classes=LABELS)
    classes = []
    for index, label_name in enumerate(LABEL_NAMES):
        fpr, tpr, _ = roc_curve(binary[:, index], probabilities[:, index])
        classes.append((label_name, fpr, tpr, auc(fpr, tpr)))
    micro_fpr, micro_tpr, _ = roc_curve(binary.ravel(), probabilities.ravel())
    all_fpr = np.unique(np.concatenate([item[1] for item in classes]))
    macro_tpr = np.mean([np.interp(all_fpr, item[1], item[2]) for item in classes], axis=0)
    return {
        "classes": classes,
        "micro": (micro_fpr, micro_tpr, auc(micro_fpr, micro_tpr)),
        "macro": (all_fpr, macro_tpr, auc(all_fpr, macro_tpr)),
        "macro_ovr": roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"),
        "weighted_ovr": roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted"),
        "macro_ovo": roc_auc_score(labels, probabilities, multi_class="ovo", average="macro"),
        "weighted_ovo": roc_auc_score(labels, probabilities, multi_class="ovo", average="weighted"),
    }


def plot_individual_roc(name: str, slug: str, curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    for (label_name, fpr, tpr, score), color in zip(curves["classes"], CLASS_COLORS):
        ax.plot(fpr, tpr, linewidth=2.2, color=color, label=f"{label_name} (AUC={score:.4f})")
    macro_fpr, macro_tpr, macro_auc = curves["macro"]
    micro_fpr, micro_tpr, micro_auc = curves["micro"]
    ax.plot(macro_fpr, macro_tpr, color="#7A5195", linewidth=2.4, linestyle="--", label=f"Macro average (AUC={macro_auc:.4f})")
    ax.plot(micro_fpr, micro_tpr, color="#F58518", linewidth=2.2, linestyle=":", label=f"Micro average (AUC={micro_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.5, linestyle="--", label="Chance")
    ax.set_title(f"{name} ROC-AUC Curves", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{slug}_roc_auc_curves.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def score_row(name: str, curves: dict) -> dict:
    return {
        "model": name,
        "roc_auc_macro_ovr": curves["macro_ovr"],
        "roc_auc_weighted_ovr": curves["weighted_ovr"],
        "roc_auc_macro_ovo": curves["macro_ovo"],
        "roc_auc_weighted_ovo": curves["weighted_ovo"],
        "class_0_roc_auc_ovr": curves["classes"][0][3],
        "class_1_roc_auc_ovr": curves["classes"][1][3],
        "class_2_roc_auc_ovr": curves["classes"][2][3],
        "micro_roc_auc": curves["micro"][2],
    }


def plot_combined_roc(all_curves: dict[str, tuple[str, dict, str]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    for _, (name, curves, color) in all_curves.items():
        fpr, tpr, score = curves["micro"]
        ax.plot(fpr, tpr, linewidth=2.2, color=color, label=f"{name} (micro AUC={score:.4f})")
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.5, linestyle="--", label="Chance")
    ax.set_title("Combined ROC-AUC Curves", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_roc_curves.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Override model-specific inference batch sizes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(TEST_PATH, dtype=str, keep_default_na=False)
    labels = frame["label"].astype(int).to_numpy()
    texts = frame["model_input_text"].tolist()
    device = torch.device(args.device)
    raw_logits = {}
    probabilities = {}

    for slug, spec in MODEL_SPECS.items():
        logits = collect_logits(slug, spec, texts, device, args.batch_size)
        raw_logits[slug] = logits
        probabilities[slug] = softmax(logits + load_bias(spec["checkpoint"]))
        validate_accuracy(spec["name"], labels, probabilities[slug], expected_accuracy(spec["metrics"]))

    with (METRICS_DIR / "fraudformer_fusion_metrics.json").open(encoding="utf-8") as handle:
        fusion_metrics = json.load(handle)
    weights = fusion_metrics["weights_tuned_on_validation"]
    fused = (
        weights["fraudformer"] * softmax(raw_logits["fraudformer_roberta"])
        + weights["deberta"] * softmax(raw_logits["deberta"])
        + weights["roberta"] * softmax(raw_logits["roberta"])
    )
    fusion_bias = np.asarray(fusion_metrics["logit_bias_tuned_on_validation"])
    probabilities[FUSION_SLUG] = softmax(np.log(fused.clip(1e-9)) + fusion_bias)
    validate_accuracy(FUSION_NAME, labels, probabilities[FUSION_SLUG], fusion_metrics["test"]["accuracy"])

    all_curves = {}
    rows = []
    for slug, spec in MODEL_SPECS.items():
        curves = curve_data(labels, probabilities[slug])
        plot_individual_roc(spec["name"], slug, curves)
        all_curves[slug] = (spec["name"], curves, spec["color"])
        rows.append(score_row(spec["name"], curves))
    fusion_curves = curve_data(labels, probabilities[FUSION_SLUG])
    plot_individual_roc(FUSION_NAME, FUSION_SLUG, fusion_curves)
    all_curves[FUSION_SLUG] = (FUSION_NAME, fusion_curves, FUSION_COLOR)
    rows.append(score_row(FUSION_NAME, fusion_curves))

    plot_combined_roc(all_curves)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "roc_auc_scores.csv", index=False)
    print(f"Saved consistent ROC-AUC outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
