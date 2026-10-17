"""Train and evaluate FraudFormer on the repository's frozen splits."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from fraudformer import FraudFormerForSequenceClassification

LABELS = [0, 1, 2]
LABEL_NAMES = ["Legitimate", "Human-written Fraud", "AI-generated Fraud"]


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(map(int, labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def classification_metrics(logits, labels):
    predicted = np.argmax(logits, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=LABELS, zero_division=0
    )
    macro = precision_recall_fscore_support(
        labels, predicted, labels=LABELS, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        labels, predicted, labels=LABELS, average="weighted", zero_division=0
    )
    result = {
        "accuracy": accuracy_score(labels, predicted),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_precision": weighted[0],
        "weighted_recall": weighted[1],
        "weighted_f1": weighted[2],
    }
    for label, p, r, score, count in zip(LABELS, precision, recall, f1, support):
        result.update(
            {
                f"class_{label}_precision": float(p),
                f"class_{label}_recall": float(r),
                f"class_{label}_f1": float(score),
                f"class_{label}_support": float(count),
            }
        )
    return result


def trainer_metrics(prediction):
    logits, labels = prediction
    return classification_metrics(logits, labels)


def tune_logit_bias(validation_logits, validation_labels):
    """Tune two class offsets on validation only; class 0 is the reference."""
    best_bias = np.zeros(3, dtype=np.float32)
    best_score = classification_metrics(validation_logits, validation_labels)["macro_f1"]
    # Coordinate descent is inexpensive and less prone to overfitting than a dense 2-D grid.
    for step in (0.25, 0.10, 0.05):
        improved = True
        while improved:
            improved = False
            for label in (1, 2):
                for direction in (-1, 1):
                    candidate = best_bias.copy()
                    candidate[label] += direction * step
                    score = classification_metrics(
                        validation_logits + candidate, validation_labels
                    )["macro_f1"]
                    if score > best_score + 1e-12:
                        best_score, best_bias = score, candidate
                        improved = True
    return best_bias, best_score


def _class_weights(labels):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=len(LABELS))
    weights = 1.0 / np.sqrt(counts.clip(min=1))
    return (weights / weights.mean()).tolist(), counts.tolist()


def _load_splits(root):
    frames = {
        name: pd.read_csv(root / f"data/splits/{name}.csv", dtype=str, keep_default_na=False)
        for name in ("train", "validation", "test")
    }
    for name, frame in frames.items():
        labels = frame.label.astype(int)
        if set(labels) - set(LABELS):
            raise ValueError(f"Unexpected labels in {name}")
        if not frame.model_input_text.str.strip().all():
            raise ValueError(f"Empty model input in {name}")
    return frames


def run(args):
    root = Path(args.root).resolve()
    set_seed(args.seed)
    frames = _load_splits(root)
    weights, train_counts = _class_weights(frames["train"].label.astype(int))
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    datasets = {
        name: EncodedDataset(
            tokenizer(
                frame.model_input_text.tolist(), truncation=True, max_length=args.max_length
            ),
            frame.label,
        )
        for name, frame in frames.items()
    }
    if args.evaluate_checkpoint:
        model = FraudFormerForSequenceClassification.from_pretrained(
            args.evaluate_checkpoint
        )
    else:
        model = FraudFormerForSequenceClassification.from_backbone_pretrained(
            args.backbone,
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABEL_NAMES)),
            label2id={name: index for index, name in enumerate(LABEL_NAMES)},
            layer_mix_count=args.layer_mix_count,
            evidence_kernels=args.evidence_kernels,
            dropout=args.dropout,
            focal_gamma=args.focal_gamma,
            class_weights=weights,
        )
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    output = root / "models/fraudformer" / args.experiment_name
    output.mkdir(parents=True, exist_ok=True)
    use_fp16 = torch.cuda.is_available() and not args.fp32
    training_args = TrainingArguments(
        output_dir=str(output),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    started = time.time()
    if not args.evaluate_checkpoint:
        trainer.train()
    validation_output = trainer.predict(datasets["validation"])
    bias, tuned_validation_macro_f1 = tune_logit_bias(
        validation_output.predictions, validation_output.label_ids
    )
    test_output = trainer.predict(datasets["test"])
    raw_test = classification_metrics(test_output.predictions, test_output.label_ids)
    calibrated_logits = test_output.predictions + bias
    calibrated_test = classification_metrics(calibrated_logits, test_output.label_ids)
    predictions = calibrated_logits.argmax(1)
    cm = confusion_matrix(test_output.label_ids, predictions, labels=LABELS).tolist()

    checkpoint = output / "final_checkpoint"
    trainer.save_model(str(checkpoint))
    tokenizer.save_pretrained(str(checkpoint))
    (checkpoint / "calibration.json").write_text(
        json.dumps({"logit_bias": bias.tolist()}, indent=2), encoding="utf-8"
    )
    payload = {
        "configuration": {
            "model": "fraudformer",
            "architecture_version": "1.0",
            "backbone": args.backbone,
            "evaluated_checkpoint": str(args.evaluate_checkpoint) if args.evaluate_checkpoint else None,
            "backbone_frozen": args.freeze_backbone,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "epochs_requested": args.epochs,
            "max_length": args.max_length,
            "layer_mix_count": args.layer_mix_count,
            "evidence_kernels": args.evidence_kernels,
            "dropout": args.dropout,
            "focal_gamma": args.focal_gamma,
            "class_weights": weights,
            "train_class_counts": train_counts,
            "logit_bias_tuned_on_validation": bias.tolist(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "mixed_precision": "fp16" if use_fp16 else "fp32",
            "duration_seconds": time.time() - started,
            "completed_at": datetime.now(UTC).isoformat(),
        },
        "validation": {
            "raw": classification_metrics(
                validation_output.predictions, validation_output.label_ids
            ),
            "calibrated_macro_f1": tuned_validation_macro_f1,
        },
        "test_raw": raw_test,
        "test": calibrated_test,
        "confusion_matrix": cm,
        "labels": dict(zip(LABELS, LABEL_NAMES)),
    }
    results = root / "results"
    (results / "metrics").mkdir(parents=True, exist_ok=True)
    (results / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    metric_suffix = "" if args.experiment_name == "experiment_v1" else f"_{args.experiment_name}"
    (results / f"metrics/fraudformer{metric_suffix}_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(3), LABEL_NAMES, rotation=25, ha="right")
    plt.yticks(range(3), LABEL_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i][j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(
        results / f"confusion_matrices/fraudformer{metric_suffix}_confusion_matrix.png",
        dpi=160,
    )
    plt.close()
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backbone", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--layer-mix-count", type=int, default=4)
    parser.add_argument("--evidence-kernels", type=int, nargs="+", default=[3, 5, 9])
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--experiment-name", default="experiment_v1")
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        help="Skip training and evaluate an existing FraudFormer checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

