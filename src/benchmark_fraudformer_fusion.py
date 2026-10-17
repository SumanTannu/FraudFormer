"""Validation-tuned fusion of FraudFormer and retained transformer checkpoints.

This is an optional high-accuracy deployment system, not a replacement for the
standalone architectural ablation. Mixture weights and class offsets are selected
on validation data before the test predictions are scored.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from fraudformer import FraudFormerForSequenceClassification
from train_fraudformer import LABELS, LABEL_NAMES, classification_metrics, tune_logit_bias


class InferenceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, list(map(int, labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            **{key: torch.tensor(value[index]) for key, value in self.encodings.items()},
            "labels": torch.tensor(self.labels[index]),
        }


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def collect_predictions(name, checkpoint, frames, cache_dir, batch_size, max_length):
    cached = cache_dir / f"{name}.npz"
    if cached.exists():
        data = np.load(cached)
        return {split: data[split] for split in ("validation", "test")}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = (
        FraudFormerForSequenceClassification.from_pretrained(checkpoint)
        if name == "fraudformer"
        else AutoModelForSequenceClassification.from_pretrained(checkpoint)
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(cache_dir / "tmp"),
            per_device_eval_batch_size=min(batch_size, 8) if name == "deberta" else batch_size,
            report_to=[],
            # One process evaluates all components. Keep a single precision mode;
            # Accelerate's shared state can otherwise leak FP16 into DeBERTa.
            fp16=False,
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    predictions = {}
    for split in ("validation", "test"):
        frame = frames[split]
        dataset = InferenceDataset(
            tokenizer(
                frame.model_input_text.tolist(), truncation=True, max_length=max_length
            ),
            frame.label,
        )
        predictions[split] = trainer.predict(dataset).predictions
    np.savez_compressed(cached, **predictions)
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def select_fusion(validation_probabilities, labels):
    candidates = []
    # A coarse simplex is deliberate: it reduces validation overfitting.
    for fraudformer_weight in np.arange(0, 1.001, 0.1):
        for deberta_weight in np.arange(0, 1.001 - fraudformer_weight, 0.1):
            roberta_weight = 1.0 - fraudformer_weight - deberta_weight
            weights = np.array([fraudformer_weight, deberta_weight, roberta_weight])
            fused = sum(w * p for w, p in zip(weights, validation_probabilities))
            score = classification_metrics(np.log(fused.clip(1e-9)), labels)["macro_f1"]
            candidates.append((score, weights, fused))
    best = None
    for _, weights, probabilities in sorted(candidates, reverse=True, key=lambda x: x[0])[:12]:
        bias, score = tune_logit_bias(np.log(probabilities.clip(1e-9)), labels)
        if best is None or score > best[0]:
            best = (score, weights, bias)
    return best


def run(args):
    root = Path(args.root).resolve()
    frames = {
        split: pd.read_csv(root / f"data/splits/{split}.csv", dtype=str, keep_default_na=False)
        for split in ("validation", "test")
    }
    checkpoints = {
        "fraudformer": root / "models/fraudformer/experiment_v1/final_checkpoint",
        "deberta": root / "models/deberta/baseline_v1/final_checkpoint",
        "roberta": root / "models/roberta/baseline_v1/final_checkpoint",
    }
    cache_dir = root / "results/logits/fraudformer_fusion"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: collect_predictions(
            name, checkpoint, frames, cache_dir, args.batch_size, args.max_length
        )
        for name, checkpoint in checkpoints.items()
    }
    order = ["fraudformer", "deberta", "roberta"]
    validation_probabilities = [softmax(outputs[name]["validation"]) for name in order]
    labels = frames["validation"].label.astype(int).to_numpy()
    validation_score, weights, bias = select_fusion(validation_probabilities, labels)

    test_probabilities = [softmax(outputs[name]["test"]) for name in order]
    fused_test = sum(w * p for w, p in zip(weights, test_probabilities))
    fused_test_logits = np.log(fused_test.clip(1e-9)) + bias
    test_labels = frames["test"].label.astype(int).to_numpy()
    metrics = classification_metrics(fused_test_logits, test_labels)
    predictions = fused_test_logits.argmax(1)
    cm = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(test_labels, predictions):
        cm[actual, predicted] += 1
    payload = {
        "model": "fraudformer_fusion",
        "component_order": order,
        "weights_tuned_on_validation": dict(zip(order, weights.tolist())),
        "logit_bias_tuned_on_validation": bias.tolist(),
        "validation_macro_f1": validation_score,
        "test": metrics,
        "confusion_matrix": cm.tolist(),
        "labels": dict(zip(LABELS, LABEL_NAMES)),
        "methodology": "weights and bias selected on validation; test scored once after selection",
    }
    destination = root / "results/metrics/fraudformer_fusion_metrics.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

