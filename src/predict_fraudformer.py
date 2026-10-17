"""Run inference with a trained FraudFormer checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from transformers import AutoTokenizer

from fraudformer import FraudFormerForSequenceClassification


def predict(checkpoint: Path, texts: list[str], max_length: int = 256):
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("At least one non-empty job advertisement is required")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = FraudFormerForSequenceClassification.from_pretrained(checkpoint).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        logits = model(**encoded).logits
    calibration_path = checkpoint / "calibration.json"
    if calibration_path.exists():
        bias = json.loads(calibration_path.read_text(encoding="utf-8"))["logit_bias"]
        logits = logits + torch.tensor(bias, device=device)
    probabilities = logits.softmax(-1).cpu()
    results = []
    for text, probability in zip(texts, probabilities):
        predicted = int(probability.argmax())
        results.append(
            {
                "label": predicted,
                "label_name": model.config.id2label[predicted],
                "confidence": float(probability[predicted]),
                "probabilities": {
                    model.config.id2label[index]: float(value)
                    for index, value in enumerate(probability)
                },
                "text_preview": text[:160],
            }
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models/fraudformer/experiment_v1/final_checkpoint",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Text to classify; may be repeated")
    source.add_argument("--file", type=Path, help="UTF-8 text file containing one advert")
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inputs = args.text if args.text is not None else [args.file.read_text(encoding="utf-8")]
    json.dump(predict(args.checkpoint, inputs, args.max_length), sys.stdout, indent=2)
    print()

