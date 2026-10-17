"""Local FastAPI service for the React fraud-classification interface."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fraudformer import FraudFormerForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = ["Legitimate", "Human-written Fraud", "AI-generated Fraud"]

MODEL_REGISTRY = {
    "fraudformer-deberta": {
        "name": "FraudFormer–DeBERTa",
        "description": "Best observed model · 95.17% test macro-F1",
        "path": ROOT / "models/fraudformer/experiment_deberta_frozen/final_checkpoint",
        "custom": True,
        "recommended": True,
    },
    "deberta": {
        "name": "DeBERTa",
        "description": "Strong baseline · 95.11% test macro-F1",
        "path": ROOT / "models/deberta/baseline_v1/final_checkpoint",
        "custom": False,
    },
    "roberta": {
        "name": "RoBERTa",
        "description": "Baseline · 92.51% test macro-F1",
        "path": ROOT / "models/roberta/baseline_v1/final_checkpoint",
        "custom": False,
    },
    "bert": {
        "name": "BERT",
        "description": "Baseline · 90.99% test macro-F1",
        "path": ROOT / "models/bert/baseline_v1/final_checkpoint",
        "custom": False,
    },
    "fraudformer-roberta": {
        "name": "FraudFormer–RoBERTa",
        "description": "Custom architecture ablation · 92.04% test macro-F1",
        "path": ROOT / "models/fraudformer/experiment_v1/final_checkpoint",
        "custom": True,
    },
}


class PredictionRequest(BaseModel):
    model: str
    text: str = Field(min_length=20, max_length=100_000)


class LoadedModel:
    key: str | None = None
    tokenizer = None
    model = None
    calibration: torch.Tensor | None = None


loaded = LoadedModel()
model_lock = threading.Lock()
app = FastAPI(title="FraudFormer Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def available_models():
    return [
        {
            "id": key,
            "name": details["name"],
            "description": details["description"],
            "recommended": details.get("recommended", False),
        }
        for key, details in MODEL_REGISTRY.items()
        if (details["path"] / "config.json").exists()
    ]


def load_model(key: str):
    if key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown model")
    details = MODEL_REGISTRY[key]
    checkpoint = details["path"]
    if not (checkpoint / "config.json").exists():
        raise HTTPException(status_code=503, detail=f"Checkpoint is unavailable: {checkpoint}")
    if loaded.key == key:
        return

    # Keep only one large transformer on the accelerator at a time.
    loaded.model = None
    loaded.tokenizer = None
    loaded.calibration = None
    loaded.key = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loaded.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if details["custom"]:
        loaded.model = FraudFormerForSequenceClassification.from_pretrained(checkpoint)
    else:
        loaded.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded.model.to(device).eval()
    calibration_path = checkpoint / "calibration.json"
    if calibration_path.exists():
        bias = json.loads(calibration_path.read_text(encoding="utf-8"))["logit_bias"]
        loaded.calibration = torch.tensor(bias, dtype=torch.float, device=device)
    loaded.key = key


@app.get("/api/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.get("/api/models")
def models():
    return {"models": available_models()}


@app.post("/api/predict")
def predict(request: PredictionRequest):
    clean_text = request.text.strip()
    if len(clean_text) < 20:
        raise HTTPException(status_code=422, detail="Please provide a complete job posting")
    started = time.perf_counter()
    with model_lock:
        load_model(request.model)
        device = next(loaded.model.parameters()).device
        encoded = loaded.tokenizer(
            clean_text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = loaded.model(**encoded).logits[0]
            if loaded.calibration is not None:
                logits = logits + loaded.calibration
            probabilities = logits.softmax(-1).cpu().tolist()

    predicted = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    return {
        "model": request.model,
        "model_name": MODEL_REGISTRY[request.model]["name"],
        "label": predicted,
        "label_name": LABEL_NAMES[predicted],
        "confidence": probabilities[predicted],
        "probabilities": [
            {"label": label, "value": probability}
            for label, probability in zip(LABEL_NAMES, probabilities)
        ],
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "truncated_to_tokens": 256,
    }


# In production/local demo mode, FastAPI serves the compiled React application.
# API routes are registered first so the root static mount cannot shadow them.
frontend_dist = ROOT / "web/dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)

