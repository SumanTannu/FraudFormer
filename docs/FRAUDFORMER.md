# FraudFormer v1

FraudFormer is this project's custom three-class recruitment-fraud transformer. It
uses a pretrained BERT-family encoder for language knowledge, but replaces the
standard single-vector classification head with a domain-specific architecture.
This is a new project design; novelty relative to all published literature must be
established by a formal literature review and ablation study.

## Architecture

1. **Learned layer mixture** combines the last four encoder layers instead of
   assuming that the final layer is always the best task representation.
2. **Multi-scale discrepancy extractor** uses depthwise token-context filters of
   widths 3, 5, and 9 and represents the absolute disagreement between a token and
   each local context. This targets locally inserted fraud cues and stylistic shifts.
3. **Query-conditioned evidence pooling** uses the global advertisement vector to
   select the most relevant discrepancy tokens.
4. **Three-view gating** dynamically combines the first-token, masked global-mean,
   and local-evidence views for each advertisement.
5. **Imbalance-aware focal objective** uses square-root inverse-frequency weights
   calculated from the training split only. It focuses learning on the rare
   human-written-fraud class without copying or resampling records.
6. **Validation-only logit calibration** chooses two class offsets on validation
   predictions and freezes them before the test split is evaluated.

Only `model_input_text` enters the network. Provenance, source IDs, review fields,
and labels remain audit-only fields, preserving the existing leakage boundary.

## Train and evaluate

From the repository root:

```powershell
python src/train_fraudformer.py
```

The defaults use `roberta-base`, the frozen split, seed 42, a 256-token limit,
mixed precision when CUDA is available, and early stopping on validation macro-F1.
Outputs are written to:

- `models/fraudformer/experiment_v1/final_checkpoint/`
- `results/metrics/fraudformer_metrics.json`
- `results/confusion_matrices/fraudformer_confusion_matrix.png`

The comparable headline is test macro-F1. Beating a baseline should only be claimed
after the untouched test evaluation completes. For a robust research claim, repeat
the complete run with multiple seeds and report mean, standard deviation, confidence
intervals, and paired significance tests.

An optional deployment fusion can be benchmarked after the standalone model:

```powershell
python src/benchmark_fraudformer_fusion.py
```

It combines retained FraudFormer, DeBERTa, and RoBERTa predictions. Its mixture
weights and class offsets are selected strictly on validation data. Report it as an
ensemble system, separately from the standalone FraudFormer architecture.

## Inference

```powershell
python src/predict_fraudformer.py --text "Full job advertisement text..."
```

The loader automatically applies the validation-fitted logit bias saved beside the
checkpoint. Predictions are assistive risk signals, not proof that an employer or
advertisement is fraudulent.

## Recorded result

The strongest completed variant uses the fine-tuned DeBERTa encoder as a frozen
language backbone and trains the FraudFormer layers above it. The best validation
checkpoint was epoch 1; epoch 2 did not improve, and the later run was stopped when
the laptop GPU reached sustained thermal throttling. Its validation-fitted offsets
were frozen before test scoring.

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| DeBERTa baseline | 0.98449 | 0.95112 |
| FraudFormer-DeBERTa | **0.98529** | **0.95166** |

This is the best observed single-run score in the repository, an improvement of
0.00055 macro-F1. The margin is too small to establish statistical superiority;
multi-seed runs and paired bootstrap or randomization testing are still required.
The full record is `results/metrics/fraudformer_experiment_deberta_frozen_metrics.json`.
The reproducible training command is:

```powershell
python src/train_fraudformer.py --backbone models/deberta/baseline_v1/final_checkpoint --experiment-name experiment_deberta_frozen --freeze-backbone --fp32 --epochs 5 --batch-size 16 --eval-batch-size 8 --learning-rate 3e-4
```

## Required ablations for a dissertation claim

Train otherwise identical variants without: (a) layer mixing, (b) discrepancy
branches, (c) view gating, (d) focal weighting, and (e) calibration. This separates
architectural benefit from optimization and threshold effects.

