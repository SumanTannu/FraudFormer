import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from transformers import BertConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudformer import FraudFormerConfig, FraudFormerForSequenceClassification


def tiny_model():
    backbone = BertConfig(
        vocab_size=101,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
    )
    config = FraudFormerConfig(
        backbone_config=backbone.to_dict(),
        num_labels=3,
        layer_mix_count=2,
        evidence_kernels=[3, 5],
        class_weights=[0.5, 2.0, 0.5],
    )
    return FraudFormerForSequenceClassification(config)


def test_forward_loss_and_shape():
    model = tiny_model()
    input_ids = torch.randint(0, 100, (3, 12))
    mask = torch.tensor([[1] * 12, [1] * 9 + [0] * 3, [1] * 5 + [0] * 7])
    output = model(input_ids=input_ids, attention_mask=mask, labels=torch.tensor([0, 1, 2]))
    assert output.logits.shape == (3, 3)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.layer_weights.grad is not None


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model().eval()
    input_ids = torch.randint(0, 100, (2, 8))
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        expected = model(input_ids=input_ids, attention_mask=mask).logits
    model.save_pretrained(tmp_path)
    restored = FraudFormerForSequenceClassification.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(input_ids=input_ids, attention_mask=mask).logits
    torch.testing.assert_close(actual, expected)

