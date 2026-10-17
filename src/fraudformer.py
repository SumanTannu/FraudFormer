"""FraudFormer: a domain-specific transformer for recruitment-fraud detection.

The encoder can inherit a pretrained BERT-family backbone, while the task head is
purpose-built for long, internally inconsistent job advertisements.  It mixes the
last encoder layers, compares each token with several local contexts, pools the
most suspicious evidence, and learns how much to trust global versus local views.
"""
from __future__ import annotations

import os
from typing import Iterable

# Keep this PyTorch-only module isolated from optional, incompatible TF/JAX installs.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class FraudFormerConfig(PretrainedConfig):
    model_type = "fraudformer"

    def __init__(
        self,
        backbone_config: dict | None = None,
        num_labels: int = 3,
        layer_mix_count: int = 4,
        evidence_kernels: Iterable[int] = (3, 5, 9),
        dropout: float = 0.15,
        focal_gamma: float = 1.5,
        class_weights: Iterable[float] | None = None,
        id2label: dict | None = None,
        label2id: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            num_labels=num_labels, id2label=id2label, label2id=label2id, **kwargs
        )
        self.backbone_config = backbone_config
        self.layer_mix_count = layer_mix_count
        self.evidence_kernels = list(evidence_kernels)
        self.dropout = dropout
        self.focal_gamma = focal_gamma
        self.class_weights = list(class_weights) if class_weights is not None else None


def _restore_backbone_config(payload: dict):
    payload = dict(payload)
    model_type = payload.pop("model_type")
    # These serialization fields are not constructor arguments on every config.
    payload.pop("transformers_version", None)
    payload.pop("_name_or_path", None)
    return AutoConfig.for_model(model_type, **payload)


class MultiScaleDiscrepancy(nn.Module):
    """Detect tokens that disagree with short-, medium-, and long-range context."""

    def __init__(self, hidden_size: int, kernels: Iterable[int], dropout: float):
        super().__init__()
        kernels = list(kernels)
        if not kernels or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("evidence_kernels must be non-empty positive odd integers")
        self.contexts = nn.ModuleList(
            nn.Conv1d(hidden_size, hidden_size, k, padding=k // 2, groups=hidden_size)
            for k in kernels
        )
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * len(kernels), hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        channels_first = hidden.transpose(1, 2)
        discrepancies = [
            (hidden - convolution(channels_first).transpose(1, 2)).abs()
            for convolution in self.contexts
        ]
        return self.combine(torch.cat(discrepancies, dim=-1)) * mask.unsqueeze(-1)


class FraudFormerForSequenceClassification(PreTrainedModel):
    config_class = FraudFormerConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    def __init__(self, config: FraudFormerConfig, backbone: nn.Module | None = None):
        super().__init__(config)
        if config.backbone_config is None:
            raise ValueError("FraudFormerConfig.backbone_config is required")
        backbone_config = _restore_backbone_config(config.backbone_config)
        backbone_config.output_hidden_states = True
        self.backbone = backbone if backbone is not None else AutoModel.from_config(backbone_config)
        hidden_size = backbone_config.hidden_size
        available_layers = int(getattr(backbone_config, "num_hidden_layers", 1)) + 1
        self.mix_count = min(config.layer_mix_count, available_layers)
        self.layer_weights = nn.Parameter(torch.zeros(self.mix_count))
        self.discrepancy = MultiScaleDiscrepancy(
            hidden_size, config.evidence_kernels, config.dropout
        )
        self.evidence_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.global_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.evidence_score = nn.Linear(hidden_size, 1, bias=False)
        self.view_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.GELU(), nn.Linear(hidden_size, 3)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_labels),
        )
        weights = config.class_weights or [1.0] * config.num_labels
        if len(weights) != config.num_labels:
            raise ValueError("class_weights must contain one value per label")
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float))

    @classmethod
    def from_backbone_pretrained(cls, backbone_name: str, **config_kwargs):
        backbone = AutoModel.from_pretrained(backbone_name)
        backbone.config.output_hidden_states = True
        config = FraudFormerConfig(
            backbone_config=backbone.config.to_dict(), **config_kwargs
        )
        return cls(config, backbone=backbone)

    def _set_gradient_checkpointing(self, module, value=False):
        if module is self.backbone and hasattr(module, "gradient_checkpointing_enable"):
            if value:
                module.gradient_checkpointing_enable()
            else:
                module.gradient_checkpointing_disable()

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=-1)
        rows = torch.arange(labels.shape[0], device=labels.device)
        true_log_probability = log_probabilities[rows, labels]
        focal_factor = (1.0 - true_log_probability.exp()).pow(self.config.focal_gamma)
        return (
            -true_log_probability
            * focal_factor
            * self.class_weights.to(logits.device)[labels]
        ).mean()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ):
        # Recent Trainer versions pass this loss-normalization hint to models
        # accepting **kwargs; it is not a valid backbone argument.
        kwargs.pop("num_items_in_batch", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        backbone_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            backbone_inputs["token_type_ids"] = token_type_ids
        outputs = self.backbone(
            **backbone_inputs,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
            **kwargs,
        )
        selected = torch.stack(outputs.hidden_states[-self.mix_count :], dim=0)
        hidden = torch.einsum("l,lbsd->bsd", self.layer_weights.softmax(0), selected)
        float_mask = attention_mask.to(hidden.dtype)
        masked_hidden = hidden * float_mask.unsqueeze(-1)
        global_mean = masked_hidden.sum(1) / float_mask.sum(1, keepdim=True).clamp_min(1.0)
        cls_view = hidden[:, 0]

        discrepancy = self.discrepancy(hidden, float_mask)
        evidence_logits = self.evidence_score(
            torch.tanh(
                self.evidence_key(discrepancy) + self.global_query(cls_view).unsqueeze(1)
            )
        ).squeeze(-1)
        evidence_logits = evidence_logits.masked_fill(attention_mask == 0, -1e4)
        evidence_attention = evidence_logits.softmax(dim=-1)
        evidence_view = torch.bmm(evidence_attention.unsqueeze(1), discrepancy).squeeze(1)

        views = torch.stack((cls_view, global_mean, evidence_view), dim=1)
        gate = self.view_gate(torch.cat((cls_view, global_mean, evidence_view), dim=-1))
        fused = (views * gate.softmax(-1).unsqueeze(-1)).sum(1)
        logits = self.classifier(torch.cat((fused, cls_view, evidence_view), dim=-1))
        loss = self._loss(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

