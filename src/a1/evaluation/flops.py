"""Arithmetic per prediction, counted from the layers (D1.7, locked 14 Oct).

Counts multiply-adds in matrix products for one predicted token and reports
FLOPs = 2 x multiply-adds, the convention of Kaplan et al. (2020, section 2.1,
Table 1). Embedding lookups, biases, norms, activations and the softmax are
left out: they are small next to the matrix products. Three parts:

- core: every weight matrix of the model except the output layer (recurrent
  matrices, attention and feed-forward layers, projections);
- output: the shared output layer, OUTPUT_DIM x |V| (Kaplan's "De-embed"),
  identical in every model;
- attention (Transformer only): each query is compared with, and averages over,
  every position it can see, 2 x d_model multiply-adds per visible position and
  layer (scores and weighted sum). With a causal mask a prediction at position t
  sees t positions, so the count uses the mean context of the workload.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from a1.models.transformer import TransformerLM


def _weight_matrices(module: nn.Module, skip: nn.Module) -> int:
    total = 0
    for child in module.modules():
        if child is skip:
            continue
        if isinstance(child, nn.Linear):
            total += child.weight.numel()
        elif isinstance(child, nn.RNNBase):
            weights = (p for name, p in child.named_parameters() if name.startswith("weight"))
            total += sum(p.numel() for p in weights)
    return total


def forward_flops(model: nn.Module, mean_context: float) -> dict[str, Any]:
    """FLOPs per prediction: core, output layer, attention and total."""
    output_layer = model.head.output
    core = _weight_matrices(model, skip=output_layer)
    output = output_layer.weight.numel()
    attention = 0.0
    if isinstance(model, TransformerLM):
        attention = 2.0 * len(model.blocks) * model.width * mean_context
    multiply_adds = core + output + attention
    return {
        "core": 2.0 * core,
        "output_layer": 2.0 * output,
        "attention": 2.0 * attention,
        "total": 2.0 * multiply_adds,
        "mean_context": mean_context,
    }
