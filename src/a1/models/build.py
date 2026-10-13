"""Build a model from a config and count its parameters the way the report does.

A config is a dict such as {"family": "lstm", "hidden": 210, "layers": 1}.
Parameter counts follow D0.6 part 2:

- word tables: the input table E and the output layer, identical in every model;
- core: everything else (recurrent or attention layers, projections, norms);
- total: both together.

Kaplan et al. (2020) leave the vocabulary tables out of the model size because
that "produces significantly cleaner scaling laws"; Lecture3_Demo's exit
problem 1 also counts weights "ignoring biases and the common embedding/output
layers". The three models' core counts are kept within 10% of each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from a1.models.recurrent import RecurrentLM
from a1.models.transformer import TransformerLM

FAMILIES = ("rnn", "lstm", "transformer")


def build_model(config: dict[str, Any], vocab_size: int) -> nn.Module:
    """A fresh model with the given settings."""
    family = config["family"]
    dropout = float(config.get("dropout", 0.0))
    if family in ("rnn", "lstm"):
        return RecurrentLM(
            vocab_size,
            cell=family,
            hidden=int(config["hidden"]),
            layers=int(config.get("layers", 1)),
            dropout=dropout,
            forget_bias=float(config.get("forget_bias", 1.0)),
        )
    if family == "transformer":
        return TransformerLM(
            vocab_size,
            width=int(config["width"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            feedforward=int(config["feedforward"]),
            dropout=dropout,
            activation=config.get("activation", "relu"),
        )
    raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}")


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total, word-table and core parameter counts, plus how many are trainable."""
    total = sum(p.numel() for p in model.parameters())
    output = model.head.output
    tables = model.embedding.weight.numel() + output.weight.numel() + output.bias.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "word_tables": tables, "core": total - tables, "trainable": trainable}


def describe(config: dict[str, Any]) -> str:
    """A short label such as 'lstm H=210 x1' for tables and logs.

    Part II settings add the input table and the seed, for example
    'lstm H=248 x1, E glove.npy frozen, seed 4214'.
    """
    family = config["family"]
    if family == "transformer":
        label = (
            f"transformer d={config['width']} x{config['layers']} "
            f"h{config['heads']} ff{config['feedforward']}"
        )
    else:
        label = f"{family} H={config['hidden']} x{config.get('layers', 1)}"
    if "embedding" in config or "freeze" in config:
        table = Path(config["embedding"]).name if config.get("embedding") else "random"
        label += f", E {table} {'frozen' if config.get('freeze') else 'trainable'}"
    if "seed" in config:
        label += f", seed {config['seed']}"
    return label
