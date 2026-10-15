"""Turning each review into one fixed-length list of numbers (D3.2, locked 15 Oct).

Language-model features. A review goes into a frozen language model as
<bos> w1 ... wn, with no <eos> (the models never read <eos> as input). At input
position t the model has read <bos> and the first t words, so positions 1..n
hold the states after each word and position n has read the whole review.

Two layers:
- "pre_output": the 100 numbers that feed the output layer (the projection in
  OutputHead). Every model has this layer at the same size, so every model's
  features have the same length; Wieting and Kiela (2019, Section 6) advise
  comparing encoders at "the same sentence embedding dimensionality".
- "hidden": the top hidden state h^(t) itself (100 to 1,120 numbers), as an
  appendix check on the Part I models that the 100-number layer keeps the
  sentiment.

Two poolings over positions 1..n:
- "last": the state at position n, after the last word;
- "mean": the average of the states at positions 1..n.
Position 0 (the state after <bos> alone) is the same for every review and is
never read; nor is padding, which comes after position n and, the models being
causal, cannot change any state up to n.

Reference features (D3.2): the average of the review's word vectors in an input
table E (no language model on top), word and word-pair counts, and the length
in words.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from scipy import sparse
from torch import nn

from a1.data.sequences import EncodedReviews, collate, make_batches
from a1.models.build import build_model

LAYERS = ("pre_output", "hidden")
POOLINGS = ("last", "mean")


@torch.no_grad()
def pooled_features(
    model: nn.Module,
    reviews: EncodedReviews,
    batch_tokens: int,
    device: torch.device,
    layers: tuple[str, ...] = ("pre_output",),
) -> dict[tuple[str, str], np.ndarray]:
    """Features of every review, keyed by (layer, pooling), one row per review in order."""
    unknown = set(layers) - set(LAYERS)
    if unknown:
        raise ValueError(f"Unknown layers {sorted(unknown)}; expected some of {LAYERS}")
    lengths = reviews.lengths
    if lengths.size and lengths.min() < 1:
        raise ValueError("Every review needs at least one token")
    model.eval()
    features: dict[tuple[str, str], np.ndarray] = {}
    for index in make_batches(lengths, batch_tokens):
        inputs, _ = collate(reviews, index, device)
        states = model.hidden_states(inputs)
        values = {}
        if "hidden" in layers:
            values["hidden"] = states
        if "pre_output" in layers:
            values["pre_output"] = model.head.projection(states)
        count = torch.as_tensor(lengths[index], device=device, dtype=torch.long)
        positions = torch.arange(states.size(1), device=device)
        inside = (positions[None, :] >= 1) & (positions[None, :] <= count[:, None])
        rows = torch.arange(index.size, device=device)
        for layer, layer_values in values.items():
            weights = inside.to(layer_values.dtype)[..., None]
            pooled = {
                "last": layer_values[rows, count],
                "mean": (layer_values * weights).sum(dim=1) / count[:, None].to(layer_values.dtype),
            }
            for pooling, vectors in pooled.items():
                key = (layer, pooling)
                if key not in features:
                    features[key] = np.zeros((len(reviews), vectors.size(-1)), dtype=np.float32)
                features[key][index] = vectors.float().cpu().numpy()
    return features


def untrained_copy(spec: dict, vocab_size: int, seed: int) -> nn.Module:
    """A model exactly as its training run started: same settings, same seed, no updates.

    train_lm.py seeds torch with the run's seed and then builds the model, so
    doing the same gives the starting weights; the caller checks the input
    table against the start fingerprint in the run record.
    """
    torch.manual_seed(seed)
    return build_model(spec, vocab_size)


def parameter_fingerprint(model: nn.Module) -> str:
    """SHA-256 of every parameter's float32 bytes, to show extraction changed nothing."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def token_counts(reviews: EncodedReviews, vocab_size: int) -> sparse.csr_matrix:
    """How often each token occurs in each review: a reviews x |V| sparse matrix."""
    rows = np.repeat(np.arange(len(reviews)), reviews.lengths)
    values = np.ones(rows.size, dtype=np.float64)
    return sparse.csr_matrix(
        (values, (rows, reviews.ids.astype(np.int64))), shape=(len(reviews), vocab_size)
    )


def average_vectors(reviews: EncodedReviews, table: np.ndarray) -> np.ndarray:
    """Each review's average word vector from an input table E (|V| x d), as float32."""
    counts = token_counts(reviews, table.shape[0])
    lengths = reviews.lengths.astype(np.float64)[:, None]
    return np.asarray((counts @ table.astype(np.float64)) / lengths, dtype=np.float32)


def ngram_documents(reviews: EncodedReviews, orders: tuple[int, ...] = (1, 2)) -> list[list[str]]:
    """Each review as its token n-grams ("12", "12 845", ...), in the shared vocabulary.

    The language models see rare words as <unk>, so the counts use the same
    token ids and see exactly the same words.
    """
    documents = []
    for i in range(len(reviews)):
        tokens = [str(token) for token in reviews.review(i).tolist()]
        grams: list[str] = []
        for order in orders:
            grams.extend(" ".join(tokens[k : k + order]) for k in range(len(tokens) - order + 1))
        documents.append(grams)
    return documents


def length_feature(words: np.ndarray) -> np.ndarray:
    """Length in words on a log scale, as a one-column feature."""
    return np.log1p(np.asarray(words, dtype=np.float64))[:, None].astype(np.float32)
