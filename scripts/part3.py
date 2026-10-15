"""Part III: frozen language models as feature extractors for sentiment (D3.1 to D3.3).

Usage:
    caffeinate -i uv run python scripts/part3.py 2>&1 | tee results/logs/part3.log

Run it once D3.1 to D3.3 are locked and the code is committed. It is the only
script that reads part3_test's labels, and nothing is chosen on them. Steps:

1. Data (D3.1): the two Part III piles, labelled (1-2 stars negative, 4-5
   positive, 3 left out) and balanced at random within each pile; the counts
   must match the config. Copy checks against lm_train and part3_train, and
   the five cross-validation folds, split by product.
2. Checks before any feature is used: every language model reproduces, on the
   validation pile, the best validation J its run recorded (as in
   evaluate.py), and Part II checkpoints' input tables match their runs.
3. Features (D3.2): for every trained language model, the 100 numbers that
   feed its output layer, at the last word and averaged over the words (plus
   the top hidden state for the Part I S and M models); every parameter is
   fingerprinted before and after, so the models provably stay frozen.
   Untrained copies of the S models; average word vectors of three input
   tables; word and word-pair counts; length. Language-model features are
   cached in data/part3/features/ (git-ignored).
4. Classifier (D3.3): logistic regression for every representation, C chosen
   by 5-fold cross-validation on part3_train, then one score on part3_test.
5. Analysis (D3.3): bootstrap intervals, the comparisons fixed in the config
   (McNemar, Holm), the ranking (Kendall's tau) overall, within groups and
   without the shortest reviews, the luck yardstick, accuracy by length group,
   and costs.

Outputs: results/part3/part3.json (numbers only) and per-review test
predictions in data/part3/predictions.npz (git-ignored) for the error
analysis (D3.4). No review text is printed or written.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from torch import nn  # noqa: E402

from a1.data.sequences import encode_piles, load_piles  # noqa: E402
from a1.data.vocab import Vocabulary  # noqa: E402
from a1.evaluation.perplexity import evaluate  # noqa: E402
from a1.evaluation.timing import pick_device, release_memory, synchronize  # noqa: E402
from a1.models.build import build_model, describe  # noqa: E402
from a1.models.pretrained import file_sha256, load_table, table_sha256  # noqa: E402
from a1.part3.classify import (  # noqa: E402
    ClassifierSettings,
    CountFeatures,
    DenseFeatures,
    fit_and_test,
    make_folds,
)
from a1.part3.data import (  # noqa: E402
    LabelledReviews,
    balance,
    copy_checks,
    describe_set,
    group_mask,
    group_name,
    read_part3,
)
from a1.part3.features import (  # noqa: E402
    average_vectors,
    length_feature,
    ngram_documents,
    parameter_fingerprint,
    pooled_features,
    untrained_copy,
)
from a1.part3.stats import (  # noqa: E402
    agreement_interval,
    bootstrap_accuracies,
    class_scores,
    holm,
    interval,
    mcnemar,
    pair_agreement,
    wilson,
)
from a1.provenance import code_version, machine_info  # noqa: E402

VALID_TOLERANCE = 1e-4  # as in evaluate.py
FEATURE_CACHE_VERSION = 2
CLASSIFIER_VERSION = 1  # change it if classify.py changes what a fit returns
WORD_COUNTS = "word_counts"
LENGTH = "length"


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class Context:
    config: dict[str, Any]
    vocab: Vocabulary
    train: LabelledReviews
    test: LabelledReviews
    folds: list[tuple[np.ndarray, np.ndarray]]
    device: torch.device
    settings: ClassifierSettings
    reviews_sha256: str
    folds_sha256: str
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    correct: dict[str, np.ndarray] = field(default_factory=dict)
    probability: dict[str, np.ndarray] = field(default_factory=dict)
    predicted: dict[str, np.ndarray] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def runs(self) -> Path:
        return Path(self.config["runs"])

    @property
    def checkpoints(self) -> Path:
        return Path(self.config["checkpoints"])

    @property
    def cache(self) -> Path:
        return Path(self.config["cache"])

    @property
    def layer(self) -> str:
        return str(self.config["layer"])

    @property
    def poolings(self) -> list[str]:
        return list(self.config["poolings"])

    @property
    def batch_tokens(self) -> int:
        return int(self.config["batch_tokens"])


def trained_models(config: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for group in config["models"].values():
        names.extend(name for name in group if name not in names)
    return names


def load_record(ctx: Context, name: str) -> dict[str, Any]:
    return json.loads((ctx.runs / name / "run.json").read_text())


def run_seed(record: dict[str, Any]) -> int:
    """The seed a run was trained with (Part I records keep it in the training config)."""
    return int(record.get("seed", record["summary"]["config"]["seed"]))


def load_trained(ctx: Context, name: str) -> tuple[nn.Module, dict[str, Any]]:
    """A trained model with its best weights; a Part II input table must match its run."""
    record = load_record(ctx, name)
    model = build_model(record["spec"], len(ctx.vocab))
    state = torch.load(ctx.checkpoints / f"{name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    expected = (record.get("embedding") or {}).get("end_sha256")
    if expected and table_sha256(model.embedding.weight.detach().cpu()) != expected:
        raise SystemExit(f"{name}: the checkpoint's input table is not the one its run ended with")
    return model.to(ctx.device).eval(), record


def free(model: Any, device: torch.device) -> None:
    del model
    release_memory(device)


def safe(name: str) -> str:
    return name.replace("/", "__")


# ---------------------------------------------------------------------------
# Step 1 and 2: data and checks


def read_data(
    config: dict[str, Any], vocab: Vocabulary
) -> tuple[Any, LabelledReviews, LabelledReviews]:
    piles = load_piles(Path(config["piles"]))
    part3 = {name: piles[name] for name in ("part3_train", "part3_test")}
    others = {name: pile for name, pile in piles.items() if name not in part3}
    reading = read_part3(Path(config["corpus"]), part3, others, vocab)
    if reading.shared_products:
        raise SystemExit(
            f"{reading.shared_products} products appear in a Part III pile and another"
        )
    seed = int(config["seed"])
    train = balance(reading.sets["part3_train"], seed)
    test = balance(reading.sets["part3_test"], seed)
    expected = config["expected_reviews"]
    for name, data in (("part3_train", train), ("part3_test", test)):
        if len(data) != int(expected[name]):
            raise SystemExit(
                f"{name} has {len(data):,} balanced reviews; the config expects {expected[name]:,}"
            )
    return reading, train, test


def check_models(ctx: Context, names: list[str]) -> dict[str, dict[str, float]]:
    """Every language model must be the one its training run kept (as in evaluate.py)."""
    piles = load_piles(Path(ctx.config["piles"]))
    valid = encode_piles(Path(ctx.config["corpus"]), {"lm_valid": piles["lm_valid"]}, ctx.vocab)[
        "lm_valid"
    ]
    checks: dict[str, dict[str, float]] = {}
    problems: list[str] = []
    for name in names:
        try:
            model, record = load_trained(ctx, name)
        except (SystemExit, FileNotFoundError, KeyError) as error:
            problems.append(f"{name}: {error}")
            continue
        j = evaluate(model, valid, ctx.batch_tokens, ctx.device)["J"]
        recorded = float(record["summary"]["best_valid_J"])
        checks[name] = {"valid_J": j, "valid_J_recorded": recorded, "difference": j - recorded}
        log(f"  {name}: validation J {j:.5f} (recorded {recorded:.5f})")
        if abs(j - recorded) > VALID_TOLERANCE:
            problems.append(f"{name}: validation J differs from its record by {j - recorded:.6f}")
        free(model, ctx.device)
    if problems:
        raise SystemExit("Stopped before any Part III feature:\n  " + "\n  ".join(problems))
    return checks


def fold_summary(ctx: Context) -> list[dict[str, Any]]:
    rows = []
    for fit, held in ctx.folds:
        rows.append(
            {
                "fit_reviews": int(fit.size),
                "held_out_reviews": int(held.size),
                "held_out_products": int(np.unique(ctx.train.products[held]).size),
                "held_out_positive_share": float(ctx.train.labels[held].mean()),
                "products_on_both_sides": int(
                    np.intersect1d(ctx.train.products[fit], ctx.train.products[held]).size
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Step 3 and 4: features and classifier, saved one representation at a time
#
# Every representation's classifier result is saved as soon as it is done
# (data/part3/classified/), keyed by everything that determines it: its
# features' source, the reviews, the folds and the classifier settings. A run
# that stops part-way (a closed lid, Ctrl-C) picks up where it stopped, and an
# unchanged representation is never fitted twice.


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an .npz file in one step: a stopped run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez(partial, **arrays)
    os.replace(partial, path)


def read_npz(path: Path) -> dict[str, np.ndarray] | None:
    """An .npz file's arrays, or None if it is missing or unreadable."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


def cached(
    ctx: Context, key: str, source: str, compute: Any
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Language-model features from data/part3/features/, or computed and saved there."""
    path = ctx.cache / "features" / f"{safe(key)}.npz"
    meta = {"version": FEATURE_CACHE_VERSION, "source": source, "reviews": ctx.reviews_sha256}
    stored = read_npz(path)
    if stored is not None and "info" in stored:
        info = json.loads(str(stored["info"]))
        if all(info.get(k) == v for k, v in meta.items()):
            features = {}
            for name in stored:
                if name.startswith("train__"):
                    _, layer, pooling = name.split("__")
                    features[(layer, pooling)] = (stored[name], stored[f"test__{layer}__{pooling}"])
            return features, info | {"from_cache": True}
    features, info = compute()
    arrays = {}
    for (layer, pooling), (train, test) in features.items():
        arrays[f"train__{layer}__{pooling}"] = train
        arrays[f"test__{layer}__{pooling}"] = test
    info = meta | info
    save_npz(path, info=np.asarray(json.dumps(info)), **arrays)
    return features, info | {"from_cache": False}


def extract(
    ctx: Context, model: nn.Module, layers: tuple[str, ...]
) -> tuple[dict, dict[str, float]]:
    """Features of the training and test reviews, each pass timed on its own.

    The test pass comes second, so the GPU has already prepared its programs
    for every batch width; its time is the warm cost of reading 24,362 reviews.
    """
    seconds = {}
    pooled = {}
    for part, data in (("train", ctx.train), ("test", ctx.test)):
        synchronize(ctx.device)
        start = time.perf_counter()
        pooled[part] = pooled_features(model, data.reviews, ctx.batch_tokens, ctx.device, layers)
        synchronize(ctx.device)
        seconds[f"{part}_seconds"] = time.perf_counter() - start
    return {key: (pooled["train"][key], pooled["test"][key]) for key in pooled["train"]}, seconds


def trained_features(ctx: Context, name: str, layers: tuple[str, ...]) -> dict:
    source = model_source(ctx, name, layers)

    def compute() -> tuple[dict, dict[str, Any]]:
        model, record = load_trained(ctx, name)
        before = parameter_fingerprint(model)
        features, seconds = extract(ctx, model, layers)
        if parameter_fingerprint(model) != before:
            raise SystemExit(f"{name}: a parameter changed during feature extraction")
        info = {"label": describe(record["spec"]), "parameters_unchanged": True, **seconds}
        free(model, ctx.device)
        return features, info

    if "hidden" in layers:  # up to 1,120 numbers per review: not kept on disk
        features, info = compute()
        info = info | {"source": source, "from_cache": False}
    else:
        features, info = cached(ctx, name, source, compute)
    ctx.sources.setdefault(name, {})[",".join(layers)] = info
    return features


def untrained_features(ctx: Context, name: str, layers: tuple[str, ...]) -> dict:
    record = load_record(ctx, name)
    seed = run_seed(record)
    source = model_source(ctx, name, layers, untrained=True)

    def compute() -> tuple[dict, dict[str, Any]]:
        model = untrained_copy(record["spec"], len(ctx.vocab), seed).to(ctx.device).eval()
        start_table = table_sha256(model.embedding.weight.detach().cpu())
        features, seconds = extract(ctx, model, layers)
        info = {"label": describe(record["spec"]), "seed": seed, **seconds}
        info["input_table_sha256"] = start_table
        free(model, ctx.device)
        return features, info

    features, info = cached(ctx, f"untrained/{name}", source, compute)
    ctx.sources.setdefault(f"untrained/{name}", {})[",".join(layers)] = info
    return features


def model_source(ctx: Context, name: str, layers: tuple[str, ...], untrained: bool = False) -> str:
    """What a model's features depend on: its weights (or its seed) and the layers read."""
    if untrained:
        record = load_record(ctx, name)
        weights = f"untrained {json.dumps(record['spec'], sort_keys=True)} seed {run_seed(record)}"
    else:
        weights = file_sha256(ctx.checkpoints / f"{name}.pt")
    return f"{weights}|{','.join(layers)}"


def check_untrained_procedure(ctx: Context) -> dict[str, Any]:
    """Does building with a run's seed give the start it recorded? Checked on a Part II run.

    Part I runs did not record their starting table, so the check uses the
    first Part II run that started from a random table with its own seed.
    """
    for name in trained_models(ctx.config):
        record = load_record(ctx, name)
        start = (record.get("embedding") or {}).get("start_sha256")
        if start and record["embedding"].get("table") == "random start":
            model = untrained_copy(record["spec"], len(ctx.vocab), run_seed(record))
            same = table_sha256(model.embedding.weight.detach().cpu()) == start
            if not same:
                raise SystemExit(
                    f"Rebuilding {name} with seed {run_seed(record)} gives another start"
                )
            return {"checked_on": name, "same_start": True}
    return {"checked_on": None, "same_start": None}


def table_features(ctx: Context, name: str, spec: dict[str, Any]) -> tuple[DenseFeatures, str]:
    if "checkpoint" in spec:
        model, _ = load_trained(ctx, spec["checkpoint"])
        table = model.embedding.weight.detach().cpu().numpy().astype(np.float32)
        free(model, ctx.device)
        origin = f"checkpoint {spec['checkpoint']}"
    else:
        table = load_table(Path(spec["file"]), len(ctx.vocab), 100)
        origin = spec["file"]
    fingerprint = table_sha256(table)
    ctx.sources[f"table/{name}"] = {"table": origin, "table_sha256": fingerprint}
    features = DenseFeatures(
        average_vectors(ctx.train.reviews, table), average_vectors(ctx.test.reviews, table)
    )
    return features, fingerprint


def count_features(ctx: Context) -> CountFeatures:
    settings = ctx.config["word_counts"]
    orders = tuple(int(order) for order in settings["orders"])
    vectorizer = CountVectorizer(analyzer=_identity, dtype=np.float64)
    train = vectorizer.fit_transform(ngram_documents(ctx.train.reviews, orders))
    test = vectorizer.transform(ngram_documents(ctx.test.reviews, orders))
    ctx.sources[WORD_COUNTS] = {"orders": list(orders), "ngrams_in_part3_train": train.shape[1]}
    return CountFeatures(
        train, test, int(settings["min_documents"]), bool(settings["sublinear_tf"])
    )


def _identity(document: list[str]) -> list[str]:
    return document


def result_key(ctx: Context, name: str, source: str) -> str:
    payload = {
        "representation": name,
        "source": source,
        "reviews": ctx.reviews_sha256,
        "folds": ctx.folds_sha256,
        "classifier": asdict(ctx.settings),
        "version": CLASSIFIER_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def reuse(ctx: Context, name: str, key: str) -> dict[str, Any] | None:
    """Load a representation classified by an earlier, identical run, if there is one."""
    stored = read_npz(ctx.cache / "classified" / f"{safe(name)}.npz")
    if stored is None or "result" not in stored:
        return None
    result = json.loads(str(stored["result"]))
    if result.get("key") != key:
        return None
    ctx.results[name] = result
    ctx.correct[name] = stored["correct"].astype(bool)
    ctx.probability[name] = stored["probability"]
    ctx.predicted[name] = stored["predicted"]
    log(f"  {name:52s} done in an earlier run; reused")
    return result


def classify(
    ctx: Context,
    name: str,
    features: DenseFeatures | CountFeatures,
    kind: str,
    key: str,
    source_info: dict[str, Any] | None,
) -> None:
    result = fit_and_test(features, ctx.train.labels, ctx.test.labels, ctx.folds, ctx.settings)
    correct = result.pop("correct")
    probability = result.pop("probability")
    predicted = result.pop("predicted")
    result["kind"] = kind
    result["key"] = key
    result["source_info"] = source_info  # how the features were made, kept for a reuse
    result["test"] = class_scores(ctx.test.labels, predicted)
    ctx.results[name] = result
    ctx.correct[name] = correct
    ctx.probability[name] = probability
    ctx.predicted[name] = predicted
    save_npz(
        ctx.cache / "classified" / f"{safe(name)}.npz",
        result=np.asarray(json.dumps(result, default=_plain)),
        correct=correct,
        probability=probability,
        predicted=predicted,
    )
    notes = " (at the limit of the C range)" if result["at_limit"] else ""
    if not result["final_fit_converged"] or result["unconverged_fold_fits"]:
        notes += " (a fit stopped at max_iter; see unconverged_fold_fits)"
    validated = result["mean_accuracy"][str(result["power"])]
    log(
        f"  {name:52s} C = 1e{result['power']:<3d} cross-validated {validated:.4f}  "
        f"test {result['test']['accuracy']:.4f}  ({result['seconds']:.0f} s){notes}"
    )


def language_model(
    ctx: Context, name: str, layers: tuple[str, ...], untrained: bool = False
) -> None:
    """Classify every (layer, pooling) of one model, reading its features only if needed."""
    prefix = "untrained/" if untrained else ""
    owner = f"{prefix}{name}"
    source = model_source(ctx, name, layers, untrained)
    wanted = {f"{owner}/{layer}/{pooling}": layer for layer in layers for pooling in ctx.poolings}
    keys = {rep: result_key(ctx, rep, source) for rep in wanted}
    missing = []
    for rep, layer in wanted.items():
        result = reuse(ctx, rep, keys[rep])
        if result is None:
            missing.append(rep)
        elif result.get("source_info"):
            ctx.sources.setdefault(owner, {}).setdefault(layer, result["source_info"])
    if not missing:
        return
    make = untrained_features if untrained else trained_features
    features = make(ctx, name, layers)
    info = ctx.sources[owner][",".join(layers)]
    for (layer, pooling), (train, test) in sorted(features.items()):
        rep = f"{owner}/{layer}/{pooling}"
        if rep in missing:
            if untrained:
                kind = "untrained"
            else:
                kind = "language_model" if layer == ctx.layer else "hidden_check"
            classify(ctx, rep, DenseFeatures(train, test), kind, keys[rep], info)


def reference(ctx: Context, name: str, kind: str, source: str, make: Any) -> None:
    """Classify a reference representation (tables, word counts, length) unless done before."""
    key = result_key(ctx, name, source)
    result = reuse(ctx, name, key)
    if result is not None:
        if result.get("source_info"):
            ctx.sources.setdefault(name, result["source_info"])
        return
    features = make()
    classify(ctx, name, features, kind, key, ctx.sources.get(name))


# ---------------------------------------------------------------------------
# Step 5: analysis


def resolve(ctx: Context, short: str, pooling: str) -> str:
    """A config name ('lstm-S-full', 'untrained/lstm-S-full', 'word_counts') at a pooling."""
    if short in (WORD_COUNTS, LENGTH) or short.startswith("table/"):
        return short
    return f"{short}/{ctx.layer}/{pooling}"


def analyse(ctx: Context, test_j: dict[str, float]) -> dict[str, Any]:
    names = list(ctx.results)
    matrix = np.stack([ctx.correct[name] for name in names])
    n_boot, seed = int(ctx.config["bootstrap"]), int(ctx.config["seed"])
    samples = bootstrap_accuracies(matrix, n_boot, seed)
    by_product = bootstrap_accuracies(matrix, n_boot, seed, groups=ctx.test.products)
    column = {name: i for i, name in enumerate(names)}

    for name in names:
        row = ctx.results[name]
        correct = ctx.correct[name]
        row["test"]["interval"] = interval(samples[:, column[name]])
        row["test"]["interval_by_product"] = interval(by_product[:, column[name]])
        row["test"]["by_length_group"] = {}
        for low, high in ctx.config["length_groups"]:
            mask = group_mask(ctx.test.words, (low, high))
            row["test"]["by_length_group"][group_name((low, high))] = {
                "reviews": int(mask.sum()),
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
                "interval": wilson(float(correct[mask].sum()), int(mask.sum())),
            }
        for minimum in ctx.config["minimum_words"]:
            mask = ctx.test.words >= minimum
            row["test"][f"accuracy_at_least_{minimum}_words"] = float(correct[mask].mean())

    comparisons = []
    for first, second in ctx.config["comparisons"]:
        for pooling in ctx.poolings:
            comparisons.append(
                (resolve(ctx, first, pooling), resolve(ctx, second, pooling), pooling)
            )
    for model in ctx.config["ranking"]:
        comparisons.append(
            (f"{model}/{ctx.layer}/last", f"{model}/{ctx.layer}/mean", "last vs mean")
        )
    rows = []
    for first, second, pooling in comparisons:
        test = mcnemar(ctx.correct[first], ctx.correct[second])
        difference = samples[:, column[first]] - samples[:, column[second]]
        product_difference = by_product[:, column[first]] - by_product[:, column[second]]
        rows.append(
            {
                "first": first,
                "second": second,
                "pooling": pooling,
                "first_accuracy": ctx.results[first]["test"]["accuracy"],
                "second_accuracy": ctx.results[second]["test"]["accuracy"],
                "difference": ctx.results[first]["test"]["accuracy"]
                - ctx.results[second]["test"]["accuracy"],
                "difference_interval": interval(difference),
                "difference_interval_by_product": interval(product_difference),
                **test,
            }
        )
    for row, adjusted in zip(rows, holm([row["p"] for row in rows]), strict=True):
        row["p_holm"] = adjusted
        row["significant_at_0.05"] = adjusted < 0.05
        low, high = row["difference_interval_by_product"]
        row["product_interval_excludes_0"] = bool(low > 0 or high < 0)

    ranking = {}
    for pooling in ctx.poolings:
        models = list(ctx.config["ranking"])
        reps = [f"{model}/{ctx.layer}/{pooling}" for model in models]
        j = np.asarray([test_j[model] for model in models])
        accuracy = np.asarray([ctx.results[rep]["test"]["accuracy"] for rep in reps])
        overall = pair_agreement(j, accuracy)
        overall["tau_interval"] = agreement_interval(j, samples[:, [column[r] for r in reps]])
        overall["tau_interval_by_product"] = agreement_interval(
            j, by_product[:, [column[r] for r in reps]]
        )
        groups = {}
        for group, members in ctx.config["ranking_groups"].items():
            member_j = np.asarray([test_j[m] for m in members])
            member_accuracy = np.asarray(
                [ctx.results[f"{m}/{ctx.layer}/{pooling}"]["test"]["accuracy"] for m in members]
            )
            groups[group] = pair_agreement(member_j, member_accuracy)
        without_short = {}
        for minimum in ctx.config["minimum_words"]:
            mask = ctx.test.words >= minimum
            subset = np.stack([ctx.correct[rep][mask] for rep in reps])
            agreement = pair_agreement(j, subset.mean(axis=1))
            agreement["tau_interval"] = agreement_interval(
                j, bootstrap_accuracies(subset, n_boot, seed)
            )
            agreement["test_reviews"] = int(mask.sum())
            without_short[f"at_least_{minimum}_words"] = agreement
        ranking[pooling] = {
            "models": [
                {"model": m, "test_J": float(jj), "accuracy": float(a)}
                for m, jj, a in zip(models, j, accuracy, strict=True)
            ],
            "overall": overall,
            "within_groups": groups,
            "without_short_reviews": without_short,
        }

    luck = {}
    for pooling in ctx.poolings:
        spreads = {}
        for group, members in ctx.config["seed_groups"].items():
            values = [
                ctx.results[f"{m}/{ctx.layer}/{pooling}"]["test"]["accuracy"] for m in members
            ]
            spreads[group] = {
                "accuracies": values,
                "sd": float(np.std(values, ddof=1)),
                "range": float(max(values) - min(values)),
            }
        luck[pooling] = {
            "settings": spreads,
            "mean_sd": float(np.mean([s["sd"] for s in spreads.values()])),
        }

    costs = {}
    for model in trained_models(ctx.config):
        record = load_record(ctx, model)
        extraction = ctx.sources.get(model, {}).get(ctx.layer, {})
        costs[model] = {
            "test_J": test_j.get(model),
            "training_minutes": record["summary"]["train_seconds"] / 60.0,
            "feature_seconds_test_pile": extraction.get("test_seconds"),
            "accuracy": {
                pooling: ctx.results[f"{model}/{ctx.layer}/{pooling}"]["test"]["accuracy"]
                for pooling in ctx.poolings
            },
        }
    return {"comparisons": rows, "ranking": ranking, "luck": luck, "costs": costs}


def save(path: Path, payload: dict[str, Any], device: torch.device) -> None:
    payload = {
        **payload,
        "code": code_version(),
        "machine": machine_info(str(device)),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    log(f"Wrote {path}")


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot write {type(value)} to JSON")


def print_summary(analysis: dict[str, Any]) -> None:
    log("\nRanking: does the better next-token predictor also classify better?")
    for pooling, block in analysis["ranking"].items():
        overall = block["overall"]
        low, high = overall["tau_interval"]
        tau = "n/a" if overall["tau"] is None else f"{overall['tau']:.2f}"
        p = "n/a" if overall["p"] is None else f"{overall['p']:.3g}"
        log(
            f"  {pooling:5s}: {overall['same_order']} of {overall['pairs']} pairs in the same "
            f"order; tau {tau} (95% {low:.2f} to {high:.2f}), p {p}"
        )
    log("\nLuck yardstick (sd of test accuracy across seeds, averaged over the settings):")
    for pooling, block in analysis["luck"].items():
        log(f"  {pooling:5s}: {block['mean_sd']:.4f}")
    rows = analysis["comparisons"]
    significant = sum(row["significant_at_0.05"] for row in rows)
    robust = sum(row["significant_at_0.05"] and row["product_interval_excludes_0"] for row in rows)
    log(
        f"\n{significant} of {len(rows)} fixed comparisons differ after the Holm correction; "
        f"{robust} of them also when whole products are resampled (details in part3.json)."
    )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/part3.yaml"))
    parser.add_argument("--device", help="override the config's device (tests use cpu)")
    parser.add_argument(
        "--without-hidden-check",
        action="store_true",
        help="skip the appendix check on the top hidden state (the slowest part)",
    )
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = pick_device(args.device or config.get("device"))
    vocab = Vocabulary.load(Path(config["vocabulary"]))
    test_scores = json.loads(Path(config["test_scores"]).read_text())
    test_j = {name: row["test"]["J"] for name, row in test_scores["models"].items()}
    missing = [m for m in config["ranking"] if m not in test_j]
    if missing:
        raise SystemExit(f"No test J for {missing} in {config['test_scores']}")

    log("== Data (D3.1) ==")
    started = time.perf_counter()
    reading, train, test = read_data(config, vocab)
    log(
        f"  part3_train {len(train):,} reviews, part3_test {len(test):,} (balanced); "
        f"read in {time.perf_counter() - started:.0f} s"
    )
    seed = int(config["seed"])
    classifier = config["classifier"]
    folds = make_folds(train.labels, train.products, int(classifier["folds"]), seed)
    reviews_digest = hashlib.sha256()
    for data in (train, test):
        reviews_digest.update(data.reviews.review_ids.astype(np.int64).tobytes())
    folds_digest = hashlib.sha256()
    for _, held in folds:
        folds_digest.update(held.astype(np.int64).tobytes())
    ctx = Context(
        config=config,
        vocab=vocab,
        train=train,
        test=test,
        folds=folds,
        device=device,
        settings=ClassifierSettings(
            low=int(classifier["powers"][0]),
            high=int(classifier["powers"][1]),
            lowest=int(classifier["limits"][0]),
            highest=int(classifier["limits"][1]),
            max_iter=int(classifier["max_iter"]),
        ),
        reviews_sha256=reviews_digest.hexdigest(),
        folds_sha256=folds_digest.hexdigest(),
    )
    data_record = {
        "before_balancing": {name: describe_set(s) for name, s in reading.sets.items()},
        "three_star_reviews_left_out": reading.three_stars,
        "part3_train": describe_set(train),
        "part3_test": describe_set(test),
        "products_shared_with_other_piles": reading.shared_products,
        "copies": copy_checks(test, train, reading.lm_train_texts),
        "folds": fold_summary(ctx),
    }
    copies = data_record["copies"]
    log(
        f"  test reviews whose exact text is in lm_train: {copies['share_in_lm_train']:.1%}; "
        f"in part3_train: {copies['share_in_part3_train']:.1%}"
    )

    log("\n== Checks before any Part III feature ==")
    models = trained_models(config)
    checks = check_models(ctx, models)
    procedure = check_untrained_procedure(ctx)
    log(f"  untrained copies rebuild a run's start: checked on {procedure['checked_on']}")

    log("\n== Features and classifier (D3.2, D3.3) ==")
    for name in models:
        language_model(ctx, name, (ctx.layer,))
    for name in config["untrained"]:
        language_model(ctx, name, (ctx.layer,), untrained=True)
    for name, spec in config["tables"].items():
        if "checkpoint" in spec:
            origin = file_sha256(ctx.checkpoints / f"{spec['checkpoint']}.pt")
        else:
            origin = file_sha256(Path(spec["file"]))
        reference(
            ctx,
            f"table/{name}",
            "table_average",
            f"table {origin}",
            lambda name=name, spec=spec: table_features(ctx, name, spec)[0],
        )
    words = json.dumps(config["word_counts"], sort_keys=True)
    reference(ctx, WORD_COUNTS, "word_counts", f"counts {words}", lambda: count_features(ctx))
    reference(
        ctx,
        LENGTH,
        "length",
        "log words",
        lambda: DenseFeatures(length_feature(train.words), length_feature(test.words)),
    )
    if args.without_hidden_check:
        log("\n(Appendix check on the top hidden state skipped: --without-hidden-check)")
    else:
        log("\n== Appendix check: the top hidden state (D3.2) ==")
        for name in config["hidden_check"]:
            language_model(ctx, name, ("hidden",))

    log("\n== Analysis (D3.3) ==")
    analysis = analyse(ctx, test_j)
    save(
        Path(config["results"]) / "part3.json",
        {
            "data": data_record,
            "checks": {"validation_J": checks, "untrained_start": procedure},
            "hidden_check_included": not args.without_hidden_check,
            "sources": ctx.sources,
            "representations": ctx.results,
            **analysis,
        },
        device,
    )
    names = list(ctx.results)
    predictions = ctx.cache / "predictions.npz"
    save_npz(
        predictions,
        review_ids=test.reviews.review_ids,
        labels=test.labels,
        stars=test.stars,
        words=test.words,
        products=test.products,
        names=np.asarray(names),
        probability=np.stack([ctx.probability[name] for name in names]),
        predicted=np.stack([ctx.predicted[name] for name in names]).astype(np.int8),
    )
    log(f"Wrote {predictions} (per-review test predictions; git-ignored)")
    print_summary(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
