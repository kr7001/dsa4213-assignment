"""Build Part II's input tables: our own skip-gram vectors and GloVe, aligned to the vocabulary.

Usage:
    uv run python scripts/build_embeddings.py [--config configs/part2.yaml] [--force]
        [--glove path/to/glove-file]

Reads only the language-model training pile (lm_train) and the GloVe file
(the config's, or --glove: a Stanford .zip or its extracted .txt). GloVe is
read first, so a damaged download fails in seconds, before the skip-gram
training.
Writes, in data/embeddings/ (git-ignored; each is |V| x 100, row i = token id i):

- skipgram.npy: our skip-gram vectors as trained (settings: config "skipgram",
  defaults in src/a1/embedding_tables.py, from the Lecture 2 demo);
- skipgram-scaled.npy: the same vectors times one number, so their root mean
  square (RMS) matches the GloVe table's;
- glove.npy: GloVe 2024 aligned to the vocabulary, missing words filled as
  described in src/a1/embedding_tables.py;
- random.npy: uniform random rows with GloVe's RMS, a table with no information.

and results/part2/embeddings.json: the skip-gram training time (the cost of
our own embeddings, reported apart from language-model training), GloVe
coverage of the vocabulary and of training inputs, the norm check of every
table, the Lecture 2 word-similarity and analogy checks on the same words,
nearest neighbours of six fixed words, file fingerprints, the code version and
the machine. The norm check also covers the Part I LSTM's random start (seed
4213) and, if its checkpoint exists, its trained table, for comparison.

Finished outputs are kept: the script stops early if they all exist, unless
--force is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from a1.data.sequences import encode_piles, load_piles
from a1.data.vocab import Vocabulary
from a1.embedding_tables import (
    check_skipgram_settings,
    coverage,
    ensure_finite,
    glove_table,
    glove_wanted,
    informative_rows,
    input_counts,
    intrinsic_scores,
    nearest_neighbours,
    number_tokens_used,
    random_table,
    read_glove,
    review_sentences,
    rms,
    scaled_to,
    table_stats,
    train_skipgram,
)
from a1.models.heads import EMBEDDING_DIM, make_embedding
from a1.models.pretrained import file_sha256, table_sha256
from a1.provenance import code_version, machine_info, peak_process_memory

RESULTS = Path("results/part2/embeddings.json")
PART1_CHECKPOINT = Path("checkpoints/lstm-S-full.pt")
TABLES = ("skipgram", "skipgram-scaled", "glove", "random")


def log(message: str) -> None:
    print(message, flush=True)


def save_table(folder: Path, name: str, table: np.ndarray) -> dict[str, Any]:
    ensure_finite(table, name)
    path = folder / f"{name}.npy"
    np.save(path, table.astype(np.float32))
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "table_sha256": table_sha256(table),
    }


def part1_tables(vocab_size: int, seed: int) -> dict[str, np.ndarray]:
    """The Part I LSTM's table: its random start (seed 4213) and, if saved, its trained state."""
    torch.manual_seed(seed)
    tables = {"part1-random-start": make_embedding(vocab_size).weight.detach().numpy().copy()}
    if PART1_CHECKPOINT.exists():
        state = torch.load(PART1_CHECKPOINT, map_location="cpu", weights_only=True)
        tables["part1-trained"] = state["embedding.weight"].numpy()
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/part2.yaml"))
    parser.add_argument("--force", action="store_true", help="rebuild even if the outputs exist")
    parser.add_argument("--glove", type=Path, help="the GloVe file, instead of the config's")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    options = settings["embeddings"]
    folder = Path(options["out"])
    glove_path = args.glove or Path(options["glove"])
    skipgram_settings = check_skipgram_settings(options.get("skipgram", {}))
    seed = int(settings["seed"])
    outputs = [folder / f"{name}.npy" for name in TABLES] + [RESULTS]
    if not args.force and all(path.exists() for path in outputs):
        log(f"The tables and {RESULTS} already exist; nothing to do (use --force to rebuild)")
        return 0
    for path, hint in (
        (Path(settings["vocabulary"]), "run scripts/build_vocab.py first"),
        (glove_path, "download glove.2024.wikigiga.100d.zip (see README, Part II)"),
    ):
        if not path.exists():
            print(f"Missing {path}: {hint}", file=sys.stderr)
            return 1
    if skipgram_settings["vector_size"] != EMBEDDING_DIM:
        print(f"vector_size must be {EMBEDDING_DIM} (D2.2)", file=sys.stderr)
        return 1

    started = time.perf_counter()
    vocab = Vocabulary.load(Path(settings["vocabulary"]))
    piles = load_piles(Path(settings["piles"]))
    train = encode_piles(Path(settings["corpus"]), {"lm_train": piles["lm_train"]}, vocab)[
        "lm_train"
    ]
    counts = input_counts(train, len(vocab))
    log(f"|V| = {len(vocab):,}; lm_train: {len(train):,} reviews, {train.ids.size:,} tokens")

    log(f"Reading {glove_path} ...")
    vectors, numbers, glove_file = read_glove(glove_path, glove_wanted(vocab), EMBEDDING_DIM)
    glove, sources = glove_table(vocab, vectors, numbers, EMBEDDING_DIM, seed)
    glove_coverage = coverage(sources, vocab, counts)
    shares = glove_coverage["by_source"]
    log(
        f"  {glove_file['words']:,} GloVe words; rows: "
        + ", ".join(f"{s} {v['types']:,}" for s, v in shares.items())
    )
    found, words = glove_coverage["word_types_with_glove_vector"], glove_coverage["word_types"]
    log(
        f"  vocabulary words with a real GloVe vector: {found:,} of {words:,} "
        f"({glove_coverage['word_type_share_with_glove_vector']:.1%}), covering "
        f"{glove_coverage['input_share_with_glove_vector']:.2%} of training inputs"
    )

    log("Training skip-gram (one CPU thread) ...")
    sentences = review_sentences(train, vocab)
    skipgram, skipgram_record = train_skipgram(sentences, vocab, skipgram_settings)
    del sentences
    log(
        f"  {skipgram_record['tokens_processed']:,} tokens in {skipgram_record['seconds']:.0f} s "
        f"(CPU {skipgram_record['cpu_seconds']:.0f} s)"
    )

    target = rms(glove)
    skipgram_scaled, factor = scaled_to(skipgram, target)
    fixed_random = random_table(len(vocab), EMBEDDING_DIM, target, seed)
    tables = {
        "skipgram": skipgram,
        "skipgram-scaled": skipgram_scaled,
        "glove": glove,
        "random": fixed_random,
    }
    folder.mkdir(parents=True, exist_ok=True)
    files = {name: save_table(folder, name, table) for name, table in tables.items()}
    norm_check = {name: table_stats(table, counts) for name, table in tables.items()}
    for name, table in part1_tables(len(vocab), seed).items():
        norm_check[name] = table_stats(table, counts)
    log("Norm check (RMS of entries; median row length):")
    for name, stats in norm_check.items():
        log(f"  {name:<20} RMS {stats['rms']:.4f}  median row {stats['row_norm_median']:.3f}")

    rows = informative_rows(sources, vocab)
    compared = {"skipgram": skipgram, "glove": glove, "random": fixed_random}
    intrinsic = intrinsic_scores(compared, vocab, rows)
    neighbours = nearest_neighbours(compared, vocab, rows)
    log(f"Lecture 2 checks on the {intrinsic['words']:,} words every table has:")
    for name in compared:
        scores = intrinsic[name]
        rho, accuracy = scores["wordsim353_spearman"], scores["analogy_accuracy"]
        log(
            f"  {name:<9} WordSim353 rho {'n/a' if rho is None else f'{rho:.3f}'} on "
            f"{scores['wordsim353_pairs']}/{scores['wordsim353_pairs_total']} pairs; analogies "
            f"{'n/a' if accuracy is None else f'{accuracy:.1%}'} on "
            f"{scores['analogy_questions']:,}/{scores['analogy_questions_total']:,}"
        )

    summary = {
        "vocabulary_size": len(vocab),
        "lm_train": {"reviews": len(train), "tokens": int(train.ids.size)},
        "skipgram": skipgram_record,
        "glove": {
            "file": str(glove_path),
            "file_sha256": file_sha256(glove_path),
            **glove_file,
            "coverage": glove_coverage,
            "number_tokens": number_tokens_used(numbers),
            "file_has_unk_vector": "<unk>" in vectors,
            "random_rows_seed": seed,
        },
        "scale": {
            "target_rms": target,
            "skipgram_rms": rms(skipgram),
            "skipgram_factor": factor,
            "random_rms": rms(fixed_random),
        },
        "tables": files,
        "norm_check": norm_check,
        "intrinsic": intrinsic,
        "neighbours": neighbours,
        "seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak_process_memory(),
        "code": code_version(),
        "machine": machine_info("cpu"),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {', '.join(f['path'] for f in files.values())} and {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
