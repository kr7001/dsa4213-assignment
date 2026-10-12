"""Evidence for the vocabulary decision: its size and the contraction rule.

Three questions, answered on the clean corpus:

1. How many tokens of products the vocabulary has never seen would become
   <unk>, for each vocabulary size? The products are split into two halves by
   their split key. A vocabulary built from one half is tested on the other,
   both ways round. This describes the corpus, not the final piles, so it uses
   no model and no test pile.
2. How common are contractions such as "don't" and "it's", which the course
   tokenizer keeps whole?
3. How much of the vocabulary do public GloVe vectors cover, with contractions
   kept whole or split the way GloVe's own tokenizer splits them ("do" + "n't")?

Every vocabulary also holds four special tokens, <pad>, <unk>, <bos> and
<eos>, as in Lecture4_Demo, so |V| is the number of corpus tokens kept plus 4.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
HALF = 0.5  # products with a split key below this form half A, the rest half B
SIZES = (2_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 50_000)
MIN_COUNTS = (1, 2, 3, 5, 10, 20, 50)
COVERAGE_SIZES = (10_000, 20_000, 30_000)
CHOSEN_MIN_COUNT = 5  # decision of 12 Oct: keep tokens seen 5+ times in the training set
TOP_MISSING = 25
TOP_CONTRACTIONS = 15
# GloVe's tokenizer splits "don't" into "do" + "n't" and "it's" into "it" + "'s".
CLITICS = ("'s", "'m", "'re", "'ve", "'ll", "'d")


def iter_corpus(path: Path) -> Iterator[tuple[int, list[str]]]:
    """Yield (half, tokens) for every review: half 0 if the split key is below HALF."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield (0 if row["split_key"] < HALF else 1), row["text"].split(" ")


def count_halves(path: Path) -> tuple[list[Counter], list[int]]:
    """Token counts and review counts of each half."""
    counts: list[Counter] = [Counter(), Counter()]
    reviews = [0, 0]
    for half, tokens in iter_corpus(path):
        counts[half].update(tokens)
        reviews[half] += 1
    return counts, reviews


def frequency_order(counts: Counter) -> list[str]:
    """Tokens from most to least frequent, ties broken alphabetically (Lecture4_Demo)."""
    return sorted(counts, key=lambda token: (-counts[token], token))


def worst_ranks(path: Path, orders: list[list[str]]) -> list[np.ndarray]:
    """For every review, the highest frequency rank of its tokens in the other half.

    A review has an <unk> under a vocabulary of the K most frequent tokens of
    the other half exactly when this rank is K or more. Tokens the other half
    never saw get a rank past its last token.
    """
    ranks = [{token: i for i, token in enumerate(order)} for order in orders]
    worst: list[list[int]] = [[], []]
    for half, tokens in iter_corpus(path):
        other = ranks[1 - half]
        unseen = len(orders[1 - half])
        worst[half].append(max(other.get(token, unseen) for token in tokens))
    return [np.array(values, dtype=np.int64) for values in worst]


def unknown_rates(
    build_counts: Counter,
    held_counts: Counter,
    held_worst: np.ndarray,
    sizes: tuple[int, ...] = SIZES,
) -> list[dict[str, Any]]:
    """For each vocabulary size, how much of held-out text becomes <unk>.

    The vocabulary is the ``size`` most frequent tokens of the build half.

    Args:
        build_counts: Token counts of the half the vocabulary is built from.
        held_counts: Token counts of the other half.
        held_worst: :func:`worst_ranks` of the other half's reviews.
        sizes: Vocabulary sizes to try, before the special tokens.

    Returns:
        One row per size: |V| (with the special tokens), the smallest build
        count that made it in, the <unk> share of held-out tokens and of build
        tokens, and the share of held-out reviews with at least one <unk>.
    """
    order = frequency_order(build_counts)
    held_total = sum(held_counts.values())
    build_total = sum(build_counts.values())
    rows = []
    for size in (*sizes, len(order)):
        if size > len(order):
            continue
        kept = order[:size]
        held_known = sum(held_counts[token] for token in kept)
        build_known = sum(build_counts[token] for token in kept)
        rows.append(
            {
                "kept_tokens": size,
                "vocabulary_size": size + len(SPECIAL_TOKENS),
                "smallest_count_kept": build_counts[kept[-1]],
                "held_out_unk_share": 1 - held_known / held_total,
                "build_unk_share": 1 - build_known / build_total,
                "held_out_reviews_with_unk": float((held_worst >= size).mean()),
            }
        )
    return rows


def min_count_rows(
    build_counts: Counter, held_counts: Counter, minimums: tuple[int, ...] = MIN_COUNTS
) -> list[dict[str, Any]]:
    """The <unk> share of held-out tokens for the rule "keep tokens seen k+ times"."""
    held_total = sum(held_counts.values())
    rows = []
    for minimum in minimums:
        kept = {token for token, n in build_counts.items() if n >= minimum}
        known = sum(n for token, n in held_counts.items() if token in kept)
        rows.append(
            {
                "min_count": minimum,
                "vocabulary_size": len(kept) + len(SPECIAL_TOKENS),
                "held_out_unk_share": 1 - known / held_total,
            }
        )
    return rows


def contractions(counts: Counter) -> dict[str, Any]:
    """How common tokens with an apostrophe are."""
    total = sum(counts.values())
    with_apostrophe = {token: n for token, n in counts.items() if "'" in token}
    tokens = sum(with_apostrophe.values())
    top = sorted(with_apostrophe.items(), key=lambda item: (-item[1], item[0]))
    return {
        "types": len(with_apostrophe),
        "tokens": tokens,
        "share_of_tokens": tokens / total,
        "most_common": dict(top[:TOP_CONTRACTIONS]),
    }


def split_like_glove(token: str) -> list[str]:
    """Split a contraction the way GloVe's tokenizer does: "don't" -> "do" + "n't"."""
    if token.endswith("n't") and len(token) > 3:
        return [token[:-3], "n't"]  # "can't" -> "ca" + "n't", "won't" -> "wo" + "n't"
    for clitic in CLITICS:
        if token.endswith(clitic) and len(token) > len(clitic):
            return [token[: -len(clitic)], clitic]
    return [token]


def read_glove_words(path: Path) -> set[str]:
    """The words of a GloVe file: Stanford .txt or .zip, or gensim-data .gz.

    Only the first field of each line is read. A first line made of two numbers
    (the gensim header "400000 50") is skipped.
    """

    def words(lines: Iterator[str]) -> set[str]:
        found: set[str] = set()
        for number, line in enumerate(lines):
            word = line.split(" ", 1)[0]
            if number == 0 and len(line.split()) == 2 and all(p.isdigit() for p in line.split()):
                continue
            found.add(word)
        return found

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                (m for m in archive.infolist() if m.filename.endswith(".txt")),
                key=lambda m: m.file_size,
            )
            with archive.open(members[0]) as raw:
                return words(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return words(handle)
    with path.open(encoding="utf-8", errors="replace") as handle:
        return words(handle)


def _covered(token: str, glove: set[str], split: bool) -> bool:
    parts = split_like_glove(token) if split else [token]
    return all(part in glove for part in parts)


def glove_coverage(
    build_counts: Counter,
    held_counts: Counter,
    glove: set[str],
    sizes: tuple[int, ...] = COVERAGE_SIZES,
) -> list[dict[str, Any]]:
    """How much of each vocabulary GloVe covers, with contractions whole or split.

    "Token share" weights each vocabulary entry by how often it occurs in the
    held-out half, so it says how many of the tokens a model reads would get a
    public vector.
    """
    order = frequency_order(build_counts)
    rows = []
    for size in sizes:
        kept = order[:size]
        if not kept:
            continue
        mass = sum(held_counts[token] for token in kept)
        row: dict[str, Any] = {"kept_tokens": len(kept)}
        for name, split in (("whole", False), ("split", True)):
            hits = [token for token in kept if _covered(token, glove, split)]
            row[f"types_covered_{name}"] = len(hits) / len(kept)
            row[f"token_share_covered_{name}"] = (
                sum(held_counts[t] for t in hits) / mass if mass else None
            )
        contraction_types = [token for token in kept if "'" in token]
        row["contraction_types"] = len(contraction_types)
        row["contraction_types_covered_whole"] = sum(t in glove for t in contraction_types)
        row["contraction_types_covered_split"] = sum(
            _covered(t, glove, True) for t in contraction_types
        )
        missing = [token for token in kept if not _covered(token, glove, True)]
        row["most_frequent_missing_after_split"] = {
            token: build_counts[token] for token in missing[:TOP_MISSING]
        }
        rows.append(row)
    return rows


def summarize_vocabulary(corpus: Path, glove_paths: list[Path] | None = None) -> dict[str, Any]:
    """Every number the vocabulary decision needs, as one JSON-ready dict."""
    counts, reviews = count_halves(corpus)
    orders = [frequency_order(c) for c in counts]
    worst = worst_ranks(corpus, orders)
    summary: dict[str, Any] = {
        "halves": {
            "split_key_below": HALF,
            "reviews": reviews,
            "tokens": [sum(c.values()) for c in counts],
        },
        "special_tokens": list(SPECIAL_TOKENS),
        "by_size": {
            "built_on_a_tested_on_b": unknown_rates(counts[0], counts[1], worst[1]),
            "built_on_b_tested_on_a": unknown_rates(counts[1], counts[0], worst[0]),
        },
        "by_min_count": {
            "built_on_a_tested_on_b": min_count_rows(counts[0], counts[1]),
            "built_on_b_tested_on_a": min_count_rows(counts[1], counts[0]),
        },
        "contractions": contractions(counts[0] + counts[1]),
        "chosen_rule": {
            "min_count": CHOSEN_MIN_COUNT,
            "kept_tokens_on_a": sum(1 for n in counts[0].values() if n >= CHOSEN_MIN_COUNT),
            "kept_tokens_on_whole_corpus": sum(
                1 for n in (counts[0] + counts[1]).values() if n >= CHOSEN_MIN_COUNT
            ),
        },
        "glove": {},
    }
    # Tokens seen 5+ times are exactly the first "kept_tokens_on_a" in frequency
    # order, so the chosen vocabulary is one of the sizes measured for coverage.
    sizes = tuple(sorted({*COVERAGE_SIZES, summary["chosen_rule"]["kept_tokens_on_a"]}))
    for path in glove_paths or []:
        glove = read_glove_words(path)
        summary["glove"][path.name] = {
            "words": len(glove),
            "coverage_built_on_a": glove_coverage(counts[0], counts[1], glove, sizes),
        }
    return summary


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_vocabulary(summary: dict[str, Any]) -> list[str]:
    """Render the summary as plain-text lines."""
    halves = summary["halves"]
    lines = [
        f"Halves by split key: {halves['reviews'][0]:,} and {halves['reviews'][1]:,} reviews, "
        f"{halves['tokens'][0]:,} and {halves['tokens'][1]:,} tokens.",
        "",
        "== Vocabulary size (built on half A, tested on half B; B on A in brackets) ==",
        "|V| | smallest count kept | held-out tokens -> <unk> | held-out reviews with an <unk>",
    ]
    a_rows = summary["by_size"]["built_on_a_tested_on_b"]
    b_rows = {r["kept_tokens"]: r for r in summary["by_size"]["built_on_b_tested_on_a"]}
    for row in a_rows:
        other = b_rows.get(row["kept_tokens"])
        bracket = f" ({other['held_out_unk_share']:.2%})" if other else ""
        lines.append(
            f"{row['vocabulary_size']:>7,} | {row['smallest_count_kept']:>6,} | "
            f"{row['held_out_unk_share']:.2%}{bracket} | {row['held_out_reviews_with_unk']:.1%}"
        )
    lines += ["", "== Minimum count (built on A, tested on B) =="]
    for row in summary["by_min_count"]["built_on_a_tested_on_b"]:
        lines.append(
            f"seen {row['min_count']}+ times: |V| = {row['vocabulary_size']:,}, "
            f"held-out <unk> {row['held_out_unk_share']:.2%}"
        )
    c = summary["contractions"]
    lines += [
        "",
        f"Contractions: {c['types']:,} types, {c['tokens']:,} tokens "
        f"({c['share_of_tokens']:.2%} of all tokens); most common: "
        + ", ".join(f"{t} {n:,}" for t, n in list(c["most_common"].items())[:8]),
    ]
    chosen = summary["chosen_rule"]
    lines.append(
        f"Chosen rule, seen {chosen['min_count']}+ times: {chosen['kept_tokens_on_a']:,} tokens "
        f"on half A, {chosen['kept_tokens_on_whole_corpus']:,} on the whole corpus."
    )
    for name, glove in summary["glove"].items():
        lines += ["", f"== GloVe coverage: {name} ({glove['words']:,} words) =="]
        for row in glove["coverage_built_on_a"]:
            mark = " (chosen rule)" if row["kept_tokens"] == chosen["kept_tokens_on_a"] else ""
            lines.append(
                f"{row['kept_tokens']:,} kept{mark}: types "
                f"{row['types_covered_whole']:.1%} whole / "
                f"{row['types_covered_split']:.1%} split; tokens "
                f"{_percent(row['token_share_covered_whole'])} whole / "
                f"{_percent(row['token_share_covered_split'])} split; contractions "
                f"{row['contraction_types_covered_whole']}/{row['contraction_types']} whole, "
                f"{row['contraction_types_covered_split']}/{row['contraction_types']} split"
            )
    return lines
