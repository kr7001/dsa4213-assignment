"""Checks on the clean corpus that scripts/prepare_data.py writes.

Every check that backs a preprocessing decision or a stated limitation, run on
the file the models will read:

1. Structure: the file matches the report (rows, tokens, fingerprint), every
   token fits the tokenizer, lengths are inside the step 3 limits, no product
   is over the cap, and each product has a single split key.
2. Duplicates: no person repeats a text, no text of 14+ words appears twice,
   and how often different people wrote near-identical reviews, by length.
   That last count is the evidence for the 14-word line in dedup.py.
3. Leftovers: what the cleaning could not fix or chose to keep: broken
   accented words, split contractions, runs of punctuation, markup traces,
   repetitive reviews and leftover non-English function words. It also looks
   for email addresses, which cleaning removes, to confirm none are left.

Near-identical reviews are found by an exact similarity join with prefix
filtering (Chaudhuri, Ganti and Kaushik, 2006; Bayardo, Ma and Srikant, 2007):
it finds every pair of distinct texts whose sets of distinct words overlap by
90% or more (Jaccard), without comparing all pairs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from a1.data.clean import TOKEN_PATTERN
from a1.data.dedup import LONG_MIN_WORDS, NEAR_COPY_JACCARD, OTHER_PERSON_MIN_WORDS

MAX_TOKENS = 512

# Near-identical: two different texts whose distinct words overlap by 90%+
# (Jaccard), the same measure and level as the same-person near-copy rule.
NEAR_OVERLAP = NEAR_COPY_JACCARD
# Safety stop. Natural text needs about 20 million comparisons here; text with
# a tiny vocabulary (such as synthetic test data) could need billions.
MAX_COMPARISONS = 100_000_000

# Accented words the a-z tokenizer breaks, and the tokens they become.
ACCENT_FRAGMENTS = {
    "fiancé(e)": "fianc",
    "crème": "cr me",
    "L'Oréal": "l'or al",
    "Lancôme": "lanc me",
    "décolleté": "d collet",
    "décolletage": "d colletage",
    "soufflé": "souffl",
    "rosé": "ros",
    "brûlée": "br l",
    "jalapeño": "jalape o",
    "Hermès": "herm s",
    "piña colada": "pi a colada",
    "protégé": "prot",
    "sauté": "saut",
    "glacé": "glac",
    "provençal": "proven al",
    "cliché": "clich",
    "Avène": "av ne",
    "entrée": "entr e",
    "purée": "pur e",
    "café": "caf",
    "Kérastase": "k rastase",
    "Chloé": "chlo",
    "naïve": "na ve",
    "résumé": "r sum",
}

# Contractions: (first part, second part). Whole: "don't"; split: "don t".
_HEADS = {
    "t": "don doesn didn can isn won wasn aren haven wouldn couldn shouldn hasn weren ain hadn "
    "mustn needn",
    "s": "it that there what he she who let here where everyone",
    "m": "i",
    "ve": "i you we they could would should might",
    "ll": "i you we they it she he that",
    "re": "you we they",
    "d": "i you we they she he",
}
CONTRACTIONS = [(head, tail) for tail, heads in _HEADS.items() for head in heads.split(" ")]

PUNCTUATION = frozenset(".,!?;:")
LONG_RUN = 4  # a run of 4+ identical marks, such as "! ! ! !"

# An email address that escaped cleaning would read "name gmail . com" after
# tokenizing (the tokenizer drops "@"). Cleaning removes them, so expect none.
_EMAIL = re.compile(
    r"(?:^| )(?:<num>|[a-z][a-z']*) (?:gmail|yahoo|hotmail|aol|outlook|icloud|msn|comcast|live"
    r"|ymail|protonmail|mail) \. (?:com|net|org)(?= |$)"
)

MARKUP_TOKENS = (
    "br", "nbsp", "quot", "amp", "lt", "gt", "videoid", "asin", "http", "https", "www", "href",
)  # fmt: skip

REPETITIVE_MIN_WORDS = 20  # a review of 20+ words ...
REPETITIVE_SHARE = 0.5  # ... where one word is more than half of the words

# Function words that are rare in English, per language, written as the
# tokenizer leaves them (no accents). A review with 4+ distinct ones is flagged.
FOREIGN_WORDS = {
    "es": "el la los las que es muy pero para por con una producto me encanta bien bueno lo mi "
    "esta este todo cabello piel recomiendo",
    "fr": "les avec une des mais tres vraiment c'est j'ai nous pour dans aussi sont cette produit "
    "je est pas bien",
    "pt": "muito produto uma bom recomendo voce nao isso mais bem otimo ele ela cabelo pele "
    "gostei para com",
    "it": "molto prodotto sono della questo buono ottimo anche perche consiglio capelli pelle "
    "bello che non per con",
    "de": "und ist nicht sehr das ich mit auch gut haare haut aber nur wie schon produkt kann "
    "bei eine einen",
    "tl": "ang ng mga sa maganda po ko talaga naman siya yung ganda salamat lang pero kasi",
}
FOREIGN_MIN_WORDS = 4
EXAMPLES = 5


@dataclass
class Corpus:
    """The clean corpus, one entry per review, plus its fingerprint."""

    ids: np.ndarray
    product: np.ndarray  # product index into product_names
    user: np.ndarray  # user index into user_names
    time: np.ndarray
    stars: np.ndarray
    split_key: np.ndarray
    text: list[str]
    product_names: list[str]
    user_names: list[str]
    schemas: Counter
    sha256: str

    def __len__(self) -> int:
        return len(self.text)


def load_corpus(path: Path) -> Corpus:
    """Read the corpus and compute the SHA-256 of its uncompressed content."""
    digest = hashlib.sha256()
    columns: dict[str, list] = defaultdict(list)
    products: dict[str, int] = {}
    users: dict[str, int] = {}
    schemas: Counter = Counter()
    with gzip.open(path, "rb") as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            schemas[tuple(sorted(row))] += 1
            columns["ids"].append(row["id"])
            columns["product"].append(products.setdefault(row["product"], len(products)))
            columns["user"].append(users.setdefault(row["user"], len(users)))
            columns["time"].append(row["time"])
            columns["stars"].append(row["stars"])
            columns["split_key"].append(row["split_key"])
            columns["text"].append(row["text"])
    return Corpus(
        ids=np.array(columns["ids"], dtype=np.int64),
        product=np.array(columns["product"], dtype=np.int64),
        user=np.array(columns["user"], dtype=np.int64),
        time=np.array(columns["time"], dtype=np.int64),
        stars=np.array(columns["stars"], dtype=np.int64),
        split_key=np.array(columns["split_key"], dtype=float),
        text=columns["text"],
        product_names=list(products),
        user_names=list(users),
        schemas=schemas,
        sha256=digest.hexdigest(),
    )


def word_list(text: str) -> list[str]:
    """The word tokens of a corpus text (no punctuation, no "<num>")."""
    return [token for token in text.split() if token[0].isalpha()]


def structure(corpus: Corpus, report: dict[str, Any] | None) -> dict[str, Any]:
    """Check the file against the report and against every step's limits.

    Args:
        corpus: Output of :func:`load_corpus`.
        report: results/preprocess/report.json, or None to skip the comparison.

    Returns:
        Every structural fact, with the expected value where there is one.
    """
    tokens = np.zeros(len(corpus), dtype=np.int64)
    words = np.zeros(len(corpus), dtype=np.int64)
    invalid: Counter = Counter()
    for i, text in enumerate(corpus.text):
        parts = text.split(" ")
        tokens[i] = len(parts)
        words[i] = sum(1 for token in parts if token[0].isalpha())
        for token in parts:
            if not TOKEN_PATTERN.fullmatch(token):
                invalid[token] += 1
    per_product = np.bincount(corpus.product)
    keys_per_product = [set() for _ in corpus.product_names]
    for product, key in zip(corpus.product, corpus.split_key, strict=True):
        keys_per_product[product].add(float(key))
    product_keys = np.array([min(keys) for keys in keys_per_product])
    years = Counter(datetime.fromtimestamp(t / 1000, tz=UTC).year for t in corpus.time)
    per_user = np.bincount(corpus.user)
    cap = None if report is None else report["cap"]["cap"]
    expected = None if report is None else report["final"]
    fingerprint = None if report is None or report.get("corpus") is None else report["corpus"]
    return {
        "rows": len(corpus),
        "tokens": int(tokens.sum()),
        "matches_report": None
        if expected is None
        else {
            "rows": len(corpus) == expected["reviews"],
            "tokens": int(tokens.sum()) == expected["tokens"],
            "fingerprint": None
            if fingerprint is None
            else corpus.sha256 == fingerprint["sha256_uncompressed"],
        },
        "sha256_uncompressed": corpus.sha256,
        "schemas": {", ".join(keys): n for keys, n in corpus.schemas.items()},
        "ids_unique_and_increasing": bool(np.all(np.diff(corpus.ids) > 0)),
        "invalid_tokens": sum(invalid.values()),
        "invalid_token_examples": [token for token, _ in invalid.most_common(EXAMPLES)],
        "fewest_words": int(words.min()),
        "most_tokens": int(tokens.max()),
        "within_length_limits": bool(words.min() >= 1 and tokens.max() <= MAX_TOKENS),
        "words_percentiles": _percentiles(words),
        "tokens_percentiles": _percentiles(tokens),
        "stars": {str(s): int((corpus.stars == s).sum()) for s in range(1, 6)},
        "years": [min(years), max(years)],
        "products": len(corpus.product_names),
        "most_reviews_per_product": int(per_product.max()),
        "products_at_cap": None if cap is None else int((per_product == cap).sum()),
        "within_cap": None if cap is None else bool(per_product.max() <= cap),
        "products_with_several_split_keys": sum(1 for keys in keys_per_product if len(keys) > 1),
        "split_keys_per_tenth": np.histogram(product_keys, bins=10, range=(0, 1))[0].tolist(),
        "users": len(corpus.user_names),
        "share_of_reviews_by_users_with_2_plus": float((per_user[corpus.user] > 1).mean()),
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    points = (1, 5, 25, 50, 75, 95, 99)
    return {f"p{p}": float(v) for p, v in zip(points, np.percentile(values, points), strict=True)}


@dataclass
class NearPair:
    """Two different texts sharing most of their distinct words."""

    first: int  # a review with the first text
    second: int  # a review with the second text
    words: int  # words in the longer of the two
    overlap: float  # Jaccard overlap of their distinct words
    same_product: bool  # whether any review of one shares a product with the other
    same_person: bool  # whether any user wrote both texts


def exact_repeats(corpus: Corpus) -> dict[str, Any]:
    """Count texts that appear more than once, by who wrote them and their length."""
    by_text: dict[str, list[int]] = defaultdict(list)
    for i, text in enumerate(corpus.text):
        by_text[text].append(i)
    same_person = 0
    long_other = 0
    kept_other: Counter = Counter()
    for text, members in by_text.items():
        if len(members) < 2:
            continue
        users = [int(corpus.user[i]) for i in members]
        same_person += len(users) - len(set(users))
        words = len(word_list(text))
        extra = len(set(users)) - 1
        if words >= OTHER_PERSON_MIN_WORDS:
            long_other += extra
        elif extra:
            kept_other[_length_label(words)] += extra
    return {
        "same_person_same_text": same_person,
        f"copies_of_{OTHER_PERSON_MIN_WORDS}_plus_words_by_other_people": long_other,
        "copies_by_other_people_kept": dict(sorted(kept_other.items(), key=_label_order)),
    }


def _length_label(words: int) -> str:
    return "1-6" if words < LONG_MIN_WORDS else str(words)


def _label_order(item: tuple[str, int]) -> int:
    return 0 if item[0] == "1-6" else int(item[0])


def near_identical_pairs(
    corpus: Corpus,
    min_words: int = LONG_MIN_WORDS,
    threshold: float = NEAR_OVERLAP,
    max_comparisons: int = MAX_COMPARISONS,
) -> tuple[list[NearPair], bool]:
    """Find every pair of different texts of ``min_words``+ words that nearly match.

    Each distinct text is compared once, so exact copies count as one text. Two
    texts nearly match when their sets of distinct words overlap by at least
    ``threshold`` (Jaccard). The search is exact: prefix filtering only skips
    pairs that cannot reach the threshold. Words are ordered from rarest to most
    common; if two sets overlap by ``threshold`` or more, their first
    ``size - ceil(threshold * size) + 1`` words must share at least one word, so
    only texts sharing one of those words are compared. Texts are taken from
    smallest to largest, and a pair whose sizes are too far apart is skipped.

    Args:
        corpus: Output of :func:`load_corpus`.
        min_words: Shortest text compared, in words.
        threshold: Smallest Jaccard overlap of distinct words reported.
        max_comparisons: Stop early after this many comparisons.

    Returns:
        Every such pair, by the same person or not (see ``NearPair.same_person``),
        and whether the search finished (False if it hit ``max_comparisons``).
    """
    first_review: dict[str, int] = {}
    users: dict[str, set[int]] = defaultdict(set)
    products: dict[str, set[int]] = defaultdict(set)
    for i, text in enumerate(corpus.text):
        first_review.setdefault(text, i)
        users[text].add(int(corpus.user[i]))
        products[text].add(int(corpus.product[i]))

    word_ids: dict[str, int] = {}
    texts: list[str] = []
    sizes: list[int] = []
    sets: list[frozenset[int]] = []
    for text in first_review:
        words = word_list(text)
        if len(words) < min_words:
            continue
        texts.append(text)
        sizes.append(len(words))
        sets.append(frozenset(word_ids.setdefault(word, len(word_ids)) for word in words))
    frequency = Counter(word for word_set in sets for word in word_set)

    def rarest_first(word: int) -> tuple[int, int]:
        return frequency[word], word

    index: dict[int, list[int]] = defaultdict(list)
    skip: dict[int, int] = defaultdict(int)  # entries too small for every later text
    pairs: list[NearPair] = []
    comparisons = 0
    for i in sorted(range(len(sets)), key=lambda k: len(sets[k])):
        x = sets[i]
        size = len(x)
        prefix = sorted(x, key=rarest_first)[: size - math.ceil(threshold * size - 1e-9) + 1]
        smallest = threshold * size - 1e-9
        candidates: set[int] = set()
        for word in prefix:
            entries = index[word]
            start = skip[word]
            while start < len(entries) and len(sets[entries[start]]) < smallest:
                start += 1
            skip[word] = start
            candidates.update(entries[start:])
        comparisons += len(candidates)
        if comparisons > max_comparisons:
            return pairs, False
        for j in candidates:
            y = sets[j]
            shared = len(x & y)
            overlap = shared / (size + len(y) - shared)
            if overlap >= threshold:
                first, second = texts[j], texts[i]
                pairs.append(
                    NearPair(
                        first=first_review[first],
                        second=first_review[second],
                        words=max(sizes[i], sizes[j]),
                        overlap=overlap,
                        same_product=not products[first].isdisjoint(products[second]),
                        same_person=not users[first].isdisjoint(users[second]),
                    )
                )
        for word in prefix:
            index[word].append(i)
    return pairs, True


def summarize_near_pairs(pairs: list[NearPair], complete: bool = True) -> dict[str, Any]:
    """Count near-identical pairs by different people, by the longer text's length."""
    other = [pair for pair in pairs if not pair.same_person]
    by_length = Counter(pair.words for pair in other)
    line = OTHER_PERSON_MIN_WORDS
    above = [pair for pair in other if pair.words >= line]
    return {
        "search_complete": complete,
        "overlap_at_least": NEAR_OVERLAP,
        "pairs_by_other_people": len(other),
        "by_words_of_longer_review": {str(k): by_length[k] for k in sorted(by_length)},
        f"pairs_below_{line}_words": len(other) - len(above),
        f"pairs_of_{line}_plus_words": len(above),
        f"longest_pair_below_{line}_words": max(
            (pair.words for pair in other if pair.words < line), default=None
        ),
        f"shortest_pair_of_{line}_plus_words": min((pair.words for pair in above), default=None),
        f"pairs_of_{line}_plus_words_on_the_same_product": sum(p.same_product for p in above),
        "pairs_by_the_same_person": len(pairs) - len(other),
    }


def leftovers(corpus: Corpus) -> dict[str, Any]:
    """Count what the cleaning could not fix or chose to keep."""
    tokens_total = 0
    vocabulary: Counter = Counter()
    split_pairs = frozenset(CONTRACTIONS)
    split = 0
    accent_reviews: Counter = Counter()
    accent_any = 0
    contraction_reviews = 0
    accent_needles = {name: f" {tokens} " for name, tokens in ACCENT_FRAGMENTS.items()}
    run_tokens: Counter = Counter()
    run_reviews = 0
    longest_run = 0
    email_reviews = 0
    repetitive = 0
    foreign_sets = {lang: frozenset(words.split()) for lang, words in FOREIGN_WORDS.items()}
    foreign: Counter = Counter()
    for text in corpus.text:
        parts = text.split(" ")
        tokens_total += len(parts)
        vocabulary.update(parts)
        found = sum(1 for pair in zip(parts, parts[1:], strict=False) if pair in split_pairs)
        split += found
        contraction_reviews += found > 0
        padded = f" {text} "
        hits = [name for name, needle in accent_needles.items() if needle in padded]
        accent_reviews.update(hits)
        accent_any += bool(hits)
        has_run = False
        start = 0
        while start < len(parts):
            end = start
            while end + 1 < len(parts) and parts[end + 1] == parts[start]:
                end += 1
            length = end - start + 1
            if parts[start] in PUNCTUATION and length >= LONG_RUN:
                run_tokens[parts[start]] += length
                longest_run = max(longest_run, length)
                has_run = True
            start = end + 1
        run_reviews += has_run
        email_reviews += bool(_EMAIL.search(text))
        words = [token for token in parts if token[0].isalpha()]
        if len(words) >= REPETITIVE_MIN_WORDS:
            top = Counter(words).most_common(1)[0][1]
            repetitive += top / len(words) > REPETITIVE_SHARE
        distinct = set(words)
        for lang, markers in foreign_sets.items():
            if len(distinct & markers) >= FOREIGN_MIN_WORDS:
                foreign[lang] += 1
    whole = sum(vocabulary[f"{a}'{b}"] for a, b in CONTRACTIONS)
    reviews = len(corpus)
    return {
        "accent_fragments": {
            "reviews_with_any": accent_any,
            "share_of_reviews": accent_any / reviews,
            "reviews_by_word": dict(accent_reviews.most_common()),
            "note": "only the listed words are searched, so this is a lower bound",
        },
        "contractions": {
            "whole": whole,
            "split": split,
            "share_split": split / (split + whole) if split + whole else None,
            "reviews_with_a_split_one": contraction_reviews,
        },
        "punctuation_runs": {
            f"tokens_in_runs_of_{LONG_RUN}_plus": sum(run_tokens.values()),
            "share_of_tokens": sum(run_tokens.values()) / tokens_total,
            "by_mark": dict(run_tokens.most_common()),
            "reviews": run_reviews,
            "share_of_reviews": run_reviews / reviews,
            "longest_run": longest_run,
        },
        "email_addresses": {"reviews": email_reviews},
        "markup_tokens": {token: vocabulary[token] for token in MARKUP_TOKENS},
        "num_tokens": vocabulary["<num>"],
        "repetitive_reviews": repetitive,
        f"reviews_with_{FOREIGN_MIN_WORDS}_plus_foreign_function_words": {
            lang: foreign[lang] for lang in FOREIGN_WORDS
        },
    }


def run_checks(corpus: Corpus, report: dict[str, Any] | None) -> tuple[dict[str, Any], list]:
    """Run every check. Returns the results and the near-identical pairs."""
    pairs, complete = near_identical_pairs(corpus)
    results = {
        "structure": structure(corpus, report),
        "duplicates": {
            "exact": exact_repeats(corpus),
            "near_identical": summarize_near_pairs(pairs, complete),
        },
        "leftovers": leftovers(corpus),
    }
    return results, pairs


def example_lines(corpus: Corpus, pairs: list[NearPair]) -> dict[str, list[str]]:
    """Readable examples of the near-identical pairs, for checking by eye."""
    by_length: dict[int, list[NearPair]] = defaultdict(list)
    for pair in sorted(pairs, key=lambda pair: (-pair.overlap, pair.first, pair.second)):
        if not pair.same_person:
            by_length[pair.words].append(pair)

    def show(pair: NearPair) -> str:
        where = "same product" if pair.same_product else "different products"
        return (
            f"{pair.words} words, {pair.overlap:.0%} shared, {where}\n"
            f"    A: {corpus.text[pair.first][:200]}\n"
            f"    B: {corpus.text[pair.second][:200]}"
        )

    below = [
        show(pair)
        for words in sorted(by_length)
        if words < OTHER_PERSON_MIN_WORDS
        for pair in by_length[words][:2]
    ]
    above = [
        show(pair)
        for words in sorted(by_length)
        if words >= OTHER_PERSON_MIN_WORDS
        for pair in by_length[words]
    ]
    return {
        f"Near-identical reviews by different people, below {OTHER_PERSON_MIN_WORDS} words "
        "(two per length)": below,
        f"Near-identical reviews by different people, {OTHER_PERSON_MIN_WORDS}+ words (all)": above,
    }


def format_checks(results: dict[str, Any]) -> list[str]:
    """Render the main results as plain-text lines."""
    s, d, left = results["structure"], results["duplicates"], results["leftovers"]
    lines = ["== Structure =="]
    lines.append(
        f"{s['rows']:,} reviews, {s['tokens']:,} tokens, {s['products']:,} products, "
        f"{s['users']:,} users, years {s['years'][0]}-{s['years'][1]}."
    )
    if s["matches_report"] is not None:
        lines.append(f"Matches report.json: {s['matches_report']}")
    lines.append(
        f"Invalid tokens: {s['invalid_tokens']}; words per review at least {s['fewest_words']}; "
        f"tokens per review at most {s['most_tokens']}; at most {s['most_reviews_per_product']} "
        f"reviews per product; products with several split keys: "
        f"{s['products_with_several_split_keys']}."
    )
    lines.append(f"Split keys per tenth of [0, 1): {s['split_keys_per_tenth']}")
    exact = d["exact"]
    lines += ["", "== Duplicates =="]
    lines.append(f"Same person, same text: {exact['same_person_same_text']}")
    long_key = f"copies_of_{OTHER_PERSON_MIN_WORDS}_plus_words_by_other_people"
    lines.append(f"Copies of {OTHER_PERSON_MIN_WORDS}+ words by other people: {exact[long_key]}")
    lines.append(
        "Copies by other people kept, by words: "
        + ", ".join(f"{k}: {v:,}" for k, v in exact["copies_by_other_people_kept"].items())
    )
    near = d["near_identical"]
    if not near["search_complete"]:
        lines.append("WARNING: the near-identical search stopped early; its counts are partial.")
    lines.append(
        f"Near-identical by other people ({near['overlap_at_least']:.0%}+ shared words), "
        "by words of the longer review: "
        + ", ".join(f"{k}: {v}" for k, v in near["by_words_of_longer_review"].items())
    )
    lines.append(f"Near-identical by the same person: {near['pairs_by_the_same_person']}")
    lines += ["", "== Leftovers =="]
    accents = left["accent_fragments"]
    lines.append(
        f"Broken accented words (listed ones): {accents['reviews_with_any']:,} reviews "
        f"({accents['share_of_reviews']:.2%})"
    )
    c = left["contractions"]
    lines.append(
        f"Contractions: {c['whole']:,} whole, {c['split']:,} split ({c['share_split']:.2%})"
    )
    runs = left["punctuation_runs"]
    lines.append(
        f"Runs of {LONG_RUN}+ identical marks: {runs['share_of_tokens']:.2%} of tokens, "
        f"{runs['reviews']:,} reviews, longest {runs['longest_run']}"
    )
    lines.append(f"Email addresses: {left['email_addresses']['reviews']} reviews")
    lines.append(f"Repetitive reviews: {left['repetitive_reviews']}")
    foreign = left[f"reviews_with_{FOREIGN_MIN_WORDS}_plus_foreign_function_words"]
    lines.append(
        f"Reviews with {FOREIGN_MIN_WORDS}+ function words of one other language: "
        + ", ".join(f"{k}: {v}" for k, v in foreign.items())
    )
    return lines
