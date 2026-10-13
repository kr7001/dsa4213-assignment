"""The input tables of Part II: our own skip-gram vectors and the public GloVe vectors.

Every table is a |V| x d float32 array (d = 100, D2.2) whose row i is the vector
of vocabulary id i, saved as .npy under data/embeddings/ (git-ignored;
scripts/build_embeddings.py rebuilds them). The <pad> row is always zero.

Own vectors (D2.3). Skip-gram with negative sampling, trained with gensim on
the language-model training pile only, with the settings of the Lecture 2 demo
(lecture2_word_embeddings.py, section 10) except vector_size = 100 (D2.2) and
min_count = 1, because the text is already mapped to the vocabulary. The text
is exactly what the language models read: each review as <bos> w1 ... wn <eos>,
with every word outside the vocabulary already <unk>, so every vocabulary token
gets a vector. One worker thread and a fixed seed make the run repeatable
(gensim's documentation). gensim 4.4 draws the starting vectors from NumPy's
generator seeded with that seed, not from Python's string hash, so a test
checks that Python processes with different hash seeds give identical vectors.

Public vectors (D2.4, D2.5). GloVe 2024 Wikipedia + Gigaword, 100 dimensions
(Carlson et al., 2025), whose training text was lowercased and tokenized by
Stanford CoreNLP. Each vocabulary token gets, in this order:

1. "exact": the GloVe vector of the same token (special tokens excepted: the
   GloVe file ends with a "<unk>" vector of its own, an average of rare GloVe
   words, which is not used);
2. "contraction": for a contraction CoreNLP splits ("don't" -> "do" + "n't",
   "it's" -> "it" + "'s"), the average of its pieces (D0.4, locked 12 Oct);
3. "assimilation": for a word CoreNLP's splitAssimilations rule splits
   ("cannot" -> "can" + "not", "gonna" -> "gon" + "na", "dont" -> "do" + "nt"),
   the average of its pieces;
4. "number": for <num>, which replaced every number in our text, the average
   of the number tokens among GloVe's NUMBER_TOP most frequent words;
5. "pad": zeros for <pad>, which never affects a prediction;
6. "random": anything else, including <unk>, <bos> and <eos>: every entry
   drawn uniformly from [-a, a], with a chosen so the mean square of the
   entries equals that of the exact GloVe rows ("the same variance as the
   pre-trained ones", Kim, 2014).

The random rows of the GloVe table and the fixed random table use separate
random streams of the same seed, so no row of one repeats a row of the other.

Scale (D2.6). With a fixed table, its overall size works like a learning-rate
setting for the first layer, so the tables can be put on one scale: the root
mean square (RMS) of the entries. Multiplying a table by one number changes no
cosine similarity, nearest neighbour or analogy answer.
"""

from __future__ import annotations

import gzip
import io
import math
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from a1.data.clean import _NUMBER as NUMBER_PATTERN
from a1.data.clean import NUM_TOKEN
from a1.data.sequences import EncodedReviews
from a1.data.vocab import BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, Vocabulary
from a1.data.vocabulary_checks import split_like_glove

# The Lecture 2 demo's Word2Vec call, with vector_size = 100 (D2.2) and
# min_count = 1 (the text is already mapped to the vocabulary).
SKIPGRAM_SETTINGS: dict[str, Any] = {
    "vector_size": 100,
    "window": 5,
    "min_count": 1,
    "sg": 1,
    "negative": 5,
    "ns_exponent": 0.75,
    "sample": 0.0,
    "shrink_windows": False,
    "epochs": 15,
    "workers": 1,
    "seed": 4213,
}

# CoreNLP 4.4 PTBLexer.flex (splitAssimilations, on by default): ASSIMILATIONS3
# keeps the last three letters apart, ASSIMILATIONS2 the last two. "nno" is what
# is left of "dunno" after the first split. Forms that never occur in our
# lowercase word tokens ('tis, 'twas, 'd've) are left out.
ASSIMILATIONS3 = ("cannot", "dunno")
ASSIMILATIONS2 = (
    "gonna",
    "gotta",
    "lemme",
    "gimme",
    "wanna",
    "nno",
    "aint",
    "dont",
    "doesnt",
    "didnt",
    "theyre",
)
NUMBER_TOP = 10_000  # <num> averages the number tokens among GloVe's most frequent words
SOURCES = ("exact", "contraction", "assimilation", "number", "pad", "random")
# Words whose nearest neighbours are listed, fixed before any table was built:
# a rating word, a product property, a price word, a failure, a product type, a purpose.
QUERY_WORDS = ("great", "smell", "cheap", "broke", "moisturizer", "gift")
NEIGHBOURS = 5
# Separate random streams of one seed: the GloVe table's random rows, the random table.
GLOVE_RANDOM_STREAM, RANDOM_TABLE_STREAM = 1, 2


def random_stream(seed: int, stream: int) -> np.random.Generator:
    """One of several independent NumPy generators derived from the same seed."""
    return np.random.default_rng([seed, stream])


def split_assimilation(token: str) -> list[str]:
    """Split a word the way CoreNLP's splitAssimilations rule does: "gonna" -> "gon" + "na"."""
    if token in ASSIMILATIONS3:
        return [token[:-3], *split_assimilation(token[-3:])]
    if token in ASSIMILATIONS2:
        return [token[:-2], token[-2:]]
    return [token]


def input_counts(reviews: EncodedReviews, vocab_size: int) -> np.ndarray:
    """How often each token is read as an input: every token, plus one <bos> per review."""
    counts = np.bincount(reviews.ids, minlength=vocab_size).astype(np.int64)
    counts[BOS_ID] += len(reviews)
    return counts


def review_sentences(reviews: EncodedReviews, vocab: Vocabulary) -> list[list[str]]:
    """Each review as <bos> w1 ... wn <eos>, with words outside the vocabulary as <unk>."""
    tokens = vocab.tokens
    bos, eos = tokens[BOS_ID], tokens[EOS_ID]
    return [[bos, *(tokens[i] for i in reviews.review(r)), eos] for r in range(len(reviews))]


def check_skipgram_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """The full settings, with the defaults above filled in; one worker only."""
    unknown = set(settings) - set(SKIPGRAM_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown skip-gram settings: {sorted(unknown)}")
    merged = {**SKIPGRAM_SETTINGS, **settings}
    if merged["workers"] != 1:
        raise ValueError("workers must be 1: more threads make the result change between runs")
    return merged


def train_skipgram(
    sentences: list[list[str]], vocab: Vocabulary, settings: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train skip-gram on the sentences; return the |V| x d table and the training record.

    The record holds the wall-clock and CPU time of gensim's vocabulary pass
    and training (the cost of our own embeddings, reported apart from language
    model training), the corpus size and the tokens that got no vector.
    """
    import gensim
    from gensim.models import Word2Vec

    settings = check_skipgram_settings(settings)
    wall, cpu = time.perf_counter(), time.process_time()
    model = Word2Vec(sentences=sentences, **settings)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    table = np.zeros((len(vocab), settings["vector_size"]), dtype=np.float32)
    without = []
    for i, token in enumerate(vocab.tokens):
        if token in model.wv.key_to_index:
            table[i] = model.wv[token]
        elif i != PAD_ID:
            without.append(token)
    table[PAD_ID] = 0.0
    record = {
        "settings": settings,
        "gensim": gensim.__version__,
        "seconds": wall,
        "cpu_seconds": cpu,
        "threads": 1,
        "reviews": model.corpus_count,
        "tokens_per_epoch": model.corpus_total_words,
        "tokens_processed": model.corpus_total_words * settings["epochs"],
        "words_with_vectors": len(model.wv),
        "vocabulary_tokens_without_vector": without,
    }
    return table, record


def _glove_lines(path: Path) -> Iterator[str]:
    """Lines of a GloVe file: Stanford .zip (its largest .txt), gzip or plain text."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist() if m.filename.endswith(".txt")]
            if not members:
                raise ValueError(f"{path} holds no .txt file")
            member = max(members, key=lambda m: m.file_size)
            with archive.open(member) as raw:
                yield from io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def read_glove(
    path: Path, wanted: set[str], dim: int, number_top: int = NUMBER_TOP
) -> tuple[dict[str, np.ndarray], list[str], dict[str, Any]]:
    """Read the vectors of the wanted words, and of GloVe's frequent number tokens.

    GloVe files list words from most to least frequent, so the number tokens
    among the first ``number_top`` lines are the most frequent numbers. A
    first line of two integers (a gensim header) is skipped; a repeated word
    keeps its first vector.

    Returns:
        The vectors found, the number tokens found (most frequent first) and
        facts about the file: its word count and first ten words.
    """
    vectors: dict[str, np.ndarray] = {}
    numbers: list[str] = []
    first_words: list[str] = []
    rank = 0
    for line_number, line in enumerate(_glove_lines(path)):
        word, _, rest = line.rstrip("\n").partition(" ")
        if line_number == 0 and word.isdigit() and rest.strip().isdigit():
            continue
        if len(first_words) < 10:
            first_words.append(word)
        is_number = rank < number_top and NUMBER_PATTERN.fullmatch(word) is not None
        if (word in wanted or is_number) and word not in vectors:
            vector = np.asarray(rest.split(), dtype=np.float32)
            if vector.shape != (dim,):
                raise ValueError(f"{path}: {word!r} has {vector.size} numbers, expected {dim}")
            vectors[word] = vector
            if is_number:
                numbers.append(word)
        rank += 1
    return vectors, numbers, {"words": rank, "first_words": first_words}


def glove_wanted(vocab: Vocabulary) -> set[str]:
    """Every GloVe word the table could use: vocabulary tokens and their pieces."""
    wanted = set(vocab.tokens)
    for token in vocab.tokens:
        wanted.update(split_like_glove(token))
        wanted.update(split_assimilation(token))
    return wanted


def glove_table(
    vocab: Vocabulary,
    vectors: dict[str, np.ndarray],
    numbers: list[str],
    dim: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """The GloVe table in vocabulary order, and where each row came from (SOURCES)."""
    table = np.zeros((len(vocab), dim), dtype=np.float32)
    sources: list[str] = []
    for i, token in enumerate(vocab.tokens):
        contraction = split_like_glove(token)
        assimilation = split_assimilation(token)
        if i == PAD_ID:
            source = "pad"
        elif token in SPECIAL_TOKENS:
            source = "random"
        elif token in vectors:
            table[i], source = vectors[token], "exact"
        elif len(contraction) > 1 and all(p in vectors for p in contraction):
            table[i] = np.mean([vectors[p] for p in contraction], axis=0)
            source = "contraction"
        elif len(assimilation) > 1 and all(p in vectors for p in assimilation):
            table[i] = np.mean([vectors[p] for p in assimilation], axis=0)
            source = "assimilation"
        elif token == NUM_TOKEN and numbers:
            table[i], source = np.mean([vectors[n] for n in numbers], axis=0), "number"
        else:
            source = "random"
        sources.append(source)
    exact = np.array([s == "exact" for s in sources])
    if not exact.any():
        raise ValueError("No vocabulary token has a GloVe vector")
    bound = math.sqrt(3.0 * float(np.mean(np.square(table[exact], dtype=np.float64))))
    missing = np.flatnonzero(np.array([s == "random" for s in sources]))
    rng = random_stream(seed, GLOVE_RANDOM_STREAM)
    table[missing] = rng.uniform(-bound, bound, size=(missing.size, dim)).astype(np.float32)
    return table, sources


def rms(table: np.ndarray) -> float:
    """Root mean square of the entries, <pad> row left out: the table's scale."""
    rows = np.ones(table.shape[0], dtype=bool)
    rows[PAD_ID] = False
    return float(np.sqrt(np.mean(np.square(table[rows], dtype=np.float64))))


def scaled_to(table: np.ndarray, target_rms: float) -> tuple[np.ndarray, float]:
    """The table times one number, so its RMS is target_rms; also returns the number."""
    factor = target_rms / rms(table)
    return (table * factor).astype(np.float32), factor


def random_table(vocab_size: int, dim: int, target_rms: float, seed: int) -> np.ndarray:
    """A table with no information: every row but <pad> uniform in [-a, a], RMS target_rms."""
    bound = math.sqrt(3.0) * target_rms
    rng = random_stream(seed, RANDOM_TABLE_STREAM)
    table = rng.uniform(-bound, bound, size=(vocab_size, dim))
    table[PAD_ID] = 0.0
    return table.astype(np.float32)


def table_stats(table: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """The norm check: scale and row lengths, also weighted by how often a row is read."""
    rows = np.ones(table.shape[0], dtype=bool)
    rows[PAD_ID] = False
    norms = np.linalg.norm(table[rows].astype(np.float64), axis=1)
    weights = counts[rows].astype(np.float64)
    return {
        "rms": rms(table),
        "row_norm_mean": float(norms.mean()),
        "row_norm_median": float(np.median(norms)),
        "row_norm_p10": float(np.percentile(norms, 10)),
        "row_norm_p90": float(np.percentile(norms, 90)),
        "row_norm_input_weighted": float(np.sqrt(np.sum(weights * norms**2) / weights.sum())),
    }


def coverage(sources: list[str], vocab: Vocabulary, counts: np.ndarray) -> dict[str, Any]:
    """Rows per source, and the share of training inputs each source covers.

    The headline figures count vocabulary words (every token but the four
    special ones; <num> counts as a word) that get a real GloVe vector: exact,
    or built from the pieces CoreNLP splits them into.
    """
    total = float(counts.sum())
    by_source = {}
    for source in SOURCES:
        ids = [i for i, s in enumerate(sources) if s == source]
        by_source[source] = {
            "types": len(ids),
            "type_share": len(ids) / (len(vocab) - 1),  # <pad> is not a word
            "input_share": float(counts[ids].sum()) / total if ids else 0.0,
        }
    random_ids = sorted(
        (i for i, s in enumerate(sources) if s == "random"), key=lambda i: -counts[i]
    )
    words = [i for i, token in enumerate(vocab.tokens) if token not in SPECIAL_TOKENS]
    real = [i for i in words if sources[i] in ("exact", "contraction", "assimilation")]
    return {
        "word_types": len(words),
        "word_types_with_glove_vector": len(real),
        "word_type_share_with_glove_vector": len(real) / len(words) if words else 0.0,
        "input_share_with_glove_vector": float(counts[real].sum()) / total if real else 0.0,
        "by_source": by_source,
        "most_frequent_random": {vocab.tokens[i]: int(counts[i]) for i in random_ids[:25]},
        "assimilations": {
            vocab.tokens[i]: split_assimilation(vocab.tokens[i])
            for i, s in enumerate(sources)
            if s == "assimilation"
        },
    }


def informative_rows(sources: list[str], vocab: Vocabulary) -> np.ndarray:
    """Word rows with a real GloVe vector (exact or built from pieces), special tokens left out."""
    keep = {"exact", "contraction", "assimilation"}
    return np.array(
        [
            s in keep and token not in SPECIAL_TOKENS
            for s, token in zip(sources, vocab.tokens, strict=True)
        ]
    )


def intrinsic_scores(
    tables: dict[str, np.ndarray], vocab: Vocabulary, rows: np.ndarray
) -> dict[str, Any]:
    """Lecture 2's checks on every table, over the same words: WordSim353 and analogies.

    Only words that have a real vector in every table are used, so every table
    answers the same questions; coverage is reported next to each score. As in
    the Lecture 2 demo, WordSim353 words are lowercased, a pair counts when both
    words have a vector, and the score is Spearman's rho between the human
    scores and the cosines. Analogies use gensim's evaluate_word_analogies
    (3CosAdd among the same words). A score needs at least 3 pairs or 1 question.
    """
    from gensim.models import KeyedVectors
    from gensim.test.utils import datapath
    from scipy.stats import spearmanr

    words = [vocab.tokens[i] for i in np.flatnonzero(rows)]
    position = {word: k for k, word in enumerate(words)}
    pairs = []
    for line in Path(datapath("wordsim353.tsv")).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            first, second, human = line.split("\t")
            pairs.append((first.lower(), second.lower(), float(human)))
    covered = [
        (position[a], position[b], h) for a, b, h in pairs if a in position and b in position
    ]
    questions = [
        line
        for line in Path(datapath("questions-words.txt")).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(":")
    ]
    scores: dict[str, Any] = {"words": len(words)}
    for name, table in tables.items():
        unit = table[rows].astype(np.float64)
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        cosines = [float(unit[a] @ unit[b]) for a, b, _ in covered]
        rho = (
            float(spearmanr([h for _, _, h in covered], cosines).statistic)
            if len(covered) >= 3
            else None
        )
        vectors = KeyedVectors(vector_size=table.shape[1])
        vectors.add_vectors(words, table[rows])
        accuracy, sections = vectors.evaluate_word_analogies(datapath("questions-words.txt"))
        answered = len(sections[-1]["correct"]) + len(sections[-1]["incorrect"])
        scores[name] = {
            "wordsim353_spearman": rho,
            "wordsim353_pairs": len(covered),
            "wordsim353_pairs_total": len(pairs),
            "analogy_accuracy": float(accuracy) if answered else None,
            "analogy_questions": answered,
            "analogy_questions_total": len(questions),
        }
    return scores


def nearest_neighbours(
    tables: dict[str, np.ndarray],
    vocab: Vocabulary,
    rows: np.ndarray,
    queries: tuple[str, ...] = QUERY_WORDS,
    k: int = NEIGHBOURS,
) -> dict[str, dict[str, list[str]]]:
    """The k most similar words (cosine) to each query word, among the same words in every table."""
    ids = np.flatnonzero(rows)
    result: dict[str, dict[str, list[str]]] = {}
    for name, table in tables.items():
        unit = table[ids].astype(np.float64)
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        result[name] = {}
        for query in queries:
            if query not in vocab or not rows[vocab.index[query]]:
                continue
            query_id = vocab.index[query]
            similarity = unit @ unit[np.searchsorted(ids, query_id)]
            order = [j for j in np.argsort(-similarity, kind="stable") if ids[j] != query_id]
            result[name][query] = [vocab.tokens[ids[j]] for j in order[:k]]
    return result


def number_tokens_used(numbers: list[str]) -> dict[str, Any]:
    """What went into the <num> vector."""
    return {"count": len(numbers), "most_frequent": numbers[:20]}


def ensure_finite(table: np.ndarray, name: str) -> None:
    if not np.isfinite(table).all():
        raise ValueError(f"The {name} table has values that are not finite")
