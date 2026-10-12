# DSA4213 Assignment 1

Language models, word embeddings and learned representations, compared on
Amazon beauty-product reviews (NUS DSA4213, Semester 1 AY2026/27).

## Setup

1. Install [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the
   exact dependency versions pinned in `uv.lock`:

       curl -LsSf https://astral.sh/uv/install.sh | sh

   (or `brew install uv`). Open a new terminal afterwards.
2. From this folder:

       uv sync

## Data

The `All_Beauty` reviews from Amazon Reviews 2023 (Hou et al., 2024). The
dataset states no licence, so this repository holds the download script and
never the data itself.

    uv run python scripts/download_data.py

This saves `data/raw/All_Beauty.jsonl.gz`, checks that it has the expected
format, and prints its SHA-256 checksum.

## Step 1: describe the data before deciding anything

    uv run python scripts/explore_data.py

This prints the statistics behind each preprocessing decision (corpus size, star
ratings, reviews per product, review lengths, repeated texts and raw-text noise)
and writes them to `results/eda/`:

- `summary.json`: every number
- `ratings.png`, `lengths.png`, `products.png`: charts
- `examples.txt`: review excerpts behind each check, for reading by eye
  (git-ignored, because it quotes the data)

## Step 1b: short reviews and the minimum length

    uv run python scripts/explore_short_reviews.py

This cleans and tokenizes every review exactly as the models will see it
(`src/a1/data/clean.py`). For each number of words, it measures:

- how many reviews and tokens there are;
- how positive the reviews are;
- how often the exact text repeats;
- how well a vote among identical texts predicts positive versus negative.

It also measures what each minimum length would remove, and how well length
alone predicts sentiment, which is a baseline for Part III. It writes
`results/eda/short_reviews.json` and `results/eda/examples_short_reviews.txt`
(git-ignored).

## Step 1c: English only, reviews per product, duplicates

    uv run python scripts/explore_filters.py

After the step 3 length rules, this measures three things:

- **Language:** what fastText's language identifier calls each review, by
  length and confidence. Short reviews are also judged by word evidence learnt
  from confidently identified long reviews.
- **Reviews per product:** how concentrated reviews are and what each cap
  costs. It also measures how alike the ratings of one product are, and how
  much a random product-level split varies with and without a cap.
- **Duplicates:** exact copies, split by whether the same user wrote them and
  whether they sit under the same product. It also finds near copies by the
  same user.

It writes `results/eda/filters.json` and `results/eda/examples_filters.txt`
(git-ignored).

## Step 1d: the English-only rule

    uv run python scripts/explore_language.py

This applies the rule in `src/a1/data/language_filter.py` and counts what it
removes, by length group and by rule. It also writes random samples of what it
removes and keeps, for checking by eye. Output goes to
`results/eda/language_rule.json` and `results/eda/examples_language.txt`
(git-ignored).

## Step 2: build the clean corpus

    uv run python scripts/prepare_data.py

This applies every locked data decision in order: cleaning, lengths, English
only, duplicates and the cap per product (see `src/a1/data/prepare.py`). It
takes a few minutes and writes:

- `data/processed/reviews.jsonl.gz`: the clean corpus, one review per line
  (never committed)
- `results/preprocess/report.json`: counts after every step, by star; the cap
  recomputed on 50 random halves of the products (it should barely move); and
  the corpus fingerprint
- `results/preprocess/examples_check.txt`: random samples of what each rule
  removed or kept, for checking by hand (git-ignored)

Every product gets a fixed random split key, and the piles are cut from it
later.

The fingerprint is the SHA-256 of the uncompressed corpus. The pipeline is
deterministic (seed 4213), so a re-run on the same raw file gives the same
value. To check a copy of the corpus:

    gunzip -c data/processed/reviews.jsonl.gz | shasum -a 256

It must print the `sha256_uncompressed` value in `report.json`.

## Step 3: check the clean corpus

    uv run python scripts/check_corpus.py

This runs every check in `src/a1/data/corpus_checks.py` on the file the models
will read (about two minutes):

- **Structure:** the file matches `report.json` (rows, tokens, fingerprint),
  every token fits the tokenizer, every review is within the length limits,
  no product is over the cap, and each product has one split key.
- **Duplicates:** no person repeats a text, and no text of 14+ words appears
  twice. It also counts, by length, how often different people wrote
  near-identical reviews (90%+ of their distinct words shared). This is the
  evidence for the 14-word copy rule: such pairs appear at every length up to
  13 words, and the longer ones read as copies. The search is exact (prefix
  filtering), not sampled.
- **Leftovers:** the stated limitations, measured: accented words the a-z
  tokenizer breaks, split contractions, runs of punctuation, markup traces,
  repetitive reviews and leftover non-English function words. It also
  confirms that no email address is left (cleaning removes them).

It writes `results/preprocess/corpus_checks.json` and
`results/preprocess/examples_corpus_checks.txt` (git-ignored, it quotes the
data). `report.json` and `corpus_checks.json` hold counts only, so they are
committed.

## Step 4: measure the vocabulary choices

    uv run python scripts/explore_vocabulary.py --glove path/to/glove-file

This splits the products into two halves by their split key, builds a
vocabulary from one half and measures how much text of the other half would
become `<unk>`, for each vocabulary size and each minimum count (both ways
round). It also counts contractions such as "don't", and with `--glove` it
measures how much of the vocabulary public GloVe vectors cover, with
contractions kept whole or split the way GloVe's tokenizer splits them
("do" + "n't"). A GloVe file can be a Stanford `.txt` or `.zip` (such as
`glove.6B.zip` or `glove.2024.wikigiga.50d.zip` from
https://nlp.stanford.edu/projects/glove/) or a gensim-data `.gz`; only its
words are read. Output: `results/eda/vocabulary.json`.

## Checks

    uv run pytest
    uv run ruff check . && uv run ruff format --check .

## Sources

- Hou, Y., Li, J., He, Z., Yan, A., Chen, X. and McAuley, J. (2024). Bridging
  Language and Items for Retrieval and Recommendation. arXiv:2403.03952.
  Data: https://amazon-reviews-2023.github.io/
- The tokenizer is adapted from DSA4213 Lecture1_Demo.ipynb (D. Zhou, 2026).
- Language identification: fastText lid.176 (Joulin, A., Grave, E., Bojanowski,
  P. and Mikolov, T. (2017). Bag of Tricks for Efficient Text Classification.
  EACL; Joulin et al. (2016). FastText.zip: Compressing text classification
  models. arXiv:1612.03651), using the compressed model bundled with the
  fast-langdetect package (https://github.com/LlmKira/fast-langdetect).
- The cap per product uses Kish's design effect: Kish, L. (1965). Survey
  Sampling. Wiley. Capping reviews per item follows Maas, A. L., Daly, R. E.,
  Pham, P. T., Huang, D., Ng, A. Y. and Potts, C. (2011). Learning Word Vectors
  for Sentiment Analysis. ACL 2011, which keeps at most 30 reviews per movie.
- Removing duplicates before splitting follows Lee, K., Ippolito, D., Nystrom,
  A., Zhang, C., Eck, D., Callison-Burch, C. and Carlini, N. (2022).
  Deduplicating Training Data Makes Language Models Better. ACL 2022,
  pp. 8424-8445.
- GloVe vectors: Pennington, J., Socher, R. and Manning, C. D. (2014). GloVe:
  Global Vectors for Word Representation. EMNLP 2014; and Carlson, R., Bauer,
  J. and Manning, C. D. (2025). A New Pair of GloVes. arXiv:2507.18103.
- The near-identical search uses prefix filtering: Chaudhuri, S., Ganti, V.
  and Kaushik, R. (2006). A Primitive Operator for Similarity Joins in Data
  Cleaning. ICDE 2006; Bayardo, R. J., Ma, Y. and Srikant, R. (2007). Scaling
  Up All Pairs Similarity Search. WWW 2007, pp. 131-140.
