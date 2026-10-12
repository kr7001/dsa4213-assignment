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
- `results/preprocess/report.json`: counts after every step, by star
- `results/preprocess/examples_check.txt`: random samples of what each rule
  removed or kept, for checking by hand (git-ignored)

Every product gets a fixed random split key, and the piles are cut from it
later.

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
  Sampling. Wiley.
