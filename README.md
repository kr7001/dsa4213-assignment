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

## Checks

    uv run pytest
    uv run ruff check . && uv run ruff format --check .

## Sources

- Hou, Y., Li, J., He, Z., Yan, A., Chen, X. and McAuley, J. (2024). Bridging
  Language and Items for Retrieval and Recommendation. arXiv:2403.03952.
  Data: https://amazon-reviews-2023.github.io/
- The tokenizer is adapted from DSA4213 Lecture1_Demo.ipynb (D. Zhou, 2026).
