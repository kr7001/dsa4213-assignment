"""Evidence for what counts as a duplicate (step 5).

1. Exact copies: reviews whose cleaned text is identical. Every copy after the
   first (in time order) is classed by whether an earlier copy came from the
   same user, and whether one sits under the same product.
2. Leakage risk: a copied text under several products can land in both the
   training and the test split of a product-level split.
3. Near copies by the same user: two reviews of 7+ words by one person whose
   sets of words overlap by at least NEAR_COPY_JACCARD (shared words divided
   by all words used), without being identical.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from a1.data.review_table import LENGTH_GROUPS, SHORT_MAX_WORDS, ReviewTable, group_name

NEAR_COPY_JACCARD = 0.8
LONG_MIN_WORDS = SHORT_MAX_WORDS + 1
SPLIT_SHARES = (0.8, 0.1, 0.1)
MAX_REVIEWS_PER_USER_COMPARED = 400  # bounds the pairwise work for the busiest users
TOP_GROUPS = 10
NEAR_COPY_SAMPLES = 10
CATEGORIES = (
    "same user, same product",
    "same user, other product",
    "other user, same product",
    "other user, other product",
)


@dataclass
class DuplicateSamples:
    """Which reviews to quote, chosen before their text is fetched."""

    copied_by_several_users: list[tuple[int, str]] = field(default_factory=list)
    copied_by_one_user: list[tuple[int, str]] = field(default_factory=list)
    near_copies: list[tuple[int, int, str]] = field(default_factory=list)

    def indices(self) -> set[int]:
        found = {i for i, _ in self.copied_by_several_users + self.copied_by_one_user}
        for a, b, _ in self.near_copies:
            found.update((a, b))
        return found

    def render(self, excerpts: dict[int, str]) -> dict[str, list[str]]:
        return {
            "Most copied texts of 7+ words, several users": [
                f"{label} | {excerpts.get(i, '')}" for i, label in self.copied_by_several_users
            ],
            "Most copied texts of 7+ words, one user": [
                f"{label} | {excerpts.get(i, '')}" for i, label in self.copied_by_one_user
            ],
            "Random near copies by the same user (pairs)": [
                f"{label}\n    A: {excerpts.get(a, '')}\n    B: {excerpts.get(b, '')}"
                for a, b, label in self.near_copies
            ],
        }


def summarize_duplicates(
    table: ReviewTable, kept: np.ndarray, seed: int
) -> tuple[dict[str, Any], DuplicateSamples]:
    """Measure exact and near copies among the reviews kept so far.

    Args:
        table: The review table.
        kept: Boolean mask of reviews kept so far.
        seed: Seed for sampling the busiest users and the quoted pairs.

    Returns:
        Numbers for the summary file, and the reviews to quote.
    """
    rng = random.Random(seed)
    samples = DuplicateSamples()
    exact = _exact_copies(table, kept, samples)
    near = _near_copies(table, kept, rng, samples)
    return {"exact_copies": exact, "near_copies_same_user": near}, samples


def _sentiment(rating: int) -> str:
    return "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"


def _exact_copies(table: ReviewTable, kept: np.ndarray, samples: DuplicateSamples) -> dict:
    index = np.flatnonzero(kept)
    order = index[np.lexsort((index, table.timestamp[index], table.text_id[index]))]
    text = table.text_id[order]
    starts = np.flatnonzero(np.concatenate(([True], text[1:] != text[:-1])))
    ends = np.append(starts[1:], order.size)

    rows = {group_name(*g): _empty_row() for g in LENGTH_GROUPS}
    long_groups: list[tuple[int, int, int, int, Counter[int]]] = []
    split_texts = split_reviews = 0.0
    for start, end in zip(starts, ends, strict=True):
        if end - start < 2:
            continue
        members = order[start:end]
        words = int(table.words[members[0]])
        row = rows[_group_of(words)]
        users_seen: set[int] = set()
        products_seen: set[int] = set()
        for position, member in enumerate(members):
            user, product = int(table.user[member]), int(table.product[member])
            if position:
                same_user, same_product = user in users_seen, product in products_seen
                row["extra_copies"][CATEGORIES[2 * (not same_user) + (not same_product)]] += 1
            users_seen.add(user)
            products_seen.add(product)
        sentiments = {_sentiment(int(r)) for r in table.rating[members]}
        row["texts"] += 1
        row["reviews"] += int(members.size)
        row["texts_under_2plus_products"] += len(products_seen) > 1
        row["texts_by_2plus_users"] += len(users_seen) > 1
        row["texts_with_positive_and_negative"] += {"positive", "negative"} <= sentiments
        if words >= LONG_MIN_WORDS:
            k = len(products_seen)
            if k > 1:
                p_split = 1 - sum(share**k for share in SPLIT_SHARES)
                split_texts += p_split
                split_reviews += p_split * members.size
            long_groups.append(
                (
                    int(members.size),
                    len(users_seen),
                    k,
                    int(members[0]),
                    Counter(int(r) for r in table.rating[members]),
                )
            )

    def label(size: int, users: int, products: int, ratings: Counter[int]) -> str:
        stars = ", ".join(f"{star}*: {n}" for star, n in sorted(ratings.items()))
        return f"{size} copies, {users} users, {products} products, {stars}"

    long_groups.sort(key=lambda g: (-g[0], g[3]))
    samples.copied_by_several_users = [
        (first, label(size, users, products, ratings))
        for size, users, products, first, ratings in long_groups
        if users > 1
    ][:TOP_GROUPS]
    samples.copied_by_one_user = [
        (first, label(size, users, products, ratings))
        for size, users, products, first, ratings in long_groups
        if users == 1
    ][:TOP_GROUPS]

    total_extra = {
        name: sum(row["extra_copies"][name] for row in rows.values()) for name in CATEGORIES
    }
    return {
        "rule": "identical token sequence after the step 1 cleaning",
        "reviews": int(kept.sum()),
        "by_length_group": [{"group": name} | row for name, row in rows.items()],
        "extra_copies_total": total_extra,
        "split_risk_7plus_words": {
            "split_shares": SPLIT_SHARES,
            "expected_texts_split_across_splits": split_texts,
            "expected_reviews_in_those_texts": split_reviews,
        },
    }


def _empty_row() -> dict[str, Any]:
    return {
        "texts": 0,
        "reviews": 0,
        "extra_copies": dict.fromkeys(CATEGORIES, 0),
        "texts_under_2plus_products": 0,
        "texts_by_2plus_users": 0,
        "texts_with_positive_and_negative": 0,
    }


def _group_of(words: int) -> str:
    for low, high in LENGTH_GROUPS:
        if words >= low and (high is None or words <= high):
            return group_name(low, high)
    raise ValueError(f"No length group for {words} words")


def _near_copies(
    table: ReviewTable, kept: np.ndarray, rng: random.Random, samples: DuplicateSamples
) -> dict[str, Any]:
    long = np.flatnonzero(kept & (table.words >= LONG_MIN_WORDS))
    order = long[np.argsort(table.user[long], kind="stable")]
    user = table.user[order]
    starts = np.flatnonzero(np.concatenate(([True], user[1:] != user[:-1])))
    ends = np.append(starts[1:], order.size)

    pairs: list[tuple[int, int, float]] = []
    users_with_2plus = compared = users_sampled = 0
    for start, end in zip(starts, ends, strict=True):
        if end - start < 2:
            continue
        users_with_2plus += 1
        members = [int(m) for m in order[start:end]]
        if len(members) > MAX_REVIEWS_PER_USER_COMPARED:
            members = sorted(rng.sample(members, MAX_REVIEWS_PER_USER_COMPARED))
            users_sampled += 1
        sets = [frozenset(table.word_set(m).tolist()) for m in members]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                compared += 1
                if table.text_id[members[a]] == table.text_id[members[b]]:
                    continue  # an exact copy, counted above
                size_a, size_b = len(sets[a]), len(sets[b])
                if min(size_a, size_b) < NEAR_COPY_JACCARD * max(size_a, size_b):
                    continue  # too different in size to reach the threshold
                shared = len(sets[a] & sets[b])
                jaccard = shared / (size_a + size_b - shared)
                if jaccard >= NEAR_COPY_JACCARD:
                    pairs.append((members[a], members[b], jaccard))

    involved = {i for a, b, _ in pairs for i in (a, b)}
    same_product = sum(table.product[a] == table.product[b] for a, b, _ in pairs)
    removable = len(involved) - _count_clusters(pairs)
    for a, b, jaccard in rng.sample(pairs, min(NEAR_COPY_SAMPLES, len(pairs))):
        place = "same product" if table.product[a] == table.product[b] else "other products"
        stars = f"{table.rating[a]}* and {table.rating[b]}*"
        samples.near_copies.append((a, b, f"overlap {jaccard:.2f}, {place}, {stars}"))
    return {
        "rule": (
            f"same user, 7+ words each, word overlap (Jaccard) {NEAR_COPY_JACCARD}+, not identical"
        ),
        "users_with_2plus_long_reviews": users_with_2plus,
        "users_sampled_down": users_sampled,
        "pairs_compared": compared,
        "near_copy_pairs": len(pairs),
        "pairs_with_overlap_0.9plus": sum(j >= 0.9 for _, _, j in pairs),
        "pairs_both_21plus_words": int(
            sum(min(table.words[a], table.words[b]) >= 21 for a, b, _ in pairs)
        ),
        "pairs_same_product": int(same_product),
        "pairs_other_products": len(pairs) - int(same_product),
        "reviews_involved": len(involved),
        "reviews_removed_if_one_kept_per_cluster": removable,
    }


def _count_clusters(pairs: list[tuple[int, int, float]]) -> int:
    """Number of groups formed by linking each pair (union-find)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        parent[find(a)] = find(b)
    return len({find(x) for x in parent})


def format_duplicates(summary: dict[str, Any]) -> list[str]:
    """Render the duplicate numbers as report lines."""
    exact, near = summary["exact_copies"], summary["near_copies_same_user"]
    lines = [
        "== Duplicates ==",
        f"Exact copies ({exact['rule']}), by length group. Extra copies = copies after the "
        "first, by who wrote them and where:",
        "group | texts | reviews | extra copies: same user+product / same user, other "
        "product / other user, same product / other user+product | under 2+ products | "
        "both positive and negative",
    ]
    for row in exact["by_length_group"]:
        extra = row["extra_copies"]
        lines.append(
            f"{row['group']:>11} | {row['texts']:,} | {row['reviews']:,} | "
            + " / ".join(f"{extra[name]:,}" for name in CATEGORIES)
            + f" | {row['texts_under_2plus_products']:,} | "
            f"{row['texts_with_positive_and_negative']:,}"
        )
    risk = exact["split_risk_7plus_words"]
    lines += [
        f"Texts of 7+ words under 2+ products: in a random 80/10/10 product split, about "
        f"{risk['expected_texts_split_across_splits']:,.0f} texts "
        f"({risk['expected_reviews_in_those_texts']:,.0f} reviews) would sit in two splits.",
        f"Near copies by the same user ({near['rule']}):",
        f"  {near['near_copy_pairs']:,} pairs ({near['pairs_with_overlap_0.9plus']:,} at 0.9+, "
        f"{near['pairs_both_21plus_words']:,} with both reviews 21+ words), "
        f"{near['pairs_same_product']:,} under the same product and "
        f"{near['pairs_other_products']:,} under other products; "
        f"{near['reviews_involved']:,} reviews involved, "
        f"{near['reviews_removed_if_one_kept_per_cluster']:,} removed if one is kept per group",
        f"  ({near['users_with_2plus_long_reviews']:,} users with 2+ long reviews, "
        f"{near['pairs_compared']:,} pairs compared, {near['users_sampled_down']} busiest users "
        f"sampled down to {MAX_REVIEWS_PER_USER_COMPARED} reviews)",
    ]
    return lines
