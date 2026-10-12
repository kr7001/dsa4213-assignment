"""The vocabulary every language model shares.

Decisions of 12 Oct (D0.4):

- It is built from the training pile only, so no validation or test text can
  shape it.
- It keeps every token seen at least 5 times there. Every other token becomes
  <unk>, and the share of <unk> targets is reported next to each perplexity.
- It starts with the four special tokens of Lecture4_Demo, in this order:
  <pad> (filler that is never scored), <unk>, <bos> (start of a review) and
  <eos> (end of a review). "<num>" is an ordinary token.
- Ties in frequency are broken alphabetically, so the ids are the same on
  every machine.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = (PAD, UNK, BOS, EOS)
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(len(SPECIAL_TOKENS))
MIN_COUNT = 5


class Vocabulary:
    """A fixed list of tokens; a token's id is its position in the list."""

    def __init__(self, tokens: Sequence[str], min_count: int | None = None) -> None:
        if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"The vocabulary must start with {SPECIAL_TOKENS}")
        if len(set(tokens)) != len(tokens):
            raise ValueError("The vocabulary has a repeated token")
        self.tokens = tuple(tokens)
        self.min_count = min_count
        self.index = {token: i for i, token in enumerate(self.tokens)}

    @classmethod
    def build(cls, counts: Counter, min_count: int = MIN_COUNT) -> Vocabulary:
        """Special tokens, then tokens seen min_count+ times, most frequent first."""
        kept = [t for t, n in counts.items() if n >= min_count and t not in SPECIAL_TOKENS]
        kept.sort(key=lambda token: (-counts[token], token))
        return cls([*SPECIAL_TOKENS, *kept], min_count=min_count)

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self.index

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Token ids, with <unk> for tokens outside the vocabulary."""
        index = self.index
        return [index.get(token, UNK_ID) for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.tokens[i] for i in ids]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"min_count": self.min_count, "size": len(self), "tokens": list(self.tokens)}
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["tokens"], min_count=payload.get("min_count"))
