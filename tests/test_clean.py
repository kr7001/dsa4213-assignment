"""Tests for the step 1 cleaning rules and the tokenizer that keeps <num>."""

from a1.data.clean import clean_text, count_words, is_word, tokenize


def _tokens(raw: str) -> list[str]:
    return tokenize(clean_text(raw))


def test_html_tags_and_links_become_spaces() -> None:
    assert _tokens("Great.<br /><br />Would buy again") == [
        "great", ".", "would", "buy", "again",
    ]  # fmt: skip
    assert _tokens("see https://example.com/x?a=1&b=2 for more") == ["see", "for", "more"]
    assert _tokens("or www.example.com today") == ["or", "today"]


def test_video_tags_become_spaces() -> None:
    raw = "[[VIDEOID:2f730fcc9114ef94c63f15083f0a03d5]] Great brush"
    assert _tokens(raw) == ["great", "brush"]


def test_product_links_keep_the_name() -> None:
    raw = "Better than [[ASIN:B000YB5ZSY Mountain Falls Nail Polish Remover]] for me"
    assert _tokens(raw) == [
        "better", "than", "mountain", "falls", "nail", "polish", "remover", "for", "me",
    ]  # fmt: skip
    assert _tokens("see [[ASIN:B000YB5ZSY]]") == ["see"]


def test_escaped_characters_are_decoded() -> None:
    # Without decoding, the course tokenizer would give ["don", ";", "t"].
    assert _tokens("don&#39;t") == ["don't"]
    assert _tokens("a &quot;must have&quot;") == ["a", "must", "have"]


def test_ampersand_becomes_and() -> None:
    assert _tokens("Tom &amp; Jerry") == ["tom", "and", "jerry"]
    assert _tokens("shampoo&conditioner") == ["shampoo", "and", "conditioner"]


def test_numbers_become_one_num_token() -> None:
    assert _tokens("2.5 oz for $12.99") == ["<num>", "oz", "for", "<num>"]
    assert _tokens("1,000 stars") == ["<num>", "stars"]
    assert _tokens("10/10") == ["<num>", "<num>"]
    assert _tokens("SPF30") == ["spf", "<num>"]


def test_hearts_drop_like_emoji() -> None:
    assert _tokens("I <3 this") == ["i", "this"]
    assert _tokens("I &lt;3 this") == ["i", "this"]  # escaped heart
    assert _tokens("love it <333") == ["love", "it"]
    assert _tokens("broke </3") == ["broke"]
    assert _tokens("size 3") == ["size", "<num>"]  # a plain 3 is still a number


def test_emoji_and_curly_apostrophe() -> None:
    assert _tokens("Love it \U0001f60d") == ["love", "it"]
    assert _tokens("it’s nice") == ["it's", "nice"]


def test_word_counting_skips_punctuation_and_numbers() -> None:
    tokens = _tokens("5 stars!! Love it.")
    assert tokens == ["<num>", "stars", "!", "!", "love", "it", "."]
    assert count_words(tokens) == 3
    assert not is_word("<num>") and not is_word("!") and is_word("don't")
    assert count_words(_tokens("10/10 !!!")) == 0
    assert count_words(_tokens("\U0001f44d")) == 0
