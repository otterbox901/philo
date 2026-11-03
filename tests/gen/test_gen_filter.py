"""Tests for the input filters and the train/val split (no API calls)."""

import pandas as pd
import pytest

from gen.filter import apply_filters, content_words, overlap_ratio, split_by_quote

# Content words: unexamined, life, worth, living  ("the", "is", "not" are stopwords)
QUOTE = "The unexamined life is not worth living."
QUOTES = {"q1": QUOTE, "q2": "Happiness depends upon ourselves."}


def row(text, quote_id="q1", mode="question"):
    return {"input": text, "quote_id": quote_id, "mode": mode}


# --- overlap ----------------------------------------------------------------------


def test_content_words_ignores_stopwords_case_and_punctuation():
    assert content_words(QUOTE) == {"unexamined", "life", "worth", "living"}
    assert content_words("Isn't it WORTH it?!") == {"worth"}


def test_overlap_ratio():
    assert overlap_ratio("is my life going anywhere", QUOTE) == 0.25          # 1 of 4
    assert overlap_ratio("is this life worth it", QUOTE) == 0.5               # 2 of 4
    assert overlap_ratio("the is not", QUOTE) == 0.0                          # only stopwords shared
    assert overlap_ratio("anything", "the and of") == 0.0                     # quote has no content words


def test_overlap_filter_threshold():
    # 5 content words: sharing 2 is exactly 40% (kept), sharing 3 is 60% (dropped).
    quotes = {"q": "Courage grows slowly through daily practice."}
    kept, dropped = apply_filters(
        [row("courage and practice matter to me", "q"), row("courage grows with practice", "q")], quotes
    )
    assert [r["input"] for r in kept] == ["courage and practice matter to me"]
    assert dropped["overlap"] == 1


# --- dedup ------------------------------------------------------------------------


def test_dedup_ignores_case_and_whitespace_within_a_mode():
    rows = [
        row("Why do I feel stuck?"),
        row("  why do i   FEEL stuck?  ", quote_id="q2"),  # same text, other quote, same mode
        row("why do I feel stuck?", mode="advice"),         # same text, other mode: kept
    ]
    kept, dropped = apply_filters(rows, QUOTES)
    assert [(r["quote_id"], r["mode"]) for r in kept] == [("q1", "question"), ("q1", "advice")]
    assert dropped["duplicate"] == 1


def test_empty_and_too_long_inputs_are_dropped():
    kept, dropped = apply_filters([row("   "), row("x" * 1001), row("y" * 1000)], QUOTES)
    assert [len(r["input"]) for r in kept] == [1000]
    assert dropped["empty"] == 1 and dropped["too_long"] == 1


# --- train/val split ----------------------------------------------------------------


@pytest.fixture
def pairs():
    return pd.DataFrame(
        [{"input": f"text {q}-{i}", "quote_id": f"q{q}", "mode": "question", "philosopher": "x"}
         for q in range(200) for i in range(5)]
    )


def test_split_never_puts_a_quote_in_both(pairs):
    train, val = split_by_quote(pairs, 0.1, seed=42)
    assert set(train["quote_id"]).isdisjoint(set(val["quote_id"]))
    assert len(train) + len(val) == len(pairs)
    assert val["quote_id"].nunique() == 20  # 10% of 200 quotes
    # every quote keeps all 5 of its pairs in its split
    assert (val.groupby("quote_id").size() == 5).all()
    assert (train.groupby("quote_id").size() == 5).all()


def test_split_is_reproducible_and_ignores_row_order(pairs):
    _, val_a = split_by_quote(pairs, 0.1, seed=42)
    _, val_b = split_by_quote(pairs.sample(frac=1, random_state=0), 0.1, seed=42)
    assert set(val_a["quote_id"]) == set(val_b["quote_id"])
