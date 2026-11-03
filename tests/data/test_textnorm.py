import pytest

from textnorm import author_key, clean_author, clean_quote, dedupe_key, strip_wrapping_quotes


# --- author_key: initials ------------------------------------------------------


@pytest.mark.parametrize("name", ["C. S. Lewis", "C.S. Lewis", "C.S.Lewis", "CS Lewis", "c s lewis"])
def test_initials_collapse(name):
    assert author_key(name) == "cs lewis"


def test_multiple_initials_collapse():
    assert author_key("J. R. R. Tolkien") == "jrr tolkien"
    assert author_key("G.K. Chesterton") == author_key("G. K. Chesterton") == "gk chesterton"


def test_single_middle_initial_is_kept_as_its_own_token():
    assert author_key("Viktor E. Frankl") == "viktor e frankl"


# --- author_key: accents and punctuation ---------------------------------------


@pytest.mark.parametrize(
    "accented, plain",
    [
        ("Antoine de Saint-Exupéry", "Antoine de Saint Exupery"),
        ("Thích Nhất Hạnh", "Thich Nhat Hanh"),
        ("Søren Kierkegaard", "Soren Kierkegaard"),
        ("Friedrich Nietzsche", "FRIEDRICH NIETZSCHE"),
    ],
)
def test_accents_and_case_are_ignored(accented, plain):
    assert author_key(accented) == author_key(plain)


def test_accent_stripping_output():
    assert author_key("Søren Kierkegaard") == "soren kierkegaard"


def test_apostrophes_are_dropped_and_other_punctuation_splits_words():
    assert author_key("Flannery O'Connor") == author_key("Flannery O’Connor") == "flannery oconnor"
    assert author_key("Jalal ad-Din Muhammad Rumi") == "jalal ad din muhammad rumi"


def test_whitespace_collapses():
    assert author_key("  Lao    Tzu ") == "lao tzu"


# --- clean_author: book suffix and trimming -------------------------------------


def test_book_suffix_is_dropped():
    assert clean_author("Billy Graham, Billy Graham in Quotes", drop_book_suffix=True) == "Billy Graham"


def test_book_suffix_splits_on_first_comma_only():
    raw = "Sunday Adelaja, How To Become Great Through Time Conversion: Are you wasting time, spending time"
    assert clean_author(raw, drop_book_suffix=True) == "Sunday Adelaja"


def test_name_without_comma_is_unchanged_by_suffix_stripping():
    assert clean_author("Ralph Waldo Emerson", drop_book_suffix=True) == "Ralph Waldo Emerson"


def test_comma_kept_when_suffix_stripping_is_off():
    assert clean_author("Smith, John") == "Smith, John"


def test_trailing_punctuation_and_whitespace_are_trimmed():
    assert clean_author("  Oscar Wilde,  ") == "Oscar Wilde"
    assert clean_author("(Anonymous)") == "Anonymous"


def test_author_is_nfkc_normalized():
    # Fullwidth comma is NFKC-normalized to "," before the suffix split.
    assert clean_author("Rumi，Masnavi", drop_book_suffix=True) == "Rumi"


# --- quote text: wrapping quote marks -------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("“Be yourself.”", "Be yourself."),
        ('"Be yourself."', "Be yourself."),
        ("‘Be yourself.’", "Be yourself."),
        ("'Be yourself.'", "Be yourself."),
        ("«Be yourself.»", "Be yourself."),
        ('"“Be yourself.”"', "Be yourself."),
        ("“ Be yourself. ”", "Be yourself."),
        ("“Be yourself.", "Be yourself."),
        ("Be yourself.”", "Be yourself."),
        ("“She said “hi” to me.”", "She said “hi” to me."),
        ("‘I don’t know.’", "I don’t know."),
    ],
)
def test_wrapping_quotes_are_stripped(raw, expected):
    assert strip_wrapping_quotes(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "“Yes,” he said, “go.”",
        '"Yes," he said, "go."',
        "'Tis the season.",
        "He called it “the end.”",
        "No quotes here.",
    ],
)
def test_non_wrapping_quotes_are_kept(text):
    assert strip_wrapping_quotes(text) == text


def test_clean_quote_normalizes_whitespace_then_strips_quotes():
    assert clean_quote("  “Hello\n\n   world…”  ") == "Hello world..."


# --- dedupe key -------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        ("To be, or not to be!", "to be or NOT to be"),
        ("“To be, or not to be.”", "To be -- or not to be"),
        ("Café society", "cafe society"),
        ("Don’t panic.", "Dont panic"),
    ],
)
def test_dedupe_key_ignores_punctuation_case_and_accents(a, b):
    assert dedupe_key(a) == dedupe_key(b)


def test_dedupe_key_distinguishes_different_words():
    assert dedupe_key("To be or not to be") != dedupe_key("To be or not to see")


def test_dedupe_key_is_alphanumeric_only():
    assert dedupe_key("Hello, world: 42!") == "helloworld42"
