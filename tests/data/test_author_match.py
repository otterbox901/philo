import pytest

from build_corpus import AuthorMatcher, Match, NameEntry, parse_name_list

PEOPLE = [
    ("Q9204", "C. S. Lewis", ["Clive Staples Lewis"]),
    ("Q9358", "Friedrich Nietzsche", ["Friedrich Wilhelm Nietzsche"]),
    ("Q9333", "Laozi", ["Lao-tzu"]),
    ("Q183167", "G. K. Chesterton", ["Gilbert Keith Chesterton"]),
]


@pytest.fixture
def matcher():
    return AuthorMatcher(
        PEOPLE,
        allow=[
            NameEntry("Lao Tzu", ("Laozi", "Lao Tse")),
            NameEntry("G.K. Chesterton"),
            NameEntry("Oscar Wilde"),
        ],
    )


# --- matching ---------------------------------------------------------------------


def test_wikidata_label_matches_across_spelling_variants(matcher):
    assert matcher.match("C.S. Lewis") == Match("C. S. Lewis", "Q9204")


def test_wikidata_alias_matches_to_label(matcher):
    assert matcher.match("Clive Staples Lewis") == Match("C. S. Lewis", "Q9204")
    assert matcher.match("friedrich wilhelm nietzsche") == Match("Friedrich Nietzsche", "Q9358")


def test_allowlist_alias_matches_to_canonical_name_with_wikidata_qid(matcher):
    assert matcher.match("Lao Tse") == Match("Lao Tzu", "Q9333")
    assert matcher.match("Laozi") == Match("Lao Tzu", "Q9333")


def test_allowlist_canonical_name_also_covers_wikidata_only_aliases(matcher):
    assert matcher.match("Gilbert Keith Chesterton") == Match("G.K. Chesterton", "Q183167")
    assert matcher.match("Lao-tzu") == Match("Lao Tzu", "Q9333")


def test_allowlist_author_without_wikidata_entry_has_no_qid(matcher):
    assert matcher.match("Oscar Wilde") == Match("Oscar Wilde", None)


def test_unknown_author_does_not_match(matcher):
    assert matcher.match("Stephen King") is None
    assert matcher.match("") is None


def test_shared_key_prefers_label_over_alias_then_lowest_qid():
    m = AuthorMatcher(
        [
            ("Q50", "John Smith", []),
            ("Q10", "Jonathan Smith", ["John Smith"]),
            ("Q300", "Ann Lee", []),
            ("Q200", "Ann Lee", []),
        ]
    )
    assert m.match("John Smith") == Match("John Smith", "Q50")
    assert m.match("Ann Lee") == Match("Ann Lee", "Q200")
    assert m.ambiguous_keys == 2


# --- denylist -----------------------------------------------------------------------


def test_denylist_overrides_wikidata_match_including_aliases():
    m = AuthorMatcher(PEOPLE, deny=[NameEntry("Friedrich Nietzsche")])
    assert m.match("Friedrich Nietzsche") is None
    assert m.match("Friedrich Wilhelm Nietzsche") is None
    assert m.is_denied("Friedrich Wilhelm Nietzsche")
    assert m.match("C.S. Lewis") == Match("C. S. Lewis", "Q9204")


def test_denylist_overrides_allowlist():
    m = AuthorMatcher([], allow=[NameEntry("Oscar Wilde")], deny=[NameEntry("Oscar Wilde")])
    assert m.match("Oscar Wilde") is None


def test_denylist_alias_matches_spelling_variants():
    m = AuthorMatcher(PEOPLE, deny=[NameEntry("Somebody Else", ("C. S. Lewis",))])
    assert m.match("CS Lewis") is None
    assert m.match("Clive Staples Lewis") is None


# --- list file parsing ----------------------------------------------------------------


def test_parse_name_list():
    text = """
# header comment
Rumi | Jalaluddin Rumi | Jalal ad-Din Muhammad Rumi
  Oscar Wilde   # trailing comment

Lao Tzu|Laozi
"""
    assert parse_name_list(text) == [
        NameEntry("Rumi", ("Jalaluddin Rumi", "Jalal ad-Din Muhammad Rumi")),
        NameEntry("Oscar Wilde"),
        NameEntry("Lao Tzu", ("Laozi",)),
    ]
