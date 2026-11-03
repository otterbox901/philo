"""Text normalization helpers for the quote corpus.

Pure string functions shared by build_corpus.py and the tests: author cleanup and
matching keys, quote text cleanup, and the dedupe key.
"""

from __future__ import annotations

import re
import unicodedata

# Letters that carry a diacritic-like stroke but have no Unicode decomposition,
# so NFKD alone would leave them alone ("Søren" would never match "Soren").
_NON_DECOMPOSING = str.maketrans(
    {
        "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ħ": "h", "Ħ": "H",
        "ı": "i", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss",
        "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D",
    }
)
# Apostrophes, plus the modifier letters used for ayn/hamza in transliterations
# ("Ibn ʿArabi"). These are dropped rather than turned into a space so that
# "O'Connor" and "OConnor" agree.
_APOSTROPHES = re.compile("['’‘`´ʹʻʼʽʾʿ]")
_NON_WORD = re.compile(r"[^\w\s]|_")
_WHITESPACE = re.compile(r"\s+")

_DOUBLE_QUOTES = '"“”„‟«»'
_DOUBLE_OPENERS = "“„‟«"
_DOUBLE_CLOSERS = "”»"
_SINGLE_QUOTES = "'‘’‚‛"


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def strip_accents(s: str) -> str:
    decomposed = unicodedata.normalize("NFKD", s.translate(_NON_DECOMPOSING))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def collapse_whitespace(s: str) -> str:
    return _WHITESPACE.sub(" ", s).strip()


def _trimmable(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def trim_punctuation(s: str) -> str:
    """Strip whitespace and punctuation from both ends."""
    start, end = 0, len(s)
    while start < end and _trimmable(s[start]):
        start += 1
    while end > start and _trimmable(s[end - 1]):
        end -= 1
    return s[start:end]


def clean_author(raw: str, drop_book_suffix: bool = False) -> str:
    """Display-quality author name.

    drop_book_suffix handles "Author, Book Title" (Quotes 500k) by keeping only
    the text before the first comma.
    """
    s = nfkc(raw)
    if drop_book_suffix:
        s = s.split(",", 1)[0]
    return trim_punctuation(collapse_whitespace(s))


def author_key(name: str) -> str:
    """Matching key for an author name.

    Accents stripped, lowercased, punctuation removed, and runs of initials
    collapsed: "C. S. Lewis", "C.S. Lewis" and "CS Lewis" all give "cs lewis".
    """
    s = strip_accents(nfkc(name)).casefold()
    s = _APOSTROPHES.sub("", s)
    s = _NON_WORD.sub(" ", s)
    tokens: list[str] = []
    initials = ""
    for tok in s.split():
        if len(tok) == 1 and tok.isalpha():
            initials += tok
            continue
        if initials:
            tokens.append(initials)
            initials = ""
        tokens.append(tok)
    if initials:
        tokens.append(initials)
    return " ".join(tokens)


def _is_apostrophe(s: str, i: int) -> bool:
    """True if s[i] sits between two letters, like the ’ in don’t."""
    return 0 < i < len(s) - 1 and s[i - 1].isalpha() and s[i + 1].isalpha()


def _wraps(s: str) -> bool:
    """True if s[0] and s[-1] are a quote pair enclosing the whole string.

    Guards against text that merely starts and ends with quoted speech, such
    as “Yes,” he said, “go.”
    """
    first, last, inner = s[0], s[-1], s[1:-1]
    if first in _DOUBLE_QUOTES and last in _DOUBLE_QUOTES:
        if first == '"':
            return '"' not in inner
        depth = 1
        for ch in inner:
            if ch in _DOUBLE_OPENERS:
                depth += 1
            elif ch in _DOUBLE_CLOSERS:
                depth -= 1
                if depth == 0:
                    return False
        return True
    if first in _SINGLE_QUOTES and last in _SINGLE_QUOTES:
        return not any(ch in _SINGLE_QUOTES and not _is_apostrophe(inner, i) for i, ch in enumerate(inner))
    return False


def strip_wrapping_quotes(s: str) -> str:
    """Remove quote marks that wrap the whole text, straight or curly.

    A lone double quote mark at either end is also dropped when it is the only
    double quote in the text. Lone single quotes are kept, since they are
    usually apostrophes ('Tis, students').
    """
    while len(s) >= 2:
        if _wraps(s):
            s = s[1:-1].strip()
        elif s[0] in _DOUBLE_QUOTES and sum(ch in _DOUBLE_QUOTES for ch in s) == 1:
            s = s[1:].strip()
        elif s[-1] in _DOUBLE_QUOTES and sum(ch in _DOUBLE_QUOTES for ch in s) == 1:
            s = s[:-1].strip()
        else:
            break
    return s


def clean_quote(text: str) -> str:
    return strip_wrapping_quotes(collapse_whitespace(nfkc(text)))


def dedupe_key(text: str) -> str:
    """Lowercased, accent-stripped, alphanumerics-only form of a quote."""
    return "".join(ch for ch in strip_accents(nfkc(text)).casefold() if ch.isalnum())
