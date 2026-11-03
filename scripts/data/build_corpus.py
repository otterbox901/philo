"""Build the cleaned philosopher-quote corpus from the raw sources.

Inputs:  data/raw/*, data/interim/philosophers.csv, configs/author_{allow,deny}list.txt
Outputs: data/processed/quotes.parquet, data/processed/quotes_sample.csv,
         data/interim/unmatched_authors.csv, data/interim/author_conflicts.csv,
         data/interim/build_report.md

Usage: python scripts/data/build_corpus.py
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fetch_raw import RAW_DIR, read_raw
from textnorm import author_key, clean_author, clean_quote, dedupe_key, nfkc

ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
PHILOSOPHERS_CSV = INTERIM_DIR / "philosophers.csv"
ALLOWLIST = ROOT / "configs" / "author_allowlist.txt"
DENYLIST = ROOT / "configs" / "author_denylist.txt"

MIN_QUOTE_LEN = 20
MAX_QUOTE_LEN = 400
SAMPLE_SIZE = 200
UNMATCHED_TOP_N = 300
REPORT_TOP_AUTHORS = 40

OUTPUT_SCHEMA = pa.schema(
    [
        ("quote_id", pa.string()),
        ("text", pa.string()),
        ("author", pa.string()),
        ("author_qid", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("sources", pa.list_(pa.string())),
    ]
)


# --- author matching ---------------------------------------------------------


@dataclass(frozen=True)
class NameEntry:
    canonical: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)


@dataclass(frozen=True)
class Match:
    name: str
    qid: str | None


def parse_name_list(text: str) -> list[NameEntry]:
    """Parse "Canonical Name | alias | alias" lines; '#' starts a comment."""
    entries = []
    for line in text.splitlines():
        names = [n.strip() for n in line.split("#", 1)[0].split("|")]
        names = [n for n in names if n]
        if names:
            entries.append(NameEntry(names[0], tuple(names[1:])))
    return entries


def _qid_number(qid: str) -> int:
    return int(qid[1:])


class AuthorMatcher:
    """Maps raw author names to a canonical name and (if known) a Wikidata QID.

    A name matches if its author_key equals the key of a Wikidata label or
    alias, or of an allowlist name or alias. Allowlist entries take precedence
    and supply the canonical name; the denylist overrides everything.
    """

    def __init__(
        self,
        people: Iterable[tuple[str, str, Iterable[str]]],
        allow: Iterable[NameEntry] = (),
        deny: Iterable[NameEntry] = (),
    ):
        labels: dict[str, str] = {}
        label_hits: dict[str, set[str]] = {}
        alias_hits: dict[str, set[str]] = {}
        for qid, label, aliases in people:
            labels[qid] = label
            label_hits.setdefault(author_key(label), set()).add(qid)
            for alias in aliases:
                alias_hits.setdefault(author_key(alias), set()).add(qid)

        # A key shared by several people resolves to a label match over an alias
        # match, then to the lowest QID (older items are usually the better-known
        # person). Shared keys are counted so the report can flag them.
        self.ambiguous_keys = 0
        self._wikidata: dict[str, str] = {}
        for key in label_hits.keys() | alias_hits.keys():
            if not key:
                continue
            candidates = label_hits.get(key) or alias_hits[key]
            if len(label_hits.get(key, set()) | alias_hits.get(key, set())) > 1:
                self.ambiguous_keys += 1
            self._wikidata[key] = min(candidates, key=_qid_number)

        # An allowlisted person's canonical name also replaces their Wikidata
        # label, so that Wikidata-only aliases (e.g. "Gilbert Keith Chesterton")
        # land on the same name as the allowlist spelling.
        canonical: dict[str, str] = {}
        allow_index: dict[str, Match] = {}
        for entry in allow:
            keys = [author_key(n) for n in entry.names]
            qid = next((self._wikidata[k] for k in keys if k in self._wikidata), None)
            if qid is not None:
                canonical[qid] = entry.canonical
            for key in keys:
                allow_index[key] = Match(entry.canonical, qid)
        self._index: dict[str, Match] = {
            k: Match(canonical.get(q, labels[q]), q) for k, q in self._wikidata.items()
        }
        self._index.update(allow_index)

        self._deny = {author_key(n) for entry in deny for n in entry.names}

    def is_denied(self, author: str) -> bool:
        key = author_key(author)
        found = self._index.get(key)
        return key in self._deny or (found is not None and author_key(found.name) in self._deny)

    def match(self, author: str) -> Match | None:
        found = self._index.get(author_key(author))
        if found is None or self.is_denied(author):
            return None
        return found


def load_matcher() -> AuthorMatcher:
    people = pd.read_csv(PHILOSOPHERS_CSV, dtype=str, keep_default_na=False)
    return AuthorMatcher(
        ((p.qid, p.name, [a for a in p.aliases.split("|") if a]) for p in people.itertuples()),
        allow=parse_name_list(ALLOWLIST.read_text()),
        deny=parse_name_list(DENYLIST.read_text()),
    )


# --- pipeline steps -----------------------------------------------------------


def normalize_tags(tags: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(t for t in (nfkc(t).strip().lower() for t in tags) if t))


def load_sources() -> pd.DataFrame:
    hf = read_raw("english_quotes", RAW_DIR / "english_quotes")
    kg = read_raw("quotes_500k", RAW_DIR / "quotes_500k")
    frames = [
        pd.DataFrame(
            {
                "text": hf.quote,
                "author_raw": hf.author,
                "tags": hf.tags.map(normalize_tags),
                "source": "english_quotes",
            }
        ),
        pd.DataFrame(
            {
                "text": kg.quote,
                "author_raw": kg.author,
                "tags": kg.category.map(lambda c: normalize_tags(c.split(","))),
                "source": "quotes_500k",
            }
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def clean_authors(df: pd.DataFrame) -> pd.DataFrame:
    pairs = df[["author_raw", "source"]].drop_duplicates()
    cleaned = {
        (raw, src): clean_author(raw, drop_book_suffix=src == "quotes_500k")
        for raw, src in pairs.itertuples(index=False)
    }
    author_clean = [cleaned[k] for k in zip(df.author_raw, df.source)]
    return df.assign(author_clean=author_clean)


def match_authors(df: pd.DataFrame, matcher: AuthorMatcher) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Keep rows whose author matches. Also returns the unmatched-author table
    and the number of rows dropped by the denylist."""
    names = df.author_clean.unique()
    matches = {a: matcher.match(a) for a in names}
    denied = {a for a in names if matcher.is_denied(a)}

    found = df.author_clean.map(matches)
    is_match = found.notna()
    is_denied = df.author_clean.isin(denied)

    # Count per match key, so "C.S. Lewis" and "C. S. Lewis" are one entry,
    # shown under its most common spelling.
    unmatched = df.loc[~is_match & ~is_denied & df.author_clean.ne(""), "author_clean"]
    spellings = (
        pd.DataFrame({"author": unmatched, "match_key": unmatched.map(author_key)})
        .value_counts()
        .reset_index(name="count")
    )
    unmatched = (
        spellings.groupby("match_key", as_index=False)
        .agg(author=("author", "first"), count=("count", "sum"))
        .sort_values(["count", "match_key"], ascending=[False, True])
        .head(UNMATCHED_TOP_N)[["author", "match_key", "count"]]
    )

    kept = df[is_match].copy()
    kept["author"] = [m.name for m in found[is_match]]
    kept["author_qid"] = [m.qid for m in found[is_match]]
    return kept, unmatched, int(is_denied.sum())


def clean_texts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(text=df.text.map(clean_quote))
    length = df.text.str.len()
    return df[length.between(MIN_QUOTE_LEN, MAX_QUOTE_LEN)]


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge rows with the same dedupe key; returns (merged, author conflicts)."""
    df = df.assign(key=df.text.map(dedupe_key))
    is_dup = df.key.duplicated(keep=False)
    singles = df[~is_dup]
    records = [
        {"key": r.key, "text": r.text, "author": r.author, "author_qid": r.author_qid,
         "tags": sorted(r.tags), "sources": [r.source]}
        for r in singles.itertuples(index=False)
    ]
    conflicts = []
    for key, group in df[is_dup].groupby("key", sort=False):
        authors = Counter(group.author)
        # Most frequent attribution wins; ties go to the first one seen, and
        # english_quotes is loaded first.
        author = max(authors, key=authors.get)
        first = group[group.author == author].iloc[0]
        records.append(
            {
                "key": key,
                "text": first.text,
                "author": author,
                "author_qid": first.author_qid,
                "tags": sorted(set().union(*group.tags)),
                "sources": sorted(set(group.source)),
            }
        )
        if len(authors) > 1:
            conflicts.append(
                {
                    "key": key,
                    "text": first.text,
                    "chosen_author": author,
                    "attributions": "|".join(f"{a} ({n})" for a, n in authors.most_common()),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(conflicts, columns=["key", "text", "chosen_author", "attributions"])


def quote_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# --- outputs -------------------------------------------------------------------


def markdown_table(df: pd.DataFrame) -> str:
    rows = [list(map(str, df.columns)), ["---"] * len(df.columns)]
    rows += [["" if pd.isna(v) else str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def write_report(steps: list[tuple[str, int]], corpus: pd.DataFrame, notes: list[str]) -> None:
    top = (
        corpus.groupby(["author", "author_qid"], dropna=False)
        .size()
        .reset_index(name="quotes")
        .sort_values(["quotes", "author"], ascending=[False, True])
        .head(REPORT_TOP_AUTHORS)
    )
    top.insert(0, "rank", range(1, len(top) + 1))
    lines = [
        "# Corpus build report",
        "",
        "Generated by `scripts/data/build_corpus.py`.",
        "",
        "## Row counts per step",
        "",
        markdown_table(pd.DataFrame(steps, columns=["step", "rows"])),
        "",
        f"**Final corpus: {len(corpus):,} quotes by {corpus.author.nunique():,} authors.**",
        "",
        "## Notes",
        "",
        *[f"- {n}" for n in notes],
        "",
        f"## Top {REPORT_TOP_AUTHORS} authors by quote count",
        "",
        markdown_table(top),
        "",
    ]
    (INTERIM_DIR / "build_report.md").write_text("\n".join(lines))


def main() -> None:
    steps: list[tuple[str, int]] = []

    def log(step: str, df: pd.DataFrame) -> None:
        steps.append((step, len(df)))
        print(f"{step}: {len(df):,} rows")

    df = load_sources()
    log("1. load raw sources", df)
    notes = [f"Raw rows by source: {', '.join(f'{s} {n:,}' for s, n in df.source.value_counts().items())}."]

    df = clean_authors(df)
    log("2. author cleanup", df)

    matcher = load_matcher()
    df, unmatched, denied_rows = match_authors(df, matcher)
    log("3. keep matched authors", df)
    notes.append(f"{denied_rows:,} rows dropped by the denylist.")
    notes.append(
        f"{matcher.ambiguous_keys:,} Wikidata name keys are shared by more than one person; "
        "they resolve to a label match first, then the lowest QID."
    )

    df = clean_texts(df)
    log(f"4. clean text, keep {MIN_QUOTE_LEN}-{MAX_QUOTE_LEN} chars", df)

    corpus, conflicts = dedupe(df)
    log("5. exact dedupe", corpus)
    notes.append(f"{len(conflicts):,} merged quotes had conflicting author attributions (see author_conflicts.csv).")

    corpus.insert(0, "quote_id", corpus.key.map(quote_id))
    if corpus.quote_id.duplicated().any():
        raise SystemExit("quote_id collision; use more hex characters")
    conflicts.insert(0, "quote_id", conflicts.key.map(quote_id))
    corpus = corpus.drop(columns="key").sort_values("quote_id", ignore_index=True)
    log("6. assign quote_id", corpus)
    notes.append(f"Final rows by source (a quote can have several): "
                 f"{', '.join(f'{s} {n:,}' for s, n in corpus.sources.explode().value_counts().items())}.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(corpus, schema=OUTPUT_SCHEMA, preserve_index=False), PROCESSED_DIR / "quotes.parquet")

    sample = corpus.sample(n=min(SAMPLE_SIZE, len(corpus)), random_state=0)
    sample.assign(tags=sample.tags.str.join("|"), sources=sample.sources.str.join("|")).to_csv(
        PROCESSED_DIR / "quotes_sample.csv", index=False
    )
    unmatched.to_csv(INTERIM_DIR / "unmatched_authors.csv", index=False)
    conflicts.drop(columns="key").to_csv(INTERIM_DIR / "author_conflicts.csv", index=False)
    write_report(steps, corpus, notes)
    print(f"wrote outputs to {PROCESSED_DIR} and {INTERIM_DIR}")


if __name__ == "__main__":
    main()
