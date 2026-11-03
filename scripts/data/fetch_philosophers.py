"""Fetch philosophers and theologians from Wikidata into data/interim/philosophers.csv.

Selects people whose occupation (P106) is philosopher (Q4964182) or theologian
(Q1234713) and who have at least 5 sitelinks, with their English label and
English aliases. Each occupation is queried separately and paginated to stay
under the endpoint's timeout.

Usage: python scripts/data/fetch_philosophers.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "interim" / "philosophers.csv"

ENDPOINT = "https://query.wikidata.org/sparql"
# Wikimedia asks for a descriptive User-Agent; add contact info here if you have some.
USER_AGENT = "philo-quote-corpus/0.1 (personal learning project; quote corpus cleaning) python-requests"
OCCUPATIONS = {"philosopher": "Q4964182", "theologian": "Q1234713"}
MIN_SITELINKS = 5
PAGE_SIZE = 2000
MAX_RETRIES = 5

# The inner sub-select pages over a stable ordering of matching people; labels
# and aliases are attached outside it so the paging is cheap.
QUERY = """
SELECT ?person ?label (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {{
  {{
    SELECT ?person WHERE {{
      ?person wdt:P106 wd:{occupation} ;
              wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks >= {min_sitelinks})
    }}
    ORDER BY ?person
    LIMIT {limit} OFFSET {offset}
  }}
  OPTIONAL {{ ?person rdfs:label ?label . FILTER(LANG(?label) = "en") }}
  OPTIONAL {{ ?person skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }}
}}
GROUP BY ?person ?label
"""


def run_query(session: requests.Session, query: str) -> list[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(ENDPOINT, params={"query": query, "format": "json"}, timeout=120)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            wait = int(resp.headers.get("Retry-After", 0)) or 5 * 2 ** attempt
            print(f"  HTTP {resp.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["results"]["bindings"]
    raise RuntimeError("unreachable")


def fetch_occupation(session: requests.Session, occupation: str) -> list[dict]:
    rows, offset = [], 0
    while True:
        query = QUERY.format(occupation=occupation, min_sitelinks=MIN_SITELINKS, limit=PAGE_SIZE, offset=offset)
        page = run_query(session, query)
        for b in page:
            rows.append(
                {
                    "qid": b["person"]["value"].rsplit("/", 1)[-1],
                    "name": b.get("label", {}).get("value", ""),
                    "aliases": b.get("aliases", {}).get("value", ""),
                }
            )
        print(f"  offset {offset}: {len(page)} rows")
        if len(page) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE
        time.sleep(1)  # be polite to the public endpoint


def merge_aliases(values: pd.Series) -> str:
    aliases = {a for v in values for a in v.split("|") if a}
    return "|".join(sorted(aliases))


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})

    frames = []
    for name, occupation in OCCUPATIONS.items():
        print(f"{name} ({occupation})")
        frames.append(pd.DataFrame(fetch_occupation(session, occupation)))
    people = pd.concat(frames, ignore_index=True)
    print(f"{len(people):,} rows fetched, {people.qid.nunique():,} unique QIDs")

    # People with both occupations come back once per query; collapse them.
    people =people.groupby("qid", sort=False).agg(name=("name", "first"), aliases=("aliases", merge_aliases)).reset_index()
    no_label = people.name.eq("")
    print(f"dropping {no_label.sum():,} people without an English label")
    people = people[~no_label].sort_values("qid", key=lambda s: s.str[1:].astype(int))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    people.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(people):,} people to {OUT_PATH}")


if __name__ == "__main__":
    main()
