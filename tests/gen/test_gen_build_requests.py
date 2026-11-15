"""Tests for sampling and request building (no API calls)."""

import pandas as pd

from gen.build_requests import build_requests, eligible_quotes, exclude_eval, sample_quote_ids
from gen.prompts import MODES


def test_eligible_quotes_keeps_5_to_60_words():
    quotes = pd.DataFrame({"quote_id": ["a", "b", "c", "d"],
                           "text": ["one two three four", "one two three four five", " ".join(["w"] * 60), " ".join(["w"] * 61)]})
    assert eligible_quotes(quotes, 5, 60)["quote_id"].tolist() == ["b", "c"]


def test_exclude_eval_by_id_or_text():
    quotes = pd.DataFrame({"quote_id": ["a", "b", "c"], "text": ["Keep me.", "Drop  ME.", "Drop by id."]})
    kept = exclude_eval(quotes, eval_ids={"c"}, eval_texts={"drop me."})
    assert kept["quote_id"].tolist() == ["a"]


def test_sample_is_reproducible_and_ignores_row_order():
    quotes = pd.DataFrame({"quote_id": [f"q{i}" for i in range(100)], "text": ["x"] * 100})
    first = sample_quote_ids(quotes, 10, seed=42)
    shuffled = sample_quote_ids(quotes.sample(frac=1, random_state=1), 10, seed=42)
    assert first == shuffled


def test_requests_one_per_mode_and_group():
    ids = [f"q{i}" for i in range(20)]  # 20 quotes -> groups of 8, 8, 4
    requests = build_requests(ids, {q: f"text {q}" for q in ids})
    assert len(requests) == 3 * len(MODES)
    assert requests[0]["custom_id"] == "question-0"
    assert requests[-1]["custom_id"] == "nonsense-2"
    assert requests[-1]["quote_ids"] == ids[16:]
    # the shared system prompt is marked for caching; the mode is in the user message
    assert requests[0]["params"]["system"][0]["cache_control"]["type"] == "ephemeral"
    assert "Mode: nonsense" in requests[-1]["params"]["messages"][0]["content"]


def test_later_rounds_never_reuse_earlier_quotes(tmp_path, monkeypatch):
    from gen import config
    from gen.build_requests import load_or_create_sample

    quotes = pd.DataFrame({"quote_id": [f"q{i}" for i in range(30)], "text": ["one two three four five"] * 30})
    round_dirs = {1: tmp_path, 2: tmp_path / "round2"}
    (tmp_path / "sample_ids.txt").write_text("\n".join(f"q{i}" for i in range(20)) + "\n")
    monkeypatch.setattr(config, "round_dir", lambda n: round_dirs[n])
    monkeypatch.setattr(config, "ROUND", 2)
    monkeypatch.setattr(config, "ROUND_DIR", round_dirs[2])
    monkeypatch.setattr(config, "SAMPLE_IDS_PATH", round_dirs[2] / "sample_ids.txt")
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path / "no_eval")
    monkeypatch.setattr(config, "N_QUOTES", 10)

    sampled = load_or_create_sample(quotes)
    assert sorted(sampled) == sorted(f"q{i}" for i in range(20, 30))  # only the 10 unused quotes
    assert load_or_create_sample(quotes) == sampled  # second call reuses the saved sample
