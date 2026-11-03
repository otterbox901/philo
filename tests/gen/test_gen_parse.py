"""Tests for parsing the model's JSON answer (no API calls)."""

import json

import pytest

from gen.collect import ParseError, message_text, mode_of, parse_response

IDS = ["q1", "q2"]


def answer(results):
    return json.dumps({"results": results})


def test_valid_answer():
    text = answer([
        {"quote_id": "q1", "fits": True, "reason": "ok", "inputs": ["a", " b "]},
        {"quote_id": "q2", "fits": False, "reason": "too dry", "inputs": []},
    ])
    parsed = parse_response(text, IDS)
    assert parsed == [
        {"quote_id": "q1", "fits": True, "reason": "ok", "inputs": ["a", "b"]},  # inputs are stripped
        {"quote_id": "q2", "fits": False, "reason": "too dry", "inputs": []},
    ]


def test_code_fences_and_surrounding_prose():
    text = "Here you go:\n```json\n" + answer([{"quote_id": "q1", "fits": False, "reason": "x", "inputs": []}]) + "\n```"
    assert parse_response(text, IDS)[0]["quote_id"] == "q1"


def test_raw_newline_inside_string_is_accepted():
    text = '{"results": [{"quote_id": "q1", "fits": true, "reason": "r", "inputs": ["line one\nline two"]}]}'
    assert parse_response(text, IDS)[0]["inputs"] == ["line one\nline two"]


@pytest.mark.parametrize("text", [
    "",                                    # empty
    "Sorry, I can't help with that.",      # no JSON at all
    '{"results": [{"quote_id": "q1", "fi',  # cut off (e.g. hit max_tokens)
    '{"results": "nope"}',                 # results is not a list
    '{"answers": []}',                     # wrong key
    "[1, 2, 3]",                           # not an object
])
def test_malformed_output_raises(text):
    with pytest.raises(ParseError):
        parse_response(text, IDS)


def test_bad_entries_are_skipped():
    text = answer([
        {"quote_id": "unknown", "fits": True, "reason": "", "inputs": ["x"]},    # id we didn't send
        {"quote_id": "q1", "fits": "yes", "reason": "", "inputs": ["x"]},        # fits not a bool
        {"quote_id": "q2", "fits": True, "reason": "", "inputs": "not a list"},  # inputs not a list
        "just a string",
    ])
    with pytest.raises(ParseError):  # nothing usable left
        parse_response(text, IDS)


def test_repeated_quote_keeps_first_and_non_strings_dropped():
    text = answer([
        {"quote_id": "q1", "fits": True, "reason": "first", "inputs": ["a", 5, None]},
        {"quote_id": "q1", "fits": True, "reason": "second", "inputs": ["b"]},
    ])
    parsed = parse_response(text, IDS)
    assert len(parsed) == 1
    assert parsed[0]["reason"] == "first"
    assert parsed[0]["inputs"] == ["a"]


def test_rejected_quote_gets_no_inputs():
    text = answer([{"quote_id": "q1", "fits": False, "reason": "no", "inputs": ["should be ignored"]}])
    assert parse_response(text, IDS)[0]["inputs"] == []


def test_message_text_skips_thinking_blocks():
    message = {"content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "{}"}]}
    assert message_text(message) == "{}"


def test_mode_of():
    assert mode_of("nonsense-499") == "nonsense"
