"""Tests for the ranking and bootstrap helpers in eval.py (no model or data needed)."""

import numpy as np
import pytest
import torch

from retrieval.eval import cluster_bootstrap, ranks

# Three unit-length candidates and two queries with known orderings.
CANDIDATES = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
QUERIES = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
# Query 0 scores: [1.0, 0.0, 0.6] -> order 0, 2, 1
# Query 1 scores: [0.0, 1.0, 0.8] -> order 1, 2, 0


def test_ranks_puts_best_match_first():
    assert ranks(QUERIES, CANDIDATES, [0, 1]).tolist() == [1, 1]


def test_ranks_counts_candidates_that_score_higher():
    assert ranks(QUERIES, CANDIDATES, [1, 0]).tolist() == [3, 3]
    assert ranks(QUERIES, CANDIDATES, [2, 2]).tolist() == [2, 2]


def test_ranks_ties_count_in_targets_favour():
    tied = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert ranks(QUERIES[:1], tied, [1]).tolist() == [1]


def test_ranks_same_result_for_any_chunk_size():
    torch.manual_seed(0)
    queries = torch.nn.functional.normalize(torch.randn(50, 8), dim=1)
    candidates = torch.nn.functional.normalize(torch.randn(200, 8), dim=1)
    targets = torch.randint(0, 200, (50,))
    whole = ranks(queries, candidates, targets, chunk=1024)
    assert np.array_equal(whole, ranks(queries, candidates, targets, chunk=7))
    assert np.array_equal(whole, ranks(queries, candidates, targets, chunk=1))


def test_bootstrap_constant_values_have_no_spread():
    mean, lo, hi = cluster_bootstrap([1, 1, 1, 1], ["a", "a", "b", "c"])
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)


def test_bootstrap_interval_contains_the_mean():
    rng = np.random.default_rng(1)
    values = rng.integers(0, 2, size=300)
    groups = np.repeat(np.arange(30), 10)
    mean, lo, hi = cluster_bootstrap(values, groups)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_is_wider_when_groups_are_internally_identical():
    # Same 50/50 split of hits, but in the clustered case every group is all-0 or all-1,
    # so resampling groups should give a wider interval than when hits are spread evenly.
    groups = np.repeat(np.arange(20), 10)
    clustered = np.repeat([0, 1] * 10, 10)
    spread = np.tile([0, 1], 100)
    _, lo_c, hi_c = cluster_bootstrap(clustered, groups)
    _, lo_s, hi_s = cluster_bootstrap(spread, groups)
    assert hi_c - lo_c > hi_s - lo_s
    assert (hi_s - lo_s) == pytest.approx(0.0)
