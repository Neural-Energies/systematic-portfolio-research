from itertools import combinations

import numpy as np

from systematic_research.entry_discovery_search import (
    execution_metrics,
    hit_tier,
    screen_rules,
    unrank_triples,
)


def test_canonical_generator_has_no_duplicate_or_missing_triples() -> None:
    expected = np.asarray(list(combinations(range(8), 3)))
    actual = unrank_triples(np.arange(len(expected)), 8)
    np.testing.assert_array_equal(actual, expected)


def test_tiers_keep_save_cutoff_separate_from_high_tiers() -> None:
    assert hit_tier(0.53) == "watch_50_to_53"
    assert hit_tier(0.54) == "saved_above_53"
    assert hit_tier(0.7) == "tier_1_70"
    assert hit_tier(0.9) == "tier_2_90"
    assert hit_tier(1.0) == "tier_3_100"


def test_screen_counts_and_disallows_same_feature_gates() -> None:
    atoms = np.array([[15], [7], [3], [1]], dtype=np.uint64)
    definitions = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    masks = np.array([[3], [12]], dtype=np.uint64)
    hits = np.array([[1], [2], [3]], dtype=np.uint64)
    scores = screen_rules(atoms, definitions, masks, hits, np.array([0, 1, 2, 0]))
    np.testing.assert_array_equal(scores[0], [2, 1, 1, 2, 0, 0, 0, 0, 1])
    assert not scores[1].any()


def test_unresolved_trade_is_not_free_position_or_invented_profit() -> None:
    signal = np.array([7], dtype=np.uint64)
    stage = np.array([0, 0, 0])
    starts = np.array([0, 5, 10])
    ends = np.array([10, 15, 20])
    pnl = np.array([np.nan, 100.0, -25.0])
    result = execution_metrics(
        signal,
        stage,
        starts,
        ends,
        pnl,
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([0, 0, 0]),
    )
    assert result[0, 0] == 2
    assert result[0, 1] == 1
    assert result[0, 2] == -25.0
    assert result[0, 3] == 25.0


def test_high_bits_and_full_words_are_counted_without_overflow() -> None:
    maximum = np.iinfo(np.uint64).max
    atoms = np.full((3, 2), maximum, dtype=np.uint64)
    masks = np.array([[maximum, 0], [0, maximum]], dtype=np.uint64)
    hit_masks = np.full((3, 2), maximum, dtype=np.uint64)
    actual = screen_rules(
        atoms, np.array([[0, 1, 2]], dtype=np.int64), masks, hit_masks, np.arange(3, dtype=np.int64)
    )
    np.testing.assert_array_equal(actual[0], [64, 64, 64, 64, 64, 64, 64, 64, 1])
