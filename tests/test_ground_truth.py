import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import ground_truth as gt  # noqa: E402


def test_logit_matches_documented_formula():
    # Referral, 10h response, 5 touchpoints: -1.2 + 1.0 - 0.5 + 0.35*4 - 0
    assert gt.logit("Referral", 10, 5) == pytest.approx(-1.2 + 1.0 - 0.5 + 1.4)
    # over-touching penalty: 9 touchpoints -> -0.10 * 3
    assert gt.logit("Website Form", 0, 9) == pytest.approx(-1.2 + 1.4 - 0.3)


def test_faster_response_always_helps():
    for s in gt.SOURCE_EFFECT:
        assert gt.conversion_prob(s, 1, 3) > gt.conversion_prob(s, 48, 3)
        assert gt.true_uplift(s, 48, 3) > 0


def test_uplift_differs_from_likelihood():
    # A hot lead already answered fast has high likelihood but tiny uplift.
    hot_fast = gt.true_uplift("Referral", 1.5, 4)
    cold_slow = gt.true_uplift("Website Form", 72, 2)
    assert gt.conversion_prob("Referral", 1.5, 4) > gt.conversion_prob("Website Form", 72, 2)
    assert cold_slow > hot_fast


def test_sources_ordering():
    assert max(gt.SOURCE_EFFECT, key=gt.SOURCE_EFFECT.get) == "Referral"
    assert min(gt.SOURCE_EFFECT, key=gt.SOURCE_EFFECT.get) == "Social Media Ad"


def test_holdout_is_deterministic_and_sized():
    ids = [f"LD{i:07d}" for i in range(50)]
    h1, h2 = gt.assign_holdout(ids), gt.assign_holdout(list(reversed(ids)))
    assert h1 == h2
    assert len(h1) == 10
    assert h1 <= set(ids)


def test_holdout_small_and_empty():
    assert gt.assign_holdout([]) == set()
    assert len(gt.assign_holdout(["a", "b", "c", "d", "e"])) == 1


def test_describe_lists_all_terms():
    terms = gt.describe()
    assert len(terms) == 4 + len(gt.SOURCE_EFFECT)
