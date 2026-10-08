"""Planted ground-truth causal model for lead conversion (pure Python, no Spark).

The Xorbix generator produces independent noise. inject_signal.py rewrites `leads` so that
conversion follows this documented model; measure.py uses it to grade the agent.
"""
import math
import random

SEED = 42
OPEN_WINDOW_DAYS = 30          # leads created in the last N days are still open
TREATED_RESPONSE_HOURS = 1.0   # a "fast call" from the agent queue
HOLDOUT_FRACTION = 0.20

INTERCEPT = -1.2
RESPONSE_COEF = -0.05          # per hour of first-response delay
TOUCH_COEF = 0.35              # per touchpoint, up to TOUCH_CAP
TOUCH_CAP = 4
OVERTOUCH_COEF = -0.10         # per touchpoint beyond OVERTOUCH_FREE (pestering)
OVERTOUCH_FREE = 6

SOURCE_EFFECT = {
    "Referral": 1.0,
    "Walk-In": 0.6,
    "Phone Inquiry": 0.3,
    "Website Form": 0.0,
    "Insurance Directory": -0.1,
    "Community Event": -0.2,
    "Social Media Ad": -0.6,
}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(source, first_response_hours, num_touchpoints):
    return (INTERCEPT + SOURCE_EFFECT[source]
            + RESPONSE_COEF * first_response_hours
            + TOUCH_COEF * min(num_touchpoints, TOUCH_CAP)
            + OVERTOUCH_COEF * max(num_touchpoints - OVERTOUCH_FREE, 0))


def conversion_prob(source, first_response_hours, num_touchpoints):
    return sigmoid(logit(source, first_response_hours, num_touchpoints))


def true_uplift(source, first_response_hours, num_touchpoints):
    """Causal gain in conversion probability from responding in TREATED_RESPONSE_HOURS."""
    return (conversion_prob(source, TREATED_RESPONSE_HOURS, num_touchpoints)
            - conversion_prob(source, first_response_hours, num_touchpoints))


def assign_holdout(lead_ids, fraction=HOLDOUT_FRACTION, seed=SEED):
    """Deterministically pick round(fraction * n) lead ids to hold out (not called)."""
    ids = sorted(lead_ids)
    k = int(round(fraction * len(ids)))
    return set(random.Random(seed).sample(ids, k))


def describe():
    """Rows for the `ground_truth_model` transparency table."""
    rows = [("intercept", "", INTERCEPT),
            ("first_response_hours", "per hour", RESPONSE_COEF),
            ("min(num_touchpoints,4)", "per touchpoint", TOUCH_COEF),
            ("max(num_touchpoints-6,0)", "per extra touchpoint", OVERTOUCH_COEF)]
    rows += [("source", s, v) for s, v in SOURCE_EFFECT.items()]
    return rows
