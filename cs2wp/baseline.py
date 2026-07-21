"""baseline.py — the man-advantage rule (Phase 4).

The single most important comparison in the project. A model reported without a
baseline is uninterpretable.

Rule:  whichever side has more players alive wins; ties resolve to CT.

This typically lands in the mid-to-high 60s for accuracy — man advantage is the
dominant signal in Counter-Strike. The trained model must beat this by a
meaningful margin, and that delta is the project's headline metric.
"""

from __future__ import annotations


def predict_proba_t(players_alive_t, players_alive_ct):
    """Return a crude P(T win) for the man-advantage baseline.

    1.0 if T has more alive, 0.0 if CT has more, 0.0 on a tie (ties -> CT).
    These hard 0/1 outputs make the baseline's log-loss deliberately bad, which
    is part of the point: it has no calibration.
    """
    if players_alive_t > players_alive_ct:
        return 1.0
    if players_alive_ct > players_alive_t:
        return 0.0
    return 0.0  # tie -> CT wins


def predict_t_win(players_alive_t, players_alive_ct) -> int:
    """Hard 0/1 class prediction for the man-advantage baseline."""
    return int(players_alive_t > players_alive_ct)
