"""Tests for compute_meta_agent_update in calculate_ratings.

Verifies that the meta-agent approach produces symmetric team-level deltas
regardless of team size, and that variance-weighted redistribution works
correctly when players have different sigma values.
"""

import pytest

from calculate_ratings import compute_meta_agent_update, DEFAULT_MU, DEFAULT_SIGMA


def test_symmetric_teams_produce_symmetric_deltas():
    """Equal-sized teams with equal ratings should produce mirror-image updates.

    If both teams start at (25, 8.33) and team A wins, the mu increase for
    each team A player should exactly equal the mu decrease for each team B
    player. Sigmas should also update identically.
    """
    rating = (DEFAULT_MU, DEFAULT_SIGMA)
    team_a = [rating, rating, rating]
    team_b = [rating, rating, rating]

    updated_a, updated_b = compute_meta_agent_update(team_a, team_b, team_a_won=True)

    # All players on same team should get identical updates
    for mu, sigma in updated_a:
        assert mu == pytest.approx(updated_a[0][0])
        assert sigma == pytest.approx(updated_a[0][1])

    for mu, sigma in updated_b:
        assert mu == pytest.approx(updated_b[0][0])
        assert sigma == pytest.approx(updated_b[0][1])

    # Deltas should be symmetric: winner gain == loser loss
    a_delta = updated_a[0][0] - DEFAULT_MU
    b_delta = updated_b[0][0] - DEFAULT_MU

    assert a_delta > 0, "Winners should gain mu"
    assert b_delta < 0, "Losers should lose mu"
    assert a_delta == pytest.approx(-b_delta), (
        f"Deltas should be symmetric: winner gained {a_delta:.4f}, "
        f"loser lost {-b_delta:.4f}"
    )


def test_asymmetric_teams_produce_symmetric_team_deltas():
    """2 impostors vs 5 crewmates should still get symmetric TEAM-level deltas.

    This is the key property of the meta-agent approach: despite the 2v5
    asymmetry, each team's average mu change is the same magnitude (opposite
    sign). Without meta-agents, OpenSkill would give the smaller team ~27x
    larger individual deltas.
    """
    rating = (DEFAULT_MU, DEFAULT_SIGMA)
    impostors = [rating, rating]  # 2 impostors
    crewmates = [rating, rating, rating, rating, rating]  # 5 crewmates

    updated_imp, updated_crew = compute_meta_agent_update(
        impostors, crewmates, team_a_won=True
    )

    # Compute average team-level mu delta
    imp_avg_delta = sum(mu - DEFAULT_MU for mu, _ in updated_imp) / len(updated_imp)
    crew_avg_delta = sum(mu - DEFAULT_MU for mu, _ in updated_crew) / len(updated_crew)

    assert imp_avg_delta > 0, "Winning impostors should gain mu"
    assert crew_avg_delta < 0, "Losing crewmates should lose mu"
    assert imp_avg_delta == pytest.approx(-crew_avg_delta), (
        f"Team-level deltas should be symmetric: impostors avg {imp_avg_delta:.4f}, "
        f"crewmates avg {crew_avg_delta:.4f}"
    )

    # All impostors should get the same update (equal starting ratings)
    assert updated_imp[0][0] == pytest.approx(updated_imp[1][0])

    # All crewmates should get the same update (equal starting ratings)
    for mu, _ in updated_crew:
        assert mu == pytest.approx(updated_crew[0][0])


def test_variance_weighted_redistribution():
    """Players with higher sigma (more uncertainty) should receive larger mu updates.

    When one impostor is well-calibrated (low sigma) and the other is uncertain
    (high sigma), the uncertain player should absorb a larger share of the
    team's mu delta.
    """
    confident = (DEFAULT_MU, 2.0)   # Low sigma — well-known rating
    uncertain = (DEFAULT_MU, 8.0)   # High sigma — uncertain rating
    crewmates = [(DEFAULT_MU, DEFAULT_SIGMA)] * 5

    updated_imp, _ = compute_meta_agent_update(
        [confident, uncertain], crewmates, team_a_won=True
    )

    confident_delta = updated_imp[0][0] - DEFAULT_MU
    uncertain_delta = updated_imp[1][0] - DEFAULT_MU

    assert confident_delta > 0, "Even confident player should gain some mu"
    assert uncertain_delta > 0, "Uncertain player should gain mu"
    assert uncertain_delta > confident_delta, (
        f"Uncertain player (sigma=8) should get a larger update than "
        f"confident player (sigma=2): {uncertain_delta:.4f} vs {confident_delta:.4f}"
    )

    # The ratio of deltas should roughly follow the ratio of variances (64:4 = 16:1)
    delta_ratio = uncertain_delta / confident_delta
    variance_ratio = 8.0**2 / 2.0**2  # = 16
    assert delta_ratio == pytest.approx(variance_ratio, rel=0.01), (
        f"Delta ratio ({delta_ratio:.2f}) should match variance ratio "
        f"({variance_ratio:.2f})"
    )
