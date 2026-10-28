import itertools

import numpy as np
import pandas as pd
import pytest

from gamblersruin.data import ODDS_COLUMNS
from gamblersruin.evaluation import (
    backtest,
    bookmaker,
    evaluate_season,
    outcome_of,
    results_table,
    rps_each,
    score,
)
from gamblersruin.strength import StrengthConfig

TEAMS = ("A", "B", "C", "D")
UNIFORM = np.full((3, 3), 1 / 3)


def season_matches(season, start, goals, odds=(2.0, 3.5, 4.0)):
    """Double round robin; ``goals(i)`` gives the (home, away) score of the i-th match."""
    rows = []
    for i, (home, away) in enumerate(itertools.permutations(TEAMS, 2)):
        gh, ga = goals(i)
        rows.append({
            "datetime": pd.Timestamp(start) + pd.Timedelta(days=i),
            "season": season, "team_home": home, "team_away": away,
            "goals_home": gh, "goals_away": ga, "xg_home": gh + 0.2, "xg_away": ga + 0.1,
            **dict(zip(ODDS_COLUMNS, odds * (len(ODDS_COLUMNS) // 3))),
        })
    return pd.DataFrame(rows)


def two_seasons(prior_goals=lambda i: (i % 3, 1), goals=lambda i: (i % 3, (i + 1) % 2)):
    history = pd.concat([
        season_matches("2024-2025", "2024-08-01", prior_goals),
        season_matches("2025-2026", "2025-08-01", goals),
    ], ignore_index=True)
    return history


def test_outcome_of_codes_home_draw_away():
    df = pd.DataFrame({"goals_home": [2, 1, 0], "goals_away": [0, 1, 3]})
    assert outcome_of(df).tolist() == [0, 1, 2]


def test_rps_of_perfect_prediction_is_zero():
    probs = np.eye(3)
    assert rps_each(probs, np.array([0, 1, 2])).tolist() == [0, 0, 0]


def test_rps_of_worst_prediction_is_one():
    assert rps_each([[0, 0, 1]], np.array([0])).tolist() == [1.0]
    assert rps_each([[1, 0, 0]], np.array([2])).tolist() == [1.0]


def test_rps_of_uniform_prediction_by_outcome():
    # home/away wins are two ranks off the middle, a draw is one
    assert rps_each(UNIFORM, np.array([0, 1, 2])) == pytest.approx([5 / 18, 1 / 9, 5 / 18])


def test_rps_penalises_distant_misses_more_than_near_ones():
    near = rps_each([[0, 1, 0]], np.array([0]))[0]  # said draw, home won
    far = rps_each([[0, 0, 1]], np.array([0]))[0]  # said away win, home won
    assert near < far


def test_score_of_uniform_prediction():
    log_loss, rps = score(UNIFORM, np.array([0, 1, 2]))
    assert log_loss == pytest.approx(np.log(3))
    assert rps == pytest.approx((5 / 18 + 1 / 9 + 5 / 18) / 3)


def test_score_uses_probability_of_the_actual_outcome():
    log_loss, _ = score([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]], np.array([0, 2]))
    assert log_loss == pytest.approx(-(np.log(0.5) + np.log(0.7)) / 2)


def test_bookmaker_removes_margin_and_orders_home_draw_away():
    test = pd.DataFrame({"b365_h": [2.0, 1.5], "b365_d": [4.0, 4.0], "b365_a": [4.0, 8.0]})
    probs = bookmaker(test, "b365_")
    assert probs.shape == (2, 3)
    assert probs[0] == pytest.approx([0.5, 0.25, 0.25])  # fair odds are untouched
    assert probs.sum(axis=1) == pytest.approx([1, 1])
    assert probs[1].argmax() == 0  # shortest odds are the home win


def test_bookmaker_reads_closing_odds_columns():
    test = pd.DataFrame({"avg_ch": [2.0], "avg_cd": [4.0], "avg_ca": [4.0]})
    assert bookmaker(test, "avg_c")[0] == pytest.approx([0.5, 0.25, 0.25])


def test_backtest_probabilities_use_only_the_given_history():
    history = two_seasons()
    test = history[history.season == "2025-2026"].iloc[3:6]
    probs = backtest(history, StrengthConfig(recent_matches=5), test, ["2024-2025", "2025-2026"])
    assert probs.shape == (3, 3)
    assert probs.sum(axis=1) == pytest.approx([1, 1, 1])

    # a result that has not happened yet must not change an earlier prediction
    altered = history.copy()
    altered.loc[altered.index[-1], ["goals_home", "xg_home"]] = [9, 9.0]
    assert backtest(altered, StrengthConfig(recent_matches=5), test, ["2024-2025", "2025-2026"]) == \
           pytest.approx(probs)


def test_evaluate_season_baseline_is_prior_season_frequencies_only():
    # prior season: every match is a home win; the tested season never is
    history = two_seasons(prior_goals=lambda i: (2, 0), goals=lambda i: (0, 1))
    evaluation = evaluate_season(history, "2025-2026", "2024-2025", {})

    assert evaluation.outcome.tolist() == [2] * 12
    assert evaluation.predictions["League frequencies"] == pytest.approx(np.tile([1.0, 0, 0], (12, 1)))


def test_evaluate_season_skips_matches_without_odds_and_names_every_model():
    history = two_seasons()
    history.loc[history.index[15], "avg_d"] = np.nan  # a match of the tested season
    configs = {"latest 5": StrengthConfig(recent_matches=5), "latest 10": StrengthConfig(recent_matches=10)}

    evaluation = evaluate_season(history, "2025-2026", "2024-2025", configs)

    assert list(evaluation.predictions) == [
        "League frequencies", "Bookmaker (Bet365)", "Bookmaker (market average)",
        "Bookmaker (market average, closing)", "latest 5", "latest 10",
    ]
    assert len(evaluation.outcome) == 11
    assert all(len(p) == 11 for p in evaluation.predictions.values())
    assert all(np.allclose(np.asarray(p).sum(axis=1), 1) for p in evaluation.predictions.values())


def test_results_table_has_a_log_loss_and_rps_column_per_season():
    history = two_seasons()
    evaluation = evaluate_season(history, "2025-2026", "2024-2025", {"latest 5": StrengthConfig(recent_matches=5)})

    table = results_table({"2025-2026": evaluation})

    assert table.columns.tolist() == [("2025-2026 (12 matches)", "log-loss"), ("2025-2026 (12 matches)", "RPS")]
    assert table.index.tolist() == list(evaluation.predictions)
    log_loss, rps = score(evaluation.predictions["latest 5"], evaluation.outcome)
    assert table.loc["latest 5"].tolist() == pytest.approx([log_loss, rps])


def test_results_table_pools_the_seasons_when_there_are_several():
    history = two_seasons()
    configs = {"latest 5": StrengthConfig(recent_matches=5)}
    first = evaluate_season(history, "2025-2026", "2024-2025", configs)
    second = evaluate_season(history, "2025-2026", "2024-2025", {"latest 5": StrengthConfig(recent_matches=3)})

    table = results_table({"a": first, "b": second})

    assert [c[0] for c in table.columns[-2:]] == ["All seasons (24 matches)"] * 2
    pooled = score(
        np.concatenate([first.predictions["latest 5"], second.predictions["latest 5"]]),
        np.concatenate([first.outcome, second.outcome]),
    )
    assert table.loc["latest 5"].iloc[-2:].tolist() == pytest.approx(pooled)
