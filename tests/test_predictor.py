import pandas as pd
import pytest

from gamblersruin.predictor import MatchPredictor, with_ev
from tests.test_strength import random_league

SEASONS = ["2024-2025", "2025-2026"]


@pytest.fixture
def predictor():
    last = random_league(seed=1, teams=("Arsenal", "Chelsea", "Everton", "Fulham"))
    this = random_league(seed=2, teams=("Arsenal", "Chelsea", "Everton", "Newport"))
    this["season"] = SEASONS[1]
    this["datetime"] += pd.Timedelta(days=365)
    return MatchPredictor(pd.concat([last, this], ignore_index=True), SEASONS)


def test_unknown_team_raises_with_suggestions(predictor):
    with pytest.raises(ValueError, match="Did you mean.*Arsenal"):
        predictor.predict("Arsenel", "Chelsea")


def test_probabilities_sum_to_one_and_text_mentions_both_teams(predictor):
    prediction = predictor.predict("Arsenal", "Chelsea")
    assert prediction.table["model"].sum() == pytest.approx(1.0, abs=2e-3)
    assert prediction.margin is None
    assert "Arsenal vs Chelsea" in prediction.format()
    assert len(prediction.top_scores) == 5


def test_odds_add_market_probabilities_and_margin(predictor):
    prediction = predictor.predict("Arsenal", "Chelsea", odds=(2.0, 3.5, 4.0))
    assert prediction.table["market (no margin)"].sum() == pytest.approx(1.0, abs=2e-3)
    assert prediction.margin == pytest.approx(1 / 2.0 + 1 / 3.5 + 1 / 4.0 - 1)
    assert "Bookmaker margin" in prediction.format()


def test_kickoff_limits_the_matches_used(predictor):
    early = predictor.predict("Arsenal", "Chelsea", kickoff="2025-06-01")
    assert early.kickoff == pd.Timestamp("2025-06-01")
    assert early.lambda_home != predictor.predict("Arsenal", "Chelsea").lambda_home


def test_promoted_warning_names_teams_missing_from_last_season(predictor):
    assert predictor.promoted_warnings("Arsenal", "Chelsea") == []
    (warning,) = predictor.promoted_warnings("Arsenal", "Newport")
    assert "Newport" in warning and "was not in the league in 2024-2025" in warning


def test_with_ev_adds_odds_and_expected_value_without_mutating(predictor):
    table = predictor.predict("Arsenal", "Chelsea", odds=(2.0, 3.5, 4.0)).table
    result = with_ev(table, (2.0, 3.5, 4.0))
    assert "odds" not in table.columns
    assert result["EV (model)"].tolist() == (table["model"] * [2.0, 3.5, 4.0] - 1).round(3).tolist()


def test_known_teams_come_from_the_calculator(predictor):
    assert predictor.known_teams == predictor.calc.known_teams
    assert predictor.calc.unknown_teams("Arsenal", "Arsenel") == ["Arsenel"]
