import datetime

import numpy as np
import pandas as pd
import pytest

import gamblersruin.data as data
from gamblersruin.data import ODDS_COLUMNS, TEAM_NAMES, attach_odds, fetch_history, fetch_odds, missing_odds, snapshot


def fixtures(ids, **columns):
    return pd.DataFrame(columns, index=pd.Index(ids, name="id"))


def odds_frame(ids, value=2.0):
    return fixtures(ids, **{c: [value] * len(ids) for c in ODDS_COLUMNS})


def test_attach_odds_joins_on_match_id_and_adds_only_odds_columns():
    history = fixtures(["a", "b", "c"], team_home=["X", "Y", "Z"])
    odds = odds_frame(["c", "a"]).assign(referee="ignored")
    odds.loc["a", "b365_h"] = 1.5

    result = attach_odds(history, odds)

    assert list(result.index) == ["a", "b", "c"]
    assert list(result.columns) == ["team_home", *ODDS_COLUMNS]
    assert result.loc["a", "b365_h"] == 1.5
    assert result.loc["b", ODDS_COLUMNS].isna().all()  # no odds for this match


@pytest.mark.parametrize("duplicated", ["history", "odds"])
def test_attach_odds_rejects_duplicate_match_ids(duplicated):
    history = fixtures(["a", "b"], team_home=["X", "Y"])
    odds = odds_frame(["a", "b"])
    if duplicated == "history":
        history = pd.concat([history, history.iloc[:1]])
    else:
        odds = pd.concat([odds, odds.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        attach_odds(history, odds)


def test_snapshot_fetches_once_and_reads_back_the_same_frame(tmp_path):
    matches = fixtures(
        ["1628812800---brentford---arsenal", "1628899200---burnley---brighton"],
        understat_id=["16376", "007"],  # would read back as numbers
        datetime=pd.to_datetime(["2021-08-13 19:00", "2021-08-14 14:00"]),
        date=[datetime.date(2021, 8, 13), datetime.date(2021, 8, 14)],
        team_home=["Brentford", "Burnley"],
        goals_home=[2, 1],
        xg_home=[1.88818, 0.1 + 0.2],  # must come back bit for bit
        avg_h=[4.02, np.nan],
    )
    calls = []

    def fetch():
        calls.append(1)
        return matches

    path = tmp_path / "data" / "snapshot.csv"
    first, second = snapshot(path, fetch), snapshot(path, fetch)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, matches, check_exact=True)
    pd.testing.assert_frame_equal(second, matches, check_exact=True)


def test_missing_odds_flags_any_missing_column():
    odds = odds_frame(["a", "b", "c"])
    odds.loc["b", "avg_d"] = np.nan
    odds.loc["c", ODDS_COLUMNS] = np.nan
    assert missing_odds(odds).tolist() == [False, True, True]


class FakeUnderstat:
    def __init__(self, league, season):
        self.season = season

    def get_fixtures(self):
        day = {"2024-2025": "2024-08-01", "2025-2026": "2025-08-01"}[self.season]
        return pd.DataFrame({
            "datetime": pd.to_datetime([day, day]) + pd.to_timedelta([2, 1], unit="D"),
            "goals_home": [1, np.nan], "goals_away": [0, np.nan],
            "xg_home": [1.1, np.nan], "xg_away": [0.4, np.nan],
        }, index=[f"{self.season}-played", f"{self.season}-unplayed"])


def test_fetch_history_keeps_played_matches_in_date_order(monkeypatch):
    monkeypatch.setattr(data.pb.scrapers, "Understat", FakeUnderstat)
    result = fetch_history("ENG Premier League", ["2025-2026", "2024-2025"])
    assert list(result.index) == ["2024-2025-played", "2025-2026-played"]


def test_fetch_odds_maps_team_names_and_stacks_seasons(monkeypatch):
    calls = []

    class FakeFootballData:
        def __init__(self, league, season, team_mappings=None):
            calls.append((league, season, team_mappings))
            self.season = season

        def get_fixtures(self):
            return odds_frame([f"{self.season}-1"])

    monkeypatch.setattr(data.pb.scrapers, "FootballData", FakeFootballData)
    result = fetch_odds("ENG Premier League", ["2024-2025", "2025-2026"])

    assert list(result.index) == ["2024-2025-1", "2025-2026-1"]
    assert calls == [("ENG Premier League", s, TEAM_NAMES) for s in ("2024-2025", "2025-2026")]
