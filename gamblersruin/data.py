import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import penaltyblog as pb

PLAYED_COLUMNS = ["goals_home", "goals_away", "xg_home", "xg_away"]


def fetch_history(league: str, seasons: Sequence[str]) -> pd.DataFrame:
    frames = [pb.scrapers.Understat(league, season).get_fixtures() for season in seasons]

    return (
        pd.concat(frames)
        .dropna(subset=PLAYED_COLUMNS)  # played matches only
        .sort_values("datetime", kind="stable")
    )


# Understat name -> football-data names, only where they differ; match ids are built from the date
# and team names, so a name left unmapped here loses that match's odds in attach_odds
TEAM_NAMES = {
    "Manchester City": ["Man City"],
    "Manchester United": ["Man United"],
    "Newcastle United": ["Newcastle"],
    "Nottingham Forest": ["Nott'm Forest"],
    "Wolverhampton Wanderers": ["Wolves"],
}
# Bet365 and market-average odds before kickoff (home, draw, away), then market-average closing odds
ODDS_COLUMNS = ["b365_h", "b365_d", "b365_a", "avg_h", "avg_d", "avg_a", "avg_ch", "avg_cd", "avg_ca"]


def fetch_odds(league: str, seasons: Sequence[str]) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.concat(
            pb.scrapers.FootballData(league, season, team_mappings=TEAM_NAMES).get_fixtures()
            for season in seasons
        )


def attach_odds(history: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Odds columns for every ``history`` match, NaN where no odds row has the same id."""
    if not (history.index.is_unique and odds.index.is_unique):
        raise ValueError("duplicate match ids would corrupt the join")
    return history.join(odds[ODDS_COLUMNS])


def missing_odds(matches: pd.DataFrame) -> pd.Series:
    return matches[ODDS_COLUMNS].isna().any(axis=1)


def snapshot(path: str | Path, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """The matches saved at ``path``, saving ``fetch()`` there first if the file does not exist.

    A snapshot keeps results from changing as the season goes on or Understat revises its xG;
    delete the file to take a new one.
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fetch().to_csv(path)
    # read back even right after saving, so the first run sees exactly what later runs will; ids
    # stay strings, dates dates, and floats bit for bit (the default parser can be off by an ulp)
    matches = pd.read_csv(
        path, index_col="id", parse_dates=["datetime"], dtype={"id": "str", "understat_id": "str"},
        float_precision="round_trip",
    )
    matches["date"] = pd.to_datetime(matches["date"]).dt.date
    return matches
