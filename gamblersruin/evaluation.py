from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
import penaltyblog as pb

from gamblersruin.data import missing_odds
from gamblersruin.strength import StrengthCalculator, StrengthConfig


class Evaluation(NamedTuple):
    predictions: dict[str, np.ndarray]  # model name -> (matches, 3) probabilities of H, D, A
    outcome: np.ndarray  # 0 = home win, 1 = draw, 2 = away win


def outcome_of(df: pd.DataFrame) -> np.ndarray:
    return np.select([df.goals_home > df.goals_away, df.goals_home == df.goals_away], [0, 1], 2)  # H, D, A


def rps_each(probs, outcome) -> np.ndarray:
    """Ranked probability score of every match."""
    probs = np.asarray(probs)
    cum_gap = np.cumsum(probs, axis=1) - np.cumsum(np.eye(3)[outcome], axis=1)
    # the last cumulative gap is always 0, so only the first two count; 0.5 = 1 / (outcomes - 1)
    return 0.5 * (cum_gap[:, :2] ** 2).sum(axis=1)


def score(probs, outcome) -> tuple[float, float]:
    """(log-loss, mean ranked probability score); lower is better."""
    probs = np.asarray(probs)
    log_loss = -np.log(probs[np.arange(len(probs)), outcome]).mean()
    return log_loss, rps_each(probs, outcome).mean()


def backtest(history: pd.DataFrame, config: StrengthConfig, test: pd.DataFrame, seasons: Sequence[str]) -> np.ndarray:
    """Model probabilities for each ``test`` match, from the ``seasons`` matches before it."""
    calc = StrengthCalculator(history, config)
    return np.array([
        calc.predict(r.team_home, r.team_away, r.datetime, season=seasons).home_draw_away
        for r in test.itertuples()
    ])


def bookmaker(test: pd.DataFrame, prefix: str) -> np.ndarray:
    """1X2 probabilities with the bookmaker margin removed, from odds columns ``{prefix}h``, ``{prefix}d``, ``{prefix}a``."""
    return np.array([
        pb.implied.calculate_implied([h, d, a]).probabilities
        for h, d, a in zip(test[f"{prefix}h"], test[f"{prefix}d"], test[f"{prefix}a"])
    ])


def evaluate_season(
        history: pd.DataFrame, season: str, prior_season: str, configs: Mapping[str, StrengthConfig]
) -> Evaluation:
    """Predictions for the ``season`` matches that have odds, by the baselines and each config.

    Models only see ``prior_season`` and ``season`` matches played before each fixture, and
    the league-frequency baseline uses ``prior_season`` alone. Bookmaker odds are football-data's
    pre-closing odds (Friday afternoon for weekend matches, Tuesday for midweek) unless marked closing.
    """
    # every model is scored on the same matches, so those without odds are dropped for all of them
    test = history[(history.season == season) & ~missing_odds(history)]
    outcome = outcome_of(test)
    prior = history[history.season == prior_season]
    base_rate = np.bincount(outcome_of(prior), minlength=3) / len(prior)
    predictions = {
        "League frequencies": np.tile(base_rate, (len(test), 1)),
        "Bookmaker (Bet365)": bookmaker(test, "b365_"),
        "Bookmaker (market average)": bookmaker(test, "avg_"),
        "Bookmaker (market average, closing)": bookmaker(test, "avg_c"),
    }
    for name, config in configs.items():
        predictions[name] = backtest(history, config, test, [prior_season, season])
    return Evaluation(predictions, outcome)


def results_table(evaluated: Mapping[str, Evaluation]) -> pd.DataFrame:
    """Log-loss and RPS of every model, a column pair per season and, from two seasons, one pooled."""
    if len(evaluated) > 1:
        # pooled over matches, not averaged over seasons, so a season weighs by its match count
        names = next(iter(evaluated.values())).predictions
        evaluated = {
            **evaluated,
            "All seasons": Evaluation(
                {name: np.concatenate([e.predictions[name] for e in evaluated.values()]) for name in names},
                np.concatenate([e.outcome for e in evaluated.values()]),
            ),
        }
    results = {}
    for season, (predictions, outcome) in evaluated.items():
        scores = {name: score(p, outcome) for name, p in predictions.items()}
        label = f"{season} ({len(outcome)} matches)"
        results[(label, "log-loss")] = {name: s[0] for name, s in scores.items()}
        results[(label, "RPS")] = {name: s[1] for name, s in scores.items()}
    return pd.DataFrame(results)
