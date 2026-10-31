import difflib
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import penaltyblog as pb

from gamblersruin.strength import Strength, StrengthCalculator, StrengthConfig


@dataclass(frozen=True)
class MatchPrediction:
    home: str
    away: str
    kickoff: pd.Timestamp
    strength_home: Strength
    strength_away: Strength
    lambda_home: float
    lambda_away: float
    top_scores: list[tuple[float, str]]  # (probability, "h-a"), most likely first
    table: pd.DataFrame  # model (and market) probabilities for home win, draw, away win
    margin: float | None  # bookmaker margin, when odds were given

    def format(self) -> str:
        lines = [
            f"{self.home} vs {self.away} (matches before {self.kickoff:%Y-%m-%d %H:%M})",
            f"Strength (attack / defense): {self.home} {self.strength_home.attack:.2f} / {self.strength_home.defense:.2f}, "
            f"{self.away} {self.strength_away.attack:.2f} / {self.strength_away.defense:.2f}",
            f"Expected goals: {self.home} {self.lambda_home:.2f} | {self.away} {self.lambda_away:.2f}",
        ]
        if self.margin is not None:
            lines.append(f"Bookmaker margin: {self.margin:.1%}")
        lines.append("Most likely scores: " + ", ".join(f"{s} ({p:.1%})" for p, s in self.top_scores))
        return "\n".join(lines)


class MatchPredictor:
    """Predicts unplayed matches from played ``matches`` in ``seasons``.

    ``seasons`` run from the previous season to the fixture's own, which is last; teams missing from
    the first one count as promoted.
    """

    def __init__(self, matches: pd.DataFrame, seasons: Sequence[str], config: StrengthConfig = StrengthConfig()):
        self.matches = matches
        self.seasons = list(seasons)
        self.calc = StrengthCalculator(matches, config)

    @property
    def known_teams(self) -> list[str]:
        return self.calc.known_teams

    def check_known(self, *teams: str) -> None:
        """Raise for unknown names; unlike the calculator, which only warns, a prediction needs history."""
        for team in self.calc.unknown_teams(*teams):
            close = difflib.get_close_matches(team, self.known_teams, n=3)
            raise ValueError(f"Unknown team {team!r}. Did you mean {close or self.known_teams}?")

    def promoted_warnings(self, *teams: str, kickoff=None) -> list[str]:
        """Warnings for teams that were not in the league last season."""
        promoted = self.calc.promoted_teams(*teams, as_of=self._cutoff(kickoff), season=self.seasons)
        warnings = []
        for team in promoted:  # strength rests on the promoted-side prior and this season's matches
            played = ((self.matches.team_home == team) | (self.matches.team_away == team)).sum()
            warnings.append(
                f"WARNING: {team} was not in the league in {self.seasons[0]}; "
                f"its strength is the promoted-side prior plus only {played} match(es) this season"
            )
        return warnings

    def _cutoff(self, kickoff) -> pd.Timestamp:
        """``kickoff``, or just after the last played match when None, since the calculator only uses matches strictly before it."""
        if kickoff is None:
            return self.matches.datetime.max() + pd.Timedelta(seconds=1)
        return pd.Timestamp(kickoff)

    def predict(self, home: str, away: str, kickoff=None, odds=None) -> MatchPrediction:
        """Probabilities for a match that has not been played yet.

        kickoff: None uses every played match; a date or datetime only uses matches before it.
        odds:    optional decimal 1X2 odds (home, draw, away); adds the margin-free market
                 probabilities.
        """
        self.check_known(home, away)
        kickoff = self._cutoff(kickoff)

        rates = self.calc.rates(home, away, kickoff, season=self.seasons)
        grid = self.calc.grid(rates.lambda_home, rates.lambda_away)

        table = pd.DataFrame({"model": grid.home_draw_away}, index=[f"{home} win", "Draw", f"{away} win"])
        margin = None
        if odds is not None:
            market = pb.implied.calculate_implied(list(odds))
            table["market (no margin)"] = market.probabilities
            margin = market.margin

        # only scorelines up to 5 goals a side are ranked
        top_scores = sorted(
            ((grid.exact_score(h, a), f"{h}-{a}") for h in range(6) for a in range(6)), reverse=True
        )[:5]
        return MatchPrediction(
            home, away, kickoff, rates.strength_home, rates.strength_away, rates.lambda_home, rates.lambda_away,
            top_scores, table.round(3), margin,
        )


def with_ev(table: pd.DataFrame, odds) -> pd.DataFrame:
    """Add the odds and the expected value per unit staked (probability * odds - 1)."""
    table = table.copy()
    table["odds"] = odds
    table["EV (model)"] = (table["model"] * table["odds"] - 1).round(3)
    return table
