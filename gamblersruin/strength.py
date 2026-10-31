import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from penaltyblog.models import FootballProbabilityGrid
from scipy.stats import poisson

REQUIRED_COLUMNS = (
    "datetime",
    "season",
    "team_home",
    "team_away",
    "goals_home",
    "goals_away",
    "xg_home",
    "xg_away",
)
_NUMERIC_COLUMNS = ("goals_home", "goals_away", "xg_home", "xg_away")


@dataclass(frozen=True)
class StrengthConfig:
    # weight of actual goals against xG in a team's rate; the rest is xG (see blend)
    goals_weight_attack: float = 0.4
    goals_weight_defense: float = 0.15
    # pseudo-matches pulling the recent sample back to the season ratio, so with n recent matches its
    # share is n / (n + prior); defense is noisier than attack, so it is pulled harder
    attack_prior_matches: float = 10.0
    defense_prior_matches: float = 20.0
    # the same for the season sample, pulled to the league-average (or promoted) starting point
    season_prior_matches: float = 3.0
    # where teams absent from the previous season start, instead of league average (1.0); 33 promoted
    # sides, 2015-16 to 2025-26, averaged about 0.75 / 1.2 in their first season
    promoted_attack: float = 0.75
    promoted_defense: float = 1.2
    recent_matches: int = 5  # a team's latest matches, home and away together
    season_matches: int | None = 50  # most matches a team's season sample may hold; None for all
    rho: float = -0.13  # Dixon-Coles correction for low scores; 0 is plain Poisson
    max_goals: int = 15  # the score grid runs 0..max_goals a side

    def __post_init__(self) -> None:
        for name in ("goals_weight_attack", "goals_weight_defense"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        for name in (
                "attack_prior_matches", "defense_prior_matches", "season_prior_matches",
                "promoted_attack", "promoted_defense",
        ):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.recent_matches < 1:
            raise ValueError(f"recent_matches must be >= 1, got {self.recent_matches}")
        if self.season_matches is not None and self.season_matches < 1:
            raise ValueError(f"season_matches must be >= 1, got {self.season_matches}")
        if self.max_goals < 1:
            raise ValueError(f"max_goals must be >= 1, got {self.max_goals}")
        if not math.isfinite(self.rho):
            raise ValueError(f"rho must be finite, got {self.rho}")


@dataclass(frozen=True)
class Strength:
    attack: float
    defense: float


@dataclass(frozen=True)
class MatchRates:
    strength_home: Strength
    strength_away: Strength
    lambda_home: float  # expected goals
    lambda_away: float


def blend(goals: float, xg: float, goals_weight: float) -> float:
    """``xg + w * (goals - xg)``: xG-anchored average, nudged towards actual goals."""
    return xg + goals_weight * (goals - xg)


def dixon_coles_matrix(
        home_lambda: float, away_lambda: float, rho: float, max_goals: int
) -> np.ndarray:
    """Poisson outer product with the Dixon & Coles (1997) tau correction.

    tau(0,0) = 1 - l*m*rho, tau(0,1) = 1 + l*rho, tau(1,0) = 1 + m*rho, tau(1,1) = 1 - rho
    with l = home mean, m = away mean (matrix is indexed [home goals, away goals]).
    """
    if home_lambda <= 0 or away_lambda <= 0:
        raise ValueError("Expected goals (lambdas) must be strictly positive.")
    # outside these bounds one of the tau factors goes negative, giving negative probabilities
    rho_min = max(-1.0 / home_lambda, -1.0 / away_lambda)
    rho_max = min(1.0, 1.0 / (home_lambda * away_lambda))
    if not rho_min <= rho <= rho_max:
        raise ValueError(
            f"Dixon-Coles rho={rho} is out of bounds for the given lambdas. "
            f"Must be between {rho_min:.4f} and {rho_max:.4f}."
        )
    goals = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goals, home_lambda), poisson.pmf(goals, away_lambda))
    grid[0, 0] *= 1.0 - rho * home_lambda * away_lambda
    grid[0, 1] *= 1.0 + rho * home_lambda
    grid[1, 0] *= 1.0 + rho * away_lambda
    grid[1, 1] *= 1.0 - rho
    return grid


class StrengthCalculator:
    def __init__(self, matches: pd.DataFrame, config: StrengthConfig = StrengthConfig()):
        missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
        if missing:
            raise ValueError(f"matches is missing required columns: {missing}")
        df = matches.loc[:, list(REQUIRED_COLUMNS)].copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ("datetime", "season", "team_home", "team_away", *_NUMERIC_COLUMNS):
            if df[col].isna().any():
                raise ValueError(f"matches has missing values in column '{col}'")
        for col in _NUMERIC_COLUMNS:
            df[col] = df[col].astype(float)
        self.config = config
        self.known_teams = sorted(set(df["team_home"]) | set(df["team_away"]))
        self._matches = df.sort_values("datetime", kind="stable").reset_index(drop=True)

    def rates(
            self, home: str, away: str, as_of, season: str | Sequence[str] | None = None
    ) -> MatchRates:
        """Both teams' strengths and the expected goals they imply, from matches before ``as_of``."""
        self._check_known(home, away)
        return self._rates(home, away, as_of, season)

    def strengths(
            self, home: str, away: str, as_of, season: str | Sequence[str] | None = None
    ) -> tuple[Strength, Strength]:
        """(home team's home strength, away team's away strength) using matches before ``as_of``."""
        self._check_known(home, away)
        rates = self._rates(home, away, as_of, season)
        return rates.strength_home, rates.strength_away

    def expected_goals(
            self, home: str, away: str, as_of, season: str | Sequence[str] | None = None
    ) -> tuple[float, float]:
        self._check_known(home, away)
        rates = self._rates(home, away, as_of, season)
        return rates.lambda_home, rates.lambda_away

    def predict(
            self, home: str, away: str, as_of, season: str | Sequence[str] | None = None
    ) -> FootballProbabilityGrid:
        self._check_known(home, away)
        rates = self._rates(home, away, as_of, season)
        return self.grid(rates.lambda_home, rates.lambda_away)

    def grid(self, lambda_home: float, lambda_away: float) -> FootballProbabilityGrid:
        """Score probabilities for the expected goals."""
        matrix = dixon_coles_matrix(lambda_home, lambda_away, self.config.rho, self.config.max_goals)
        return FootballProbabilityGrid(
            goal_matrix=matrix,
            home_goal_expectation=lambda_home,
            away_goal_expectation=lambda_away,
            normalize=True,  # the grid stops at max_goals and tau reweights cells, so it no longer sums to 1
        )

    def unknown_teams(self, *teams: str) -> list[str]:
        """The names absent from the whole dataset."""
        return [t for t in teams if t not in self.known_teams]

    def promoted_teams(self, *teams: str, as_of, season: str | Sequence[str] | None = None) -> list[str]:
        """The teams that count as promoted for a fixture: absent from the earlier season(s) in ``season``.

        This is the test ``_team_strength`` applies; none count when there are no earlier-season matches.
        """
        _, established = self._prior_matches(as_of, season)
        return [] if established is None else [t for t in teams if t not in established]

    def _check_known(self, *teams: str) -> None:
        """Warn about names absent from the whole dataset.

        They get prior strengths (a promoted side's when a previous season is given, else league
        average), which is right for a newly promoted side but silently wrong for a typo or a
        naming mismatch.
        """
        unknown = self.unknown_teams(*teams)
        if unknown:
            warnings.warn(
                f"Unknown team(s) {unknown}: not in matches, using prior strengths. "
                "Check the spelling if they should have history.",
                UserWarning,
                stacklevel=3,
            )

    def _rates(self, home: str, away: str, as_of, season: str | Sequence[str] | None) -> MatchRates:
        prior, established = self._prior_matches(as_of, season)
        strength_home = self._team_strength(prior, home, "home", established)
        strength_away = self._team_strength(prior, away, "away", established)
        # strengths are ratios to the league average, so scale by the league's mean goals per side
        return MatchRates(
            strength_home,
            strength_away,
            strength_home.attack * strength_away.defense * prior["goals_home"].mean(),
            strength_away.attack * strength_home.defense * prior["goals_away"].mean(),
        )

    def _prior_matches(
            self, as_of, season: str | Sequence[str] | None
    ) -> tuple[pd.DataFrame, set[str] | None]:
        """Matches strictly before ``as_of`` in the fixture's season(s), and the teams that played
        in the earlier ones.

        ``season`` is one season or several, the fixture's own last (e.g. last season and this);
        it defaults to the season of the latest match played before ``as_of``. The teams are None
        when there are no earlier-season matches, and then no team counts as promoted.
        """
        before = self._matches[self._matches["datetime"] < pd.Timestamp(as_of)]
        if before.empty:
            raise ValueError(f"No matches played before {as_of}; cannot estimate league averages")
        if season is None:
            season = before["season"].iloc[-1]
        seasons = [season] if isinstance(season, str) else list(season)
        prior = before[before["season"].isin(seasons)]
        if prior.empty:
            raise ValueError(f"No matches played in season {season} before {as_of} (round 1?)")
        earlier = prior[prior["season"] != seasons[-1]]
        established = set(earlier["team_home"]) | set(earlier["team_away"]) if len(earlier) else None
        return prior, established

    def _team_strength(
            self, prior: pd.DataFrame, team: str, venue: str, established: set[str] | None
    ) -> Strength:
        cfg = self.config
        promoted = established is not None and team not in established
        target = Strength(cfg.promoted_attack, cfg.promoted_defense) if promoted else Strength(1.0, 1.0)
        # the recent window and the season cap count matches at either venue, but the samples below
        # only use the matches at the venue in question; the recent and season samples never overlap
        played = prior[(prior["team_home"] == team) | (prior["team_away"] == team)]
        window = set(played.tail(cfg.recent_matches).index)
        capped = played if cfg.season_matches is None else played.tail(cfg.season_matches)

        if venue == "home":
            rows = prior[prior["team_home"] == team]
            scored, scored_xg = "goals_home", "xg_home"
            conceded, conceded_xg = "goals_away", "xg_away"
        else:
            rows = prior[prior["team_away"] == team]
            scored, scored_xg = "goals_away", "xg_away"
            conceded, conceded_xg = "goals_home", "xg_home"

        recent_mask = rows.index.isin(window)
        season_mask = ~recent_mask & rows.index.isin(capped.index)
        attack = self._shrunk_ratio(
            rows, recent_mask, season_mask, prior, scored, scored_xg,
            cfg.goals_weight_attack, cfg.attack_prior_matches, target.attack,
        )
        defense = self._shrunk_ratio(
            rows, recent_mask, season_mask, prior, conceded, conceded_xg,
            cfg.goals_weight_defense, cfg.defense_prior_matches, target.defense,
        )
        return Strength(attack, defense)

    def _shrunk_ratio(
            self,
            rows: pd.DataFrame,
            recent_mask: np.ndarray,
            season_mask: np.ndarray,
            league: pd.DataFrame,
            goals_col: str,
            xg_col: str,
            goals_weight: float,
            recent_prior: float,
            target: float,
    ) -> float:
        """Ratio to the league average: the season sample shrunk towards ``target``, the recent sample towards the result."""
        # the baseline is blended like the samples, so the ratio compares like with like
        baseline = blend(league[goals_col].mean(), league[xg_col].mean(), goals_weight)
        if baseline <= 0:
            raise ValueError(f"League baseline for '{goals_col}' is not positive; cannot form ratios")

        def raw_ratio(sample: pd.DataFrame) -> float:
            return blend(sample[goals_col].mean(), sample[xg_col].mean(), goals_weight) / baseline

        season_rows, recent_rows = rows[season_mask], rows[recent_mask]

        season_ratio = target
        if len(season_rows):
            n = len(season_rows)
            season_ratio += n / (n + self.config.season_prior_matches) * (raw_ratio(season_rows) - target)

        if not len(recent_rows):
            return season_ratio
        n = len(recent_rows)
        return season_ratio + n / (n + recent_prior) * (raw_ratio(recent_rows) - season_ratio)
