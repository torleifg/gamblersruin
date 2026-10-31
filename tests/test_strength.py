import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from gamblersruin.strength import (
    Strength,
    StrengthConfig,
    StrengthCalculator,
    dixon_coles_matrix,
)

AS_OF = "2030-01-01"


def match(day, home, away, gh, ga, xh=None, xa=None, season="2024-2025"):
    return {
        "datetime": pd.Timestamp("2024-08-01") + pd.Timedelta(days=day),
        "season": season,
        "team_home": home,
        "team_away": away,
        "goals_home": gh,
        "goals_away": ga,
        "xg_home": gh if xh is None else xh,
        "xg_away": ga if xa is None else xa,
    }


def uniform_league(teams=("A", "B", "C", "D")):
    rows, day = [], 0
    for h in teams:
        for a in teams:
            if h != a:
                rows.append(match(day, h, a, 2, 1, 1.8, 1.1))
                day += 1
    return pd.DataFrame(rows)


def random_league(seed=0, teams=("A", "B", "C", "D", "E")):
    rng = np.random.default_rng(seed)
    rows, day = [], 0
    for h in teams:
        for a in teams:
            if h != a:
                rows.append(
                    match(day, h, a, rng.poisson(1.5), rng.poisson(1.1),
                          rng.gamma(4, 0.4), rng.gamma(4, 0.3))
                )
                day += 1
    return pd.DataFrame(rows)


def test_identical_teams_are_neutral():
    calc = StrengthCalculator(uniform_league())
    home, away = calc.strengths("A", "B", AS_OF)
    assert home == pytest.approx(Strength(1.0, 1.0))
    assert away == pytest.approx(Strength(1.0, 1.0))
    assert calc.expected_goals("A", "B", AS_OF) == pytest.approx((2.0, 1.0))


def test_future_and_other_season_matches_are_ignored():
    base = random_league()
    extra = pd.DataFrame(
        [
            match(1000, "A", "B", 9, 9, 9, 9),  # on/after as_of
            match(1001, "B", "A", 9, 9, 9, 9),
            match(-400, "A", "B", 7, 7, 7, 7, season="2023-2024"),
        ]
    )
    cutoff = base["datetime"].max() + pd.Timedelta(days=1)
    clean = StrengthCalculator(base).expected_goals("A", "B", cutoff, season="2024-2025")
    noisy = StrengthCalculator(pd.concat([base, extra])).expected_goals(
        "A", "B", cutoff, season="2024-2025"
    )
    assert noisy == pytest.approx(clean)
    # as_of itself is excluded (strictly before)
    on_cutoff = pd.DataFrame([match(0, "A", "B", 9, 9, 9, 9)])
    on_cutoff["datetime"] = cutoff
    again = StrengthCalculator(pd.concat([base, on_cutoff])).expected_goals("A", "B", cutoff)
    assert again == pytest.approx(clean)


def test_multiple_seasons_pool_matches():
    old = random_league(seed=1)
    old["season"] = "2023-2024"
    new = random_league(seed=2)
    new["datetime"] += pd.Timedelta(days=400)
    both = pd.concat([old, new])
    pooled = StrengthCalculator(both).expected_goals(
        "A", "B", AS_OF, season=["2023-2024", "2024-2025"]
    )
    relabelled = both.assign(season="2024-2025")
    assert pooled == pytest.approx(
        StrengthCalculator(relabelled).expected_goals("A", "B", AS_OF, season="2024-2025")
    )
    newest_only = StrengthCalculator(both).expected_goals("A", "B", AS_OF, season="2024-2025")
    assert pooled != pytest.approx(newest_only)


def test_matches_outside_the_recent_window_still_count():
    base = random_league(seed=3)
    changed = base.copy()
    # rewrite A's oldest matches; with a window of 4 they are far outside it, but still in the season sample
    old = changed[(changed.team_home == "A") | (changed.team_away == "A")].index[:3]
    changed.loc[old, ["goals_home", "goals_away", "xg_home", "xg_away"]] = 9.0
    cfg = StrengthConfig(recent_matches=4)
    assert StrengthCalculator(base, cfg).strengths("A", "B", AS_OF)[0] != pytest.approx(
        StrengthCalculator(changed, cfg).strengths("A", "B", AS_OF)[0]
    )


def test_season_matches_caps_the_season_sample():
    df = random_league(seed=3, teams=tuple("ABCDEFGH"))
    cfg = StrengthConfig(recent_matches=2, season_matches=6, goals_weight_attack=0.4)
    attack = StrengthCalculator(df, cfg).strengths("H", "B", AS_OF)[0].attack

    played = df[(df.team_home == "H") | (df.team_away == "H")]
    window = played.tail(cfg.recent_matches)
    capped = played.tail(cfg.season_matches)
    home = df[df.team_home == "H"]
    recent = home[home.index.isin(window.index)]
    season = home[home.index.isin(capped.index) & ~home.index.isin(window.index)]
    gw = cfg.goals_weight_attack
    b = lambda g, x: x + gw * (g - x)

    def ratio(sample):
        return b(sample.goals_home.mean(), sample.xg_home.mean()) / b(df.goals_home.mean(), df.xg_home.mean())

    r_season = 1 + len(season) / (len(season) + cfg.season_prior_matches) * (ratio(season) - 1)
    n = len(recent)
    expected = r_season + n / (n + cfg.attack_prior_matches) * (ratio(recent) - r_season)
    uncapped = StrengthCalculator(df, StrengthConfig(recent_matches=2, goals_weight_attack=0.4))
    assert len(season) == 4 and len(home) - n == 5  # the cap drops one home match
    assert attack == pytest.approx(expected)
    assert attack != pytest.approx(uncapped.strengths("H", "B", AS_OF)[0].attack)


def test_unknown_team_warns_and_is_neutral():
    calc = StrengthCalculator(uniform_league())
    with pytest.warns(UserWarning, match="Unknown team"):
        _, away = calc.strengths("A", "Z", AS_OF)
    assert away == Strength(1.0, 1.0)
    with pytest.warns(UserWarning, match="Unknown team"):
        calc.predict("Z", "B", AS_OF)


def test_no_prior_matches_raises():
    calc = StrengthCalculator(uniform_league())
    with pytest.raises(ValueError):
        calc.expected_goals("A", "B", "2024-08-01")
    with pytest.raises(ValueError):
        calc.expected_goals("A", "B", AS_OF, season="1999-2000")


def test_team_without_home_matches_is_neutral_and_lambda_finite():
    df = uniform_league()
    df = df[df["team_home"] != "A"]
    calc = StrengthCalculator(df)
    home, _ = calc.strengths("A", "B", AS_OF)
    assert home == Strength(1.0, 1.0)
    lam_h, lam_a = calc.expected_goals("A", "B", AS_OF)
    assert np.isfinite(lam_h) and np.isfinite(lam_a)
    assert lam_h > 0 and lam_a > 0
    # unknown team entirely is also safe (but warns)
    with pytest.warns(UserWarning, match="Unknown team"):
        assert calc.strengths("Nobody", "Ghost", AS_OF) == (Strength(1, 1), Strength(1, 1))


def test_xg_scale_cancels_when_goals_weight_is_zero():
    df = random_league(seed=3)
    scaled = df.assign(xg_home=df["xg_home"] * 0.8, xg_away=df["xg_away"] * 0.8)
    cfg = StrengthConfig(goals_weight_attack=0.0, goals_weight_defense=0.0)
    a = StrengthCalculator(df, cfg).expected_goals("A", "C", AS_OF)
    b = StrengthCalculator(scaled, cfg).expected_goals("A", "C", AS_OF)
    assert b == pytest.approx(a)


def test_more_matches_move_attack_monotonically_towards_raw_ratio():
    # A scores 2 at home, offset by B scoring 0 at home, so the league home mean stays 1.0.
    attacks = []
    for n in range(1, 9):
        rows = [match(i, "C", "D", 1, 1) for i in range(20)]
        rows += [match(100 + i, "A", "C", 2, 1) for i in range(n)]
        rows += [match(200 + i, "B", "C", 0, 1) for i in range(n)]
        calc = StrengthCalculator(pd.DataFrame(rows), StrengthConfig(recent_matches=100, season_matches=None))
        attacks.append(calc.strengths("A", "D", AS_OF)[0].attack)
    assert all(1.0 < x < 2.0 for x in attacks)
    assert attacks == sorted(attacks) and len(set(attacks)) == len(attacks)
    assert attacks[0] == pytest.approx(1 + 1 / (1 + 10))  # n/(n+k) with n=1, k=10


def promoted_league():
    """Last season A-D, every ratio 1.0; this season E, promoted, plays one league-average home match."""
    this = pd.DataFrame([match(400, "E", "A", 2, 1, 1.8, 1.1, season="2025-2026")])
    return pd.concat([uniform_league(), this], ignore_index=True), this.datetime.iloc[0]


def test_promoted_team_shrinks_towards_the_promoted_prior():
    df, first_match = promoted_league()
    cfg = StrengthConfig(promoted_attack=0.8, promoted_defense=1.3)
    calc = StrengthCalculator(df, cfg)
    seasons = ["2024-2025", "2025-2026"]

    home, away = calc.strengths("E", "A", first_match, season=seasons)
    assert home == pytest.approx(Strength(0.8, 1.3))
    assert away == pytest.approx(Strength(1.0, 1.0))  # A played last season

    # one league-average match moves E 1 / (1 + prior matches) of the way to 1.0
    home, _ = calc.strengths("E", "A", AS_OF, season=seasons)
    assert home.attack == pytest.approx(0.8 + (1.0 - 0.8) / (1 + cfg.attack_prior_matches))
    assert home.defense == pytest.approx(1.3 + (1.0 - 1.3) / (1 + cfg.defense_prior_matches))


def test_no_team_counts_as_promoted_without_earlier_season_matches():
    df, _ = promoted_league()
    calc = StrengthCalculator(df, StrengthConfig(promoted_attack=0.8, promoted_defense=1.3))
    for season in ("2025-2026", ["2023-2024", "2025-2026"]):  # one season, or an earlier one without matches
        assert calc.strengths("E", "A", AS_OF, season=season)[0] == pytest.approx(Strength(1.0, 1.0))


def test_recent_and_season_samples_are_disjoint():
    df = random_league(seed=5)
    cfg = StrengthConfig(recent_matches=6, goals_weight_attack=0.4)
    calc = StrengthCalculator(df, cfg)
    attack = calc.strengths("A", "B", AS_OF)[0].attack

    played = df[(df.team_home == "A") | (df.team_away == "A")]
    window = played.tail(cfg.recent_matches)
    home = df[df.team_home == "A"]
    recent = home[home.index.isin(window.index)]
    season = home[~home.index.isin(window.index)]
    w = cfg.goals_weight_attack

    def ratio(sample):
        b = lambda g, x: x + w * (g - x)
        return b(sample.goals_home.mean(), sample.xg_home.mean()) / b(
            df.goals_home.mean(), df.xg_home.mean()
        )

    r_season = 1 + len(season) / (len(season) + cfg.season_prior_matches) * (ratio(season) - 1)
    n = len(recent)
    expected = r_season + n / (n + cfg.attack_prior_matches) * (ratio(recent) - r_season)
    assert len(season) > 0 and n > 0
    assert attack == pytest.approx(expected)


def test_grid_cells_follow_dixon_coles_paper():
    lh, la, rho = 1.6, 0.9, -0.13
    grid = dixon_coles_matrix(lh, la, rho, 15)
    p = lambda k, lam: poisson.pmf(k, lam)
    assert grid[1, 0] == pytest.approx(p(1, lh) * p(0, la) * (1 + rho * la))
    assert grid[0, 1] == pytest.approx(p(0, lh) * p(1, la) * (1 + rho * lh))
    assert grid[0, 0] == pytest.approx(p(0, lh) * p(0, la) * (1 - rho * lh * la))
    assert grid[1, 1] == pytest.approx(p(1, lh) * p(1, la) * (1 - rho))


def test_predict_probabilities_and_rho_zero_is_plain_poisson():
    df = random_league(seed=1)
    calc = StrengthCalculator(df)
    pred = calc.predict("A", "B", AS_OF)
    assert sum(pred.home_draw_away) == pytest.approx(1.0)
    assert (pred.home_goal_expectation, pred.away_goal_expectation) == pytest.approx(
        calc.expected_goals("A", "B", AS_OF)
    )

    flat = StrengthCalculator(df, StrengthConfig(rho=0.0)).predict("A", "B", AS_OF)
    lh, la = flat.home_goal_expectation, flat.away_goal_expectation
    goals = np.arange(16)
    plain = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
    assert flat.grid == pytest.approx(plain / plain.sum())


def test_rho_out_of_bounds_raises():
    with pytest.raises(ValueError):
        dixon_coles_matrix(0.5, 0.5, -3.0, 10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"goals_weight_attack": -0.1},
        {"goals_weight_defense": 1.5},
        {"attack_prior_matches": 0},
        {"defense_prior_matches": -1},
        {"season_prior_matches": 0},
        {"promoted_attack": 0},
        {"promoted_defense": -1},
        {"recent_matches": 0},
        {"season_matches": 0},
        {"max_goals": 0},
        {"rho": float("nan")},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        StrengthConfig(**kwargs)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        StrengthCalculator(uniform_league().drop(columns=["xg_home"]))


def test_rates_match_strengths_and_expected_goals():
    calc = StrengthCalculator(random_league(seed=4))
    rates = calc.rates("A", "B", AS_OF)
    assert (rates.strength_home, rates.strength_away) == calc.strengths("A", "B", AS_OF)
    assert (rates.lambda_home, rates.lambda_away) == calc.expected_goals("A", "B", AS_OF)
    assert calc.grid(rates.lambda_home, rates.lambda_away).home_draw_away == pytest.approx(
        calc.predict("A", "B", AS_OF).home_draw_away
    )


def test_promoted_teams_are_those_missing_from_earlier_seasons():
    df, _ = promoted_league()
    calc = StrengthCalculator(df)
    seasons = sorted(df["season"].unique())
    assert calc.promoted_teams("A", "E", as_of=AS_OF, season=seasons) == ["E"]
    assert calc.promoted_teams("A", "E", as_of=AS_OF, season=seasons[-1]) == []
