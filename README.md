# Gambler's ruin

An experiment in building a football prediction model and testing it against the betting market. The model rates each
team's attack and defense, home and away, against the league average from a blend of goals and expected
goals (xG). Ratings from a team's last 50 matches, over this season and last, are pulled towards a prior (league
average, or below it for promoted sides), and ratings from its last 5 towards those. The two teams' ratings give each
side's expected goals, and a Poisson score grid with the Dixon-Coles correction turns them into home, draw and away
probabilities. It uses only public data: Understat for results and xG, football-data for odds.

The model is judged on its probabilities, not on how often its pick wins or on the profit from a run of bets. Accuracy
only checks the most likely outcome, which the odds already name, and a few hundred bets can show a profit by luck.
Each match is scored with the ranked probability score (RPS) and log-loss, and the model is compared with the market
average's odds, with the bookmakers' margin removed, both before the round (pre-closing) and at kick-off (closing).
Every prediction is made only from matches played before it, and each gap to the market comes with a paired 95%
interval, so a difference within noise is reported as one.

Over five Premier League seasons, 2022-23 to 2026-27 (1,600 matches, the last season up to 25 October 2026), the model
scores
0.2018 RPS, against 0.1968 for the market before closing, 0.1951 at closing and 0.2321 for league frequencies (lower is
better). 2021-22 is only a warm-up: the season before it was played without crowds, with more away wins than home wins,
and the model would carry that into it. The model covers about 86% of the distance from that naive baseline to the
market, with ten parameters (whose defaults were tuned on these same seasons). Most of what is left looks like
information the model never sees: squad quality, transfers, injuries and lineups. The gap is widest when the favourite
is strong, and making the probabilities sharper does not close it.

Changes to the model and its parameters have not closed the gap either. Five changes were tried and none was adopted:
down-weighting older matches with an exponential half-life (short half-lives are worse), taking the recent window from
the same venue only, adjusting ratings for the strength of opponents, dropping the recent window, and pooling home and
away matches into one attack and one defense per team. No `StrengthConfig` setting reliably beats the default either:
the attack and defense priors should not be too small (under about 10 matches) or the season prior too large (over about
10), goal weights should stay away from 1, and `recent_matches` and `rho` hardly matter. The promoted-side priors (0.75
attack, 1.2 defense) are the first-season averages of the 33 promoted sides since 2015-16, and they beat having no
promoted prior by 0.0021 ± 0.0013 RPS.

All of the above is for the Premier League. The model also supports the Bundesliga, La Liga, Serie A and Ligue 1, but
its defaults were tuned and its promoted-side priors set on Premier League seasons alone, so these leagues may not give
the same result.

## Getting Started

### 1. Install Python (Mac)

Install pyenv:

```bash
brew install pyenv
```

Install Python:

```bash
pyenv install 3.15.2
```

Switch to Python version:

```bash
pyenv local 3.15.2
```

Verify Python and version:

```bash
which python
python --version
```

### 2. Set Up the Virtual Environment

In the root folder of the project, create a virtual environment for managing dependencies:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Usage

The model lives in the `gamblersruin` package and is run from the notebooks in the root folder.

- `backtest.ipynb` scores the model against the market over five Premier League seasons and produces the numbers above.
  It reads matches and odds from a snapshot in `data/`, named after the league and seasons, so the numbers do not move
  as the season goes on or Understat revises its xG. When there is no snapshot it fetches the data and saves one; delete
  the file to take a new snapshot. Snapshots are kept out of git, so a fresh clone takes its own, with the matches played
  by then. To backtest another league, set `LEAGUE` to its name in `LEAGUES` (`gamblersruin/odds.py`) and add any team names that
  differ between Understat and football-data to `TEAM_NAMES` (`gamblersruin/data.py`); the notebook prints how many
  matches it leaves out for missing odds.
- `predict.ipynb` predicts the matches of a given day in all five leagues (set `DAY` in the first cell) and compares
  them with the bookmaker odds (Norsk Tipping).

### Tests

```bash
pytest
```
