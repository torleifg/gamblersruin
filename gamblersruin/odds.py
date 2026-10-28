from collections.abc import Collection

import pandas as pd
import requests

ODDS_URL = "https://api.norsk-tipping.no/OddsenGameInfo/v1/api/events/FBL"

# Norsk Tipping name -> Understat name, only where they differ
NT_TO_UNDERSTAT = {
    "Brighton and Hove Albion": "Brighton",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Tottenham Hotspur": "Tottenham",
    "Leeds United": "Leeds",

    "Bayern München": "Bayern Munich",
    "Borussia Mönchengladbach": "Borussia M.Gladbach",
    "FC Köln": "FC Cologne",
    "FSV Mainz": "Mainz 05",
    "RB Leipzig": "RasenBallsport Leipzig",
    "SC Paderborn": "Paderborn",
    "SV 07 Elversberg": "Elversberg",
    "TSG Hoffenheim": "Hoffenheim",

    "Athletic Club Bilbao": "Athletic Club",

    "AS Monaco": "Monaco",
    "Angers SCO": "Angers",
    "FC Lorient": "Lorient",
    "Lille OSC": "Lille",
    "OGC Nice": "Nice",
    "Olympique Lyon": "Lyon",
    "Olympique Marseille": "Marseille",
    "Paris Saint-Germain": "Paris Saint Germain",
    "Racing Club De Lens": "Lens",
    "Stade Brest 29": "Brest",
    "Strasbourg Alsace": "Strasbourg",

    "AC Monza": "Monza",
    "Atalanta BC": "Atalanta",
    "Como 1907": "Como",
    "Frosinone Calcio": "Frosinone",
    "Inter Milan": "Inter",
    "Milan": "AC Milan",
    "Parma Calcio": "Parma Calcio 1913",
    "Sassuolo Calcio": "Sassuolo",
    "Torino FC": "Torino",
    "Venezia FC": "Venezia",
}

# Norsk Tipping competition id -> Understat league name
LEAGUES = {
    "35887": "ENG Premier League",
    "27571": "DEU Bundesliga 1",
    "27431": "ESP La Liga",
    "27724": "FRA Ligue 1",
    "27914": "ITA Serie A",
}

COLUMNS = [
    "kickoff", "competition_id", "competition", "home", "away", "odds_home", "odds_draw", "odds_away", "event_id",
]


def to_understat(name: str) -> str:
    return NT_TO_UNDERSTAT.get(name, name)


def kickoff_uk_naive(kickoff: pd.Timestamp) -> pd.Timestamp:
    # UK local time without a timezone, to compare with the naive datetimes of the match data
    return kickoff.tz_convert("Europe/London").tz_localize(None)


def parse_events(payload: dict, leagues: Collection[str], day=None, tz="Europe/Oslo") -> pd.DataFrame:
    """Football events in ``leagues`` kicking off on ``day`` (default today) in ``tz``, by kickoff time."""
    day = pd.Timestamp(day).date() if day is not None else pd.Timestamp.now(tz).date()

    rows = []
    for event in payload["eventList"]:
        kickoff = pd.Timestamp(event["startTime"])
        if (
                event.get("sportId") != "FBL"
                or "homeParticipant" not in event  # specials are not matches between two teams
                or event["tournament"]["id"] not in leagues
                or kickoff.tz_convert(tz).date() != day
        ):
            continue
        market = event.get("mainMarket") or {}
        odds = {}  # stays empty (NaN odds) when the main market is not 1X2
        if market.get("marketName") == "HUB":
            odds = {s["selectionValue"]: float(s["selectionOdds"]) for s in market.get("selections", [])}
        rows.append({
            "kickoff": kickoff.tz_convert(tz),
            "competition_id": event["tournament"]["id"],
            "competition": event["tournament"]["name"],
            "home": event["homeParticipant"],
            "away": event["awayParticipant"],
            "odds_home": odds.get("H", float("nan")),
            "odds_draw": odds.get("D", float("nan")),
            "odds_away": odds.get("A", float("nan")),
            "event_id": event["eventId"],
        })
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("kickoff", kind="stable").reset_index(drop=True)


def fetch_matches(leagues: Collection[str], day=None, tz="Europe/Oslo") -> pd.DataFrame:
    response = requests.get(ODDS_URL, timeout=20)
    response.raise_for_status()
    return parse_events(response.json(), leagues, day, tz)
