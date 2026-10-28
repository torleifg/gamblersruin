import pandas as pd

from gamblersruin.odds import LEAGUES, kickoff_uk_naive, parse_events, to_understat

PL = "35887"


def event(home="Arsenal", away="Chelsea", start="2026-10-25T14:00:00Z", competition=PL, odds=None, **extra):
    market = None
    if odds is not None:
        market = {
            "marketName": "HUB",
            "selections": [
                {"selectionValue": key, "selectionOdds": str(value)} for key, value in zip("HDA", odds)
            ],
        }
    return {
        "sportId": "FBL",
        "eventId": f"{home}-{away}",
        "startTime": start,
        "homeParticipant": home,
        "awayParticipant": away,
        "tournament": {"id": competition, "name": "England - Premier League"},
        "mainMarket": market,
        **extra,
    }


def parse(*events, competition_ids=(PL,), day="2026-10-25"):
    return parse_events({"eventList": list(events)}, competition_ids, day)


def test_keeps_matching_events_with_odds_sorted_by_kickoff():
    result = parse(
        event("Spurs", "Everton", start="2026-10-25T16:30:00Z", odds=(2.0, 3.5, 3.8)),
        event(odds=(1.9, 3.6, 4.0)),
    )
    assert list(result.home) == ["Arsenal", "Spurs"]
    assert result.loc[0, ["odds_home", "odds_draw", "odds_away"]].tolist() == [1.9, 3.6, 4.0]


def test_skips_other_competitions_days_sports_and_specials():
    specials = event()
    del specials["homeParticipant"]
    result = parse(
        event(competition="27914"),
        event(start="2026-10-26T14:00:00Z"),
        event(sportId="HBL"),
        specials,
    )
    assert result.empty


def test_returns_competition_id_across_leagues():
    result = parse(event(), event("Inter", "Milan", competition="27914"), competition_ids=LEAGUES)
    assert sorted(result.competition_id) == ["27914", PL]


def test_missing_odds_become_nan():
    result = parse(event())
    assert result.loc[0, ["odds_home", "odds_draw", "odds_away"]].isna().all()


def test_day_is_judged_in_the_given_timezone():
    late = event(start="2026-10-25T23:30:00Z")  # 00:30 on the 26th in Oslo
    assert parse(late).empty
    assert len(parse(late, day="2026-10-26")) == 1


def test_kickoff_is_in_oslo_time_and_converts_to_naive_uk_time():
    kickoff = parse(event(start="2026-10-25T14:00:00Z")).loc[0, "kickoff"]
    assert kickoff.hour == 15  # CET after the clocks went back
    assert kickoff_uk_naive(kickoff) == pd.Timestamp("2026-10-25 14:00")
    assert kickoff_uk_naive(kickoff).tzinfo is None


def test_name_mapping_only_changes_known_differences():
    assert to_understat("Brighton and Hove Albion") == "Brighton"
    assert to_understat("Ipswich Town") == "Ipswich"
    assert to_understat("Tottenham Hotspur") == "Tottenham"
    assert to_understat("Liverpool") == "Liverpool"
