"""Expected success model, step 2: one row per draft pick, from the cache
fetch.py fills.

The outcome is the player's NHL regular-season games in the seven seasons
after his draft (the 2018 class: 2018-19 through 2024-25), so every class up
to 2019 is fully observed and none is right-censored. Every predictor is
known on draft day: his production in his draft season (D0) and the season
before (D-1), his age at the September 15 cutoff, height, weight, position,
shooting side and country. Draft position is kept only to judge the model
against the draft; it is never a predictor.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date

WINDOW = 7  # seasons after the draft
# Regular-season games each club played: the 2012-13 lockout, and the two
# seasons COVID cut short (2019-20 ended at 68 to 71 games, 2020-21 had 56).
SEASON_GAMES = {20122013: 48, 20192020: 71, 20202021: 56}


def window_games(year: int) -> int:
    """The most NHL regular-season games a player drafted in `year` could have
    played in his seven seasons."""
    return sum(SEASON_GAMES.get(season_code(year + k), 82) for k in range(1, WINDOW + 1))


# League groups, from the NHL's own league names (older seasons use older
# names: NCAA conferences, "U-18" for the US national U18 team, "High-MN" for
# Minnesota high school). First match wins.
GROUPS: list[tuple[str, str]] = [
    ("OHL", r"^OHL$"),
    ("WHL", r"^WHL$"),
    ("QMJHL", r"^QMJHL$"),
    ("USHL", r"^USHL$"),
    ("NTDP", r"^U-?1[78]$|NTDP|USDP|^USNDT|U\.S\. National"),
    (
        "NCAA",
        r"^NCAA$|^CCHA$|^WCHA$|^H-East$|^Hockey East$|^ECAC$|^NCHC$|^Big Ten$|^B1G$|^Atlantic|^AHA$|^CHA$",
    ),
    (
        "EuroPro",
        r"^Sweden$|^SHL$|^Finland$|^Liiga$|^SM-?liiga$|^Russia$|^Rus-KHL$|^KHL$|^CzRep$|^Czechia$|^Czech$"
        r"|^Swiss$|^NL$|^NLA$|^Germany$|^DEL$|^Slovakia$|^Slovak$|^Austria$|^ICEHL$|^EBEL$",
    ),
    (
        "EuroLow",
        r"^Sweden-[23]$|^Allsvenskan$|^Russia-?[23]$|^VHL$|^Mestis$|^German-[23]$|^Slovak-2$|^Swiss-B$"
        r"|^CzRep-2$|^Czech2$|^Finland-2$",
    ),
    (
        "EuroJr",
        r"-Jr\.?$|^J20|SuperElit|Nationell|U20|^Jr\. A|^MHL$|-U18$|^J18|^Swe-|^Fin-|^Rus-Jr|^Cze-|^U18 SM",
    ),
    (
        "JrA",
        r"^BCHL$|^AJHL$|^OPJHL$|^OJHL|^CCHL$|^SJHL$|^MJHL$|^CJHL$|^NAHL$|^EJHL$|^QJHL$|^AtJHL$|^KIJHL$|^MJAHL$"
        r"|^MHL-Can|^NOJHL$|^SIJHL$|^EHL$|^USPHL",
    ),
    ("HS", r"^High-|^USHS|Prep|^HS-"),
]
POSITIONS = ("C", "L", "R", "D", "G")


def group(league: str) -> str:
    for name, pattern in GROUPS:
        if re.search(pattern, league or ""):
            return name
    return "Other"


def season_code(year: int) -> int:
    """The season that ends in `year`: 2018 -> 20172018."""
    return (year - 1) * 10000 + year


def _season(rows: list[dict], code: int) -> dict | None:
    """A season's regular-season line in his main league (most games), with
    the games he played everywhere that season."""
    here = [
        r
        for r in rows
        if r.get("season") == code and r.get("gameTypeId") == 2 and (r.get("gamesPlayed") or 0) > 0
    ]
    if not here:
        return None
    main = max(here, key=lambda r: (r.get("gamesPlayed") or 0, r.get("points") or 0))
    same = [r for r in here if r["leagueAbbrev"] == main["leagueAbbrev"]]  # two clubs in one league
    gp = sum(r.get("gamesPlayed") or 0 for r in same)
    out = {
        "league": main["leagueAbbrev"],
        "group": group(main["leagueAbbrev"]),
        "gp": gp,
        "gp_all": sum(r.get("gamesPlayed") or 0 for r in here),
    }
    for k in ("goals", "assists", "points", "pim"):
        out[k] = sum(r.get(k) or 0 for r in same)
    if main.get("savePctg") is not None or main.get("goalsAgainstAvg") is not None:
        out["sv"] = main.get("savePctg")
        out["gaa"] = main.get("goalsAgainstAvg")
        out["wins"] = sum(r.get("wins") or 0 for r in same)
    return out


def _plain(s: str) -> str:
    import unicodedata

    return "".join(
        c
        for c in unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
        if c.isalpha()
    )


def css(cache: str, year: int) -> dict[tuple[str, str], tuple[int, int]]:
    """Central Scouting's final rank for each listed player that year, keyed
    by (last name, birth date): (category, rank). Category 1 North American
    skaters, 2 international skaters, 3 and 4 the goalies. A player missing
    from the final list keeps his midterm rank."""
    out = {}
    for cat in (1, 2, 3, 4):
        path = f"{cache}/css_{year}_{cat}.json"
        if not os.path.exists(path):
            continue
        for r in (json.load(open(path)) or {}).get("rankings") or []:
            rank = r.get("finalRank") or r.get("midtermRank")
            if rank and r.get("birthDate") and r.get("lastName"):
                out[(_plain(r["lastName"]), r["birthDate"])] = (cat, int(rank))
    return out


# International events at under-18 level (the U18 Worlds, the Hlinka Gretzky
# and its earlier names, the World U17 Challenge) and at under-20 (the World
# Juniors), as the NHL's records name them.
U18_EVENTS = re.compile(r"^WJ18|^WJC-18|Hlinka|^WHC-17|^U-17$|^WU17")
U20_EVENTS = re.compile(r"^WJC-A$|^WJC-20$|^WJC$")


def _extras(rows: list[dict], year: int) -> dict:
    """What else the NHL's records hold from before the draft: the draft
    season's playoffs in his main league, international events at U18 and
    U20 in his draft season or the one before, plus-minus and power-play
    goals where the league reports them, and the season two years back."""
    d0, d1, d2 = season_code(year), season_code(year - 1), season_code(year - 2)
    main = _season(rows, d0) or {}
    out: dict = {}
    po = [
        r
        for r in rows
        if r.get("season") == d0 and r.get("gameTypeId") == 3 and r.get("leagueAbbrev") == main.get("league")
    ]
    pgp = sum(r.get("gamesPlayed") or 0 for r in po)
    out["po_gp"] = pgp
    out["po_points"] = sum(r.get("points") or 0 for r in po)
    for name, pattern in (("u18", U18_EVENTS), ("u20", U20_EVENTS)):
        ev = [r for r in rows if r.get("season") in (d0, d1) and pattern.search(r.get("leagueAbbrev") or "")]
        out[f"{name}_gp"] = sum(r.get("gamesPlayed") or 0 for r in ev)
        out[f"{name}_points"] = sum(r.get("points") or 0 for r in ev)
    reg = [
        r
        for r in rows
        if r.get("season") == d0 and r.get("gameTypeId") == 2 and r.get("leagueAbbrev") == main.get("league")
    ]
    pm = [r.get("plusMinus") for r in reg if r.get("plusMinus") is not None]
    out["plus_minus"] = sum(pm) if pm else None
    ppg_goals = [r.get("powerPlayGoals") for r in reg if r.get("powerPlayGoals") is not None]
    shg = [r.get("shorthandedGoals") or 0 for r in reg if r.get("powerPlayGoals") is not None]
    out["pp_goals"] = sum(ppg_goals) if ppg_goals else None
    out["sh_goals"] = sum(shg) if ppg_goals else None
    out["d2"] = _season(rows, d2)
    return out


def rows(cache: str, first: int, last: int) -> list[dict]:
    matched = json.load(open(f"{cache}/matched.json"))
    out = []
    for year in range(first, last + 1):
        scouted = css(cache, year)
        picks = json.load(open(f"{cache}/picks_{year}.json")).get("picks") or []
        ids = matched.get(str(year), {})
        for p in picks:
            if not p.get("lastName"):
                continue
            pid = ids.get(str(p["overallPick"]))
            base = {
                "year": year,
                "overall": p["overallPick"],
                "round": p["round"],
                "team": p.get("teamAbbrev"),
                "name": f"{(p.get('firstName') or {}).get('default', '')} {p['lastName']['default']}".strip(),
                "player_id": pid,
            }
            path = f"{cache}/landing/{pid}.json"
            if not pid or not os.path.exists(path):
                out.append({**base, "matched": False})
                continue
            land = json.load(open(path))
            totals = land.get("seasonTotals") or []
            nhl = [
                r
                for r in totals
                if r.get("leagueAbbrev") == "NHL"
                and r.get("gameTypeId") == 2
                and season_code(year + 1) <= r.get("season", 0) <= season_code(year + WINDOW)
            ]
            last_seen = max((r.get("season", 0) for r in totals), default=0)
            born = land.get("birthDate")
            age = None
            if born:
                b = date.fromisoformat(born)
                age = (date(year, 9, 15) - b).days / 365.25
            pos = land.get("position") or p.get("positionCode")
            out.append(
                {
                    **base,
                    "matched": True,
                    "position": pos if pos in POSITIONS else None,
                    "shoots": land.get("shootsCatches"),
                    "country": land.get("birthCountry") or p.get("countryCode"),
                    "height": land.get("heightInInches") or p.get("height"),
                    "weight": land.get("weightInPounds") or p.get("weight"),
                    "age": age,
                    # Central Scouting (None: not listed, or no lists that year)
                    "css": scouted.get((_plain(land.get("lastName", {}).get("default", "")), born))
                    if scouted
                    else None,
                    "css_year": bool(scouted),
                    "d0": _season(totals, season_code(year)),
                    "d1": _season(totals, season_code(year - 1)),
                    "extra": _extras(totals, year),
                    "nhl_gp": sum(r.get("gamesPlayed") or 0 for r in nhl),
                    "window_games": window_games(year),
                    # seasons seen so far after the draft, for classes still inside the window
                    "seasons_seen": max(0, min(WINDOW, (last_seen // 10000) - year)),
                }
            )
    return out
