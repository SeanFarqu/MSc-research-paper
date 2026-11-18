"""Expected success model, step 1: every draft pick 2005 onward with his
NHL.com record, busts included, from the NHL's public API (cached).

For each class: the full pick list (/v1/draft/picks/{year}/all), then each
pick matched to his NHL player id by the NHL's player search and confirmed by
the landing record's own draft year and overall pick, so nobody is matched by
name alone. The landing record carries every season he played in any league
(seasonTotals), which is where draft-year production and NHL games come from.

    python fetch.py CACHE 2005 2026

Writes CACHE/picks_<year>.json, CACHE/landing/<id>.json and CACHE/matched.json
({year: {overall: id}}); reruns read the cache.
"""

from __future__ import annotations

import json
import os
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import httpx

CACHE, FIRST, LAST = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
API = "https://api-web.nhle.com/v1"
SEARCH = "https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=40&q={q}"
os.makedirs(f"{CACHE}/landing", exist_ok=True)
os.makedirs(f"{CACHE}/search", exist_ok=True)
client = httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0"})


def get(url: str, path: str):
    if os.path.exists(path):
        return json.load(open(path))
    for attempt in range(4):
        try:
            r = client.get(url)
            if r.status_code == 404:
                body = None
                break
            r.raise_for_status()
            body = r.json()
            break
        except (httpx.HTTPError, ValueError):
            time.sleep(2 * (attempt + 1))
    else:
        return None
    with open(path + ".tmp", "w") as f:
        json.dump(body, f)
    os.replace(path + ".tmp", path)
    return body


def norm(s: str) -> str:
    plain = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return "".join(c for c in plain if c.isalpha())


def landing(pid: str):
    return get(f"{API}/player/{pid}/landing", f"{CACHE}/landing/{pid}.json")


def resolve(year: int, pick: dict) -> str | None:
    if not pick.get("lastName"):
        return None  # a forfeited or void pick has no player
    first, last = (pick.get("firstName") or {}).get("default", ""), pick["lastName"]["default"]
    tries = [f"{first} {last}".strip(), last, last.split("-")[-1], last.split(" ")[-1]]
    seen = set()
    for q in tries:
        if not q or q in seen:
            continue
        seen.add(q)
        hits = get(SEARCH.format(q=quote(q)), f"{CACHE}/search/{norm(q)}.json") or []
        for h in hits:
            name = norm(h.get("name", ""))
            if norm(last.split(" ")[-1]) not in name and norm(last) not in name:
                continue
            d = (landing(h["playerId"]) or {}).get("draftDetails") or {}
            if d.get("year") == year and d.get("overallPick") == pick["overallPick"]:
                return str(h["playerId"])
    return None


matched_path = f"{CACHE}/matched.json"
matched = json.load(open(matched_path)) if os.path.exists(matched_path) else {}
for year in range(FIRST, LAST + 1):
    picks = (get(f"{API}/draft/picks/{year}/all", f"{CACHE}/picks_{year}.json") or {}).get("picks") or []
    done = matched.get(str(year), {})
    todo = [p for p in picks if str(p["overallPick"]) not in done]
    with ThreadPoolExecutor(4) as pool:
        for p, pid in zip(todo, pool.map(resolve, [year] * len(todo), todo), strict=True):
            done[str(p["overallPick"])] = pid
    matched[str(year)] = done
    with open(matched_path, "w") as f:
        json.dump(matched, f)
    hit = sum(1 for v in done.values() if v)
    print(f"{year}: {len(picks)} picks, {hit} matched", flush=True)

# NHL Central Scouting's final lists (published before each draft, back to
# 2008): 1 North American skaters, 2 international skaters, 3 North American
# goalies, 4 international goalies.
for year in range(max(FIRST, 2008), LAST + 1):
    for cat in (1, 2, 3, 4):
        get(f"{API}/draft/rankings/{year}/{cat}", f"{CACHE}/css_{year}_{cat}.json")
print("central scouting lists cached", flush=True)
