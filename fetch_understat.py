"""
Fetch match results + xG for a league season from Understat.

Run this LOCALLY (needs internet):
    pip install requests
    python fetch_understat.py EPL 2025
    python fetch_understat.py La_liga 2025 --out laliga.csv

Leagues: EPL, La_liga, Bundesliga, Serie_A, Ligue_1, RFPL
Output CSV columns: date, home, away, home_goals, away_goals, home_xg, away_xg
"""

import argparse

import requests

# Understat serves this via an AJAX endpoint now (not embedded in the page
# HTML like it used to be) — it 404s without the XHR header, since that's
# how the server tells an AJAX call apart from a direct page request.
URL = "https://understat.com/getLeagueData/{league}/{season}"


def fetch(league: str, season: str) -> list:
    resp = requests.get(
        URL.format(league=league, season=season),
        headers={
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"https://understat.com/league/{league}/{season}",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "dates" not in payload:
        raise RuntimeError("Unexpected response shape — Understat may have changed again.")
    return payload["dates"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("league")
    ap.add_argument("season")
    ap.add_argument("--out", default="matches.csv")
    args = ap.parse_args()

    rows = []
    for match in fetch(args.league, args.season):
        if not match.get("isResult"):
            continue  # skip fixtures not yet played
        rows.append(
            {
                "date": match["datetime"][:10],
                "home": match["h"]["title"],
                "away": match["a"]["title"],
                "home_goals": int(match["goals"]["h"]),
                "away_goals": int(match["goals"]["a"]),
                "home_xg": round(float(match["xG"]["h"]), 3),
                "away_xg": round(float(match["xG"]["a"]), 3),
            }
        )

    import csv

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matches to {args.out}")


if __name__ == "__main__":
    main()
