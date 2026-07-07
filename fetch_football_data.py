"""
Fetch FINISHED match results from football-data.org as a fallback to
fetch_understat.py (no xG here — just goals — which model.py handles fine,
it just skips the xG blend and fits on actual goals).

Needs the same FOOTBALL_DATA_API_KEY env var as fixtures_api.py.

Usage:
    python fetch_football_data.py PL --season 2025 --out matches.csv
"""

import argparse
import csv
import os
import sys

import requests

API_URL = "https://api.football-data.org/v4/competitions/{code}/matches"


def fetch(competition: str, season: str) -> list:
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        sys.exit(
            "Missing FOOTBALL_DATA_API_KEY. Get a free key at "
            "https://www.football-data.org/client/register and set it as "
            "an environment variable first."
        )

    resp = requests.get(
        API_URL.format(code=competition),
        headers={"X-Auth-Token": key},
        params={"status": "FINISHED", "season": season},
        timeout=30,
    )
    resp.raise_for_status()
    matches = resp.json().get("matches", [])

    rows = []
    for m in matches:
        ft = m["score"]["fullTime"]
        if ft["home"] is None or ft["away"] is None:
            continue
        rows.append(
            {
                "date": m["utcDate"][:10],
                "home": m["homeTeam"]["name"],
                "away": m["awayTeam"]["name"],
                "home_goals": ft["home"],
                "away_goals": ft["away"],
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("competition", help="Competition code, e.g. PL, PD, BL1, SA, FL1")
    ap.add_argument("--season", default="2025", help="Season start year, e.g. 2025 for 2025/26")
    ap.add_argument("--out", default="matches.csv")
    args = ap.parse_args()

    rows = fetch(args.competition, args.season)
    if not rows:
        sys.exit("No finished matches returned — check the competition code and season.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matches to {args.out}")


if __name__ == "__main__":
    main()
