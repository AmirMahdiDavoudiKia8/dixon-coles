"""
Fetch upcoming (scheduled) fixtures with dates from football-data.org.

Needs a free API key: sign up at https://www.football-data.org/client/register
then set it as an environment variable before running:

    Windows (PowerShell):  $env:FOOTBALL_DATA_API_KEY = "your-key"
    Windows (cmd):         set FOOTBALL_DATA_API_KEY=your-key

Usage:
    python fixtures_api.py PL --days 14 --out fixtures.json

Competition codes: PL (Premier League), PD (La Liga), BL1 (Bundesliga),
SA (Serie A), FL1 (Ligue 1).
"""

import argparse
import json
import os
import sys
from datetime import date, timedelta

import requests

API_URL = "https://api.football-data.org/v4/competitions/{code}/matches"


def fetch(competition: str, days: int) -> list:
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        sys.exit(
            "Missing FOOTBALL_DATA_API_KEY. Get a free key at "
            "https://www.football-data.org/client/register and set it as "
            "an environment variable first."
        )

    date_from = date.today().isoformat()
    date_to = (date.today() + timedelta(days=days)).isoformat()

    resp = requests.get(
        API_URL.format(code=competition),
        headers={"X-Auth-Token": key},
        params={"status": "SCHEDULED", "dateFrom": date_from, "dateTo": date_to},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for m in payload.get("matches", []):
        rows.append(
            {
                "id": m["id"],
                "date": m["utcDate"][:10],
                "time": m["utcDate"][11:16],
                "competition": payload["competition"]["name"],
                "matchday": m.get("matchday"),
                "home": m["homeTeam"]["name"],
                "away": m["awayTeam"]["name"],
                "home_crest": m["homeTeam"].get("crest"),
                "away_crest": m["awayTeam"].get("crest"),
            }
        )
    rows.sort(key=lambda r: (r["date"], r["time"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("competition", help="Competition code, e.g. PL, PD, BL1, SA, FL1")
    ap.add_argument("--days", type=int, default=14, help="Look-ahead window in days")
    ap.add_argument("--out", default="fixtures.json")
    args = ap.parse_args()

    rows = fetch(args.competition, args.days)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} upcoming fixtures to {args.out}")


if __name__ == "__main__":
    main()
