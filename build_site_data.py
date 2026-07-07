"""
Build docs/data.json: fit Dixon-Coles on historical results, predict every
upcoming fixture, and write everything the interactive site needs (score
matrix, fair odds, top scorelines). No market odds here — the site takes
those as user input and does the blend/EV/Kelly math client-side.

Usage:
    python build_site_data.py matches.csv fixtures.json --out docs/data.json
"""

import argparse
import json
import re

import pandas as pd

from model import DixonColes

GRID = 6  # display grid is 0..GRID-1 goals per side


def normalize(name: str) -> str:
    name = re.sub(r"\b(FC|CF|AFC|CD|SC)\b", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip().lower()


def match_team(name: str, known: list, index: dict) -> str | None:
    """
    Exact match after normalizing, else whole-string containment (handles
    Understat's short forms, e.g. "Brighton" inside "Brighton & Hove
    Albion", "Leeds" inside "Leeds United"). Deliberately NOT fuzzy/edit-
    distance matching: two clubs that both end in "United" (Leeds United,
    Newcastle United) can score a deceptively high similarity ratio and
    silently swap one team's fixtures for the other's.
    """
    norm = normalize(name)
    if norm in index:
        return index[norm]
    candidates = [(len(k), v) for k, v in index.items() if k in norm or norm in k]
    if not candidates:
        return None
    candidates.sort(reverse=True)  # longest/most specific match wins
    return candidates[0][1]


def build_matrix(pred) -> list:
    return [
        [round(float(pred.matrix[i, j]) * 100, 2) for j in range(GRID)]
        for i in range(GRID)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="CSV of historical matches (from fetch_understat.py)")
    ap.add_argument("fixtures", help="JSON of upcoming fixtures (from fixtures_api.py)")
    ap.add_argument("--out", default="docs/data.json")
    ap.add_argument("--xg-weight", type=float, default=0.7)
    ap.add_argument("--half-life", type=float, default=365.0)
    args = ap.parse_args()

    matches = pd.read_csv(args.data)
    model = DixonColes(xg_weight=args.xg_weight, half_life_days=args.half_life)
    model.fit(matches)

    known = model.teams_
    index = {normalize(t): t for t in known}

    with open(args.fixtures, encoding="utf-8") as f:
        fixtures = json.load(f)

    out_fixtures, skipped = [], []
    for fx in fixtures:
        home = match_team(fx["home"], known, index)
        away = match_team(fx["away"], known, index)
        if not home or not away:
            skipped.append(fx)
            continue

        pred = model.predict(home, away)
        out_fixtures.append(
            {
                "id": fx["id"],
                "date": fx["date"],
                "time": fx.get("time"),
                "competition": fx.get("competition"),
                "home": fx["home"],
                "away": fx["away"],
                "home_crest": fx.get("home_crest"),
                "away_crest": fx.get("away_crest"),
                "lambda_home": round(pred.lambda_home, 3),
                "lambda_away": round(pred.lambda_away, 3),
                "rho": round(model.params_["rho"], 3),
                "home_adv": round(model.params_["home_adv"], 3),
                "p_home": round(pred.p_home, 4),
                "p_draw": round(pred.p_draw, 4),
                "p_away": round(pred.p_away, 4),
                "p_over_2_5": round(pred.p_over(2.5), 4),
                "p_under_2_5": round(pred.p_under(2.5), 4),
                "fair_odds": pred.fair_odds(),
                "top_scores": pred.top_scores(5),
                "matrix": build_matrix(pred),
            }
        )

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "fixtures": out_fixtures,
        "skipped": [{"home": s["home"], "away": s["away"], "date": s["date"]} for s in skipped],
    }

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(out_fixtures)} predicted fixtures to {args.out}")
    if skipped:
        print(f"Skipped {len(skipped)} fixtures (team not found in historical data):")
        for s in skipped:
            print(f"  {s['date']}  {s['home']} vs {s['away']}")


if __name__ == "__main__":
    main()
