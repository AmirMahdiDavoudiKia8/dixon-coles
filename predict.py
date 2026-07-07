"""
Predict a fixture and evaluate market odds.

    python predict.py matches.csv "Arsenal" "Chelsea" \
        --market-1x2 2.20,3.50,3.40 --market-over 2.15 \
        --bankroll 1000 --trust 0.5

--trust is how much you believe your model vs the market (0.5 = blend
equally with the margin-free market probabilities). Use 1.0 only once
your tracked CLV has earned it.
"""

import argparse

import pandas as pd

from model import (
    DixonColes,
    blend,
    expected_value,
    kelly_fraction,
    strip_margin,
)


def report_selection(name, model_p, market_odds, market_p, trust, bankroll):
    p = blend(model_p, market_p, trust)
    ev = expected_value(p, market_odds)
    stake = kelly_fraction(p, market_odds) * bankroll
    flag = "  <-- BET" if ev > 0.02 else ""
    print(
        f"  {name:<10} model {model_p:6.1%} | market {market_p:6.1%} "
        f"| blended {p:6.1%} | odds {market_odds:5.2f} "
        f"| EV {ev:+6.1%} | 1/4-Kelly stake {stake:8.2f}{flag}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="CSV of historical matches")
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--market-1x2", help="home,draw,away decimal odds")
    ap.add_argument("--market-over", type=float, help="over 2.5 decimal odds")
    ap.add_argument("--market-under", type=float, help="under 2.5 decimal odds")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--trust", type=float, default=0.5)
    ap.add_argument("--xg-weight", type=float, default=0.7)
    ap.add_argument("--half-life", type=float, default=365.0)
    args = ap.parse_args()

    matches = pd.read_csv(args.data)
    model = DixonColes(xg_weight=args.xg_weight, half_life_days=args.half_life)
    model.fit(matches)

    pred = model.predict(args.home, args.away)
    print(f"\n{pred.home} vs {pred.away}")
    print(f"Expected goals: {pred.lambda_home:.2f} - {pred.lambda_away:.2f}")
    print(f"Model rho (low-score correction): {model.params_['rho']:+.3f}")
    print(f"Most likely scores: {pred.top_scores(5)}")
    print(f"Fair odds: {pred.fair_odds()}\n")

    if args.market_1x2:
        oh, od, oa = (float(x) for x in args.market_1x2.split(","))
        overround = sum(1 / o for o in (oh, od, oa)) - 1
        mh, md, ma = strip_margin([oh, od, oa])
        print(f"1X2 market (overround {overround:.1%}):")
        report_selection("Home", pred.p_home, oh, mh, args.trust, args.bankroll)
        report_selection("Draw", pred.p_draw, od, md, args.trust, args.bankroll)
        report_selection("Away", pred.p_away, oa, ma, args.trust, args.bankroll)

    if args.market_over and args.market_under:
        mo, mu = strip_margin([args.market_over, args.market_under])
        print("Totals market:")
        report_selection("Over 2.5", pred.p_over(2.5), args.market_over, mo, args.trust, args.bankroll)
        report_selection("Under 2.5", pred.p_under(2.5), args.market_under, mu, args.trust, args.bankroll)
    elif args.market_over:
        # no under price given: assume symmetric 3% margin to estimate market prob
        mo = min(0.99, implied := 1 / args.market_over / 1.03)
        print("Totals market (margin estimated at 3%):")
        report_selection("Over 2.5", pred.p_over(2.5), args.market_over, mo, args.trust, args.bankroll)

    print("\nReminder: log every bet in tracker.py — CLV is the verdict, not results.")


if __name__ == "__main__":
    main()
