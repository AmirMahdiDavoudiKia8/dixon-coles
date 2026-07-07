"""
Bet tracker: log bets, then let closing line value judge you.

Log a bet (do this the moment you place it):
    python tracker.py add --match "Arsenal v Chelsea" --market 1X2 \
        --selection Home --odds 2.20 --stake 7 --model-prob 0.47

Settle it after kickoff (closing odds + result W/L/P):
    python tracker.py settle 1 --closing 2.05 --result L

Report:
    python tracker.py report
"""

import argparse
import os

import pandas as pd

FILE = os.path.join(os.path.dirname(__file__), "bets.csv")
COLS = [
    "id", "placed", "match", "market", "selection",
    "odds", "closing", "stake", "model_prob", "result",
]


def load() -> pd.DataFrame:
    if os.path.exists(FILE):
        return pd.read_csv(FILE)
    return pd.DataFrame(columns=COLS)


def save(df: pd.DataFrame):
    df.to_csv(FILE, index=False)


def add(args):
    df = load()
    row = {
        "id": (df["id"].max() + 1) if len(df) else 1,
        "placed": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "match": args.match,
        "market": args.market,
        "selection": args.selection,
        "odds": args.odds,
        "closing": None,
        "stake": args.stake,
        "model_prob": args.model_prob,
        "result": "pending",
    }
    save(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    print(f"Logged bet #{row['id']}: {args.selection} @ {args.odds} on {args.match}")


def settle(args):
    df = load()
    mask = df["id"] == args.bet_id
    if not mask.any():
        raise SystemExit(f"No bet with id {args.bet_id}")
    df.loc[mask, "closing"] = args.closing
    df.loc[mask, "result"] = args.result.upper()
    save(df)
    clv = args.closing and (df.loc[mask, "odds"].iloc[0] / args.closing - 1)
    print(f"Settled #{args.bet_id} ({args.result.upper()}), CLV {clv:+.1%}")


def report(_args):
    df = load()
    settled = df[df["result"].isin(["W", "L", "P"])].copy()
    if settled.empty:
        print("No settled bets yet.")
        return

    settled["clv"] = settled["odds"] / settled["closing"] - 1
    settled["pnl"] = settled.apply(
        lambda r: r.stake * (r.odds - 1) if r.result == "W"
        else (-r.stake if r.result == "L" else 0.0),
        axis=1,
    )
    n = len(settled)
    staked = settled["stake"].sum()
    pnl = settled["pnl"].sum()

    print(f"Settled bets: {n}")
    print(f"Total staked: {staked:.2f} | P/L: {pnl:+.2f} | ROI: {pnl/staked:+.1%}")
    print(f"Average CLV:  {settled['clv'].mean():+.2%}")
    print(f"CLV-positive: {(settled['clv'] > 0).mean():.0%} of bets")
    print()
    if n < 200:
        print(f"({n}/200 bets — too early to judge anything. Keep logging.)")
    elif settled["clv"].mean() > 0.01:
        print("Sustained positive CLV: the model is beating the market. Continue.")
    elif settled["clv"].mean() > -0.01:
        print("CLV around zero: no demonstrated edge yet. Do not increase stakes.")
    else:
        print("Negative CLV: the market is beating the model. Stop staking; fix the model.")
    print("\nBy market:")
    print(
        settled.groupby("market")
        .agg(n=("id", "count"), avg_clv=("clv", "mean"), pnl=("pnl", "sum"))
        .round(3)
        .to_string()
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    a = sub.add_parser("add")
    a.add_argument("--match", required=True)
    a.add_argument("--market", required=True)
    a.add_argument("--selection", required=True)
    a.add_argument("--odds", type=float, required=True)
    a.add_argument("--stake", type=float, required=True)
    a.add_argument("--model-prob", type=float, required=True)
    a.set_defaults(func=add)

    s = sub.add_parser("settle")
    s.add_argument("bet_id", type=int)
    s.add_argument("--closing", type=float, required=True)
    s.add_argument("--result", required=True, choices=["W", "L", "P", "w", "l", "p"])
    s.set_defaults(func=settle)

    r = sub.add_parser("report")
    r.set_defaults(func=report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
