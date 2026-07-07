# Dixon-Coles football betting model

A complete pipeline: Understat xG data → Dixon-Coles model → score matrix →
fair odds → EV and quarter-Kelly staking → CLV tracking.

## Setup (local machine)

```bash
pip install numpy scipy pandas requests
```

## Workflow

**1. Get data** (needs internet — leagues: EPL, La_liga, Bundesliga, Serie_A, Ligue_1, RFPL):

```bash
python fetch_understat.py EPL 2025 --out matches.csv
```

**2. Predict a fixture and evaluate the market:**

```bash
python predict.py matches.csv "Arsenal" "Chelsea" \
    --market-1x2 2.20,3.50,3.40 --market-over 2.15 --market-under 1.75 \
    --bankroll 1000 --trust 0.5
```

`--trust 0.5` blends your model 50/50 with the margin-free market
probabilities. Do not raise it until your tracked CLV has earned it.

**3. Log every bet the moment you place it:**

```bash
python tracker.py add --match "Arsenal v Chelsea" --market 1X2 \
    --selection Home --odds 2.20 --stake 7 --model-prob 0.47
```

**4. Settle after kickoff with the closing odds:**

```bash
python tracker.py settle 1 --closing 2.05 --result L
python tracker.py report
```

## Validation

`python test_simulation.py` simulates a season with known team strengths
and verifies the model recovers them (expect ~0.95 correlation). Run it
after any change to `model.py`.

## The rules (non-negotiable)

1. **Paper-trade first.** Log predictions against real closing lines for
   2–3 months with zero money. Only stake if paper CLV is positive.
2. **CLV is the verdict, not results.** ~200 bets before judging anything.
   Positive average CLV → continue. Negative → stop and fix the model.
3. **Quarter-Kelly, always.** The stake sizes will feel absurdly small.
   That is what survival looks like.
4. **Big model-vs-market gaps mean the market knows something you don't**
   (injury, rotation, stale data). Investigate before betting, never after.
5. **Bankroll is entertainment money, fully separate, already written off.**

## Files

- `model.py` — Dixon-Coles fitting, score matrix, market pricing, Kelly/EV math
- `fetch_understat.py` — pulls match results + xG from Understat (run locally)
- `predict.py` — CLI: fixture → fair odds → EV vs market → stake
- `tracker.py` — bet log + CLV report (the feedback loop)
- `test_simulation.py` — parameter-recovery validation on synthetic data
