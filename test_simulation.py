"""
Validation: simulate a season with KNOWN team strengths, fit the model,
and check it recovers the truth. If this fails, nothing else matters.
"""

import numpy as np
import pandas as pd

from model import DixonColes

rng = np.random.default_rng(42)

# --- ground truth ------------------------------------------------------
N = 20
teams = [f"Team{i:02d}" for i in range(N)]
true_att = rng.normal(0, 0.25, N); true_att -= true_att.mean()
true_def = rng.normal(0, 0.20, N); true_def -= true_def.mean()
TRUE_MU, TRUE_HOME_ADV = 0.05, 0.25

# --- simulate double round-robin ---------------------------------------
rows = []
day = pd.Timestamp("2025-08-01")
for h in range(N):
    for a in range(N):
        if h == a:
            continue
        lam = np.exp(TRUE_MU + TRUE_HOME_ADV + true_att[h] + true_def[a])
        nu = np.exp(TRUE_MU + true_att[a] + true_def[h])
        rows.append({
            "date": (day + pd.Timedelta(days=len(rows) // 10 * 3)).date(),
            "home": teams[h], "away": teams[a],
            "home_goals": rng.poisson(lam), "away_goals": rng.poisson(nu),
            # xG = true rate + measurement noise
            "home_xg": max(0.05, lam + rng.normal(0, 0.35)),
            "away_xg": max(0.05, nu + rng.normal(0, 0.35)),
        })
matches = pd.DataFrame(rows)
print(f"Simulated {len(matches)} matches, "
      f"avg goals/match {matches[['home_goals','away_goals']].sum(axis=1).mean():.2f}")

# --- fit and compare ----------------------------------------------------
model = DixonColes(xg_weight=0.7).fit(matches)
p = model.params_

fit_att = np.array([p["attack"][t] for t in teams])
fit_def = np.array([p["defense"][t] for t in teams])
print(f"\nAttack recovery:  corr = {np.corrcoef(true_att, fit_att)[0,1]:.3f}")
print(f"Defense recovery: corr = {np.corrcoef(true_def, fit_def)[0,1]:.3f}")
print(f"Home advantage:   true {TRUE_HOME_ADV:.3f}  fitted {p['home_adv']:.3f}")
print(f"rho fitted:       {p['rho']:+.3f} (simulated data is independent, so ~0 expected)")

# --- one fixture end to end ---------------------------------------------
best, worst = teams[int(np.argmax(true_att - true_def))], teams[int(np.argmin(true_att - true_def))]
pred = model.predict(best, worst)
print(f"\nStrongest vs weakest: {best} vs {worst}")
print(f"lambda: {pred.lambda_home:.2f} - {pred.lambda_away:.2f}")
print(f"1X2 probs: H {pred.p_home:.1%} / D {pred.p_draw:.1%} / A {pred.p_away:.1%}"
      f"  (sum {pred.p_home + pred.p_draw + pred.p_away:.4f})")
print(f"Fair odds: {pred.fair_odds()}")
print(f"Top scores: {pred.top_scores(3)}")

matches.to_csv("sim_matches.csv", index=False)
print("\nSaved sim_matches.csv for the predict.py demo.")
