"""
Dixon-Coles football match model.

Pipeline (matches the theory exactly):
  1. Fit per-team attack/defense strengths + home advantage + rho (the
     low-score correction) by weighted maximum likelihood on historical
     matches, with exponential time decay so recent form matters more.
  2. For a fixture, produce lambda_home / lambda_away.
  3. Build the Poisson score matrix, apply the Dixon-Coles tau correction
     to the 0-0 / 1-0 / 0-1 / 1-1 cells, renormalise.
  4. Derive fair probabilities/odds for 1X2 and totals from the matrix.

Optionally blends xG with actual goals as the "observed" scoring signal
(xg_weight=0.7 means the likelihood target is 0.7*xG + 0.3*goals).
Dropping the k! term from the Poisson log-likelihood makes non-integer
targets valid; the tau term is always evaluated on actual goals.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10  # matrix dimension; P(>10 goals) is negligible


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

@dataclass
class FixturePrediction:
    home: str
    away: str
    lambda_home: float
    lambda_away: float
    matrix: np.ndarray  # P(home=i, away=j)

    # --- markets, all derived from the matrix ---
    @property
    def p_home(self) -> float:
        return float(np.tril(self.matrix, -1).sum())

    @property
    def p_draw(self) -> float:
        return float(np.trace(self.matrix))

    @property
    def p_away(self) -> float:
        return float(np.triu(self.matrix, 1).sum())

    def p_over(self, line: float = 2.5) -> float:
        i, j = np.indices(self.matrix.shape)
        return float(self.matrix[(i + j) > line].sum())

    def p_under(self, line: float = 2.5) -> float:
        return 1.0 - self.p_over(line)

    def p_btts(self) -> bool:
        return float(self.matrix[1:, 1:].sum())

    def fair_odds(self) -> dict:
        f = lambda p: round(1.0 / p, 3) if p > 0 else float("inf")
        return {
            "home": f(self.p_home),
            "draw": f(self.p_draw),
            "away": f(self.p_away),
            "over_2.5": f(self.p_over(2.5)),
            "under_2.5": f(self.p_under(2.5)),
        }

    def top_scores(self, n: int = 5) -> list:
        flat = [
            (i, j, self.matrix[i, j])
            for i in range(self.matrix.shape[0])
            for j in range(self.matrix.shape[1])
        ]
        flat.sort(key=lambda t: -t[2])
        return [(f"{i}-{j}", round(p, 4)) for i, j, p in flat[:n]]


class DixonColes:
    def __init__(self, xg_weight: float = 0.7, half_life_days: float = 365.0):
        """
        xg_weight: how much the likelihood trusts xG vs actual goals (0..1).
        half_life_days: a match this many days old carries half the weight
                        of a match played today.
        """
        self.xg_weight = xg_weight
        self.xi = np.log(2) / half_life_days
        self.teams_: list | None = None
        self.params_: dict | None = None

    # ---- internal helpers -------------------------------------------

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        """Dixon-Coles correction factor for low-scoring cells."""
        tau = np.ones_like(lam)
        tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau)
        tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
        tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
        tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
        return np.clip(tau, 1e-10, None)

    def _unpack(self, vec, n):
        # attack: n-1 free params, last team = -sum (identifiability)
        att = np.append(vec[: n - 1], -vec[: n - 1].sum())
        dfn = np.append(vec[n - 1 : 2 * n - 2], -vec[n - 1 : 2 * n - 2].sum())
        mu, home_adv, rho = vec[2 * n - 2 : 2 * n + 1]
        return att, dfn, mu, home_adv, rho

    # ---- fitting -----------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> "DixonColes":
        """
        matches columns: date, home, away, home_goals, away_goals,
                         and optionally home_xg, away_xg.
        """
        m = matches.copy()
        m["date"] = pd.to_datetime(m["date"])
        self.teams_ = sorted(set(m["home"]) | set(m["away"]))
        n = len(self.teams_)
        idx = {t: k for k, t in enumerate(self.teams_)}

        hi = m["home"].map(idx).to_numpy()
        ai = m["away"].map(idx).to_numpy()
        hg = m["home_goals"].to_numpy(float)
        ag = m["away_goals"].to_numpy(float)

        if "home_xg" in m.columns and m["home_xg"].notna().all():
            w = self.xg_weight
            h_target = w * m["home_xg"].to_numpy(float) + (1 - w) * hg
            a_target = w * m["away_xg"].to_numpy(float) + (1 - w) * ag
        else:
            h_target, a_target = hg, ag

        days_ago = (m["date"].max() - m["date"]).dt.days.to_numpy(float)
        weights = np.exp(-self.xi * days_ago)

        def nll(vec):
            att, dfn, mu, home_adv, rho = self._unpack(vec, n)
            lam = np.exp(mu + home_adv + att[hi] + dfn[ai])  # home rate
            nu = np.exp(mu + att[ai] + dfn[hi])              # away rate
            ll = (
                h_target * np.log(lam) - lam
                + a_target * np.log(nu) - nu
                + np.log(self._tau(hg, ag, lam, nu, rho))
            )
            return -(weights * ll).sum()

        x0 = np.zeros(2 * n + 1)
        x0[2 * n - 2] = 0.1  # mu
        x0[2 * n - 1] = 0.2  # home advantage
        bounds = [(-3, 3)] * (2 * n - 2) + [(-2, 2), (-1, 1), (-0.2, 0.2)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            raise RuntimeError(f"Fit failed: {res.message}")

        att, dfn, mu, home_adv, rho = self._unpack(res.x, n)
        self.params_ = {
            "attack": dict(zip(self.teams_, att)),
            "defense": dict(zip(self.teams_, dfn)),
            "mu": float(mu),
            "home_adv": float(home_adv),
            "rho": float(rho),
        }
        return self

    # ---- prediction --------------------------------------------------

    def rates(self, home: str, away: str) -> tuple:
        p = self.params_
        lam = np.exp(p["mu"] + p["home_adv"] + p["attack"][home] + p["defense"][away])
        nu = np.exp(p["mu"] + p["attack"][away] + p["defense"][home])
        return float(lam), float(nu)

    def predict(self, home: str, away: str) -> FixturePrediction:
        for t in (home, away):
            if t not in self.params_["attack"]:
                raise KeyError(f"Unknown team: {t!r}. Known: {self.teams_}")
        lam, nu = self.rates(home, away)
        goals = np.arange(MAX_GOALS + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, nu))

        rho = self.params_["rho"]
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            matrix[i, j] *= self._tau(
                np.array(i), np.array(j), np.array(lam), np.array(nu), rho
            )
        matrix /= matrix.sum()
        return FixturePrediction(home, away, lam, nu, matrix)

    def table(self) -> pd.DataFrame:
        p = self.params_
        return (
            pd.DataFrame(
                {"attack": p["attack"], "defense": p["defense"]}
            )
            .assign(rating=lambda d: d.attack - d.defense)
            .sort_values("rating", ascending=False)
            .round(3)
        )


# ----------------------------------------------------------------------
# Betting math
# ----------------------------------------------------------------------

def implied_prob(odds: float) -> float:
    return 1.0 / odds


def strip_margin(odds: list) -> list:
    """Remove the overround from a set of mutually exclusive odds."""
    probs = np.array([1.0 / o for o in odds])
    return list(probs / probs.sum())


def expected_value(p: float, odds: float) -> float:
    """EV per unit staked."""
    return p * (odds - 1) - (1 - p)


def kelly_fraction(p: float, odds: float, multiplier: float = 0.25) -> float:
    """Fractional Kelly stake as share of bankroll (0 if -EV)."""
    b = odds - 1
    f = (b * p - (1 - p)) / b
    return max(0.0, f * multiplier)


def blend(model_p: float, market_p: float, trust: float = 0.5) -> float:
    """Shrink model probability toward the margin-free market probability."""
    return trust * model_p + (1 - trust) * market_p
