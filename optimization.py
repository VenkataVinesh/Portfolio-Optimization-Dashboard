"""Markowitz mean-variance optimisation, long-only, solved numerically with SciPy SLSQP.

    python optimization.py      # prints max-Sharpe weights, writes efficient_frontier.png

The returns are SYNTHETIC (seeded), so the numbers illustrate the method, not a market.
"""
from __future__ import annotations

import numpy as np
import scipy.optimize as sco

RISK_FREE = 0.02


def synthetic_market(num_assets: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Annualised expected returns and a covariance matrix that is positive-definite by construction.

    Covariance = B B^T + D (one-factor model plus idiosyncratic variance), which is PD for any
    loadings B and strictly positive D. The previous version rescaled a random correlation
    matrix and reset its diagonal, which can produce a matrix that is not PSD.
    """
    rng = np.random.default_rng(seed)
    mean_returns = rng.uniform(0.05, 0.20, num_assets)
    loadings = rng.uniform(0.05, 0.20, (num_assets, 1))
    idio = rng.uniform(0.10, 0.25, num_assets) ** 2
    cov = loadings @ loadings.T + np.diag(idio)
    return mean_returns, cov


def performance(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = RISK_FREE) -> float:
    ret, vol = performance(w, mu, cov)
    return (ret - rf) / vol


def _solve(objective, n: int, extra_constraints=()) -> np.ndarray:
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *extra_constraints]
    res = sco.minimize(
        objective, np.full(n, 1.0 / n), method="SLSQP",
        bounds=[(0.0, 1.0)] * n, constraints=constraints, options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = RISK_FREE) -> np.ndarray:
    return _solve(lambda w: -sharpe(w, mu, cov, rf), len(mu))


def min_variance(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return _solve(lambda w: w @ cov @ w, len(mu))


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, points: int = 40) -> np.ndarray:
    """Long-only efficient frontier: min variance for each target return.

    Returns an array of shape (points, 2) with columns (volatility, return), from the
    minimum-variance portfolio up to the highest-return asset.
    """
    lo = performance(min_variance(mu, cov), mu, cov)[0]
    targets = np.linspace(lo, mu.max(), points)
    out = []
    for t in targets:
        w = _solve(lambda w: w @ cov @ w, len(mu), ({"type": "eq", "fun": lambda w, t=t: w @ mu - t},))
        out.append(performance(w, mu, cov)[::-1])
    return np.array(out)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = ["A", "B", "C", "D", "E"]
    mu, cov = synthetic_market(len(assets))
    w = max_sharpe(mu, cov)
    ret, vol = performance(w, mu, cov)
    print("Max-Sharpe portfolio (synthetic data, long-only):")
    for a, x in zip(assets, w):
        print(f"  {a}: {x:6.2%}")
    print(f"  expected return {ret:.2%}  volatility {vol:.2%}  Sharpe {sharpe(w, mu, cov):.3f}")

    frontier = efficient_frontier(mu, cov)
    rng = np.random.default_rng(0)
    cloud = rng.dirichlet(np.ones(len(mu)), 2000)
    cloud_perf = np.array([performance(x, mu, cov) for x in cloud])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(cloud_perf[:, 1], cloud_perf[:, 0], s=4, c="#bbbbbb", label="2,000 random long-only portfolios")
    ax.plot(frontier[:, 0], frontier[:, 1], color="#2b6cb0", lw=2, label="efficient frontier (SLSQP)")
    ax.scatter(np.sqrt(np.diag(cov)), mu, color="#444", marker="x", label="single assets")
    ax.scatter(vol, ret, color="#c53030", marker="*", s=220, zorder=5, label=f"max Sharpe ({sharpe(w, mu, cov):.2f})")
    xs = np.linspace(0, frontier[:, 0].max(), 2)
    ax.plot(xs, RISK_FREE + sharpe(w, mu, cov) * xs, color="#c53030", lw=0.8, ls="--", label="capital market line")
    ax.set_xlabel("volatility (annualised)")
    ax.set_ylabel("expected return (annualised)")
    ax.set_title("Markowitz efficient frontier, long-only (synthetic returns)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("efficient_frontier.png", dpi=120)
    print("saved efficient_frontier.png")


if __name__ == "__main__":
    main()
