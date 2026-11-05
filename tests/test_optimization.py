import numpy as np

from optimization import efficient_frontier, max_sharpe, min_variance, performance, sharpe, synthetic_market


def test_covariance_is_positive_definite_for_many_seeds():
    for seed in range(50):
        _, cov = synthetic_market(8, seed=seed)
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > 0


def test_max_sharpe_weights_are_long_only_and_sum_to_one():
    mu, cov = synthetic_market()
    w = max_sharpe(mu, cov)
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= -1e-9).all()


def test_max_sharpe_beats_every_frontier_point_and_random_portfolios():
    mu, cov = synthetic_market()
    best = sharpe(max_sharpe(mu, cov), mu, cov)
    for vol, ret in efficient_frontier(mu, cov, points=15):
        assert (ret - 0.02) / vol <= best + 1e-6
    rng = np.random.default_rng(1)
    for w in rng.dirichlet(np.ones(len(mu)), 500):
        assert sharpe(w, mu, cov) <= best + 1e-6


def test_frontier_dominates_random_portfolios():
    mu, cov = synthetic_market()
    frontier = efficient_frontier(mu, cov, points=30)
    rng = np.random.default_rng(2)
    for w in rng.dirichlet(np.ones(len(mu)), 300):
        ret, vol = performance(w, mu, cov)
        # The frontier volatility at this return level is never worse than the random portfolio's.
        if frontier[0, 1] <= ret <= frontier[-1, 1]:
            assert np.interp(ret, frontier[:, 1], frontier[:, 0]) <= vol + 1e-6


def test_min_variance_is_the_left_end_of_the_frontier():
    mu, cov = synthetic_market()
    _, vol = performance(min_variance(mu, cov), mu, cov)
    assert abs(efficient_frontier(mu, cov, points=5)[0, 0] - vol) < 1e-6
