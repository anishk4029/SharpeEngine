# optimizer.py
import numpy as np
from scipy.optimize import minimize


def _portfolio_var(w, cov):
    w = np.asarray(w)
    return float(w @ cov @ w)


def min_variance_for_target(mu, cov, target_return, allow_short=False):
    """
    Minimize variance for a given target return:
        min w' Σ w
        s.t. sum(w) = 1
             w' μ >= target_return
             w_i >= 0  (if allow_short=False)
    """
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = len(mu)

    def obj(w):
        return _portfolio_var(w, cov)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w @ mu - target_return}
    ]

    bounds = None if allow_short else [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    if not res.success:
        return None

    w_opt = res.x
    var_opt = _portfolio_var(w_opt, cov)
    ret_opt = float(w_opt @ mu)
    vol_opt = np.sqrt(var_opt)

    return w_opt, ret_opt, vol_opt


def efficient_frontier(mu, cov, n_points=50, allow_short=False):
    """
    Build an efficient frontier by sweeping target returns between min(mu) and max(mu).
    Returns list of dicts: {"target", "return", "vol", "weights"}.
    """
    mu = np.asarray(mu)
    mu_min, mu_max = float(mu.min()), float(mu.max())
    targets = np.linspace(mu_min, mu_max, n_points)

    frontier = []
    for R in targets:
        res = min_variance_for_target(mu, cov, R, allow_short=allow_short)
        if res is None:
            continue
        w, ret, vol = res
        frontier.append(
            {"target": float(R), "return": float(ret), "vol": float(vol), "weights": w}
        )

    return frontier


def max_sharpe(mu, cov, rf, allow_short=False):
    """
    Maximize Sharpe ratio:
        max (w' μ - rf) / sqrt(w' Σ w)
        s.t. sum(w) = 1
             w_i >= 0 (if allow_short=False)
    """
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = len(mu)

    def neg_sharpe(w):
        w = np.asarray(w)
        ret = float(w @ mu)
        var = _portfolio_var(w, cov)
        vol = np.sqrt(var)
        if vol <= 0:
            return 1e9
        return -(ret - rf) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = None if allow_short else [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    if not res.success:
        return None

    w_opt = res.x
    ret_opt = float(w_opt @ mu)
    var_opt = _portfolio_var(w_opt, cov)
    vol_opt = np.sqrt(var_opt)
    sharpe_opt = (ret_opt - rf) / vol_opt if vol_opt > 0 else np.nan

    return w_opt, ret_opt, vol_opt, sharpe_opt
