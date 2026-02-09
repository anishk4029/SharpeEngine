# metrics.py
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(price_df: pd.DataFrame):
    """
    Given a price DataFrame of Adjusted Close prices, compute:
    - daily log returns
    - annualized expected returns (mu)
    - annualized covariance matrix
    - correlation matrix
    - annualized volatilities
    """
    returns = np.log(price_df / price_df.shift(1)).dropna()

    mu_annual = returns.mean() * TRADING_DAYS
    cov_daily = returns.cov()
    cov_annual = cov_daily * TRADING_DAYS
    corr = returns.corr()

    vol_annual = np.sqrt(np.diag(cov_annual))
    vol_series = pd.Series(vol_annual, index=price_df.columns)

    return {
        "returns": returns,
        "mu": mu_annual,
        "cov": cov_annual,
        "corr": corr,
        "vol": vol_series
    }


def portfolio_stats(weights, mu_vec, cov_mat, rf=0.0):
    """
    weights: array-like, shape (n,)
    mu_vec: array-like, expected returns, shape (n,)
    cov_mat: array-like, covariance matrix, shape (n, n)
    rf: risk-free rate
    """
    w = np.asarray(weights)
    mu_vec = np.asarray(mu_vec)
    cov_mat = np.asarray(cov_mat)

    ret = w @ mu_vec
    var = w @ cov_mat @ w
    vol = np.sqrt(var) if var >= 0 else np.nan
    sharpe = (ret - rf) / vol if vol > 0 else np.nan

    return {"return": ret, "vol": vol, "var": var, "sharpe": sharpe}


def portfolio_investment_corr(returns_df: pd.DataFrame, weights, new_ticker: str) -> float:
    """
    Compute correlation between portfolio (defined by `weights` over columns of returns_df)
    and a specific asset `new_ticker` in the same DataFrame.
    """
    if new_ticker not in returns_df.columns:
        raise ValueError(f"{new_ticker} not in returns DataFrame columns.")

    w = np.asarray(weights)
    r_port = returns_df.dot(w)
    r_new = returns_df[new_ticker]

    return float(r_port.corr(r_new))
