# portfolio_logic.py
import numpy as np


def normalize_weights(amounts_dict, available_assets=None):
    """
    Takes a dict {ticker: amount} and returns dict {ticker: weight}.
    Optionally filters to `available_assets` (list of tickers).
    """
    if available_assets is not None:
        filtered = {t: a for t, a in amounts_dict.items() if t in available_assets}
    else:
        filtered = dict(amounts_dict)

    total = sum(float(a) for a in filtered.values() if a is not None)

    if total <= 0:
        # Default equally-weighted if nothing valid
        n = len(filtered)
        if n == 0:
            return {}
        return {t: 1.0 / n for t in filtered.keys()}

    return {t: float(a) / total for t, a in filtered.items()}


def investment_is_attractive(mu_port, vol_port, mu_i, vol_i, corr_pi, rf):
    """
    Implements the condition:
        rf + Sharpe_p * sigma_i * corr_pi < mu_i
    where Sharpe_p = (mu_port - rf) / vol_port.
    Returns (bool, info_dict).
    """
    if vol_port <= 0:
        # Sharpe undefined; be conservative.
        return False, {"sharpe_port": float("nan"), "threshold": float("nan")}

    sharpe_p = (mu_port - rf) / vol_port
    threshold = rf + sharpe_p * vol_i * corr_pi

    return mu_i > threshold, {
        "sharpe_port": sharpe_p,
        "threshold": threshold
    }
