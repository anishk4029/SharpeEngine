# data.py
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def load_price_data(tickers, annual_returns):
    """
    Download daily prices for the given tickers for enough history to compute
    `annual_returns` year-to-year changes.
    Returns a DataFrame with Date index and one column per ticker (prices).
    Handles both single-ticker and multi-ticker yfinance formats.
    """
    years_back = int(annual_returns) + 1
    end = datetime.today()
    start = end - timedelta(days=int(365.25 * years_back))

    raw = yf.download(tickers, start=start, end=end, progress=False)

    if raw is None or raw.empty:
        return pd.DataFrame()

    # If we already have a simple single-level column DataFrame
    if not isinstance(raw.columns, pd.MultiIndex):
        # Try Adj Close or Close
        for col in ["Adj Close", "Close"]:
            if col in raw.columns:
                prices = raw[col].to_frame() if isinstance(raw[col], pd.Series) else raw[[col]]
                prices.columns = [tickers] if isinstance(tickers, str) else prices.columns
                return prices.dropna(how="all")
        # If neither exists, just return whatever we have
        return raw.dropna(how="all")

    # MultiIndex columns case (most common with multiple tickers)
    # Columns might look like:
    #   level 0: ticker, level 1: OHLCV  OR
    #   level 0: OHLCV,  level 1: ticker
    lvl0 = list(raw.columns.get_level_values(0))
    lvl1 = list(raw.columns.get_level_values(1))

    prices = None

    # Case 1: price field in level 0 (e.g., ('Adj Close','AAPL'))
    if "Adj Close" in lvl0:
        prices = raw.xs("Adj Close", axis=1, level=0)
    elif "Close" in lvl0:
        prices = raw.xs("Close", axis=1, level=0)

    # Case 2: price field in level 1 (e.g., ('AAPL','Adj Close'))
    elif "Adj Close" in lvl1:
        prices = raw.xs("Adj Close", axis=1, level=1)
    elif "Close" in lvl1:
        prices = raw.xs("Close", axis=1, level=1)

    # Fallback: if still nothing, just try to coerce something sensible
    if prices is None:
        # Try to pick any "Close"-like thing
        try:
            prices = raw.xs("Close", axis=1, level=0)
        except Exception:
            prices = raw.copy()

    # Ensure DataFrame and drop all-null columns
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all")

    # Make sure column names are tickers, not ('Adj Close', 'AAPL') etc.
    # After xs, columns should usually be tickers already.
    return prices
