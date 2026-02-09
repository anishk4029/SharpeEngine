# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import importlib.util

from data import load_price_data
from metrics import compute_metrics, portfolio_stats, portfolio_investment_corr
from optimizer import efficient_frontier, max_sharpe
from portfolio_logic import normalize_weights, investment_is_attractive
from ocr import extract_shares_from_image
from scipy.stats import t as student_t  # for t-based confidence intervals
from scipy.stats import norm


# ------------------ HELPERS ------------------ #

def find_closest_frontier_point(frontier, current_ret, current_vol):
    """
    Given a list of frontier points and the current portfolio's
    (return, vol), find the frontier portfolio closest in (μ, σ) space.
    """
    best = None
    min_dist = float("inf")
    for p in frontier:
        d = (p["return"] - current_ret) ** 2 + (p["vol"] - current_vol) ** 2
        if d < min_dist:
            min_dist = d
            best = p
    return best


def portfolio_return_ci(returns_df, weights, mu_index, conf_level=0.95, annualization_factor=252):
    """
    Compute a t-based confidence interval for the *annualized* expected return
    of a portfolio with given weights.

    returns_df: DataFrame of daily returns for each asset.
    weights: 1D array-like of weights aligned with mu_index.
    mu_index: index (tickers) defining the asset order.
    conf_level: e.g. 0.90, 0.95, 0.975, 0.99
    """
    cols = list(mu_index)
    rets = returns_df[cols].values  # shape (T, N)
    w = np.asarray(weights, dtype=float)
    if rets.shape[1] != w.shape[0]:
        return None, None

    # daily portfolio returns
    r_p = rets @ w  # shape (T,)
    n = len(r_p)
    if n <= 1:
        return None, None

    mean_daily = np.mean(r_p)
    sd_daily = np.std(r_p, ddof=1)
    if sd_daily == 0:
        # no variability -> degenerate CI
        annual_mean = mean_daily * annualization_factor
        return annual_mean, annual_mean

    se_daily = sd_daily / np.sqrt(n)
    df = n - 1
    alpha = 1.0 - conf_level
    tcrit = student_t.ppf(1 - alpha / 2, df)

    lower_daily = mean_daily - tcrit * se_daily
    upper_daily = mean_daily + tcrit * se_daily

    # annualize
    lower_annual = lower_daily * annualization_factor
    upper_annual = upper_daily * annualization_factor
    return lower_annual, upper_annual


def portfolio_prob_negative_annual_return(returns_df, weights, mu_index):
    """
    Estimate probability that the portfolio's *annual* return is negative.
    Uses t-distribution when sample size n < 30; otherwise normal approximation.
    """
    cols = list(mu_index)
    rets = returns_df[cols].values  # shape (T, N)
    w = np.asarray(weights, dtype=float)
    if rets.shape[1] != w.shape[0]:
        return None, None, None, None

    # Daily portfolio log returns
    r_p = rets @ w  # shape (T,)
    if len(r_p) <= 1:
        return None, None, None, None

    # Convert daily log returns to annual log returns, then to simple returns
    r_p_series = pd.Series(r_p, index=returns_df.index)
    annual_log = r_p_series.groupby(r_p_series.index.year).sum()
    annual_returns = np.expm1(annual_log.values)  # simple annual returns

    n = len(annual_returns)
    if n <= 1:
        return None, None, None, None

    mean_annual = float(np.mean(annual_returns))
    sd_annual = float(np.std(annual_returns, ddof=1))
    if sd_annual == 0:
        prob = 1.0 if mean_annual < 0 else 0.0
        return prob, n, "degenerate", mean_annual

    x = (0.0 - mean_annual) / sd_annual

    if n < 30:
        prob = float(student_t.cdf(x, df=n - 1))
        dist = "t"
    else:
        prob = float(norm.cdf(x))
        dist = "normal"

    return prob, n, dist, mean_annual


def ocr_is_available():
    return importlib.util.find_spec("easyocr") is not None


# ------------------ PAGE CONFIG ------------------ #

st.set_page_config(
    page_title="SharpeEngine | Portfolio Optimizer",
    layout="wide"
)

st.title("🧠 SharpeEngine")

st.markdown(
    """
Welcome to **SharpeEngine** – a quantitative toolkit for building smarter portfolios.

SharpeEngine helps you:

- Use historical price data to estimate **expected returns**, **volatility**, **covariances**, and **correlations**
- Build the **efficient frontier** and identify a **max-Sharpe portfolio**
- Input your **current portfolio** across stocks and ETFs and see its risk/return profile
- Interactively visualize **expected return vs variance (risk)**
- Check whether adding a new investment satisfies your custom **Sharpe-based attractiveness condition**
- Upload a screenshot of your **Robinhood** holdings and auto-fill your portfolio
"""
)

# ------------------ SIDEBAR CONTROLS ------------------ #

st.sidebar.header("SharpeEngine Settings")

# Default tickers – user can change to anything
tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="NVDA, AMD, SMP, VOO, IVV"
)

annual_returns = st.sidebar.selectbox(
    "Annual return periods (year-to-year changes)",
    options=[1, 3, 5, 10, 15, 20, 30],
    index=2  # default 5 years
)

# Default RF = 5%
rf = st.sidebar.number_input(
    "Risk-free rate (annual, decimal)",
    value=0.05,
    min_value=-0.05,
    max_value=0.20,
    step=0.005
)

allow_short = st.sidebar.checkbox("Allow short selling in optimization?", value=False)

n_points = st.sidebar.slider(
    "Efficient frontier resolution (number of portfolios)",
    min_value=10,
    max_value=100,
    value=40,
    step=5
)

allow_sells = st.sidebar.checkbox(
    "Allow selling when suggesting trades?",
    value=True
)

# Confidence level selector for CIs
conf_level = st.sidebar.selectbox(
    "Confidence level for return intervals",
    options=[0.90, 0.95, 0.975, 0.99],
    format_func=lambda x: f"{int(x * 100)}%"
)

# Parse tickers from user input
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(tickers) == 0:
    st.warning("Please enter at least one ticker in the sidebar to get started with SharpeEngine.")
    st.stop()

# ------------------ OCR INPUT (OPTIONAL) ------------------ #

ocr_available = ocr_is_available()
if not ocr_available:
    st.info(
        "OCR is optional and not installed in this environment. "
        "To enable it, install `easyocr` (and its dependencies)."
    )

uploaded_image = None
if ocr_available:
    uploaded_image = st.file_uploader(
        "Upload Robinhood screenshot (optional)",
        type=["png", "jpg", "jpeg"]
    )

ocr_positions = {}
raw_text = ""
if uploaded_image is not None:
    with st.spinner("Running OCR on screenshot..."):
        try:
            ocr_positions, raw_text = extract_shares_from_image(uploaded_image)
        except Exception as e:
            st.warning(f"OCR failed to run: {e}")
            ocr_positions = {}
            raw_text = ""

# If OCR found tickers not in the sidebar list, include them
extra_tickers = [t for t in ocr_positions.keys() if t not in tickers]
if extra_tickers:
    tickers = tickers + sorted(extra_tickers)

# ------------------ LOAD PRICE DATA ------------------ #

with st.spinner("Loading historical price data..."):
    price_df = load_price_data(tickers, annual_returns)

if price_df is None or price_df.empty:
    st.error("No price data returned. Check ticker symbols or try a different lookback window.")
    st.stop()

# Warn about tickers that had no data
loaded_tickers = set(price_df.columns.astype(str))
requested_tickers = set(tickers)
missing = requested_tickers - loaded_tickers

if missing:
    st.warning(
        "No data found for the following tickers, so they were ignored: "
        + ", ".join(sorted(missing))
    )

# ------------------ PRICE SNAPSHOTS TABLE ------------------ #

st.subheader("📊 Price Snapshots (Current & Prior Annual Points)")

start_date_global = price_df.index.min()
end_date_global = price_df.index.max()

years_back = annual_returns + 1
st.write(
    f"Historical data from **{start_date_global.date()}** to "
    f"**{end_date_global.date()}** "
    f"({years_back} year window requested for "
    f"{annual_returns} annual return periods)."
)

# Use exactly the number of years requested by the user
max_years = annual_returns

rows = []
for ticker in price_df.columns:
    s = price_df[ticker].dropna()
    if s.empty:
        continue

    # Years available for this specific ticker
    first_date = s.index.min()
    years_available = int((end_date_global - first_date).days // 365.25)

    row = {
        "Ticker": ticker,
        "Years Available": years_available
    }

    # Current + kY Ago columns
    for k in range(0, max_years + 1):
        if k == 0:
            target_date = end_date_global
            label = "Current"
        else:
            target_date = end_date_global - pd.DateOffset(years=k)
            label = f"{k}Y Ago"

        # Last available price on or before target_date
        s_up_to = s.loc[:target_date]
        if not s_up_to.empty:
            value = s_up_to.iloc[-1]
            if isinstance(value, float) and np.isnan(value):
                value = "N/A"
        else:
            value = "N/A"

        row[label] = value

    rows.append(row)

snapshot_df = pd.DataFrame(rows).set_index("Ticker")
st.dataframe(snapshot_df)

# ------------------ METRICS ------------------ #

metrics = compute_metrics(price_df)
mu = metrics["mu"]           # expected annual returns (Series)
cov = metrics["cov"]         # annualized covariance matrix (DataFrame)
corr = metrics["corr"]       # correlation matrix
vol = metrics["vol"]         # annual volatilities (Series)
returns_df = metrics["returns"]  # daily log returns

st.subheader("📌 Asset Metrics (Annualized)")
metrics_df = pd.DataFrame({
    "Expected Return": mu,
    "Volatility": vol
})
st.dataframe(metrics_df.style.format("{:.2%}"))

with st.expander("🔗 Correlation Matrix"):
    st.dataframe(corr.style.format("{:.2f}"))

# ------------------ EFFICIENT FRONTIER ------------------ #

st.subheader("🚀 Efficient Frontier (SharpeEngine Core)")

frontier = efficient_frontier(mu.values, cov.values, n_points=n_points, allow_short=allow_short)
if len(frontier) == 0:
    st.error("Could not generate an efficient frontier. Try adjusting tickers or lookback.")
    st.stop()

ms_result = max_sharpe(mu.values, cov.values, rf, allow_short=allow_short)
if ms_result is not None:
    w_ms, ret_ms, vol_ms, sharpe_ms = ms_result
    max_sharpe_stats = {
        "weights": w_ms,
        "return": ret_ms,
        "vol": vol_ms,
        "sharpe": sharpe_ms
    }
else:
    max_sharpe_stats = None

# ------------------ CURRENT PORTFOLIO INPUT ------------------ #

st.subheader("📂 Current Portfolio")

st.markdown(
    "You can either:\n"
    "- Edit the table manually, **or**\n"
    "- Upload a screenshot of your **Robinhood** holdings and let SharpeEngine fill in share counts."
)

show_ocr_debug = False
if ocr_available:
    show_ocr_debug = st.checkbox("Show OCR debug text", value=False)

if uploaded_image is not None:
    if ocr_positions:
        pretty = ", ".join([f"{t}: {s} shares" for t, s in ocr_positions.items()])
        st.success(f"Detected positions from screenshot: {pretty}")
    else:
        st.warning(
            "Could not detect any 'TICKER ... shares' patterns in the screenshot. "
            "You may need a clearer crop or try again."
        )

    if show_ocr_debug:
        st.text_area("OCR raw text", raw_text, height=200)

# Default rows: use OCR shares -> dollar amount when available,
# otherwise fall back to a generic 1000.0 amount.
last_prices = price_df.iloc[-1]

default_rows = []
for t in mu.index:
    if t in ocr_positions:
        shares = ocr_positions[t]
        price = float(last_prices.get(t, np.nan))
        amount = shares * price if not np.isnan(price) and price > 0 else 0.0
    else:
        amount = 1000.0  # fallback if no OCR data

    default_rows.append({"Ticker": t, "Amount": amount})

portfolio_df = st.data_editor(
    pd.DataFrame(default_rows),
    num_rows="dynamic",
    key="portfolio_editor"
)

# Clean and align portfolio input
portfolio_df["Ticker"] = portfolio_df["Ticker"].astype(str).str.upper()
portfolio_df = portfolio_df[portfolio_df["Ticker"].isin(mu.index)]

current_portfolio_stats = None
w_port_dict = None
w_vec = None
closest_point = None
w_closest = None

if portfolio_df.empty:
    st.info("No valid tickers in current portfolio (matching loaded price data).")
else:
    tickers_port = portfolio_df["Ticker"].tolist()
    amounts = portfolio_df["Amount"].astype(float).tolist()

    # Normalize to weights, but only using assets that exist in the current universe
    w_port_dict = normalize_weights(
        dict(zip(tickers_port, amounts)),
        available_assets=mu.index.tolist()
    )

    # Convert dict to weight vector in the same order as mu.index
    w_vec = np.array([w_port_dict.get(t, 0.0) for t in mu.index])

    current_portfolio_stats = portfolio_stats(w_vec, mu.values, cov.values, rf=rf)

    st.markdown("**Current Portfolio Stats (based on selected lookback):**")
    cp = current_portfolio_stats
    st.write(
        f"- Expected Return: **{cp['return']:.2%}**  \n"
        f"- Volatility: **{cp['vol']:.2%}**  \n"
        f"- Variance: **{cp['var']:.4f}**  \n"
        f"- Sharpe Ratio: **{cp['sharpe']:.3f}** (using risk-free {rf:.2%})"
    )

    # Precompute closest frontier point (for later CI & rebalancing)
    if len(frontier) > 0:
        closest_point = find_closest_frontier_point(
            frontier,
            current_ret=cp["return"],
            current_vol=cp["vol"]
        )
        if closest_point is not None:
            w_closest = np.array(closest_point["weights"])

# ------------------ REBALANCING TO EFFICIENT FRONTIER ------------------ #

st.subheader("📈 Rebalancing to Efficient Frontier")

if current_portfolio_stats is not None and len(frontier) > 0:
    cp = current_portfolio_stats

    # Current dollar amounts per ticker (grouped)
    amounts_by_ticker = portfolio_df.groupby("Ticker")["Amount"].sum()
    total_value = float(amounts_by_ticker.sum())
    price_series = price_df.iloc[-1][mu.index]

    if total_value <= 0:
        st.info("Total portfolio value is zero or negative; cannot compute rebalancing.")
    else:
        if allow_sells:
            # -------- Buy & Sell to closest frontier point (same total value) -------- #
            target_point = closest_point

            if target_point is None:
                st.info("Could not identify a suitable frontier portfolio.")
            else:
                w_target = np.array(target_point["weights"])  # weights in same order as mu.values
                target_ret = target_point["return"]
                target_vol = target_point["vol"]

                st.write(
                    f"Closest efficient-frontier portfolio has:  \n"
                    f"- Expected Return: **{target_ret:.2%}**  \n"
                    f"- Volatility: **{target_vol:.2%}**  \n"
                    f"- Sharpe Ratio: **{((target_ret - rf) / target_vol if target_vol > 0 else float('nan')):.3f}**"
                )

                rebalance_rows = []
                total_sale_proceeds = 0.0
                total_buy_cost = 0.0

                for i, t in enumerate(mu.index):
                    price = float(price_series.get(t, np.nan))

                    # Current amount & shares
                    curr_amt = float(amounts_by_ticker.get(t, 0.0))
                    curr_shares = (
                        curr_amt / price if price > 0 and not np.isnan(price) else 0.0
                    )

                    # Target amount & shares (same total value, different weights)
                    target_weight = float(w_target[i])
                    target_amt = target_weight * total_value
                    target_shares = (
                        target_amt / price if price > 0 and not np.isnan(price) else 0.0
                    )

                    delta_shares = target_shares - curr_shares

                    if delta_shares > 1e-6:
                        action = "Buy"
                    elif delta_shares < -1e-6:
                        action = "Sell"
                    else:
                        action = "Hold"

                    # Cash flows per ticker
                    buy_cost = 0.0
                    sale_proceeds = 0.0
                    if action == "Buy":
                        buy_cost = delta_shares * price  # positive cash out
                        total_buy_cost += buy_cost
                    elif action == "Sell":
                        sale_proceeds = -delta_shares * price  # delta_shares is negative
                        total_sale_proceeds += sale_proceeds

                    rebalance_rows.append({
                        "Ticker": t,
                        "Price": price,
                        "Current Shares": curr_shares,
                        "Target Shares": target_shares,
                        "Δ Shares": delta_shares,
                        "Action": action,
                        "Sale Proceeds ($)": sale_proceeds,
                        "Buy Cost ($)": buy_cost,
                    })

                rebalance_df = pd.DataFrame(rebalance_rows).set_index("Ticker")

                net_cash = total_sale_proceeds - total_buy_cost

                st.markdown(
                    "To move your current portfolio **onto the efficient frontier** "
                    "(closest point), SharpeEngine suggests the following trades "
                    "(buys **and** sells allowed):"
                )

                st.dataframe(
                    rebalance_df.style.format({
                        "Price": "{:.2f}",
                        "Current Shares": "{:.4f}",
                        "Target Shares": "{:.4f}",
                        "Δ Shares": "{:.4f}",
                        "Sale Proceeds ($)": "${:,.2f}",
                        "Buy Cost ($)": "${:,.2f}",
                    })
                )

                st.markdown(
                    f"- **Total sale proceeds:** ${total_sale_proceeds:,.2f}  \n"
                    f"- **Total buy cost:** ${total_buy_cost:,.2f}  \n"
                    f"- **Net cash from rebalance (sales - buys):** ${net_cash:,.2f}"
                )

        else:
            # -------- Buys-only (no selling) -------- #
            st.markdown(
                "Buys-only mode selected: SharpeEngine will find a feasible efficient-frontier "
                "portfolio that you can reach **only by adding new cash (no selling)**."
            )

            # Current amounts vector aligned to mu.index
            a_vec = np.array([float(amounts_by_ticker.get(t, 0.0)) for t in mu.index])
            V0 = total_value

            best_point = None
            best_V_target = None
            best_dist = float("inf")

            for p in frontier:
                w = np.array(p["weights"])
                # Check feasibility: if we already hold a ticker but target weight is zero,
                # reaching that portfolio would require selling -> infeasible in buys-only.
                infeasible = False
                V_req_list = []

                for i, t in enumerate(mu.index):
                    a_i = a_vec[i]
                    w_i = w[i]

                    if a_i > 1e-8 and w_i <= 1e-8:
                        infeasible = True
                        break

                    if a_i > 1e-8 and w_i > 1e-8:
                        V_req_list.append(a_i / w_i)

                if infeasible:
                    continue

                if len(V_req_list) == 0:
                    V_req = V0
                else:
                    V_req = max(V_req_list)

                V_target = max(V_req, V0)

                d = (p["return"] - cp["return"]) ** 2 + (p["vol"] - cp["vol"]) ** 2

                if d < best_dist:
                    best_dist = d
                    best_point = p
                    best_V_target = V_target

            if best_point is None:
                st.info(
                    "No efficient-frontier portfolio can be reached without selling at least "
                    "one existing holding. Try enabling selling in the sidebar."
                )
            else:
                w_target = np.array(best_point["weights"])
                target_ret = best_point["return"]
                target_vol = best_point["vol"]
                V_target = best_V_target
                additional_cash = V_target - V0

                st.write(
                    f"Closest **buys-only** efficient-frontier portfolio has:  \n"
                    f"- Expected Return: **{target_ret:.2%}**  \n"
                    f"- Volatility: **{target_vol:.2%}**  \n"
                    f"- Current portfolio value: **${V0:,.2f}**  \n"
                    f"- Target portfolio value: **${V_target:,.2f}**  \n"
                    f"- Additional cash required (buys only): **${additional_cash:,.2f}**"
                )

                target_amounts = w_target * V_target

                rebalance_rows = []
                for i, t in enumerate(mu.index):
                    price = float(price_series.get(t, np.nan))
                    curr_amt = a_vec[i]
                    curr_shares = (
                        curr_amt / price if price > 0 and not np.isnan(price) else 0.0
                    )

                    tgt_amt = target_amounts[i]
                    tgt_shares = (
                        tgt_amt / price if price > 0 and not np.isnan(price) else 0.0
                    )

                    delta_shares = max(0.0, tgt_shares - curr_shares)
                    action = "Buy" if delta_shares > 1e-6 else "Hold"

                    rebalance_rows.append({
                        "Ticker": t,
                        "Price": price,
                        "Current Shares": curr_shares,
                        "Target Shares": tgt_shares,
                        "Δ Shares (Buys Only)": delta_shares,
                        "Action": action
                    })

                rebalance_df = pd.DataFrame(rebalance_rows).set_index("Ticker")

                st.markdown(
                    "To move your portfolio **toward the efficient frontier** using only "
                    "**additional purchases (no selling)**, SharpeEngine suggests:"
                )

                st.dataframe(
                    rebalance_df.style.format({
                        "Price": "{:.2f}",
                        "Current Shares": "{:.4f}",
                        "Target Shares": "{:.4f}",
                        "Δ Shares (Buys Only)": "{:.4f}"
                    })
                )
else:
    st.info("Define a current portfolio and efficient frontier to see rebalancing suggestions.")

# ------------------ REBALANCING TO MAX-SHARPE PORTFOLIO ------------------ #

st.subheader("⭐ Rebalancing to Max-Sharpe Portfolio")

if current_portfolio_stats is not None and max_sharpe_stats is not None:
    # Current dollar amounts per ticker (grouped)
    amounts_by_ticker = portfolio_df.groupby("Ticker")["Amount"].sum()
    total_value = float(amounts_by_ticker.sum())
    price_series = price_df.iloc[-1][mu.index]

    if total_value <= 0:
        st.info("Total portfolio value is zero or negative; cannot compute rebalancing.")
    else:
        w_target = np.array(max_sharpe_stats["weights"])
        target_ret = max_sharpe_stats["return"]
        target_vol = max_sharpe_stats["vol"]

        st.write(
            f"Max-Sharpe portfolio has:  \n"
            f"- Expected Return: **{target_ret:.2%}**  \n"
            f"- Volatility: **{target_vol:.2%}**  \n"
            f"- Sharpe Ratio: **{((target_ret - rf) / target_vol if target_vol > 0 else float('nan')):.3f}**"
        )

        if allow_sells:
            # -------- Buy & Sell to max-sharpe point (same total value) -------- #
            rebalance_rows = []
            total_sale_proceeds = 0.0
            total_buy_cost = 0.0

            for i, t in enumerate(mu.index):
                price = float(price_series.get(t, np.nan))

                # Current amount & shares
                curr_amt = float(amounts_by_ticker.get(t, 0.0))
                curr_shares = (
                    curr_amt / price if price > 0 and not np.isnan(price) else 0.0
                )

                # Target amount & shares (same total value, different weights)
                target_weight = float(w_target[i])
                target_amt = target_weight * total_value
                target_shares = (
                    target_amt / price if price > 0 and not np.isnan(price) else 0.0
                )

                delta_shares = target_shares - curr_shares

                if delta_shares > 1e-6:
                    action = "Buy"
                elif delta_shares < -1e-6:
                    action = "Sell"
                else:
                    action = "Hold"

                # Cash flows per ticker
                buy_cost = 0.0
                sale_proceeds = 0.0
                if action == "Buy":
                    buy_cost = delta_shares * price  # positive cash out
                    total_buy_cost += buy_cost
                elif action == "Sell":
                    sale_proceeds = -delta_shares * price  # delta_shares is negative
                    total_sale_proceeds += sale_proceeds

                rebalance_rows.append({
                    "Ticker": t,
                    "Price": price,
                    "Current Shares": curr_shares,
                    "Target Shares": target_shares,
                    "Δ Shares": delta_shares,
                    "Action": action,
                    "Sale Proceeds ($)": sale_proceeds,
                    "Buy Cost ($)": buy_cost,
                })

            rebalance_df = pd.DataFrame(rebalance_rows).set_index("Ticker")

            net_cash = total_sale_proceeds - total_buy_cost

            st.markdown(
                "To move your current portfolio **to the max-Sharpe portfolio**, "
                "SharpeEngine suggests the following trades "
                "(buys **and** sells allowed):"
            )

            st.dataframe(
                rebalance_df.style.format({
                    "Price": "{:.2f}",
                    "Current Shares": "{:.4f}",
                    "Target Shares": "{:.4f}",
                    "Δ Shares": "{:.4f}",
                    "Sale Proceeds ($)": "${:,.2f}",
                    "Buy Cost ($)": "${:,.2f}",
                })
            )

            st.markdown(
                f"- **Total sale proceeds:** ${total_sale_proceeds:,.2f}  \n"
                f"- **Total buy cost:** ${total_buy_cost:,.2f}  \n"
                f"- **Net cash from rebalance (sales - buys):** ${net_cash:,.2f}"
            )

        else:
            # -------- Buys-only (no selling) -------- #
            st.markdown(
                "Buys-only mode selected: SharpeEngine will determine how much additional "
                "cash is required to reach the max-Sharpe portfolio **without selling**."
            )

            # Current amounts vector aligned to mu.index
            a_vec = np.array([float(amounts_by_ticker.get(t, 0.0)) for t in mu.index])
            V0 = total_value

            infeasible = False
            V_req_list = []
            for i, t in enumerate(mu.index):
                a_i = a_vec[i]
                w_i = w_target[i]

                if a_i > 1e-8 and w_i <= 1e-8:
                    infeasible = True
                    break

                if a_i > 1e-8 and w_i > 1e-8:
                    V_req_list.append(a_i / w_i)

            if infeasible:
                st.info(
                    "Your current holdings include assets that have zero weight in the "
                    "max-Sharpe portfolio. Reaching it without selling is not feasible."
                )
            else:
                V_req = max(V_req_list) if len(V_req_list) > 0 else V0
                V_target = max(V_req, V0)
                additional_cash = V_target - V0

                st.write(
                    f"- Current portfolio value: **${V0:,.2f}**  \n"
                    f"- Target portfolio value: **${V_target:,.2f}**  \n"
                    f"- Additional cash required (buys only): **${additional_cash:,.2f}**"
                )

                target_amounts = w_target * V_target

                rebalance_rows = []
                for i, t in enumerate(mu.index):
                    price = float(price_series.get(t, np.nan))
                    curr_amt = a_vec[i]
                    curr_shares = (
                        curr_amt / price if price > 0 and not np.isnan(price) else 0.0
                    )

                    tgt_amt = target_amounts[i]
                    tgt_shares = (
                        tgt_amt / price if price > 0 and not np.isnan(price) else 0.0
                    )

                    delta_shares = max(0.0, tgt_shares - curr_shares)
                    action = "Buy" if delta_shares > 1e-6 else "Hold"

                    rebalance_rows.append({
                        "Ticker": t,
                        "Price": price,
                        "Current Shares": curr_shares,
                        "Target Shares": tgt_shares,
                        "Δ Shares (Buys Only)": delta_shares,
                        "Action": action
                    })

                rebalance_df = pd.DataFrame(rebalance_rows).set_index("Ticker")

                st.markdown(
                    "To move your portfolio **toward the max-Sharpe portfolio** using only "
                    "**additional purchases (no selling)**, SharpeEngine suggests:"
                )

                st.dataframe(
                    rebalance_df.style.format({
                        "Price": "{:.2f}",
                        "Current Shares": "{:.4f}",
                        "Target Shares": "{:.4f}",
                        "Δ Shares (Buys Only)": "{:.4f}"
                    })
                )
else:
    st.info("Define a current portfolio and compute the max-Sharpe portfolio to see rebalancing suggestions.")

# ------------------ CONFIDENCE INTERVALS FOR EXPECTED RETURNS ------------------ #

st.subheader("📐 Confidence Intervals for Annual Expected Returns")

if current_portfolio_stats is not None:
    # Current portfolio CI
    ci_curr = portfolio_return_ci(
        returns_df=returns_df,
        weights=w_vec,
        mu_index=mu.index,
        conf_level=conf_level
    )

    if ci_curr[0] is not None:
        st.markdown(
            f"**Current portfolio:** There is **{int(conf_level * 100)}%** confidence that "
            f"the **annual expected return** lies between "
            f"**{ci_curr[0]:.2%}** and **{ci_curr[1]:.2%}**."
        )
    else:
        st.info("Not enough data to compute a confidence interval for the current portfolio.")

    # Probability of negative annual return (current portfolio)
    prob_neg, n_obs, dist_used, mean_annual = portfolio_prob_negative_annual_return(
        returns_df=returns_df,
        weights=w_vec,
        mu_index=mu.index
    )
    if prob_neg is not None:
        st.markdown(
            f"**Current portfolio:** Estimated probability of a **negative annual return** is "
            f"**{prob_neg:.2%}** "
            f"(using {dist_used}-distribution; n={n_obs})."
        )
    else:
        st.info("Not enough data to estimate probability of a negative annual return for the current portfolio.")

    # Closest efficient-frontier portfolio CI
    if w_closest is not None:
        ci_closest = portfolio_return_ci(
            returns_df=returns_df,
            weights=w_closest,
            mu_index=mu.index,
            conf_level=conf_level
        )
        if ci_closest[0] is not None:
            st.markdown(
                f"**Closest efficient-frontier portfolio:** There is **{int(conf_level * 100)}%** "
                f"confidence that the **annual expected return** lies between "
                f"**{ci_closest[0]:.2%}** and **{ci_closest[1]:.2%}**."
            )
        else:
            st.info("Not enough data to compute a confidence interval for the closest frontier portfolio.")

        prob_neg_close, n_obs_close, dist_used_close, _ = portfolio_prob_negative_annual_return(
            returns_df=returns_df,
            weights=w_closest,
            mu_index=mu.index
        )
        if prob_neg_close is not None:
            st.markdown(
                f"**Closest efficient-frontier portfolio:** Estimated probability of a "
                f"**negative annual return** is **{prob_neg_close:.2%}** "
                f"(using {dist_used_close}-distribution; n={n_obs_close})."
            )
        else:
            st.info(
                "Not enough data to estimate probability of a negative annual return "
                "for the closest frontier portfolio."
            )
else:
    st.info("Set up a valid current portfolio to compute confidence intervals.")

# Max Sharpe portfolio CI (doesn't require your current portfolio)
if max_sharpe_stats is not None:
    ci_ms = portfolio_return_ci(
        returns_df=returns_df,
        weights=max_sharpe_stats["weights"],
        mu_index=mu.index,
        conf_level=conf_level
    )
    if ci_ms[0] is not None:
        st.markdown(
            f"**Max-Sharpe portfolio:** There is **{int(conf_level * 100)}%** confidence that "
            f"the **annual expected return** lies between "
            f"**{ci_ms[0]:.2%}** and **{ci_ms[1]:.2%}**."
        )
    else:
        st.info("Not enough data to compute a confidence interval for the max-Sharpe portfolio.")

    prob_neg_ms, n_obs_ms, dist_used_ms, _ = portfolio_prob_negative_annual_return(
        returns_df=returns_df,
        weights=max_sharpe_stats["weights"],
        mu_index=mu.index
    )
    if prob_neg_ms is not None:
        st.markdown(
            f"**Max-Sharpe portfolio:** Estimated probability of a **negative annual return** "
            f"is **{prob_neg_ms:.2%}** (using {dist_used_ms}-distribution; n={n_obs_ms})."
        )
    else:
        st.info("Not enough data to estimate probability of a negative annual return for the max-Sharpe portfolio.")

# ------------------ PLOT: RETURN VS VARIANCE ------------------ #

x_frontier = [p["vol"] ** 2 for p in frontier]
y_frontier = [p["return"] for p in frontier]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_frontier,
    y=y_frontier,
    mode="lines+markers",
    name="Efficient Frontier"
))

# Current portfolio point
if current_portfolio_stats is not None:
    fig.add_trace(go.Scatter(
        x=[current_portfolio_stats["vol"] ** 2],
        y=[current_portfolio_stats["return"]],
        mode="markers",
        marker=dict(size=11, symbol="x"),
        name="Current Portfolio"
    ))

# Max Sharpe portfolio
if max_sharpe_stats is not None:
    fig.add_trace(go.Scatter(
        x=[max_sharpe_stats["vol"] ** 2],
        y=[max_sharpe_stats["return"]],
        mode="markers",
        marker=dict(size=11),
        name="Max Sharpe Portfolio"
    ))

fig.update_layout(
    xaxis_title="Variance (σ²)",
    yaxis_title="Expected Return (μ)",
    title="SharpeEngine: Expected Return vs Variance",
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig, use_container_width=True)

# ------------------ MATH EXPLAINER + NEW INVESTMENT ATTRACTIVENESS CHECK ------------------ #

with st.expander("📘 Math Behind SharpeEngine Investment Check"):
    st.markdown(r"""
    The investment attractiveness condition used by SharpeEngine is:

    $$
    r_f + S_p \cdot \sigma_i \cdot \rho_{p,i} < \mathbb{E}[R_i]
    $$

    where:
    """)
    st.latex(r"r_f = \text{risk-free rate}")
    st.latex(r"S_p = \text{Sharpe ratio of your current portfolio}")
    st.latex(r"\sigma_i = \text{volatility of the candidate investment}")
    st.latex(r"\rho_{p,i} = \text{correlation between your portfolio and the investment}")
    st.latex(r"\mathbb{E}[R_i] = \text{expected return of the investment}")

st.subheader("🧮 New Investment Attractiveness (Sharpe Condition)")

candidate_ticker = st.selectbox(
    "Candidate investment ticker to test",
    options=["(none)"] + list(mu.index)
)

if candidate_ticker != "(none)" and current_portfolio_stats is not None and w_port_dict is not None:
    # Candidate stats
    mu_i = mu[candidate_ticker]
    vol_i = vol[candidate_ticker]

    # Correlation with portfolio
    w_vec_for_corr = np.array([w_port_dict.get(t, 0.0) for t in mu.index])
    corr_pi = portfolio_investment_corr(returns_df[mu.index], w_vec_for_corr, candidate_ticker)

    is_good, info = investment_is_attractive(
        current_portfolio_stats["return"],
        current_portfolio_stats["vol"],
        mu_i,
        vol_i,
        corr_pi,
        rf
    )

    threshold = info["threshold"]
    sharpe_p = info["sharpe_port"]

    st.write(f"**Candidate:** `{candidate_ticker}`")
    st.write(f"- Expected Return of Investment: **{mu_i:.2%}**")
    st.write(f"- Volatility of Investment: **{vol_i:.2%}**")
    st.write(f"- Correlation with Portfolio: **{corr_pi:.3f}**")
    st.write(f"- Portfolio Sharpe Ratio (SharpeEngine): **{sharpe_p:.3f}**")
    st.write(f"- Threshold: **{threshold:.2%}** (must be **less** than investment expected return)")

    if is_good:
        st.success(
            "✅ SharpeEngine verdict: Condition satisfied. "
            "The investment's expected return is high enough given your portfolio's risk/return profile."
        )
    else:
        st.warning(
            "⚠️ SharpeEngine verdict: Condition NOT satisfied. "
            "The investment does **not** clear your Sharpe-based hurdle."
        )
elif candidate_ticker != "(none)" and current_portfolio_stats is None:
    st.info("Set up a valid current portfolio above to run the SharpeEngine attractiveness check.")
