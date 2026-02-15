import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from io import StringIO

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hyperliquid Wallet Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

/* Global */
.stApp {
    font-family: 'DM Sans', sans-serif;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f1116 0%, #1a1d26 100%);
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 16px 20px;
    transition: border-color 0.2s;
}
div[data-testid="stMetric"]:hover {
    border-color: #4a9eff;
}
div[data-testid="stMetric"] label {
    font-family: 'DM Sans', sans-serif !important;
    color: #8b8fa3 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0c10;
    border-right: 1px solid #1a1d26;
}

/* Tables */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* Headers */
h1, h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
}

/* Code / mono */
code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# API FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

HL_INFO_URL = "https://api.hyperliquid.xyz/info"


@st.cache_data(ttl=60, show_spinner=False)
def fetch_fills(address: str, start_time: int, end_time: int, max_pages: int = 50) -> list:
    """Fetch all fills for an address using paginated userFillsByTime."""
    all_fills = []
    current_end = end_time
    pages = 0

    while pages < max_pages:
        try:
            resp = requests.post(
                HL_INFO_URL,
                json={
                    "type": "userFillsByTime",
                    "user": address,
                    "startTime": start_time,
                    "endTime": current_end,
                },
                timeout=15,
            )
            resp.raise_for_status()
            fills = resp.json()
        except Exception as e:
            st.warning(f"API error on page {pages + 1}: {e}")
            break

        if not fills:
            break

        all_fills.extend(fills)
        oldest_time = fills[-1]["time"]

        # If we got the same timestamp, we're done
        if oldest_time >= current_end:
            break

        current_end = oldest_time - 1
        pages += 1
        time.sleep(0.3)

    return all_fills


@st.cache_data(ttl=60, show_spinner=False)
def fetch_account_state(address: str) -> dict | None:
    """Fetch current open positions and account value."""
    try:
        resp = requests.post(
            HL_INFO_URL,
            json={"type": "clearinghouseState", "user": address},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fills_to_dataframe(fills: list) -> pd.DataFrame:
    """Convert raw API fills list to a clean DataFrame."""
    if not fills:
        return pd.DataFrame()

    df = pd.DataFrame(fills)
    df["px"] = pd.to_numeric(df["px"], errors="coerce")
    df["sz"] = pd.to_numeric(df["sz"], errors="coerce")
    df["closedPnl"] = pd.to_numeric(df["closedPnl"], errors="coerce")
    df["fee"] = pd.to_numeric(df.get("fee", 0), errors="coerce").fillna(0)
    df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["date"] = df["datetime"].dt.date
    df["notional"] = df["px"] * df["sz"]
    df["side_label"] = df["side"].map({"B": "Buy", "A": "Sell"})

    # Deduplicate by tid
    if "tid" in df.columns:
        df = df.drop_duplicates(subset="tid")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────


def analyze_pnl(df: pd.DataFrame) -> dict:
    """Compute comprehensive PnL statistics from fills DataFrame."""
    if df.empty:
        return {}

    # closedPnl from the API is the realized PnL per fill
    closing_fills = df[df["closedPnl"] != 0].copy()

    total_realized_pnl = closing_fills["closedPnl"].sum()
    total_fees = df["fee"].sum()
    net_pnl = total_realized_pnl - total_fees

    winners = closing_fills[closing_fills["closedPnl"] > 0]
    losers = closing_fills[closing_fills["closedPnl"] < 0]

    win_count = len(winners)
    loss_count = len(losers)
    total_closing = win_count + loss_count
    win_rate = win_count / total_closing * 100 if total_closing > 0 else 0

    gross_profit = winners["closedPnl"].sum() if len(winners) > 0 else 0
    gross_loss = losers["closedPnl"].sum() if len(losers) > 0 else 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    avg_win = winners["closedPnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["closedPnl"].mean() if len(losers) > 0 else 0
    best_trade = closing_fills["closedPnl"].max() if len(closing_fills) > 0 else 0
    worst_trade = closing_fills["closedPnl"].min() if len(closing_fills) > 0 else 0

    # Daily PnL
    daily_pnl = closing_fills.groupby("date")["closedPnl"].sum()
    daily_fees = df.groupby("date")["fee"].sum()
    daily_net = daily_pnl.subtract(daily_fees, fill_value=0)

    profitable_days = (daily_net > 0).sum()
    total_days = len(daily_net)

    # Sharpe
    if len(daily_net) > 1 and daily_net.std() > 0:
        sharpe = daily_net.mean() / daily_net.std() * np.sqrt(365)
    else:
        sharpe = None

    # Max drawdown on cumulative PnL
    cum = daily_net.cumsum()
    peak = cum.expanding().max()
    drawdown = cum - peak
    max_dd = drawdown.min() if len(drawdown) > 0 else 0

    # Volume
    total_volume = df["notional"].sum()

    # By coin
    coin_pnl = closing_fills.groupby("coin").agg(
        pnl=("closedPnl", "sum"),
        trades=("closedPnl", "count"),
        wins=("closedPnl", lambda x: (x > 0).sum()),
        losses=("closedPnl", lambda x: (x < 0).sum()),
        best=("closedPnl", "max"),
        worst=("closedPnl", "min"),
        gross_win=("closedPnl", lambda x: x[x > 0].sum()),
        gross_loss=("closedPnl", lambda x: x[x < 0].sum()),
    ).sort_values("pnl", ascending=False)
    coin_pnl["win_rate"] = coin_pnl["wins"] / coin_pnl["trades"] * 100
    coin_pnl["avg_win"] = coin_pnl["gross_win"] / coin_pnl["wins"].replace(0, np.nan)
    coin_pnl["avg_loss"] = coin_pnl["gross_loss"] / coin_pnl["losses"].replace(0, np.nan)

    # Coin volume
    coin_vol = df.groupby("coin")["notional"].sum()
    coin_pnl["volume"] = coin_vol

    return {
        "total_realized_pnl": total_realized_pnl,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "total_fills": len(df),
        "closing_fills": total_closing,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "daily_pnl": daily_pnl,
        "daily_net": daily_net,
        "profitable_days": profitable_days,
        "total_days": total_days,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_volume": total_volume,
        "coin_pnl": coin_pnl,
    }


def analyze_copy_feasibility(df: pd.DataFrame, your_account: float, trader_equity_est: float) -> dict:
    """Analyze whether copy trading is feasible at a given account size."""
    if df.empty:
        return {}

    ratio = your_account / trader_equity_est if trader_equity_est > 0 else 0
    scaled_notionals = df["notional"] * ratio

    min_order = 10  # Hyperliquid typical minimum
    below_min = (scaled_notionals < min_order).mean() * 100
    median_scaled = scaled_notionals.median()
    max_scaled = scaled_notionals.max()

    # Estimated fees at your scale
    your_volume = df["notional"].sum() * ratio
    your_fees = your_volume * 0.00035

    return {
        "scale_ratio": ratio,
        "pct_below_min": below_min,
        "median_trade_size": median_scaled,
        "max_trade_size": max_scaled,
        "estimated_volume": your_volume,
        "estimated_fees": your_fees,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CHART FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#c0c4d6"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
)


def chart_cumulative_pnl(daily_net: pd.Series) -> go.Figure:
    cum = daily_net.cumsum()
    colors = ["#00c853" if v >= 0 else "#ff1744" for v in cum.values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        mode="lines+markers",
        line=dict(color="#4a9eff", width=2.5),
        marker=dict(size=6, color=colors, line=dict(width=1, color="#1a1d26")),
        fill="tozeroy",
        fillcolor="rgba(74,158,255,0.08)",
        hovertemplate="<b>%{x}</b><br>Cumulative PnL: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, title="Cumulative Net PnL", height=380)
    fig.update_yaxes(tickprefix="$")
    return fig


def chart_daily_pnl(daily_net: pd.Series) -> go.Figure:
    colors = ["#00c853" if v >= 0 else "#ff1744" for v in daily_net.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_net.index, y=daily_net.values,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>PnL: $%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, title="Daily Net PnL", height=340)
    fig.update_yaxes(tickprefix="$")
    return fig


def chart_pnl_by_coin(coin_pnl: pd.DataFrame) -> go.Figure:
    sorted_df = coin_pnl.sort_values("pnl")
    colors = ["#00c853" if v >= 0 else "#ff1744" for v in sorted_df["pnl"].values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_df.index, x=sorted_df["pnl"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>PnL: $%{x:,.2f}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, title="Realized PnL by Token", height=max(300, len(sorted_df) * 32))
    fig.update_xaxes(tickprefix="$")
    return fig


def chart_win_rate_by_coin(coin_pnl: pd.DataFrame) -> go.Figure:
    sorted_df = coin_pnl.sort_values("win_rate")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_df.index, x=sorted_df["win_rate"],
        orientation="h",
        marker_color="#4a9eff",
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Win Rate: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(**CHART_LAYOUT, title="Win Rate by Token", height=max(300, len(sorted_df) * 32))
    fig.update_xaxes(ticksuffix="%", range=[0, 105])
    return fig


def chart_trade_size_distribution(df: pd.DataFrame) -> go.Figure:
    notionals = df["notional"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=notionals,
        nbinsx=60,
        marker_color="#4a9eff",
        marker_line_width=0,
        hovertemplate="Size: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, title="Trade Size Distribution (Notional)", height=340)
    fig.update_xaxes(tickprefix="$")
    return fig


def chart_hourly_activity(df: pd.DataFrame) -> go.Figure:
    hours = df["datetime"].dt.hour.value_counts().sort_index()
    full_hours = pd.Series(0, index=range(24))
    full_hours.update(hours)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=full_hours.index, y=full_hours.values,
        marker_color="#4a9eff",
        marker_line_width=0,
        hovertemplate="Hour: %{x}:00 UTC<br>Trades: %{y}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, title="Trading Activity by Hour (UTC)", height=300)
    fig.update_xaxes(dtick=1)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Hyperliquid Analyzer")
    st.caption("Analyze any wallet's perp trading history")

    st.markdown("---")

    address = st.text_input(
        "Wallet address",
        placeholder="0x...",
        help="The Hyperliquid wallet address to analyze",
    )

    st.markdown("##### Date range")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("From", value=datetime.now() - timedelta(days=90))
    with col_e:
        end_date = st.date_input("To", value=datetime.now())

    st.markdown("---")

    st.markdown("##### Copy trade analysis")
    your_account_size = st.number_input(
        "Your account size ($)",
        min_value=100,
        max_value=10_000_000,
        value=1000,
        step=100,
    )
    est_trader_equity = st.number_input(
        "Est. trader equity ($)",
        min_value=1000,
        max_value=100_000_000,
        value=100_000,
        step=10_000,
        help="Estimate of the trader's account size. Check their max exposure and divide by leverage.",
    )

    st.markdown("---")

    fetch_btn = st.button("🔍  Analyze wallet", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("##### Or upload a CSV")
    uploaded = st.file_uploader("Upload Hyperliquid CSV export", type=["csv"])


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# Hyperliquid Wallet Analyzer")
st.caption("Perp PnL breakdown · copy trade feasibility · full trade history via API")

if not fetch_btn and uploaded is None:
    st.markdown("---")
    st.info("Enter a wallet address in the sidebar and click **Analyze wallet**, or upload a CSV export.")
    st.stop()

# ── Fetch or load data ──
df = pd.DataFrame()

if fetch_btn and address:
    if not address.startswith("0x") or len(address) != 42:
        st.error("Invalid address. Must be a 42-character hex address starting with 0x.")
        st.stop()

    start_ms = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
    end_ms = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1000)

    with st.spinner("Fetching trade history from Hyperliquid API..."):
        fills = fetch_fills(address, start_ms, end_ms)

    if not fills:
        st.warning("No fills found for this address in the selected date range.")
        st.stop()

    st.success(f"Fetched **{len(fills):,}** fills")
    df = fills_to_dataframe(fills)

elif uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        # Check if it's the Hyperliquid UI export format (has 'class' column)
        if "class" in raw.columns:
            raw = raw[raw["class"] == "PERP"].copy()
            raw["datetime"] = pd.to_datetime(raw["time_iso"])
            raw["date"] = raw["datetime"].dt.date
            raw["coin"] = raw["token"].apply(lambda x: x.split(":")[-1] if ":" in str(x) else x)
            raw["sz"] = raw["amount"].abs()
            raw["notional"] = raw["USDAmount"].abs()
            raw["side"] = raw["amount"].apply(lambda x: "B" if x > 0 else "A")
            raw["side_label"] = raw["side"].map({"B": "Buy", "A": "Sell"})
            raw["dir"] = raw["type"]

            # For CSV exports, closedPnl isn't directly available,
            # so we reconstruct using FIFO matching
            raw["closedPnl"] = 0.0
            raw["fee"] = raw.get("fee", 0.0)

            # FIFO PnL reconstruction
            mask = raw["px"].replace([np.inf, -np.inf], np.nan).isna()
            raw.loc[mask, "px"] = (raw.loc[mask, "USDAmount"].abs() / raw.loc[mask, "amount"].abs())

            for coin in raw["coin"].unique():
                coin_mask = raw["coin"] == coin
                coin_df = raw[coin_mask].sort_values("datetime")
                long_q, short_q = [], []

                for idx, row in coin_df.iterrows():
                    tp = row["type"]
                    amt = abs(row["amount"])
                    px_val = row["px"]

                    if "Open Long" in str(tp):
                        long_q.append((amt, px_val))
                    elif "Open Short" in str(tp):
                        short_q.append((amt, px_val))
                    elif "Close Long" in str(tp):
                        cq = amt
                        pnl = 0
                        while cq > 0.0001 and long_q:
                            oq, op = long_q[0]
                            m = min(cq, oq)
                            pnl += m * (px_val - op)
                            cq -= m
                            if m >= oq - 0.0001:
                                long_q.pop(0)
                            else:
                                long_q[0] = (oq - m, op)
                        raw.loc[idx, "closedPnl"] = pnl
                    elif "Close Short" in str(tp):
                        cq = amt
                        pnl = 0
                        while cq > 0.0001 and short_q:
                            oq, op = short_q[0]
                            m = min(cq, oq)
                            pnl += m * (op - px_val)
                            cq -= m
                            if m >= oq - 0.0001:
                                short_q.pop(0)
                            else:
                                short_q[0] = (oq - m, op)
                        raw.loc[idx, "closedPnl"] = pnl

            df = raw.sort_values("datetime").reset_index(drop=True)
        else:
            # Assume API-format JSON-like CSV
            df = fills_to_dataframe(raw.to_dict("records"))
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

if df.empty:
    st.warning("No data to analyze.")
    st.stop()

# ── Run analysis ──
stats = analyze_pnl(df)
copy_stats = analyze_copy_feasibility(df, your_account_size, est_trader_equity)

# ── Account state (if fetched from API) ──
acct_state = None
if fetch_btn and address:
    acct_state = fetch_account_state(address)


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

# ── Top metrics ──
st.markdown("---")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net PnL", f"${stats['net_pnl']:+,.2f}")
m2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
m3.metric("Profit Factor", f"{stats['profit_factor']:.2f}" if stats["profit_factor"] != float("inf") else "∞")
m4.metric("Total Fills", f"{stats['total_fills']:,}")
m5.metric("Total Fees", f"${stats['total_fees']:,.2f}")

m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("Avg Win", f"${stats['avg_win']:+,.2f}")
m7.metric("Avg Loss", f"${stats['avg_loss']:+,.2f}")
m8.metric("Best Trade", f"${stats['best_trade']:+,.2f}")
m9.metric("Worst Trade", f"${stats['worst_trade']:+,.2f}")
m10.metric("Max Drawdown", f"${stats['max_drawdown']:+,.2f}")

st.markdown("---")

# ── Charts ──
tab_overview, tab_tokens, tab_copy, tab_positions, tab_data = st.tabs([
    "📈 Overview", "🪙 By Token", "📋 Copy Trade", "💰 Open Positions", "📄 Raw Data"
])

with tab_overview:
    col_a, col_b = st.columns(2)
    with col_a:
        if len(stats["daily_net"]) > 0:
            st.plotly_chart(chart_cumulative_pnl(stats["daily_net"]), use_container_width=True)
    with col_b:
        if len(stats["daily_net"]) > 0:
            st.plotly_chart(chart_daily_pnl(stats["daily_net"]), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(chart_trade_size_distribution(df), use_container_width=True)
    with col_d:
        st.plotly_chart(chart_hourly_activity(df), use_container_width=True)

    st.markdown("##### Daily Breakdown")
    daily_table = stats["daily_net"].reset_index()
    daily_table.columns = ["Date", "Net PnL"]
    daily_table["Cumulative"] = daily_table["Net PnL"].cumsum()
    daily_table["Result"] = daily_table["Net PnL"].apply(lambda x: "✅ Profit" if x > 0 else "❌ Loss")
    st.dataframe(
        daily_table.style.format({"Net PnL": "${:+,.2f}", "Cumulative": "${:+,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
    if stats["sharpe"] is not None:
        st.caption(f"Annualized Sharpe: **{stats['sharpe']:.2f}** · "
                   f"Profitable days: **{stats['profitable_days']}/{stats['total_days']}**")

with tab_tokens:
    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(chart_pnl_by_coin(stats["coin_pnl"]), use_container_width=True)
    with col_f:
        st.plotly_chart(chart_win_rate_by_coin(stats["coin_pnl"]), use_container_width=True)

    st.markdown("##### Token Breakdown")
    display_coins = stats["coin_pnl"][["pnl", "trades", "win_rate", "avg_win", "avg_loss", "best", "worst", "volume"]].copy()
    display_coins.columns = ["PnL", "Trades", "Win %", "Avg Win", "Avg Loss", "Best", "Worst", "Volume"]
    st.dataframe(
        display_coins.style.format({
            "PnL": "${:+,.2f}", "Win %": "{:.1f}%",
            "Avg Win": "${:+,.2f}", "Avg Loss": "${:+,.2f}",
            "Best": "${:+,.2f}", "Worst": "${:+,.2f}",
            "Volume": "${:,.0f}",
        }),
        use_container_width=True,
    )

    # PnL concentration
    total_abs = stats["coin_pnl"]["pnl"].abs().sum()
    if total_abs > 0:
        st.markdown("##### PnL Concentration")
        conc = stats["coin_pnl"]["pnl"].abs() / total_abs * 100
        conc = conc.sort_values(ascending=False)
        top1 = conc.iloc[0] if len(conc) > 0 else 0
        top3 = conc.iloc[:3].sum() if len(conc) >= 3 else conc.sum()
        if top1 > 50:
            st.warning(f"⚠️ **Highly concentrated** — top token is {top1:.0f}% of total PnL. Top 3 tokens account for {top3:.0f}%.")
        elif top3 > 80:
            st.warning(f"⚠️ **Moderately concentrated** — top 3 tokens account for {top3:.0f}% of total PnL.")
        else:
            st.success(f"✅ **Well diversified** — top 3 tokens account for {top3:.0f}% of total PnL.")

with tab_copy:
    st.markdown("### Copy Trade Feasibility")
    st.markdown(f"**Your account:** ${your_account_size:,} · **Est. trader equity:** ${est_trader_equity:,} · **Scale ratio:** {copy_stats.get('scale_ratio', 0):.6f}")

    st.markdown("---")

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Trades below $10 min", f"{copy_stats.get('pct_below_min', 0):.1f}%")
    cc2.metric("Your median trade", f"${copy_stats.get('median_trade_size', 0):,.2f}")
    cc3.metric("Your est. fees", f"${copy_stats.get('estimated_fees', 0):,.2f}")

    pct = copy_stats.get("pct_below_min", 0)
    if pct > 60:
        st.error(f"🚫 **Not feasible.** {pct:.0f}% of trades would be below Hyperliquid's minimum order size. "
                 f"You'd miss the majority of entries/exits, completely breaking the strategy.")
        min_viable = your_account_size / copy_stats["scale_ratio"] * 0.01 if copy_stats.get("scale_ratio", 0) > 0 else 0
        if min_viable > 0:
            # Estimate: need pct_below < 20%, so find account size where that holds
            for test_acct in [5000, 10000, 25000, 50000, 100000]:
                test_ratio = test_acct / est_trader_equity
                test_below = (df["notional"] * test_ratio < 10).mean() * 100
                if test_below < 20:
                    st.info(f"💡 **Minimum suggested account:** ~${test_acct:,} (would reduce missed trades to ~{test_below:.0f}%)")
                    break
    elif pct > 30:
        st.warning(f"⚠️ **Marginal.** {pct:.0f}% of trades would be too small. You'd miss a significant chunk of the strategy.")
    elif pct > 10:
        st.info(f"ℹ️ **Mostly feasible.** {pct:.0f}% of trades below minimum — some smaller scaling entries would be missed.")
    else:
        st.success(f"✅ **Feasible.** Only {pct:.0f}% of trades below minimum order size.")

    # Slippage warning for low-liquidity tokens
    st.markdown("---")
    st.markdown("##### Additional Risks")
    st.markdown("""
    - **Slippage:** You always enter *after* the trader. On fast-moving or thin markets this can significantly erode edge.
    - **Timing:** Copy trade bots have latency. Even milliseconds matter on perps.
    - **Open positions:** You inherit whatever the trader is currently holding — check the Open Positions tab.
    - **Concentration:** If PnL is driven by 1-2 big trades, you're betting on lightning striking twice.
    - **Funding rates:** Long/short funding costs apply to your positions too.
    """)

with tab_positions:
    if acct_state and "assetPositions" in acct_state:
        positions = acct_state["assetPositions"]
        if positions:
            pos_data = []
            for p in positions:
                pos_info = p.get("position", {})
                pos_data.append({
                    "Coin": pos_info.get("coin", "?"),
                    "Size": float(pos_info.get("szi", 0)),
                    "Entry Px": float(pos_info.get("entryPx", 0)),
                    "Notional": abs(float(pos_info.get("szi", 0)) * float(pos_info.get("entryPx", 0))),
                    "Unrealized PnL": float(pos_info.get("unrealizedPnl", 0)),
                    "Leverage": pos_info.get("leverage", {}).get("value", "?"),
                    "Side": "LONG" if float(pos_info.get("szi", 0)) > 0 else "SHORT",
                })
            pos_df = pd.DataFrame(pos_data).sort_values("Notional", ascending=False)

            total_long = pos_df[pos_df["Side"] == "LONG"]["Notional"].sum()
            total_short = pos_df[pos_df["Side"] == "SHORT"]["Notional"].sum()
            total_unreal = pos_df["Unrealized PnL"].sum()

            p1, p2, p3 = st.columns(3)
            p1.metric("Long Exposure", f"${total_long:,.0f}")
            p2.metric("Short Exposure", f"${total_short:,.0f}")
            p3.metric("Unrealized PnL", f"${total_unreal:+,.2f}")

            st.dataframe(
                pos_df.style.format({
                    "Size": "{:,.4f}", "Entry Px": "${:,.4f}",
                    "Notional": "${:,.0f}", "Unrealized PnL": "${:+,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            if total_long + total_short > 0:
                net_label = "net LONG" if total_long > total_short else "net SHORT"
                net_val = abs(total_long - total_short)
                st.caption(f"Total open exposure: ${total_long + total_short:,.0f} ({net_label} ${net_val:,.0f})")
        else:
            st.info("No open positions.")
    else:
        st.info("Open positions are only available when fetching directly from the API (not CSV upload). "
                "Enter a wallet address and click Analyze.")

    # Account value
    if acct_state and "marginSummary" in acct_state:
        ms = acct_state["marginSummary"]
        st.markdown("---")
        st.markdown("##### Account Summary")
        a1, a2, a3 = st.columns(3)
        a1.metric("Account Value", f"${float(ms.get('accountValue', 0)):,.2f}")
        a2.metric("Total Margin Used", f"${float(ms.get('totalMarginUsed', 0)):,.2f}")
        a3.metric("Total Notional", f"${float(ms.get('totalNtlPos', 0)):,.2f}")

with tab_data:
    st.markdown("##### Raw Fills Data")
    st.caption(f"{len(df):,} fills")

    # Pick columns to show based on what's available
    show_cols = [c for c in ["datetime", "coin", "dir", "side_label", "sz", "px", "notional", "closedPnl", "fee"] if c in df.columns]
    if not show_cols:
        show_cols = df.columns.tolist()[:10]

    st.dataframe(
        df[show_cols].sort_values("datetime", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    # Download
    csv_export = df.to_csv(index=False)
    st.download_button(
        "📥 Download full data as CSV",
        csv_export,
        file_name=f"hl_fills_{address[:10] if address else 'upload'}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
