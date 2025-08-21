# =========================
# Portfolio Simulator with Fees, FX, and ROI (Monthly shows ROI% curve only by default)
# - Fixed mode: one-time buy with fees (price-based portfolio + asset lines)
# - Monthly mode: DCA; main chart shows portfolio ROI% over time (money-weighted, net of fees)
#   NEW: optional overlay toggle to show asset price indices on a secondary right axis
# - Auto-detects asset type / currency / exchange; allows per-asset overrides
# - Two tables: Performance Details (always), Monthly DCA Metrics (only in Monthly mode)
# =========================

import streamlit as st          # UI framework to build the app
import yfinance as yf           # Market data downloader (prices, metadata) from Yahoo Finance
import pandas as pd             # Data analysis library (tables, time series)
import numpy as np              # Numerical helpers
import plotly.graph_objects as go  # Interactive plotting (Plotly)
from datetime import datetime, timedelta  # Date utilities

# =========================
# Page / Theme (visual style + small CSS helpers)
# - set_page_config: page title & layout
# - st.markdown with CSS: defines colors & styles used across the app
# =========================
st.set_page_config(page_title="Portfolio Simulator", layout="wide")

st.markdown("""
<style>
:root{
  --bg:#0f1115; --muted:#9aa3b2; --text:#f3f4f6; --accent:#2563eb;
}
html, body, [class*="css"] { background: var(--bg); color: var(--text); }
h1,h2,h3 { letter-spacing:.2px; } h1{font-weight:800;} h2{font-weight:700;} h3{font-weight:700;}
.help { color: var(--muted); }

/* Hero (clean, light) */
.hero {
  background: linear-gradient(180deg, #ffffff, #f6f7fb);
  color: #0f172a; border-radius: 18px; padding: 20px 24px; border: 1px solid #eaecef;
  box-shadow: 0 8px 28px rgba(15,17,26,0.15);
}
.hero h1 { margin: 0 0 6px 0; color:#0b1220; }
.hero .sub { color:#475569; }

/* Divider: thin horizontal separator */
.divider { width:100%; height:1px; background: rgba(226,232,240,0.25); margin: 12px 0 18px 0; }

/* Small "chips" showing per-asset performance under the legend */
.cards { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
.card {
  display:flex; align-items:center; gap:8px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.14);
  padding:6px 10px; border-radius:12px; font-size:13px;
}
.card .swatch { width:10px; height:10px; border-radius:999px; display:inline-block; }
.card .lbl { color:#e5e7eb; }
.card .val.pos { color:#10b981; font-weight:700; }
.card .val.neg { color:#ef4444; font-weight:700; }

/* Sidebar titles & helpers */
.sidebar-title { font-weight:700; font-size:1.05rem; margin:4px 0 8px 0; }
.sidebar-subtle { color:var(--muted); font-size:0.9rem; }

/* Primary buttons & slider accent color */
a, .stDownloadButton button, .stButton button { border-radius: 10px !important; }
.stButton button { background: var(--accent); color:white; border:0; }
.stSlider > div > div > div { background: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Plot colors & layout helpers (readability on dark theme)
# - PLOT_COLORS: color palette for traces
# - base_layout(): common Plotly layout shared by charts
# =========================
PLOT_COLORS = [
    "#2563eb", "#f59e0b", "#06b6d4", "#f472b6", "#84cc16",
    "#eab308", "#ef4444", "#10b981", "#a78bfa", "#f97316",
]
LEGEND_BG = "rgba(255,255,255,0.98)"
LEGEND_TEXT = "#0f172a"
LEGEND_BORDER = "#e2e8f0"
TITLE_TEXT = "#e5e7eb"
AXIS_TEXT = "#e5e7eb"
GRID_COLOR = "rgba(148,163,184,0.25)"

def base_layout(title_text=None, right_margin=220, height=640):
    """
    Returns a dict of Plotly layout properties for consistent styling.
    - title_text: optional chart title
    - right_margin: leave space for legend on the right
    - height: figure height in pixels
    """
    return dict(
        title=dict(text=title_text or "", font=dict(color=TITLE_TEXT, size=20, family="Inter, Arial, sans-serif"),
                   x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text="Date", font=dict(color=AXIS_TEXT)),
                   tickfont=dict(color=AXIS_TEXT), gridcolor=GRID_COLOR),
        yaxis=dict(title=dict(text="", font=dict(color=AXIS_TEXT)),
                   tickfont=dict(color=AXIS_TEXT), gridcolor=GRID_COLOR),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top",
                    bgcolor=LEGEND_BG, bordercolor=LEGEND_BORDER, borderwidth=1,
                    font=dict(color=LEGEND_TEXT, size=12)),
        margin=dict(r=right_margin, t=60, b=40, l=60),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

# =========================
# Currency helpers (symbols + FX detection + FX series)
# - sym_for(): pretty currency symbol for display
# - detect_currency(): get a ticker's native currency from yfinance
# - fetch_fx_series(): download FX rate series to convert prices to base currency
# =========================
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥", "HKD": "HK$",
    "CHF": "CHF", "CAD": "C$", "AUD": "A$", "NZD": "NZ$", "SEK": "kr",
    "NOK": "kr", "DKK": "kr", "INR": "₹", "SGD": "S$", "ZAR": "R",
    "TRY": "₺", "MXN": "Mex$", "BRL": "R$", "PLN": "zł"
}
def sym_for(code: str) -> str:
    """Return a friendly currency symbol (fallback to the code itself)."""
    return CURRENCY_SYMBOLS.get(code, code + " ")

@st.cache_data(show_spinner=False)
def detect_currency(ticker: str) -> str:
    """
    Try to detect a ticker's native currency using yfinance:
    1) fast_info.currency, else 2) info['currency'], else default 'USD'.
    """
    try:
        t = yf.Ticker(ticker)
        cur = getattr(t, "fast_info", None)
        if cur and getattr(cur, "currency", None): return str(cur.currency)
        info = getattr(t, "info", {}) or {}
        if info.get("currency"): return str(info["currency"])
    except Exception:
        pass
    return "USD"

@st.cache_data(show_spinner=False)
def fetch_fx_series(src: str, dst: str, start_dt: datetime, end_dt: datetime) -> pd.Series:
    """
    Download an FX conversion series converting from src->dst for the given date range.
    - First try direct pair (e.g., EURUSD=X).
    - If not found, try inverse (USD EUR).
    - If still not found, triangulate via USD.
    Returns a daily Series of conversion multipliers aligned to the date range.
    """
    if src == dst:
        idx = pd.date_range(start_dt, end_dt, freq="D")
        return pd.Series(1.0, index=idx)

    def _dl(pair):
        try:
            df = yf.download(pair, start=start_dt, end=end_dt, progress=False, auto_adjust=False)
            if "Close" in df and not df["Close"].empty: return df["Close"].rename(pair)
        except Exception:
            pass
        return None

    # Try direct pair
    direct = _dl(f"{src}{dst}=X")
    if direct is not None: return direct
    # Try inverse pair, invert values
    inv = _dl(f"{dst}{src}=X")
    if inv is not None:
        inv = inv.replace(0, np.nan)
        return 1.0 / inv

    # Triangulate via USD
    if src != "USD":
        src_usd = _dl(f"{src}USD=X") or _dl(f"USD{src}=X")
        if src_usd is not None and src_usd.name.startswith("USD"):
            src_usd = 1.0 / src_usd.replace(0, np.nan)
    else:
        src_usd = pd.Series(1.0, index=pd.date_range(start_dt, end_dt, freq="D"))

    if dst != "USD":
        usd_dst = _dl(f"USD{dst}=X") or _dl(f"{dst}USD=X")
        if usd_dst is not None and usd_dst.name.endswith("USD=X"):
            usd_dst = 1.0 / usd_dst.replace(0, np.nan)
    else:
        usd_dst = pd.Series(1.0, index=pd.date_range(start_dt, end_dt, freq="D"))

    if src_usd is not None and usd_dst is not None:
        fx = pd.concat([src_usd, usd_dst], axis=1).ffill().bfill()
        return (fx.iloc[:, 0] * fx.iloc[:, 1]).rename(f"{src}->{dst}")

    # Fallback: identity (no conversion)
    idx = pd.date_range(start_dt, end_dt, freq="D")
    return pd.Series(1.0, index=idx)

# =========================
# Asset metadata helpers (type + exchange detection and mapping)
# - ASSET_TYPES/EXCHANGES: allowed UI choices
# - map_exchange(): normalize yfinance exchange names into friendly buckets
# - detect_metadata(): best-effort guess of asset type & exchange
# =========================
ASSET_TYPES = ["AUTO", "EQUITY", "ETF", "COMMODITY"]
EXCHANGES = ["AUTO", "Euronext", "US (NYSE/Nasdaq)", "LSE", "XETRA"]

def map_exchange(info_exchange: str) -> str:
    """
    Map raw yfinance exchange/market string into one of our standard buckets
    (Euronext / US / LSE / XETRA). Default to US if unknown.
    """
    if not info_exchange:
        return "US (NYSE/Nasdaq)"
    s = info_exchange.upper()
    if "EURONEXT" in s or "BRU" in s or "PAR" in s or "AMS" in s:
        return "Euronext"
    if "XETRA" in s or "GER" in s or "DEU" in s:
        return "XETRA"
    if "LSE" in s or "LON" in s or "LSEIOB" in s:
        return "LSE"
    return "US (NYSE/Nasdaq)"

@st.cache_data(show_spinner=False)
def detect_metadata(ticker: str):
    """
    Best-effort detection of (asset_type, exchange) from yfinance.
    Returns ('EQUITY' or 'ETF' or 'COMMODITY', exchange_bucket).
    Falls back to 'EQUITY' and 'US (NYSE/Nasdaq)' if unknown.
    """
    a_type, exch = "EQUITY", "US (NYSE/Nasdaq)"
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
        qtype = str(info.get("quoteType", "")).upper()
        if qtype in {"EQUITY","ETF","FUND"}:
            a_type = "ETF" if qtype in {"ETF","FUND"} else "EQUITY"
        elif qtype in {"COMMODITY","FUTURE"}:
            a_type = "COMMODITY"
        else:
            a_type = "EQUITY"
        exch_raw = info.get("exchange") or info.get("market") or ""
        exch = map_exchange(str(exch_raw))
    except Exception:
        pass
    return a_type, exch

# =========================
# Broker fee model (simplified, approximate)
# - BROKERS: selectable brokers in the UI
# - BROKER_FEES: nested dict with flat fee, percentage fee, and FX markup per exchange & asset_type
# - fee_for(): compute fee (in base currency) for a trade value
# =========================
BROKERS = ["Bolero", "DEGIRO", "Saxo", "Interactive Brokers", "MEXEM"]

BROKER_FEES = {
    # NOTE: These are approximate placeholders for demo purposes.
    "Bolero": {
        "Euronext": {"EQUITY": {"flat": 7.5,  "percent": 0.0,  "fx_pct": 0.0},
                     "ETF":    {"flat": 5.0,  "percent": 0.0,  "fx_pct": 0.0},
                     "COMMODITY": {"flat": 10.0, "percent": 0.0, "fx_pct": 0.0}},
        "US (NYSE/Nasdaq)": {"EQUITY": {"flat": 15.0, "percent": 0.0,  "fx_pct": 0.0},
                             "ETF":    {"flat": 15.0, "percent": 0.0,  "fx_pct": 0.0}},
        "LSE": {"EQUITY": {"flat": 12.0, "percent": 0.0, "fx_pct": 0.0},
                "ETF":    {"flat": 12.0, "percent": 0.0, "fx_pct": 0.0}},
        "XETRA":{"EQUITY": {"flat": 12.0, "percent": 0.0, "fx_pct": 0.0},
                 "ETF":    {"flat": 12.0, "percent": 0.0, "fx_pct": 0.0}},
    },
    "DEGIRO": {
        "Euronext": {"EQUITY": {"flat": 3.0,  "percent": 0.0,  "fx_pct": 0.0},
                     "ETF":    {"flat": 1.0,  "percent": 0.0,  "fx_pct": 0.0}}, # core ETFs ~€1 handling
        "US (NYSE/Nasdaq)": {"EQUITY": {"flat": 1.0,  "percent": 0.004, "fx_pct": 0.0},
                             "ETF":    {"flat": 3.0,  "percent": 0.0,   "fx_pct": 0.0}},
        "LSE": {"EQUITY": {"flat": 2.0, "percent": 0.0, "fx_pct": 0.0},
                "ETF":    {"flat": 2.0, "percent": 0.0, "fx_pct": 0.0}},
        "XETRA":{"EQUITY": {"flat": 3.0, "percent": 0.0, "fx_pct": 0.0},
                 "ETF":    {"flat": 3.0, "percent": 0.0, "fx_pct": 0.0}},
    },
    "Saxo": {
        "Euronext": {"EQUITY": {"flat": 3.0, "percent": 0.0, "fx_pct": 0.25},
                     "ETF":    {"flat": 3.0, "percent": 0.0, "fx_pct": 0.25}},
        "US (NYSE/Nasdaq)": {"EQUITY": {"flat": 1.0, "percent": 0.0, "fx_pct": 0.25},
                             "ETF":    {"flat": 1.0, "percent": 0.0, "fx_pct": 0.25}},
        "LSE": {"EQUITY": {"flat": 8.0, "percent": 0.0, "fx_pct": 0.25},
                "ETF":    {"flat": 8.0, "percent": 0.0, "fx_pct": 0.25}},
        "XETRA":{"EQUITY": {"flat": 5.0, "percent": 0.0, "fx_pct": 0.25},
                 "ETF":    {"flat": 5.0, "percent": 0.0, "fx_pct": 0.25}},
    },
    "Interactive Brokers": {
        "Euronext": {"EQUITY": {"flat": 1.25, "percent": 0.05, "fx_pct": 0.0},
                     "ETF":    {"flat": 1.25, "percent": 0.05, "fx_pct": 0.0}},
        "US (NYSE/Nasdaq)": {"EQUITY": {"flat": 1.0, "percent": 0.0035, "fx_pct": 0.0},
                             "ETF":    {"flat": 1.0, "percent": 0.0035, "fx_pct": 0.0}},
        "LSE": {"EQUITY": {"flat": 1.25, "percent": 0.05, "fx_pct": 0.0},
                "ETF":    {"flat": 1.25, "percent": 0.05, "fx_pct": 0.0}},
        "XETRA":{"EQUITY": {"flat": 1.25, "percent": 0.05, "fx_pct": 0.0},
                 "ETF":    {"flat": 1.25, "percent": 0.05, "fx_pct": 0.0}},
    },
    "MEXEM": {
        "Euronext": {"EQUITY": {"flat": 1.0, "percent": 0.05, "fx_pct": 0.0},
                     "ETF":    {"flat": 1.0, "percent": 0.05, "fx_pct": 0.0}},
        "US (NYSE/Nasdaq)": {"EQUITY": {"flat": 1.0, "percent": 0.0035, "fx_pct": 0.0},
                             "ETF":    {"flat": 1.0, "percent": 0.0035, "fx_pct": 0.0}},
        "LSE": {"EQUITY": {"flat": 1.0, "percent": 0.05, "fx_pct": 0.0},
                "ETF":    {"flat": 1.0, "percent": 0.05, "fx_pct": 0.0}},
        "XETRA":{"EQUITY": {"flat": 1.0, "percent": 0.05, "fx_pct": 0.0}},
    },
}

def fee_for(broker: str, exchange: str, asset_type: str, trade_value_base: float,
            asset_ccy: str, base_ccy: str) -> float:
    """
    Compute per-order fee in base currency for a given broker/exchange/asset_type.
    fee = flat + percent * trade_value + (optional FX % if asset currency != base currency)
    """
    if broker not in BROKER_FEES:
        return 0.0
    ex = BROKER_FEES[broker].get(exchange, {})
    at = ex.get(asset_type, ex.get("EQUITY", {}))
    flat = float(at.get("flat", 0.0))
    pct  = float(at.get("percent", 0.0))
    fx_pct = float(at.get("fx_pct", 0.0)) if asset_ccy != base_ccy else 0.0
    fee = flat + (pct/100.0)*trade_value_base + (fx_pct/100.0)*trade_value_base
    return max(fee, 0.0)

# =========================
# Risk/Perf helpers
# - annualized_sharpe(): risk-adjusted return using daily series (~252 trading days)
# - compute_drawdown(): running drawdown of a cumulative index (peak-to-trough)
# - annualized_from_prices(): convert total growth to annualized growth
# =========================
def annualized_sharpe(daily_returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    """
    Annualized Sharpe ratio using daily returns:
    sqrt(252) * mean(excess) / std
    """
    rets = daily_returns.dropna()
    if rets.std(ddof=0) == 0 or len(rets) < 2: return float("nan")
    excess = rets - risk_free_daily
    return float(np.sqrt(252) * excess.mean() / excess.std(ddof=0))

def compute_drawdown(cum: pd.Series) -> pd.Series:
    """Compute drawdown: current / running_max - 1."""
    return cum / cum.cummax() - 1.0

def annualized_from_prices(start_val: float, end_val: float, years: float) -> float:
    """
    Annualize a growth factor over 'years':
    (end / start)^(1/years) - 1
    Returns NaN if inputs invalid.
    """
    if years is None or years <= 0 or start_val is None or end_val is None:
        return float("nan")
    if np.isnan(start_val) or np.isnan(end_val) or start_val <= 0:
        return float("nan")
    return (end_val / start_val) ** (1.0 / years) - 1.0

# =========================
# DCA helper
# - month_start_dates(): pick the first trading day in each calendar month from an index
# =========================
def month_start_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the first available date per month (based on the provided trading index)."""
    if len(index) == 0:
        return index
    df = pd.DataFrame(index=index)
    firsts = df.groupby([index.year, index.month]).apply(lambda x: x.index.min())
    return pd.DatetimeIndex(firsts.values)

# =========================
# Hero (header): pretty page header at the top of the app
# =========================
st.markdown(
    """
<div class="hero">
  <h1>Portfolio Simulator</h1>
  <div class="sub">FX-converted performance, money-weighted ROI (Monthly), and broker fees.</div>
</div>
""",
    unsafe_allow_html=True
)

# =========================
# State (default assets)
# - st.session_state.assets: keeps asset list between UI interactions
# =========================
if "assets" not in st.session_state:
    st.session_state.assets = [
        {"ticker": "AAPL", "weight": 40.0},
        {"ticker": "MSFT", "weight": 40.0},
        {"ticker": "GLD",  "weight": 20.0},
    ]

# =========================
# Sidebar (inputs: assets, dates, currency, broker/fees, mode)
# - Lets the user configure portfolio, timeframe, base currency, broker, and investing mode
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Portfolio Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtle">Per-asset row. Auto-normalize keeps total at 100%.</div>', unsafe_allow_html=True)

    # Number of assets (adds/removes rows)
    n_assets = st.number_input("Number of assets", 1, 20, value=len(st.session_state.assets), step=1)
    if n_assets > len(st.session_state.assets):
        st.session_state.assets += [{"ticker": "", "weight": 0.0} for _ in range(n_assets - len(st.session_state.assets))]
    elif n_assets < len(st.session_state.assets):
        st.session_state.assets = st.session_state.assets[:n_assets]

    # Auto-normalize toggles re-scaling weights to sum to 100%
    auto_normalize = st.toggle("Auto-normalize to 100%", value=True)

    st.markdown("**Assets**")
    # Optional advanced controls to override detected asset type/exchange
    show_advanced = st.checkbox("Advanced per-asset overrides (asset type & exchange)", value=False)
    for i in range(n_assets):
        c1, c2 = st.columns([2, 1], gap="small")
        with c1:
            # Ticker input (uppercased/trimmed)
            st.session_state.assets[i]["ticker"] = st.text_input(
                f"Ticker {i+1}", value=st.session_state.assets[i]["ticker"], key=f"ticker_{i}"
            ).upper().strip()
        with c2:
            # Weight input (percentage)
            st.session_state.assets[i]["weight"] = st.number_input(
                f"Weight {i+1} (%)", 0.0, 100.0, value=float(st.session_state.assets[i]["weight"]),
                step=1.0, key=f"weight_{i}"
            )

        # Advanced overrides (if enabled)
        if show_advanced and st.session_state.assets[i]["ticker"]:
            detected_type, detected_ex = detect_metadata(st.session_state.assets[i]["ticker"])
            st.session_state.assets[i]["asset_type"] = st.selectbox(
                f"Asset type {i+1}", ASSET_TYPES, index=ASSET_TYPES.index(detected_type) if detected_type in ASSET_TYPES else 0,
                key=f"type_{i}"
            )
            st.session_state.assets[i]["exchange"] = st.selectbox(
                f"Exchange {i+1}", EXCHANGES, index=EXCHANGES.index(detected_ex) if detected_ex in EXCHANGES else 0,
                key=f"ex_{i}"
            )
        else:
            # AUTO means: let the app detect it
            st.session_state.assets[i]["asset_type"] = "AUTO"
            st.session_state.assets[i]["exchange"] = "AUTO"

    # If auto-normalize, rescale all weights to sum exactly 100%
    if auto_normalize:
        total = sum(a["weight"] for a in st.session_state.assets) or 1.0
        for a in st.session_state.assets:
            a["weight"] = round(a["weight"] * 100.0 / total, 4)

    # Date range slider
    today = datetime.today().date()
    default_start = (datetime.today() - timedelta(days=365)).date()
    min_slider_date = (datetime.today() - timedelta(days=365*15)).date()
    date_start, date_end = st.slider(
        "Historical date range", min_value=min_slider_date, max_value=today,
        value=(default_start, today), format="YYYY-MM-DD"
    )

    # Base currency (for conversions) and broker (for fees)
    BASE_CHOICES = ["USD","EUR","GBP","CHF","JPY","CAD","AUD","NZD","SEK","NOK","DKK","INR","SGD","HKD","ZAR","TRY","MXN","BRL","PLN","CNY"]
    base_currency = st.selectbox("Base currency", BASE_CHOICES, index=1)  # default EUR
    broker = st.selectbox("Broker (fees applied to buys)", BROKERS, index=1)

    # Amount & investing mode
    amount = st.number_input(f"Amount ({base_currency})", min_value=0.0, value=1000.0, step=100.0)
    invest_mode = st.radio("Mode", options=["Fixed", "Monthly"], horizontal=True,
                           help="Fixed: one lump-sum buy (fees once). Monthly: DCA; chart shows portfolio ROI% over time (net of fees).")
    overlay_assets = st.toggle("Overlay asset price lines (Monthly)", value=False,
                               help="When Monthly is selected, optionally show asset price indices (right axis) under the ROI% curve.")

    st.markdown("---")
    run_click = st.button("Run")  # triggers computation

# =========================
# Main computation & plotting
# - After pressing "Run", we:
#   1) validate inputs
#   2) download prices
#   3) FX-convert to base currency
#   4) compute returns, portfolio, fees, and ROI
#   5) build charts & tables
# =========================
if run_click:
    # ---- Collect sanitized inputs
    tickers = [a["ticker"] for a in st.session_state.assets if a["ticker"]]
    weights = [float(a["weight"]) for a in st.session_state.assets if a["ticker"]]

    # Basic validation
    if len(tickers) != len(weights) or len(tickers) == 0:
        st.error("Please enter a valid set of tickers and weights.")
    else:
        total_w = round(sum(weights), 6)
        if abs(total_w - 100.0) > 1e-3:
            st.warning(f"Weights sum to {total_w}%, not 100%. (Enable Auto-normalize for a clean 100%.)")

        # Build date boundaries and display currency symbol
        start_dt = datetime.combine(date_start, datetime.min.time())
        end_dt   = datetime.combine(date_end,   datetime.min.time())
        sym = sym_for(base_currency)

        # ---- Detect per-asset metadata (type, exchange, currency)
        currencies = {}   # map ticker -> native currency (e.g., 'USD')
        asset_types = []  # parallel list to tickers (e.g., 'EQUITY', 'ETF')
        exchanges  = []   # parallel list to tickers (e.g., 'Euronext')

        for a in st.session_state.assets:
            if not a["ticker"]:
                continue
            t = a["ticker"]
            currencies[t] = detect_currency(t)
            # Asset type: override if user chose one; else detect
            if a.get("asset_type") and a["asset_type"] != "AUTO":
                a_type = a["asset_type"]
            else:
                a_type, _ex = detect_metadata(t)
            asset_types.append(a_type if a_type in ASSET_TYPES else "EQUITY")
            # Exchange: override if user chose one; else detect
            if a.get("exchange") and a["exchange"] != "AUTO":
                exch = a["exchange"]
            else:
                _t, detected_ex = detect_metadata(t)
                exch = detected_ex
            exchanges.append(exch if exch in EXCHANGES else "US (NYSE/Nasdaq)")

        # ---- Download prices (native CCY), then convert to base CCY
        raw_price = pd.DataFrame()  # rows=dates, cols=tickers, values=close in native CCY
        for t in tickers:
            try:
                df = yf.download(t, start=start_dt, end=end_dt, progress=False, auto_adjust=True)
                if "Close" in df and not df["Close"].empty:
                    raw_price[t] = df["Close"]
                else:
                    st.warning(f"No price data for {t}. Skipping.")
            except Exception as e:
                st.warning(f"{t} error: {e}")

        if raw_price.empty or len(raw_price) < 2:
            st.error("No valid data was retrieved for the given tickers/date range.")
        else:
            # FX conversion to base currency (e.g., EUR)
            price_base = pd.DataFrame(index=raw_price.index)
            for t in tickers:
                fx = fetch_fx_series(currencies.get(t, "USD"), base_currency, start_dt, end_dt)
                price_base[t] = raw_price[t] * fx.reindex(raw_price.index).ffill().bfill()

            # ---- Price-based returns (used for asset lines, Sharpe, annualized)
            rets      = price_base.pct_change().fillna(0.0)                 # daily % returns per asset
            weighted  = rets.mul([w/100.0 for w in weights], axis=1)        # weighted daily returns
            port_rets = weighted.sum(axis=1)                                # daily portfolio return
            port_cum  = (1.0 + port_rets).cumprod()                         # portfolio index (base=1)
            asset_cum = (1.0 + rets).cumprod()                              # per-asset indices (base=1)

            years = (price_base.index[-1] - price_base.index[0]).days / 365.25 if len(price_base.index) >= 2 else np.nan

            # ---- Fees + investment math (Fixed vs Monthly)
            start_amount_port  = float("nan")            # total contributions (excl. fees)
            fees_total_port    = 0.0                     # total fees paid
            end_amount_port    = float("nan")            # end portfolio market value
            roi_port_pct       = float("nan")            # ROI % net of fees
            asset_fees         = {t: 0.0 for t in tickers}  # fee tracker per asset

            # For Monthly metrics table (DCA):
            invest_dates = None                          # dates we invest (first trading day per month)
            units_per_asset = {t: 0.0 for t in tickers}  # total units accumulated per asset
            net_contrib_per_asset = {t: 0.0 for t in tickers}   # net contributions (after fees)
            gross_contrib_per_asset = {t: 0.0 for t in tickers} # gross contributions (before fees)
            fees_per_asset = {t: 0.0 for t in tickers}          # total fees per asset

            roi_pct_series = None  # time series (Monthly mode): ROI% over time

            if invest_mode == "Fixed":
                # ----- Fixed mode: spend 'amount' once at start date, pay fees once
                start_idx = price_base.index[0]
                start_amount_port = float(amount)

                for t, w, a_type, exch in zip(tickers, weights, asset_types, exchanges):
                    alloc = start_amount_port * (w/100.0)  # money to this asset
                    fee   = fee_for(broker, exch, a_type, alloc, currencies[t], base_currency)
                    asset_fees[t] += fee
                    fees_per_asset[t] += fee
                    fees_total_port += fee
                    net_alloc = max(alloc - fee, 0.0)      # money left to buy shares
                    gross_contrib_per_asset[t] += alloc
                    net_contrib_per_asset[t]  += net_alloc
                    p0 = float(price_base.loc[start_idx, t])
                    units_per_asset[t] = (net_alloc / p0) if p0 > 0 else 0.0

                last_row = price_base.iloc[-1]
                # Market value at the end = units * last price (sum over assets)
                end_amount_port = sum(units_per_asset[t] * float(last_row.get(t, np.nan)) for t in tickers)

                start_cash_out = start_amount_port + fees_total_port  # total money spent incl. fees
                roi_port_pct = ((end_amount_port - start_cash_out) / start_cash_out * 100.0) if start_cash_out > 0 else float("nan")

            else:
                # =========================
                # Monthly DCA — CORRECT ROI SERIES
                # - Invest 'amount' split by weights each month (first trading day).
                # - Track units AFTER each buy date (cumulative).
                # - cash_out_daily: contributions + that day's fees (sum over assets).
                # - ROI_t = (PortfolioValue_t / CumulativeCashOut_t - 1) * 100; first buy ROI set to 0%.
                # =========================
                invest_dates = month_start_dates(price_base.index)
                n_months = len(invest_dates)
                start_amount_port = float(amount) * n_months  # contributions only (no fees)

                # Prepare daily trackers
                units_daily = pd.DataFrame(0.0, index=price_base.index, columns=tickers)  # units held each day (built via cumsum)
                cash_out_daily = pd.Series(0.0, index=price_base.index)                   # contributions + fees per day

                for d in invest_dates:
                    day_fees_total = 0.0
                    buy_units_row = {t: 0.0 for t in tickers}

                    for t, w, a_type, exch in zip(tickers, weights, asset_types, exchanges):
                        alloc = float(amount) * (w/100.0)  # this day's gross allocation for asset t
                        fee   = fee_for(broker, exch, a_type, alloc, currencies[t], base_currency)
                        day_fees_total += fee
                        fees_per_asset[t] += fee
                        gross_contrib_per_asset[t] += alloc
                        net_alloc = max(alloc - fee, 0.0)
                        net_contrib_per_asset[t]  += net_alloc

                        px = price_base.loc[d, t] if d in price_base.index else np.nan
                        if pd.notnull(px) and px > 0:
                            buy_units_row[t] = net_alloc / float(px)
                            units_per_asset[t] += buy_units_row[t]  # running total for final tables

                    # add today's purchased units; cumulative sum later will propagate holdings forward
                    for t in tickers:
                        units_daily.loc[d, t] += buy_units_row[t]

                    # Today we spent 'amount' (split by weights) + total fees
                    cash_out_daily.loc[d] = float(amount) + day_fees_total

                # From "units bought on day" -> "units held each day"
                units_daily = units_daily.cumsum()

                # Portfolio value each day = sum(units * price)
                V_daily = (units_daily * price_base[tickers]).sum(axis=1)

                # Cumulative cash-out (contrib + fees)
                cash_cum = cash_out_daily.cumsum()
                # Total fees = total cash out - pure contributions
                fees_total_port = float(cash_out_daily.sum() - (amount * n_months))

                # ROI series (money-weighted)
                roi_pct_series = (V_daily / cash_cum - 1.0) * 100.0
                roi_pct_series[cash_cum <= 0] = np.nan
                if n_months > 0:
                    first_d = invest_dates[0]
                    if first_d in roi_pct_series.index:
                        roi_pct_series.loc[first_d] = 0.0  # start at 0% on first investment day

                end_amount_port = float(V_daily.iloc[-1])
                start_cash_out  = float(cash_cum.iloc[-1])
                roi_port_pct    = ((end_amount_port - start_cash_out) / start_cash_out * 100.0) if start_cash_out > 0 else float("nan")

            # =========================
            # CHART — Portfolio & Assets
            # - Fixed: show price indices (portfolio + assets)
            # - Monthly: show portfolio ROI% (left axis); optional asset price indices on right axis
            # =========================
            st.markdown("### Portfolio Performance")
            mode_note = " · Mode: <b>Monthly (ROI %)</b>" if (invest_mode == "Monthly") else " · Mode: <b>Fixed (Price)</b>"
            st.markdown(
                f'<div class="divider"></div><div style="color:#cbd5e1;font-size:13px;">'
                f'Base currency: <b>{base_currency}</b> · Range: <b>{date_start} → {date_end}</b>{mode_note} · Broker: <b>{broker}</b>'
                f'</div>',
                unsafe_allow_html=True
            )

            fig = go.Figure()
            if invest_mode == "Monthly" and roi_pct_series is not None and roi_pct_series.notna().sum() > 0:
                # (Left) Portfolio ROI% curve (net of fees)
                fig.add_trace(go.Scatter(
                    x=roi_pct_series.index, y=roi_pct_series.values, mode="lines",
                    name="PORTFOLIO ROI (net, %)", line=dict(width=4, color=PLOT_COLORS[0])
                ))

                # Optional overlay of asset price indices (right axis) for context
                right_margin = 60
                if overlay_assets:
                    for idx, t in enumerate(asset_cum.columns):
                        fig.add_trace(go.Scatter(
                            x=asset_cum.index, y=asset_cum[t].values, mode="lines",
                            name=f"{t} (Price idx)", opacity=0.6,
                            line=dict(color=PLOT_COLORS[(idx+1) % len(PLOT_COLORS)]),
                            yaxis="y2"
                        ))
                    right_margin = 220  # space for legend

                    # Base layout + both axes
                    layout_main = base_layout(title_text="", height=640, right_margin=right_margin)
                    fig.update_layout(**layout_main)
                    fig.update_layout(
                        yaxis=dict(
                            title=dict(text="ROI (%)", font=dict(color=AXIS_TEXT)),
                            tickfont=dict(color=AXIS_TEXT),
                            gridcolor=GRID_COLOR,
                            ticksuffix="%"
                        ),
                        yaxis2=dict(
                            title=dict(text="Price index (base=1)", font=dict(color=AXIS_TEXT)),
                            tickfont=dict(color=AXIS_TEXT),
                            overlaying="y",
                            side="right",
                            showgrid=False
                        )
                    )
                else:
                    layout_main = base_layout(title_text="", height=640, right_margin=60)
                    fig.update_layout(**layout_main)
                    fig.update_layout(
                        yaxis=dict(
                            title=dict(text="ROI (%)", font=dict(color=AXIS_TEXT)),
                            tickfont=dict(color=AXIS_TEXT),
                            gridcolor=GRID_COLOR,
                            ticksuffix="%"
                        )
                    )

            else:
                # Fixed mode: show price indices (portfolio + assets)
                fig.add_trace(go.Scatter(
                    x=port_cum.index, y=port_cum.values, mode="lines",
                    name="PORTFOLIO (Price index)", line=dict(width=4, color=PLOT_COLORS[0])
                ))
                for idx, t in enumerate(asset_cum.columns):
                    fig.add_trace(go.Scatter(
                        x=asset_cum.index, y=asset_cum[t].values, mode="lines",
                        name=t, opacity=0.9, line=dict(color=PLOT_COLORS[(idx+1) % len(PLOT_COLORS)])
                    ))
                layout_main = base_layout(title_text="Growth (Base = 1)", height=640, right_margin=220)
                fig.update_layout(**layout_main)
                fig.update_layout(
                    yaxis=dict(
                        title=dict(text="Cumulative Return", font=dict(color=AXIS_TEXT)),
                        tickfont=dict(color=AXIS_TEXT),
                        gridcolor=GRID_COLOR
                    )
                )

            st.plotly_chart(fig, use_container_width=True)

            # =========================
            # SUMMARY — Amounts & ROI (net of fees)
            # - Quick recap under the chart
            # =========================
            st.markdown(
                f"<div style='margin-top:6px;color:#cbd5e1;font-size:13px;'>"
                f"Contributions: <b>{sym}{start_amount_port:,.2f}</b> · "
                f"Fees Paid: <b>{sym}{fees_total_port:,.2f}</b> · "
                f"End Value: <b>{sym}{end_amount_port:,.2f}</b> · "
                f"ROI (net): <b>{roi_port_pct:.2f}%</b>"
                f"</div>",
                unsafe_allow_html=True
            )

            # =========================
            # CHIPS — Reference price performance per asset (not fee-adjusted)
            # - These are pure price moves over the selected period
            # =========================
            perf_cards = []
            port_perf = float((port_cum.iloc[-1] - 1.0) * 100.0)
            perf_cards.append(("PORTFOLIO (Price)", port_perf, PLOT_COLORS[0]))
            for i, t in enumerate(tickers):
                s = price_base[t].dropna()
                perf = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) * 100.0 if len(s) > 1 else float("nan")
                perf_cards.append((t, perf, PLOT_COLORS[(i+1) % len(PLOT_COLORS)]))

            chips_html = ['<div class="cards">']
            for label, v, color in perf_cards:
                cls = "pos" if pd.notnull(v) and v >= 0 else "neg"
                val = "—" if pd.isna(v) else f"{v:.2f}%"
                chips_html.append(
                    f'<div class="card"><span class="swatch" style="background:{color};"></span>'
                    f'<span class="lbl">{label}</span><span class="val {cls}">{val}</span></div>'
                )
            chips_html.append('</div>')
            st.markdown("\n".join(chips_html), unsafe_allow_html=True)

            # =========================
            # TABLE 1 — Performance Details (Converted to Base CCY, net of fees)
            # - Shows price performance, annualized % (price-based), and ROI% (net of fees)
            # - Portfolio row: in Monthly mode, Performance/Annualized are price-reference only; ROI is money-weighted
            # =========================
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### Performance Details (Net of Fees)")

            rows = []

            # ---- Portfolio row
            if invest_mode == "Monthly":
                rows.append({
                    "Ticker": "PORTFOLIO", "Native CCY": "—", "Currency": base_currency, "Weight %": 100.0,
                    "Start Price": 1.00, "End Price": float("nan"),  # not meaningful for ROI in Monthly
                    "Performance %": float((port_cum.iloc[-1]-1.0)*100.0),  # price-based reference
                    "Annualized %": float(annualized_from_prices(1.0, float(port_cum.iloc[-1]), years) * 100.0) if pd.notnull(port_cum.iloc[-1]) else float("nan"),
                    "ROI % (net)": roi_port_pct,                         # money-weighted
                    "Contributions": start_amount_port,
                    "Fees Paid": fees_total_port,
                    "End Amount": end_amount_port,
                    "Sharpe": annualized_sharpe(port_rets)
                })
            else:
                rows.append({
                    "Ticker": "PORTFOLIO", "Native CCY": "—", "Currency": base_currency, "Weight %": 100.0,
                    "Start Price": 1.00, "End Price": float(port_cum.iloc[-1]),
                    "Performance %": float((port_cum.iloc[-1]-1.0)*100.0),
                    "Annualized %": float(annualized_from_prices(1.0, float(port_cum.iloc[-1]), years) * 100.0),
                    "ROI % (net)": roi_port_pct,                         # lump-sum ROI incl. fees
                    "Contributions": float(amount),
                    "Fees Paid": fees_total_port,
                    "End Amount": end_amount_port,
                    "Sharpe": annualized_sharpe(port_rets)
                })

            # ---- Per-asset rows
            for t, w in zip(tickers, weights):
                s = price_base[t].dropna()
                if len(s) > 1:
                    start_p, end_p = float(s.iloc[0]), float(s.iloc[-1])
                    perf = (end_p/start_p - 1.0) * 100.0
                    ann  = annualized_from_prices(start_p, end_p, years)
                else:
                    start_p = end_p = perf = float("nan")
                    ann = float("nan")

                if invest_mode == "Fixed":
                    # Lump-sum allocation for this asset
                    alloc = float(amount) * (w/100.0)
                    fee   = BROKER_FEES and fee_for(broker, exchanges[tickers.index(t)], asset_types[tickers.index(t)], alloc, currencies[t], base_currency) or 0.0
                    end_amt = units_per_asset[t] * float(price_base.iloc[-1].get(t, np.nan))
                    base_cash = alloc + fee
                    roi_t = ((end_amt - base_cash) / base_cash * 100.0) if base_cash > 0 and pd.notnull(end_amt) else float("nan")
                    contrib = alloc
                else:
                    # Monthly DCA contribution for this asset (across months)
                    invest_dates_local = month_start_dates(price_base.index)
                    contrib = float(amount) * len(invest_dates_local) * (w/100.0)
                    # fees_per_asset[t] already accumulated during Monthly loop
                    end_amt = units_per_asset[t] * float(price_base.iloc[-1].get(t, np.nan))
                    base_cash = contrib + fees_per_asset[t]
                    roi_t = ((end_amt - base_cash) / base_cash * 100.0) if base_cash > 0 and pd.notnull(end_amt) else float("nan")

                rows.append({
                    "Ticker": t,
                    "Native CCY": currencies.get(t, "USD"),
                    "Currency": base_currency,
                    "Weight %": float(w),
                    "Start Price": start_p,
                    "End Price": end_p,
                    "Performance %": perf,
                    "Annualized %": float(ann * 100.0) if pd.notnull(ann) else float("nan"),
                    "ROI % (net)": roi_t,
                    "Contributions": contrib,
                    "Fees Paid": fees_per_asset[t],
                    "End Amount": end_amt,
                    "Sharpe": annualized_sharpe(rets[t])
                })

            metrics_df = pd.DataFrame(rows).set_index("Ticker")[
                ["Native CCY","Currency","Weight %","Start Price","End Price","Performance %","Annualized %","ROI % (net)","Contributions","Fees Paid","End Amount","Sharpe"]
            ]

            # Pretty formatting for Table 1
            show_df = metrics_df.copy()
            show_df["Weight %"]        = show_df["Weight %"].map(lambda x: f"{x:.2f}%")
            show_df["Start Price"]     = show_df["Start Price"].map(lambda x: f"{sym}{x:,.2f}" if pd.notnull(x) else "—")
            show_df["End Price"]       = show_df["End Price"].map(lambda x: f"{sym}{x:,.2f}" if pd.notnull(x) else "—")
            for col in ["Performance %", "Annualized %", "ROI % (net)"]:
                if col in show_df.columns:
                    show_df[col] = show_df[col].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "—")
            show_df["Contributions"]   = show_df["Contributions"].map(lambda x: f"{sym}{x:,.2f}" if pd.notnull(x) else "—")
            show_df["Fees Paid"]       = show_df["Fees Paid"].map(lambda x: f"{sym}{x:,.2f}" if pd.notnull(x) else "—")
            show_df["End Amount"]      = show_df["End Amount"].map(lambda x: f"{sym}{x:,.2f}" if pd.notnull(x) else "—")
            show_df["Sharpe"]          = show_df["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "—")

            col_config = {
                "Native CCY":     st.column_config.Column(width=90, help="Asset’s original currency"),
                "Currency":       st.column_config.Column(width=90, help="Conversion target (base)"),
                "Weight %":       st.column_config.Column(width=90),
                "Start Price":    st.column_config.Column(width=120),
                "End Price":      st.column_config.Column(width=120),
                "Performance %":  st.column_config.Column(width=120),
                "Annualized %":   st.column_config.Column(width=130, help="Geometric annualized return (price-based)"),
                "ROI % (net)":    st.column_config.Column(width=110, help="(End - Contributions - Fees) / (Contributions + Fees)"),
                "Contributions":  st.column_config.Column(width=130),
                "Fees Paid":      st.column_config.Column(width=120),
                "End Amount":     st.column_config.Column(width=130),
                "Sharpe":         st.column_config.Column(width=80),
            }
            st.dataframe(show_df, use_container_width=True, height=560, column_config=col_config)

            # =========================
            # TABLE 2 — Monthly Investing: per-asset DCA metrics (only in Monthly)
            # - Extra insights useful for DCA:
            #   Avg Cost, Premium/Discount, Break-even month, Underwater streak,
            #   Contribution efficiency, TWR (price-based), monthly volatility
            # =========================
            if invest_mode == "Monthly":
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("### Monthly Investing — Per-Asset DCA Metrics")

                last_prices = price_base.iloc[-1]           # last available price per asset
                monthly_prices = price_base.resample("M").last()  # month-end prices
                per_asset_rows = []
                invest_dates_local = month_start_dates(price_base.index)

                for t, w in zip(tickers, weights):
                    cur_price = float(last_prices.get(t, np.nan))
                    units = units_per_asset.get(t, 0.0)
                    net_contrib = net_contrib_per_asset.get(t, 0.0)
                    fee_total = fees_per_asset.get(t, 0.0)

                    # Avg cost = (net contributions) / (units)
                    avg_cost = (net_contrib / units) if (units and units > 0) else np.nan
                    prem = (cur_price/avg_cost - 1.0) * 100.0 if (pd.notnull(cur_price) and pd.notnull(avg_cost) and avg_cost>0) else np.nan

                    # Break-even tracking on invest dates (first month where value >= contributions+fees)
                    break_month_str = "—"
                    underwater_streak = 0
                    max_underwater = 0
                    units_so_far = 0.0
                    contrib_so_far = 0.0
                    fees_so_far = 0.0

                    for d in invest_dates_local:
                        alloc = float(amount) * (w/100.0)  # this month's gross contribution to asset t
                        fee   = fee_for(broker, exchanges[tickers.index(t)], asset_types[tickers.index(t)],
                                        alloc, currencies[t], base_currency)
                        px = price_base.loc[d, t] if d in price_base.index else np.nan
                        if pd.notnull(px) and px > 0:
                            net_alloc = max(alloc - fee, 0.0)
                            units_so_far += net_alloc / float(px)
                            contrib_so_far += alloc
                            fees_so_far    += fee

                        # value vs cash-out so far
                        if d in price_base.index and pd.notnull(price_base.loc[d, t]):
                            val = units_so_far * float(price_base.loc[d, t])
                        else:
                            val = np.nan
                        base_cash = contrib_so_far + fees_so_far
                        if pd.notnull(val) and base_cash > 0:
                            if val >= base_cash and break_month_str == "—":
                                break_month_str = d.strftime("%b %Y")
                            if val < base_cash:
                                underwater_streak += 1
                                max_underwater = max(max_underwater, underwater_streak)
                            else:
                                underwater_streak = 0

                    # Contribution efficiency: proportion of buys below today's price
                    buys_below = 0
                    for d in invest_dates_local:
                        px = price_base.loc[d, t] if d in price_base.index else np.nan
                        if pd.notnull(px) and pd.notnull(cur_price) and float(px) < cur_price:
                            buys_below += 1
                    eff = (buys_below / len(invest_dates_local) * 100.0) if len(invest_dates_local) > 0 else np.nan

                    # Time-weighted return (price-only) for context
                    twr_pct = float((asset_cum[t].iloc[-1] - 1.0) * 100.0) if t in asset_cum.columns else np.nan

                    # Monthly price volatility (std of monthly returns)
                    mp = monthly_prices[t].dropna()
                    mr = mp.pct_change().dropna()
                    vol_m = float(np.nanstd(mr)) * 100.0 if len(mr) > 0 else np.nan

                    per_asset_rows.append({
                        "Ticker": t,
                        "Avg Cost": avg_cost,
                        "Current Price": cur_price,
                        "Premium/Discount %": prem,
                        "Break-even Month": break_month_str,
                        "Max Underwater (months)": int(max_underwater) if not np.isnan(max_underwater) else "—",
                        "Contribution Efficiency %": eff,
                        "TWR % (Price)": twr_pct,
                        "Monthly Volatility %": vol_m,
                    })

                dca_df = pd.DataFrame(per_asset_rows).set_index("Ticker")

                # Format & show Table 2
                fmt = dca_df.copy()
                fmt["Avg Cost"]            = fmt["Avg Cost"].map(lambda x: f"{sym}{x:,.2f}" if pd.notnull(x) else "—")
                fmt["Current Price"]       = fmt["Current Price"].map(lambda x: f"{sym}{x:,.2f}" if pd.notnull(x) else "—")
                fmt["Premium/Discount %"]  = fmt["Premium/Discount %"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "—")
                fmt["Contribution Efficiency %"] = fmt["Contribution Efficiency %"].map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "—")
                fmt["TWR % (Price)"]       = fmt["TWR % (Price)"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "—")
                fmt["Monthly Volatility %"] = fmt["Monthly Volatility %"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "—")

                col_config_dca = {
                    "Avg Cost": st.column_config.Column(width=110, help="Weighted average purchase price (net of fees)"),
                    "Current Price": st.column_config.Column(width=120),
                    "Premium/Discount %": st.column_config.Column(width=160, help="(Current / Avg Cost) - 1"),
                    "Break-even Month": st.column_config.Column(width=150, help="First month value >= contributions+fees"),
                    "Max Underwater (months)": st.column_config.Column(width=190, help="Longest streak value < cash-out"),
                    "Contribution Efficiency %": st.column_config.Column(width=190, help="Share of buys below today's price"),
                    "TWR % (Price)": st.column_config.Column(width=130, help="Time-weighted price return (reference)"),
                    "Monthly Volatility %": st.column_config.Column(width=160, help="Std dev of monthly price returns"),
                }
                st.dataframe(fmt, use_container_width=True, height=420, column_config=col_config_dca)

            # =========================
            # DRAWDOWN — price-based portfolio drawdown (reference market risk)
            # - Independent of cash contributions; uses the price-based index
            # =========================
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📉 Portfolio Drawdown (Price-based index)")
            dd = (port_cum / port_cum.cummax() - 1.0)
            max_dd = float(dd.min() * 100.0)

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd.index, y=(dd*100.0).values, mode="lines",
                                        name="Drawdown", fill="tozeroy", line=dict(color="#ef4444")))

            layout_dd = base_layout(title_text=f"Max Drawdown: {max_dd:.2f}%", height=420, right_margin=220)
            fig_dd.update_layout(**layout_dd)
            fig_dd.update_layout(
                yaxis=dict(
                    title=dict(text="Drawdown (%)", font=dict(color=AXIS_TEXT)),
                    tickfont=dict(color=AXIS_TEXT),
                    gridcolor=GRID_COLOR,
                    ticksuffix="%"
                )
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            st.caption(
                "Max drawdown is the largest peak-to-trough fall of the **price-based portfolio index** over the selected period — "
                "a proxy for market risk (independent of your contribution timing)."
            )
else:
    # Initial hint before the first run
    st.markdown('Set your assets and click **Run** in the sidebar.')
