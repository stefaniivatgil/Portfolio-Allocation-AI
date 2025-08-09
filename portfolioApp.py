import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# =========================
# Page / Theme
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

/* Subtle divider */
.divider { width:100%; height:1px; background: rgba(226,232,240,0.25); margin: 12px 0 18px 0; }

/* Perf cards (under legend) */
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

/* Sidebar */
.sidebar-title { font-weight:700; font-size:1.05rem; margin:4px 0 8px 0; }
.sidebar-subtle { color:var(--muted); font-size:0.9rem; }

/* Controls */
a, .stDownloadButton button, .stButton button { border-radius: 10px !important; }
.stButton button { background: var(--accent); color:white; border:0; }
.stSlider > div > div > div { background: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Plot colors & layout helpers
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
# Currency helpers (FX conversion)
# =========================
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥", "HKD": "HK$",
    "CHF": "CHF", "CAD": "C$", "AUD": "A$", "NZD": "NZ$", "SEK": "kr",
    "NOK": "kr", "DKK": "kr", "INR": "₹", "SGD": "S$", "ZAR": "R",
    "TRY": "₺", "MXN": "Mex$", "BRL": "R$", "PLN": "zł"
}

@st.cache_data(show_spinner=False)
def detect_currency(ticker: str) -> str:
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

    direct = _dl(f"{src}{dst}=X")
    if direct is not None: return direct
    inv = _dl(f"{dst}{src}=X")
    if inv is not None:
        inv = inv.replace(0, np.nan)
        return 1.0 / inv

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

    idx = pd.date_range(start_dt, end_dt, freq="D")
    return pd.Series(1.0, index=idx)

def sym_for(code: str) -> str:
    return CURRENCY_SYMBOLS.get(code, code + " ")

# =========================
# Hero
# =========================
st.markdown(
    """
<div class="hero">
  <h1>Portfolio Simulator</h1>
  <div class="sub">FX converted portfolio simulator.</div>
</div>
""",
    unsafe_allow_html=True
)

# =========================
# State
# =========================
if "assets" not in st.session_state:
    st.session_state.assets = [
        {"ticker": "AAPL", "weight": 40.0},
        {"ticker": "MSFT", "weight": 40.0},
        {"ticker": "GLD",  "weight": 20.0},
    ]

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Portfolio Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtle">One row per asset. Auto-normalize keeps total at 100%.</div>', unsafe_allow_html=True)

    n_assets = st.number_input("Number of assets", 1, 20, value=len(st.session_state.assets), step=1)
    if n_assets > len(st.session_state.assets):
        st.session_state.assets += [{"ticker": "", "weight": 0.0} for _ in range(n_assets - len(st.session_state.assets))]
    elif n_assets < len(st.session_state.assets):
        st.session_state.assets = st.session_state.assets[:n_assets]

    auto_normalize = st.toggle("Auto-normalize to 100%", value=True)

    st.markdown("**Assets**")
    for i in range(n_assets):
        c1, c2 = st.columns([2, 1], gap="small")
        with c1:
            st.session_state.assets[i]["ticker"] = st.text_input(
                f"Ticker {i+1}", value=st.session_state.assets[i]["ticker"], key=f"ticker_{i}"
            ).upper().strip()
        with c2:
            st.session_state.assets[i]["weight"] = st.number_input(
                f"Weight {i+1} (%)", 0.0, 100.0, value=float(st.session_state.assets[i]["weight"]),
                step=1.0, key=f"weight_{i}"
            )

    if auto_normalize:
        total = sum(a["weight"] for a in st.session_state.assets) or 1.0
        for a in st.session_state.assets:
            a["weight"] = round(a["weight"] * 100.0 / total, 4)

    today = datetime.today().date()
    default_start = (datetime.today() - timedelta(days=365)).date()
    min_slider_date = (datetime.today() - timedelta(days=365*15)).date()

    date_start, date_end = st.slider(
        "Historical date range", min_value=min_slider_date, max_value=today,
        value=(default_start, today), format="YYYY-MM-DD"
    )

    BASE_CHOICES = ["USD","EUR","GBP","CHF","JPY","CAD","AUD","NZD","SEK","NOK","DKK","INR","SGD","HKD","ZAR","TRY","MXN","BRL","PLN","CNY"]
    base_currency = st.selectbox("Base currency", BASE_CHOICES, index=1)  # default EUR

    show_indicators = st.toggle("Show Technical Indicators (SMA20 + RSI)", value=False)

    st.markdown("---")
    saved_name = st.text_input("Portfolio Name")
    load_file = st.file_uploader("Load Portfolio JSON", type="json")

    colA, colB = st.columns(2)
    with colA: save_click = st.button("Save")
    with colB: run_click  = st.button("Run")

if load_file is not None:
    try:
        loaded = json.load(load_file)
        if "assets" in loaded and isinstance(loaded["assets"], list):
            st.session_state.assets = loaded["assets"]
        else:
            tickers = [t.strip().upper() for t in loaded.get("tickers","").split(",") if t.strip()]
            weights = [float(w) for w in loaded.get("weights","").split(",") if w.strip()]
            st.session_state.assets = [{"ticker": t, "weight": w} for t, w in zip(tickers, weights)]
        st.success("Portfolio loaded. Press Run.")
    except Exception as e:
        st.error(f"Failed to load: {e}")

if save_click:
    if saved_name:
        payload = {"assets": st.session_state.assets}
        st.download_button("Download JSON", data=json.dumps(payload, indent=2),
                           file_name=f"{saved_name}.json", mime="application/json",
                           use_container_width=True)
    else:
        st.warning("Give your portfolio a name first.")

# =========================
# Helpers
# =========================
def annualized_sharpe(daily_returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    rets = daily_returns.dropna()
    if rets.std(ddof=0) == 0 or len(rets) < 2: return float("nan")
    excess = rets - risk_free_daily
    return float(np.sqrt(252) * excess.mean() / excess.std(ddof=0))

def compute_drawdown(cum: pd.Series) -> pd.Series:
    return cum / cum.cummax() - 1.0

def annualized_from_prices(start_val: float, end_val: float, years: float) -> float:
    """
    Geometric annualized return based on start/end values over 'years' (calendar years).
    Returns NaN if years <= 0 or invalid inputs.
    """
    if years is None or years <= 0 or start_val is None or end_val is None:
        return float("nan")
    if np.isnan(start_val) or np.isnan(end_val) or start_val <= 0:
        return float("nan")
    return (end_val / start_val) ** (1.0 / years) - 1.0

# =========================
# Main
# =========================
if run_click:
    tickers = [a["ticker"] for a in st.session_state.assets if a["ticker"]]
    weights = [float(a["weight"]) for a in st.session_state.assets if a["ticker"]]

    if len(tickers) != len(weights) or len(tickers) == 0:
        st.error("Please enter a valid set of tickers and weights.")
    else:
        total_w = round(sum(weights), 6)
        if abs(total_w - 100.0) > 1e-3:
            st.warning(f"Weights sum to {total_w}%, not 100%. (Enable Auto-normalize for a clean 100%.)")

        start_dt = datetime.combine(date_start, datetime.min.time())
        end_dt   = datetime.combine(date_end,   datetime.min.time())

        currencies = {t: detect_currency(t) for t in tickers}

        # Download prices
        raw_price = pd.DataFrame()
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
            # Convert to base currency
            price_base = pd.DataFrame(index=raw_price.index)
            for t in tickers:
                fx = fetch_fx_series(currencies.get(t, "USD"), base_currency, start_dt, end_dt)
                price_base[t] = raw_price[t] * fx.reindex(raw_price.index).ffill().bfill()

            # Returns & cumulative
            rets      = price_base.pct_change().fillna(0.0)
            weighted  = rets.mul([w/100.0 for w in weights], axis=1)
            port_rets = weighted.sum(axis=1)
            port_cum  = (1.0 + port_rets).cumprod()
            asset_cum = (1.0 + rets).cumprod()

            # Calendar years for annualization (same window for all)
            years = (price_base.index[-1] - price_base.index[0]).days / 365.25 if len(price_base.index) >= 2 else np.nan

            # === 1) MAIN CHART
            st.markdown("### Portfolio & Assets — Growth (Base = 1)")
            st.markdown(
                f'<div class="divider"></div><div style="color:#cbd5e1;font-size:13px;">'
                f'Base currency: <b>{base_currency}</b> · Range: <b>{date_start} → {date_end}</b></div>',
                unsafe_allow_html=True
            )

            fig = go.Figure()
            # Portfolio line
            fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, mode="lines",
                                     name="PORTFOLIO", line=dict(width=4, color=PLOT_COLORS[0])))
            # Each asset
            for idx, t in enumerate(asset_cum.columns):
                fig.add_trace(go.Scatter(x=asset_cum.index, y=asset_cum[t].values, mode="lines",
                                         name=t, opacity=0.9, line=dict(color=PLOT_COLORS[(idx+1) % len(PLOT_COLORS)])))
            layout_main = base_layout(title_text=f"{date_start} → {date_end}", height=640, right_margin=220)
            fig.update_layout(**layout_main, yaxis_title_text="Cumulative Return")
            st.plotly_chart(fig, use_container_width=True)

            # === 2) PERFORMANCE CARDS (under legend)
            perf_cards = []
            port_perf = float((port_cum.iloc[-1] - 1.0) * 100.0)
            perf_cards.append(("PORTFOLIO", port_perf, PLOT_COLORS[0]))
            for i, t in enumerate(tickers):
                s = price_base[t].dropna()
                if len(s) > 1:
                    perf = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) * 100.0
                else:
                    perf = float("nan")
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

            # === 3) DETAILS TABLE (with Annualized %)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### Performance Details (Converted to Base Currency)")

            rows = [{
                "Ticker":"PORTFOLIO","Native CCY":"—","Currency":base_currency,"Weight %":100.0,
                "Start Price":1.00,"End Price":float(port_cum.iloc[-1]),
                "Performance %": float((port_cum.iloc[-1]-1.0)*100.0),
                "Annualized %": float(annualized_from_prices(1.0, float(port_cum.iloc[-1]), years) * 100.0),
                "Sharpe": annualized_sharpe(port_rets)
            }]
            for t, w in zip(tickers, weights):
                s = price_base[t].dropna()
                if len(s) > 1:
                    start_p, end_p = float(s.iloc[0]), float(s.iloc[-1])
                    perf = (end_p/start_p - 1.0) * 100.0
                    ann  = annualized_from_prices(start_p, end_p, years)
                else:
                    start_p = end_p = perf = float("nan")
                    ann = float("nan")
                rows.append({
                    "Ticker": t,
                    "Native CCY": currencies.get(t, "USD"),
                    "Currency": base_currency,
                    "Weight %": float(w),
                    "Start Price": start_p,
                    "End Price": end_p,
                    "Performance %": perf,
                    "Annualized %": float(ann * 100.0) if pd.notnull(ann) else float("nan"),
                    "Sharpe": annualized_sharpe(rets[t])
                })
            metrics_df = pd.DataFrame(rows).set_index("Ticker")[
                ["Native CCY","Currency","Weight %","Start Price","End Price","Performance %","Annualized %","Sharpe"]
            ]

            sym = sym_for(base_currency)
            show_df = metrics_df.copy()
            show_df["Weight %"]      = show_df["Weight %"].map(lambda x: f"{x:.2f}%")
            show_df["Start Price"]   = show_df["Start Price"].map(lambda x: f"{sym}{x:,.2f}" if pd.notnull(x) else "—")
            show_df["End Price"]     = show_df["End Price"].map(lambda x: f"{sym}{x:,.2f}" if pd.notnull(x) else "—")
            show_df["Performance %"] = show_df["Performance %"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "—")
            show_df["Annualized %"]  = show_df["Annualized %"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "—")
            show_df["Sharpe"]        = show_df["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "—")

            st.dataframe(
                show_df,
                use_container_width=True,
                height=560,
                column_config={
                    "Native CCY": st.column_config.Column(width=90, help="Asset’s original currency"),
                    "Currency":   st.column_config.Column(width=90, help="Conversion target (base)"),
                    "Weight %":   st.column_config.Column(width=90),
                    "Start Price": st.column_config.Column(width=120),
                    "End Price":   st.column_config.Column(width=120),
                    "Performance %": st.column_config.Column(width=120),
                    "Annualized %": st.column_config.Column(width=120, help="Geometric annualized return over selected window"),
                    "Sharpe":        st.column_config.Column(width=80),
                }
            )

            # === 4) DRAWDOWN (after the table)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📉 Portfolio Drawdown")
            dd = (port_cum / port_cum.cummax() - 1.0)
            max_dd = float(dd.min() * 100.0)

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd.index, y=(dd*100.0).values, mode="lines",
                                        name="Drawdown", fill="tozeroy", line=dict(color="#ef4444")))
            layout_dd = base_layout(title_text=f"Max Drawdown: {max_dd:.2f}%", height=420, right_margin=220)
            fig_dd.update_layout(**layout_dd, yaxis_title_text="Drawdown (%)", yaxis_ticksuffix="%")
            st.plotly_chart(fig_dd, use_container_width=True)

            st.caption(
                "Max drawdown is the largest peak-to-trough fall of the **portfolio index** over the selected period — "
                "a quick proxy for worst-case loss before a recovery."
            )
else:
    st.markdown('Set your assets and click **Run** in the sidebar.')
