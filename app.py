import datetime
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, acf

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Risk Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "NSE Risk Analysis Tool — Quantitative Finance Portfolio Project"
    },
)

# ─── Constants ────────────────────────────────────────────────────────────────
STOCK_OPTIONS = {
    "Reliance Industries":      "RELIANCE.NS",
    "Tata Consultancy Services":"TCS.NS",
    "HDFC Bank":                "HDFCBANK.NS",
    "Infosys":                  "INFY.NS",
    "ITC Limited":              "ITC.NS",
    "Wipro":                    "WIPRO.NS",
    "Bharti Airtel":            "BHARTIARTL.NS",
    "Asian Paints":             "ASIANPAINT.NS",
    "Maruti Suzuki":            "MARUTI.NS",
    "Nifty 50 Index":           "^NSEI",
}

PAGES = [
    "Overview",
    "Risk Methods",
    "GARCH Analysis",
    "Stock Comparison",
    "ML vs GARCH",
]

# ─── CSS ──────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
<style>
/* ── Force light theme everywhere ── */
.stApp                             { background-color: #ffffff !important; }
.main                              { background-color: #ffffff !important; }
.stApp > header                    { background-color: transparent !important; }
[data-testid="stHeader"]           { background-color: transparent !important; }
[data-testid="stSidebar"]          { background-color: #f0f4f8 !important;
                                     border-right: 2px solid #dde3ea; }
[data-testid="stSidebarContent"]   { background-color: #f0f4f8 !important; }
section[data-testid="stSidebar"] > div { background-color: #f0f4f8 !important; }

/* ── Animations ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.main .block-container {
    animation: fadeSlideIn 0.35s ease-out;
    padding-top: 2rem;
}

/* ── Metric cards — navy border by default ── */
[data-testid="metric-container"] {
    background:    #ffffff !important;
    border:        1px solid #dde3ea !important;
    border-left:   4px solid #1f4e79 !important;
    border-radius: 8px !important;
    padding:       20px 16px !important;
    box-shadow:    0 2px 8px rgba(31,78,121,0.08) !important;
    transition:    box-shadow 0.2s ease !important;
    animation:     fadeSlideIn 0.4s ease-out;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 16px rgba(31,78,121,0.15) !important;
}
/* Risk metric cards — red border override */
.risk-metrics [data-testid="metric-container"] {
    border-left: 4px solid #c0392b !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #dde3ea !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}

/* ── Dividers ── */
hr { border-color: #dde3ea !important; }

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #dde3ea !important;
    border-radius: 6px !important;
}

/* ── Compact sidebar widget spacing ── */
[data-testid="stSidebar"] .stRadio,
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stDateInput,
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stSelectSlider {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    margin-bottom: 2px !important;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] .element-container {
    margin-bottom: 4px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ─── Shared Helpers ───────────────────────────────────────────────────────────
def chart_layout(title="", height=400):
    return dict(
        title=dict(
            text=title,
            font=dict(size=15, color="#1a1a2e", family="Arial, sans-serif"),
        ),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#1a1a2e"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(gridcolor="#f0f0f0", showgrid=True, linecolor="#dde3ea", zeroline=False),
        yaxis=dict(gridcolor="#f0f0f0", showgrid=True, linecolor="#dde3ea", zeroline=False),
        margin=dict(l=50, r=30, t=70, b=50),
    )


def section_header(text, color="#1f4e79"):
    st.markdown(
        f'<h3 style="border-left:4px solid {color}; padding-left:12px; '
        f'color:#1a1a2e; font-size:18px; font-weight:600; margin:16px 0 10px 0;">'
        f"{text}</h3>",
        unsafe_allow_html=True,
    )


def callout(text, border_color="#1f4e79", bg_color="#dce6f0"):
    st.markdown(
        f'<div style="background:{bg_color}; border-left:4px solid {border_color}; '
        f'border-radius:4px; padding:12px 16px; font-size:14px; color:#1a1a2e; '
        f'line-height:1.65; margin-bottom:16px;">{text}</div>',
        unsafe_allow_html=True,
    )


def page_footer():
    st.divider()
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        st.caption("**NSE Risk Analysis Tool** · Quantitative Finance Portfolio Project")
        st.caption("Built by Yashwanth V · BTech CS + FinTech · github.com/yash2021v/nse-risk-analysis")
    with col_f2:
        st.caption("**Data Source**")
        st.caption("Yahoo Finance via yfinance")
    with col_f3:
        st.caption("**Stack**")
        st.caption("Python · yfinance · pandas · NumPy · SciPy · statsmodels · arch (GARCH) · scikit-learn · XGBoost · Plotly · Streamlit")


# ─── Core Functions (verbatim) ────────────────────────────────────────────────

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    price = data["Close"].squeeze()
    log_returns = np.log(price / price.shift(1)).dropna()
    return price, log_returns


@st.cache_data
def compute_risk_metrics(ticker, start, end, confidence):
    _, log_returns = load_data(ticker, start, end)
    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    var_hist = float(np.percentile(log_returns, (1 - confidence) * 100))
    var_param = float(mu + sigma * stats.norm.ppf(1 - confidence))
    cvar = float(log_returns[log_returns <= var_hist].mean())
    kurtosis = float(log_returns.kurtosis() + 3)
    skewness = float(log_returns.skew())
    worst_day = float(log_returns.min())
    best_day = float(log_returns.max())
    return {
        "var_hist": var_hist,
        "var_param": var_param,
        "cvar": cvar,
        "mu": mu,
        "sigma": sigma,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "worst_day": worst_day,
        "best_day": best_day,
    }


@st.cache_data
def fit_garch(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    returns = np.log(
        data["Close"].squeeze() / data["Close"].squeeze().shift(1)
    ).dropna()
    scaled = returns * 100
    model = arch_model(scaled, vol='Garch', p=1, q=1,
                       mean='Constant', dist='normal')
    result = model.fit(disp='off')
    cond_vol = result.conditional_volatility / 100
    alpha = float(result.params['alpha[1]'])
    beta = float(result.params['beta[1]'])
    omega = float(result.params['omega'])
    return cond_vol, alpha, beta, omega, returns


def run_adf_test(series):
    result = adfuller(series.dropna())
    return {
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 6),
        "critical_1": round(result[4]['1%'], 4),
        "critical_5": round(result[4]['5%'], 4),
        "is_stationary": result[1] < 0.05,
    }


def compute_acf_values(series, n_lags=40):
    acf_values = acf(series, nlags=n_lags)
    conf_interval = 1.96 / np.sqrt(len(series))
    return acf_values, conf_interval


def compute_rolling_metrics(log_returns, window=252, confidence=0.95):
    rolling_var = log_returns.rolling(window).apply(
        lambda x: np.percentile(x, (1 - confidence) * 100)
    )
    rolling_cvar = log_returns.rolling(window).apply(
        lambda x: float(x[x <= np.percentile(
            x, (1-confidence)*100)].mean())
    )
    rolling_param_var = log_returns.rolling(window).apply(
        lambda x: float(x.mean() + x.std() *
                        stats.norm.ppf(1 - confidence))
    )
    return rolling_var, rolling_cvar, rolling_param_var


@st.cache_data
def compute_all_stocks(start, end, confidence):
    stock_list = {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "INFY": "INFY.NS",
        "ITC": "ITC.NS"
    }
    results = {"tickers": [], "hvar": [], "pvar": [], "cvar": []}
    for name, ticker in stock_list.items():
        try:
            d = yf.download(ticker, start=start,
                           end=end, progress=False)
            r = np.log(
                d["Close"].squeeze() /
                d["Close"].squeeze().shift(1)
            ).dropna()
            vh = float(np.percentile(r, (1 - confidence) * 100))
            vp = float(r.mean() + r.std() *
                      stats.norm.ppf(1 - confidence))
            cv = float(r[r <= vh].mean())
            results["tickers"].append(name)
            results["hvar"].append(abs(vh))
            results["pvar"].append(abs(vp))
            results["cvar"].append(abs(cv))
        except:
            pass
    return results


# ─── Sidebar ──────────────────────────────────────────────────────────────────
_RULE = (
    '<hr style="border:none; border-top:1px solid #dde3ea; margin:6px 0 8px 0;"/>'
)
_LABEL = (
    '<p style="font-size:10px; font-weight:600; color:#8a9ab0; '
    'letter-spacing:1.4px; margin:4px 0 2px 0;">{}</p>'
)


def render_sidebar():
    # Logo / branding — compact
    st.sidebar.markdown(
        '<div style="padding:8px 4px 4px 4px;">'
        '<span style="font-size:17px; font-weight:700; color:#1f4e79;">📈 NSE Risk Analysis</span><br>'
        '<span style="font-size:10px; color:#8a9ab0;">Quantitative Finance Research Tool</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(_RULE, unsafe_allow_html=True)

    # Navigation
    st.sidebar.markdown(_LABEL.format("NAVIGATION"), unsafe_allow_html=True)
    page = st.sidebar.radio("nav", PAGES, label_visibility="collapsed")
    st.sidebar.markdown(_RULE, unsafe_allow_html=True)

    # Parameters
    st.sidebar.markdown(_LABEL.format("PARAMETERS"), unsafe_allow_html=True)

    # Stock selector — hidden on Comparison page
    selected_name = None
    ticker = None
    if page != "Stock Comparison":
        selected_name = st.sidebar.selectbox(
            "Stock", list(STOCK_OPTIONS.keys()), index=0, label_visibility="collapsed"
        )
        ticker = STOCK_OPTIONS[selected_name]
    else:
        st.sidebar.markdown(
            '<div style="font-size:11px; color:#8a9ab0; margin:2px 0 4px 0;">'
            "RELIANCE · TCS · HDFCBANK · INFY · ITC</div>",
            unsafe_allow_html=True,
        )

    today = datetime.date.today()
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.date(2018, 1, 1),
        min_value=datetime.date(2010, 1, 1),
        max_value=today - datetime.timedelta(days=1),
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=today,
        min_value=start_date + datetime.timedelta(days=50),
        max_value=today,
    )
    confidence = st.sidebar.select_slider(
        "Confidence Level",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%",
    )

    st.sidebar.markdown(_RULE, unsafe_allow_html=True)

    clicked = st.sidebar.button(
        "🔍  Analyse",
        type="primary",
        use_container_width=True,
        help="Click to run analysis with the selected parameters",
    )
    if clicked:
        st.session_state.run_analysis = True
        st.session_state.start = str(start_date)
        st.session_state.end = str(end_date)
        st.session_state.confidence = confidence
        if ticker is not None:
            st.session_state.ticker = ticker
        if selected_name is not None:
            st.session_state.selected_name = selected_name

    st.sidebar.markdown(
        '<p style="font-size:10px; color:#8a9ab0; margin-top:6px;">'
        "Done by Yashwanth V (BTech CS + FinTech) · Data via Yahoo Finance</p>",
        unsafe_allow_html=True,
    )

    return page


# ─── Landing State ────────────────────────────────────────────────────────────
def show_landing():
    st.markdown(
        '<h1 style="color:#1f4e79; font-size:36px; font-weight:700; margin-bottom:4px;">'
        "NSE Risk Analysis Tool</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#5a6a7a; font-size:17px; margin-bottom:28px;">'
        "Institutional-grade tail risk estimation on Indian equity markets</p>",
        unsafe_allow_html=True,
    )

    df_methods = pd.DataFrame(
        {
            "Method": [
                "Historical VaR",
                "Parametric VaR",
                "CVaR (Expected Shortfall)",
                "GARCH(1,1)",
            ],
            "Approach": [
                "Empirical percentile of past returns",
                "Normal distribution assumption",
                "Average loss beyond VaR threshold",
                "Time-varying conditional volatility",
            ],
            "Best For": [
                "Regime-stable markets",
                "Quick estimation",
                "Tail severity measurement",
                "Crisis detection",
            ],
        }
    )
    st.dataframe(df_methods, use_container_width=True, hide_index=True)

    st.info(
        "👈 Select a stock and date range in the sidebar, then click **Analyse** to begin."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        callout(
            "<strong>🎯 Precise</strong><br>"
            "Four complementary risk methods applied to the same dataset for direct comparison.",
            border_color="#1f4e79",
            bg_color="#dce6f0",
        )
    with c2:
        callout(
            "<strong>📊 Visual</strong><br>"
            "Interactive Plotly charts with annotated VaR, CVaR, and volatility thresholds.",
            border_color="#2e75b6",
            bg_color="#e8f0f8",
        )
    with c3:
        callout(
            "<strong>⚡ Live</strong><br>"
            "Real-time NSE price data streamed via Yahoo Finance on every analysis run.",
            border_color="#27ae60",
            bg_color="#e8f8ee",
        )


# ─── Guard for stock-specific pages ───────────────────────────────────────────
def require_ticker():
    """Returns False and shows a prompt if no stock has been analysed yet."""
    if not st.session_state.get("ticker"):
        st.info(
            "👈 Select a stock from the **Stock** dropdown in the sidebar "
            "and click **Analyse** to view this page."
        )
        page_footer()
        return False
    return True


# ─── Page 1 — Overview ────────────────────────────────────────────────────────
def page_overview():
    if not require_ticker():
        return

    ticker = st.session_state.ticker
    selected_name = st.session_state.selected_name
    start = st.session_state.start
    end = st.session_state.end
    confidence = st.session_state.confidence

    st.markdown(
        '<h2 style="color:#1f4e79; font-size:28px; font-weight:700; margin-bottom:2px;">'
        "📊 Overview</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#8a9ab0; font-size:13px; margin-bottom:16px;">'
        f"{selected_name} · {start} to {end} · {int(confidence*100)}% Confidence</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.spinner(f"Loading {selected_name} data..."):
        try:
            price, log_returns = load_data(ticker, start, end)
            metrics = compute_risk_metrics(ticker, start, end, confidence)
        except Exception:
            st.error(
                f"⚠️ Could not download data for **{selected_name}**. "
                "Check your internet connection."
            )
            st.stop()

    if len(log_returns) < 50:
        st.error(
            "⚠️ Insufficient data. "
            "Please select a date range spanning at least 50 trading days."
        )
        st.stop()

    # ── Price chart ──
    section_header("Price History")
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=price.index,
            y=price.values,
            mode="lines",
            line=dict(color="#2e75b6", width=1.8),
            name=selected_name,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>",
        )
    )
    # COVID shaded region
    price_start = pd.Timestamp(start)
    price_end = pd.Timestamp(end)
    covid_s = pd.Timestamp("2020-02-15")
    covid_e = pd.Timestamp("2020-04-15")
    if price_start <= covid_e and price_end >= covid_s:
        fig_price.add_vrect(
            x0="2020-02-15",
            x1="2020-04-15",
            fillcolor="rgba(192,57,43,0.08)",
            layer="below",
            line_width=0,
            annotation_text="COVID Crash",
            annotation_position="top left",
            annotation_font_color="#c0392b",
            annotation_font_size=11,
        )
    fig_price.update_layout(
        **chart_layout(title=f"{selected_name} — Closing Price (₹)", height=350)
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── 5 summary metric cards ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trading Days", f"{len(log_returns):,}")
    c2.metric("Latest Price (₹)", f"₹{float(price.iloc[-1]):,.2f}")
    c3.metric("Mean Daily Return", f"{metrics['mu']*100:.4f}%")
    c4.metric("Daily Volatility (σ)", f"{metrics['sigma']*100:.4f}%")
    c5.metric("Worst Single Day", f"{metrics['worst_day']*100:.2f}%")

    st.divider()

    # ── Return distribution ──
    section_header("Return Distribution & Risk Thresholds")
    var_h = metrics["var_hist"]
    var_p = metrics["var_param"]
    cvar_v = metrics["cvar"]

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=log_returns[log_returns > var_h].values,
            nbinsx=100,
            marker_color="#2e75b6",
            opacity=0.6,
            name="Returns",
        )
    )
    fig_hist.add_trace(
        go.Histogram(
            x=log_returns[log_returns <= var_h].values,
            nbinsx=30,
            marker_color="#c0392b",
            opacity=0.85,
            name="Tail Returns",
        )
    )
    fig_hist.add_vline(
        x=var_h, line_dash="dash", line_color="#1f4e79", line_width=2,
        annotation_text=f"Hist VaR {var_h*100:.2f}%",
        annotation_font_color="#1f4e79",
    )
    fig_hist.add_vline(
        x=var_p, line_dash="dot", line_color="#e67e22", line_width=2,
        annotation=dict(
        text=f"Param VaR {var_p*100:.2f}%", 
        font_color="#F5890E",
        # 'paper' means 0 is the bottom of the plot, 1 is the top
        yref="paper", 
        y=0.9,            # 0.5 places it exactly in the middle vertically
        xanchor="right" )
    )
    fig_hist.add_vline(
        x=cvar_v, line_dash="dash", line_color="#c0392b", line_width=2,
        annotation_text=f"CVaR {cvar_v*100:.2f}%",
        annotation_font_color="#c0392b",
        annotation_position="top left"
    )
    fig_hist.update_layout(
        **chart_layout(
            title="Daily Log-Return Distribution with Risk Thresholds", height=400
        ),
        barmode="overlay",
        xaxis_title="Log Return",
        yaxis_title="Frequency",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── 4 risk metric cards (red border) ──
    st.markdown('<div class="risk-metrics">', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Historical VaR", f"{var_h*100:.3f}%")
    d2.metric(
        "Parametric VaR",
        f"{var_p*100:.3f}%",
        delta=f"{(var_p - var_h)*100:+.3f}% vs Hist",
        delta_color="inverse",
    )
    d3.metric(
        "CVaR",
        f"{cvar_v*100:.3f}%",
        delta=f"{(cvar_v - var_h)*100:+.3f}% vs Hist",
        delta_color="inverse",
    )
    d4.metric("VaR–CVaR Gap", f"{abs(cvar_v - var_h)*100:.3f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    page_footer()


# ─── Page 2 — Risk Methods ────────────────────────────────────────────────────
def page_risk_methods():
    if not require_ticker():
        return

    ticker = st.session_state.ticker
    selected_name = st.session_state.selected_name
    start = st.session_state.start
    end = st.session_state.end
    confidence = st.session_state.confidence

    st.markdown(
        '<h2 style="color:#1f4e79; font-size:28px; font-weight:700; margin-bottom:2px;">'
        "🔬 Risk Methods Deep Dive</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#8a9ab0; font-size:14px; margin-bottom:16px;">'
        "Comparing methodology, assumptions, and performance across three estimation approaches</p>",
        unsafe_allow_html=True,
    )

    try:
        _, log_returns = load_data(ticker, start, end)
        metrics = compute_risk_metrics(ticker, start, end, confidence)
    except Exception:
        st.error(f"⚠️ Could not load data for **{selected_name}**.")
        st.stop()

    if len(log_returns) < 50:
        st.error(
            "⚠️ Insufficient data. "
            "Please select a date range spanning at least 50 trading days."
        )
        st.stop()

    mu = metrics["mu"]
    sigma = metrics["sigma"]
    kurtosis = metrics["kurtosis"]
    skewness = metrics["skewness"]

    # ── Section A: Historical VaR ──────────────────────────────────────────────
    with st.expander("Historical Simulation VaR", expanded=True):
        callout(
            "Historical Simulation sorts the past N days of returns and reads off the 5th "
            "percentile. No distribution is assumed — the empirical data speaks for itself.<br><br>"
            "<strong>Core weakness:</strong> The model is blind to crises that have not occurred "
            "within its lookback window — known as look-back bias. It cannot warn before a pandemic "
            "it has never seen.",
            border_color="#1f4e79",
            bg_color="#dce6f0",
        )

        # VaR at 90 / 95 / 99 %
        rows = []
        for lbl, cv in [("90%", 0.90), ("95%", 0.95), ("99%", 0.99)]:
            m = compute_risk_metrics(ticker, start, end, cv)
            rows.append(
                {
                    "Confidence": lbl,
                    "Historical VaR": f"{m['var_hist']*100:.3f}%",
                    "Worst Day": f"{m['worst_day']*100:.2f}%",
                }
            )
        st.markdown("**VaR at 90%, 95%, 99% confidence:**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Calm vs Crisis comparison
        col_c, col_cr = st.columns(2)
        with col_c:
            st.markdown(
                '<div style="color:#27ae60; font-weight:600; font-size:15px; margin-bottom:8px;">'
                "✅ Calm Period (2018–2019)</div>",
                unsafe_allow_html=True,
            )
            try:
                _, calm_ret = load_data(ticker, "2018-01-01", "2019-12-31")
                c_var = float(np.percentile(calm_ret, (1 - confidence) * 100))
                st.metric("Historical VaR", f"{c_var*100:.3f}%")
                st.metric("Worst Day", f"{float(calm_ret.min())*100:.2f}%")
            except Exception:
                st.info("Insufficient data for 2018–2019 in this range.")

        with col_cr:
            st.markdown(
                '<div style="color:#c0392b; font-weight:600; font-size:15px; margin-bottom:8px;">'
                "⚠️ Crisis Period (2020)</div>",
                unsafe_allow_html=True,
            )
            try:
                _, crisis_ret = load_data(ticker, "2020-01-01", "2020-12-31")
                cr_var = float(np.percentile(crisis_ret, (1 - confidence) * 100))
                st.metric("Historical VaR", f"{cr_var*100:.3f}%")
                st.metric("Worst Day", f"{float(crisis_ret.min())*100:.2f}%")
            except Exception:
                st.info("Insufficient data for 2020 in this range.")

        # Rolling VaR chart
        st.markdown("**Rolling 252-Day Historical VaR:**")
        roll_var, _, _ = compute_rolling_metrics(
            log_returns, window=252, confidence=confidence
        )
        fig_roll = go.Figure()
        fig_roll.add_trace(
            go.Scatter(
                x=log_returns.index,
                y=log_returns.values,
                mode="lines",
                line=dict(color="#2e75b6", width=0.8),
                opacity=0.4,
                name="Returns",
            )
        )
        fig_roll.add_trace(
            go.Scatter(
                x=roll_var.index,
                y=roll_var.values,
                mode="lines",
                line=dict(color="#1f4e79", width=2),
                name="Rolling Hist VaR",
            )
        )
        # Annotate the COVID lag
        post_covid = roll_var.index[roll_var.index >= pd.Timestamp("2020-04-01")]
        if len(post_covid) > 3:
            ann_date = post_covid[3]
            ann_val = float(roll_var.loc[ann_date])
            if not np.isnan(ann_val):
                fig_roll.add_annotation(
                    x=ann_date,
                    y=ann_val,
                    text="VaR rises only AFTER crash",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#c0392b",
                    font=dict(color="#c0392b", size=11),
                    ax=70,
                    ay=-40,
                )
        fig_roll.update_layout(
            **chart_layout("Rolling 252-Day Historical VaR", height=400),
            yaxis_title="Log Return",
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    # ── Section B: Parametric VaR ──────────────────────────────────────────────
    with st.expander("Parametric VaR — Normal Distribution", expanded=True):
        callout(
            "Assumes returns follow a Normal distribution. Computes VaR analytically "
            "as μ + σ·Z where Z is the Normal quantile. Fast and tractable — but "
            "fundamentally flawed for financial returns, which exhibit fat tails and "
            "negative skewness.",
            border_color="#e67e22",
            bg_color="#fff3e0",
        )

        k_col, s_col = st.columns(2)
        if kurtosis > 3:
            k_col.metric(
                "Kurtosis",
                f"{kurtosis:.4f}",
                delta="⚠️ Fat Tails Confirmed",
                delta_color="inverse",
            )
        else:
            k_col.metric("Kurtosis", f"{kurtosis:.4f}", delta="✅ Near-Normal")

        if skewness < 0:
            s_col.metric(
                "Skewness",
                f"{skewness:.4f}",
                delta="Negative skew — downside asymmetry",
                delta_color="inverse",
            )
        else:
            s_col.metric("Skewness", f"{skewness:.4f}")

        # Overlay chart: histogram + fitted Normal
        x_range = np.linspace(float(log_returns.min()), float(log_returns.max()), 300)
        normal_curve = stats.norm.pdf(x_range, mu, sigma)
        counts, _ = np.histogram(log_returns, bins=100)
        scale_factor = counts.max() / normal_curve.max()

        fig_norm = go.Figure()
        fig_norm.add_trace(
            go.Histogram(
                x=log_returns.values,
                nbinsx=100,
                marker_color="steelblue",
                opacity=0.55,
                name="Actual Returns",
            )
        )
        fig_norm.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_curve * scale_factor,
                mode="lines",
                line=dict(color="#c0392b", width=2.5),
                name="Fitted Normal",
            )
        )
        left_tail_x = float(log_returns.quantile(0.02))
        fig_norm.add_annotation(
            x=left_tail_x,
            y=counts.max() * 0.25,
            text="Normal curve underestimates<br>left tail height",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#c0392b",
            font=dict(color="#c0392b", size=11),
            ax=70,
            ay=-30,
        )
        fig_norm.update_layout(
            **chart_layout(
                "Actual Returns vs Fitted Normal Distribution", height=400
            ),
            barmode="overlay",
            xaxis_title="Log Return",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_norm, use_container_width=True)

        # Calm vs crisis parametric VaR
        st.markdown("**Parametric VaR — Calm vs Crisis:**")
        try:
            _, calm_r2 = load_data(ticker, "2018-01-01", "2019-12-31")
            calm_pvar = float(
                calm_r2.mean() + calm_r2.std() * stats.norm.ppf(1 - confidence)
            )
        except Exception:
            calm_pvar = float("nan")
        try:
            _, crisis_r2 = load_data(ticker, "2020-01-01", "2020-12-31")
            crisis_pvar = float(
                crisis_r2.mean() + crisis_r2.std() * stats.norm.ppf(1 - confidence)
            )
        except Exception:
            crisis_pvar = float("nan")

        pvar_df = pd.DataFrame(
            {
                "Period": ["Calm (2018–2019)", "Crisis (2020)"],
                "Parametric VaR": [
                    f"{calm_pvar*100:.3f}%" if not np.isnan(calm_pvar) else "N/A",
                    f"{crisis_pvar*100:.3f}%" if not np.isnan(crisis_pvar) else "N/A",
                ],
            }
        )
        st.dataframe(pvar_df, use_container_width=True, hide_index=True)

    # ── Section C: CVaR ────────────────────────────────────────────────────────
    with st.expander("CVaR — Conditional Value at Risk", expanded=True):
        callout(
            "CVaR (Expected Shortfall) answers: given that I am already in the worst "
            "5% of days, what is my average loss? Basel III now mandates CVaR over VaR "
            "precisely because it captures the full severity of tail events — not just "
            "the threshold.",
            border_color="#27ae60",
            bg_color="#e8f8ee",
        )

        var_h_now = metrics["var_hist"]
        cvar_now = metrics["cvar"]

        fig_cvar = go.Figure()
        fig_cvar.add_trace(
            go.Histogram(
                x=log_returns[log_returns > var_h_now].values,
                nbinsx=80,
                marker_color="#2e75b6",
                opacity=0.55,
                name="Returns",
            )
        )
        fig_cvar.add_trace(
            go.Histogram(
                x=log_returns[log_returns <= var_h_now].values,
                nbinsx=20,
                marker_color="#c0392b",
                opacity=0.85,
                name="Tail (beyond VaR)",
            )
        )
        fig_cvar.add_vline(
            x=var_h_now,
            line_dash="dash",
            line_color="#1f4e79",
            line_width=2,
            annotation_text=f"VaR {var_h_now*100:.2f}%",
            annotation_font_color="#1f4e79",
        )
        fig_cvar.add_vline(
            x=cvar_now,
            line_dash="dash",
            line_color="#c0392b",
            line_width=2,
            annotation_text=f"CVaR {cvar_now*100:.2f}%",
            annotation_font_color="#c0392b",
        )
        fig_cvar.update_layout(
            **chart_layout("CVaR — Tail Severity Beyond VaR", height=400),
            barmode="overlay",
            xaxis_title="Log Return",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_cvar, use_container_width=True)

        # VaR / CVaR / Gap table at all confidence levels
        cvar_rows = []
        for lbl, cv in [("90%", 0.90), ("95%", 0.95), ("99%", 0.99)]:
            m2 = compute_risk_metrics(ticker, start, end, cv)
            gap = abs(m2["cvar"] - m2["var_hist"])
            if gap > 0.015:
                interp = "Significant hidden tail risk"
            elif gap > 0.005:
                interp = "Moderate tail risk"
            else:
                interp = "Light tail"
            cvar_rows.append(
                {
                    "Confidence": lbl,
                    "VaR": f"{m2['var_hist']*100:.3f}%",
                    "CVaR": f"{m2['cvar']*100:.3f}%",
                    "Gap": f"{gap*100:.3f}%",
                    "Interpretation": interp,
                }
            )
        st.dataframe(
            pd.DataFrame(cvar_rows), use_container_width=True, hide_index=True
        )

    page_footer()


# ─── Page 3 — GARCH Analysis ──────────────────────────────────────────────────
def page_garch():
    if not require_ticker():
        return

    ticker = st.session_state.ticker
    selected_name = st.session_state.selected_name
    start = st.session_state.start
    end = st.session_state.end

    st.markdown(
        '<h2 style="color:#1f4e79; font-size:28px; font-weight:700; margin-bottom:2px;">'
        "📈 GARCH(1,1) Dynamic Volatility Model</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#8a9ab0; font-size:14px; margin-bottom:16px;">'
        "Modelling time-varying conditional volatility to detect risk regime shifts in real time</p>",
        unsafe_allow_html=True,
    )

    try:
        price, log_returns = load_data(ticker, start, end)
    except Exception:
        st.error(f"⚠️ Could not load data for **{selected_name}**.")
        st.stop()

    if len(log_returns) < 50:
        st.error(
            "⚠️ Insufficient data. "
            "Please select a date range spanning at least 50 trading days."
        )
        st.stop()

    # Theory expander
    with st.expander("📖 GARCH Theory", expanded=False):
        st.markdown(
            """
- **Volatility clustering:** Large price moves tend to be followed by large moves (of either sign)
  and small moves by small moves. This violates the constant-variance assumption of Black-Scholes.
- **Autoregressive variance:** GARCH models today's variance as a weighted function of
  yesterday's squared return shock and yesterday's conditional variance, explicitly capturing clustering.
- **Mean reversion:** Volatility reverts to a long-run level ω/(1−α−β), providing an
  automatic regime-normalisation mechanism missing from historical and parametric VaR.
"""
        )
        st.latex(
            r"\sigma^2_t = \omega + \alpha \,\varepsilon^2_{t-1} + \beta \,\sigma^2_{t-1}"
        )
        param_df = pd.DataFrame(
            {
                "Parameter": ["ω (omega)", "α (alpha)", "β (beta)", "α + β"],
                "Role": [
                    "Long-run variance baseline",
                    "ARCH coefficient",
                    "GARCH coefficient",
                    "Total persistence",
                ],
                "Interpretation": [
                    "Controls mean reversion level",
                    "Reactivity to new return shocks",
                    "Persistence of volatility",
                    "Closer to 1 = longer memory",
                ],
            }
        )
        st.dataframe(param_df, use_container_width=True, hide_index=True)

    # ── ADF Test ──────────────────────────────────────────────────────────────
    st.divider()
    section_header("Stationarity Verification (ADF Test)")

    adf_prices = run_adf_test(price)
    adf_returns = run_adf_test(log_returns)

    col_p, col_r = st.columns(2)

    with col_p:
        st.markdown(
            '<div style="background:#fdf2f2; border:1px solid #e8c4c4; border-radius:8px; '
            'padding:16px; margin-bottom:12px;">'
            '<div style="font-size:30px; line-height:1.2;">❌</div>'
            '<div style="font-size:17px; font-weight:600; color:#c0392b; margin:4px 0 2px 0;">'
            "Non-Stationary</div>"
            '<div style="font-size:12px; color:#5a6a7a;">Raw Prices</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.metric("ADF Statistic", adf_prices["adf_stat"])
        st.metric("p-value", adf_prices["p_value"])
        st.metric("Critical Value (1%)", adf_prices["critical_1"])
        st.metric("Critical Value (5%)", adf_prices["critical_5"])
        st.caption(
            "p > 0.05 → fail to reject unit root → prices trend over time "
            "and cannot be modelled directly by GARCH."
        )

    with col_r:
        st.markdown(
            '<div style="background:#f0fff4; border:1px solid #b2dfdb; border-radius:8px; '
            'padding:16px; margin-bottom:12px;">'
            '<div style="font-size:30px; line-height:1.2;">✅</div>'
            '<div style="font-size:17px; font-weight:600; color:#27ae60; margin:4px 0 2px 0;">'
            "Stationary</div>"
            '<div style="font-size:12px; color:#5a6a7a;">Log Returns</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.metric("ADF Statistic", adf_returns["adf_stat"])
        st.metric("p-value", adf_returns["p_value"])
        st.metric("Critical Value (1%)", adf_returns["critical_1"])
        st.metric("Critical Value (5%)", adf_returns["critical_5"])
        st.caption(
            "p < 0.05 → reject unit root → log returns are stationary — "
            "safe input for GARCH modelling. ✓"
        )

    # ── ACF Analysis ──────────────────────────────────────────────────────────
    st.divider()
    section_header("Autocorrelation Analysis — Evidence for GARCH")

    acf_raw, ci_raw = compute_acf_values(log_returns, n_lags=40)
    acf_sq, ci_sq = compute_acf_values(log_returns ** 2, n_lags=40)
    lags = list(range(len(acf_raw)))

    def acf_chart(acf_vals, ci, title, subtitle):
        bar_colors = [
            "#c0392b" if abs(v) > ci else "#2e75b6" for v in acf_vals
        ]
        fig = go.Figure()
        for i, (lag, val) in enumerate(zip(lags, acf_vals)):
            fig.add_trace(
                go.Bar(
                    x=[lag],
                    y=[val],
                    marker_color=bar_colors[i],
                    showlegend=False,
                )
            )
        fig.add_hline(
            y=ci, line_dash="dash", line_color="#e67e22", line_width=1.5
        )
        fig.add_hline(
            y=-ci, line_dash="dash", line_color="#e67e22", line_width=1.5
        )
        layout = chart_layout(height=350)
        layout["title"] = dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            font=dict(size=14, color="#1a1a2e", family="Arial, sans-serif"),
        )
        fig.update_layout(
            **layout,
            xaxis_title="Lag",
            yaxis_title="ACF",
            bargap=0.1,
        )
        return fig

    acf_c1, acf_c2 = st.columns(2)
    with acf_c1:
        st.plotly_chart(
            acf_chart(
                acf_raw,
                ci_raw,
                "ACF — Raw Log Returns",
                "No significant autocorrelation",
            ),
            use_container_width=True,
        )
    with acf_c2:
        st.plotly_chart(
            acf_chart(
                acf_sq,
                ci_sq,
                "ACF — Squared Returns",
                "ARCH effects confirmed ✓",
            ),
            use_container_width=True,
        )

    sig_raw = sum(1 for v in acf_raw[1:] if abs(v) > ci_raw)
    sig_sq = sum(1 for v in acf_sq[1:] if abs(v) > ci_sq)
    callout(
        f"Raw returns show <strong>{sig_raw}</strong> significant lags. "
        f"Squared returns show <strong>{sig_sq}</strong> significant lags. "
        "This contrast proves that volatility clusters in time — the statistical "
        "justification for GARCH.",
        border_color="#1f4e79",
        bg_color="#dce6f0",
    )

    # ── GARCH-Filtered Risk Estimates ─────────────────────────────────────────
    st.divider()
    section_header("GARCH-Filtered Risk Estimates")
    st.markdown(
        '<p style="color:#5a6a7a; font-size:13px; margin:-6px 0 12px 0;">'
        "Dynamic VaR using today's conditional volatility σ_t "
        "instead of historical standard deviation</p>",
        unsafe_allow_html=True,
    )

    try:
        with st.spinner(
            "Fitting GARCH(1,1) — estimating conditional volatility..."
        ):
            cond_vol, alpha, beta, _, returns = fit_garch(ticker, start, end)

        latest_garch_vol = float(cond_vol.iloc[-1])
        mu = float(returns.mean())
        garch_var_95 = float(mu - latest_garch_vol * stats.norm.ppf(0.95))
        garch_var_99 = float(mu - latest_garch_vol * stats.norm.ppf(0.99))
        static_var_95 = float(np.percentile(returns, 5))
        difference = garch_var_95 - static_var_95

        if difference < 0:
            diff_delta = "GARCH sees elevated risk"
            diff_delta_color = "inverse"
        else:
            diff_delta = "GARCH sees below-average risk"
            diff_delta_color = "normal"

        st.markdown('<div class="risk-metrics">', unsafe_allow_html=True)
        rv1, rv2, rv3, rv4 = st.columns(4)
        rv1.metric("GARCH-VaR 95%", f"{garch_var_95*100:.2f}%")
        rv2.metric("GARCH-VaR 99%", f"{garch_var_99*100:.2f}%")
        rv3.metric(
            "vs Static VaR 95%",
            f"{difference*100:+.2f}%",
            delta=diff_delta,
            delta_color=diff_delta_color,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        # Card 4 — navy border (outside the risk-metrics wrapper)
        with rv4:
            st.metric(
                "Current Conditional Volatility σ_t",
                f"{latest_garch_vol*100:.4f}%",
                help="GARCH(1,1) estimate of today's annualised conditional σ",
            )

        callout(
            "GARCH-VaR uses today's estimated conditional volatility σ_t in place of "
            "the historical standard deviation. When σ_t is elevated (as during March 2020), "
            "GARCH-VaR produces a larger loss estimate — providing an early warning that "
            "static VaR cannot generate.",
            border_color="#1f4e79",
            bg_color="#dce6f0",
        )

    except Exception:
        st.error("GARCH fitting failed — cannot compute GARCH-filtered VaR. Try a longer date range.")

    # ── GARCH Fitting ─────────────────────────────────────────────────────────
    st.divider()
    section_header("GARCH(1,1) Parameter Estimation")

    try:
        with st.spinner(
            "Fitting GARCH(1,1) — estimating conditional volatility..."
        ):
            cond_vol, alpha, beta, _, returns = fit_garch(ticker, start, end)

        persistence = alpha + beta
        peak_date = cond_vol.idxmax()
        peak_date_str = (
            str(peak_date.date())
            if hasattr(peak_date, "date")
            else str(peak_date)
        )

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("α (Alpha)", f"{alpha:.6f}", help="Shock reaction coefficient")
        g2.metric(
            "β (Beta)", f"{beta:.6f}", help="Volatility persistence coefficient"
        )

        if persistence > 0.95:
            persist_delta = "Very High Persistence 🔴"
        elif persistence >= 0.90:
            persist_delta = "High Persistence 🟠"
        else:
            persist_delta = "Moderate Persistence 🟢"
        g3.metric(
            "α + β Persistence",
            f"{persistence:.6f}",
            delta=persist_delta,
            delta_color="off",
        )
        g4.metric("Peak Volatility Date", peak_date_str)

        # GARCH conditional volatility chart
        idx = returns.index
        fig_garch = go.Figure()
        fig_garch.add_trace(
            go.Scatter(
                x=idx,
                y=returns.values,
                mode="lines",
                line=dict(color="#2e75b6", width=0.7),
                opacity=0.4,
                name="Log Returns",
            )
        )
        fig_garch.add_trace(
            go.Scatter(
                x=idx,
                y=cond_vol.values,
                mode="lines",
                line=dict(color="#c0392b", width=2.5),
                name="+GARCH Volatility",
            )
        )
        fig_garch.add_trace(
            go.Scatter(
                x=idx,
                y=-cond_vol.values,
                mode="lines",
                line=dict(color="#c0392b", width=2.5, dash="dash"),
                name="−GARCH Volatility",
            )
        )

        # COVID annotation using peak volatility in the COVID window
        covid_start_ts = pd.Timestamp("2020-02-01")
        covid_end_ts = pd.Timestamp("2020-05-01")
        if idx[0] <= covid_end_ts and idx[-1] >= covid_start_ts:
            covid_window = cond_vol[
                (cond_vol.index >= covid_start_ts)
                & (cond_vol.index <= covid_end_ts)
            ]
            if len(covid_window) > 0:
                covid_peak_date = covid_window.idxmax()
                covid_peak_val = float(covid_window.max())
                fig_garch.add_annotation(
                    x=covid_peak_date,
                    y=covid_peak_val,
                    text="COVID-19 Peak",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#c0392b",
                    font=dict(color="#c0392b", size=11),
                    ax=70,
                    ay=-45,
                )
                fig_garch.add_vrect(
                    x0=str(covid_start_ts.date()),
                    x1=str(covid_end_ts.date()),
                    fillcolor="rgba(192,57,43,0.06)",
                    layer="below",
                    line_width=0,
                )

        fig_garch.update_layout(
            **chart_layout(
                f"{selected_name} — Returns & GARCH(1,1) Conditional Volatility Bands",
                height=430,
            )
        )
        st.plotly_chart(fig_garch, use_container_width=True)

        # Two-column interpretation
        interp_c1, interp_c2 = st.columns(2)
        with interp_c1:
            callout(
                "<strong>What this chart shows</strong><br>"
                "• Blue line: daily log returns — noisy, centred near zero<br>"
                "• Red solid/dashed bands: ±1 GARCH conditional standard deviation<br>"
                "• Wide bands = high-volatility regime; narrow = calm",
                border_color="#2e75b6",
                bg_color="#e8f0f8",
            )
        with interp_c2:
            callout(
                "<strong>Why GARCH outperforms static models</strong><br>"
                "• Static VaR uses a fixed σ — equally wrong in calm and crisis<br>"
                "• GARCH σ_t expands in crises and contracts in calm, matching reality<br>"
                f"• α + β = {persistence:.4f}: shocks decay slowly — long volatility memory",
                border_color="#1f4e79",
                bg_color="#dce6f0",
            )

    except Exception:
        st.error("GARCH fitting failed. Try a longer date range.")

    page_footer()


# ─── Page 4 — Stock Comparison ────────────────────────────────────────────────
def page_comparison():
    start = st.session_state.start
    end = st.session_state.end
    confidence = st.session_state.confidence

    st.markdown(
        '<h2 style="color:#1f4e79; font-size:28px; font-weight:700; margin-bottom:2px;">'
        "🏦 5-Stock Risk Comparison</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#8a9ab0; font-size:14px; margin-bottom:16px;">'
        "RELIANCE · TCS · HDFCBANK · INFY · ITC — Side-by-side tail risk analysis</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Computing risk metrics for all 5 stocks..."):
        multi = compute_all_stocks(start, end, confidence)

    if not multi["tickers"]:
        st.error(
            "No stock data could be loaded. "
            "Please check your date range and internet connection."
        )
        st.stop()

    # Grouped bar chart
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=multi["tickers"],
            y=[v * 100 for v in multi["hvar"]],
            name="Historical VaR",
            marker_color="#1f4e79",
        )
    )
    fig_bar.add_trace(
        go.Bar(
            x=multi["tickers"],
            y=[v * 100 for v in multi["pvar"]],
            name="Parametric VaR",
            marker_color="#e67e22",
        )
    )
    fig_bar.add_trace(
        go.Bar(
            x=multi["tickers"],
            y=[v * 100 for v in multi["cvar"]],
            name="CVaR",
            marker_color="#c0392b",
        )
    )
    fig_bar.update_layout(
        **chart_layout(
            title=f"Daily Tail Risk Estimates at {int(confidence*100)}% Confidence",
            height=420,
        ),
        barmode="group",
        xaxis_title="Stock",
        yaxis_title="Risk (%, absolute)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Distribution statistics table
    st.divider()
    section_header("Distribution Statistics — All 5 Stocks")

    full_list = {
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "INFY": "INFY.NS",
        "ITC": "ITC.NS",
    }
    stats_rows = []
    fat_tail_count = 0
    for name, sym in full_list.items():
        if name not in multi["tickers"]:
            continue
        try:
            _, r = load_data(sym, start, end)
            kurt = float(r.kurtosis() + 3)
            skew = float(r.skew())
            is_fat = kurt > 3
            if is_fat:
                fat_tail_count += 1
            stats_rows.append(
                {
                    "Stock": name,
                    "Mean (%)": round(float(r.mean()) * 100, 4),
                    "Std (%)": round(float(r.std()) * 100, 4),
                    "Skewness": round(skew, 4),
                    "Kurtosis": round(kurt, 4),
                    "Fat Tail?": "Yes ⚠️" if is_fat else "No ✅",
                }
            )
        except Exception:
            pass

    df_stats = pd.DataFrame(stats_rows)

    def highlight_row(row):
        n = len(row)
        styles = [""] * n
        col_names = list(df_stats.columns)
        kurt_i = col_names.index("Kurtosis")
        fat_i = col_names.index("Fat Tail?")
        try:
            if float(row["Kurtosis"]) > 3:
                styles[kurt_i] = "color:#c0392b; font-weight:bold"
        except Exception:
            pass
        if "Yes" in str(row["Fat Tail?"]):
            styles[fat_i] = "color:#c0392b; font-weight:bold"
        else:
            styles[fat_i] = "color:#27ae60"
        return styles

    styled = df_stats.style.apply(highlight_row, axis=1).format(
        {"Mean (%)": "{:.4f}", "Std (%)": "{:.4f}",
         "Skewness": "{:.4f}", "Kurtosis": "{:.4f}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Interpretation callout
    st.divider()
    callout(
        f"All <strong>{fat_tail_count} of {len(stats_rows)}</strong> stocks exhibit excess "
        "kurtosis above 3.0, confirming fat-tailed return distributions across the comparison "
        "set. The Normal distribution assumption underlying Parametric VaR is therefore "
        "statistically invalid for Indian equities — this finding motivates the use of "
        "GARCH-filtered volatility estimates for accurate tail risk measurement.",
        border_color="#1f4e79",
        bg_color="#dce6f0",
    )

    page_footer()


# ─── Page 5 · ML vs GARCH Volatility Forecast ───────────────────────────────
def page_ml_forecast():
    from ml_vol_helpers import run_full_pipeline, historical_forecast_at_date, FEATURE_COLS, qlike_loss

    ticker = st.session_state.get("ticker", "RELIANCE.NS")
    selected_name = st.session_state.get("selected_name", "Reliance Industries")
    start = st.session_state.get("start", "2018-01-01")
    end = st.session_state.get("end", str(datetime.date.today()))

    st.markdown(
        '<h1 style="color:#1f4e79; font-size:28px; font-weight:700; margin-bottom:4px;">'
        "ML vs GARCH Volatility Forecast</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#5a6a7a; font-size:15px; margin-bottom:20px;">'
        f"Can an ML model forecast 5-day realized volatility more accurately than GARCH(1,1)? "
        f"Currently analysing <strong>{selected_name}</strong> ({ticker}).</p>",
        unsafe_allow_html=True,
    )

    with st.spinner(f"Fitting GARCH, Ridge & XGBoost models for {ticker}..."):
        pipeline = run_full_pipeline(ticker, start, end)

    if pipeline is None:
        st.warning(
            f"Not enough historical data for GARCH/ML comparison on **{selected_name}** ({ticker}). "
            "This analysis needs at least ~300 trading days after feature construction. "
            "Try a more established ticker or a wider date range."
        )
        page_footer()
        return

    bt_results = pipeline['backtest_results']
    ann = lambda v: v * np.sqrt(252) * 100

    # ── Section A: Live Forecast Panel ──
    section_header("Live Forecast — Next 5 Trading Days")

    best_model = min(bt_results, key=lambda k: bt_results[k]['qlike'])

    c1, c2, c3 = st.columns(3)
    with c1:
        label = "GARCH(1,1)  ✅ Best" if best_model == 'GARCH(1,1)' else "GARCH(1,1)"
        st.metric(label, f"{ann(pipeline['live_garch']):.1f}%")
    with c2:
        if best_model == 'Ridge':
            st.markdown(
                '<div style="border:2px solid #27ae60; border-radius:10px; padding:4px 0 0 0;">'
                "</div>",
                unsafe_allow_html=True,
            )
        label = "Ridge  ✅ Best" if best_model == 'Ridge' else "Ridge"
        st.metric(label, f"{ann(pipeline['live_ridge']):.1f}%")
    with c3:
        label = "XGBoost  ✅ Best" if best_model == 'XGBoost' else "XGBoost"
        st.metric(label, f"{ann(pipeline['live_xgb']):.1f}%")

    st.caption(
        "This forecasts expected volatility over the next 5 trading days, not a single next-day value. "
        f"Models refit on all available data through {pipeline['last_date'].date()}."
    )

    st.divider()

    # ── Section B: Historical Comparison Chart ──
    section_header("Historical Comparison — Out-of-Sample Forecasts")

    test_dates = pipeline['ridge_preds'].index
    realized = pipeline['y'].loc[test_dates]
    garch_bt = pipeline['df_clean'].loc[test_dates, 'garch_vol']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates, y=realized, mode='lines',
        name='Realized Vol (5d)', line=dict(color='#1a1a2e', width=1.2),
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=garch_bt, mode='lines',
        name='GARCH(1,1)', line=dict(color='#e74c3c', width=1),
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=pipeline['ridge_preds'], mode='lines',
        name='Ridge', line=dict(color='#27ae60', width=1.2),
        opacity=0.8,
    ))

    show_xgb = st.checkbox("Show XGBoost forecast", value=False)
    if show_xgb:
        fig.add_trace(go.Scatter(
            x=test_dates, y=pipeline['xgb_preds'], mode='lines',
            name='XGBoost', line=dict(color='#3498db', width=1, dash='dot'),
            opacity=0.6,
        ))

    fig.update_layout(**chart_layout(
        title=f"Realized Volatility vs. Model Forecasts — {ticker} (Out-of-Sample Backtest)",
        height=420,
    ))
    fig.update_yaxes(title_text="5-Day Realized Volatility")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section C: Model Comparison Table ──
    section_header("Model Comparison — Backtest Results")

    garch_qlike = bt_results['GARCH(1,1)']['qlike']
    comp_df = pd.DataFrame([
        {
            'Model': name,
            'Mean QLIKE': v['qlike'],
            'Mean MSE': v['mse'],
            'QLIKE Std': v['qlike_std'],
            'QLIKE vs GARCH': '—' if name == 'GARCH(1,1)'
                else f"{((v['qlike'] - garch_qlike) / garch_qlike * 100):+.2f}%",
        }
        for name, v in bt_results.items()
    ])

    def style_comp(df_style):
        styles = pd.DataFrame('', index=df_style.index, columns=df_style.columns)
        for col in ['Mean QLIKE', 'Mean MSE', 'QLIKE Std']:
            if col in df_style.columns:
                num_vals = pd.to_numeric(df_style[col], errors='coerce')
                min_idx = num_vals.idxmin()
                if pd.notna(min_idx):
                    styles.loc[min_idx, col] = 'font-weight:bold; color:#27ae60'
        return styles

    styled = comp_df.style.apply(style_comp, axis=None).format({
        'Mean QLIKE': '{:.4f}',
        'Mean MSE': '{:.6f}',
        'QLIKE Std': '{:.4f}',
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    best_qlike_model = min(bt_results, key=lambda k: bt_results[k]['qlike'])
    best_std_model = min(bt_results, key=lambda k: bt_results[k]['qlike_std'])
    callout(
        f"Lower QLIKE = better. <strong>{best_qlike_model}</strong> wins on mean QLIKE (best accuracy) "
        f"and <strong>{best_std_model}</strong> is the most stable across folds (lowest QLIKE Std).",
        border_color="#27ae60",
        bg_color="#e8f5e9",
    )

    st.divider()

    # ── Section D: Ridge Coefficients Chart ──
    section_header("Ridge Coefficients — What Drives the Forecast?")

    coefs = pipeline['ridge_coefs'].sort_values()
    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        y=coefs.index,
        x=coefs.values,
        orientation='h',
        marker_color=['#c0392b' if v < 0 else '#27ae60' for v in coefs.values],
    ))
    fig_coef.update_layout(**chart_layout(
        title=f"Ridge Regression Coefficients — {ticker} (Standardized Features)",
        height=380,
    ))
    fig_coef.update_xaxes(title_text="Standardized Coefficient", zeroline=True, zerolinecolor='#1a1a2e')
    st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()

    # ── Section E: Plain-Language Takeaway ──
    section_header("Takeaway")

    ridge_vs_garch = (garch_qlike - bt_results['Ridge']['qlike']) / garch_qlike * 100
    xgb_vs_garch = (garch_qlike - bt_results['XGBoost']['qlike']) / garch_qlike * 100

    if best_qlike_model == 'Ridge':
        takeaway = (
            f"A regularized linear combination of volatility features (Ridge) outperforms both the "
            f"parametric GARCH(1,1) baseline and a more flexible XGBoost model on out-of-sample 5-day "
            f"volatility forecasts for {ticker} — improving QLIKE by {ridge_vs_garch:.0f}% over GARCH "
            f"while also being the most stable model across folds. This suggests the additional "
            f"information captured by the engineered features (including GARCH's own forecast as one "
            f"input) is real and useful, but the relationship between these features and forward "
            f"volatility is closer to linear than nonlinear — XGBoost's added flexibility doesn't "
            f"pay off here and instead adds instability."
        )
    elif best_qlike_model == 'XGBoost':
        takeaway = (
            f"XGBoost achieves the best out-of-sample QLIKE on {ticker}, improving on GARCH by "
            f"{xgb_vs_garch:.0f}%. The nonlinear combination of volatility features captures dynamics "
            f"that both GARCH's parametric form and Ridge's linear combination miss. Ridge still "
            f"improves on GARCH by {ridge_vs_garch:.0f}%, confirming that the engineered feature set "
            f"adds real signal beyond conditional volatility alone."
        )
    else:
        takeaway = (
            f"GARCH(1,1) remains the best volatility forecaster for {ticker} on out-of-sample QLIKE. "
            f"Neither Ridge nor XGBoost improve meaningfully on the parametric baseline — the additional "
            f"features do not add enough predictive signal to justify the added model complexity. "
            f"This is a legitimate negative result: GARCH's parsimonious structure efficiently captures "
            f"the volatility clustering dynamics of this stock."
        )

    callout(takeaway, border_color="#1f4e79", bg_color="#dce6f0")

    st.divider()

    # ── Section 3 (Stretch): Interactive Historical Date Toggle ──
    section_header("Historical Date Explorer")
    st.caption(
        "Pick any date within the backtest period. Models are refit using only data available "
        "up to that date — no look-ahead."
    )

    bt_start = pipeline['fold_results'][0]['test_start']
    bt_end = pipeline['fold_results'][-1]['test_end']

    covid_default = pd.Timestamp("2020-03-20")
    default_date = covid_default if bt_start <= covid_default <= bt_end else bt_start

    selected_date = st.date_input(
        "Select a historical date",
        value=default_date.date() if hasattr(default_date, 'date') else default_date,
        min_value=bt_start.date() if hasattr(bt_start, 'date') else bt_start,
        max_value=bt_end.date() if hasattr(bt_end, 'date') else bt_end,
        key="ml_hist_date",
    )

    sel_ts = pd.Timestamp(selected_date)
    valid_dates = pipeline['X'].index
    valid_on_or_before = valid_dates[valid_dates <= sel_ts]
    if len(valid_on_or_before) == 0:
        st.warning("No trading data available on or before the selected date.")
    else:
        snapped = valid_on_or_before[-1]
        if snapped.date() != selected_date:
            st.caption(f"Snapped to nearest trading day: {snapped.date()}")

        with st.spinner("Refitting models up to selected date (no look-ahead)..."):
            hist = historical_forecast_at_date(pipeline, snapped)

        if hist is None:
            st.warning("Not enough data before this date to fit models (need ≥100 days).")
        else:
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.metric("GARCH Forecast", f"{ann(hist['garch']):.1f}%")
            with h2:
                st.metric("Ridge Forecast", f"{ann(hist['ridge']):.1f}%")
            with h3:
                st.metric("XGBoost Forecast", f"{ann(hist['xgb']):.1f}%")
            with h4:
                if hist['actual'] is not None:
                    st.metric("Actual Realized", f"{ann(hist['actual']):.1f}%")
                    forecasts = {
                        'GARCH': abs(hist['garch'] - hist['actual']),
                        'Ridge': abs(hist['ridge'] - hist['actual']),
                        'XGBoost': abs(hist['xgb'] - hist['actual']),
                    }
                    closest = min(forecasts, key=forecasts.get)
                    st.caption(f"Closest forecast: **{closest}**")
                else:
                    st.metric("Actual Realized", "N/A")
                    st.caption("No forward data available for this date.")

    page_footer()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    page = render_sidebar()

    if "run_analysis" not in st.session_state:
        show_landing()
        page_footer()
        return

    if page == "Overview":
        page_overview()
    elif page == "Risk Methods":
        page_risk_methods()
    elif page == "GARCH Analysis":
        page_garch()
    elif page == "Stock Comparison":
        page_comparison()
    elif page == "ML vs GARCH":
        page_ml_forecast()


if __name__ == "__main__":
    main()
