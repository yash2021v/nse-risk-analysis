# NSE Risk Analysis Tool — with ML-Enhanced Volatility Forecasting

An interactive tool for tail-risk estimation and volatility forecasting on live NSE equity data — built to test, on real Indian market data, both classical risk methods (VaR, CVaR, GARCH) and whether machine learning can improve on GARCH's own volatility forecasts.

**Live app:** https://nse-risk-analyser.streamlit.app

---

## What this does

1. **Tail-risk comparison** — Historical VaR, Parametric VaR, and Conditional VaR (CVaR/Expected Shortfall) on any NSE stock, at configurable confidence levels.
2. **GARCH(1,1) volatility modeling** — conditional volatility estimation, with ADF stationarity testing and ACF analysis used to justify the GARCH specification (confirming ARCH effects exist before fitting).
3. **ML vs GARCH volatility forecasting** — tests whether a machine learning model, given GARCH's own forecast plus engineered volatility features, can out-predict GARCH's formula on **5-day forward realized volatility**. Evaluated out-of-sample with `TimeSeriesSplit` (5 folds) using QLIKE, the standard loss function in the volatility-forecasting literature.

---

## Key finding

| Model | Mean QLIKE | Mean MSE | QLIKE Std | QLIKE vs GARCH |
|---|---|---|---|---|
| GARCH(1,1) | 0.3676 | 0.000055 | 0.0810 | baseline |
| **Ridge** | **0.3006** | 0.000042 | **0.0554** | **−18.2%** |
| XGBoost | 0.3436 | 0.000055 | 0.1346 | −6.6% |
(Lower QLike value =better model)
A regularized linear model (Ridge) on engineered volatility features beats both the parametric GARCH(1,1) baseline and a more flexible XGBoost model — improving QLIKE by 18% over GARCH while also being the most stable model across folds (lowest QLIKE Std). This suggests the additional information captured by the feature set (including GARCH's own forecast as one input) is genuinely useful, but the relationship between these features and forward volatility is closer to linear than nonlinear — XGBoost's extra flexibility doesn't pay off here and instead adds instability.

This result is consistent with the established (and still debated) literature on GARCH-vs-ML volatility forecasting; this project is an empirical test on an underexplored slice of that literature — NSE equity data — rather than a claim of new theory.

---

## Methodology (ML layer)

- **Data:** live NSE daily price data via `yfinance`
- **Target:** 5-day forward realized volatility — `log_returns.shift(-1).rolling(5).std()`
- **Features (11):** lagged returns, rolling returns (5d/20d), rolling volatility (5d/20d/60d), vol ratio, GARCH conditional volatility and its % change, 5-day max drawdown — all properly lagged to avoid look-ahead leakage
- **Models compared:** GARCH(1,1) (baseline), Ridge Regression (sanity-check), XGBoost Regressor (main ML candidate)
- **Validation:** `TimeSeriesSplit`, 5 folds — no look-ahead bias
- **Metric:** QLIKE (primary, penalizes underestimating volatility more heavily), MSE (secondary)

---

## App pages

1. **Overview** — stock selector, price/returns summary
2. **Risk Methods** — Historical VaR, Parametric VaR, CVaR comparison
3. **GARCH Analysis** — conditional volatility, ADF test, ACF diagnostics
4. **Stock Comparison** — cross-stock risk metric comparison
5. **ML vs GARCH Forecast** — live 5-day-forward forecast (GARCH, Ridge, XGBoost), historical forecast-vs-realized chart, backtest results table, Ridge feature coefficients, and an interactive date picker to inspect what each model would have forecast on any historical date vs. what actually happened — all driven by the same stock selector used across the app

---

## Tech stack

Python · yfinance · pandas · NumPy · SciPy · statsmodels · arch (GARCH) · scikit-learn · XGBoost · Plotly · Streamlit

---

## Project structure

```
project1_risk_analysis/
├── app.py                          # Streamlit application (all pages)
├── ml_volatility_forecasting.py    # ML layer: features, models, backtest
├── requirements.txt
└── README.md
```

---

## Running locally

```bash
git clone <your-repo-url>
cd project1_risk_analysis
pip install -r requirements.txt
streamlit run app.py
```

---

## Limitations

- Backtest results shown above are for a single stock (RELIANCE.NS) over one sample period; the live app's stock selector lets results be checked across tickers, but per-ticker QLIKE comparisons will vary
- Equal treatment of model complexity costs is not factored in — Ridge's win is partly a statement about this specific feature set and target horizon, not a general claim that linear models beat ML for volatility forecasting
- 5-day horizon only; no transaction-cost or portfolio-level implications are modeled

---

## Author

Yashwanth V — BTech CS + FinTech
Linkedin - https://www.linkedin.com/in/yashwanth-velaga-852a2232a/
