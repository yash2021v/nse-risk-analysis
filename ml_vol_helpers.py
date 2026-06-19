"""
Helper functions for the ML Volatility Forecasting Streamlit page.
Reuses load_data / fit_garch patterns from the main app.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import streamlit as st

FEATURE_COLS = [
    'return_lag1', 'return_lag2',
    'rolling_return_5d', 'rolling_return_20d',
    'rolling_vol_5d', 'rolling_vol_20d', 'rolling_vol_60d',
    'vol_ratio',
    'garch_vol', 'garch_vol_change',
    'max_drawdown_5d',
]

MIN_TRADING_DAYS = 300


def qlike_loss(y_true, y_pred):
    var_true = y_true ** 2
    var_pred = np.clip(y_pred ** 2, 1e-12, None)
    return np.mean(var_true / var_pred - np.log(var_true / var_pred) - 1)


def build_features(log_returns, garch_cond_vol):
    df = pd.DataFrame(index=log_returns.index)
    df['returns'] = log_returns
    df['return_lag1'] = df['returns'].shift(1)
    df['return_lag2'] = df['returns'].shift(2)
    df['rolling_return_5d'] = df['returns'].shift(1).rolling(5).mean()
    df['rolling_return_20d'] = df['returns'].shift(1).rolling(20).mean()
    df['rolling_vol_5d'] = df['returns'].shift(1).rolling(5).std()
    df['rolling_vol_20d'] = df['returns'].shift(1).rolling(20).std()
    df['rolling_vol_60d'] = df['returns'].shift(1).rolling(60).std()
    df['vol_ratio'] = df['rolling_vol_5d'] / df['rolling_vol_60d']
    df['garch_vol'] = garch_cond_vol.shift(1)
    df['garch_vol_change'] = df['garch_vol'].pct_change()
    df['max_drawdown_5d'] = df['returns'].shift(1).rolling(5).min().abs()
    return df


def fit_garch_raw(log_returns):
    scaled = log_returns * 100
    model = arch_model(scaled, vol='Garch', p=1, q=1,
                       mean='Constant', dist='normal')
    result = model.fit(disp='off')
    cond_vol = result.conditional_volatility / 100
    return cond_vol, result


@st.cache_data(ttl=3600)
def run_full_pipeline(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        price = data["Close"][ticker] if ticker in data["Close"].columns else data["Close"].iloc[:, 0]
    else:
        price = data["Close"]
    price = price.squeeze()
    log_returns = np.log(price / price.shift(1)).dropna()

    if len(log_returns) < MIN_TRADING_DAYS:
        return None

    cond_vol, garch_result = fit_garch_raw(log_returns)
    df = build_features(log_returns, cond_vol)

    target = log_returns.shift(-1).rolling(5).std()
    df['target'] = target

    df_clean = df.dropna(subset=FEATURE_COLS + ['target']).copy()
    X = df_clean[FEATURE_COLS]
    y = df_clean['target']

    if len(X) < MIN_TRADING_DAYS:
        return None

    # --- Backtest with TimeSeriesSplit ---
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_preds = pd.Series(dtype=float)
    ridge_preds = pd.Series(dtype=float)
    garch_preds = pd.Series(dtype=float)
    fold_results = []

    garch_fold_qlikes, ridge_fold_qlikes, xgb_fold_qlikes = [], [], []
    garch_fold_mses, ridge_fold_mses, xgb_fold_mses = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_sc, y_train)
        r_pred = np.clip(ridge.predict(X_te_sc), 1e-8, None)

        xgb = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        x_pred = np.clip(xgb.predict(X_test), 1e-8, None)

        g_pred = df_clean['garch_vol'].iloc[test_idx].values

        ridge_preds = pd.concat([ridge_preds, pd.Series(r_pred, index=y_test.index)])
        xgb_preds = pd.concat([xgb_preds, pd.Series(x_pred, index=y_test.index)])
        garch_preds = pd.concat([garch_preds, pd.Series(g_pred, index=y_test.index)])

        garch_fold_qlikes.append(qlike_loss(y_test.values, g_pred))
        ridge_fold_qlikes.append(qlike_loss(y_test.values, r_pred))
        xgb_fold_qlikes.append(qlike_loss(y_test.values, x_pred))
        garch_fold_mses.append(float(np.mean((y_test.values - g_pred) ** 2)))
        ridge_fold_mses.append(float(np.mean((y_test.values - r_pred) ** 2)))
        xgb_fold_mses.append(float(np.mean((y_test.values - x_pred) ** 2)))

        fold_results.append({
            'fold': fold + 1,
            'test_start': y_test.index[0],
            'test_end': y_test.index[-1],
        })

    backtest_results = {
        'GARCH(1,1)': {
            'qlike': float(np.mean(garch_fold_qlikes)),
            'mse': float(np.mean(garch_fold_mses)),
            'qlike_std': float(np.std(garch_fold_qlikes)),
        },
        'Ridge': {
            'qlike': float(np.mean(ridge_fold_qlikes)),
            'mse': float(np.mean(ridge_fold_mses)),
            'qlike_std': float(np.std(ridge_fold_qlikes)),
        },
        'XGBoost': {
            'qlike': float(np.mean(xgb_fold_qlikes)),
            'mse': float(np.mean(xgb_fold_mses)),
            'qlike_std': float(np.std(xgb_fold_qlikes)),
        },
    }

    # --- Full-data refit for live forecast ---
    scaler_full = StandardScaler()
    X_sc_full = scaler_full.fit_transform(X)
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_sc_full, y)

    xgb_full = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb_full.fit(X, y)

    last_row = X.iloc[[-1]]
    last_row_sc = scaler_full.transform(last_row)

    live_garch = float(df_clean['garch_vol'].iloc[-1])
    live_ridge = float(np.clip(ridge_full.predict(last_row_sc), 1e-8, None)[0])
    live_xgb = float(np.clip(xgb_full.predict(last_row), 1e-8, None)[0])

    ridge_coefs = pd.Series(ridge_full.coef_, index=FEATURE_COLS)

    return {
        'df_clean': df_clean,
        'X': X,
        'y': y,
        'ridge_preds': ridge_preds,
        'xgb_preds': xgb_preds,
        'garch_preds': garch_preds,
        'fold_results': fold_results,
        'backtest_results': backtest_results,
        'live_garch': live_garch,
        'live_ridge': live_ridge,
        'live_xgb': live_xgb,
        'ridge_coefs': ridge_coefs,
        'scaler_full': scaler_full,
        'ridge_full': ridge_full,
        'xgb_full': xgb_full,
        'log_returns': log_returns,
        'cond_vol': cond_vol,
        'last_date': df_clean.index[-1],
        'ticker': ticker,
    }


def historical_forecast_at_date(pipeline, date_idx):
    """Refit models using only data up to `date_idx` (no look-ahead)."""
    X = pipeline['X']
    y = pipeline['y']
    df_clean = pipeline['df_clean']

    mask = X.index <= date_idx
    if mask.sum() < 100:
        return None

    X_train = X.loc[mask]
    y_train = y.loc[mask]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_sc, y_train)

    xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)

    last_row = X_train.iloc[[-1]]
    last_row_sc = scaler.transform(last_row)

    garch_f = float(df_clean.loc[mask, 'garch_vol'].iloc[-1])
    ridge_f = float(np.clip(ridge.predict(last_row_sc), 1e-8, None)[0])
    xgb_f = float(np.clip(xgb.predict(last_row), 1e-8, None)[0])

    future_mask = y.index > date_idx
    future_vals = y.loc[future_mask]
    actual = float(future_vals.iloc[0]) if len(future_vals) > 0 else None

    return {
        'garch': garch_f,
        'ridge': ridge_f,
        'xgb': xgb_f,
        'actual': actual,
    }
