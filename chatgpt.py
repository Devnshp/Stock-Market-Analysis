# streamlit_stock_forecast_css.py
"""
Streamlit app that:
- Loads user CSV (Date, Close)
- Trains ARIMA, SARIMA, XGBoost, LSTM
- Backtests on last N days (configurable)
- Computes MAE, RMSE, R2
- Retrains on full history & forecasts next 90 business days
- Visualizes results, shows Prediction Summary & Risk Analysis
- Highlights best model visually
- Uses custom CSS for improved UI/UX
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------------------
# CSS Styling
# ---------------------------
st.set_page_config(layout="wide", page_title="Stock Forecast Hub", page_icon="📈")
st.markdown(
    """
    <style>
    /* Page background */
    .reportview-container {
        background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
        color: #e6eef8;
    }
    /* Sidebar */
    .css-1d391kg { background-color: rgba(255,255,255,0.03); }
    /* Headings */
    h1, h2, h3, .css-1v0mbdj {
        color: #e6eef8 !important;
    }
    /* Card-like containers */
    .card {
        background: rgba(255,255,255,0.04);
        padding: 14px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        margin-bottom: 12px;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#06b6d4,#7c3aed);
        color: white;
        border: none;
        padding: 8px 18px;
        border-radius: 8px;
    }
    /* Metrics */
    .metric-label { color: #bcd3ff !important; }
    .metric-value { color: #ffffff !important; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        # create demo set (business days)
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=600)
        # synthetic but plausible price series
        price = 150 + np.cumsum(np.random.normal(0, 1.2, len(idx)))
        df = pd.DataFrame({'Date': idx, 'Close': price})
    else:
        df = pd.read_csv(uploaded_file)
    if 'Date' not in df.columns:
        raise ValueError("CSV must include a 'Date' column")
    if 'Close' not in df.columns:
        raise ValueError("CSV must include a 'Close' column")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    df = df[['Close']].dropna()
    return df

def time_series_train_test_split(series, test_size):
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # r2 can be negative; handle constant arrays
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = np.nan
    return mae, rmse, r2

# ARIMA
def train_arima(train, order=(5,1,0), forecast_horizon=90):
    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=forecast_horizon)
    forecast.index = pd.bdate_range(start=train.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    return fitted, forecast

# SARIMA
def train_sarima(train, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_horizon=90):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=forecast_horizon).predicted_mean
    forecast.index = pd.bdate_range(start=train.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    return fitted, forecast

# XGBoost with lags
def create_lag_features(series, lags=10):
    df = pd.DataFrame(series).rename(columns={series.name:'Close'}).copy()
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df = df.dropna()
    return df

def train_xgboost(train, test, lags=10, forecast_horizon=90):
    full = pd.concat([train, test])
    df = create_lag_features(full, lags=lags)
    # split into train & test within df
    train_df = df[df.index < test.index[0]]
    test_df = df[df.index >= test.index[0]]
    X_train = train_df.drop('Close', axis=1).values
    y_train = train_df['Close'].values
    X_test = test_df.drop('Close', axis=1).values
    y_test = test_df['Close'].values
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, verbosity=0)
    model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_test, y_test)], verbose=False)
    preds_test = model.predict(X_test)
    # iterative forecasting
    last_window = df.drop('Close', axis=1).iloc[-1].values.reshape(1, -1)
    future_preds = []
    window = last_window.copy()
    for _ in range(forecast_horizon):
        p = model.predict(window)[0]
        future_preds.append(p)
        window = np.roll(window, -1)
        window[0, -1] = p
    future_index = pd.bdate_range(start=full.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    future_series = pd.Series(future_preds, index=future_index)
    preds_test_series = pd.Series(preds_test, index=test_df.index)
    return model, preds_test_series, future_series

# LSTM
def create_sequences(values, seq_len=30):
    X, y = [], []
    for i in range(len(values)-seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)

def train_lstm(train, test, seq_len=30, epochs=25, batch_size=16, forecast_horizon=90):
    full = pd.concat([train, test])
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full.values.reshape(-1,1)).flatten()
    train_scaled = scaled_full[:len(train)]
    test_scaled = scaled_full[len(train):]
    X_train, y_train = create_sequences(train_scaled, seq_len=seq_len)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    # model
    model = Sequential([
        LSTM(64, input_shape=(seq_len,1), return_sequences=False),
        Dropout(0.12),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=6, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)
    # create test sequences (tail of train + test)
    inputs = np.concatenate([train_scaled[-seq_len:], test_scaled])
    X_test_seq, y_test_seq = create_sequences(inputs, seq_len=seq_len)
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))
    preds_scaled = model.predict(X_test_seq).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()[:len(test)]
    preds_series = pd.Series(preds, index=test.index)
    # iterative forecast
    last_seq = scaled_full[-seq_len:].tolist()
    future_preds = []
    for _ in range(forecast_horizon):
        seq_arr = np.array(last_seq[-seq_len:]).reshape(1, seq_len, 1)
        p_scaled = float(model.predict(seq_arr).flatten()[0])
        last_seq.append(p_scaled)
        p = scaler.inverse_transform(np.array([[p_scaled]]))[0,0]
        future_preds.append(p)
    future_index = pd.bdate_range(start=full.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
    future_series = pd.Series(future_preds, index=future_index)
    return model, preds_series, future_series

def prediction_summary(pred_series):
    s = pred_series.dropna()
    starting_price = float(s.iloc[0])
    end_price = float(s.iloc[-1])
    change = end_price - starting_price
    change_pct = (change / starting_price) * 100 if starting_price != 0 else np.nan
    avg_pred = float(s.mean())
    vol = float(s.std())
    min_pred = float(s.min())
    max_pred = float(s.max())
    return {
        'Starting Price': starting_price,
        'Predicted End Price': end_price,
        'Predicted Change': change,
        'Predicted Change %': change_pct,
        'Average Predicted Price': avg_pred,
        'Prediction Volatility': vol,
        'Min Predicted Price': min_pred,
        'Max Predicted Price': max_pred
    }

# ---------------------------
# Streamlit UI layout
# ---------------------------
st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><h1>📈 Stock Forecasting Hub</h1><div style='color:#9aaed6'>ARIMA • SARIMA • XGBoost • LSTM</div></div>", unsafe_allow_html=True)
st.write("")  # spacing

with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload CSV (Date, Close)", type=['csv'])
    test_size = st.slider("Backtest horizon (days)", min_value=30, max_value=180, value=90, step=10)
    forecast_horizon = st.number_input("Forecast horizon (business days)", min_value=7, max_value=180, value=90)
    arima_order_str = st.text_input("ARIMA order (p,d,q)", value="5,1,0")
    sarima_order_str = st.text_input("SARIMA order (p,d,q)", value="1,1,1")
    sarima_seasonal_str = st.text_input("SARIMA seasonal (P,D,Q,s)", value="1,1,1,12")
    xgb_lags = st.slider("XGBoost lags", min_value=3, max_value=40, value=10)
    lstm_seq_len = st.slider("LSTM sequence length", min_value=5, max_value=60, value=30)
    lstm_epochs = st.slider("LSTM epochs", min_value=5, max_value=100, value=25)
    run_btn = st.button("Run models")

# Load data
try:
    data = load_data(uploaded_file)
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

col_main, col_side = st.columns([3,1])
with col_main:
    st.subheader("Price series")
    fig0 = px.line(data.reset_index(), x='Date', y='Close', title="Close Price (historical)")
    st.plotly_chart(fig0, use_container_width=True)
with col_side:
    st.metric("Last Close", f"{data['Close'].iloc[-1]:.2f}")

if run_btn:
    tic = time.time()
    st.info("Training models... this may take a few minutes (LSTM & XGBoost included).")
    # parse orders
    try:
        arima_order = tuple(int(x.strip()) for x in arima_order_str.split(','))
    except:
        arima_order = (5,1,0)
    try:
        sarima_order = tuple(int(x.strip()) for x in sarima_order_str.split(','))
        sarima_seasonal = tuple(int(x.strip()) for x in sarima_seasonal_str.split(','))
    except:
        sarima_order = (1,1,1)
        sarima_seasonal = (1,1,1,12)

    # split
    train, test = time_series_train_test_split(data['Close'], test_size=int(test_size))

    # ARIMA - backtest on test horizon
    try:
        arima_fit, arima_test_fore = train_arima(train, order=arima_order, forecast_horizon=len(test))
        arima_preds_test = pd.Series(arima_test_fore.values, index=test.index)
        arima_mae, arima_rmse, arima_r2 = compute_metrics(test.values, arima_preds_test.values)
        # retrain on full
        _, arima_future = train_arima(data['Close'], order=arima_order, forecast_horizon=int(forecast_horizon))
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")
        arima_mae = arima_rmse = arima_r2 = np.nan
        arima_future = pd.Series([np.nan]*int(forecast_horizon), index=pd.bdate_range(start=data.index[-1]+pd.Timedelta(days=1), periods=int(forecast_horizon)))

    # SARIMA
    try:
        sarima_fit, sarima_test_fore = train_sarima(train, order=sarima_order, seasonal_order=sarima_seasonal, forecast_horizon=len(test))
        sarima_preds_test = pd.Series(sarima_test_fore.values, index=test.index)
        sarima_mae, sarima_rmse, sarima_r2 = compute_metrics(test.values, sarima_preds_test.values)
        _, sarima_future = train_sarima(data['Close'], order=sarima_order, seasonal_order=sarima_seasonal, forecast_horizon=int(forecast_horizon))
    except Exception as e:
        st.warning(f"SARIMA failed: {e}")
        sarima_mae = sarima_rmse = sarima_r2 = np.nan
        sarima_future = pd.Series([np.nan]*int(forecast_horizon), index=pd.bdate_range(start=data.index[-1]+pd.Timedelta(days=1), periods=int(forecast_horizon)))

    # XGBoost
    try:
        xgb_model, xgb_preds_test, xgb_future = train_xgboost(train, test, lags=xgb_lags, forecast_horizon=int(forecast_horizon))
        xgb_mae, xgb_rmse, xgb_r2 = compute_metrics(test.values, xgb_preds_test.values)
    except Exception as e:
        st.warning(f"XGBoost failed: {e}")
        xgb_mae = xgb_rmse = xgb_r2 = np.nan
        xgb_future = pd.Series([np.nan]*int(forecast_horizon), index=pd.bdate_range(start=data.index[-1]+pd.Timedelta(days=1), periods=int(forecast_horizon)))

    # LSTM
    try:
        lstm_model, lstm_preds_test, lstm_future = train_lstm(train, test, seq_len=int(lstm_seq_len), epochs=int(lstm_epochs), forecast_horizon=int(forecast_horizon))
        lstm_mae, lstm_rmse, lstm_r2 = compute_metrics(test.values, lstm_preds_test.values)
    except Exception as e:
        st.warning(f"LSTM failed: {e}")
        lstm_mae = lstm_rmse = lstm_r2 = np.nan
        lstm_future = pd.Series([np.nan]*int(forecast_horizon), index=pd.bdate_range(start=data.index[-1]+pd.Timedelta(days=1), periods=int(forecast_horizon)))

    # metrics summary
    metrics_df = pd.DataFrame({
        "Model": ["ARIMA", "SARIMA", "XGBoost", "LSTM"],
        "MAE": [arima_mae, sarima_mae, xgb_mae, lstm_mae],
        "RMSE": [arima_rmse, sarima_rmse, xgb_rmse, lstm_rmse],
        "R2": [arima_r2, sarima_r2, xgb_r2, lstm_r2]
    })

    # best model by RMSE (lower better)
    best_idx = metrics_df['RMSE'].idxmin()
    best_model_name = metrics_df.loc[best_idx, 'Model']
    future_map = {"ARIMA": arima_future, "SARIMA": sarima_future, "XGBoost": xgb_future, "LSTM": lstm_future}
    best_future = future_map.get(best_model_name)

    elapsed = time.time() - tic
    st.success(f"Models trained. Best (by RMSE): {best_model_name}. Took {elapsed:.1f}s.")

    # ---------------------------
    # Plots: Metrics & Backtest
    # ---------------------------
    st.subheader("Model comparison (backtest)")
    # Build grouped bar chart for MAE, RMSE, R2
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(name="MAE", x=metrics_df['Model'], y=metrics_df['MAE'], marker=dict(line=dict(width=0))))
    fig_metrics.add_trace(go.Bar(name="RMSE", x=metrics_df['Model'], y=metrics_df['RMSE'], marker=dict(line=dict(width=0))))
    fig_metrics.add_trace(go.Bar(name="R2", x=metrics_df['Model'], y=metrics_df['R2'], marker=dict(line=dict(width=0))))
    fig_metrics.update_layout(title="MAE, RMSE, R² by Model (on backtest)", barmode='group', template='plotly_dark')
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Also create individual RMSE/MAE/R2 barplots (three-bar plot style)
    st.markdown("### Individual metric plots")
    cols = st.columns(3)
    with cols[0]:
        fig_rmse = px.bar(metrics_df, x='Model', y='RMSE', title='RMSE', text='RMSE')
        st.plotly_chart(fig_rmse, use_container_width=True)
    with cols[1]:
        fig_mae = px.bar(metrics_df, x='Model', y='MAE', title='MAE', text='MAE')
        st.plotly_chart(fig_mae, use_container_width=True)
    with cols[2]:
        fig_r2 = px.bar(metrics_df, x='Model', y='R2', title='R²', text='R2')
        st.plotly_chart(fig_r2, use_container_width=True)

    # Backtest predictions vs actual
    st.subheader(f"Backtest: last {int(test_size)} days — Actual vs Predictions")
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name='Actual', line=dict(width=3)))
    # add predictions if available
    try:
        fig_bt.add_trace(go.Scatter(x=arima_preds_test.index, y=arima_preds_test.values, mode='lines', name='ARIMA'))
    except: pass
    try:
        fig_bt.add_trace(go.Scatter(x=sarima_preds_test.index, y=sarima_preds_test.values, mode='lines', name='SARIMA'))
    except: pass
    try:
        fig_bt.add_trace(go.Scatter(x=xgb_preds_test.index, y=xgb_preds_test.values, mode='lines', name='XGBoost'))
    except: pass
    try:
        fig_bt.add_trace(go.Scatter(x=lstm_preds_test.index, y=lstm_preds_test.values, mode='lines', name='LSTM'))
    except: pass
    fig_bt.update_layout(template='plotly_dark')
    st.plotly_chart(fig_bt, use_container_width=True)

    # ---------------------------
    # Forecast visualization
    # ---------------------------
    st.subheader(f"{int(forecast_horizon)}-day Forecast — All models (best highlighted)")
    fig_fut = go.Figure()
    fig_fut.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical', line=dict(color='gray')))
    # to reduce clutter: plot lighter lines for models, thicker for best
    try:
        fig_fut.add_trace(go.Scatter(x=arima_future.index, y=arima_future.values, mode='lines', name='ARIMA Forecast', line=dict(width=1)))
    except: pass
    try:
        fig_fut.add_trace(go.Scatter(x=sarima_future.index, y=sarima_future.values, mode='lines', name='SARIMA Forecast', line=dict(width=1)))
    except: pass
    try:
        fig_fut.add_trace(go.Scatter(x=xgb_future.index, y=xgb_future.values, mode='lines', name='XGBoost Forecast', line=dict(width=1)))
    except: pass
    try:
        fig_fut.add_trace(go.Scatter(x=lstm_future.index, y=lstm_future.values, mode='lines', name='LSTM Forecast', line=dict(width=1)))
    except: pass
    # highlight best
    if best_future is not None:
        fig_fut.add_trace(go.Scatter(x=best_future.index, y=best_future.values, mode='lines', name=f'Best: {best_model_name}', line=dict(width=4)))
    fig_fut.update_layout(template='plotly_dark')
    st.plotly_chart(fig_fut, use_container_width=True)

    # ---------------------------
    # Summary & Risk Analysis
    # ---------------------------
    st.subheader("Prediction Summary (best model)")
    if best_future is None or best_future.isna().all():
        st.warning("Best model forecast unavailable.")
    else:
        summary = prediction_summary(best_future)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**Starting Price:** {summary['Starting Price']:.2f}")
            st.markdown(f"**Predicted End Price:** {summary['Predicted End Price']:.2f}")
            st.markdown(f"**Predicted Change:** {summary['Predicted Change']:.2f}")
            st.markdown(f"**Predicted Change %:** {summary['Predicted Change %']:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**Average Predicted Price:** {summary['Average Predicted Price']:.2f}")
            st.markdown(f"**Prediction Volatility (std):** {summary['Prediction Volatility']:.4f}")
            st.markdown(f"**Min Predicted Price:** {summary['Min Predicted Price']:.2f}")
            st.markdown(f"**Max Predicted Price:** {summary['Max Predicted Price']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Risk Analysis")
    curr_price = float(data['Close'].iloc[-1])
    if best_future is not None and not best_future.isna().all():
        pred_range = (summary['Min Predicted Price'], summary['Max Predicted Price'])
        st.markdown(f"- **Current Price:** {curr_price:.2f}")
        st.markdown(f"- **Prediction Range (min, max):** ({pred_range[0]:.2f}, {pred_range[1]:.2f})")
        st.markdown(f"- **Prediction Volatility (std):** {summary['Prediction Volatility']:.4f}")
    else:
        st.markdown("- Predictions are not available to compute risk analysis.")

    # Forecast table & download
    all_forecasts = pd.DataFrame({
        "Date": pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=int(forecast_horizon)),
        "ARIMA": arima_future.values if 'arima_future' in locals() else [np.nan]*int(forecast_horizon),
        "SARIMA": sarima_future.values if 'sarima_future' in locals() else [np.nan]*int(forecast_horizon),
        "XGBoost": xgb_future.values if 'xgb_future' in locals() else [np.nan]*int(forecast_horizon),
        "LSTM": lstm_future.values if 'lstm_future' in locals() else [np.nan]*int(forecast_horizon)
    })
    all_forecasts = all_forecasts.set_index('Date')
    st.dataframe(all_forecasts.style.format("{:.2f}"))

    csv = all_forecasts.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("Download forecast CSV", csv, file_name="stock_forecasts.csv", mime="text/csv")

    # Analysis section
    st.header("Analysis")
    st.markdown("""
    **What we did**
    - Trained four models (ARIMA, SARIMA, XGBoost, LSTM) and backtested on the last N days.
    - Metrics (MAE, RMSE, R²) computed on the backtest window.
    - Retrained each model on full history and produced a multi-model 90-day forecast. Best model selected by lowest RMSE.

    **Notes & caveats**
    - Forecasts are model outputs only — not financial advice.
    - ARIMA/SARIMA capture linear & seasonal structure; XGBoost captures non-linear effects via lag features; LSTM can learn complex temporal patterns but needs sufficient data.
    - Data quality, regime changes, macro news, and events are not captured here.
    """)

    st.header("Summary")
    st.markdown(f"- **Best model (by RMSE):** {best_model_name}")
    st.markdown(f"- **Backtest horizon:** {int(test_size)} days")
    st.markdown(f"- **Forecast horizon:** {int(forecast_horizon)} business days")

    st.balloons()
else:
    st.info("Upload data and click 'Run models' to train and forecast. Default demo data will be used if you don't upload a CSV.")
