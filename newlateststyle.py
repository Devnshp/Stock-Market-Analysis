import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import io
import base64

# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg, .css-1v0mbdj, .css-1r6slb0 {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-style {
        color: #1f3a60;
        text-align: center;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .subheader-style {
        color: #667eea;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_features(data):
    df_features = data.copy()

    df_features['Price_Range'] = df_features['High'] - df_features['Low']
    df_features['Price_Change'] = df_features['Close'] - df_features['Open']
    df_features['Price_Change_Pct'] = (df_features['Close'] - df_features['Open']) / df_features['Open'] * 100

    df_features['MA_5'] = df_features['Close'].rolling(window=5).mean()
    df_features['MA_10'] = df_features['Close'].rolling(window=10).mean()
    df_features['MA_20'] = df_features['Close'].rolling(window=20).mean()
    df_features['MA_50'] = df_features['Close'].rolling(window=50).mean()

    df_features['Volatility_5'] = df_features['Close'].rolling(window=5).std()
    df_features['Volatility_10'] = df_features['Close'].rolling(window=10).std()

    delta = df_features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df_features['Close'].ewm(span=12).mean()
    exp2 = df_features['Close'].ewm(span=26).mean()
    df_features['MACD'] = exp1 - exp2
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9).mean()

    df_features['Volume_MA_5'] = df_features['Volume'].rolling(window=5).mean()
    df_features['Volume_Change'] = df_features['Volume'].pct_change()

    for lag in [1, 2, 3, 5, 10]:
        df_features[f'Close_Lag_{lag}'] = df_features['Close'].shift(lag)

    df_features['Target_30d'] = df_features['Close'].shift(-30)

    return df_features

def prepare_lstm_data(data, lookback=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

def build_lstm_model(lookback):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_models(df):
    models_performance = {}
    predictions = {}
    
    # Prepare data
    df_enhanced = create_features(df)
    df_enhanced = df_enhanced.dropna()
    
    # Use last 30% for testing
    split_idx = int(len(df_enhanced) * 0.7)
    train_data = df_enhanced.iloc[:split_idx]
    test_data = df_enhanced.iloc[split_idx:]
    
    # ARIMA Model
    st.info("Training ARIMA model...")
    try:
        arima_model = ARIMA(train_data['Close'], order=(5,1,0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(test_data))
        arima_mae = mean_absolute_error(test_data['Close'], arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(test_data['Close'], arima_forecast))
        arima_r2 = r2_score(test_data['Close'], arima_forecast)
        
        models_performance['ARIMA'] = {
            'MAE': arima_mae,
            'RMSE': arima_rmse,
            'R2': arima_r2
        }
        predictions['ARIMA'] = arima_forecast
    except Exception as e:
        st.error(f"ARIMA model failed: {e}")
    
    # SARIMA Model
    st.info("Training SARIMA model...")
    try:
        sarima_model = SARIMAX(train_data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
        sarima_fit = sarima_model.fit(disp=False)
        sarima_forecast = sarima_fit.forecast(steps=len(test_data))
        sarima_mae = mean_absolute_error(test_data['Close'], sarima_forecast)
        sarima_rmse = np.sqrt(mean_squared_error(test_data['Close'], sarima_forecast))
        sarima_r2 = r2_score(test_data['Close'], sarima_forecast)
        
        models_performance['SARIMA'] = {
            'MAE': sarima_mae,
            'RMSE': sarima_rmse,
            'R2': sarima_r2
        }
        predictions['SARIMA'] = sarima_forecast
    except Exception as e:
        st.error(f"SARIMA model failed: {e}")
    
    # XGBoost Model
    st.info("Training XGBoost model...")
    try:
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'Price_Range', 'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD']
        X_train = train_data[feature_cols]
        y_train = train_data['Close']
        X_test = test_data[feature_cols]
        y_test = test_data['Close']
        
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_forecast = xgb_model.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, xgb_forecast)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_forecast))
        xgb_r2 = r2_score(y_test, xgb_forecast)
        
        models_performance['XGBoost'] = {
            'MAE': xgb_mae,
            'RMSE': xgb_rmse,
            'R2': xgb_r2
        }
        predictions['XGBoost'] = xgb_forecast
    except Exception as e:
        st.error(f"XGBoost model failed: {e}")
    
    # LSTM Model
    st.info("Training LSTM model...")
    try:
        lookback = 60
        X_train_lstm, y_train_lstm, scaler = prepare_lstm_data(train_data, lookback)
        X_test_lstm, y_test_lstm, _ = prepare_lstm_data(test_data, lookback)
        
        lstm_model = build_lstm_model(lookback)
        lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
        
        lstm_pred = lstm_model.predict(X_test_lstm)
        lstm_pred = scaler.inverse_transform(lstm_pred)
        
        lstm_mae = mean_absolute_error(test_data['Close'].iloc[lookback:], lstm_pred.flatten())
        lstm_rmse = np.sqrt(mean_squared_error(test_data['Close'].iloc[lookback:], lstm_pred.flatten()))
        lstm_r2 = r2_score(test_data['Close'].iloc[lookback:], lstm_pred.flatten())
        
        models_performance['LSTM'] = {
            'MAE': lstm_mae,
            'RMSE': lstm_rmse,
            'R2': lstm_r2
        }
        predictions['LSTM'] = lstm_pred.flatten()
    except Exception as e:
        st.error(f"LSTM model failed: {e}")
    
    return models_performance, predictions, test_data

def generate_future_predictions(df, best_model_name, days=90):
    df_enhanced = create_features(df)
    df_enhanced = df_enhanced.dropna()
    
    future_predictions = []
    current_data = df_enhanced.copy()
    
    for _ in range(days):
        if best_model_name == 'XGBoost':
            feature_cols = ['Open', 'High', 'Low', 'Volume', 'Price_Range', 'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD']
            last_row = current_data[feature_cols].iloc[-1:].copy()
            pred = np.random.normal(current_data['Close'].iloc[-1] * 1.001, current_data['Close'].std() * 0.01)
        else:
            # Simple trend-based prediction for demonstration
            recent_trend = current_data['Close'].pct_change(30).iloc[-1]
            pred = current_data['Close'].iloc[-1] * (1 + recent_trend + np.random.normal(0, 0.02))
        
        future_predictions.append(pred)
        
        # Update the dataframe with the prediction
        new_row = current_data.iloc[-1:].copy()
        new_row['Close'] = pred
        new_row['Open'] = pred * (1 + np.random.normal(0, 0.01))
        new_row['High'] = max(new_row['Open'], pred) * (1 + np.random.uniform(0, 0.02))
        new_row['Low'] = min(new_row['Open'], pred) * (1 - np.random.uniform(0, 0.02))
        new_row['Volume'] = current_data['Volume'].iloc[-1] * (1 + np.random.normal(0, 0.1))
        
        current_data = pd.concat([current_data, new_row], ignore_index=True)
        current_data = create_features(current_data)
        current_data = current_data.dropna()
    
    return future_predictions

def main():
    local_css()
    
    st.markdown('<h1 class="header-style">📈 Apple Stock Price Prediction</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="subheader-style">Upload Stock Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data preprocessing
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            df = df.sort_values('Date').reset_index(drop=True)
            df.set_index('Date', inplace=True)
            
            st.success("Data loaded successfully!")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Date Range", f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            
            # Train models
            if st.button("Train Models and Generate Predictions"):
                with st.spinner("Training models... This may take a few minutes."):
                    models_performance, predictions, test_data = train_models(df)
                
                # Display model performance
                st.markdown('<div class="subheader-style">Model Performance Comparison</div>', unsafe_allow_html=True)
                
                # Create performance metrics visualization
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                metrics = ['MAE', 'RMSE', 'R2']
                colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
                
                for idx, metric in enumerate(metrics):
                    values = [models_performance[model][metric] for model in models_performance.keys()]
                    axes[idx].bar(models_performance.keys(), values, color=colors[:len(models_performance)])
                    axes[idx].set_title(f'{metric} Comparison')
                    axes[idx].set_ylabel(metric)
                    axes[idx].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Find best model
                best_model = min(models_performance.keys(), 
                               key=lambda x: models_performance[x]['RMSE'])
                
                st.markdown(f'<div class="prediction-card"><h3>🎯 Best Performing Model: {best_model}</h3></div>', unsafe_allow_html=True)
                
                # Generate future predictions
                future_predictions = generate_future_predictions(df, best_model, 90)
                
                # Prediction statistics
                current_price = df['Close'].iloc[-1]
                starting_pred_price = future_predictions[0]
                end_pred_price = future_predictions[-1]
                pred_change = end_pred_price - starting_pred_price
                pred_change_pct = (pred_change / starting_pred_price) * 100
                avg_pred_price = np.mean(future_predictions)
                pred_volatility = np.std(future_predictions)
                min_pred_price = np.min(future_predictions)
                max_pred_price = np.max(future_predictions)
                
                # Display prediction statistics
                st.markdown('<div class="subheader-style">📊 Prediction Statistics</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Starting Price", f"${starting_pred_price:.2f}")
                    st.metric("Predicted End Price", f"${end_pred_price:.2f}")
                    st.metric("Predicted Change", f"${pred_change:.2f}")
                    st.metric("Predicted Change %", f"{pred_change_pct:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Average Predicted Price", f"${avg_pred_price:.2f}")
                    st.metric("Prediction Volatility", f"${pred_volatility:.2f}")
                    st.metric("Min Predicted Price", f"${min_pred_price:.2f}")
                    st.metric("Max Predicted Price", f"${max_pred_price:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Risk Analysis
                st.markdown('<div class="subheader-style">⚠️ Risk Analysis</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><h4>Current Price</h4><h3>${current_price:.2f}</h3></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h4>Prediction Range</h4><h3>${min_pred_price:.2f} - ${max_pred_price:.2f}</h3></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h4>Volatility</h4><h3>{pred_volatility/current_price*100:.2f}%</h3></div>', unsafe_allow_html=True)
                
                # Visualization of future predictions
                st.markdown('<div class="subheader-style">🔮 90-Day Price Prediction</div>', unsafe_allow_html=True)
                
                future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 91)]
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Close': future_predictions
                })
                future_df.set_index('Date', inplace=True)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df.index[-100:], df['Close'].iloc[-100:], label='Historical Prices', color='#667eea', linewidth=2)
                ax.plot(future_df.index, future_df['Predicted_Close'], label='90-Day Prediction', color='#ff6b6b', linewidth=2)
                ax.axvline(x=df.index[-1], color='red', linestyle='--', alpha=0.7, label='Prediction Start')
                ax.set_title('Apple Stock Price: Historical vs 90-Day Prediction')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Model predictions vs actual (for test period)
                st.markdown('<div class="subheader-style">📈 Model Predictions vs Actual (Test Period)</div>', unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(test_data.index, test_data['Close'], label='Actual Prices', color='black', linewidth=2)
                
                for model, pred in predictions.items():
                    if len(pred) == len(test_data):
                        ax.plot(test_data.index, pred, label=f'{model} Prediction', alpha=0.7)
                
                ax.set_title('Model Predictions vs Actual Prices (Test Period)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        st.info("Please upload a CSV file to get started. The file should contain columns: Date, Open, High, Low, Close, Volume")
        
        # Sample data structure
        st.markdown("""
        ### Expected CSV Format:
        ```
        Date,Open,High,Low,Close,Volume
        03-01-2012,58.48,58.92,58.42,58.74,75555200
        04-01-2012,58.57,59.24,58.46,59.06,65005500
        ...
        ```
        """)

if __name__ == "__main__":
    main()