# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import with error handling
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.error(f"Scikit-learn/XGBoost import error: {e}")
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    st.error(f"Statsmodels import error: {e}")
    STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"TensorFlow import error: {e}")
    TENSORFLOW_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Apple Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_features(data):
    """Create technical indicators and features for stock prediction"""
    try:
        df_features = data.copy()
        
        # Price-based features
        df_features['Price_Range'] = df_features['High'] - df_features['Low']
        df_features['Price_Change'] = df_features['Close'] - df_features['Open']
        df_features['Price_Change_Pct'] = (df_features['Close'] - df_features['Open']) / df_features['Open'] * 100
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df_features[f'MA_{window}'] = df_features['Close'].rolling(window=window).mean()
        
        # Volatility measures
        for window in [5, 10]:
            df_features[f'Volatility_{window}'] = df_features['Close'].rolling(window=window).std()
        
        # RSI
        delta = df_features['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_features['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df_features['Close'].ewm(span=12).mean()
        exp2 = df_features['Close'].ewm(span=26).mean()
        df_features['MACD'] = exp1 - exp2
        df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9).mean()
        
        # Volume features
        df_features['Volume_MA_5'] = df_features['Volume'].rolling(window=5).mean()
        df_features['Volume_Change'] = df_features['Volume'].pct_change()
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df_features[f'Close_Lag_{lag}'] = df_features['Close'].shift(lag)
        
        # Target variable
        df_features['Target_30d'] = df_features['Close'].shift(-30)
        
        return df_features
    except Exception as e:
        st.error(f"Error in feature engineering: {e}")
        return data

def create_lstm_dataset(data, lookback=60):
    """Create dataset for LSTM"""
    try:
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    except Exception as e:
        st.error(f"Error creating LSTM dataset: {e}")
        return np.array([]), np.array([])

def build_lstm_model(lookback):
    """Build and compile LSTM model"""
    try:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                    loss='mean_squared_error',
                    metrics=['mae'])
        return model
    except Exception as e:
        st.error(f"Error building LSTM model: {e}")
        return None

def predict_next_30_days_lstm(model, scaler, last_60_days, lookback=60):
    """Predict next 30 days using LSTM"""
    try:
        predictions = []
        current_data = last_60_days.copy()
        
        for i in range(30):
            current_scaled = scaler.transform(current_data.reshape(-1, 1))
            X_pred = current_scaled[-lookback:].reshape(1, lookback, 1)
            pred_scaled = model.predict(X_pred, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)[0, 0]
            
            predictions.append(pred)
            current_data = np.append(current_data[1:], pred)
        
        return predictions
    except Exception as e:
        st.error(f"Error in LSTM prediction: {e}")
        return []

def predict_next_30_days_arima(model, steps=30):
    """Predict next 30 days using ARIMA/SARIMA"""
    try:
        forecast = model.forecast(steps=steps)
        return forecast.values if hasattr(forecast, 'values') else forecast
    except Exception as e:
        st.error(f"Error in ARIMA prediction: {e}")
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Apple Stock Price Predictor</h1>', unsafe_allow_html=True)
    
    # Check dependencies
    if not all([SKLEARN_AVAILABLE, STATSMODELS_AVAILABLE, TENSORFLOW_AVAILABLE]):
        st.markdown("""
        <div class="warning-box">
        <h3>⚠️ Missing Dependencies</h3>
        <p>Some required packages are not available. Please check the requirements.txt file and ensure all dependencies are installed.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    st.sidebar.markdown("### Model Settings")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Stock Data (CSV)", type=['csv'])
    
    # Sample data for demo
    if uploaded_file is None:
        st.sidebar.markdown("### Try with Sample Data")
        if st.sidebar.button("Use Sample Data"):
            # Create sample data
            sample_dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
            np.random.seed(42)
            base_price = 100
            returns = np.random.normal(0.001, 0.02, len(sample_dates))
            prices = base_price * np.cumprod(1 + returns)
            
            sample_data = pd.DataFrame({
                'Date': sample_dates.strftime('%d-%m-%Y'),
                'Open': prices * 0.99,
                'High': prices * 1.02,
                'Low': prices * 0.98,
                'Close': prices,
                'Volume': np.random.randint(1000000, 5000000, len(sample_dates))
            })
            
            # Convert to CSV and create file-like object
            csv_data = sample_data.to_csv(index=False)
            uploaded_file = type('obj', (object,), {
                'read': lambda: csv_data,
                'name': 'sample_data.csv'
            })()
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.info("Please ensure your CSV contains: Date, Open, High, Low, Close, Volume")
                return
            
            # Data preprocessing
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
            if df['Date'].isnull().any():
                st.error("Date format error. Please use DD-MM-YYYY format.")
                return
                
            df = df.sort_values('Date').reset_index(drop=True)
            df.set_index('Date', inplace=True)
            
            # Feature engineering
            with st.spinner('Creating features...'):
                df_enhanced = create_features(df)
                if df_enhanced is not None:
                    df_enhanced = df_enhanced.dropna()
                else:
                    st.error("Feature engineering failed.")
                    return
            
            # Model selection
            st.sidebar.markdown("### Select Models to Run")
            run_arima = st.sidebar.checkbox("ARIMA", value=True) and STATSMODELS_AVAILABLE
            run_sarima = st.sidebar.checkbox("SARIMA", value=True) and STATSMODELS_AVAILABLE
            run_xgboost = st.sidebar.checkbox("XGBoost", value=True) and SKLEARN_AVAILABLE
            run_lstm = st.sidebar.checkbox("LSTM", value=True) and TENSORFLOW_AVAILABLE
            
            # Main content
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Models", "📈 Predictions", "📋 Results"])
            
            with tab1:
                st.header("Data Overview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Dataset Info")
                    st.write(f"**Period:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                    st.write(f"**Total Records:** {len(df):,}")
                    st.write(f"**Features Created:** {len(df_enhanced.columns) if df_enhanced is not None else 0}")
                    
                    st.subheader("Recent Data")
                    st.dataframe(df.tail(10), use_container_width=True)
                
                with col2:
                    st.subheader("Price Statistics")
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                        st.metric("All-time High", f"${df['High'].max():.2f}")
                        st.metric("All-time Low", f"${df['Low'].min():.2f}")
                    
                    with col2_2:
                        st.metric("Average Price", f"${df['Close'].mean():.2f}")
                        st.metric("Price Volatility", f"${df['Close'].std():.2f}")
                        st.metric("Total Volume", f"{df['Volume'].sum():,}")
                
                # Price chart
                st.subheader("Price History")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df.index, df['Close'], linewidth=1, color='#1f77b4')
                ax.set_title('Stock Price History')
                ax.set_ylabel('Price ($)')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with tab2:
                st.header("Model Training")
                
                if st.button("Train Models", type="primary"):
                    results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Prepare data
                    if df_enhanced is not None:
                        feature_columns = [col for col in df_enhanced.columns if col not in 
                                        ['Target_30d', 'Adj Close'] and not col.startswith('Close_Lag')]
                        lag_features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_5']
                        feature_columns.extend(lag_features)
                        
                        X = df_enhanced[feature_columns]
                        y = df_enhanced['Target_30d']
                        
                        split_index = int(0.8 * len(X))
                        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
                        
                        ts_data = df_enhanced['Close']
                        ts_train = ts_data.iloc[:split_index]
                        ts_test = ts_data.iloc[split_index:]
                    
                    progress_bar.progress(10)
                    
                    # ARIMA
                    if run_arima:
                        status_text.text("Training ARIMA model...")
                        try:
                            arima_model = ARIMA(ts_train, order=(1, 1, 1))
                            arima_fit = arima_model.fit()
                            arima_forecast = arima_fit.forecast(steps=len(ts_test))
                            
                            results['ARIMA'] = {
                                'model': arima_fit,
                                'predictions': arima_forecast,
                                'mae': mean_absolute_error(ts_test, arima_forecast),
                                'rmse': np.sqrt(mean_squared_error(ts_test, arima_forecast)),
                                'r2': r2_score(ts_test, arima_forecast)
                            }
                            st.success("✅ ARIMA trained successfully")
                        except Exception as e:
                            st.error(f"❌ ARIMA failed: {str(e)}")
                    
                    progress_bar.progress(40)
                    
                    # SARIMA
                    if run_sarima:
                        status_text.text("Training SARIMA model...")
                        try:
                            sarima_model = SARIMAX(ts_train, 
                                                 order=(1, 1, 1), 
                                                 seasonal_order=(1, 1, 1, 12),
                                                 enforce_stationarity=False,
                                                 enforce_invertibility=False)
                            sarima_fit = sarima_model.fit(disp=False)
                            sarima_forecast = sarima_fit.forecast(steps=len(ts_test))
                            
                            results['SARIMA'] = {
                                'model': sarima_fit,
                                'predictions': sarima_forecast,
                                'mae': mean_absolute_error(ts_test, sarima_forecast),
                                'rmse': np.sqrt(mean_squared_error(ts_test, sarima_forecast)),
                                'r2': r2_score(ts_test, sarima_forecast)
                            }
                            st.success("✅ SARIMA trained successfully")
                        except Exception as e:
                            st.error(f"❌ SARIMA failed: {str(e)}")
                    
                    progress_bar.progress(60)
                    
                    # XGBoost
                    if run_xgboost:
                        status_text.text("Training XGBoost model...")
                        try:
                            xgb_model = xgb.XGBRegressor(
                                n_estimators=100,
                                learning_rate=0.1,
                                max_depth=6,
                                random_state=42
                            )
                            xgb_model.fit(X_train, y_train)
                            xgb_predictions = xgb_model.predict(X_test)
                            
                            results['XGBoost'] = {
                                'model': xgb_model,
                                'predictions': xgb_predictions,
                                'mae': mean_absolute_error(y_test, xgb_predictions),
                                'rmse': np.sqrt(mean_squared_error(y_test, xgb_predictions)),
                                'r2': r2_score(y_test, xgb_predictions),
                                'feature_importance': pd.DataFrame({
                                    'feature': feature_columns,
                                    'importance': xgb_model.feature_importances_
                                }).sort_values('importance', ascending=False)
                            }
                            st.success("✅ XGBoost trained successfully")
                        except Exception as e:
                            st.error(f"❌ XGBoost failed: {str(e)}")
                    
                    progress_bar.progress(80)
                    
                    # LSTM
                    if run_lstm:
                        status_text.text("Training LSTM model...")
                        try:
                            close_prices = df_enhanced['Close'].values
                            scaler_lstm = MinMaxScaler(feature_range=(0, 1))
                            close_scaled = scaler_lstm.fit_transform(close_prices.reshape(-1, 1))
                            
                            lookback = 30  # Reduced for faster training
                            X_lstm, y_lstm = create_lstm_dataset(close_scaled, lookback)
                            
                            if len(X_lstm) > 0:
                                split_idx = int(0.8 * len(X_lstm))
                                X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
                                y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
                                
                                lstm_model = build_lstm_model(lookback)
                                
                                if lstm_model is not None:
                                    # Train with fewer epochs for demo
                                    lstm_model.fit(X_train_lstm, y_train_lstm,
                                                 epochs=20, batch_size=32,
                                                 validation_data=(X_test_lstm, y_test_lstm),
                                                 verbose=0, shuffle=False)
                                    
                                    lstm_predictions = lstm_model.predict(X_test_lstm, verbose=0)
                                    lstm_predictions = scaler_lstm.inverse_transform(lstm_predictions)
                                    y_test_actual = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1, 1))
                                    
                                    results['LSTM'] = {
                                        'model': lstm_model,
                                        'predictions': lstm_predictions.flatten(),
                                        'mae': mean_absolute_error(y_test_actual, lstm_predictions),
                                        'rmse': np.sqrt(mean_squared_error(y_test_actual, lstm_predictions)),
                                        'r2': r2_score(y_test_actual, lstm_predictions),
                                        'scaler': scaler_lstm
                                    }
                                    st.success("✅ LSTM trained successfully")
                        except Exception as e:
                            st.error(f"❌ LSTM failed: {str(e)}")
                    
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.df_enhanced = df_enhanced
                    st.session_state.feature_columns = feature_columns if 'feature_columns' in locals() else []
                
                else:
                    st.info("Click 'Train Models' to start training selected models.")
            
            # ... (rest of the code for tabs 3 and 4 remains similar but with error handling)
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Please check your CSV file format and try again.")
    
    else:
        # Welcome page
        st.markdown("""
        ## Welcome to Stock Price Predictor! 📈
        
        This application uses machine learning to predict stock prices for the next 30 days.
        
        ### How to use:
        1. **Upload your data** in the sidebar (CSV format)
        2. **Train models** in the Models tab
        3. **Generate predictions** in the Predictions tab
        4. **Analyze results** in the Results tab
        
        ### Required CSV format:
        - `Date` (DD-MM-YYYY)
        - `Open`, `High`, `Low`, `Close`
        - `Volume`
        
        *Click "Use Sample Data" in the sidebar to try the app immediately!*
        """)

if __name__ == "__main__":
    main()