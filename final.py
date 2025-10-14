import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Stock Price Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .prediction-positive {
        color: #00cc96;
        font-weight: bold;
        font-size: 1.1em;
    }
    .prediction-negative {
        color: #ef553b;
        font-weight: bold;
        font-size: 1.1em;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stButton button {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .stButton button:hover {
        background: linear-gradient(45deg, #ff7f0e, #1f77b4);
        color: white;
    }
    .risk-high { color: #ef553b; font-weight: bold; }
    .risk-medium { color: #fba905; font-weight: bold; }
    .risk-low { color: #00cc96; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample stock data if no file is uploaded"""
    dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
    np.random.seed(42)
    
    # Generate realistic stock price data
    prices = [100.0]
    for i in range(1, len(dates)):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df

def create_features(df):
    """Create technical indicators and features based on Apple_forecast.ipynb"""
    df_features = df.copy()

    # Basic price features
    df_features['Price_Range'] = df_features['High'] - df_features['Low']
    df_features['Price_Change'] = df_features['Close'] - df_features['Open']
    df_features['Price_Change_Pct'] = (df_features['Close'] - df_features['Open']) / df_features['Open'] * 100

    # Moving averages
    for window in [5, 10, 20, 50]:
        df_features[f'MA_{window}'] = df_features['Close'].rolling(window=window).mean()

    # Volatility
    df_features['Volatility_5'] = df_features['Close'].rolling(window=5).std()
    df_features['Volatility_10'] = df_features['Close'].rolling(window=10).std()

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

    # Target variable (future price)
    df_features['Target_30d'] = df_features['Close'].shift(-30)

    return df_features.dropna()

def train_arima_model(train_data, future_days=90):
    """Train ARIMA model based on Apple_forecast.ipynb"""
    try:
        # Use auto ARIMA to find best parameters or use fixed ones
        model = ARIMA(train_data['Close'], order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=future_days)
        return np.array(forecast)
    except Exception as e:
        st.error(f"ARIMA Error: {e}")
        # Fallback to simple prediction
        last_price = train_data['Close'].iloc[-1]
        return np.full(future_days, last_price)

def train_sarima_model(train_data, future_days=90):
    """Train SARIMA model based on Apple_forecast.ipynb"""
    try:
        model = SARIMAX(train_data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 30))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=future_days)
        return np.array(forecast)
    except Exception as e:
        st.error(f"SARIMA Error: {e}")
        # Fallback to simple prediction
        last_price = train_data['Close'].iloc[-1]
        return np.full(future_days, last_price)

def train_xgboost_model(train_data, future_days=90):
    """Train XGBoost model based on Apple_forecast.ipynb"""
    try:
        # Select features (excluding target and date columns)
        feature_columns = ['Open', 'High', 'Low', 'Volume', 'Price_Range', 
                          'Price_Change_Pct', 'MA_5', 'MA_10', 'MA_20', 'MA_50',
                          'Volatility_5', 'Volatility_10', 'RSI', 'MACD', 'MACD_Signal',
                          'Volume_MA_5', 'Volume_Change']
        
        # Add lag features
        for lag in [1, 2, 3, 5, 10]:
            feature_columns.append(f'Close_Lag_{lag}')
        
        available_features = [f for f in feature_columns if f in train_data.columns]
        
        X = train_data[available_features].fillna(method='ffill').fillna(method='bfill')
        y = train_data['Close']
        
        # Remove any rows with NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            return None
            
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=10
        )
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Generate future predictions using recursive forecasting
        last_features = X.iloc[-1:].copy()
        predictions = []
        
        for i in range(future_days):
            # Prepare current features
            current_features = last_features.copy()
            current_scaled = scaler.transform(current_features)
            
            # Make prediction
            pred = model.predict(current_scaled)[0]
            predictions.append(pred)
            
            # Update features for next prediction
            if i < future_days - 1:
                # Update price-related features
                last_features['Open'] = pred * (1 + np.random.normal(0, 0.001))
                last_features['High'] = pred * (1 + abs(np.random.normal(0, 0.005)))
                last_features['Low'] = pred * (1 - abs(np.random.normal(0, 0.005)))
                last_features['Close'] = pred
                
                # Update moving averages and other indicators
                for window in [5, 10, 20, 50]:
                    if f'MA_{window}' in last_features.columns:
                        last_features[f'MA_{window}'] = pred  # Simplified
                
                # Update lag features
                for lag in [1, 2, 3, 5, 10]:
                    if f'Close_Lag_{lag}' in last_features.columns:
                        if lag == 1:
                            last_features[f'Close_Lag_{lag}'] = train_data['Close'].iloc[-1]
                        else:
                            last_features[f'Close_Lag_{lag}'] = last_features[f'Close_Lag_{lag-1}']
        
        return np.array(predictions)
    except Exception as e:
        st.error(f"XGBoost Error: {e}")
        return None

def train_lstm_model(train_data, future_days=90):
    """Train LSTM model based on Apple_forecast.ipynb"""
    try:
        # Use more data points for better training
        data = train_data['Close'].values
        
        if len(data) < 60:
            return None
            
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        seq_length = 60
        X, y = create_sequences(scaled_data, seq_length)
        
        if len(X) == 0:
            return None
            
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Enhanced LSTM model based on Apple_forecast.ipynb
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X, y, 
            epochs=50, 
            batch_size=32, 
            verbose=0, 
            validation_split=0.2,
            shuffle=False
        )
        
        # Make future predictions
        last_sequence = scaled_data[-seq_length:]
        predictions = []
        
        for _ in range(future_days):
            pred = model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
            predictions.append(pred[0, 0])
            # Update sequence
            last_sequence = np.append(last_sequence[1:], pred)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()
        
    except Exception as e:
        st.error(f"LSTM Error: {e}")
        return None

def calculate_model_metrics(actual, predicted):
    """Calculate performance metrics with error handling"""
    if predicted is None or len(predicted) == 0:
        return None
        
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    if min_len == 0:
        return None
        
    return {
        'MAE': mean_absolute_error(actual, predicted),
        'R2': max(0, r2_score(actual, predicted)),  # Ensure non-negative R²
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAPE': np.mean(np.abs((actual - predicted) / actual)) * 100
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Stock Price Predictor Pro</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload Your Stock Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload your stock data CSV file with columns: Date, Open, High, Low, Close, Volume"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Use uploaded file or generate sample data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File successfully uploaded! Shape: {df.shape}")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return
    else:
        df = generate_sample_data()
        st.info("📊 Using sample data for demonstration. Upload your own CSV file for real analysis.")
    
    # Show data preview
    with st.expander("📊 Data Preview"):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df.head(10))
        with col2:
            st.write(f"**Dataset shape:** {df.shape}")
            st.write(f"**Date range:** {df['Date'].min()} to {df['Date'].max()}")
            st.write(f"**Columns:** {', '.join(df.columns)}")
    
    # Data preprocessing
    st.subheader("🛠️ Data Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_column = st.selectbox(
            "Select Date Column", 
            options=df.columns,
            index=0 if 'Date' in df.columns else 0
        )
    
    with col2:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        close_column = st.selectbox(
            "Select Close Price Column", 
            options=numeric_columns,
            index=numeric_columns.index('Close') if 'Close' in numeric_columns else 0
        )
    
    # Process data
    try:
        df_processed = df.copy()
        df_processed['Date'] = pd.to_datetime(df_processed[date_column], errors='coerce')
        df_processed = df_processed.dropna(subset=['Date'])
        df_processed = df_processed.sort_values('Date').reset_index(drop=True)
        df_processed.set_index('Date', inplace=True)
        
        # Ensure close column exists
        if close_column != 'Close':
            df_processed['Close'] = df_processed[close_column]
        
        # Create features
        with st.spinner("🔄 Creating technical indicators and features..."):
            df_enhanced = create_features(df_processed)
        
        st.success(f"✅ Data preprocessing completed! Enhanced dataset shape: {df_enhanced.shape}")
        
        # Show feature engineering results
        with st.expander("🔧 Feature Engineering Details"):
            st.write("**Created Technical Indicators:**")
            feature_cols = [col for col in df_enhanced.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            st.write(f"Total features created: {len(feature_cols)}")
            st.write(feature_cols)
            
    except Exception as e:
        st.error(f"❌ Error in data preprocessing: {e}")
        return
    
    # Display current market overview
    st.subheader("📈 Current Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df_enhanced['Close'].iloc[-1] if len(df_enhanced) > 0 else 0
    prev_price = df_enhanced['Close'].iloc[-2] if len(df_enhanced) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💰 Current Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        rsi = df_enhanced['RSI'].iloc[-1] if len(df_enhanced) > 0 else 50
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        st.metric("📊 RSI", f"{rsi:.1f}", rsi_status)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        volume = df_enhanced['Volume'].iloc[-1] if len(df_enhanced) > 0 else 0
        volume_ma = df_enhanced['Volume_MA_5'].iloc[-1] if len(df_enhanced) > 0 else volume
        volume_change = ((volume - volume_ma) / volume_ma) * 100
        st.metric("📈 Volume", f"{volume:,.0f}", f"{volume_change:+.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        volatility = df_enhanced['Close'].pct_change().std() * 100 * np.sqrt(252)  # Annualized
        st.metric("⚡ Volatility", f"{volatility:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/179/179309.png", width=80)
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.subheader("🤖 Model Settings")
    
    # Model selection
    st.sidebar.markdown("**Select Models to Train:**")
    use_arima = st.sidebar.checkbox("ARIMA", value=True)
    use_sarima = st.sidebar.checkbox("SARIMA", value=True)
    use_xgboost = st.sidebar.checkbox("XGBoost", value=True)
    use_lstm = st.sidebar.checkbox("LSTM", value=True)
    
    # Prediction settings
    st.sidebar.subheader("📅 Prediction Settings")
    prediction_days = st.sidebar.slider("Prediction Days", 30, 180, 90)
    train_ratio = st.sidebar.slider("Training Data Ratio", 0.6, 0.9, 0.8)
    
    # Advanced settings
    st.sidebar.subheader("🔧 Advanced Settings")
    show_training_details = st.sidebar.checkbox("Show Training Details", value=False)
    
    # Calculate split index
    split_index = int(len(df_enhanced) * train_ratio)
    train_data = df_enhanced[:split_index]
    test_data = df_enhanced[split_index:]
    
    st.sidebar.info(f"""
    **Data Split:**
    - Training: {len(train_data)} days ({train_ratio*100:.0f}%)
    - Testing: {len(test_data)} days ({(1-train_ratio)*100:.0f}%)
    - Total: {len(df_enhanced)} days
    """)
    
    # Run predictions button
    if st.sidebar.button("🚀 Run All Model Predictions", use_container_width=True, type="primary"):
        if not (use_arima or use_sarima or use_xgboost or use_lstm):
            st.error("❌ Please select at least one model to train!")
            return
            
        with st.spinner("🤖 Training all selected models and generating predictions..."):
            future_predictions = {}
            test_predictions = {}
            metrics = {}
            training_details = {}
            
            # Train selected models
            if use_arima:
                with st.spinner("Training ARIMA model..."):
                    future_pred = train_arima_model(train_data, prediction_days)
                    test_pred = train_arima_model(train_data, len(test_data))
                    if future_pred is not None:
                        future_predictions['ARIMA'] = future_pred
                        test_predictions['ARIMA'] = test_pred
                        metrics['ARIMA'] = calculate_model_metrics(test_data['Close'].values, test_pred)
                        training_details['ARIMA'] = "ARIMA(2,1,2) - Traditional time series model"
            
            if use_sarima:
                with st.spinner("Training SARIMA model..."):
                    future_pred = train_sarima_model(train_data, prediction_days)
                    test_pred = train_sarima_model(train_data, len(test_data))
                    if future_pred is not None:
                        future_predictions['SARIMA'] = future_pred
                        test_predictions['SARIMA'] = test_pred
                        metrics['SARIMA'] = calculate_model_metrics(test_data['Close'].values, test_pred)
                        training_details['SARIMA'] = "SARIMA(1,1,1)(1,1,1,30) - Seasonal ARIMA"
            
            if use_xgboost:
                with st.spinner("Training XGBoost model..."):
                    future_pred = train_xgboost_model(train_data, prediction_days)
                    test_pred = train_xgboost_model(train_data, len(test_data))
                    if future_pred is not None:
                        future_predictions['XGBoost'] = future_pred
                        test_predictions['XGBoost'] = test_pred
                        metrics['XGBoost'] = calculate_model_metrics(test_data['Close'].values, test_pred)
                        training_details['XGBoost'] = "XGBoost with technical indicators - Ensemble learning"
            
            if use_lstm:
                with st.spinner("Training LSTM model..."):
                    future_pred = train_lstm_model(train_data, prediction_days)
                    test_pred = train_lstm_model(train_data, len(test_data))
                    if future_pred is not None:
                        future_predictions['LSTM'] = future_pred
                        test_predictions['LSTM'] = test_pred
                        metrics['LSTM'] = calculate_model_metrics(test_data['Close'].values, test_pred)
                        training_details['LSTM'] = "LSTM with 3 layers - Deep learning for sequences"
            
            # Store results in session state
            st.session_state.future_predictions = future_predictions
            st.session_state.test_predictions = test_predictions
            st.session_state.metrics = metrics
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
            st.session_state.df_enhanced = df_enhanced
            st.session_state.prediction_days = prediction_days
            st.session_state.training_details = training_details
            
            st.success("✅ All models trained successfully!")
    
    # Display results if available
    if 'future_predictions' in st.session_state:
        future_predictions = st.session_state.future_predictions
        test_predictions = st.session_state.test_predictions
        metrics = st.session_state.metrics
        df_enhanced = st.session_state.df_enhanced
        prediction_days = st.session_state.prediction_days
        training_details = st.session_state.training_details
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔮 Predictions", 
            "📊 Performance", 
            "📈 Summary", 
            "📉 Analysis",
            "🤖 Models",
            "🔧 Training"
        ])
        
        with tab1:
            st.subheader(f"🔮 Future Price Predictions ({prediction_days} Days)")
            
            # Create future dates
            last_date = df_enhanced.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
            
            # Plot predictions
            fig = go.Figure()
            
            # Historical data (last 200 days or available data)
            historical_days = min(200, len(df_enhanced))
            fig.add_trace(go.Scatter(
                x=df_enhanced.index[-historical_days:],
                y=df_enhanced['Close'][-historical_days:],
                mode='lines',
                name='Historical Prices',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Future predictions
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            color_idx = 0
            for model, preds in future_predictions.items():
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=preds,
                    mode='lines',
                    name=f'{model} Prediction',
                    line=dict(color=colors[color_idx % len(colors)], width=2.5, dash='dash'),
                    opacity=0.8
                ))
                color_idx += 1
            
            fig.update_layout(
                title=f'Stock Price Predictions (Next {prediction_days} Days)',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white',
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show prediction values
            with st.expander("📋 Prediction Values"):
                pred_df = pd.DataFrame({'Date': future_dates})
                for model, preds in future_predictions.items():
                    pred_df[model] = preds
                st.dataframe(pred_df.style.format("{:.2f}"))
        
        with tab2:
            st.subheader("📊 Model Performance Comparison")
            
            if metrics:
                # Create metrics comparison
                models = [m for m in metrics.keys() if metrics[m] is not None]
                
                if models:
                    # Bar charts for metrics
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('📏 Mean Absolute Error (MAE)', '📈 R-squared (R²)', 
                                      '🎯 Root Mean Square Error (RMSE)', '📊 Mean Absolute Percentage Error (MAPE)'),
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1
                    )
                    
                    # MAE
                    fig.add_trace(
                        go.Bar(x=models, y=[metrics[m]['MAE'] for m in models], 
                              name='MAE', marker_color='#ff7f0e'),
                        row=1, col=1
                    )
                    
                    # R²
                    fig.add_trace(
                        go.Bar(x=models, y=[metrics[m]['R2'] for m in models], 
                              name='R²', marker_color='#2ca02c'),
                        row=1, col=2
                    )
                    
                    # RMSE
                    fig.add_trace(
                        go.Bar(x=models, y=[metrics[m]['RMSE'] for m in models], 
                              name='RMSE', marker_color='#d62728'),
                        row=2, col=1
                    )
                    
                    # MAPE
                    fig.add_trace(
                        go.Bar(x=models, y=[metrics[m]['MAPE'] for m in models], 
                              name='MAPE', marker_color='#9467bd'),
                        row=2, col=2
                    )
                    
                    fig.update_layout(
                        height=600,
                        showlegend=False,
                        template='plotly_white',
                        title_text="Model Performance Metrics Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Best performing model
                    best_model_rmse = min([(m, metrics[m]['RMSE']) for m in models], key=lambda x: x[1])
                    best_model_r2 = max([(m, metrics[m]['R2']) for m in models], key=lambda x: x[1])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"🏆 **Best Model (Lowest RMSE)**: {best_model_rmse[0]} (RMSE: {best_model_rmse[1]:.4f})")
                    with col2:
                        st.success(f"🎯 **Best Model (Highest R²)**: {best_model_r2[0]} (R²: {best_model_r2[1]:.4f})")
                    
                    # Metrics table
                    st.subheader("Detailed Metrics Table")
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(
                        metrics_df.style.format("{:.4f}")
                        .background_gradient(cmap='Blues', subset=['R2'])
                        .background_gradient(cmap='Reds_r', subset=['MAE', 'RMSE', 'MAPE'])
                    )
                else:
                    st.warning("No valid metrics available from the trained models.")
            else:
                st.warning("No metrics available. Please check model configurations.")
        
        with tab3:
            st.subheader("📈 Prediction Summary")
            
            if future_predictions and metrics:
                # Use best model for summary
                valid_models = [m for m in metrics.keys() if metrics[m] is not None]
                if valid_models:
                    best_model = min(valid_models, key=lambda x: metrics[x]['RMSE'])
                    pred_prices = future_predictions[best_model]
                    
                    current_price = df_enhanced['Close'].iloc[-1]
                    start_pred_price = pred_prices[0]
                    end_pred_price = pred_prices[-1]
                    total_change = end_pred_price - current_price
                    total_change_pct = (total_change / current_price) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.subheader("📊 Prediction Statistics")
                        st.write(f"**Best Model**: {best_model}")
                        st.write(f"**Current Price**: ${current_price:.2f}")
                        st.write(f"**Predicted Start Price**: ${start_pred_price:.2f}")
                        st.write(f"**Predicted End Price**: ${end_pred_price:.2f}")
                        
                        change_class = "prediction-positive" if total_change >= 0 else "prediction-negative"
                        st.markdown(f"**Total Predicted Change**: <span class='{change_class}'>${total_change:.2f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Total Predicted Change %**: <span class='{change_class}'>{total_change_pct:+.2f}%</span>", unsafe_allow_html=True)
                        
                        st.write(f"**Average Predicted Price**: ${np.mean(pred_prices):.2f}")
                        st.write(f"**Prediction Range**: ${np.min(pred_prices):.2f} - ${np.max(pred_prices):.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.subheader("⚡ Risk Analysis")
                        
                        volatility = (np.std(pred_prices) / np.mean(pred_prices)) * 100
                        st.write(f"**Prediction Volatility**: {volatility:.2f}%")
                        
                        # Risk assessment
                        if volatility < 2:
                            risk_level = "Low"
                            risk_class = "risk-low"
                        elif volatility < 5:
                            risk_level = "Medium"
                            risk_class = "risk-medium"
                        else:
                            risk_level = "High"
                            risk_class = "risk-high"
                        
                        st.markdown(f"**Risk Level**: <span class='{risk_class}'>{risk_level}</span>", unsafe_allow_html=True)
                        
                        # Confidence score based on R²
                        confidence = metrics[best_model]['R2'] * 100
                        st.write(f"**Model Confidence**: {confidence:.1f}%")
                        
                        # Trend analysis
                        price_trend = "Bullish" if total_change_pct > 2 else "Bearish" if total_change_pct < -2 else "Neutral"
                        trend_color = "prediction-positive" if price_trend == "Bullish" else "prediction-negative" if price_trend == "Bearish" else ""
                        st.markdown(f"**Price Trend**: <span class='{trend_color}'>{price_trend}</span>", unsafe_allow_html=True)
                        
                        # Recommendation
                        if total_change_pct > 10:
                            recommendation = "📈 Strong Buy"
                            rec_color = "prediction-positive"
                        elif total_change_pct > 2:
                            recommendation = "📊 Buy"
                            rec_color = "prediction-positive"
                        elif total_change_pct > -2:
                            recommendation = "⚖️ Hold"
                            rec_color = ""
                        elif total_change_pct > -10:
                            recommendation = "📉 Sell"
                            rec_color = "prediction-negative"
                        else:
                            recommendation = "🚨 Strong Sell"
                            rec_color = "prediction-negative"
                        
                        st.markdown(f"**Recommendation**: <span class='{rec_color}'>{recommendation}</span>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.subheader("📉 Technical Analysis")
            
            # Technical indicators plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price with Moving Averages', 'Relative Strength Index (RSI)', 
                              'Moving Average Convergence Divergence (MACD)', 'Volume Analysis'),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Price with MAs (show only last 100 points for clarity)
            display_data = df_enhanced.tail(100)
            
            fig.add_trace(
                go.Scatter(x=display_data.index, y=display_data['Close'], name='Close', line=dict(color='#1f77b4', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=display_data.index, y=display_data['MA_20'], name='MA 20', line=dict(color='orange', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=display_data.index, y=display_data['MA_50'], name='MA 50', line=dict(color='red', width=1)),
                row=1, col=1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(x=display_data.index, y=display_data['RSI'], name='RSI', line=dict(color='purple', width=2)),
                row=1, col=2
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=2)
            
            # MACD
            fig.add_trace(
                go.Scatter(x=display_data.index, y=display_data['MACD'], name='MACD', line=dict(color='blue', width=2)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=display_data.index, y=display_data['MACD_Signal'], name='Signal', line=dict(color='red', width=2)),
                row=2, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=display_data.index, y=display_data['Volume'], name='Volume', marker_color='lightblue', opacity=0.6),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=display_data.index, y=display_data['Volume_MA_5'], name='Volume MA 5', line=dict(color='darkblue', width=2)),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("🤖 Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **ARIMA (AutoRegressive Integrated Moving Average)**
                - **Type**: Statistical time series model
                - **Best for**: Linear patterns, stationary data
                - **Strengths**: Fast, interpretable, good for short-term forecasts
                - **Limitations**: Assumes linear relationships
                
                **SARIMA (Seasonal ARIMA)**
                - **Type**: Statistical model with seasonality
                - **Best for**: Data with seasonal patterns
                - **Strengths**: Handles seasonality, good for periodic data
                - **Limitations**: More parameters to tune
                """)
            
            with col2:
                st.markdown("""
                **XGBoost (Extreme Gradient Boosting)**
                - **Type**: Ensemble machine learning
                - **Best for**: Complex non-linear relationships
                - **Strengths**: Handles features well, robust to outliers
                - **Limitations**: Requires feature engineering
                
                **LSTM (Long Short-Term Memory)**
                - **Type**: Deep learning recurrent network
                - **Best for**: Sequential data with long-term dependencies
                - **Strengths**: Captures complex patterns, no manual feature engineering needed
                - **Limitations**: Computationally intensive, requires more data
                """)
        
        with tab6:
            st.subheader("🔧 Training Details")
            
            if training_details:
                for model, details in training_details.items():
                    st.markdown(f"**{model}**")
                    st.info(details)
                    
                    if model in metrics and metrics[model] is not None:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"{metrics[model]['MAE']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{metrics[model]['RMSE']:.4f}")
                        with col3:
                            st.metric("R²", f"{metrics[model]['R2']:.4f}")
                        with col4:
                            st.metric("MAPE", f"{metrics[model]['MAPE']:.2f}%")
                    st.markdown("---")
            else:
                st.info("No training details available. Run predictions to see model specifications.")
    
    else:
        # Instructions when no predictions have been run
        st.info("""
        🎯 **Ready to generate predictions!**
        
        To get started:
        1. Select the models you want to use in the sidebar (all are selected by default)
        2. Adjust prediction settings (90 days recommended)
        3. Click **'Run All Model Predictions'** to train all selected models and generate forecasts
        
        📊 **Available Models:**
        - **ARIMA**: Traditional time series forecasting
        - **SARIMA**: Seasonal time series analysis  
        - **XGBoost**: Powerful gradient boosting with technical indicators
        - **LSTM**: Advanced deep learning for sequences
        
        All models are trained automatically and evaluated based on their actual performance.
        The system will show you which model performs best for your specific data.
        """)

if __name__ == "__main__":
    main()