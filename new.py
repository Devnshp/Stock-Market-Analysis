import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import plotly.express as px
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
</style>
""", unsafe_allow_html=True)

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
    
    if uploaded_file is not None:
        # Load and process data
        @st.cache_data
        def load_data(uploaded_file):
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                
                # Display file info
                st.success(f"✅ File successfully uploaded! Shape: {df.shape}")
                
                # Show preview
                with st.expander("📊 Preview Uploaded Data"):
                    st.dataframe(df.head(10))
                    st.write(f"**Columns:** {list(df.columns)}")
                    st.write(f"**Data Types:**")
                    st.write(df.dtypes)
                
                return df
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
                return None

        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Data preprocessing section
            st.subheader("🛠️ Data Preprocessing")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date_column = st.selectbox(
                    "Select Date Column", 
                    options=df.columns,
                    index=0 if 'Date' in df.columns else 0
                )
            
            with col2:
                # Auto-detect numeric columns for price data
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                close_column = st.selectbox(
                    "Select Close Price Column", 
                    options=numeric_columns,
                    index=0
                )
            
            with col3:
                date_format = st.selectbox(
                    "Date Format",
                    ["%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "Auto-detect"]
                )
            
            # Process data
            @st.cache_data
            def preprocess_data(df, date_column, close_column, date_format):
                try:
                    df_processed = df.copy()
                    
                    # Convert date column
                    if date_format == "Auto-detect":
                        df_processed['Date'] = pd.to_datetime(df_processed[date_column], infer_datetime_format=True)
                    else:
                        df_processed['Date'] = pd.to_datetime(df_processed[date_column], format=date_format)
                    
                    # Set date as index and sort
                    df_processed = df_processed.sort_values('Date').reset_index(drop=True)
                    df_processed.set_index('Date', inplace=True)
                    
                    # Ensure we have the required columns
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    available_columns = df_processed.columns.tolist()
                    
                    # If close column is specified, rename it
                    if close_column != 'Close':
                        df_processed['Close'] = df_processed[close_column]
                    
                    st.success("✅ Data preprocessing completed successfully!")
                    return df_processed
                    
                except Exception as e:
                    st.error(f"❌ Error in data preprocessing: {e}")
                    return None

            # Process the data
            df_processed = preprocess_data(df, date_column, close_column, date_format)
            
            if df_processed is not None:
                # Feature engineering
                def create_features(data):
                    df_features = data.copy()

                    # Basic price features
                    df_features['Price_Range'] = df_features['High'] - df_features['Low']
                    df_features['Price_Change'] = df_features['Close'] - df_features['Open']
                    df_features['Price_Change_Pct'] = (df_features['Close'] - df_features['Open']) / df_features['Open'] * 100

                    # Moving averages
                    df_features['MA_5'] = df_features['Close'].rolling(window=5).mean()
                    df_features['MA_10'] = df_features['Close'].rolling(window=10).mean()
                    df_features['MA_20'] = df_features['Close'].rolling(window=20).mean()
                    df_features['MA_50'] = df_features['Close'].rolling(window=50).mean()

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

                    return df_features

                # Create enhanced features
                df_enhanced = create_features(df_processed)
                df_enhanced = df_enhanced.dropna()

                # Sidebar configuration
                st.sidebar.image("https://cdn-icons-png.flaticon.com/512/179/179309.png", width=80)
                st.sidebar.title("⚙️ Configuration")
                
                st.sidebar.subheader("🔧 Model Settings")
                
                # Model selection
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    use_arima = st.checkbox("ARIMA", value=True)
                    use_sarima = st.checkbox("SARIMA", value=True)
                with col2:
                    use_xgboost = st.checkbox("XGBoost", value=True)
                    use_lstm = st.checkbox("LSTM", value=True)
                
                # Prediction settings
                st.sidebar.subheader("📅 Prediction Settings")
                prediction_days = st.sidebar.slider("Prediction Days", 30, 90, 60)
                train_ratio = st.sidebar.slider("Training Data Ratio", 0.7, 0.9, 0.8)

                # Calculate split index
                split_index = int(len(df_enhanced) * train_ratio)
                train_data = df_enhanced[:split_index]
                test_data = df_enhanced[split_index:]

                # Model training functions
                def train_arima(train_data, future_days=60):
                    try:
                        model = ARIMA(train_data['Close'], order=(2,1,2))
                        model_fit = model.fit()
                        predictions = model_fit.forecast(steps=future_days)
                        return predictions
                    except Exception as e:
                        st.error(f"ARIMA training error: {e}")
                        return None

                def train_sarima(train_data, future_days=60):
                    try:
                        model = SARIMAX(train_data['Close'], order=(1,1,1), seasonal_order=(1,1,1,12))
                        model_fit = model.fit(disp=False)
                        predictions = model_fit.forecast(steps=future_days)
                        return predictions
                    except Exception as e:
                        st.error(f"SARIMA training error: {e}")
                        return None

                def train_xgboost(train_data, future_days=60):
                    try:
                        features = ['Open', 'High', 'Low', 'Volume', 'Price_Range', 'MA_5', 'MA_20', 'RSI', 'MACD']
                        features = [f for f in features if f in train_data.columns]
                        
                        # Prepare training data
                        X_train = train_data[features].dropna()
                        y_train = train_data.loc[X_train.index, 'Close']
                        
                        model = xgb.XGBRegressor(
                            n_estimators=100, 
                            learning_rate=0.1, 
                            max_depth=6,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        
                        # For future prediction
                        last_features = train_data[features].iloc[-1:].values
                        predictions = []
                        
                        for _ in range(future_days):
                            pred = model.predict(last_features)[0]
                            predictions.append(pred)
                            # Update features for next prediction
                            last_features[0][0] = pred  # Update Open
                            last_features[0][1] = pred * 1.01  # Update High
                            last_features[0][2] = pred * 0.99  # Update Low
                        
                        return np.array(predictions)
                    except Exception as e:
                        st.error(f"XGBoost training error: {e}")
                        return None

                def train_lstm(train_data, future_days=60):
                    try:
                        # Prepare data
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))
                        
                        # Create sequences
                        def create_sequences(data, seq_length):
                            X, y = [], []
                            for i in range(seq_length, len(data)):
                                X.append(data[i-seq_length:i, 0])
                                y.append(data[i, 0])
                            return np.array(X), np.array(y)
                        
                        seq_length = 60
                        X, y = create_sequences(scaled_data, seq_length)
                        X = X.reshape((X.shape[0], X.shape[1], 1))
                        
                        # Build model
                        model = Sequential([
                            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                            Dropout(0.2),
                            LSTM(50, return_sequences=False),
                            Dropout(0.2),
                            Dense(25),
                            Dense(1)
                        ])
                        
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        model.fit(X, y, batch_size=32, epochs=20, verbose=0, validation_split=0.1)
                        
                        # Make future predictions
                        last_sequence = scaled_data[-seq_length:]
                        predictions = []
                        
                        for _ in range(future_days):
                            pred = model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
                            predictions.append(pred[0, 0])
                            last_sequence = np.append(last_sequence[1:], pred)
                        
                        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                        return predictions.flatten()
                    except Exception as e:
                        st.error(f"LSTM training error: {e}")
                        return None

                # Run predictions button
                if st.sidebar.button("🚀 Run Predictions", use_container_width=True):
                    with st.spinner("🔄 Training models and generating predictions..."):
                        future_predictions = {}
                        metrics = {}
                        
                        # Train models and get future predictions
                        if use_arima:
                            with st.spinner("Training ARIMA model..."):
                                future_pred = train_arima(train_data, prediction_days)
                                if future_pred is not None:
                                    future_predictions['ARIMA'] = future_pred
                                    
                                    # Calculate metrics on test data
                                    test_pred = train_arima(train_data, len(test_data))
                                    if test_pred is not None:
                                        metrics['ARIMA'] = {
                                            'MAE': mean_absolute_error(test_data['Close'][:len(test_pred)], test_pred),
                                            'R2': r2_score(test_data['Close'][:len(test_pred)], test_pred),
                                            'RMSE': np.sqrt(mean_squared_error(test_data['Close'][:len(test_pred)], test_pred))
                                        }

                        if use_sarima:
                            with st.spinner("Training SARIMA model..."):
                                future_pred = train_sarima(train_data, prediction_days)
                                if future_pred is not None:
                                    future_predictions['SARIMA'] = future_pred
                                    
                                    test_pred = train_sarima(train_data, len(test_data))
                                    if test_pred is not None:
                                        metrics['SARIMA'] = {
                                            'MAE': mean_absolute_error(test_data['Close'][:len(test_pred)], test_pred),
                                            'R2': r2_score(test_data['Close'][:len(test_pred)], test_pred),
                                            'RMSE': np.sqrt(mean_squared_error(test_data['Close'][:len(test_pred)], test_pred))
                                        }

                        if use_xgboost:
                            with st.spinner("Training XGBoost model..."):
                                future_pred = train_xgboost(train_data, prediction_days)
                                if future_pred is not None:
                                    future_predictions['XGBoost'] = future_pred
                                    
                                    test_pred = train_xgboost(train_data, len(test_data))
                                    if test_pred is not None:
                                        metrics['XGBoost'] = {
                                            'MAE': mean_absolute_error(test_data['Close'][:len(test_pred)], test_pred),
                                            'R2': r2_score(test_data['Close'][:len(test_pred)], test_pred),
                                            'RMSE': np.sqrt(mean_squared_error(test_data['Close'][:len(test_pred)], test_pred))
                                        }

                        if use_lstm:
                            with st.spinner("Training LSTM model..."):
                                future_pred = train_lstm(train_data, prediction_days)
                                if future_pred is not None:
                                    future_predictions['LSTM'] = future_pred
                                    
                                    test_pred = train_lstm(train_data, len(test_data))
                                    if test_pred is not None:
                                        metrics['LSTM'] = {
                                            'MAE': mean_absolute_error(test_data['Close'][:len(test_pred)], test_pred),
                                            'R2': r2_score(test_data['Close'][:len(test_pred)], test_pred),
                                            'RMSE': np.sqrt(mean_squared_error(test_data['Close'][:len(test_pred)], test_pred))
                                        }

                        # Store results
                        st.session_state.future_predictions = future_predictions
                        st.session_state.metrics = metrics
                        st.session_state.train_data = train_data
                        st.session_state.test_data = test_data
                        st.session_state.df_enhanced = df_enhanced

                # Display results if available
                if 'future_predictions' in st.session_state:
                    future_predictions = st.session_state.future_predictions
                    metrics = st.session_state.metrics
                    train_data = st.session_state.train_data
                    test_data = st.session_state.test_data
                    df_enhanced = st.session_state.df_enhanced
                    
                    # Create tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Overview", 
                        "📈 Predictions", 
                        "⚖️ Model Metrics", 
                        "📋 Stock Summary", 
                        "🔍 Technical Analysis"
                    ])

                    with tab1:
                        st.subheader("📈 Stock Overview & Performance")
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        current_price = df_enhanced['Close'].iloc[-1]
                        
                        with col1:
                            st.metric(
                                "Current Price", 
                                f"${current_price:.2f}",
                                delta=f"${df_enhanced['Close'].iloc[-1] - df_enhanced['Close'].iloc[-2]:.2f}"
                            )
                        
                        with col2:
                            st.metric(
                                "30 Day Volatility", 
                                f"{df_enhanced['Close'].pct_change().std() * 100:.2f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "RSI", 
                                f"{df_enhanced['RSI'].iloc[-1]:.1f}"
                            )
                        
                        with col4:
                            best_model = min(metrics.items(), key=lambda x: x[1]['RMSE'])[0] if metrics else "N/A"
                            st.metric("Best Model", best_model)

                        # Price chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_enhanced.index, 
                            y=df_enhanced['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        fig.update_layout(
                            title='Stock Price History',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        st.subheader("🔮 Future Price Predictions")
                        
                        # Create future dates
                        last_date = df_enhanced.index[-1]
                        future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
                        
                        # Plot predictions
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=df_enhanced.index[-100:],
                            y=df_enhanced['Close'][-100:],
                            mode='lines',
                            name='Historical',
                            line=dict(color='#1f77b4', width=3)
                        ))
                        
                        # Future predictions
                        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                        for i, (model, preds) in enumerate(future_predictions.items()):
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=preds,
                                mode='lines',
                                name=f'{model} Prediction',
                                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                            ))
                        
                        fig.update_layout(
                            title=f'Stock Price Predictions (Next {prediction_days} Days)',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            template='plotly_white',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with tab3:
                        st.subheader("⚖️ Model Performance Comparison")
                        
                        if metrics:
                            # Create metrics comparison
                            models = list(metrics.keys())
                            
                            # Bar charts for metrics
                            fig = make_subplots(
                                rows=1, cols=3,
                                subplot_titles=('Mean Absolute Error (MAE)', 'R-squared (R²)', 'Root Mean Square Error (RMSE)')
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
                                row=1, col=3
                            )
                            
                            fig.update_layout(
                                height=400,
                                showlegend=False,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Best performing model
                            best_model = min(metrics.items(), key=lambda x: x[1]['RMSE'])[0]
                            st.success(f"🎯 **Best Performing Model**: {best_model} (Lowest RMSE)")
                            
                            # Metrics table
                            st.subheader("Detailed Metrics")
                            metrics_df = pd.DataFrame(metrics).T
                            st.dataframe(metrics_df.style.format("{:.4f}").background_gradient(cmap='Blues'))
                            
                        else:
                            st.warning("No metrics available. Please check model configurations.")

                    with tab4:
                        st.subheader("📋 Stock Prediction Summary")
                        
                        if future_predictions:
                            # Use the best model for summary
                            best_model = min(metrics.items(), key=lambda x: x[1]['RMSE'])[0] if metrics else list(future_predictions.keys())[0]
                            pred_prices = future_predictions[best_model]
                            
                            current_price = df_enhanced['Close'].iloc[-1]
                            start_pred_price = pred_prices[0]
                            end_pred_price = pred_prices[-1]
                            price_change = end_pred_price - start_pred_price
                            price_change_pct = (price_change / start_pred_price) * 100
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.subheader("📊 Prediction Statistics")
                                st.write(f"**Starting Price**: ${start_pred_price:.2f}")
                                st.write(f"**Predicted End Price**: ${end_pred_price:.2f}")
                                
                                change_class = "prediction-positive" if price_change >= 0 else "prediction-negative"
                                st.markdown(f"**Predicted Change**: <span class='{change_class}'>${price_change:.2f}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Predicted Change %**: <span class='{change_class}'>{price_change_pct:.2f}%</span>", unsafe_allow_html=True)
                                
                                st.write(f"**Average Predicted Price**: ${np.mean(pred_prices):.2f}")
                                st.write(f"**Prediction Model**: {best_model}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.subheader("⚡ Risk Analysis")
                                st.write(f"**Current Price**: ${current_price:.2f}")
                                st.write(f"**Prediction Range**: ${np.min(pred_prices):.2f} - ${np.max(pred_prices):.2f}")
                                
                                volatility = (np.std(pred_prices) / np.mean(pred_prices)) * 100
                                st.write(f"**Prediction Volatility**: {volatility:.2f}%")
                                st.write(f"**Min Predicted Price**: ${np.min(pred_prices):.2f}")
                                st.write(f"**Max Predicted Price**: ${np.max(pred_prices):.2f}")
                                
                                # Risk indicator
                                if volatility < 2:
                                    risk_level = "Low"
                                    risk_color = "green"
                                elif volatility < 5:
                                    risk_level = "Medium"
                                    risk_color = "orange"
                                else:
                                    risk_level = "High"
                                    risk_color = "red"
                                
                                st.markdown(f"**Risk Level**: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                    with tab5:
                        st.subheader("🔍 Technical Analysis")
                        
                        # Technical indicators
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Price with Moving Averages', 'Relative Strength Index (RSI)', 
                                          'MACD', 'Trading Volume'),
                            vertical_spacing=0.1,
                            horizontal_spacing=0.1
                        )
                        
                        # Price with MAs
                        fig.add_trace(
                            go.Scatter(x=df_enhanced.index, y=df_enhanced['Close'], name='Close', line=dict(color='#1f77b4')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df_enhanced.index, y=df_enhanced['MA_20'], name='MA 20', line=dict(color='orange')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df_enhanced.index, y=df_enhanced['MA_50'], name='MA 50', line=dict(color='red')),
                            row=1, col=1
                        )
                        
                        # RSI
                        fig.add_trace(
                            go.Scatter(x=df_enhanced.index, y=df_enhanced['RSI'], name='RSI', line=dict(color='purple')),
                            row=1, col=2
                        )
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
                        
                        # MACD
                        fig.add_trace(
                            go.Scatter(x=df_enhanced.index, y=df_enhanced['MACD'], name='MACD', line=dict(color='blue')),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df_enhanced.index, y=df_enhanced['MACD_Signal'], name='Signal', line=dict(color='red')),
                            row=2, col=1
                        )
                        
                        # Volume
                        fig.add_trace(
                            go.Bar(x=df_enhanced.index, y=df_enhanced['Volume'], name='Volume', marker_color='lightblue'),
                            row=2, col=2
                        )
                        
                        fig.update_layout(height=600, showlegend=True, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    # Instructions
                    st.info("""
                    🎯 **Ready to generate predictions!**
                    
                    To get started:
                    1. Select the models you want to use in the sidebar
                    2. Adjust prediction settings
                    3. Click **'Run Predictions'** to generate forecasts
                    
                    This app uses multiple machine learning models to predict stock prices 
                    for the next 60 days and provides comprehensive analysis.
                    """)
            
            else:
                st.error("❌ Data preprocessing failed. Please check your data format.")
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        <div style='text-align: center; padding: 4rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <h2>🎯 Welcome to Stock Price Predictor Pro!</h2>
            <p style='font-size: 1.2em;'>Upload your stock data CSV file to get started with advanced price predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='model-card'>
                <h4>📊 Multiple Models</h4>
                <p>ARIMA, SARIMA, XGBoost, LSTM</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='model-card'>
                <h4>🔮 Future Predictions</h4>
                <p>60-day price forecasts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='model-card'>
                <h4>📈 Advanced Analytics</h4>
                <p>Technical indicators & risk analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        # File format instructions
        with st.expander("📋 Expected File Format"):
            st.write("""
            Your CSV file should contain the following columns:
            - **Date**: Date of the trading day
            - **Open**: Opening price
            - **High**: Highest price of the day
            - **Low**: Lowest price of the day
            - **Close**: Closing price
            - **Volume**: Trading volume
            
            Example:
            """)
            sample_data = pd.DataFrame({
                'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'Open': [150.0, 152.5, 151.8],
                'High': [155.0, 154.2, 153.5],
                'Low': [149.5, 151.0, 150.2],
                'Close': [153.2, 152.8, 152.0],
                'Volume': [1000000, 1200000, 950000]
            })
            st.dataframe(sample_data)

if __name__ == "__main__":
    main()