-----

# 📈 Apple Stock Price Predictor

This is a comprehensive web application built with Streamlit that uses various machine learning and time series models to analyze and predict Apple (AAPL) stock prices. The application allows users to upload their own stock data, train multiple models, compare their performance, and generate a 30-day price forecast.

## ✨ Features

  * **Interactive Dashboard**: A user-friendly web interface powered by Streamlit.
  * **Data Upload**: Upload your own stock data in CSV format.
  * **Data Overview**: View historical price charts, key statistics (like all-time high/low), and recent data.
  * **Automated Feature Engineering**: Automatically creates technical indicators like Moving Averages (MA), RSI, MACD, and lag features.
  * **Multi-Model Training**: Train and evaluate four different powerful models:
      * `ARIMA`: Autoregressive Integrated Moving Average
      * `SARIMA`: Seasonal ARIMA
      * `XGBoost`: A gradient-boosting powerhouse for regression.
      * `LSTM`: A deep learning (Recurrent Neural Network) model for time series forecasting.
  * **Performance Comparison**: Visually compare models based on metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
  * **30-Day Forecasting**: Select the best model to predict stock prices for the next 30 days and visualize the forecast against historical data.
  * **Feature Importance**: For the XGBoost model, view a chart of the most influential features in predicting the stock price.

## 🛠️ Technologies & Libraries Used

The project is built using Python and relies on the following major libraries:

  * **Web Framework**: `streamlit` 
  * **Data Manipulation**: `pandas`, `numpy` 
  * **Machine Learning**: `scikit-learn`, `xgboost` 
  * **Deep Learning**: `tensorflow` 
  * **Time Series Analysis**: `statsmodels` 
  * **Data Visualization**: `matplotlib`, `seaborn`, `plotly` 

All dependencies are listed in the `requirements.txt` file.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

  * Python 3.8 or higher
  * `pip` (Python package installer)

### Installation & Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/apple-stock-predictor.git
    cd apple-stock-predictor
    ```

2.  **Create and Activate a Virtual Environment**
    It is highly recommended to use a virtual environment to manage project-specific dependencies.

      * **Create the environment:**
        ```bash
        python -m venv venv
        ```
      * **Activate the environment:**
          * On **Windows**:
            ```bash
            venv\Scripts\activate
            ```
          * On **macOS & Linux**:
            ```bash
            source venv/bin/activate
            ```

    Your terminal prompt should now be prefixed with `(venv)`.

3.  **Install Required Packages**
    Install all the necessary libraries from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Application**
    Once the installation is complete, you can launch the app with the following command:

    ```bash
    streamlit run apple.py
    ```

    Your web browser should automatically open with the application running.

## 🖥️ How to Use the App

1.  **Launch the app** using the `streamlit run apple.py` command.
2.  **Upload Data**: Use the file uploader in the sidebar to upload a CSV file containing stock data.
3.  **Data Format**: The CSV file must contain the following columns:
      * `Date` (in `DD-MM-YYYY` format)
      * `Open`
      * `High`
      * `Low`
      * `Close`
      * `Volume`
4.  **Explore Data**: Navigate to the **📊 Data Overview** tab to see visualizations and statistics.
5.  **Train Models**: Go to the **🤖 Models** tab. Select the models you want to run (ARIMA, SARIMA, XGBoost, LSTM) and click the **"Train Models"** button.
6.  **Generate Predictions**: After training, go to the **📈 Predictions** tab. Choose a model from the dropdown list and click **"Generate 30-Day Prediction"**.
7.  **Analyze Results**: Switch to the **📋 Results** tab to see a detailed comparison of model performance metrics and feature importance charts.
