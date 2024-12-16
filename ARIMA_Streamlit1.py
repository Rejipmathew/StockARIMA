import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Streamlit application title and description
st.title("📈 Stock Price Forecasting with ARIMA")
st.markdown("""
This application uses the **ARIMA** model to forecast stock prices.
Enter the stock ticker symbol, select the date range, and adjust the ARIMA parameters to see the forecast results.
""")

# Step 1: User inputs for stock ticker and date range
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, TSLA)", value='TSLA')
start_date = st.date_input("Start Date", value=pd.to_datetime('2020-11-01'))
 
end_date = st.date_input("End Date", value=pd.to_datetime('2024-12-16'))
forecast_days = st.slider("Select Forecast Days", min_value=1, max_value=180, value=30)

# Step 2: Fetch stock data using yfinance
@st.cache_data
def load_data(ticker, start_date, end_date):
    """Fetches historical stock data using yfinance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']].dropna()  # Ensure there are no missing values
    return data

# Load data
data = load_data(ticker, start_date, end_date)

# Check if data is available
if not data.empty:
    st.subheader(f"Historical Closing Prices for {ticker}")
    st.line_chart(data['Close'])

    # Step 3: Data preprocessing (differencing for stationarity)
    data['Differenced'] = data['Close'].diff().dropna()

    # Step 4: ACF and PACF plots for parameter selection
    st.subheader("ACF and PACF Plots")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(data['Differenced'].dropna(), ax=axes[0], lags=30)
    plot_pacf(data['Differenced'].dropna(), ax=axes[1], lags=30)
    st.pyplot(fig)

    # Step 5: Build and fit the ARIMA model
    st.subheader("ARIMA Model Training")
    
    # Separate sliders for ARIMA (p, d, q) parameters
    p = st.slider("Select AR (p) parameter", 0, 5, value=5)
    d = st.slider("Select I (d) parameter", 0, 2, value=1)
    q = st.slider("Select MA (q) parameter", 0, 5, value=2)

    # Build the ARIMA model
    try:
        model = ARIMA(data['Close'], order=(p, d, q))
        fitted_model = model.fit()

        # Display model summary
        st.text("Model Summary")
        st.text(fitted_model.summary())

        # Output the p-values
        st.subheader("P-Values of ARIMA Model Parameters")
        p_values = fitted_model.pvalues
        st.write(p_values)

        # Step 6: Forecast future stock prices
        st.subheader(f"Forecasting Next {forecast_days} Days")
        forecast = fitted_model.get_forecast(steps=forecast_days)

        # Generate the date range for the forecast
        last_date = data.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

        # Extract forecast results and confidence intervals
        forecast_mean = forecast.predicted_mean
        forecast_conf_int = forecast.conf_int()

        # Plotting the forecast results with confidence intervals
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Close'], label='Historical Data')
        ax.plot(forecast_index, forecast_mean, label='Forecast', color='orange')
        ax.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='gray', alpha=0.3)
        ax.set_title(f'{ticker} Stock Price Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Output the confidence intervals
        st.subheader("Confidence Intervals for Forecasted Values")
        st.write(forecast_conf_int)

        # Step 7: Evaluate model performance using RMSE
        st.subheader("Model Evaluation")
        predictions = fitted_model.fittedvalues
        # Adjust prediction length to match the actual data length
        predictions = predictions[1:]
        rmse = np.sqrt(mean_squared_error(data['Close'][1:], predictions))
        st.write(f"Root Mean Square Error (RMSE): {rmse:.2f}")

    except Exception as e:
        st.error(f"Error in fitting ARIMA model: {e}")

else:
    st.error("No data found for the specified ticker. Please try another symbol.")
