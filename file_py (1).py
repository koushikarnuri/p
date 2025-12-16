import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -------------------------------
# Title
# -------------------------------
st.title("Apple Stock Price Prediction using ARIMA")

# -------------------------------
# Load dataset (training data)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df["Close"]

train_data = load_data()

# -------------------------------
# Load ARIMA model
# -------------------------------
try:
    model = joblib.load("arima_model.joblib")
    st.success("ARIMA model loaded successfully!")
except FileNotFoundError:
    st.error("❌ 'arima_model.joblib' not found. Upload it to the repo.")
    st.stop()

# -------------------------------
# Display training info
# -------------------------------
last_train_date = train_data.index[-1]
last_train_value = train_data.iloc[-1]

st.write(f"📅 Model trained up to: **{last_train_date.date()}**")
st.write(f"💰 Last observed close price: **${last_train_value:.2f}**")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Prediction Settings")
num_days = st.sidebar.slider("Forecast days", 1, 90, 30)

# -------------------------------
# Forecast
# -------------------------------
if st.sidebar.button("Generate Forecast"):

    st.subheader(f"📈 Forecast for next {num_days} days")

    try:
        forecast = model.forecast(steps=num_days)

        forecast_dates = pd.date_range(
            start=last_train_date + pd.Timedelta(days=1),
            periods=num_days,
            freq="B"
        )

        forecast_df = pd.DataFrame({
            "Predicted Close": forecast
        }, index=forecast_dates)

        st.dataframe(forecast_df)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train_data.index, train_data, label="Historical Data")
        ax.plot(forecast_df.index, forecast_df["Predicted Close"],
                label="Forecast", linestyle="--")

        ax.set_title("Apple Stock Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Forecasting error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.write("⚠️ **Disclaimer:** This is for educational purposes only. Not financial advice.")
