import streamlit as st
import matplotlib.pyplot as plt

from datetime import date

from src.data_loader import load_stock_data
from src.predict import predict_future_price
from src.preprocessing import create_features
from src.train import train_model
from src.utils import POPULAR_STOCKS

# PAGE CONFIGURATION 

st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    layout="wide"
)

st.title("Stock Price Prediction Dasboard")
st.markdown("Predict future STOCKS prices using Machine Learning and Yahoo finance Data")

# SIDEBAAR 

st.sidebar.header("Dashboard Settings")

selected_company = st.sidebar.selectbox(
    "Choose Company", list(POPULAR_STOCKS.keys())
    )

default_ticker = POPULAR_STOCKS[selected_company]

ticker = st.sidebar.text_input(
    "OR Enter Custom Yahoo Finance Stocks", value=default_ticker
)

year = st.sidebar.slider(
    "Years of Historical Data",
    min_value=1,
    max_value=10,
    value=5
)

prediction_days = st.sidebar.selectbox(
    "Predict Hoizon (Days)",
    [1,7,30]
)


# STOCK GUIDE 

st.sidebar.markdown("----")
st.sidebar.markdown("Popular Tickers")

st.sidebar.info("""
🇺🇸 US Stocks

AAPL → Apple

MSFT → Microsoft

NVDA → NVIDIA

TSLA → Tesla

AMZN → Amazon

GOOGL → Google

META → Meta


🇮🇳 Indian Stocks

RELIANCE.NS → Reliance

TCS.NS → TCS

INFY.NS → Infosys

SBIN.NS → SBI

HDFCBANK.NS → HDFC Bank

ITC.NS → ITC
                
""")

run_button = st.sidebar.button("Run Prediction")


# MAIN APP

if run_button:

    try:

        today = date.today()
        start_date = date(
            today.year - year,
            today.month,
            today.day
        )

        # load data 
        df = load_stock_data(
            ticker, start_date
        )

        if df.empty:
            st.error("Invalid Ticker...")
            st.stop()

        # Company information card 
        st.success(
            f"""
            Company : {selected_company},
            Ticker : {ticker},
            Historical Data: {year} years,
            Prediction Horizon : {prediction_days} days"""
        )    

        # stock price chart 

        st.subheader("Stock Closing Price ")

        fig , ax = plt.subplot()
        ax.plot(df["Date"], df["Close"])
        ax.set_tittle(f"{ticker} Stock Price")

        st.pyplot(fig)

        # DATA PREVIEW 

        st.subheader("Latest Market Data")

        st.dataframe(df.tail(10), use_container_width=True)

        # FEATURE ENGINEERING 

        processed_df = create_features(df, prediction_days)

        # TRAIN MODEL
        model, mae, r2s = train_model(processed_df)

        st.subheader("Model Performannce")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("MAE", round(mae,2))

        with col2:
            st.metric("R2S", round(r2s,3))


        # PREDICTION 
        latest_row = processed_df.iloc[-1]

        predicted_price = predict_future_price(model, latest_row)
        current_price = float(processed_df.iloc[-1]["Close"])

        difference = predicted_price - current_price 

        st.subheader("Future Pridiction")

        p1,p2,p3 = st.columns(3) 

        with p1:
            st.metric("Current price", f"${current_price:.2f}")

        with p2:
            st.metric(f"Predicted Price ({prediction_days} days)",f"${predicted_price:.2f}")

        with p3:
            st.metric("Expected Change", f"${difference:.2f}")

        # BUY / SELL SIGNAL 
        st.subheader("AI Signal")

        if predicted_price > current_price:
            st.success(f"Bullish Signal: Model Predicts the rise of ${difference:.2f}")

        else:
            st.error(f"Bearish Signal: Model predicts the drop of ${abs(difference):.2f}")            

    except Exception as e:
        st.error(f"Error: {str(e)}")
                      