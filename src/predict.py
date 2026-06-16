import pandas as pd

def predict_future_price(model, latest_row):

    input_df = pd.DataFrame({
        "Open": [latest_row["Open"]],
        "High": [latest_row["High"]],
        "Low": [latest_row["Low"]],
        "Volume": [latest_row["Volume"]],
        "Day": [latest_row["Day"]],
        "Month": [latest_row["Month"]],
        "Year": [latest_row["Year"]]
    })

    prediction = model.predict(input_df)

    return prediction[0]