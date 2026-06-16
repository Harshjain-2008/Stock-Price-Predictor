import pandas as pd

FEATURES = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Day",
    "Month",
    "Year"
]

def predict_future_price(model, latest_row):

    input_df = pd.DataFrame(
        [[
            float(latest_row["Open"]),
            float(latest_row["High"]),
            float(latest_row["Low"]),
            float(latest_row["Volume"]),
            int(latest_row["Day"]),
            int(latest_row["Month"]),
            int(latest_row["Year"])
        ]],
        columns=FEATURES
    )

    prediction = model.predict(input_df)

    return prediction[0]