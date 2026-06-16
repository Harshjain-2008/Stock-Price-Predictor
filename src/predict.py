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
        [latest_row[FEATURES].values],
        columns=FEATURES
    )

    prediction = model.predict(input_df)

    return float(prediction[0])