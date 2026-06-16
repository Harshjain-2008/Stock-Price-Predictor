def predict_future_price(model, latest_row):

    prediction = model.predict([
        latest_row["Open"],
        latest_row["High"],
        latest_row["Low"],
        latest_row["Volume"],
        latest_row["Day"],
        latest_row["Month"],
        latest_row["Year"]

    ])

    return prediction[0]

