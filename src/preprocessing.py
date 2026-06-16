def create_features(df, prediction_days):

    df["Day"] = df['Date'].dt.day
    df["Month"] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    df["Target"] = df["Close"].shift(-prediction_days)

    df.dropna(inplace=True)

    return df 