from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_model(df):

    features = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Day",
        "Month",
        "Year"
    ]

    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(n_estimators=200,random_state=42)

    model.fit(X_train,y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test,prediction)
    r2s = r2_score(y_test,prediction)

    joblib.dump(model, "models/stock_model.pkl")

    return model,mae,r2s

