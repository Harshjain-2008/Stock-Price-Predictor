import yfinance as yf
import pandas as pd


def load_stock_data(ticker, start_date):

    df = yf.download(
        ticker,
        start=start_date,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df