import yfinance as yf

def load_stock_data(ticker, start_date):

    df = yf.download(
        ticker,
        start=start_date,
        progress=False
    )

    df.reset_index(inplace=True)

    return df