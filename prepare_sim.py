import yfinance as yf
import numpy as np
import json

def get_params(ticker, years=5):
    """
    Fetch historical adjusted close prices for `ticker`,
    calculate annualized expected return (mu) and volatility (sigma).
    """
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, period="max")  # fetch full history
    close = data["Close"].dropna()

    if len(close) < years * 252:
        print(f"⚠️  {ticker}: Only {len(close)} trading days available (less than {years} years). Using all available data.")
        close = close
    else:
        close = close.iloc[-int(years * 252):]  # use only last N years

    # Compute daily log returns
    log_returns = np.log(close / close.shift(1)).dropna()

    mu = log_returns.mean().item() * 252
    sigma = log_returns.std().item() * np.sqrt(252)
    current_price = close.iloc[-1].item()

    result = {
        "ticker": ticker,
        "currentPrice": round(current_price, 2),
        "expectedReturns": round(mu, 4),
        "volatility": round(sigma, 4)
    }

    print(f"✔️  {ticker}: μ = {result['expectedReturns']}  σ = {result['volatility']}  S₀ = {result['currentPrice']}")
    return result

# Example: simulate for these tickers
tickers = {
    "MSFT": 5,
    "META": 10,
    "PLTR": 3,
    "TSLA": 5,
    "GME": 5
}

for ticker, years in tickers.items():
    result = get_params(ticker, years)
