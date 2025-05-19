import yfinance as yf
import numpy as np
import subprocess

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

def main():
    print("Welcome to monte carlo")
    stock = input("Select a stock: ")
    years = int(input("Select number of years: "))

    result = get_params(stock, years)

    print("Starting Monte Carlo simulation...")
    subprocess.call([
        "./cmake-build-debug/opencl_monte_carlo",
        str(result["currentPrice"]),
        str(result["expectedReturns"]),
        str(result["volatility"]),
        str(1 / 252),
        str(int(years) * 252)
    ])

if __name__ == "__main__":
    main()
