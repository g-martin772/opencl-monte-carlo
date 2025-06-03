import yfinance as yf
import numpy as np
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

def get_params(ticker, years=5):
    """
    Fetch historical adjusted close prices for `ticker`,
    calculate annualized expected return (mu) and volatility (sigma).
    """
    info = yf.Ticker(ticker).info
    long_name = info.get("longName", "N/A")

    print(f"Fetching data for {ticker}...\n")

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
        "longName": long_name,
        "currentPrice": round(current_price, 2),
        "expectedReturns": round(mu, 4),
        "volatility": round(sigma, 4)
    }

    print(f"✔️  {ticker}: μ = {result['expectedReturns']}  σ = {result['volatility']}  S₀ = {result['currentPrice']}")
    return result

def main():
    print("\nWelcome to Monte Carlo")
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

    return result

if __name__ == "__main__":
    result = main()
    df = pd.read_csv("price.csv", sep=";", header=None)

    plt.figure(figsize=(16, 8))

    for i in range(df.shape[0]):
        plt.plot(df.iloc[i], label=f"Sim {i}", linewidth=1)

    plt.xlabel("Tage")
    plt.ylabel("Preis")
    plt.title(f"Preis Sim ({df.shape[0]}x): {result['longName']} ({result['ticker']}) start at {result['currentPrice']}$\n")
    plt.grid(True)
    plt.show()
