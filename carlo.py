import yfinance as yf
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt

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

import time
import numpy as np
import matplotlib.pyplot as plt

def gbm_simulation():


    ticker = input("Enter a stock ticker for GBM simulation (e.g. AAPL): ").strip().upper()
    years = 5           #int(input("Enter number of years to simulate: "))
    paths = 1000        #int(input("How many paths to simulate? (e.g. 10): "))

    result = get_params(ticker, years)
    S0 = result["currentPrice"]
    mu = result["expectedReturns"]
    sigma = result["volatility"]

    T = years
    dt = 1 / 252
    N = int(T / dt)
    t = np.linspace(0, T, N)

    start_time = time.time()

    # Simuliere GBM Pfade
    simulations = np.zeros((paths, N))
    for i in range(paths):
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)  # Wiener Prozess
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)
        simulations[i] = S

    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    print(f"\nSimulation completed in {duration_ms:.2f} milliseconds.")

    # Plot
    plt.figure(figsize=(12, 6))
    for i in range(paths):
        plt.plot(t, simulations[i], linewidth=1)
    plt.xlabel("Time (Years)")
    plt.ylabel("Simulated Price")
    plt.title(f"Geometric Brownian Motion Simulation for {ticker}")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    gbm_simulation()

