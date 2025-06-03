import yfinance as yf
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import time

def get_params(ticker, years=5):
    info = yf.Ticker(ticker).info
    long_name = info.get("longName", "N/A")
    print(f"Fetching data for {ticker}...\n")

    data = yf.download(ticker, period="max")
    close = data["Close"].dropna()

    if len(close) < years * 252:
        print(f"⚠️ Only {len(close)} trading days available – using all.")
    else:
        close = close.iloc[-int(years * 252):]

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

    print(f"\n✔️  {ticker}: μ = {result['expectedReturns']}  σ = {result['volatility']}  S₀ = {result['currentPrice']}")
    return result

def run_gpu_sim(result, years):
    print("Starting GPU-sim...")
    subprocess.call([
        "./cmake-build-debug/opencl_monte_carlo",
        str(result["currentPrice"]),
        str(result["expectedReturns"]),
        str(result["volatility"]),
        str(1 / 252),
        str(int(years) * 252)
    ])
    df = pd.read_csv("price.csv", sep=";", header=None)
    return df

def run_cpu_sim(result, years, paths=1000):
    print("Starting CPU-sim...")
    S0 = result["currentPrice"]
    mu = result["expectedReturns"]
    sigma = result["volatility"]

    T = years
    dt = 1 / 252
    N = int(T / dt)
    t = np.linspace(0, T, N)

    simulations = np.zeros((paths, N))
    start_time = time.time()
    for i in range(paths):
        W = np.random.standard_normal(N)
        W = np.cumsum(W) * np.sqrt(dt)
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)
        simulations[i] = S
    end_time = time.time()

    print(f"Sim duration: {(end_time - start_time)*1000:.2f} ms")
    df = pd.DataFrame(simulations)
    return df

def plot_simulation(df, result, years):
    mean_per_day = df.mean(axis=0)

    plt.figure(figsize=(16, 8))

    for i in range(df.shape[0]):
        #plt.plot(df.iloc[i], color='gray', alpha=0.2, linewidth=0.5)
        plt.plot(df.iloc[i], linewidth=1)
    #plt.plot(mean_per_day, color='black', linewidth=2, label="Average Path")

    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.title(f"Monte Carlo ({df.shape[0]}x): {result['longName']} ({result['ticker']}) – Start: {result['currentPrice']}$")
    plt.grid(True)
    plt.show()

def main():
    print("\nWelcome to the Monte Carlo sim!\n")
    stock = input("Select a stock (Ticker e.g. AAPL, MSFT): ").strip().upper()
    years = int(input("Years of sim: "))
    mode = input("Mode (1=CPU / 2=GPU): ").strip()

    result = get_params(stock, years)

    if mode == "2":
        df = run_gpu_sim(result, years)
    else:
        df = run_cpu_sim(result, years)

    plot_simulation(df, result, years)

if __name__ == "__main__":
    main()
