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

def main():
    print("Welcome to monte carlo")
    stock = input("Select a stock: ")
    years = int(input("Select number of years: "))

    result = get_params(stock, years)

    print("Starting Monte Carlo simulation...")

    executable = os.path.join("cmake-build-debug", "opencl_monte_carlo.exe")

    subprocess.call([
        executable,
        str(result["currentPrice"]),
        str(result["expectedReturns"]),
        str(result["volatility"]),
        str(1 / 252),
        str(int(years) * 252)
    ])

def portfolio():
    print("Portfolio Simulation and Visualization")
    tickers = input("Enter tickers separated by commas (e.g. AAPL,MSFT,GOOGL): ").split(",")
    years = int(input("Select number of years: "))

    returns = []
    volatilities = []
    log_returns_all = {}
    valid_tickers = []

    for ticker in tickers:
        ticker = ticker.strip().upper()
        try:
            print(f"Fetching data for {ticker}...")
            data = yf.download(ticker, period="max")
            close = data["Close"].dropna()
            if len(close) < years * 252:
                close = close
            else:
                close = close.iloc[-int(years * 252):]

            log_returns = np.log(close / close.shift(1)).dropna()
            log_returns_all[ticker] = log_returns
            returns.append(log_returns.mean() * 252)
            volatilities.append(log_returns.std() * np.sqrt(252))
            valid_tickers.append(ticker)
        except Exception as e:
            print(f"Error with {ticker}: {e}")

    if len(valid_tickers) < 2:
        print("Not enough valid data to build a portfolio.")
        return

    aligned_returns = [log_returns_all[ticker] for ticker in valid_tickers]
    returns_df = np.column_stack(aligned_returns)

    cov_matrix = np.cov(returns_df.T) * 252

    n = len(valid_tickers)
    weights = np.array([1/n] * n)

    port_return = np.dot(weights, returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.scatter(volatilities[i], returns[i], label=valid_tickers[i], s=100)
        plt.text(volatilities[i], returns[i], valid_tickers[i], fontsize=9, ha='right')

    plt.scatter(port_volatility, port_return, c='red', marker='X', s=150, label='Average')
    plt.text(port_volatility, port_return, "Average", fontsize=10, fontweight='bold', ha='left', color='black')

    plt.xlabel("Volatility (σ)")
    plt.ylabel("Expected Return (μ)")
    plt.title("Portfolio: Risk vs. Return")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"\nPortfolio Summary (Equal Weights):")
    print(f"Expected Return (μ): {port_return:.4f}")
    print(f"Volatility (σ): {port_volatility:.4f}")

def gbm_simulation():
    import matplotlib.pyplot as plt

    ticker = input("Enter a stock ticker for GBM simulation (e.g. AAPL): ").strip().upper()
    years = int(input("Enter number of years to simulate: "))
    paths = int(input("How many paths to simulate? (e.g. 10): "))

    result = get_params(ticker, years)
    S0 = result["currentPrice"]
    mu = result["expectedReturns"]
    sigma = result["volatility"]

    T = years
    dt = 1/252
    N = int(T / dt)
    t = np.linspace(0, T, N)

    # Simuliere GBM Pfade
    simulations = np.zeros((paths, N))
    for i in range(paths):
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)  # Wiener Prozess
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)
        simulations[i] = S

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
    #main()
    #portfolio()
    gbm_simulation()
