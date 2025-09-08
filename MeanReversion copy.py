#Importing Required Libraries 
import sys
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Utilities
def safe_print(msg):
    print(msg, flush=True)

def compute_metrics(returns, cumulative, trades, leverage=1.0):
    returns = returns.fillna(0)
    n = len(returns)
    metrics = {}
    if n > 0 and len(cumulative) > 0:
        try:
            total_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1
            years = n / 252.0
            cagr = ((cumulative.iloc[-1] / cumulative.iloc[0]) ** (1.0 / years) - 1) if years > 0 else np.nan
        except Exception:
            total_return = (1 + returns).prod() - 1
            cagr = ((1 + total_return) ** (252.0 / n) - 1) if n > 0 else np.nan
    else:
        total_return = np.nan
        cagr = np.nan

    ann_vol = returns.std() * np.sqrt(252) if n > 1 else np.nan
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else np.nan

    running_max = cumulative.cummax() if len(cumulative) > 0 else pd.Series()
    drawdown = (cumulative - running_max) / running_max if len(cumulative) > 0 else pd.Series()
    max_dd = drawdown.min() * 100 if not drawdown.empty else np.nan

    trades_arr = np.array(trades) if trades is not None else np.array([])
    win_rate = (trades_arr > 0).mean() * 100 if trades_arr.size > 0 else np.nan

    metrics["Total Return (%)"] = total_return * 100 if not np.isnan(total_return) else np.nan
    metrics["Annual Return (%)"] = cagr * 100 if not np.isnan(cagr) else np.nan
    metrics["Annual Volatility (%)"] = ann_vol * 100 if not np.isnan(ann_vol) else np.nan
    metrics["Sharpe Ratio"] = sharpe if not np.isnan(sharpe) else np.nan
    metrics["Max Drawdown (%)"] = max_dd
    metrics["Win Rate (%)"] = win_rate
    metrics["Total Trades"] = int(trades_arr.size)
    metrics["Avg Return/Trade (%)"] = trades_arr.mean() * 100 if trades_arr.size > 0 else np.nan
    metrics["Avg Win (%)"] = trades_arr[trades_arr > 0].mean() * 100 if (trades_arr.size > 0 and any(trades_arr > 0)) else np.nan
    metrics["Avg Loss (%)"] = trades_arr[trades_arr < 0].mean() * 100 if (trades_arr.size > 0 and any(trades_arr < 0)) else np.nan
    metrics["Leverage Used"] = leverage
    return metrics

# Simple strategy
class SimpleMeanReversion:
    def __init__(self, initial_capital=100000, leverage=1.0, short=5, long=20, threshold=0.005):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.short = short
        self.long = long
        self.threshold = threshold
        self.ticker = "SPY"

    def fetch_data(self, start, end, debug=False):
        df = yf.download(self.ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError(f"No data downloaded for {self.ticker}")
        if debug:
            safe_print(f"[DEBUG] SPY raw columns: {df.columns.tolist()}")
        if "Adj Close" in df.columns:
            series = df["Adj Close"].dropna()
        elif "Close" in df.columns:
            series = df["Close"].dropna()
        else:
            series = df.iloc[:, 0].dropna()
        series.name = self.ticker
        return series

    def run_backtest(self, price_series):
        if isinstance(price_series, pd.DataFrame):
            s = price_series.iloc[:, 0].dropna()
        elif isinstance(price_series, pd.Series):
            s = price_series.dropna()
        else:
            s = pd.Series(price_series).dropna()
            if len(s) <= 1:
                raise ValueError("Provided price data cannot be coerced to a valid series")

        df = pd.DataFrame({self.ticker: s})
        df["MA_Short"] = df[self.ticker].rolling(self.short).mean()
        df["MA_Long"] = df[self.ticker].rolling(self.long).mean()
        df["SpreadPct"] = (df["MA_Short"] - df["MA_Long"]) / df[self.ticker]

        df["Signal"] = np.where(df["SpreadPct"] > self.threshold, 1,
                        np.where(df["SpreadPct"] < -self.threshold, -1, 0))

        df["Return"] = df["Signal"].shift(1) * df[self.ticker].pct_change() * self.leverage
        df["Return"] = df["Return"].fillna(0)
        cumulative = (1 + df["Return"]).cumprod() * self.initial_capital
        trades = df["Return"][df["Signal"] != 0].tolist()
        metrics = compute_metrics(df["Return"], cumulative, trades, leverage=self.leverage)
        return df, cumulative, metrics

    def plot(self, cumulative):
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative.index, cumulative.values, label="Equity Curve", color="blue")
        plt.title("Simple Mean Reversion Strategy (SPY)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.show()


# Advanced strategy
class AdvancedSimons:
    def __init__(self, initial_capital=100000, leverage=1.8, short=10, long=30, z_threshold=1.0):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.short = short
        self.long = long
        self.z_threshold = z_threshold
        self.tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "GOOG"]

    def fetch_data(self, start, end, debug=False):
        tickers = self.tickers.copy()
        df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, group_by='ticker')

        if df.empty:
            raise RuntimeError("No data downloaded")

        adj_list = []
        for t in tickers:
            try:
                series = df[t]['Adj Close']
            except (KeyError, TypeError):
                # fallback if no multi-level columns
                if 'Adj Close' in df.columns:
                    series = df['Adj Close'][t] if t in df['Adj Close'] else df.iloc[:, 0]
                elif 'Close' in df.columns:
                    series = df['Close'][t] if t in df['Close'] else df.iloc[:, 0]
                else:
                    series = df[t].iloc[:, 0]
            adj_list.append(series.rename(t))

        prices = pd.concat(adj_list, axis=1)
        prices = prices.dropna(how='any')

        if debug:
            safe_print(f"[DEBUG] final prices shape: {prices.shape}, columns: {prices.columns.tolist()}")
        return prices

    def run_backtest(self, prices):
        returns_df = pd.DataFrame(index=prices.index)
        trades = []

        for col in prices.columns:
            df = pd.DataFrame({col: prices[col]})
            df['MA_Short'] = df[col].rolling(self.short).mean()
            df['MA_Long'] = df[col].rolling(self.long).mean()
            df['Spread'] = df['MA_Short'] - df['MA_Long']
            df['Vol'] = df[col].rolling(self.long).std()
            df['ZScore'] = df['Spread'] / df['Vol']
            df['Signal'] = np.where(df['ZScore'] > self.z_threshold, 1,
                             np.where(df['ZScore'] < -self.z_threshold, -1, 0))
            df['Return'] = df['Signal'].shift(1) * df[col].pct_change()
            returns_df[col] = df['Return'].fillna(0)
            trades.extend(df['Return'][df['Signal'] != 0].dropna().tolist())

        # Inverse-vol weighting
        vol = returns_df.rolling(20).std()
        inv_vol = 1 / vol.replace(0, np.nan)
        inv_vol = inv_vol.fillna(method='ffill').fillna(method='bfill').fillna(0)
        weights = inv_vol.div(inv_vol.sum(axis=1).replace(0, 1), axis=0)

        portfolio_returns = (returns_df.fillna(0) * weights).sum(axis=1) * self.leverage
        portfolio_cumulative = (1 + portfolio_returns).cumprod() * self.initial_capital
        metrics = compute_metrics(portfolio_returns, portfolio_cumulative, trades, leverage=self.leverage)
        return portfolio_cumulative, portfolio_returns, metrics

    def plot(self, cumulative, returns):
        drawdown = (cumulative / cumulative.cummax() - 1) * 100
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        axes[0].plot(cumulative.index, cumulative.values, label="Equity Curve", color="blue")
        axes[0].set_ylabel("Portfolio Value"); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(returns.index, returns.values, label="Daily Returns", color="orange")
        axes[1].set_ylabel("Return"); axes[1].legend(); axes[1].grid(True)
        axes[2].plot(drawdown.index, drawdown.values, label="Drawdown (%)", color="red")
        axes[2].set_ylabel("Drawdown (%)"); axes[2].legend(); axes[2].grid(True)
        plt.xlabel("Date"); plt.tight_layout(); plt.show()


# Synthetic test
def synthetic_test():
    safe_print("Running synthetic tests (no network).")
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2018-01-01", "2020-12-31")
    n = len(dates)
    steps = rng.normal(loc=0.0002, scale=0.01, size=n)
    p = 100 * np.exp(np.cumsum(steps))
    series = pd.Series(p, index=dates, name="SPY")
    s = SimpleMeanReversion()
    df, cum, metrics = s.run_backtest(series)
    safe_print("Simple synthetic metrics sample:")
    safe_print(metrics)

    assets = {}
    for i, ticker in enumerate(["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META"]):
        steps = rng.normal(loc=0.0005, scale=0.02, size=n)
        assets[ticker] = 100 * np.exp(np.cumsum(steps))
    prices = pd.DataFrame(assets, index=dates)
    adv = AdvancedSimons()
    cum2, rts, metrics2 = adv.run_backtest(prices)
    safe_print("Advanced synthetic metrics sample:")
    safe_print(metrics2)
    safe_print("Synthetic tests done.")
    return


# Run Backtest Function
def run_backtest(strategy="simple", start="2018-01-01", end=None, debug=False, test=False):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if test:
        synthetic_test()
        return
    try:
        if strategy == "simple":
            s = SimpleMeanReversion()
            series = s.fetch_data(start=start, end=end, debug=debug)
            df, cumulative, metrics = s.run_backtest(series)
            safe_print("\n=== Simple Strategy Metrics ===")
            for k, v in metrics.items():
                safe_print(f"{k:25s}: {v}")
            s.plot(cumulative)
        else:
            s = AdvancedSimons()
            prices = s.fetch_data(start=start, end=end, debug=debug)
            cumulative, returns, metrics = s.run_backtest(prices)
            safe_print("\n=== Advanced Strategy Metrics ===")
            for k, v in metrics.items():
                safe_print(f"{k:25s}: {v}")
            s.plot(cumulative, returns)
    except Exception:
        safe_print("ERROR: exception during run:")
        traceback.print_exc()

# Example Run
if __name__ == "__main__":
    # Simple strategy
    run_backtest(strategy="simple", start="2018-01-01", end="2025-01-01")
    # Advanced strategy
    run_backtest(strategy="advanced", start="2018-01-01", end="2025-01-01")
    # Synthetic test (uncomment to run)
    # run_backtest(test=True)
