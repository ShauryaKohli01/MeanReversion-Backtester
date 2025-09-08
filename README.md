# Introduction and Origins


Mean reversion is a classic trading concept based on the idea that asset prices tend to revert to their historical average over time. 
This principle has been used in finance for decades, with origins in statistical arbitrage and quantitative trading. 
Legendary hedge fund manager Jim Simons and his firm, Renaissance Technologies, famously applied sophisticated mean reversion strategies acr
oss multiple assets, using statistical models, z-scores, and risk weighting to exploit temporary mispricings in the market.

<img width="489" height="230" alt="image" src="https://github.com/user-attachments/assets/4f93eff3-8afc-4c03-b6e2-8c467082c45a" />



In our Python implementation, we developed a framework to backtest mean reversion strategies both on a single asset (SPY) and a portfolio of US tickers. The code uses yfinance to fetch historical prices, computes moving averages, spreads, and z-scores, generates trading signals, calculates returns, and evaluates key metrics like Sharpe ratio, drawdown, and win rate. It also produces visualizations to understand performance over time.

# Simple Mean Reversion Strategy

The Simple Mean Reversion Strategy focuses on a single asset, SPY, and is designed to capitalize on temporary 
deviations from its historical average price. It computes both short-term and long-term moving averages to identify when the current price is significantly above or below its recent trend. 


By calculating the percentage spread between these moving averages, the strategy generates buy or sell signals whenever the spread crosses a predefined threshold, indicating a likely reversion toward the mean. Leverage can be applied to amplify returns, allowing the strategy to take larger positions relative to the capital. Throughout the backtest, it tracks daily returns and computes cumulative portfolio value to monitor performance over time. Key metrics, such as total return, average return per trade, and win rate, provide insight into the strategy’s effectiveness. 

Finally, a visual equity curve allows users to see how the portfolio grows, highlighting profitable periods as well as drawdowns, giving a clear picture of risk and reward.



# Advanced Simons-style Strategy

The Advanced Simons-style Strategy builds upon the simple mean reversion approach by applying it to a portfolio of multiple US tickers, 
such as AAPL, MSFT, NVDA, and others. For each asset, it computes rolling z-scores to standardize price deviations, 
ensuring that signals are comparable across stocks with different price ranges and volatilities. 
To manage risk, the strategy applies inverse-volatility weighting, giving more weight to less volatile assets and less to highly volatile ones,
thereby balancing the overall portfolio exposure. Individual asset returns are aggregated into a portfolio return, and leverage can be applied to enhance gains. 


The strategy is designed to handle missing data and periods of low volatility, preventing calculation errors and maintaining robustness. Throughout the backtest, it tracks cumulative portfolio value and daily returns, while computing key performance metrics similar to the simple strategy, such as total return, Sharpe ratio, and drawdowns. 
This approach provides a more diversified and risk-adjusted method for capturing mean reversion opportunities across multiple assets.

# Graphs

Equity Curve: Shows portfolio growth over time. Smooth upward trends indicate profitable periods; dips reflect losses or drawdowns.

<img width="1146" height="576" alt="image" src="https://github.com/user-attachments/assets/c9dacaa3-cdc3-455a-837e-849f5f261542" />


Drawdown / Daily Returns (Advanced): Drawdown visualizes losses from peak value, while daily returns show volatility and consistency of performance. Both help assess risk-adjusted returns.

<img width="1461" height="871" alt="image" src="https://github.com/user-attachments/assets/01015c16-522b-44c6-8782-cb8b704b93fd" />

# Conclusion

Mean reversion strategies aim to exploit temporary price deviations from a historical average. The Python code provides a clear framework to test both simple and multi-asset mean reversion strategies, incorporating key quantitative techniques like z-scores and inverse-volatility weighting. The generated metrics and graphs allow traders to understand returns, risk, and the effectiveness of their strategies, making this a practical tool for exploring statistical trading methods inspired by pioneers like Jim Simons.
