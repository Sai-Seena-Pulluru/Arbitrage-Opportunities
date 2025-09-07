# ANALYSIS OF HISTORICAL ARBITRAGE OPPORTUNITIES IN CRYPTOCURRENCY MARKETS

# **Objective**

To investigate and simulate cryptocurrency arbitrage opportunities across major exchanges by comparing Direct Arbitrage and Indirect Arbitrage. The study evaluates profitability, stability, and risk-adjusted performance using historical data.

# **Background**

 - Cryptocurrency markets are highly volatile and fragmented, often creating temporary price discrepancies.

 - _Direct Arbitrage:_ Exploits price differences for the same asset between exchanges.

 - _Indirect Arbitrage:_ Uses multi-step conversions across different assets to identify profitable trading paths.

 - Systematic detection and simulation of these opportunities help traders align strategies with their risk appetite and investment goals.

# **Methodology**

_1. Data Collection_

 - Historical OHLCV data gathered from major exchanges using public APIs.

 - Focus on liquid cryptocurrency pairs for reliable comparison.

_2. Preprocessing_

 - Unified schema across exchanges.

 - Filled missing intervals to ensure consistent time-series.

 - Merged datasets by timestamp for cross-exchange analysis.

_3. Arbitrage Simulation_

 - Direct Arbitrage: Compare identical pairs across exchanges; simulate capital growth over time.

 - Indirect Arbitrage: Apply graph-based search (BFS) to evaluate multi-step profitable paths.

 - Performance Metrics: Evaluated using Net Profit, Profit Percentage, Capital Growth, Standard Deviation, and Sharpe Ratio.

# **Results**

_Direct Arbitrage_

 - Net Profit: High absolute profits generated across trades.

 - Profit Percentage: Strong relative returns compared to initial capital.

 - Capital Growth: Rapid increase in simulated investment but with noticeable fluctuations.

 - Standard Deviation: High variability in profit across trades, reflecting volatility.

 - Sharpe Ratio: Moderate, indicating profits come with significant risk.

 - Suitable for short-term aggressive traders willing to tolerate high fluctuations.

_Indirect Arbitrage_

 - Net Profit: Lower total profits than direct arbitrage, but consistent across paths.

 - Profit Percentage: Modest but steady gains relative to starting capital.

 - Capital Growth: Stable upward trend with fewer drawdowns.

 - Standard Deviation: Low, showing reduced variability between trades.

 - Sharpe Ratio: High, demonstrating excellent risk-adjusted returns.

 - Better suited for risk-averse or institutional strategies prioritizing stability.

_Key Insights_

 - There is a trade-off between maximizing profits (direct arbitrage) and maximizing stability (indirect arbitrage).

 - Liquidity, exchange efficiency, and trading path optimization strongly influence outcomes.

 - Risk-adjusted measures (Sharpe Ratio, Std. Dev.) show that indirect arbitrage is more efficient long-term, even if direct arbitrage earns more in absolute terms.

# **Conclusion**

Cryptocurrency arbitrage can be systematically identified and analyzed using historical data.

 - Direct arbitrage delivers higher profits but with greater volatility.

 - Indirect arbitrage provides steadier returns and stronger risk-adjusted performance.

 - Traders and institutions can choose strategies based on risk tolerance.

This framework establishes a foundation for predictive modeling, real-time arbitrage detection, and multi-exchange analysis, making it scalable for live trading applications.

# **Installation**

_1. Clone the repository_

  git clone https://github.com/Sai-Seena-Pulluru/Arbitrage-Opportunities.git

_2. Install dependencies_

  pip install tkcalendar
  
  pip install pandas
  
  pip install numpy
  
  pip install requests
  
  pip install python-binance
  
  pip install matplotlib
  
  pip install seaborn

# **Author**

Sai Seena Pulluru

Toronto Metropolitan University • MSc Data Science & Analytics • 2025
