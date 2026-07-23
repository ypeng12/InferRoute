# AI Stock Trading Modes & Architecture Guidelines

## User Trading Modes Preference
The user is building an AI Autonomous Stock Trading Platform (炒股的大模型全自动托管平台).
The AI agent supports 3 distinct AI trading modes while sharing a unified stock domain knowledge base:

1. **`AGGRESSIVE_DAY_TRADING` (日内激进高频/突击模式)**
   - High intraday frequency (1-min bar scanning), Donchian Breakout + Volume Spike.
   - Strict intraday exit (no overnight hold, 15:55 PM force liquidation or $500 daily target).
   - Multi-factor weights: Momentum (40%), Volume (30%), Volatility (20%), RSI (10%).

2. **`SWING_TREND_INVESTING` (中低频趋势/波段投资模式)**
   - Low trade frequency (1-3 trades per week, multi-day holding).
   - Uses EMA 9/21 trend ribbon, Donchian 20-day high channel breakout, trailing stop-loss.
   - Allows overnight holding and multi-day trend rides.

3. **`OPTIONS_QUANT_ML` (大数据/机器学习期权量化模式)**
   - Daily & Weekly Options (0DTE/7DTE Call/Put Spreads & Straddles).
   - Implied Volatility (IV) Rank & Skew analysis, Big Data & Machine Learning quantitative scoring.
   - High-leverage option contract selection, Delta/Gamma risk hedging.

All modes share common technical indicators, market depth data (Level 2), and Alpaca Broker order execution engines.
