# AI Stock Trading Modes & Architecture Guidelines

## User Trading Modes Preference
The user is building an AI Autonomous Stock Trading Platform (炒股的大模型全自动托管平台).
The AI agent supports 3 distinct AI trading modes/profiles while sharing a unified stock domain knowledge base:

1. **`INTRADAY_HIGH_FREQ_SNIPER` (🔥 默认：激进高频日内操盘手)**
   - DEFAULT mode: Very active trading (scanning 1-min bars), high momentum breakouts, full-day trading.
   - Multi-factor weights: Momentum (45%), Volume (35%), Volatility (15%), RSI (5%).

2. **`INTRADAY_DAILY_TARGET_500` (🎯 $500 目标日内止盈收工手)**
   - Reaches $500 daily profit target -> locks profit and stops trading for the day.
   - Strict intraday exit (no overnight hold, 15:55 PM force liquidation, $300 daily max loss).
   - Multi-factor weights: Momentum (35%), Volume (30%), Volatility (25%), RSI (10%).

3. **`OPTIONS_QUANT_TRADER` (🧠 大数据/ML 期权操盘手)**
   - Daily & Weekly Options (0DTE/7DTE Call/Put Spreads & Straddles).
   - Implied Volatility (IV) Rank & Skew analysis, Big Data & Machine Learning quantitative scoring.
   - High-leverage option contract selection, Delta/Gamma risk hedging.

All modes share common technical indicators, market depth data (Level 2), and Alpaca Broker order execution engines.
