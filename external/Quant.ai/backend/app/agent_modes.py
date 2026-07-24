# backend/app/agent_modes.py
"""
AI Stock Trading Modes Configuration Module.
Defines 3 distinct AI Agent trading profiles with shared domain knowledge.
"""

from typing import Dict, List

TRADING_MODES: Dict[str, Dict] = {
    "INTRADAY_HIGH_FREQ_SNIPER": {
        "id": "INTRADAY_HIGH_FREQ_SNIPER",
        "name": "🔥 激进高频日内操盘手 (默认)",
        "icon": "⚡",
        "description": "【默认推荐】全天高频盯盘与极速狙击。专注 1分钟 K线突破与成交量突击，交易非常频繁，全天捕捉动量行情。",
        "bar_interval": "1m",
        "hold_overnight": False,
        "daily_target": None,
        "daily_stop_loss": 500.0,
        "max_trades_per_day": 50,
        "scoring_weights": {
            "momentum": 0.45,
            "volume_ratio": 0.35,
            "volatility": 0.15,
            "rsi": 0.05
        }
    },
    "INTRADAY_DAILY_TARGET_500": {
        "id": "INTRADAY_DAILY_TARGET_500",
        "name": "🎯 $500 目标日内止盈收工手",
        "icon": "🎯",
        "description": "稳健日内小目标模式。每日净收益达到 $500 立即自动锁定利润并终止当日交易；15:55 强制清仓绝对不持股过夜。",
        "bar_interval": "1m",
        "hold_overnight": False,
        "daily_target": 500.0,
        "daily_stop_loss": 300.0,
        "max_trades_per_day": 15,
        "scoring_weights": {
            "momentum": 0.35,
            "volume_ratio": 0.30,
            "volatility": 0.25,
            "rsi": 0.10
        }
    },
    "OPTIONS_QUANT_TRADER": {
        "id": "OPTIONS_QUANT_TRADER",
        "name": "🧠 大数据/ML 期权操盘手",
        "icon": "🧠",
        "description": "专攻每日/每周期权 (0DTE/7DTE Call/Put/Straddle)。运用大数据与 ML 机器学习模型计算 IV Rank、波动率偏斜与 Delta/Gamma 希腊字母风控。",
        "bar_interval": "15m",
        "hold_overnight": True,
        "daily_target": 1500.0,
        "daily_stop_loss": 800.0,
        "max_trades_per_day": 10,
        "scoring_weights": {
            "iv_rank_skew": 0.40,
            "ml_prediction_score": 0.30,
            "greeks_delta_gamma": 0.30
        }
    }
}

CURRENT_MODE_ID = "INTRADAY_HIGH_FREQ_SNIPER"

def get_all_modes() -> List[Dict]:
    return list(TRADING_MODES.values())

def get_current_mode() -> Dict:
    return TRADING_MODES.get(CURRENT_MODE_ID, TRADING_MODES["INTRADAY_HIGH_FREQ_SNIPER"])

def set_current_mode(mode_id: str) -> Dict:
    global CURRENT_MODE_ID
    if mode_id in TRADING_MODES:
        CURRENT_MODE_ID = mode_id
    return get_current_mode()
