# backend/app/agent_modes.py
"""
AI Stock Trading Modes Configuration Module.
Defines 3 distinct trading agent profiles with shared domain knowledge.
"""

from typing import Dict, List

TRADING_MODES: Dict[str, Dict] = {
    "AGGRESSIVE_DAY_TRADING": {
        "id": "AGGRESSIVE_DAY_TRADING",
        "name": "日内激进高频突击模式",
        "icon": "⚡",
        "description": "专注日内高频 K 线突破、成交量异动与开盘突击。日内赚满 $500 锁定收益出场，15:55 强制清仓不持股过夜。",
        "bar_interval": "1m",
        "hold_overnight": False,
        "daily_target": 500.0,
        "daily_stop_loss": 300.0,
        "max_trades_per_day": 20,
        "scoring_weights": {
            "momentum": 0.40,
            "volume_ratio": 0.30,
            "volatility": 0.20,
            "rsi": 0.10
        }
    },
    "SWING_TREND_INVESTING": {
        "id": "SWING_TREND_INVESTING",
        "name": "中低频趋势/波段投资模式",
        "icon": "📈",
        "description": "专注数日至数周的中长线主升浪趋势。依靠 EMA 均线多头与 20 日高点突破建仓，交易频率较低，支持持股过夜与追踪止损。",
        "bar_interval": "1d",
        "hold_overnight": True,
        "daily_target": None,
        "daily_stop_loss": None,
        "max_trades_per_day": 3,
        "scoring_weights": {
            "trend_alignment": 0.50,
            "donchian_breakout": 0.30,
            "rvol_sustainability": 0.20
        }
    },
    "OPTIONS_QUANT_ML": {
        "id": "OPTIONS_QUANT_ML",
        "name": "大数据/ML 期权量化模式",
        "icon": "🧠",
        "description": "利用大数据与机器学习模型对每日/每周期权 (0DTE/7DTE) 进行 IV Rank、波动率偏斜与 Delta/Gamma 对冲分析，高杠杆战术建仓 Call/Put 或跨式组合。",
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

CURRENT_MODE_ID = "AGGRESSIVE_DAY_TRADING"

def get_all_modes() -> List[Dict]:
    return list(TRADING_MODES.values())

def get_current_mode() -> Dict:
    return TRADING_MODES.get(CURRENT_MODE_ID, TRADING_MODES["AGGRESSIVE_DAY_TRADING"])

def set_current_mode(mode_id: str) -> Dict:
    global CURRENT_MODE_ID
    if mode_id in TRADING_MODES:
        CURRENT_MODE_ID = mode_id
    return get_current_mode()
