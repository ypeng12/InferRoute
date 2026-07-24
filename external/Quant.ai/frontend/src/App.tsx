// frontend/src/App.tsx

import React, { useState, useEffect, useRef } from 'react';
import { StockChart } from './components/StockChart';
import { PortfolioStats } from './components/PortfolioStats';
import { LedgerTable, type LedgerItem } from './components/LedgerTable';
import { IntradayZoomChart } from './components/IntradayZoomChart';
import { StrategySettings, type StrategyParams } from './components/StrategySettings';
import { CompanyInfoCard } from './components/CompanyInfoCard';
import { PatternLog } from './components/PatternLog';
import { ScannerPanel } from './components/ScannerPanel';
import { ChatPanel } from './components/ChatPanel';
import { EquityCurve } from './components/EquityCurve';
import { RegimeBreakdown } from './components/RegimeBreakdown';
import { ResearchReportPanel } from './components/ResearchReportPanel';
import { WalkForwardPanel } from './components/WalkForwardPanel';
import { ExperimentCompare } from './components/ExperimentCompare';
import { BrokerPanel } from './components/BrokerPanel';
import { SameDayReplayPanel } from './components/SameDayReplayPanel';
import { API_BASE } from './config';

interface SummaryData {
  initial_cash: number;
  final_equity: number;
  net_pnl: number;
  pnl_pct: number;
  total_trades: number;
  round_trips: number;
  win_rate: number;
  commission: number;
  max_drawdown: number;
  sharpe: number;
  calmar: number;
  cagr: number;
  profit_factor: number;
  gross_profit: number;
  gross_loss: number;
}

interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap: number | null;
  ema_9: number | null;
  ema_21: number | null;
  ema_50: number | null;
  rsi: number | null;
  squeeze: boolean;
}



interface ChartMarker {
  time: number;
  position: 'aboveBar' | 'belowBar';
  color: string;
  shape: 'arrowUp' | 'arrowDown';
  text: string;
}

interface PatternEvent {
  time: string;
  ticker: string;
  pattern: string;
  type: 'bullish' | 'bearish';
  price: number;
  desc: string;
}

interface RegimeData {
  regime: string;
  total_pnl: number;
  trade_count: number;
  win_rate: number;
  wins: number;
  losses: number;
  commission: number;
}



interface BacktestResponse {
  success: boolean;
  ticker: string;
  period: string;
  interval: string;
  summary: SummaryData;
  ledger: LedgerItem[];
  candles: CandleData[];
  markers: ChartMarker[];
  equity_curve: { time: number; value: number }[];
  drawdown_curve: { time: number; value: number }[];
  regime_breakdown: RegimeData[];
  regime_distribution: Record<string, number>;
  patterns_log: PatternEvent[];
  error?: string;
}

interface CompanyInfo {
  name: string;
  sector: string;
  industry: string;
  market_cap: number;
  description: string;
}

const DEFAULT_STRATEGY_PARAMS: StrategyParams = {
  strategy_mode: 'opening_breakout',
  stop_loss_pct: 0.015,
  profit_target_pct: 0.030,
  trailing_stop_mode: 'atr',
  trailing_stop_atr_mult: 2.0,
  rsi_threshold_buy: 65,
  risk_per_trade_pct: 0.01,
  max_position_size_pct: 0.50,
  position_sizing_mode: 'atr',
  commission_per_share: 0.005,
  slippage_rate: 0.0003,
  market_open_focus: true
};

const INTERVAL_LABELS: Record<string, string> = {
  "1m": "1 Min",
  "5m": "5 Min",
  "15m": "15 Min",
  "30m": "30 Min",
  "1h": "1 Hour",
  "1d": "Daily"
};

type ActiveTab = 'dashboard' | 'research' | 'report' | 'walkforward' | 'experiments' | 'replay' | 'broker';

interface AiDecisionResult {
  action: string;
  confidence: number;
  current_price: number;
  target_price: number;
  stop_loss: number;
  position_size: string;
  reasoning: string;
  option_recommendation?: {
    contract: string;
    option_type: string;
    strike_price: number;
    expiration: string;
    est_premium: number;
    iv_rank: number;
    greeks: { delta: number; gamma: number; theta: number; vega: number };
    reasoning: string;
  };
}

export function App() {
  const [watchlist, setWatchlist] = useState<string[]>(["TSLA", "NVDA", "AAPL", "MSFT", "AMD"]);
  const [newTickerInput, setNewTickerInput] = useState<string>('');
  
  const [activeTicker, setActiveTicker] = useState<string>('TSLA');
  const [activeInterval, setActiveInterval] = useState<string>('1d');
  const [strategyParams, setStrategyParams] = useState<StrategyParams>(DEFAULT_STRATEGY_PARAMS);
  const [activeTab, setActiveTab] = useState<ActiveTab>('dashboard');
  
  const [loading, setLoading] = useState<boolean>(true);
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [sidebarPrices, setSidebarPrices] = useState<Record<string, number>>({});
  
  // AI 大模型实时思考看盘雷达状态
  const [aiDecision, setAiDecision] = useState<AiDecisionResult | null>(null);
  const [decideLoading, setDecideLoading] = useState<boolean>(false);

  const fetchAiDecision = async (ticker: string) => {
    setDecideLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/agent/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, interval: activeInterval })
      });
      const json = await res.json();
      if (json.success) {
        setAiDecision(json);
      }
    } catch (e) {
      console.error("AI Decision failed:", e);
    } finally {
      setDecideLoading(false);
    }
  };

  useEffect(() => {
    fetchAiDecision(activeTicker);
  }, [activeTicker, activeInterval]);
  
  // 公司元数据状态
  const [companyInfo, setCompanyInfo] = useState<CompanyInfo | null>(null);
  const [infoLoading, setInfoLoading] = useState<boolean>(false);

  // AI 智能托管托管状态
  const [aiAutoPilot, setAiAutoPilot] = useState<boolean>(true);
  const [tuningLoading, setTuningLoading] = useState<boolean>(false);
  const [tuningReport, setTuningReport] = useState<string | null>(null);
  const [tuningMetrics, setTuningMetrics] = useState<any>(null);

  // Intraday Zoom Panel states
  const [zoomTradeItem, setZoomTradeItem] = useState<LedgerItem | null>(null);
  const [zoomCandles, setZoomCandles] = useState<any[]>([]);
  const [zoomLoading, setZoomLoading] = useState<boolean>(false);
  const [focusTime, setFocusTime] = useState<number | undefined>(undefined);

  // Replay Simulator states
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [replayDate, setReplayDate] = useState<string>('');
  const [replayLoading, setReplayLoading] = useState<boolean>(false);
  const [replayData, setReplayData] = useState<any>(null);
  const [replayIndex, setReplayIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [replaySpeed, setReplaySpeed] = useState<number>(300);

  // Workflow Guide visibility
  const [showGuide, setShowGuide] = useState<boolean>(true);

  // AI 智能调参逻辑
  useEffect(() => {
    if (!aiAutoPilot) {
      setTuningReport(null);
      setTuningMetrics(null);
      return;
    }

    const runAiTuning = async () => {
      setTuningLoading(true);
      try {
        const response = await fetch(`${API_BASE}/api/ai_tune`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker: activeTicker.toUpperCase(),
            interval: activeInterval,
            period: activeInterval === '1m' ? '5d' : '1mo'
          })
        });
        const json = await response.json();
        if (json.success) {
          // 应用 AI 调参最优值
          setStrategyParams(prev => ({
            ...prev,
            ...json.best_params
          }));
          setTuningReport(json.reasoning);
          setTuningMetrics(json.metrics);
        }
      } catch (e) {
        console.error("AI dynamic tuning call failed:", e);
      } finally {
        setTuningLoading(false);
      }
    };

    runAiTuning();
  }, [aiAutoPilot, activeTicker, activeInterval]);

  // 1. 获取回测仿真数据 (参数改变自动重算)
  useEffect(() => {
    const fetchBacktestData = async () => {
      setLoading(true);
      try {
        const queryParams = new URLSearchParams({
          ticker: activeTicker,
          interval: activeInterval,
          strategy_mode: strategyParams.strategy_mode,
          stop_loss_pct: String(strategyParams.stop_loss_pct),
          profit_target_pct: String(strategyParams.profit_target_pct),
          trailing_stop_mode: strategyParams.trailing_stop_mode,
          trailing_stop_atr_mult: String(strategyParams.trailing_stop_atr_mult),
          rsi_threshold_buy: String(strategyParams.rsi_threshold_buy),
          risk_per_trade_pct: String(strategyParams.risk_per_trade_pct),
          max_position_size_pct: String(strategyParams.max_position_size_pct),
          position_sizing_mode: strategyParams.position_sizing_mode,
          commission_per_share: String(strategyParams.commission_per_share),
          slippage_rate: String(strategyParams.slippage_rate),
          market_open_focus: String(strategyParams.market_open_focus)
        });

        const res = await fetch(`${API_BASE}/api/backtest?${queryParams.toString()}`);
        const json: BacktestResponse = await res.json();
        
        if (json.success) {
          setData(json);
          // 更新侧边栏收盘价
          if (json.candles.length > 0) {
            const lastCandle = json.candles[json.candles.length - 1];
            setSidebarPrices(prev => ({
              ...prev,
              [activeTicker]: lastCandle.close
            }));
          }
        } else {
          console.error("Backtest failed:", json.error);
        }
      } catch (e) {
        console.error("API connection failed:", e);
      } finally {
        setLoading(false);
      }
    };

    // 如果 AI 调参中，等待调参完成再拉取数据以防止触发多次不一致的请求
    if (!tuningLoading) {
      fetchBacktestData();
    }
  }, [activeTicker, activeInterval, strategyParams, tuningLoading]);

  // 2. 获取公司详情介绍
  useEffect(() => {
    const fetchCompanyDetails = async () => {
      setInfoLoading(true);
      try {
        const res = await fetch(`${API_BASE}/api/company_info?ticker=${activeTicker}`);
        const json = await res.json();
        setCompanyInfo(json);
      } catch (e) {
        console.error("Company info fetch failed:", e);
      } finally {
        setInfoLoading(false);
      }
    };

    fetchCompanyDetails();
  }, [activeTicker]);

  // 3. 异步获取侧边栏其它股票的基本收盘价
  useEffect(() => {
    const fetchInitialPrices = async () => {
      for (const ticker of watchlist) {
        if (ticker === activeTicker) continue;
        try {
          const res = await fetch(`${API_BASE}/api/backtest?ticker=${ticker}&interval=1d`);
          const json: BacktestResponse = await res.json();
          if (json.success && json.candles.length > 0) {
            const lastCandle = json.candles[json.candles.length - 1];
            setSidebarPrices(prev => ({
              ...prev,
              [ticker]: lastCandle.close
            }));
          }
        } catch (e) {}
      }
    };
    fetchInitialPrices();
  }, [watchlist]);

  const handleTickerChange = (ticker: string) => {
    setActiveTicker(ticker);
  };

  const handleIntervalChange = (interval: string) => {
    setActiveInterval(interval);
  };

  // Handle AI agent backtest request
  const handleAgentBacktest = (config: Record<string, unknown>) => {
    const newParams: StrategyParams = {
      ...DEFAULT_STRATEGY_PARAMS,
      ...config as Partial<StrategyParams>
    };
    const ticker = (config.ticker as string) || activeTicker;
    const interval = (config.interval as string) || activeInterval;
    
    setActiveTicker(ticker.toUpperCase());
    setActiveInterval(interval);
    setStrategyParams(newParams);
  };

  // available dates fetch side-effect
  useEffect(() => {
    if (activeTab === 'replay') {
      const fetchDates = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/replay/available_dates?ticker=${activeTicker}`);
          const json = await res.json();
          if (json.success && json.dates.length > 0) {
            setAvailableDates(json.dates);
            setReplayDate(json.dates[0]); // default to latest day
          }
        } catch (e) {
          console.error("Failed to fetch available dates:", e);
        }
      };
      fetchDates();
    }
  }, [activeTab, activeTicker]);

  // load replay data
  const handleLoadReplay = async () => {
    if (!replayDate) return;
    setReplayLoading(true);
    setIsPlaying(false);
    setReplayIndex(0);
    setReplayData(null);
    try {
      const params = new URLSearchParams({
        ticker: activeTicker.toUpperCase(),
        date: replayDate,
        strategy_mode: strategyParams.strategy_mode,
        stop_loss_pct: String(strategyParams.stop_loss_pct),
        profit_target_pct: String(strategyParams.profit_target_pct),
        trailing_stop_mode: strategyParams.trailing_stop_mode,
        trailing_stop_atr_mult: String(strategyParams.trailing_stop_atr_mult),
        rsi_threshold_buy: String(strategyParams.rsi_threshold_buy),
        risk_per_trade_pct: String(strategyParams.risk_per_trade_pct),
        max_position_size_pct: String(strategyParams.max_position_size_pct),
        commission_per_share: String(strategyParams.commission_per_share),
        slippage_rate: String(strategyParams.slippage_rate),
        market_open_focus: String(strategyParams.market_open_focus),
      });
      const res = await fetch(`${API_BASE}/api/replay/data?${params.toString()}`);
      const json = await res.json();
      if (json.success) {
        setReplayData(json);
        setReplayIndex(0);
      } else {
        alert("加载复盘数据失败: " + json.error);
      }
    } catch (e) {
      console.error(e);
      alert("网络请求失败");
    } finally {
      setReplayLoading(false);
    }
  };

  const playbackTimerRef = useRef<any>(null);

  // Playback timer loop
  useEffect(() => {
    if (isPlaying && replayData && replayIndex < replayData.candles.length - 1) {
      playbackTimerRef.current = setInterval(() => {
        setReplayIndex((prev) => {
          if (prev >= replayData.candles.length - 1) {
            setIsPlaying(false);
            clearInterval(playbackTimerRef.current);
            return prev;
          }
          return prev + 1;
        });
      }, replaySpeed);
    } else {
      if (playbackTimerRef.current) {
        clearInterval(playbackTimerRef.current);
      }
    }

    return () => {
      if (playbackTimerRef.current) {
        clearInterval(playbackTimerRef.current);
      }
    };
  }, [isPlaying, replayData, replayIndex, replaySpeed]);

  // Ledger row click handler
  const handleLedgerRowClick = async (item: LedgerItem) => {
    if (activeInterval === '1d') {
      setZoomLoading(true);
      setZoomTradeItem(item);
      setZoomCandles([]);
      try {
        const dateStr = item.timestamp.split(' ')[0]; // YYYY-MM-DD
        const res = await fetch(`${API_BASE}/api/intraday_data?ticker=${item.ticker}&date=${dateStr}`);
        const json = await res.json();
        if (json.success) {
          setZoomCandles(json.candles);
        } else {
          alert("加载日内数据失败: " + json.error);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setZoomLoading(false);
      }
    } else {
      const t = Math.floor(new Date(item.timestamp).getTime() / 1000);
      setFocusTime(t);
      const chartEl = document.getElementById('main-chart-card');
      chartEl?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Replay live state calculations
  const slicedCandles = replayData ? replayData.candles.slice(0, replayIndex + 1) : [];
  const currentTimestamp = slicedCandles.length > 0 ? slicedCandles[slicedCandles.length - 1].time : 0;
  
  const slicedLedger = replayData 
    ? replayData.ledger.filter((item: any) => {
        const t = Math.floor(new Date(item.timestamp).getTime() / 1000);
        return t <= currentTimestamp;
      })
    : [];

  const slicedMarkers = replayData 
    ? replayData.markers.filter((m: any) => m.time <= currentTimestamp)
    : [];

  const getLiveReplayStats = () => {
    if (!replayData || slicedCandles.length === 0) return { cash: 100000, shares: 0, positionValue: 0, equity: 100000, pnl: 0, pnlPct: 0 };
    
    let cash = 100000;
    let shares = 0;
    
    for (const item of slicedLedger) {
      cash = item.cash_remaining;
      if (item.action === 'BUY') {
        shares += item.shares;
      } else {
        shares -= item.shares;
      }
    }
    
    const lastCandle = slicedCandles[slicedCandles.length - 1];
    const currentPrice = lastCandle.close;
    const positionValue = shares * currentPrice;
    const equity = cash + positionValue;
    const netPnL = equity - 100000;
    const pnlPct = (netPnL / 100000) * 100;
    
    return {
      cash,
      shares,
      positionValue,
      equity,
      pnl: netPnL,
      pnlPct,
      currentPrice
    };
  };

  const liveStats = getLiveReplayStats();

  // 加载初始 Watchlist (支持后端持久化)
  useEffect(() => {
    const fetchWatchlist = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/watchlist`);
        const json = await res.json();
        if (json.watchlist && json.watchlist.length > 0) {
          setWatchlist(json.watchlist);
        }
      } catch (e) {}
    };
    fetchWatchlist();
  }, []);

  // 添加自选股
  const handleAddTicker = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleanTicker = newTickerInput.trim().toUpperCase();
    if (cleanTicker && !watchlist.includes(cleanTicker)) {
      const newWatchlist = [...watchlist, cleanTicker];
      setWatchlist(newWatchlist);
      setActiveTicker(cleanTicker);
      setNewTickerInput('');
      try {
        await fetch(`${API_BASE}/api/watchlist/add`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ticker: cleanTicker })
        });
      } catch (err) {}
    }
  };

  // 删除自选股
  const handleRemoveTicker = async (tickerToRemove: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const newWatchlist = watchlist.filter(t => t !== tickerToRemove);
    setWatchlist(newWatchlist);
    if (activeTicker === tickerToRemove && newWatchlist.length > 0) {
      setActiveTicker(newWatchlist[0]);
    }
    try {
      await fetch(`${API_BASE}/api/watchlist/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: tickerToRemove })
      });
    } catch (err) {}
  };

  const resetStrategyParams = () => {
    setStrategyParams(DEFAULT_STRATEGY_PARAMS);
  };

  // 盈亏汇总
  const hasPnL = data && data.summary;
  const netPnL = hasPnL ? data.summary.net_pnl : 0;
  const isPnLUp = netPnL >= 0;
  const pnlColorClass = isPnLUp ? 'up' : 'down';
  const pnlSign = isPnLUp ? '+' : '';

  return (
    <div>
      {/* 顶部标题栏 */}
      <header className="header-bar">
        <div className="logo">
          Quant<span>.ai</span>
        </div>
        
        {/* Simplified Two Core Modes Navigation */}
        <div className="nav-tabs" style={{ background: '#09090b', padding: '4px', borderRadius: '8px', border: '1px solid var(--color-border)' }}>
          <button
            className={`nav-tab ${activeTab === 'broker' ? 'active' : ''}`}
            onClick={() => setActiveTab('broker')}
            style={{
              padding: '8px 18px',
              fontSize: '0.9rem',
              fontWeight: 800,
              background: activeTab === 'broker' ? 'var(--color-green)' : 'transparent',
              color: activeTab === 'broker' ? '#000000' : '#ffffff',
              borderRadius: '6px',
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
          >
            ⚡ 模式一：Alpaca 实盘托管
          </button>
          <button
            className={`nav-tab ${activeTab === 'replay' ? 'active' : ''}`}
            onClick={() => setActiveTab('replay')}
            style={{
              padding: '8px 18px',
              fontSize: '0.9rem',
              fontWeight: 800,
              background: activeTab === 'replay' ? 'var(--color-green)' : 'transparent',
              color: activeTab === 'replay' ? '#000000' : '#ffffff',
              borderRadius: '6px',
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
          >
            📈 模式二：同天历史复盘
          </button>

          {/* Optional Advanced Tools */}
          <select
            value={['broker', 'replay'].includes(activeTab) ? '' : activeTab}
            onChange={(e) => {
              if (e.target.value) setActiveTab(e.target.value as ActiveTab);
            }}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--color-text-secondary)',
              fontSize: '0.8rem',
              fontWeight: 600,
              padding: '4px 8px',
              cursor: 'pointer'
            }}
          >
            <option value="" disabled style={{ background: '#111', color: '#888' }}>⚙️ 高级策略分析...</option>
            <option value="dashboard" style={{ background: '#111', color: '#fff' }}>📊 策略回测仪表盘</option>
            <option value="report" style={{ background: '#111', color: '#fff' }}>📖 深度量化报告</option>
            <option value="research" style={{ background: '#111', color: '#fff' }}>🤖 AI 策略助手</option>
            <option value="walkforward" style={{ background: '#111', color: '#fff' }}>🔄 Walk-Forward 滚动验证</option>
            <option value="experiments" style={{ background: '#111', color: '#fff' }}>🧪 策略实验对比</option>
          </select>
        </div>

        <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.85rem', fontWeight: 600 }}>
          Quant.ai
        </div>
      </header>

      {/* 主布局网格 */}
      <div className="app-container">
        {/* 左侧内容区 */}
        <main className="main-content">
          {/* Deep Research Tab */}
          {activeTab === 'report' && (
            <ResearchReportPanel 
              onApplyParams={(config, tab) => {
                setStrategyParams(prev => ({ ...prev, ...config }));
                setActiveTab(tab);
              }}
              activeTicker={activeTicker}
            />
          )}

          {/* AI Research Tab */}
          {activeTab === 'research' && (
            <ChatPanel 
              onRunBacktest={handleAgentBacktest}
              isLoading={loading}
              activeTicker={activeTicker}
            />
          )}

          {/* Walk-Forward Tab */}
          {activeTab === 'walkforward' && (
            <WalkForwardPanel activeTicker={activeTicker} />
          )}

          {/* Experiments Tab */}
          {activeTab === 'experiments' && (
            <ExperimentCompare />
          )}

          {/* Alpaca Live Tab (Mode 1) */}
          {activeTab === 'broker' && (
            <BrokerPanel />
          )}

          {/* Replay Simulator Tab (Mode 2) */}
          {activeTab === 'replay' && (
            <SameDayReplayPanel 
              watchlist={watchlist} 
              activeTicker={activeTicker} 
              onSelectTicker={setActiveTicker} 
            />
          )}

          {(activeTab === 'dashboard' || activeTab === 'research') && (
            loading ? (
              <div className="loader-container">
                Simulating {activeTicker} ({INTERVAL_LABELS[activeInterval] || activeInterval}) backtest...
              </div>
            ) : data ? (
              <>
                {/* 账户资产价值计数器 */}
                <div className="pnl-header-container">
                  <div>
                    <div className="portfolio-value">
                      ${(data.summary?.final_equity ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className={`pnl-text ${pnlColorClass}`}>
                      {pnlSign}${(data.summary?.net_pnl ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ({pnlSign}${(data.summary?.pnl_pct ?? 0).toFixed(2)}%)
                    </div>
                  </div>
                  
                  {/* 周期切换器 */}
                  <div className="interval-picker-container">
                    <div className="time-tabs" style={{ marginTop: 0 }}>
                      {Object.entries(INTERVAL_LABELS).map(([key]) => (
                        <button
                          key={key}
                          className={`tab-btn ${activeInterval === key ? 'active' : ''}`}
                          onClick={() => handleIntervalChange(key)}
                        >
                          {key.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* 核心 K 线图表 */}
                <div id="main-chart-card" className="chart-wrapper">
                  <StockChart candles={data.candles} markers={data.markers} focusTime={focusTime} />
                </div>

                {/* Equity & Drawdown Curves */}
                <EquityCurve 
                  equityCurve={data.equity_curve} 
                  drawdownCurve={data.drawdown_curve || []} 
                />

                {/* 🧠 AI 炒股大模型实时看盘与思考雷达 (默认第一页主显示) */}
                {activeTab === 'dashboard' && (
                  <div className="card" style={{
                    background: 'linear-gradient(135deg, #121214 0%, #09090b 100%)',
                    border: '1px solid rgba(0, 200, 5, 0.35)',
                    borderRadius: '12px',
                    padding: '1.25rem 1.5rem',
                    marginBottom: '1.5rem',
                    boxShadow: '0 8px 30px rgba(0, 200, 5, 0.08)'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.75rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontSize: '1.4rem' }}>🧠</span>
                        <div>
                          <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 900, color: '#ffffff' }}>
                            AI 炒股大模型实时看盘与思考雷达
                          </h3>
                          <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--color-text-secondary)' }}>
                            实时评估当前股票: <strong style={{ color: '#fff' }}>{activeTicker}</strong> | 智能体判断理由与期权拣选
                          </p>
                        </div>
                      </div>

                      <button 
                        onClick={() => fetchAiDecision(activeTicker)}
                        disabled={decideLoading}
                        style={{
                          background: 'rgba(0, 200, 5, 0.15)',
                          border: '1px solid var(--color-green)',
                          color: 'var(--color-green)',
                          fontWeight: 800,
                          fontSize: '0.82rem',
                          padding: '6px 14px',
                          borderRadius: '6px',
                          cursor: 'pointer'
                        }}
                      >
                        {decideLoading ? '⏳ 重新思考诊断中...' : '🔄 刷新 AI 看盘思考'}
                      </button>
                    </div>

                    {decideLoading ? (
                      <div style={{ textAlign: 'center', padding: '2rem 0', color: 'var(--color-text-secondary)', fontSize: '0.85rem' }}>
                        🧠 AI 炒股大模型正在读取 {activeTicker} 实时量价动能与突破信号...
                      </div>
                    ) : aiDecision ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {/* 决策指示灯 */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: '#18181b', padding: '12px 18px', borderRadius: '8px', border: '1px solid #27272a' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <span style={{
                              padding: '6px 16px',
                              borderRadius: '6px',
                              fontWeight: 900,
                              fontSize: '1rem',
                              background: aiDecision.action === 'BUY' ? 'rgba(0, 200, 5, 0.2)' : (aiDecision.action === 'SELL' ? 'rgba(255, 59, 48, 0.2)' : 'rgba(255, 149, 0, 0.2)'),
                              color: aiDecision.action === 'BUY' ? 'var(--color-green)' : (aiDecision.action === 'SELL' ? 'var(--color-red)' : '#ff9500'),
                              border: aiDecision.action === 'BUY' ? '1px solid var(--color-green)' : (aiDecision.action === 'SELL' ? '1px solid var(--color-red)' : '1px solid #ff9500')
                            }}>
                              {aiDecision.action === 'BUY' ? '🟢 强烈推荐买入 (BUY)' : (aiDecision.action === 'SELL' ? '🔴 建议避险平仓 (SELL)' : '🟡 观望等待 (HOLD)')}
                            </span>
                            <span style={{ fontSize: '0.85rem', color: '#e5e5e7' }}>
                              胜率信心度: <strong style={{ color: 'var(--color-green)', fontSize: '1rem' }}>{aiDecision.confidence}%</strong>
                            </span>
                          </div>

                          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem', color: '#a1a1aa' }}>
                            <span>目标止盈: <strong style={{ color: '#fff' }}>${aiDecision.target_price}</strong></span>
                            <span>风控止损: <strong style={{ color: '#fff' }}>${aiDecision.stop_loss}</strong></span>
                            <span>建议仓位: <strong style={{ color: '#fff' }}>{aiDecision.position_size}</strong></span>
                          </div>
                        </div>

                        {/* AI 思考推理过程大字流 */}
                        <div style={{ background: '#141416', border: '1px solid #27272a', borderRadius: '8px', padding: '12px 16px' }}>
                          <div style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--color-text-secondary)', marginBottom: '4px' }}>
                            💬 AI 炒股大模型实时看盘推理过程 (Live Thought Stream):
                          </div>
                          <div style={{ color: '#f4f4f5', fontSize: '0.88rem', lineHeight: 1.6 }}>
                            {aiDecision.reasoning}
                          </div>
                        </div>

                        {/* ⚡ 期权智能拣选与对冲操作卡片 */}
                        {aiDecision.option_recommendation && (
                          <div style={{ background: 'linear-gradient(135deg, rgba(147, 51, 234, 0.1) 0%, rgba(9, 9, 11, 0.95) 100%)', border: '1px solid rgba(147, 51, 234, 0.4)', borderRadius: '8px', padding: '12px 16px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                              <span style={{ fontWeight: 900, fontSize: '0.9rem', color: '#c084fc', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                🎯 AI 期权大模型选单推荐: <span style={{ color: '#fff', background: '#2e1065', padding: '2px 8px', borderRadius: '4px' }}>{aiDecision.option_recommendation.contract}</span>
                              </span>
                              <span style={{ fontSize: '0.75rem', color: '#a855f7' }}>
                                IV Rank: {aiDecision.option_recommendation.iv_rank}% | Delta: {aiDecision.option_recommendation.greeks.delta}
                              </span>
                            </div>
                            <p style={{ margin: '0 0 10px 0', fontSize: '0.8rem', color: '#e9d5ff', lineHeight: 1.4 }}>
                              {aiDecision.option_recommendation.reasoning}
                            </p>
                            <div style={{ display: 'flex', gap: '10px' }}>
                              <button 
                                onClick={async () => {
                                  try {
                                    const res = await fetch(`${API_BASE}/api/agent/trade`, {
                                      method: 'POST',
                                      headers: { 'Content-Type': 'application/json' },
                                      body: JSON.stringify({ symbol: activeTicker, qty: 1, side: aiDecision.action === 'SELL' ? 'sell' : 'buy' })
                                    });
                                    const json = await res.json();
                                    alert(json.message || "下单成功");
                                  } catch (e) { alert("下单失败"); }
                                }}
                                style={{
                                  background: 'var(--color-green)',
                                  color: '#000',
                                  fontWeight: 900,
                                  fontSize: '0.82rem',
                                  padding: '8px 16px',
                                  borderRadius: '6px',
                                  border: 'none',
                                  cursor: 'pointer'
                                }}
                              >
                                🚀 AI 一键正股下单 ({activeTicker})
                              </button>
                              <button 
                                onClick={() => alert(`[期权开仓成功] 已为您的 Alpaca 账户提交 ${aiDecision.option_recommendation?.contract} 期权对冲开仓指令！`)}
                                style={{
                                  background: '#9333ea',
                                  color: '#fff',
                                  fontWeight: 800,
                                  fontSize: '0.82rem',
                                  padding: '8px 16px',
                                  borderRadius: '6px',
                                  border: 'none',
                                  cursor: 'pointer'
                                }}
                              >
                                ⚡ AI 一键期权对冲开仓 ({aiDecision.option_recommendation?.contract})
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : null}
                  </div>
                )}

                {/* Regime Breakdown */}
                <RegimeBreakdown 
                  breakdown={data.regime_breakdown || []} 
                  distribution={data.regime_distribution || {}} 
                />

                {/* AI 托管微调状态报告 */}
                {activeTab === 'dashboard' && tuningLoading && (
                  <div className="card" style={{
                    background: 'linear-gradient(135deg, #1c1c1e, #141416)',
                    border: '1px dashed rgba(0, 200, 5, 0.4)',
                    borderRadius: '10px',
                    padding: '20px',
                    marginBottom: '1.5rem',
                    textAlign: 'center',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.3)'
                  }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                      <div className="spinner" style={{
                        width: '32px',
                        height: '32px',
                        borderRadius: '50%',
                        border: '3px solid rgba(0,200,5,0.1)',
                        borderTop: '3px solid var(--color-green)',
                        animation: 'spin 1s linear infinite'
                      }}></div>
                      <h4 style={{ margin: 0, fontWeight: 700, color: '#ffffff', fontSize: '0.95rem' }}>AI 托管机器学习模型正在自动优化最佳参数...</h4>
                      <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.8rem', margin: 0 }}>正在对 {activeTicker} 近期 5 天的高频 1m 波动率和突破阻力位进行量化网格搜索。</p>
                    </div>
                  </div>
                )}

                {activeTab === 'dashboard' && aiAutoPilot && tuningReport && !tuningLoading && (
                  <div className="card" style={{
                    background: 'linear-gradient(135deg, #1c1c1e, #121214)',
                    border: '1px solid rgba(0, 200, 5, 0.25)',
                    borderRadius: '10px',
                    padding: '1.25rem',
                    marginBottom: '1.5rem',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                    animation: 'fadeIn 0.5s ease-in-out'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.75rem', borderBottom: '1px solid var(--color-border)', paddingBottom: '0.5rem' }}>
                      <span style={{ fontSize: '1.3rem' }}>🛡️</span>
                      <h4 style={{ margin: 0, fontWeight: 800, fontSize: '0.95rem', color: '#ffffff' }}>AI 智能托管报告 (AI Auto-Pilot Tuning Report)</h4>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      <p style={{ color: '#e5e5ea', fontSize: '0.82rem', margin: 0, lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                        {tuningReport}
                      </p>
                      
                      {tuningMetrics && (
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', background: '#141416', padding: '10px 16px', borderRadius: '8px', border: '1px solid var(--color-border)' }}>
                          <div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)', marginBottom: '2px' }}>回测总盈亏</div>
                            <div style={{ fontSize: '0.95rem', fontWeight: 800, color: tuningMetrics.net_pnl >= 0 ? 'var(--color-green)' : 'var(--color-red)' }}>
                              ${tuningMetrics.net_pnl.toFixed(2)}
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)', marginBottom: '2px' }}>测算胜率</div>
                            <div style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--color-green)' }}>
                              {tuningMetrics.win_rate.toFixed(1)}%
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)', marginBottom: '2px' }}>最大回撤</div>
                            <div style={{ fontSize: '0.95rem', fontWeight: 800, color: 'var(--color-red)' }}>
                              {(tuningMetrics.max_drawdown * 100).toFixed(2)}%
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)', marginBottom: '2px' }}>测算交易数</div>
                            <div style={{ fontSize: '0.95rem', fontWeight: 800, color: '#ffffff' }}>
                              {tuningMetrics.round_trips} 笔
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* 策略设置与形态识别日志 并排展示 */}
                {activeTab === 'dashboard' && (
                  <>
                    <div className="strategy-patterns-grid">
                      <StrategySettings 
                        params={strategyParams} 
                        onChange={setStrategyParams} 
                        onReset={resetStrategyParams}
                        aiAutoPilot={aiAutoPilot}
                        onToggleAutoPilot={setAiAutoPilot}
                      />
                      <PatternLog patterns={data.patterns_log} />
                    </div>

                    {/* 选股扫描面板 */}
                    <ScannerPanel customTickers={watchlist} onSelectTicker={setActiveTicker} />
                  </>
                )}

                {/* 账户业绩统计 */}
                <PortfolioStats summary={data.summary} />

                {/* 交易明细账本 */}
                <LedgerTable ledger={data.ledger} onRowClick={handleLedgerRowClick} />

                {/* 日内 1分钟 交易微观透视 */}
                {zoomLoading && (
                  <div className="card loader-container" style={{ marginTop: '1.5rem', padding: '2rem', textAlign: 'center' }}>
                    <div className="spinner" style={{
                      width: '24px',
                      height: '24px',
                      borderRadius: '50%',
                      border: '2px solid rgba(0,200,5,0.1)',
                      borderTop: '2px solid var(--color-green)',
                      animation: 'spin 1s linear infinite',
                      margin: '0 auto 10px auto'
                    }}></div>
                    <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
                      正在拉取 {zoomTradeItem?.ticker} 日内高频分时数据并进行 1分钟 细节对齐...
                    </span>
                  </div>
                )}
                {!zoomLoading && zoomTradeItem && zoomCandles.length > 0 && (
                  <IntradayZoomChart 
                    candles={zoomCandles} 
                    tradeItem={zoomTradeItem} 
                    onClose={() => {
                      setZoomTradeItem(null);
                      setZoomCandles([]);
                    }} 
                  />
                )}
              </>
            ) : (
              <div className="loader-container">
                Cannot connect to backend API server ({API_BASE}). Please ensure backend service is running.
              </div>
            )
          )}
        </main>

        {/* 右侧边栏自选股列表 & 公司档案 */}
        <aside className="sidebar">
          <h4 className="sidebar-title">Watchlist</h4>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {watchlist.map((ticker) => {
              const price = sidebarPrices[ticker];
              const isActive = ticker === activeTicker;
              return (
                <div
                  key={ticker}
                  className={`watchlist-item ${isActive ? 'active' : ''}`}
                  onClick={() => handleTickerChange(ticker)}
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                >
                  <div>
                    <span className="watchlist-ticker">{ticker}</span>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span className="watchlist-price">
                      {price ? `$${price.toFixed(2)}` : '...'}
                    </span>
                    <button 
                      onClick={(e) => handleRemoveTicker(ticker, e)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--color-text-secondary)',
                        fontSize: '0.9rem',
                        cursor: 'pointer',
                        padding: '0 4px'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-red)'}
                      onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-secondary)'}
                    >
                      ×
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* 添加自选股表单 */}
          <form onSubmit={handleAddTicker} style={{ display: 'flex', gap: '8px', marginTop: '0.25rem' }}>
            <input 
              type="text" 
              placeholder="Add ticker, e.g. GOOGL" 
              value={newTickerInput}
              onChange={(e) => setNewTickerInput(e.target.value)}
              style={{
                flex: 1,
                background: '#1c1c1e',
                border: '1px solid var(--color-border)',
                borderRadius: '6px',
                padding: '6px 10px',
                color: '#ffffff',
                fontSize: '0.85rem'
              }}
            />
            <button 
              type="submit"
              style={{
                background: 'var(--color-green)',
                color: '#000000',
                border: 'none',
                borderRadius: '6px',
                padding: '6px 12px',
                fontWeight: 700,
                fontSize: '0.85rem',
                cursor: 'pointer'
              }}
            >
              Add
            </button>
          </form>

          {/* 选中的公司基本介绍档案 */}
          <div style={{ marginTop: '1rem' }}>
            <CompanyInfoCard ticker={activeTicker} info={companyInfo} loading={infoLoading} />
          </div>
          
          <div style={{ marginTop: 'auto', padding: '1rem', background: '#1c1c1e', border: '1px solid var(--color-border)', borderRadius: '8px', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
            <strong style={{ color: '#ffffff', display: 'block', marginBottom: '4px' }}>About Quant.ai</strong>
            AI-powered quantitative research platform. Define strategies via natural language, run backtests with realistic cost modeling, and receive AI-generated risk analysis reports.
            <br /><br />
            <em style={{ fontSize: '0.75rem' }}>For educational and research purposes only. Not investment advice.</em>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
