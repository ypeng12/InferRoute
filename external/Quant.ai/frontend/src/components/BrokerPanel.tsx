// frontend/src/components/BrokerPanel.tsx

import { useState, useEffect } from 'react';
import { API_BASE } from '../config';

interface AccountSummary {
  success: boolean;
  account_number: string;
  status: string;
  cash: number;
  portfolio_value: number;
  buying_power: number;
  equity: number;
}

interface BrokerPosition {
  ticker: string;
  shares: number;
  avg_entry_price: number;
  market_value: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

interface BrokerOrder {
  order_id: string;
  symbol: string;
  qty: number;
  side: string;
  type: string;
  status: string;
  submitted_at: string;
  filled_at: string | null;
  filled_avg_price: number;
}

interface AgentMode {
  id: string;
  name: string;
  icon: string;
  description: string;
  bar_interval: string;
  hold_overnight: boolean;
  daily_target: number | null;
  daily_stop_loss: number | null;
  max_trades_per_day: number;
}

export function BrokerPanel() {
  const [account, setAccount] = useState<AccountSummary | null>(null);
  const [positions, setPositions] = useState<BrokerPosition[]>([]);
  const [orders, setOrders] = useState<BrokerOrder[]>([]);
  const [modes, setModes] = useState<AgentMode[]>([]);
  const [currentMode, setCurrentMode] = useState<AgentMode | null>(null);
  const [isBotRunning, setIsBotRunning] = useState<boolean>(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Poll status, account, and positions
  const fetchBrokerData = async () => {
    try {
      // 1. Fetch account stats
      const accRes = await fetch(`${API_BASE}/api/broker/account`);
      const accJson = await accRes.json();
      if (accJson.success !== false) {
        setAccount(accJson);
        setErrorMsg(null);
      } else {
        setErrorMsg(accJson.error || "无法获取 Alpaca 账户信息，请检查 Keys 配置。");
      }

      // 2. Fetch positions
      const posRes = await fetch(`${API_BASE}/api/broker/positions`);
      const posJson = await posRes.json();
      if (posJson.success) {
        setPositions(posJson.positions);
      }

      // 3. Fetch order history
      const orderRes = await fetch(`${API_BASE}/api/broker/orders`);
      const orderJson = await orderRes.json();
      if (orderJson.success && orderJson.orders) {
        setOrders(orderJson.orders);
      }

      // 4. Fetch bot status and logs
      const statusRes = await fetch(`${API_BASE}/api/live/status`);
      const statusJson = await statusRes.json();
      if (statusJson.success) {
        setIsBotRunning(statusJson.status.is_running);
        setLogs(statusJson.logs);
      }

      // 5. Fetch AI Agent Modes
      const modeRes = await fetch(`${API_BASE}/api/agent/modes`);
      const modeJson = await modeRes.json();
      if (modeJson.success) {
        setModes(modeJson.modes);
        setCurrentMode(modeJson.current_mode);
      }

    } catch (e) {
      console.error("Error fetching broker data:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectMode = async (modeId: string) => {
    setActionLoading(`mode_${modeId}`);
    try {
      const res = await fetch(`${API_BASE}/api/agent/mode/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode_id: modeId })
      });
      const json = await res.json();
      if (json.success) {
        setCurrentMode(json.current_mode);
      }
    } catch (e) {
      alert("模式切换失败");
    } finally {
      setActionLoading(null);
    }
  };

  useEffect(() => {
    fetchBrokerData();
    const interval = setInterval(fetchBrokerData, 3000); // Poll every 3 seconds
    return () => clearInterval(interval);
  }, []);

  const handleStartBot = async () => {
    setActionLoading("start");
    try {
      const res = await fetch(`${API_BASE}/api/live/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ignore_market_hours: true })
      });
      const json = await res.json();
      if (json.success) {
        setIsBotRunning(json.status.is_running);
        setLogs(json.logs || []);
      } else {
        alert("启动失败: " + (json.status?.logs?.[json.status.logs.length - 1] || "错误。"));
      }
    } catch (e) {
      alert("请求失败");
    } finally {
      setActionLoading(null);
      fetchBrokerData();
    }
  };

  const handleStopBot = async () => {
    setActionLoading("stop");
    try {
      const res = await fetch(`${API_BASE}/api/live/stop`, { method: 'POST' });
      const json = await res.json();
      setIsBotRunning(json.status.is_running);
    } catch (e) {
      alert("请求失败");
    } finally {
      setActionLoading(null);
      fetchBrokerData();
    }
  };

  const handleCancelAllOrders = async () => {
    if (!window.confirm("确定要撤销 Alpaca 账户中的所有未成交挂单吗？")) return;
    setActionLoading("cancel_orders");
    try {
      const res = await fetch(`${API_BASE}/api/broker/cancel_orders`, { method: 'POST' });
      const json = await res.json();
      alert(json.message || "撤单请求已发送");
    } catch (e) {
      alert("撤单失败");
    } finally {
      setActionLoading(null);
      fetchBrokerData();
    }
  };

  const handleForceLiquidate = async () => {
    if (!window.confirm("🚨 警告：这会以市价立即平仓所有股票持仓！确定继续吗？")) return;
    setActionLoading("liquidate");
    try {
      const res = await fetch(`${API_BASE}/api/broker/close_positions`, { method: 'POST' });
      const json = await res.json();
      alert(json.message || "清仓请求已发送");
    } catch (e) {
      alert("平仓失败");
    } finally {
      setActionLoading(null);
      fetchBrokerData();
    }
  };

  // Extract key action events for clean visual feed
  const getActionFeeds = () => {
    const actionLogs = logs.filter(l => 
      l.includes('触发买入') || l.includes('触发卖出') || l.includes('买单提交') || l.includes('平仓单提交') || l.includes('开始新一轮')
    );
    return actionLogs.slice(-6).reverse(); // Latest 6 events
  };

  const actionFeeds = getActionFeeds();

  if (loading && !account) {
    return (
      <div className="loader-container" style={{ padding: '4rem', textAlign: 'center' }}>
        正在连接 Alpaca 账户...
      </div>
    );
  }

  if (errorMsg) {
    return (
      <div className="card" style={{ padding: '2.5rem', textAlign: 'center', border: '1px solid var(--color-red)', background: 'rgba(255, 59, 48, 0.05)' }}>
        <h3 style={{ color: 'var(--color-red)', marginTop: 0, fontSize: '1.2rem' }}>🔌 Alpaca 账户未连接</h3>
        <p style={{ color: '#e5e5e7', fontSize: '0.95rem', margin: '15px 0' }}>{errorMsg}</p>
        <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
          请在配置文件 <code style={{ color: '#fff', background: '#111', padding: '2px 6px', borderRadius: '4px' }}>backend/.env</code> 中添加您的 Alpaca API Key。
        </div>
      </div>
    );
  }

  return (
    <div className="fade-in">
      {/* 顶部一键托管大操作卡片 */}
      <div 
        className="card" 
        style={{ 
          marginBottom: '1.5rem', 
          padding: '1.5rem 2rem',
          background: isBotRunning 
            ? 'linear-gradient(135deg, rgba(0, 200, 5, 0.08) 0%, rgba(9, 9, 11, 0.95) 100%)' 
            : 'linear-gradient(135deg, #18181b 0%, #09090b 100%)',
          border: isBotRunning ? '1px solid rgba(0, 200, 5, 0.4)' : '1px solid var(--color-border)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1.5rem'
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: isBotRunning ? 'var(--color-green)' : '#8e8e93',
              boxShadow: isBotRunning ? '0 0 12px var(--color-green)' : 'none'
            }}></div>
            <h2 style={{ margin: 0, fontSize: '1.4rem', fontWeight: 900, color: '#ffffff' }}>
              {isBotRunning ? '⚡ AI 量化托管交易中' : '⏸️ 托管交易已暂停'}
            </h2>
          </div>
          <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
            {isBotRunning 
              ? '系统正在后台以 1分钟 频率高频评估行情，满足形态与突破信号时将直接提交订单至您的 Alpaca 账户。' 
              : '点击右侧按钮启动 AI 托管。开启后无需手动干预，全自动执行买卖与动态风控。'}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {!isBotRunning ? (
            <button 
              onClick={handleStartBot}
              disabled={actionLoading !== null}
              style={{
                background: 'var(--color-green)',
                color: '#000000',
                fontWeight: 900,
                fontSize: '1.05rem',
                padding: '12px 28px',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                boxShadow: '0 4px 20px rgba(0, 200, 5, 0.3)',
                transition: 'all 0.2s ease'
              }}
            >
              {actionLoading === "start" ? "⏳ 启动中..." : "▶️ 开启 AI 托管买卖"}
            </button>
          ) : (
            <button 
              onClick={handleStopBot}
              disabled={actionLoading !== null}
              style={{
                background: '#3a3a3c',
                color: '#ffffff',
                fontWeight: 800,
                fontSize: '1rem',
                padding: '12px 24px',
                borderRadius: '8px',
                border: '1px solid #48484a',
                cursor: 'pointer'
              }}
            >
              {actionLoading === "stop" ? "⏳ 停止中..." : "⏸️ 暂停 AI 托管"}
            </button>
          )}
        </div>
      </div>

      {/* 🤖 AI 操盘手模式切换选择器 (3款托管账号选项) */}
      <div className="card" style={{ marginBottom: '1.5rem', padding: '1.25rem 1.5rem', background: '#0c0c0e', border: '1px solid var(--color-border)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: '#ffffff' }}>
            🤖 AI 操盘手模式切换 (3款托管账号选项)
          </h3>
          <span style={{ fontSize: '0.8rem', color: 'var(--color-green)', fontWeight: 700 }}>
            当前生效: {currentMode ? `${currentMode.icon} ${currentMode.name}` : '加载中...'}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
          {modes.map((m) => {
            const isSelected = currentMode?.id === m.id;
            return (
              <div 
                key={m.id}
                onClick={() => handleSelectMode(m.id)}
                style={{
                  background: isSelected ? 'rgba(0, 200, 5, 0.08)' : '#141416',
                  border: isSelected ? '2px solid var(--color-green)' : '1px solid #27272a',
                  borderRadius: '10px',
                  padding: '1.1rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  position: 'relative'
                }}
              >
                {isSelected && (
                  <span style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    background: 'var(--color-green)',
                    color: '#000',
                    fontWeight: 900,
                    fontSize: '0.65rem',
                    padding: '2px 6px',
                    borderRadius: '4px'
                  }}>
                    ACTIVE 生效中
                  </span>
                )}
                <div style={{ fontSize: '1.3rem', marginBottom: '6px' }}>{m.icon}</div>
                <div style={{ fontWeight: 800, fontSize: '0.95rem', color: isSelected ? '#ffffff' : '#e5e5e7', marginBottom: '6px' }}>
                  {m.name}
                </div>
                <div style={{ fontSize: '0.78rem', color: '#8e8e93', lineHeight: 1.4, marginBottom: '10px' }}>
                  {m.description}
                </div>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', fontSize: '0.7rem' }}>
                  <span style={{ background: '#27272a', color: '#d4d4d8', padding: '2px 6px', borderRadius: '4px' }}>
                    K线: {m.bar_interval}
                  </span>
                  <span style={{ background: '#27272a', color: '#d4d4d8', padding: '2px 6px', borderRadius: '4px' }}>
                    过夜: {m.hold_overnight ? '允许' : '禁止过夜'}
                  </span>
                  {m.daily_target && (
                    <span style={{ background: 'rgba(0, 200, 5, 0.15)', color: 'var(--color-green)', padding: '2px 6px', borderRadius: '4px', fontWeight: 700 }}>
                      目标: ${m.daily_target}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 极简资产大字报 */}
      {account && (
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
          <div className="stat-card" style={{ background: '#09090b', border: '1px solid var(--color-border)', padding: '1.25rem' }}>
            <span className="stat-label">总资产净值 (Equity)</span>
            <span className="stat-value" style={{ fontSize: '1.5rem', fontWeight: 900 }}>
              ${account.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className="stat-card" style={{ background: '#09090b', border: '1px solid var(--color-border)', padding: '1.25rem' }}>
            <span className="stat-label">可用现金 (Cash)</span>
            <span className="stat-value" style={{ fontSize: '1.5rem', fontWeight: 900, color: 'var(--color-green)' }}>
              ${account.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className="stat-card" style={{ background: '#09090b', border: '1px solid var(--color-border)', padding: '1.25rem' }}>
            <span className="stat-label">持仓证券总额</span>
            <span className="stat-value" style={{ fontSize: '1.5rem', fontWeight: 900 }}>
              ${(account.equity - account.cash).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className="stat-card" style={{ background: '#09090b', border: '1px solid var(--color-border)', padding: '1.25rem' }}>
            <span className="stat-label">可用购买力 (4x Power)</span>
            <span className="stat-value" style={{ fontSize: '1.5rem', fontWeight: 900, color: '#e5e5e7' }}>
              ${account.buying_power.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
        </div>
      )}

      {/* 持仓卡片 + 交易动态卡片 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: '1.5rem' }}>
        
        {/* Alpaca 持仓大盘 */}
        <div className="card" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 800, color: '#ffffff' }}>
              📋 Alpaca 账户当前持仓 (Live Positions)
            </h3>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
              模拟盘账号: <strong>{account?.account_number}</strong>
            </span>
          </div>

          {positions.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--color-text-secondary)', padding: '3.5rem 0', fontSize: '0.85rem' }}>
              目前保持空仓防守中。当机器人检测到突破买入信号时，会自动建立仓位并在此展示。
            </div>
          ) : (
            <table className="ledger-table" style={{ fontSize: '0.85rem' }}>
              <thead>
                <tr>
                  <th>代码</th>
                  <th>持股数</th>
                  <th>建仓均价</th>
                  <th>最新现价</th>
                  <th style={{ textAlign: 'right' }}>浮动盈亏</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos) => {
                  const isUp = pos.unrealized_pnl >= 0;
                  return (
                    <tr key={pos.ticker}>
                      <td style={{ fontWeight: 900, color: '#fff' }}>{pos.ticker}</td>
                      <td>{pos.shares} 股</td>
                      <td>${pos.avg_entry_price.toFixed(2)}</td>
                      <td>${pos.current_price.toFixed(2)}</td>
                      <td style={{ textAlign: 'right', fontWeight: 800, color: isUp ? 'var(--color-green)' : 'var(--color-red)' }}>
                        {isUp ? '+' : ''}${pos.unrealized_pnl.toFixed(2)} ({isUp ? '+' : ''}{pos.unrealized_pnl_pct.toFixed(2)}%)
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* 极简最新交易买卖动态 (Feed) */}
        <div className="card" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 800, color: '#ffffff' }}>
              ⚡ 机器人最新买卖动态 (Action Feed)
            </h3>

            {actionFeeds.length === 0 ? (
              <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.85rem', padding: '2rem 0', textAlign: 'center' }}>
                启动 AI 托管后，最新的买卖通知卡片将在此实时呈现。
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {actionFeeds.map((feed, idx) => (
                  <div 
                    key={idx} 
                    style={{
                      background: '#141416',
                      border: feed.includes('买单') || feed.includes('买入') ? '1px solid rgba(0, 200, 5, 0.3)' : '1px solid #333',
                      borderRadius: '8px',
                      padding: '10px 14px',
                      fontSize: '0.8rem',
                      color: feed.includes('买单') || feed.includes('买入') ? 'var(--color-green)' : (feed.includes('卖') || feed.includes('平仓') ? 'var(--color-red)' : '#ffffff'),
                      lineHeight: 1.4
                    }}
                  >
                    {feed}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 底部应急操作按钮区 */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--color-border)' }}>
            <button 
              onClick={handleCancelAllOrders}
              disabled={actionLoading !== null}
              style={{
                background: 'rgba(255, 149, 0, 0.1)',
                border: '1px solid #ff9500',
                color: '#ff9500',
                padding: '8px 12px',
                borderRadius: '6px',
                fontSize: '0.8rem',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              📯 撤销所有挂单
            </button>
            <button 
              onClick={handleForceLiquidate}
              disabled={actionLoading !== null}
              style={{
                background: 'rgba(255, 59, 48, 0.15)',
                border: '1px solid var(--color-red)',
                color: 'var(--color-red)',
                padding: '8px 12px',
                borderRadius: '6px',
                fontSize: '0.8rem',
                fontWeight: 800,
                cursor: 'pointer'
              }}
            >
              🚨 一键紧急全平仓
            </button>
          </div>
        </div>

      </div>

      {/* 📜 AI 炒股大模型下单与交易历史记录 */}
      <div className="card" style={{ marginTop: '1.5rem', padding: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: '#ffffff' }}>
            📜 AI 下单与交易历史账单 (Alpaca Broker Orders Ledger)
          </h3>
          <span style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
            实时同步最近 {orders.length} 笔订单记录
          </span>
        </div>

        {orders.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--color-text-secondary)', padding: '2rem 0', fontSize: '0.85rem' }}>
            暂无历史订单记录。
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="ledger-table" style={{ fontSize: '0.85rem', width: '100%' }}>
              <thead>
                <tr>
                  <th>股票代码</th>
                  <th>买卖方向</th>
                  <th>委托数量</th>
                  <th>订单类型</th>
                  <th>成交均价</th>
                  <th>订单状态</th>
                  <th>提交时间 (UTC)</th>
                  <th>订单 ID</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((ord) => {
                  const isBuy = ord.side === 'BUY';
                  const isFilled = ord.status === 'FILLED';
                  const isAccepted = ord.status === 'ACCEPTED';
                  
                  return (
                    <tr key={ord.order_id}>
                      <td style={{ fontWeight: 900, color: '#ffffff' }}>{ord.symbol}</td>
                      <td>
                        <span style={{
                          padding: '3px 8px',
                          borderRadius: '4px',
                          fontWeight: 800,
                          fontSize: '0.75rem',
                          background: isBuy ? 'rgba(0, 200, 5, 0.15)' : 'rgba(255, 59, 48, 0.15)',
                          color: isBuy ? 'var(--color-green)' : 'var(--color-red)',
                          border: isBuy ? '1px solid rgba(0, 200, 5, 0.3)' : '1px solid rgba(255, 59, 48, 0.3)'
                        }}>
                          {isBuy ? '买入 BUY' : '卖出 SELL'}
                        </span>
                      </td>
                      <td style={{ fontWeight: 700 }}>{ord.qty} 股</td>
                      <td style={{ color: '#8e8e93' }}>{ord.type}</td>
                      <td style={{ fontWeight: 800 }}>
                        {ord.filled_avg_price > 0 ? `$${ord.filled_avg_price.toFixed(2)}` : '--'}
                      </td>
                      <td>
                        <span style={{
                          padding: '2px 6px',
                          borderRadius: '4px',
                          fontSize: '0.75rem',
                          fontWeight: 700,
                          background: isFilled ? '#1c3829' : (isAccepted ? '#3a2e16' : '#2c2c2e'),
                          color: isFilled ? 'var(--color-green)' : (isAccepted ? '#ff9500' : '#8e8e93')
                        }}>
                          {isFilled ? '✅ 已成交 (FILLED)' : (isAccepted ? '⏳ 已受理 (ACCEPTED)' : ord.status)}
                        </span>
                      </td>
                      <td style={{ color: '#8e8e93', fontSize: '0.78rem' }}>
                        {ord.submitted_at ? ord.submitted_at.replace('T', ' ').split('.')[0] : '--'}
                      </td>
                      <td style={{ color: '#636366', fontSize: '0.75rem', fontFamily: 'monospace' }}>
                        {ord.order_id.slice(0, 8)}...
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
