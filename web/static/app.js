/* ─── Dashboard Application ────────────────────────────────────────────────── */

(function () {
  'use strict';

  // ── State ─────────────────────────────────────────────────────────────────
  let equityHistory = [];
  let ws = null;
  let reconnectTimer = null;
  let failedPolls = 0;
  const MAX_POINTS = 200;

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const equityEl = $('equity');
  const cashEl = $('cash');
  const pnlEl = $('pnl');
  const drawdownEl = $('drawdown');
  const peakEquityEl = $('peakEquity');
  const killSwitchEl = $('killSwitch');
  const timeDisplay = $('timeDisplay');
  const pricesList = $('pricesList');
  const agentGrid = $('agentGrid');
  const riskGrid = $('riskGrid');
  const riskBars = $('riskBars');
  const tradeLogBody = $('tradeLogBody');
  const orderCount = $('orderCount');
  const configGrid = $('configGrid');
  const connectionStatus = $('connectionStatus');
  const lastUpdated = $('lastUpdated');
  const exchangeName = $('exchangeName');
  const paperMode = $('paperMode');
  const equityCanvas = $('equityChart');

  // ── Clock ─────────────────────────────────────────────────────────────────
  function updateClock() {
    const now = new Date();
    timeDisplay.textContent = now.toLocaleTimeString('en-US', { hour12: false });
  }
  setInterval(updateClock, 1000);
  updateClock();

  // ── Formatting helpers ────────────────────────────────────────────────────
  function fmtUSD(v) {
    if (v == null || isNaN(v)) return '$0.00';
    const sign = v < 0 ? '-' : '';
    return sign + '$' + Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtPct(v) {
    if (v == null || isNaN(v)) return '0.00%';
    const sign = v < 0 ? '-' : '';
    return sign + Math.abs(v).toFixed(2) + '%';
  }

  function fmtNum(v) {
    if (v == null || isNaN(v)) return '0';
    return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
  }

  function fmtTime(iso) {
    if (!iso) return '--';
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', { hour12: false });
  }

  // ── Canvas / Sparkline ────────────────────────────────────────────────────
  let sparkCtx = null;
  let animFrame = null;

  function drawSparkline() {
    if (!equityCanvas) return;
    if (!sparkCtx) sparkCtx = equityCanvas.getContext('2d');
    const ctx = sparkCtx;
    const w = equityCanvas.clientWidth || equityCanvas.parentElement.clientWidth || 600;
    const h = equityCanvas.clientHeight || 80;
    equityCanvas.width = w * 2;
    equityCanvas.height = h * 2;
    equityCanvas.style.width = w + 'px';
    equityCanvas.style.height = h + 'px';
    ctx.scale(2, 2);

    ctx.clearRect(0, 0, w, h);

    if (equityHistory.length < 2) {
      ctx.fillStyle = '#5a6474';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for data...', w / 2, h / 2 + 4);
      return;
    }

    const data = equityHistory.slice(-MAX_POINTS);
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const pad = 4;
    const plotH = h - pad * 2;

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, pad, 0, h - pad);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
    gradient.addColorStop(0.5, 'rgba(59, 130, 246, 0.08)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

    // Path
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = pad + plotH - ((v - min) / range) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.lineTo(w, h - pad);
    ctx.lineTo(0, h - pad);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = pad + plotH - ((v - min) / range) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Current value label
    const last = data[data.length - 1];
    ctx.fillStyle = '#8b95a5';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('$' + last.toLocaleString('en-US', { minimumFractionDigits: 0 }), w - 4, h - 2);
  }

  // ── Update Portfolio ──────────────────────────────────────────────────────
  function updatePortfolio(p) {
    if (!p) return;
    equityEl.textContent = fmtUSD(p.equity);
    cashEl.textContent = fmtUSD(p.cash);
    pnlEl.textContent = fmtUSD(p.daily_pnl);
    pnlEl.className = 'metric-value' + (p.daily_pnl >= 0 ? ' positive' : ' negative');
    drawdownEl.textContent = fmtPct(p.drawdown_pct);
    drawdownEl.className = 'metric-value' + (p.drawdown_pct >= 5 ? ' negative' : p.drawdown_pct > 0 ? ' status-warn' : '');
    peakEquityEl.textContent = fmtUSD(p.peak_equity);
    killSwitchEl.textContent = p.kill_switch ? '● ON' : '● OFF';
    killSwitchEl.className = 'metric-value' + (p.kill_switch ? ' negative' : ' positive');

    // Track equity history
    if (p.equity != null && !isNaN(p.equity)) {
      equityHistory.push(p.equity);
      if (equityHistory.length > MAX_POINTS) equityHistory.shift();
      if (animFrame) cancelAnimationFrame(animFrame);
      animFrame = requestAnimationFrame(drawSparkline);
    }
  }

  // ── Update Market Prices ─────────────────────────────────────────────────
  function updatePrices(prices) {
    if (!prices || Object.keys(prices).length === 0) {
      pricesList.innerHTML = '<div class="placeholder">Waiting for market data...</div>';
      return;
    }
    pricesList.innerHTML = Object.entries(prices).map(([symbol, price]) => {
      const prev = equityHistory.length > 0 ? equityHistory[equityHistory.length - 1] : price;
      const change = price - prev;
      const pct = prev > 0 ? (change / prev) * 100 : 0;
      const cls = change >= 0 ? 'positive' : 'negative';
      return `
        <div class="price-item">
          <span class="price-symbol">${symbol}</span>
          <span>
            <span class="price-value ${cls}">${fmtUSD(price)}</span>
            <span class="price-change ${cls}">${change >= 0 ? '+' : ''}${fmtPct(pct)}</span>
          </span>
        </div>`;
    }).join('');
  }

  // ── Update Agents ─────────────────────────────────────────────────────────
  function updateAgents(agents) {
    if (!agents || Object.keys(agents).length === 0) {
      agentGrid.innerHTML = '<div class="loading">Waiting for agent signals...</div>';
      return;
    }
    const agentLabels = {
      technical: 'Technical',
      fundamental: 'Fundamental',
      sentiment: 'Sentiment',
      council: 'Council',
      risk: 'Risk Engine',
      execution: 'Execution',
      supervisor: 'Supervisor'
    };
    agentGrid.innerHTML = Object.entries(agents).map(([name, data]) => {
      const label = agentLabels[name] || name;
      const side = (data.side || 'HOLD').toUpperCase();
      const conf = data.confidence != null ? (data.confidence * 100).toFixed(0) + '%' : '--';
      const detail = data.detail || data.rationale || data.status || '';
      return `
        <div class="agent-card">
          <div class="agent-card-header">
            <span class="agent-name">${label}</span>
            <span class="agent-conf">${conf}</span>
          </div>
          <span class="agent-side ${side}">${side}</span>
          ${detail ? `<span class="agent-detail">${detail}</span>` : ''}
        </div>`;
    }).join('');
  }

  // ── Update Risk ───────────────────────────────────────────────────────────
  function updateRisk(risk) {
    if (!risk || Object.keys(risk).length === 0) {
      riskGrid.innerHTML = '<div class="risk-placeholder">No risk data yet</div>';
      riskBars.innerHTML = '';
      return;
    }

    const colors = ['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#a855f7'];

    riskGrid.innerHTML = Object.entries(risk).map(([symbol, r]) => `
      <div class="risk-card">
        <div class="risk-symbol">${symbol}</div>
        <div class="risk-metric-row">
          <span class="risk-metric-label">VaR 99%</span>
          <span class="risk-metric-value">${fmtPct((r.var_99 || 0) * 100)}</span>
        </div>
        <div class="risk-metric-row">
          <span class="risk-metric-label">VaR 95%</span>
          <span class="risk-metric-value">${fmtPct((r.var_95 || 0) * 100)}</span>
        </div>
        <div class="risk-metric-row">
          <span class="risk-metric-label">CVaR 99%</span>
          <span class="risk-metric-value">${fmtPct((r.cvar_99 || 0) * 100)}</span>
        </div>
        <div class="risk-metric-row">
          <span class="risk-metric-label">Kelly</span>
          <span class="risk-metric-value">${r.kelly_fractional != null ? fmtPct(r.kelly_fractional * 100) : '--'}</span>
        </div>
        <div class="risk-metric-row">
          <span class="risk-metric-label">Stop Loss</span>
          <span class="risk-metric-value">${fmtUSD(r.stop_loss_price)}</span>
        </div>
        <div class="risk-metric-row">
          <span class="risk-metric-label">Take Profit</span>
          <span class="risk-metric-value">${fmtUSD(r.take_profit_price)}</span>
        </div>
      </div>
    `).join('');

    // Risk bars
    const entries = Object.entries(risk);
    riskBars.innerHTML = `<div class="risk-bars-container">
      <div class="risk-bar-label">Portfolio Heat</div>
      <div class="risk-bar-track">
        <div class="risk-bar-fill" style="width:${Math.min((entries[0]?.[1]?.portfolio_heat || 0) * 100, 100)}%; background: ${colors[0]}"></div>
      </div>
    </div>`;
  }

  // ── Update Trades ─────────────────────────────────────────────────────────
  function updateTrades(orders) {
    if (!orders || orders.length === 0) {
      tradeLogBody.innerHTML = '<tr><td colspan="7" class="empty-row">No trades yet</td></tr>';
      orderCount.textContent = '0';
      return;
    }
    orderCount.textContent = orders.length;
    tradeLogBody.innerHTML = orders.map(o => `
      <tr>
        <td>${fmtTime(o.created_at)}</td>
        <td><strong>${o.symbol || '--'}</strong></td>
        <td class="trade-side-${o.side}">${o.side || '--'}</td>
        <td>${fmtNum(o.quantity)}</td>
        <td>${fmtUSD(o.price)}</td>
        <td>${o.type || 'MARKET'}</td>
        <td class="status-${o.status}">${o.status || '--'}</td>
      </tr>
    `).join('');
  }

  // ── Update Config ─────────────────────────────────────────────────────────
  function updateConfig(cfg) {
    if (!cfg) return;
    exchangeName.textContent = cfg.exchange || '--';
    paperMode.textContent = cfg.paper_trading ? 'ON' : 'OFF';

    const items = [
      ['Symbols', (cfg.symbols || []).join(', ')],
      ['Min Consensus', fmtPct((cfg.min_consensus_score || 0) * 100)],
      ['Max Drawdown', fmtPct((cfg.max_daily_drawdown_pct || 0) * 100)],
      ['Kelly Fraction', fmtPct((cfg.kelly_fraction || 0) * 100)],
      ['Capital', fmtUSD(cfg.initial_capital)],
    ];

    configGrid.innerHTML = items.map(([label, value]) => `
      <div class="config-item">
        <span class="config-item-label">${label}</span>
        <span class="config-item-value">${value}</span>
      </div>
    `).join('');
  }

  // ── Load Config ──────────────────────────────────────────────────────────
  async function loadConfig() {
    try {
      const res = await fetch('/api/config');
      if (res.ok) updateConfig(await res.json());
    } catch (e) {
      console.warn('Failed to load config:', e);
    }
  }

  // ── Process Snapshot ─────────────────────────────────────────────────────
  function processSnapshot(data) {
    if (!data) return;
    updatePortfolio(data.portfolio);
    updatePrices(data.prices);
    updateAgents(data.agents);
    updateRisk(data.risk);
    updateTrades(data.orders);
    lastUpdated.textContent = 'Updated ' + fmtTime(data.updated_at);
  }

  // ── WebSocket ────────────────────────────────────────────────────────────
  function connectWS() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = proto + '//' + window.location.host + '/ws';

    ws = new WebSocket(url);

    ws.onopen = () => {
      connectionStatus.textContent = '● Connected';
      connectionStatus.className = 'connected';
      failedPolls = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        processSnapshot(data);
      } catch (e) {
        console.warn('WS parse error:', e);
      }
    };

    ws.onclose = () => {
      connectionStatus.textContent = '● Disconnected (polling)';
      connectionStatus.className = 'disconnected';
      ws = null;
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws && ws.close();
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connectWS, 3000);
  }

  // ── REST Fallback Polling ────────────────────────────────────────────────
  let pollTimer = null;

  async function pollREST() {
    try {
      const res = await fetch('/api/snapshot');
      if (res.ok) {
        const data = await res.json();
        processSnapshot(data);
        failedPolls = 0;
      } else {
        throw new Error('HTTP ' + res.status);
      }
    } catch (e) {
      failedPolls++;
      if (failedPolls > 5) {
        connectionStatus.textContent = '● Server unreachable';
        connectionStatus.className = 'disconnected';
      }
    }
    pollTimer = setTimeout(pollREST, 3000);
  }

  // ── Init ────────────────────────────────────────────────────────────────
  function init() {
    loadConfig();
    // Try WebSocket first, fall back to REST polling
    if (window.WebSocket) {
      connectWS();
      // Also poll REST as backup
      pollTimer = setTimeout(pollREST, 5000);
    } else {
      pollREST();
    }

    // Re-draw sparkline on resize
    window.addEventListener('resize', () => {
      if (animFrame) cancelAnimationFrame(animFrame);
      animFrame = requestAnimationFrame(drawSparkline);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
