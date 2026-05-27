/* ─── Dashboard Application ────────────────────────────────────────────────── */

(function () {
  'use strict';

  // ── State ─────────────────────────────────────────────────────────────────
  let ws = null;
  let reconnectTimer = null;
  let failedPolls = 0;
  let equityChart = null;
  let priceChart = null;
  let equityHistory = [];
  let priceHistory = {};

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
  const orderCountBadge = $('orderCountBadge');
  const configGrid = $('configGrid');
  const connectionStatus = $('connectionStatus');
  const lastUpdated = $('lastUpdated');
  const exchangeName = $('exchangeName');
  const paperMode = $('paperMode');
  const equityCanvas = $('equityChart');
  const priceCanvas = $('priceChart');
  const featuresGrid = $('featuresGrid');
  const councilContent = $('councilContent');
  const logContainer = $('logContainer');
  const logCount = $('logCount');

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

  function fmtShortTime(iso) {
    if (!iso) return '--';
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  function fmtDate(iso) {
    if (!iso) return '--';
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  // ── Chart.js ──────────────────────────────────────────────────────────────

  function initEquityChart() {
    if (!equityCanvas || !window.Chart) return;
    const ctx = equityCanvas.getContext('2d');
    equityChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Equity',
          data: [],
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          pointHitRadius: 10,
          fill: true,
          tension: 0.3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1c1f26',
            titleColor: '#eaeef2',
            bodyColor: '#8b95a5',
            borderColor: '#2a2f3a',
            borderWidth: 1,
            cornerRadius: 6,
            padding: 10,
            callbacks: {
              label: function(ctx) { return '$' + ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 }); }
            }
          }
        },
        scales: {
          x: {
            display: false,
            grid: { display: false },
          },
          y: {
            display: true,
            grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
            ticks: {
              color: '#5a6474',
              font: { size: 10, family: 'JetBrains Mono' },
              callback: function(v) { return '$' + (v / 1000).toFixed(0) + 'k'; }
            }
          }
        },
        animation: { duration: 300 },
      }
    });
  }

  function initPriceChart() {
    if (!priceCanvas || !window.Chart) return;
    const ctx = priceCanvas.getContext('2d');
    priceChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [{
          label: 'Price',
          data: [],
          backgroundColor: [],
          borderColor: [],
          borderWidth: 1,
          borderRadius: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1c1f26',
            titleColor: '#eaeef2',
            bodyColor: '#8b95a5',
            borderColor: '#2a2f3a',
            borderWidth: 1,
            cornerRadius: 6,
            padding: 8,
            callbacks: {
              label: function(ctx) { return '$' + ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 }); }
            }
          }
        },
        scales: {
          x: { display: false, grid: { display: false } },
          y: {
            display: true,
            grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
            ticks: {
              color: '#5a6474',
              font: { size: 9, family: 'JetBrains Mono' },
              callback: function(v) { return '$' + (v / 1000).toFixed(1) + 'k'; }
            }
          }
        },
        animation: { duration: 300 },
      }
    });
  }

  function updateEquityChart(data) {
    if (!equityChart || !data || data.length < 2) return;
    const points = data.slice(-150);
    equityChart.data.labels = points.map(p => fmtDate(p.t));
    equityChart.data.datasets[0].data = points.map(p => p.v);
    equityChart.update('none');
  }

  function updatePriceChart(prices) {
    if (!priceChart || !prices || Object.keys(prices).length === 0) return;
    const entries = Object.entries(prices);
    const labels = entries.map(([s]) => s);
    const values = entries.map(([, v]) => v);
    // Compare with previous values for color
    const colors = values.map((v, i) => {
      const prev = priceHistory[labels[i]];
      if (prev !== undefined && v >= prev) return '#22c55e';
      if (prev !== undefined && v < prev) return '#ef4444';
      return '#3b82f6';
    });
    // Store for next comparison
    entries.forEach(([s, v]) => { priceHistory[s] = v; });

    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = values;
    priceChart.data.datasets[0].backgroundColor = colors;
    priceChart.data.datasets[0].borderColor = colors;
    priceChart.update('none');
  }

  // ── Update Portfolio ──────────────────────────────────────────────────────
  function updatePortfolio(p) {
    if (!p) return;
    equityEl.textContent = fmtUSD(p.equity);
    cashEl.textContent = fmtUSD(p.cash);
    pnlEl.textContent = (p.daily_pnl >= 0 ? '+' : '') + fmtUSD(p.daily_pnl);
    pnlEl.className = 'metric-value' + (p.daily_pnl >= 0 ? ' positive' : ' negative');
    drawdownEl.textContent = fmtPct(p.drawdown_pct);
    drawdownEl.className = 'metric-value' + (p.drawdown_pct >= 5 ? ' negative' : p.drawdown_pct > 0 ? ' status-warn' : '');
    peakEquityEl.textContent = fmtUSD(p.peak_equity);
    killSwitchEl.textContent = p.kill_switch ? '● ON' : '● OFF';
    killSwitchEl.className = 'metric-value' + (p.kill_switch ? ' negative' : ' positive');
  }

  // ── Update Market Prices ─────────────────────────────────────────────────
  function updatePrices(prices) {
    if (!prices || Object.keys(prices).length === 0) {
      pricesList.innerHTML = '<div class="placeholder">Waiting for market data...</div>';
      return;
    }
    // We need a simple price history for change calculation
    if (!window._priceHistory) window._priceHistory = {};
    const ph = window._priceHistory;

    pricesList.innerHTML = Object.entries(prices).map(([symbol, price]) => {
      const prev = ph[symbol] || price;
      const change = price - prev;
      const pct = prev > 0 ? (change / prev) * 100 : 0;
      const cls = change >= 0 ? 'positive' : 'negative';
      ph[symbol] = price;
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

  // ── Update Features ───────────────────────────────────────────────────────
  function updateFeatures(features) {
    if (!features || Object.keys(features).length === 0) {
      featuresGrid.innerHTML = '<div class="placeholder">No feature data yet</div>';
      return;
    }
    featuresGrid.innerHTML = Object.entries(features).map(([symbol, f]) => {
      const trendCls = (f.trend || 'HOLD').toUpperCase();
      const ofiCls = (f.ofi || 0) >= 0 ? 'positive' : 'negative';
      const cvdCls = (f.cvd_delta || 0) >= 0 ? 'positive' : 'negative';
      const frCls = (f.funding_rate || 0) >= 0 ? 'negative' : 'positive'; // neg funding = bullish
      return `
        <div class="feature-card">
          <div class="feature-header">
            <span class="feature-symbol">${symbol}</span>
            <span class="agent-side ${trendCls}">${trendCls}</span>
            <span class="feature-conf">${f.confidence ? (f.confidence * 100).toFixed(0) + '%' : '--'}</span>
          </div>
          <div class="feature-metrics">
            <div class="feature-metric ${ofiCls}">
              <span class="feature-metric-label">OFI</span>
              <span class="feature-metric-value">${f.ofi != null ? (f.ofi >= 0 ? '+' : '') + f.ofi.toFixed(3) : '--'}</span>
            </div>
            <div class="feature-metric ${cvdCls}">
              <span class="feature-metric-label">CVD Δ</span>
              <span class="feature-metric-value">${f.cvd_delta != null ? (f.cvd_delta >= 0 ? '+' : '') + f.cvd_delta.toFixed(1) : '--'}</span>
            </div>
            <div class="feature-metric ${frCls}">
              <span class="feature-metric-label">Funding</span>
              <span class="feature-metric-value">${f.funding_rate != null ? (f.funding_rate >= 0 ? '+' : '') + f.funding_rate.toFixed(4) + '%' : '--'}</span>
            </div>
            <div class="feature-metric">
              <span class="feature-metric-label">OI</span>
              <span class="feature-metric-value">${f.open_interest != null ? fmtNum(f.open_interest) : '--'}</span>
            </div>
            <div class="feature-metric">
              <span class="feature-metric-label">OI Δ</span>
              <span class="feature-metric-value">${f.open_interest_delta != null ? (f.open_interest_delta >= 0 ? '+' : '') + fmtNum(f.open_interest_delta) : '--'}</span>
            </div>
            <div class="feature-metric">
              <span class="feature-metric-label">OI-Price Corr</span>
              <span class="feature-metric-value">${f.oi_price_delta_corr != null ? (f.oi_price_delta_corr >= 0 ? '+' : '') + f.oi_price_delta_corr.toFixed(2) : '--'}</span>
            </div>
            <div class="feature-metric">
              <span class="feature-metric-label">Trade Strength</span>
              <span class="feature-metric-value">${f.trade_strength != null ? f.trade_strength.toFixed(2) : '--'}</span>
            </div>
          </div>
        </div>`;
    }).join('');
  }

  // ── Update Council ────────────────────────────────────────────────────────
  function updateCouncil(council) {
    if (!council || Object.keys(council).length === 0) {
      councilContent.innerHTML = '<div class="placeholder">Waiting for council decision...</div>';
      return;
    }
    councilContent.innerHTML = Object.entries(council).map(([symbol, c]) => {
      const sideCls = (c.final_side || 'HOLD').toUpperCase();
      const debateHtml = (c.debate_log || []).map(d => `
        <div class="debate-card debate-${d.position.toLowerCase()}">
          <div class="debate-header">
            <span class="debate-agent">${d.agent}</span>
            <span class="agent-side ${d.position}">${d.position}</span>
            <span class="debate-score">Score: ${d.score != null ? (d.score * 100).toFixed(0) + '%' : '--'}</span>
          </div>
          <p class="debate-argument">${d.argument || ''}</p>
          ${d.supporting && d.supporting.length > 0 ? `
            <div class="debate-factors">
              <span class="debate-factors-label">Supporting:</span>
              ${d.supporting.map(s => `<span class="factor-chip factor-bull">${s}</span>`).join('')}
            </div>` : ''}
          ${d.risks && d.risks.length > 0 ? `
            <div class="debate-factors">
              <span class="debate-factors-label">Risks:</span>
              ${d.risks.map(r => `<span class="factor-chip factor-bear">${r}</span>`).join('')}
            </div>` : ''}
        </div>
      `).join('');

      const consensusPct = c.consensus_score != null ? (c.consensus_score * 100).toFixed(1) : '--';
      return `
        <div class="council-symbol-group">
          <div class="council-header">
            <span class="council-symbol">${symbol}</span>
            <span class="council-verdict agent-side ${sideCls}">${sideCls}</span>
            <span class="council-consensus">Consensus: ${consensusPct}%</span>
          </div>
          <div class="council-scores">
            <div class="score-bar">
              <span class="score-bar-label">Bull</span>
              <div class="score-bar-track">
                <div class="score-bar-fill score-bar-bull" style="width:${(c.bull_score || 0) * 100}%"></div>
              </div>
              <span class="score-bar-value">${(c.bull_score || 0).toFixed(2)}</span>
            </div>
            <div class="score-bar">
              <span class="score-bar-label">Bear</span>
              <div class="score-bar-track">
                <div class="score-bar-fill score-bar-bear" style="width:${(c.bear_score || 0) * 100}%"></div>
              </div>
              <span class="score-bar-value">${(c.bear_score || 0).toFixed(2)}</span>
            </div>
          </div>
          ${c.rationale ? `<div class="council-rationale"><span class="rationale-label">PM Rationale:</span> ${c.rationale}</div>` : ''}
          <div class="debate-grid">
            ${debateHtml}
          </div>
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
        <div class="risk-metric-row">
          <span class="risk-metric-label">Position Size</span>
          <span class="risk-metric-value">${r.position_size_usd ? fmtUSD(r.position_size_usd) : '--'}</span>
        </div>
      </div>
    `).join('');

    // Risk bars — portfolio heat gauge
    const heatValues = Object.values(risk).map(r => r.var_99 || 0);
    const avgHeat = heatValues.length > 0 ? heatValues.reduce((a, b) => a + b, 0) / heatValues.length : 0;
    riskBars.innerHTML = `
      <div class="risk-bars-container">
        <div class="risk-bar-label">Portfolio Heat (Avg VaR99)</div>
        <div class="risk-bar-track">
          <div class="risk-bar-fill ${avgHeat > 0.1 ? 'risk-heat-high' : avgHeat > 0.05 ? 'risk-heat-mid' : ''}" style="width:${Math.min(avgHeat * 500, 100)}%"></div>
        </div>
      </div>`;
  }

  // ── Update Trades ─────────────────────────────────────────────────────────
  function updateTrades(orders) {
    if (!orders || orders.length === 0) {
      tradeLogBody.innerHTML = '<tr><td colspan="7" class="empty-row">No trades yet</td></tr>';
      if (orderCount) orderCount.textContent = '0';
      if (orderCountBadge) orderCountBadge.textContent = '0';
      return;
    }
    const count = orders.length;
    if (orderCount) orderCount.textContent = count;
    if (orderCountBadge) orderCountBadge.textContent = count;
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

  // ── Update Logs ───────────────────────────────────────────────────────────
  function updateLogs(logs) {
    if (!logs || logs.length === 0) {
      logContainer.innerHTML = '<div class="log-placeholder">Waiting for log events...</div>';
      if (logCount) logCount.textContent = '0';
      return;
    }
    if (logCount) logCount.textContent = logs.length;
    logContainer.innerHTML = logs.map(l => {
      const levelCls = (l.level || 'INFO').toLowerCase();
      return `<div class="log-entry log-${levelCls}">
        <span class="log-time">${fmtShortTime(l.time)}</span>
        <span class="log-level log-level-${levelCls}">${l.level || 'INFO'}</span>
        <span class="log-msg">${l.msg || ''}</span>
      </div>`;
    }).join('');
    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;
  }

  // ── Update Model Deployment ──────────────────────────────────────────────
  function updateDeployment(dep) {
    if (!dep) return;
    var verEl = $('deploymentVersion');
    var srcEl = $('deploymentSource');
    var loadedEl = $('deploymentLoaded');
    var regEl = $('deploymentRegistry');

    if (verEl) verEl.textContent = dep.version != null ? 'v' + dep.version : '--';
    if (srcEl) {
      srcEl.textContent = dep.source || 'unknown';
      srcEl.className = 'deployment-metric-value';
      if (dep.source === 'ray_serve') srcEl.classList.add('positive');
      else if (dep.source === 'local') srcEl.classList.add('status-warn');
      else if (dep.source === 'fallback') srcEl.classList.add('negative');
    }
    if (loadedEl) {
      loadedEl.textContent = dep.model_loaded ? '● Yes' : '● No';
      loadedEl.className = 'deployment-metric-value' + (dep.model_loaded ? ' positive' : ' negative');
    }
    if (regEl) regEl.textContent = dep.latest_registry_version != null ? 'v' + dep.latest_registry_version : '--';
  }

  // ── Update Serve Health ────────────────────────────────────────────────────
  function updateServeHealth(data) {
    if (!data) {
      // Try to fetch it
      fetchServeHealth();
      return;
    }
    var urlEl = $('serveUrl');
    var dotEl = $('serveStatusDot');
    var textEl = $('serveStatusText');
    var deploymentsEl = $('serveDeployments');

    if (urlEl) urlEl.textContent = data.serve_url || 'Not configured';

    if (!data.configured) {
      if (dotEl) { dotEl.className = 'serve-status-dot'; textEl.textContent = 'Not configured'; }
      if (deploymentsEl) deploymentsEl.innerHTML = '<div class="placeholder">Ray Serve is not configured. Set RAY_SERVE_URL to enable.</div>';
      return;
    }

    if (dotEl) {
      dotEl.className = 'serve-status-dot' + (data.reachable ? ' status-ok' : ' status-error');
    }
    if (textEl) {
      if (data.reachable) {
        textEl.textContent = 'Reachable';
        textEl.className = 'serve-status-text status-ok';
      } else {
        textEl.textContent = 'Unreachable' + (data.error ? ': ' + data.error : '');
        textEl.className = 'serve-status-text status-error';
      }
    }

    if (!deploymentsEl) return;
    var deps = data.deployments || {};
    if (Object.keys(deps).length === 0) {
      deploymentsEl.innerHTML = '<div class="placeholder">No deployments found</div>';
      return;
    }
    deploymentsEl.innerHTML = Object.entries(deps).map(function(_ref) {
      var name = _ref[0], status = _ref[1];
      var statusText = status.status || 'unknown';
      var statusCls = statusText === 'healthy' || statusText === 'ready' ? 'status-ok' :
                      statusText === 'error' ? 'status-error' :
                      statusText === 'unreachable' ? 'status-error' : 'status-warn';
      var versionText = status.version != null ? 'v' + status.version : '';
      var loadedText = status.model_loaded != null ? (status.model_loaded ? 'Loaded' : 'Unloaded') : '';
      return '<div class="serve-deployment-card">' +
        '<div class="serve-deployment-header">' +
          '<span class="serve-deployment-name">' + name.toUpperCase() + '</span>' +
          '<span class="serve-deployment-status ' + statusCls + '">' +
            '<span class="serve-deployment-dot"></span> ' + statusText +
          '</span>' +
        '</div>' +
        (versionText || loadedText ? '<div class="serve-deployment-details">' +
          (versionText ? '<span class="serve-deployment-version">' + versionText + '</span>' : '') +
          (loadedText ? '<span class="serve-deployment-version">' + loadedText + '</span>' : '') +
        '</div>' : '') +
        (status.error ? '<div class="serve-deployment-error">' + status.error + '</div>' : '') +
      '</div>';
    }).join('');
  }

  window._serveHealthFetching = false;

  async function fetchServeHealth() {
    if (window._serveHealthFetching) return;
    window._serveHealthFetching = true;
    try {
      var res = await fetch('/api/serve-status');
      if (res.ok) {
        var data = await res.json();
        updateServeHealth(data);
      }
    } catch (e) {}
    window._serveHealthFetching = false;
  }

  // ── Update PPO Latency Chart ──────────────────────────────────────────────
  var latencyChart = null;

  function updatePPOLatency(latencyData) {
    if (!latencyData || latencyData.length === 0) {
      return;
    }
    var canvas = $('latencyChart');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var points = latencyData.slice(-100);
    var labels = points.map(function(p) { return fmtShortTime(p.t); });
    var values = points.map(function(p) { return p.v; });

    if (latencyChart) {
      latencyChart.data.labels = labels;
      latencyChart.data.datasets[0].data = values;
      latencyChart.update('none');
      return;
    }

    if (!window.Chart) return;
    latencyChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Latency (ms)',
          data: values,
          borderColor: '#a855f7',
          backgroundColor: 'rgba(168, 85, 247, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          pointHitRadius: 8,
          fill: true,
          tension: 0.3,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1c1f26',
            titleColor: '#eaeef2',
            bodyColor: '#8b95a5',
            borderColor: '#2a2f3a',
            borderWidth: 1,
            cornerRadius: 6,
            padding: 8,
            callbacks: {
              label: function(ctx) { return ctx.parsed.y.toFixed(1) + ' ms'; }
            }
          }
        },
        scales: {
          x: { display: false, grid: { display: false } },
          y: {
            display: true,
            min: 0,
            grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
            ticks: {
              color: '#5a6474',
              font: { size: 9, family: 'JetBrains Mono' },
              callback: function(v) { return v.toFixed(0) + 'ms'; }
            }
          }
        },
        animation: { duration: 300 },
      }
    });
  }

  // ── Update Reload Events ────────────────────────────────────────────────
  function updateReloadEvents(events) {
    if (!events || events.length === 0) {
      var logEl = $('reloadLog');
      if (logEl) logEl.innerHTML = '<div class="log-placeholder">No reload events yet</div>';
      var countEl = $('reloadCount');
      if (countEl) countEl.textContent = '0';
      return;
    }
    var logEl = $('reloadLog');
    if (!logEl) return;
    var countEl = $('reloadCount');
    if (countEl) countEl.textContent = events.length;
    logEl.innerHTML = events.map(function(e) {
      var levelCls = (e.level || 'info').toLowerCase();
      var eventIcon = e.success ? '\u2713' : '\u2717';
      var eventCls = e.success ? 'reload-event-success' : 'reload-event-failure';
      if (e.event === 'model_loaded') eventIcon = '\u2B06';
      if (e.event === 'reload_started') eventIcon = '\u21BB';
      return '<div class="reload-event ' + eventCls + '">' +
        '<span class="reload-event-icon">' + eventIcon + '</span>' +
        '<span class="reload-event-time">' + fmtShortTime(e.timestamp) + '</span>' +
        '<span class="reload-event-msg">' + (e.msg || e.event || '') + '</span>' +
        '<span class="reload-event-version">' + (e.version != null ? 'v' + e.version : '') + '</span>' +
      '</div>';
    }).join('');
    logEl.scrollTop = logEl.scrollHeight;
  }

  // ── Paper Trading ──────────────────────────────────────────────────────────
  var paperEquityChart = null;

  function updatePaperTrading(data) {
    if (!data || data.mode !== 'paper') {
      var paperCard = document.getElementById('paperCard');
      if (paperCard) paperCard.style.display = 'none';
      return;
    }
    var paperCard = document.getElementById('paperCard');
    if (paperCard) paperCard.style.display = '';

    var equityEl = $('paperEquity');
    var cashEl = $('paperCash');
    var totalPnlEl = $('paperTotalPnl');
    var dailyPnlEl = $('paperDailyPnl');
    var winRateEl = $('paperWinRate');
    var sharpeEl = $('paperSharpe');
    var drawdownEl = $('paperDrawdown');
    var tradesEl = $('paperTrades');
    var positionsEl = $('paperPositions');
    var posCountEl = $('paperPositionCount');

    if (equityEl) equityEl.textContent = fmtUSD(data.equity);
    if (cashEl) cashEl.textContent = fmtUSD(data.cash);
    if (totalPnlEl) {
      totalPnlEl.textContent = (data.total_pnl >= 0 ? '+' : '') + fmtUSD(data.total_pnl);
      totalPnlEl.className = 'metric-value' + (data.total_pnl >= 0 ? ' positive' : ' negative');
    }
    if (dailyPnlEl) {
      dailyPnlEl.textContent = (data.daily_pnl >= 0 ? '+' : '') + fmtUSD(data.daily_pnl);
      dailyPnlEl.className = 'metric-value' + (data.daily_pnl >= 0 ? ' positive' : ' negative');
    }
    if (winRateEl) winRateEl.textContent = (data.win_rate * 100).toFixed(1) + '%';
    if (sharpeEl) sharpeEl.textContent = data.sharpe_ratio != null ? data.sharpe_ratio.toFixed(2) : '0.00';
    if (drawdownEl) {
      drawdownEl.textContent = data.drawdown_pct != null ? data.drawdown_pct.toFixed(2) + '%' : '0.00%';
      drawdownEl.className = 'metric-value' + (data.drawdown_pct > 5 ? ' negative' : data.drawdown_pct > 0 ? ' status-warn' : '');
    }
    if (tradesEl) tradesEl.textContent = data.total_trades || 0;

    // Open positions
    if (posCountEl) posCountEl.textContent = (data.open_positions || []).length;
    if (positionsEl) {
      var positions = data.open_positions || [];
      if (positions.length === 0) {
        positionsEl.innerHTML = '<div class="placeholder">No open positions</div>';
      } else {
        positionsEl.innerHTML = positions.map(function(p) {
          var sideCls = p.side || 'HOLD';
          var value = (p.quantity * p.entry_price).toFixed(2);
          return '<div class="position-row">' +
            '<span class="position-symbol">' + p.symbol + '</span>' +
            '<span class="agent-side ' + sideCls + '">' + sideCls + '</span>' +
            '<span class="position-qty">' + fmtNum(p.quantity) + '</span>' +
            '<span class="position-price">' + fmtUSD(p.entry_price) + '</span>' +
            '<span class="position-value">' + fmtUSD(parseFloat(value)) + '</span>' +
          '</div>';
        }).join('');
      }
    }

    // Paper equity mini-chart
    var eqHistory = data.equity_history || [];
    if (eqHistory.length >= 2) {
      var canvas = $('paperEquityChart');
      if (canvas && window.Chart) {
        var ctx = canvas.getContext('2d');
        var labels = eqHistory.map(function(p) { return fmtShortTime(p.t); });
        var values = eqHistory.map(function(p) { return p.v; });
        if (paperEquityChart) {
          paperEquityChart.data.labels = labels;
          paperEquityChart.data.datasets[0].data = values;
          paperEquityChart.update('none');
        } else {
          paperEquityChart = new Chart(ctx, {
            type: 'line',
            data: {
              labels: labels,
              datasets: [{
                label: 'Paper Equity',
                data: values,
                borderColor: '#eab308',
                backgroundColor: 'rgba(234, 179, 8, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                pointHitRadius: 8,
                fill: true,
                tension: 0.3,
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { display: false },
                tooltip: {
                  backgroundColor: '#1c1f26',
                  titleColor: '#eaeef2',
                  bodyColor: '#8b95a5',
                  borderColor: '#2a2f3a',
                  borderWidth: 1,
                  cornerRadius: 6,
                  padding: 8,
                  callbacks: {
                    label: function(ctx) { return '$' + ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 }); }
                  }
                }
              },
              scales: {
                x: { display: false, grid: { display: false } },
                y: {
                  display: true,
                  grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                  ticks: {
                    color: '#5a6474',
                    font: { size: 9, family: 'JetBrains Mono' },
                    callback: function(v) { return '$' + (v / 1000).toFixed(0) + 'k'; }
                  }
                }
              },
              animation: { duration: 300 },
            }
          });
        }
      }
    }
  }

  // ── Update Sentinel-X Status ──────────────────────────────────────────────────────
  function updateSentinelX(data) {
    if (!data) return;

    // Circuit breaker
    var cbEl = $('circuitBreaker');
    if (cbEl) {
      var state = data.circuit_breaker || 'CLOSED';
      var isOpen = state === 'OPEN';
      cbEl.innerHTML = '<span class="serve-status-dot' + (isOpen ? ' status-error' : ' status-ok') + '"></span> ' + state;
      cbEl.className = 'sentinelx-metric-value' + (isOpen ? ' negative' : ' positive');
    }

    // Rust kill switch
    var ksEl = $('rustKsStatus');
    if (ksEl) {
      var active = data.rust_ks_active || false;
      ksEl.innerHTML = '<span class="serve-status-dot' + (active ? ' status-error' : ' status-ok') + '"></span> ' + (active ? 'ACTIVE' : 'INACTIVE');
      ksEl.className = 'sentinelx-metric-value' + (active ? ' negative' : ' positive');
    }

    // Portfolio heat
    var heatEl = $('portfolioHeat');
    var heatFill = $('heatFill');
    if (heatEl && heatFill) {
      var heat = data.rust_portfolio_heat;
      if (heat != null) {
        heatEl.innerHTML = '<span class="sentinelx-heat-value">' + heat.toFixed(4) + '</span>';
        var pct = Math.min(heat * 100, 100);
        heatFill.style.width = pct + '%';
        var cls = 'sentinelx-heat-fill';
        if (heat > 0.1) cls += ' heat-high';
        else if (heat > 0.05) cls += ' heat-mid';
        heatFill.className = cls;
      } else {
        heatEl.innerHTML = '<span class="sentinelx-heat-value">--</span>';
        heatFill.style.width = '0%';
        heatFill.className = 'sentinelx-heat-fill';
      }
    }
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
    updateFeatures(data.features);
    updateCouncil(data.council);
    updateRisk(data.risk);
    updateTrades(data.orders);
    updateLogs(data.logs);
    updateEquityChart(data.equity_history);
    updatePriceChart(data.prices);
    updatePaperTrading(data.paper);
    updateDeployment(data.deployment);
    updateServeHealth(data.serve_health);
    updatePPOLatency(data.ppo_latency);
    updateReloadEvents(data.reload_events);
    updateSentinelX(data.sentinelx);
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
    // Initialize Chart.js charts
    if (window.Chart) {
      initEquityChart();
      initPriceChart();
    }

    loadConfig();

    // Initial fetch of serve health (separate from snapshot)
    fetchServeHealth();

    // Try WebSocket first, fall back to REST polling
    if (window.WebSocket) {
      connectWS();
      // Also poll REST as backup
      pollTimer = setTimeout(pollREST, 5000);
    } else {
      pollREST();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
