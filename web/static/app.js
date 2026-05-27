/* ─── Trade Agent Dashboard — Live WebSocket Client ────────────────────── */

(function () {
  "use strict";

  // ── Chart.js dark mode defaults ────────────────────────────────────────
  if (typeof Chart !== "undefined") {
    Chart.defaults.color = "#8b95a5";
    Chart.defaults.borderColor = "rgba(255,255,255,0.06)";
  }

  // ── DOM refs ───────────────────────────────────────────────────────────────
  // Header
  const $ = (id) => document.getElementById(id);
  const liveBadge = $("liveBadge");
  const timeDisplay = $("timeDisplay");
  const exchangeName = $("exchangeName");
  const paperMode = $("paperMode");
  const orderCountHeader = $("orderCount");
  const connectionStatus = $("connectionStatus");
  const lastUpdated = $("lastUpdated");

  // Portfolio
  const equityEl = $("equity");
  const cashEl = $("cash");
  const pnlEl = $("pnl");
  const drawdownEl = $("drawdown");
  const peakEquityEl = $("peakEquity");
  const killSwitchEl = $("killSwitch");

  // Market
  const pricesList = $("pricesList");

  // Agents
  const agentGrid = $("agentGrid");

  // Features
  const featuresGrid = $("featuresGrid");

  // Council
  const councilContent = $("councilContent");

  // Risk
  const riskGrid = $("riskGrid");
  const riskBars = $("riskBars");

  // Trade log
  const tradeLogBody = $("tradeLogBody");
  const orderCountBadge = $("orderCountBadge");

  // System log
  const logContainer = $("logContainer");
  const logCount = $("logCount");

  // Paper trading
  const paperCard = $("paperCard");
  const paperModeBadge = $("paperModeBadge");
  const paperEquity = $("paperEquity");
  const paperCash = $("paperCash");
  const paperTotalPnl = $("paperTotalPnl");
  const paperDailyPnl = $("paperDailyPnl");
  const paperWinRate = $("paperWinRate");
  const paperSharpe = $("paperSharpe");
  const paperDrawdown = $("paperDrawdown");
  const paperTrades = $("paperTrades");
  const paperPositions = $("paperPositions");
  const paperPositionCount = $("paperPositionCount");

  // Config
  const configGrid = $("configGrid");

  // Model Deployment
  const deploymentVersion = $("deploymentVersion");
  const deploymentSource = $("deploymentSource");
  const deploymentLoaded = $("deploymentLoaded");
  const deploymentRegistry = $("deploymentRegistry");

  // Sentinel-X
  const circuitBreakerEl = $("circuitBreaker");
  const rustKsStatus = $("rustKsStatus");
  const portfolioHeat = $("portfolioHeat");
  const heatFill = $("heatFill");

  // Ray Serve
  const serveUrl = $("serveUrl");
  const serveStatusDot = $("serveStatusDot");
  const serveStatusText = $("serveStatusText");
  const serveDeployments = $("serveDeployments");

  // ── Chart.js instances ────────────────────────────────────────────────────
  let equityChart = null;
  let priceChart = null;
  let latencyChart = null;
  let paperEquityChart = null;

  // ── Formatting helpers ─────────────────────────────────────────────────────
  function fmtCurrency(v) {
    if (v == null) return "$--";
    const abs = Math.abs(v);
    const sign = v < 0 ? "-" : "";
    if (abs >= 1_000_000) return sign + "$" + (abs / 1_000_000).toFixed(2) + "M";
    if (abs >= 1_000) return sign + "$" + (abs / 1_000).toFixed(1) + "K";
    return sign + "$" + abs.toFixed(2);
  }

  function fmtPrice(v) {
    if (v == null) return "--";
    if (v >= 1000) return v.toFixed(2);
    if (v >= 1) return v.toFixed(4);
    return v.toFixed(6);
  }

  function fmtPct(v) {
    if (v == null) return "--";
    return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
  }

  function fmtTime(iso) {
    if (!iso) return "--";
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString("en-US", { hour12: false });
    } catch {
      return iso;
    }
  }

  function updateClock() {
    if (timeDisplay) {
      timeDisplay.textContent = new Date().toLocaleTimeString("en-US", {
        hour12: false,
      });
    }
  }
  setInterval(updateClock, 1000);
  updateClock();

  // ── Helper: set metric value with flash animation ─────────────────────────
  function setMetric(el, value, className) {
    if (!el) return;
    const old = el.textContent;
    el.textContent = value;
    if (old !== value && className) {
      el.classList.remove("updating-positive", "updating-negative");
      // Force reflow
      void el.offsetWidth;
      el.classList.add(className);
    }
  }

  // ── Portfolio ──────────────────────────────────────────────────────────────
  function updatePortfolio(p) {
    if (!p) return;
    const pnl = p.daily_pnl || 0;
    const dd = p.drawdown_pct || 0;
    setMetric(equityEl, fmtCurrency(p.equity), pnl >= 0 ? "updating-positive" : "updating-negative");
    setMetric(cashEl, fmtCurrency(p.cash));
    setMetric(pnlEl, fmtCurrency(pnl), pnl >= 0 ? "updating-positive" : "updating-negative");
    setMetric(drawdownEl, fmtPct(-dd), dd > 0 ? "updating-negative" : "updating-positive");
    setMetric(peakEquityEl, fmtCurrency(p.peak_equity));
    if (killSwitchEl) {
      const active = p.kill_switch;
      killSwitchEl.textContent = active ? "● ON" : "● OFF";
      killSwitchEl.style.color = active ? "#ef4444" : "#22c55e";
    }
    if (exchangeName) {
      exchangeName.textContent = "Binance";
    }
    if (paperMode) {
      paperMode.textContent = p.kill_switch ? "HALT" : "ACTIVE";
      paperMode.className = "stat-mini-value badge " + (p.kill_switch ? "badge-live" : "badge-warn");
    }
  }

  // ── Market Prices ──────────────────────────────────────────────────────────
  let prevPrices = {};

  function updatePrices(prices) {
    if (!pricesList || !prices) return;
    const keys = Object.keys(prices);
    if (keys.length === 0) {
      pricesList.innerHTML = '<div class="placeholder">Waiting for market data...</div>';
      return;
    }
    pricesList.innerHTML = "";
    keys.forEach((sym) => {
      const price = prices[sym];
      const prev = prevPrices[sym];
      const change = prev && prev !== 0 ? ((price - prev) / prev) * 100 : 0;
      prevPrices[sym] = price;
      const item = document.createElement("div");
      item.className = "price-item";
      item.innerHTML =
        '<span class="price-symbol">' +
        sym +
        '</span><span class="price-value">' +
        fmtPrice(price) +
        '</span><span class="price-change' +
        (change >= 0 ? " positive" : " negative") +
        '">' +
        fmtPct(change) +
        "</span>";
      pricesList.appendChild(item);
    });
  }

  // ── Agent Signals ──────────────────────────────────────────────────────────
  function updateAgents(agents) {
    if (!agentGrid) return;
    const keys = Object.keys(agents || {});
    if (keys.length === 0) {
      agentGrid.innerHTML = '<div class="loading">Waiting for agent signals...</div>';
      return;
    }
    agentGrid.innerHTML = "";
    keys.forEach((name) => {
      const a = agents[name];
      const side = (a.side || "HOLD").toUpperCase();
      const card = document.createElement("div");
      card.className = "agent-card";
      card.innerHTML =
        '<div class="agent-card-header"><span class="agent-name">' +
        name +
        '</span><span class="agent-conf">' +
        (a.confidence ? (a.confidence * 100).toFixed(0) + "%" : "") +
        '</span></div><span class="agent-side ' +
        side +
        '">' +
        side +
        '</span><div class="agent-detail">' +
        (a.status || "") +
        "</div>";
      agentGrid.appendChild(card);
    });
  }

  // ── Feature Signals ────────────────────────────────────────────────────────
  function updateFeatures(features) {
    if (!featuresGrid) return;
    const keys = Object.keys(features || {});
    if (keys.length === 0) {
      featuresGrid.innerHTML = '<div class="placeholder">No feature data yet</div>';
      return;
    }
    featuresGrid.innerHTML = "";
    keys.forEach((sym) => {
      const f = features[sym];
      const trend = (f.trend || "HOLD").toUpperCase();
      const conf = f.confidence || 0;
      const metrics = [
        { label: "OFI", value: f.ofi, fmt: (v) => (v != null ? (v >= 0 ? "+" : "") + v.toFixed(3) : "N/A") },
        { label: "CVD", value: f.cvd, fmt: (v) => (v != null ? v.toFixed(0) : "N/A") },
        { label: "CVD Δ", value: f.cvd_delta, fmt: (v) => (v != null ? (v >= 0 ? "+" : "") + v.toFixed(2) : "N/A") },
        { label: "Funding", value: f.funding_rate, fmt: (v) => (v != null ? (v >= 0 ? "+" : "") + (v * 100).toFixed(4) + "%" : "N/A") },
        { label: "Funding Δ", value: f.funding_rate_delta, fmt: (v) => (v != null ? (v >= 0 ? "+" : "") + (v * 100).toFixed(4) + "%" : "N/A") },
        { label: "Open Interest", value: f.open_interest, fmt: (v) => (v != null ? v.toFixed(0) : "N/A") },
        { label: "OI Δ", value: f.open_interest_delta, fmt: (v) => (v != null ? (v >= 0 ? "+" : "") + v.toFixed(0) : "N/A") },
        { label: "OI-Price Corr", value: f.oi_price_delta_corr, fmt: (v) => (v != null ? (v >= 0 ? "+" : "") + v.toFixed(2) : "N/A") },
        { label: "Trade Strength", value: f.trade_strength, fmt: (v) => (v != null ? v.toFixed(2) : "N/A") },
      ];
      const card = document.createElement("div");
      card.className = "feature-card";
      const metricsHtml = metrics
        .map((m) => {
          const val = m.fmt(m.value);
          const isPos = typeof m.value === "number" && m.value > 0;
          const isNeg = typeof m.value === "number" && m.value < 0;
          return '<div class="feature-metric' +
            (isPos ? " positive" : isNeg ? " negative" : "") +
            '"><span class="feature-metric-label">' +
            m.label +
            '</span><span class="feature-metric-value">' +
            val +
            "</span></div>";
        })
        .join("");
      card.innerHTML =
        '<div class="feature-header"><span class="feature-symbol">' +
        sym +
        '</span><span class="feature-side agent-side ' +
        trend +
        '">' +
        trend +
        '</span><span class="feature-conf">' +
        (conf * 100).toFixed(0) +
        "%</span></div><div class=\"feature-metrics\">" +
        metricsHtml +
        "</div>";
      featuresGrid.appendChild(card);
    });
  }

  // ── Council Deliberation ───────────────────────────────────────────────────
  function updateCouncil(council) {
    if (!councilContent) return;
    const keys = Object.keys(council || {});
    if (keys.length === 0) {
      councilContent.innerHTML = '<div class="placeholder">Waiting for council decision...</div>';
      return;
    }
    councilContent.innerHTML = "";
    keys.forEach((sym) => {
      const c = council[sym];
      if (!c) return;
      const verdict = (c.final_side || "HOLD").toUpperCase();
      const bullPct = ((c.bull_score || 0) * 100).toFixed(0);
      const bearPct = ((c.bear_score || 0) * 100).toFixed(0);
      const consensus = ((c.consensus_score || 0) * 100).toFixed(0);
      const group = document.createElement("div");
      group.className = "council-symbol-group";

      let debateHtml = "";
      if (c.debate_log && c.debate_log.length > 0) {
        const cards = c.debate_log
          .map((d) => {
            const pos = (d.position || "HOLD").toUpperCase();
            const borderClass = "debate-" + pos.toLowerCase();
            const factors = (d.supporting || []).map((f) => '<span class="factor-chip factor-bull">' + f + "</span>").join(" ");
            const risks = (d.risks || []).map((r) => '<span class="factor-chip factor-bear">' + r + "</span>").join(" ");
            return (
              '<div class="debate-card ' +
              borderClass +
              '"><div class="debate-header"><span class="agent-side ' +
              pos +
              '">' +
              pos +
              '</span><span class="debate-agent">' +
              (d.agent || "?") +
              '</span><span class="debate-score">' +
              (d.score ? d.score.toFixed(2) : "") +
              '</span></div><div class="debate-argument">' +
              (d.argument || "") +
              '</div>' +
              (factors ? '<div class="debate-factors"><span class="debate-factors-label">↑</span>' + factors + "</div>" : "") +
              (risks ? '<div class="debate-factors"><span class="debate-factors-label">↓</span>' + risks + "</div>" : "") +
              "</div>"
            );
          })
          .join("");
        debateHtml = '<div class="debate-grid">' + cards + "</div>";
      }

      group.innerHTML =
        '<div class="council-header"><span class="council-symbol">' +
        sym +
        '</span><span class="agent-side ' +
        verdict +
        '">' +
        verdict +
        '</span><span class="council-consensus">Consensus: ' +
        consensus +
        "%</span></div><div class=\"council-scores\"><div class=\"score-bar\"><span class=\"score-bar-label\">Bull</span><div class=\"score-bar-track\"><div class=\"score-bar-fill score-bar-bull\" style=\"width:" +
        bullPct +
        '%\"></div></div><span class="score-bar-value">' +
        bullPct +
        '%</span></div><div class="score-bar"><span class="score-bar-label">Bear</span><div class="score-bar-track"><div class="score-bar-fill score-bar-bear" style="width:' +
        bearPct +
        '%\"></div></div><span class="score-bar-value">' +
        bearPct +
        '%</span></div></div>' +
        (c.rationale
          ? '<div class="council-rationale"><span class="rationale-label">Rationale:</span> ' + c.rationale + "</div>"
          : "") +
        debateHtml;
      councilContent.appendChild(group);
    });
  }

  // ── Risk Metrics ───────────────────────────────────────────────────────────
  function updateRisk(risk) {
    if (!riskGrid) return;
    const keys = Object.keys(risk || {});
    if (keys.length === 0) {
      riskGrid.innerHTML = '<div class="risk-placeholder">No risk data yet</div>';
      if (riskBars) riskBars.innerHTML = "";
      return;
    }
    riskGrid.innerHTML = "";
    keys.forEach((sym) => {
      const r = risk[sym];
      if (!r) return;
      const card = document.createElement("div");
      card.className = "risk-card";
      card.innerHTML =
        '<div class="risk-symbol">' +
        sym +
        '</div><div class="risk-metric-row"><span class="risk-metric-label">VaR 95%</span><span class="risk-metric-value">' +
        (r.var_95 != null ? (r.var_95 * 100).toFixed(2) + "%" : "N/A") +
        '</span></div><div class="risk-metric-row"><span class="risk-metric-label">VaR 99%</span><span class="risk-metric-value">' +
        (r.var_99 != null ? (r.var_99 * 100).toFixed(2) + "%" : "N/A") +
        '</span></div><div class="risk-metric-row"><span class="risk-metric-label">CVaR 99%</span><span class="risk-metric-value">' +
        (r.cvar_99 != null ? (r.cvar_99 * 100).toFixed(2) + "%" : "N/A") +
        '</span></div><div class="risk-metric-row"><span class="risk-metric-label">Kelly</span><span class="risk-metric-value">' +
        (r.kelly_fractional != null ? (r.kelly_fractional * 100).toFixed(1) + "%" : "N/A") +
        '</span></div><div class="risk-metric-row"><span class="risk-metric-label">Position</span><span class="risk-metric-value">' +
        fmtCurrency(r.position_size_usd) +
        '</span></div><div class="risk-metric-row"><span class="risk-metric-label">Stop Loss</span><span class="risk-metric-value">' +
        fmtPrice(r.stop_loss_price) +
        '</span></div><div class="risk-metric-row"><span class="risk-metric-label">Take Profit</span><span class="risk-metric-value">' +
        fmtPrice(r.take_profit_price) +
        "</span></div>";
      riskGrid.appendChild(card);

      // Risk bars (portfolio heat)
      if (riskBars) {
        riskBars.innerHTML = "";
        const heat = r.position_size_usd || 0;
        const maxHeat = 100000;
        const pct = Math.min((heat / maxHeat) * 100, 100);
        const bar = document.createElement("div");
        bar.className = "risk-bars-container";
        bar.innerHTML =
          '<div class="risk-bar-label">Portfolio Heat</div><div class="risk-bar-track"><div class="risk-bar-fill' +
          (pct > 70 ? " risk-heat-high" : pct > 40 ? " risk-heat-mid" : "") +
          '" style="width:' +
          pct +
          '%"></div></div>';
        riskBars.appendChild(bar);
      }
    });
  }

  // ── Trade Log ──────────────────────────────────────────────────────────────
  function updateTradeLog(orders) {
    if (!tradeLogBody) return;
    const arr = orders || [];
    if (arr.length === 0) {
      tradeLogBody.innerHTML = '<tr><td colspan="7" class="empty-row">No trades yet</td></tr>';
      if (orderCountBadge) orderCountBadge.textContent = "0";
      if (orderCountHeader) orderCountHeader.textContent = "0";
      return;
    }
    tradeLogBody.innerHTML = arr
      .slice(0, 50)
      .map((o) => {
        const side = (o.side || "HOLD").toUpperCase();
        const status = (o.status || "PENDING").toUpperCase();
        return (
          '<tr><td>' +
          fmtTime(o.created_at) +
          '</td><td>' +
          (o.symbol || "") +
          '</td><td class="trade-side-' +
          side +
          '">' +
          side +
          '</td><td>' +
          (o.quantity || 0) +
          '</td><td>' +
          fmtPrice(o.price) +
          '</td><td>' +
          (o.type || "") +
          '</td><td class="status-' +
          status +
          '">' +
          status +
          "</td></tr>"
        );
      })
      .join("");
    if (orderCountBadge) orderCountBadge.textContent = arr.length;
    if (orderCountHeader) orderCountHeader.textContent = arr.length;
  }

  // ── System Log ─────────────────────────────────────────────────────────────
  function updateLogs(logs) {
    if (!logContainer) return;
    const arr = logs || [];
    if (arr.length === 0) {
      logContainer.innerHTML = '<div class="log-placeholder">Waiting for log events...</div>';
      if (logCount) logCount.textContent = "0";
      return;
    }
    logContainer.innerHTML = arr
      .slice(-50)
      .reverse()
      .map((l) => {
        const level = (l.level || "INFO").toLowerCase();
        return (
          '<div class="log-entry' +
          (level === "error" ? " log-error" : "") +
          '"><span class="log-time">' +
          fmtTime(l.time) +
          '</span><span class="log-level log-level-' +
          level +
          '">' +
          (l.level || "INFO") +
          '</span><span class="log-msg">' +
          escapeHtml(l.msg || "") +
          "</span></div>"
        );
      })
      .join("");
    if (logCount) logCount.textContent = arr.length;
  }

  function escapeHtml(text) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  }

  // ── Paper Account ──────────────────────────────────────────────────────────
  function updatePaper(paper) {
    if (!paper || !paperCard) {
      if (paperCard) paperCard.style.display = "none";
      return;
    }
    paperCard.style.display = "";
    if (paperModeBadge) paperModeBadge.textContent = "PAPER";
    setMetric(paperEquity, fmtCurrency(paper.equity));
    setMetric(paperCash, fmtCurrency(paper.cash));
    setMetric(paperTotalPnl, fmtCurrency(paper.total_pnl), (paper.total_pnl || 0) >= 0 ? "updating-positive" : "updating-negative");
    setMetric(paperDailyPnl, fmtCurrency(paper.daily_pnl), (paper.daily_pnl || 0) >= 0 ? "updating-positive" : "updating-negative");
    setMetric(paperWinRate, paper.win_rate != null ? (paper.win_rate * 100).toFixed(1) + "%" : "0.0%");
    setMetric(paperSharpe, paper.sharpe != null ? paper.sharpe.toFixed(2) : "0.00");
    setMetric(paperDrawdown, paper.max_drawdown != null ? (paper.max_drawdown * 100).toFixed(2) + "%" : "0.00%");
    setMetric(paperTrades, paper.total_trades != null ? paper.total_trades : "0");

    // Positions
    const positions = paper.positions || [];
    if (!paperPositions) return;
    if (paperPositionCount) paperPositionCount.textContent = positions.length;
    if (positions.length === 0) {
      paperPositions.innerHTML = '<div class="placeholder">No open positions</div>';
      return;
    }
    paperPositions.innerHTML = positions
      .map(
        (p) =>
          '<div class="position-row"><span class="position-symbol">' +
          (p.symbol || "") +
          '</span><span class="position-qty">' +
          (p.quantity || 0) +
          '</span><span class="position-price">@ ' +
          fmtPrice(p.entry_price) +
          '</span><span class="position-value">' +
          fmtCurrency(p.current_value) +
          "</span></div>"
      )
      .join("");
  }

  // ── Config ─────────────────────────────────────────────────────────────────
  function updateConfig(cfg) {
    if (!configGrid || !cfg) return;
    const items = [
      { label: "Exchange", value: cfg.exchange || "--" },
      { label: "Symbols", value: (cfg.symbols || []).join(", ") },
      { label: "Paper Trading", value: cfg.paper_trading ? "Enabled" : "Disabled" },
      { label: "Capital", value: fmtCurrency(cfg.initial_capital) },
      { label: "Min Consensus", value: cfg.min_consensus_score != null ? (cfg.min_consensus_score * 100).toFixed(0) + "%" : "--" },
      { label: "Max Drawdown", value: cfg.max_daily_drawdown_pct != null ? (cfg.max_daily_drawdown_pct * 100).toFixed(1) + "%" : "--" },
      { label: "Kelly Fraction", value: cfg.kelly_fraction != null ? (cfg.kelly_fraction * 100).toFixed(0) + "%" : "--" },
    ];
    configGrid.innerHTML = items
      .map(
        (i) =>
          '<div class="config-item"><span class="config-item-label">' +
          i.label +
          '</span><span class="config-item-value">' +
          i.value +
          "</span></div>"
      )
      .join("");
  }

  // ── Model Deployment ──────────────────────────────────────────────────────
  function updateDeployment(dep) {
    if (!dep) return;
    if (deploymentVersion) deploymentVersion.textContent = dep.version || "--";
    if (deploymentSource) deploymentSource.textContent = dep.source || "--";
    if (deploymentLoaded) deploymentLoaded.textContent = dep.model_loaded ? "YES" : "NO";
    if (deploymentRegistry)
      deploymentRegistry.textContent =
        "v" + (dep.latest_registry_version || dep.local_version || "?");
    if (deploymentLoaded)
      deploymentLoaded.style.color = dep.model_loaded ? "#22c55e" : "#ef4444";
  }

  // ── Sentinel-X ────────────────────────────────────────────────────────────
  function updateSentinelX(sx) {
    if (!sx) return;
    if (circuitBreakerEl) {
      const cb = sx.circuit_breaker || "CLOSED";
      circuitBreakerEl.innerHTML =
        '<span class="serve-status-dot' +
        (cb === "CLOSED" ? " status-ok" : " status-error") +
        '"></span> ' +
        cb;
    }
    if (rustKsStatus) {
      const active = sx.rust_ks_active;
      rustKsStatus.innerHTML =
        '<span class="serve-status-dot' +
        (active ? " status-error" : " status-ok") +
        '"></span> ' +
        (active ? "ACTIVE" : "INACTIVE");
    }
    if (portfolioHeat && sx.rust_portfolio_heat != null) {
      const heat = sx.rust_portfolio_heat;
      const pct = Math.min(heat * 100, 100);
      portfolioHeat.innerHTML = '<span class="sentinelx-heat-value">' + (heat * 100).toFixed(1) + "%</span>";
      if (heatFill) {
        heatFill.style.width = pct + "%";
        heatFill.className = "sentinelx-heat-fill" + (pct > 70 ? " heat-high" : pct > 40 ? " heat-mid" : "");
      }
    }
  }

  // ── Ray Serve ─────────────────────────────────────────────────────────────
  function updateServeHealth(data) {
    if (!data) return;
    if (serveUrl) serveUrl.textContent = data.serve_url || "Not configured";
    const reachable = data.reachable;
    if (serveStatusDot) {
      serveStatusDot.className = "serve-status-dot" + (reachable ? " status-ok" : " status-error");
    }
    if (serveStatusText) {
      serveStatusText.textContent = reachable ? "Reachable" : "Unreachable";
      serveStatusText.className = "serve-status-text" + (reachable ? " status-ok" : " status-error");
    }
    if (serveDeployments) {
      const deps = data.deployments || {};
      const depNames = Object.keys(deps);
      if (depNames.length === 0) {
        serveDeployments.innerHTML = '<div class="placeholder">No deployments found</div>';
        return;
      }
      serveDeployments.innerHTML = depNames
        .map((name) => {
          const d = deps[name];
          const status = (d.status || "unknown").toLowerCase();
          const isOk = status === "ok" || status === "ready" || status === "healthy";
          return (
            '<div class="serve-deployment-card"><div class="serve-deployment-header"><span class="serve-deployment-name">' +
            name +
            '</span><span class="serve-deployment-status" style="color:' +
            (isOk ? "#22c55e" : "#ef4444") +
            '"><span class="serve-deployment-dot"></span>' +
            status +
            '</span></div>' +
            (d.version
              ? '<div class="serve-deployment-details"><span class="serve-deployment-version">v' +
                d.version +
                "</span></div>"
              : "") +
            (d.error
              ? '<div class="serve-deployment-error">' + escapeHtml(d.error) + "</div>"
              : "") +
            "</div>"
          );
        })
        .join("");
    }
  }

  // ── Charts ─────────────────────────────────────────────────────────────────
  function initEquityChart() {
    var canvas = document.getElementById("equityChart");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    equityChart = new Chart(ctx, {
      type: "line",
      data: { labels: [], datasets: [{ label: "Equity", data: [], borderColor: "#3b82f6", backgroundColor: "rgba(59, 130, 246, 0.08)", fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false, grid: { display: false } },
          y: { display: true, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#5a6474", font: { size: 10 }, callback: function (v) { return "$" + v.toFixed(0); } } },
        },
      },
    });
  }

  function initPriceChart() {
    var canvas = document.getElementById("priceChart");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    priceChart = new Chart(ctx, {
      type: "line",
      data: { labels: [], datasets: [{ label: "Price", data: [], borderColor: "#22c55e", backgroundColor: "rgba(34, 197, 94, 0.06)", fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5 }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false, grid: { display: false } },
          y: { display: true, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#5a6474", font: { size: 9 }, callback: function (v) { return "$" + v.toFixed(0); } } },
        },
      },
    });
  }

  function updateCharts(equityHistory) {
    if (!equityChart && equityHistory && equityHistory.length > 0) {
      initEquityChart();
    }
    if (equityChart && equityHistory) {
      var labels = equityHistory.map(function (e) { return e.t || ""; });
      var values = equityHistory.map(function (e) { return e.v || 0; });
      equityChart.data.labels = labels;
      equityChart.data.datasets[0].data = values;
      equityChart.update("none");
    }
  }

  function updatePriceChart(prices) {
    if (!prices) return;
    if (!priceChart) initPriceChart();
    if (!priceChart) return;
    var syms = Object.keys(prices);
    if (syms.length === 0) return;
    // Append data points over time (store in a closure)
    if (!window._priceHistory) window._priceHistory = {};
    syms.forEach(function (sym) {
      if (!window._priceHistory[sym]) window._priceHistory[sym] = [];
      window._priceHistory[sym].push(prices[sym]);
      if (window._priceHistory[sym].length > 100) window._priceHistory[sym].shift();
    });
    // Show first symbol's price
    var firstSym = syms[0];
    var data = window._priceHistory[firstSym] || [];
    priceChart.data.labels = data.map(function (_, i) { return i; });
    priceChart.data.datasets[0].data = data;
    priceChart.update("none");
  }

  // ── Main data handler ──────────────────────────────────────────────────────
  function handleData(data) {
    if (!data) return;
    updatePortfolio(data.portfolio);
    updatePrices(data.prices);
    updateAgents(data.agents);
    updateFeatures(data.features);
    updateCouncil(data.council);
    updateRisk(data.risk);
    updateTradeLog(data.orders);
    updateLogs(data.logs);
    updatePaper(data.paper);
    updateConfig(data.config);
    updateDeployment(data.deployment);
    updateSentinelX(data.sentinelx);
    updateCharts(data.equity_history);
    updatePriceChart(data.prices);
    if (lastUpdated) lastUpdated.textContent = "Updated: " + fmtTime(data.updated_at);
  }

  // ── Config fetch ───────────────────────────────────────────────────────────
  function fetchConfig() {
    fetch("/api/config")
      .then(function (r) { return r.json(); })
      .then(function (cfg) {
        updateConfig(cfg);
        window._config = cfg;
      })
      .catch(function () {});
  }
  fetchConfig();

  // ── Serve health refresh (initial) ─────────────────────────────────────────
  function fetchServeHealth() {
    fetch("/api/serve-status")
      .then(function (r) { return r.json(); })
      .then(function (data) { if (window.updateServeHealth) window.updateServeHealth(data); })
      .catch(function () {});
  }
  window.updateServeHealth = updateServeHealth;
  setTimeout(fetchServeHealth, 2000);

  // ── WebSocket connection ────────────────────────────────────────────────────
  var ws = null;
  var wsReconnectTimer = null;
  var wsPingTimer = null;

  function connectWS() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    var protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    var url = protocol + "//" + window.location.host + "/ws";

    if (connectionStatus) {
      connectionStatus.textContent = "● Connecting...";
      connectionStatus.className = "disconnected";
    }

    ws = new WebSocket(url);

    ws.onopen = function () {
      if (connectionStatus) {
        connectionStatus.textContent = "● Connected";
        connectionStatus.className = "connected";
      }
      if (liveBadge) liveBadge.textContent = "● Live";
      if (wsReconnectTimer) { clearTimeout(wsReconnectTimer); wsReconnectTimer = null; }
    };

    ws.onmessage = function (evt) {
      try {
        var data = JSON.parse(evt.data);
        handleData(data);
      } catch (e) {
        // Ignore parse errors
      }
    };

    ws.onclose = function () {
      if (connectionStatus) {
        connectionStatus.textContent = "● Disconnected";
        connectionStatus.className = "disconnected";
      }
      if (liveBadge) liveBadge.textContent = "● Reconnecting";
      ws = null;
      // Reconnect after 2s
      wsReconnectTimer = setTimeout(connectWS, 2000);
    };

    ws.onerror = function () {
      // onclose will fire after this
    };
  }

  // Connect on page load
  connectWS();

  // Ping every 30s to keep connection alive
  wsPingTimer = setInterval(function () {
    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('{"type":"ping"}');
      }
    } catch (e) {
      // WS may have closed between check and send
    }
  }, 30000);

  // Reconnect on visibility change (tab becomes active)
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWS();
      }
    }
  });

  console.log("Trade Agent Dashboard — WebSocket client loaded");
})();
