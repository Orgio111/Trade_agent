/* ─── Trading Dashboard Data Types ──────────────────────────────────────── */

export interface PortfolioData {
  equity: number;
  cash: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  drawdown_pct: number;
  peak_equity: number;
  kill_switch: boolean;
  positions: Record<string, unknown>;
}

export interface OrderData {
  order_id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  status: string;
  created_at: string;
  type: string;
}

export interface AgentSignal {
  side: string;
  confidence: number;
  status?: string;
  timestamp: string;
  [key: string]: unknown;
}

export interface RiskMetric {
  var_95?: number;
  var_99?: number;
  cvar_99?: number;
  kelly_fractional?: number;
  position_size_usd?: number;
  stop_loss_price?: number;
  take_profit_price?: number;
  timestamp: string;
}

export interface FeatureSignal {
  ofi?: number;
  cvd?: number;
  cvd_delta?: number;
  funding_rate?: number;
  funding_rate_delta?: number;
  open_interest?: number;
  open_interest_delta?: number;
  oi_price_delta_corr?: number;
  trade_strength?: number;
  trend: string;
  confidence: number;
  timestamp: string;
}

export interface DebateEntry {
  agent: string;
  position: string;
  argument: string;
  score: number;
  supporting: string[];
  risks: string[];
}

export interface CouncilDecision {
  session_id: string;
  bull_score: number;
  bear_score: number;
  consensus_score: number;
  final_side: string;
  rationale: string;
  debate_log: DebateEntry[];
  timestamp: string;
}

export interface PaperState {
  equity: number;
  cash: number;
  total_pnl: number;
  daily_pnl: number;
  win_rate: number;
  sharpe: number;
  max_drawdown: number;
  total_trades: number;
  positions: PaperPosition[];
}

export interface PaperPosition {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_value: number;
}

export interface ModelDeployment {
  version: number;
  source: string;
  local_version: number;
  latest_registry_version: number;
  model_loaded: boolean;
}

export interface SentinelXState {
  circuit_breaker: string;
  rust_ks_active: boolean;
  rust_portfolio_heat: number | null;
}

export interface ServeHealth {
  configured: boolean;
  serve_url: string;
  reachable: boolean;
  deployments: Record<string, { status: string; version?: string; error?: string }>;
}

export interface AppConfig {
  exchange: string;
  exchanges: string[];
  symbols: string[];
  paper_trading: boolean;
  initial_capital: number;
  min_consensus_score: number;
  max_daily_drawdown_pct: number;
  kelly_fraction: number;
}

export interface DashboardSnapshot {
  portfolio: PortfolioData;
  orders: OrderData[];
  agents: Record<string, AgentSignal>;
  risk: Record<string, RiskMetric>;
  prices: Record<string, number>;
  features: Record<string, FeatureSignal>;
  council: Record<string, CouncilDecision>;
  equity_history: EquityPoint[];
  paper: PaperState | null;
  deployment: ModelDeployment;
  serve_health: ServeHealth | null;
  deployment_health: Record<string, unknown>;
  ppo_latency: LatencyPoint[];
  reload_events: ReloadEvent[];
  sentinelx: SentinelXState;
  logs: LogEntry[];
  updated_at: string;
}

export interface EquityPoint {
  t: string;
  v: number;
}

export interface LatencyPoint {
  t: string;
  v: number;
}

export interface ReloadEvent {
  timestamp: string;
  [key: string]: unknown;
}

export interface LogEntry {
  time: string;
  level: string;
  msg: string;
}
