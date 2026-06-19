trading-ai/
│
├── README.md
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── pyproject.toml
│
├── config/
│   ├── settings.yaml
│   ├── exchanges.yaml
│   ├── strategies.yaml
│   └── risk.yaml
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── features/
│   ├── datasets/
│   └── parquet/
│
├── storage/
│   ├── duckdb/
│   ├── postgres/
│   └── backups/
│
├── ingestion/
│   ├── collectors/
│   ├── websocket/
│   ├── schedulers/
│   └── validators/
│
├── features/
│   ├── technical/
│   ├── statistical/
│   ├── regime/
│   └── pipelines/
│
├── strategies/
│   ├── trend/
│   ├── mean_reversion/
│   ├── breakout/
│   ├── volatility/
│   └── ensemble/
│
├── backtest/
│   ├── vectorbt/
│   ├── backtrader/
│   ├── reports/
│   └── optimization/
│
├── execution/
│   ├── order_manager/
│   ├── brokers/
│   ├── smart_routing/
│   └── paper_trading/
│
├── risk/
│   ├── sizing/
│   ├── limits/
│   ├── drawdown/
│   └── kill_switch/
│
├── portfolio/
│   ├── allocation/
│   ├── hedging/
│   └── exposure/
│
├── ai/
│   ├── agents/
│   ├── models/
│   ├── prompts/
│   ├── memory/
│   └── orchestration/
│
├── monitoring/
│   ├── logs/
│   ├── metrics/
│   ├── alerts/
│   └── dashboards/
│
├── api/
│   ├── routes/
│   ├── websocket/
│   ├── middleware/
│   └── main.py
│
├── ui/
│   ├── dashboard/
│   ├── charts/
│   ├── controls/
│   └── terminal/
│
├── notebooks/
│   ├── research/
│   ├── experiments/
│   └── analysis/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── stress/
│
└── scripts/
    ├── bootstrap/
    ├── deployment/
    ├── retraining/
    └── maintenance/
