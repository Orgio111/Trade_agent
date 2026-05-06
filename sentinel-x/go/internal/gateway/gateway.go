// Package gateway provides the low-latency API gateway that routes
// agent council decisions to the OMS and risk engine.
package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb_agents "github.com/sentinelx/go/pkg/proto/agents"
	pb_orders  "github.com/sentinelx/go/pkg/proto/orders"
	pb_risk    "github.com/sentinelx/go/pkg/proto/risk"
	"github.com/sentinelx/go/internal/oms"
)

// Gateway is the central routing layer between Python agents and Go OMS.
type Gateway struct {
	log          *zap.Logger
	executor     *oms.Executor
	agentClient  pb_agents.AgentCouncilClient
	httpServer   *http.Server

	// Metrics
	requestsTotal *prometheus.CounterVec
	requestLatency *prometheus.HistogramVec
}

// TradeSignal is the JSON payload the Python supervisor POSTs to the gateway.
type TradeSignal struct {
	SessionID      string    `json:"session_id"`
	Symbol         string    `json:"symbol"`
	Side           string    `json:"side"`
	CurrentPrice   float64   `json:"current_price"`
	ConsensusScore float64   `json:"consensus_score"`
	ConfidencePct  float64   `json:"confidence_pct"`
	ATR14          float64   `json:"atr_14"`
	Rationale      string    `json:"rationale"`
	ReturnSeries   []float64 `json:"return_series"`
}

// GatewayConfig holds all gateway configuration.
type GatewayConfig struct {
	HTTPAddr      string
	RiskUDSPath   string
	AgentUDSPath  string
	PortfolioEquity float64
}

func New(cfg GatewayConfig, log *zap.Logger) (*Gateway, error) {
	exec, err := oms.NewExecutor(cfg.RiskUDSPath, log)
	if err != nil {
		return nil, fmt.Errorf("executor init: %w", err)
	}

	agentConn, err := grpc.NewClient(
		"unix://"+cfg.AgentUDSPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("agent council connection: %w", err)
	}

	reg := prometheus.DefaultRegisterer
	reqTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "gateway_requests_total",
		Help: "Requests processed by the gateway",
	}, []string{"endpoint", "status"})

	reqLatency := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "gateway_request_latency_seconds",
		Help:    "Gateway request latency",
		Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5},
	}, []string{"endpoint"})

	reg.MustRegister(reqTotal, reqLatency)

	g := &Gateway{
		log:            log,
		executor:       exec,
		agentClient:    pb_agents.NewAgentCouncilClient(agentConn),
		requestsTotal:  reqTotal,
		requestLatency: reqLatency,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/trade", g.handleTrade)
	mux.HandleFunc("/v1/health", g.handleHealth)
	mux.Handle("/metrics", promhttp.Handler())

	g.httpServer = &http.Server{
		Addr:         cfg.HTTPAddr,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	return g, nil
}

// Start begins serving the HTTP gateway.
func (g *Gateway) Start() error {
	g.log.Info("Gateway starting", zap.String("addr", g.httpServer.Addr))
	return g.httpServer.ListenAndServe()
}

// handleTrade is the main entry: receives signal → routes through risk → executes.
func (g *Gateway) handleTrade(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	defer func() {
		g.requestLatency.WithLabelValues("/v1/trade").Observe(time.Since(start).Seconds())
	}()

	if r.Method != http.MethodPost {
		g.requestsTotal.WithLabelValues("/v1/trade", "405").Inc()
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var sig TradeSignal
	if err := json.NewDecoder(r.Body).Decode(&sig); err != nil {
		g.requestsTotal.WithLabelValues("/v1/trade", "400").Inc()
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Confidence gate: block if < 75%
	if sig.ConfidencePct < 75.0 {
		g.log.Warn("Trade blocked — insufficient confidence",
			zap.String("symbol", sig.Symbol),
			zap.Float64("confidence_pct", sig.ConfidencePct),
		)
		g.requestsTotal.WithLabelValues("/v1/trade", "blocked").Inc()
		respondJSON(w, http.StatusOK, map[string]interface{}{
			"approved": false,
			"reason":   fmt.Sprintf("confidence %.1f%% < 75%% threshold", sig.ConfidencePct),
		})
		return
	}

	// Build order
	sideVal := pb_orders.OrderSide_BUY
	if sig.Side == "SELL" { sideVal = pb_orders.OrderSide_SELL }

	order := &pb_orders.Order{
		SessionId:          sig.SessionID,
		Symbol:             sig.Symbol,
		Side:               sideVal,
		OrderType:          pb_orders.OrderType_TWAP,
		SupervisorRationale: sig.Rationale,
		CreatedAtNs:        time.Now().UnixNano(),
	}

	// Build risk request
	riskReq := &pb_risk.RiskRequest{
		SessionId:      sig.SessionID,
		Symbol:         sig.Symbol,
		Side:           sig.Side,
		CurrentPrice:   sig.CurrentPrice,
		Atr_14:         sig.ATR14,
		ConsensusScore: sig.ConsensusScore,
		ReturnSeries:   sig.ReturnSeries,
	}

	ctx, cancel := context.WithTimeout(r.Context(), 8*time.Second)
	defer cancel()

	resultCh := g.executor.Submit(ctx, &pb_orders.OrderRequest{Order: order}, riskReq)

	select {
	case result, ok := <-resultCh:
		if !ok {
			g.requestsTotal.WithLabelValues("/v1/trade", "500").Inc()
			http.Error(w, "Executor channel closed unexpectedly", http.StatusInternalServerError)
			return
		}
		g.requestsTotal.WithLabelValues("/v1/trade", "200").Inc()
		respondJSON(w, http.StatusOK, map[string]interface{}{
			"order":    result.Order,
			"approved": result.Order.Status == pb_orders.OrderStatus_FILLED,
		})
	case <-ctx.Done():
		g.requestsTotal.WithLabelValues("/v1/trade", "timeout").Inc()
		http.Error(w, "Request timed out", http.StatusGatewayTimeout)
	}
}

func (g *Gateway) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	resp, err := g.agentClient.HealthCheck(ctx, &pb_agents.HealthRequest{Service: "gateway"})
	healthy := err == nil && resp.Healthy
	status := http.StatusOK
	if !healthy { status = http.StatusServiceUnavailable }
	respondJSON(w, status, map[string]bool{"healthy": healthy})
}

func respondJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
