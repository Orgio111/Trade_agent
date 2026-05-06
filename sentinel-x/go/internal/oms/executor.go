// Package oms provides the async Order Management System.
// Routes orders through the Rust Risk Engine via gRPC/UDS,
// then submits to exchange with retry + circuit breaker.
package oms

import (
	"context"
	"encoding/base64"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/sony/gobreaker"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb_orders "github.com/sentinelx/go/pkg/proto/orders"
	pb_risk   "github.com/sentinelx/go/pkg/proto/risk"
)

const (
	maxRetries      = 4
	initialBackoff  = 100 * time.Millisecond
	maxSlippageBps  = 20.0
	orderTimeoutSec = 5
)

// OrderResult bundles the filled order with its risk snapshot.
type OrderResult struct {
	Order       *pb_orders.Order
	RiskReport  *pb_risk.RiskResponse
	SBESnapshot []byte
}

// Executor handles the full order lifecycle: risk gate → exchange → audit log.
type Executor struct {
	log         *zap.Logger
	riskClient  pb_risk.RiskEngineClient
	breaker     *gobreaker.CircuitBreaker

	// Metrics
	ordersTotal     *prometheus.CounterVec
	slippageHist    *prometheus.HistogramVec
	latencyHist     *prometheus.HistogramVec
	riskRejections  prometheus.Counter

	// Kill-switch state (shared via atomic)
	killSwitchActive int32

	mu       sync.RWMutex
	openOrds map[string]*pb_orders.Order
}

func NewExecutor(riskUDSPath string, log *zap.Logger) (*Executor, error) {
	// Connect to Rust Risk Engine over Unix Domain Socket
	conn, err := grpc.NewClient(
		"unix://"+riskUDSPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("risk engine connection failed: %w", err)
	}

	// Circuit breaker: open after 5 consecutive failures, half-open after 10s
	cb := gobreaker.NewCircuitBreaker(gobreaker.Settings{
		Name:        "risk-engine",
		MaxRequests: 1,
		Interval:    30 * time.Second,
		Timeout:     10 * time.Second,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures > 5
		},
		OnStateChange: func(name string, from, to gobreaker.State) {
			log.Warn("Circuit breaker state change",
				zap.String("name", name),
				zap.String("from", from.String()),
				zap.String("to", to.String()),
			)
		},
	})

	reg := prometheus.DefaultRegisterer
	ordersTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "oms_orders_total",
		Help: "Total orders processed by OMS",
	}, []string{"symbol", "side", "status"})

	slippageHist := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "oms_slippage_bps",
		Help:    "Order execution slippage in basis points",
		Buckets: []float64{0, 1, 2, 5, 10, 20, 50},
	}, []string{"symbol"})

	latencyHist := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "oms_order_latency_seconds",
		Help:    "End-to-end order latency",
		Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0},
	}, []string{"symbol"},
	)

	riskRejections := prometheus.NewCounter(prometheus.CounterOpts{
		Name: "oms_risk_rejections_total",
		Help: "Orders rejected by the Rust Risk Engine",
	})

	reg.MustRegister(ordersTotal, slippageHist, latencyHist, riskRejections)

	return &Executor{
		log:            log,
		riskClient:     pb_risk.NewRiskEngineClient(conn),
		breaker:        cb,
		ordersTotal:    ordersTotal,
		slippageHist:   slippageHist,
		latencyHist:    latencyHist,
		riskRejections: riskRejections,
		openOrds:       make(map[string]*pb_orders.Order),
	}, nil
}

// Submit validates risk and executes the order asynchronously.
// Returns a channel that receives the OrderResult when complete.
func (e *Executor) Submit(
	ctx context.Context,
	req *pb_orders.OrderRequest,
	riskReq *pb_risk.RiskRequest,
) <-chan OrderResult {
	ch := make(chan OrderResult, 1)

	go func() {
		defer close(ch)
		start := time.Now()

		// ── Risk gate via Rust engine ──────────────────────────────────────
		if atomic.LoadInt32(&e.killSwitchActive) == 1 {
			ch <- e.rejected(req.Order, "Kill switch active — all trading halted", nil, nil)
			return
		}

		riskResp, err := e.callRiskEngine(ctx, riskReq)
		if err != nil {
			e.log.Error("Risk engine error", zap.Error(err))
			ch <- e.rejected(req.Order, "Risk engine unreachable: "+err.Error(), nil, nil)
			return
		}
		if !riskResp.Approved {
			e.riskRejections.Inc()
			e.log.Warn("Risk rejected",
				zap.String("symbol", req.Order.Symbol),
				zap.String("reason", riskResp.RejectionReason),
			)
			ch <- e.rejected(req.Order, riskResp.RejectionReason, riskResp, nil)
			return
		}

		// Apply risk-sized quantity
		req.Order.Quantity      = riskResp.PositionSizeUnits
		req.Order.StopPrice     = riskResp.StopLossPrice

		// Decode SBE payload for audit
		sbeBytes, _ := base64.StdEncoding.DecodeString(riskResp.SbePayload)
		req.SbePayload = sbeBytes

		// ── Exchange submission with exponential backoff ───────────────────
		filled, err := e.submitWithRetry(ctx, req.Order)
		if err != nil {
			e.log.Error("Order submission failed", zap.Error(err), zap.String("order_id", req.Order.OrderId))
			ch <- e.rejected(req.Order, err.Error(), riskResp, sbeBytes)
			return
		}

		latency := time.Since(start).Seconds()
		e.latencyHist.WithLabelValues(filled.Symbol).Observe(latency)
		e.ordersTotal.WithLabelValues(filled.Symbol, filled.Side.String(), filled.Status.String()).Inc()
		if filled.SlippageBps > 0 {
			e.slippageHist.WithLabelValues(filled.Symbol).Observe(filled.SlippageBps)
		}

		e.log.Info("Order filled",
			zap.String("order_id", filled.OrderId),
			zap.String("symbol", filled.Symbol),
			zap.Float64("qty", filled.Quantity),
			zap.Float64("avg_fill", filled.AvgFillPrice),
			zap.Float64("slippage_bps", filled.SlippageBps),
			zap.Float64("latency_ms", latency*1000),
		)

		ch <- OrderResult{Order: filled, RiskReport: riskResp, SBESnapshot: sbeBytes}
	}()

	return ch
}

// callRiskEngine wraps the gRPC call in the circuit breaker.
func (e *Executor) callRiskEngine(ctx context.Context, req *pb_risk.RiskRequest) (*pb_risk.RiskResponse, error) {
	result, err := e.breaker.Execute(func() (interface{}, error) {
		tctx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
		defer cancel()
		return e.riskClient.Validate(tctx, req)
	})
	if err != nil {
		return nil, err
	}
	return result.(*pb_risk.RiskResponse), nil
}

// submitWithRetry sends the order to the exchange with exponential backoff.
func (e *Executor) submitWithRetry(ctx context.Context, order *pb_orders.Order) (*pb_orders.Order, error) {
	var lastErr error
	backoff := initialBackoff

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
				backoff = time.Duration(float64(backoff) * math.Pow(2, float64(attempt)))
			}
		}

		filled, err := e.doSubmit(ctx, order)
		if err == nil {
			return filled, nil
		}
		lastErr = err
		e.log.Warn("Order attempt failed",
			zap.Int("attempt", attempt+1),
			zap.Error(err),
		)
	}
	return nil, fmt.Errorf("exhausted %d retries: %w", maxRetries, lastErr)
}

// doSubmit is the actual exchange call (paper mode: simulated fill).
func (e *Executor) doSubmit(ctx context.Context, order *pb_orders.Order) (*pb_orders.Order, error) {
	tctx, cancel := context.WithTimeout(ctx, orderTimeoutSec*time.Second)
	defer cancel()
	_ = tctx

	// Paper mode: simulate a market fill with micro-slippage
	order.OrderId      = uuid.NewString()
	order.Status       = pb_orders.OrderStatus_FILLED
	order.AvgFillPrice = order.GetStopPrice() // placeholder
	order.SlippageBps  = 3.5
	order.FilledAtNs   = time.Now().UnixNano()

	if order.SlippageBps > maxSlippageBps {
		return nil, fmt.Errorf("slippage %.1f bps exceeds max %d bps", order.SlippageBps, maxSlippageBps)
	}
	return order, nil
}

func (e *Executor) rejected(
	o *pb_orders.Order,
	reason string,
	risk *pb_risk.RiskResponse,
	sbe []byte,
) OrderResult {
	o.Status = pb_orders.OrderStatus_REJECTED
	e.ordersTotal.WithLabelValues(o.Symbol, o.Side.String(), "REJECTED").Inc()
	return OrderResult{Order: o, RiskReport: risk, SBESnapshot: sbe}
}

// TripKillSwitch atomically halts the OMS.
func (e *Executor) TripKillSwitch() {
	atomic.StoreInt32(&e.killSwitchActive, 1)
	e.log.Error("OMS KILL SWITCH TRIPPED — rejecting all new orders")
}

// ResetKillSwitch re-enables the OMS (operator use only).
func (e *Executor) ResetKillSwitch() {
	atomic.StoreInt32(&e.killSwitchActive, 0)
	e.log.Warn("OMS kill switch manually reset")
}
