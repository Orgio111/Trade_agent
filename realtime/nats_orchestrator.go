package main

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
)

// ── Signal Types ───────────────────────────────────────────────

// BrainSignal is published by each Python brain to signals.raw.
type BrainSignal struct {
	BrainID     string                 `json:"brain_id"`
	Symbol      string                 `json:"symbol"`
	Score       float64                `json:"score"`        // -1.0 to 1.0
	Confidence  float64                `json:"confidence"`   // 0.0 to 1.0
	TimestampMs int64                  `json:"timestamp_ms"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// AggregatedSignal is published by the Go orchestrator to signals.aggregated.
type AggregatedSignal struct {
	Action      string             `json:"action"`        // BUY | SELL | HOLD
	Symbol      string             `json:"symbol"`
	FinalScore  float64            `json:"final_score"`
	ActiveBrains int               `json:"active_brains"`
	BrainScores map[string]float64 `json:"brain_scores"`
	WeightsUsed map[string]float64 `json:"weights_used"`
	TimestampMs int64              `json:"timestamp_ms"`
	Confidence  float64            `json:"confidence"`
}

// ── NATS Orchestrator ──────────────────────────────────────────

// NATSOrchestrator is Layer B of the Trinity Architecture.
// It subscribes to signals.raw, aggregates brain scores with weighted
// averaging, and publishes the final decision to signals.aggregated.
type NATSOrchestrator struct {
	cfg    *OrchestratorConfig
	nc     *nats.Conn
	js     jetstream.JetStream

	// Current window of brain signals (keyed by brain_id)
	mu        sync.RWMutex
	signals   map[string]BrainSignal // brain_id → latest signal
	symbols   map[string]string      // brain_id → symbol (all must match)

	// Last aggregation result for metrics/health
	lastResult *AggregatedSignal
}

// NewNATSOrchestrator creates and initialises the orchestrator.
func NewNATSOrchestrator(cfg *OrchestratorConfig) (*NATSOrchestrator, error) {
	nc, err := nats.Connect(cfg.NATSURL,
		nats.Name("quantex-orchestrator"),
		nats.ReconnectWait(2*time.Second),
		nats.MaxReconnects(60),
		nats.DisconnectErrHandler(func(_ *nats.Conn, err error) {
			if err != nil {
				log.Printf("⚠️  NATS disconnect: %v", err)
			}
		}),
		nats.ReconnectHandler(func(_ *nats.Conn) {
			log.Printf("✅ NATS reconnected")
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("NATS connect failed: %w", err)
	}

	js, err := jetstream.New(nc)
	if err != nil {
		nc.Close()
		return nil, fmt.Errorf("JetStream init failed: %w", err)
	}

	orch := &NATSOrchestrator{
		cfg:     cfg,
		nc:      nc,
		js:      js,
		signals: make(map[string]BrainSignal),
		symbols: make(map[string]string),
	}

	return orch, nil
}

// SetupStream ensures the JetStream stream and consumer exist.
func (o *NATSOrchestrator) SetupStream() error {
	ctx := &natsContext{}

	// Create stream if it doesn't exist
	stream, err := o.js.Stream(ctx, o.cfg.NATSStreamName)
	if err != nil {
		// Stream doesn't exist — create it
		stream, err = o.js.CreateStream(ctx, jetstream.StreamConfig{
			Name:     o.cfg.NATSStreamName,
			Subjects: []string{o.cfg.NATSSubjectRaw, o.cfg.NATSSubjectAggregated, "signals.executed"},
			Retention: jetstream.LimitsPolicy,
			MaxMsgs:  100000,
			MaxAge:   72 * time.Hour,
			Storage:  jetstream.FileStorage,
			Replicas: 1,
		})
		if err != nil {
			return fmt.Errorf("create stream failed: %w", err)
		}
		log.Printf("✅ NATS stream '%s' created", o.cfg.NATSStreamName)
	} else {
		log.Printf("✅ NATS stream '%s' exists", o.cfg.NATSStreamName)
	}

	_ = stream // stream is managed by JetStream; we subscribe directly
	return nil
}

// SubscribeRawSignals starts consuming brain signals from signals.raw.
func (o *NATSOrchestrator) SubscribeRawSignals() error {
	ctx := &natsContext{}

	_, err := o.nc.Subscribe(o.cfg.NATSSubjectRaw, func(msg *nats.Msg) {
		var sig BrainSignal
		if err := json.Unmarshal(msg.Data, &sig); err != nil {
			log.Printf("⚠️  Invalid brain signal: %v", err)
			return
		}

		o.mu.Lock()
		o.signals[sig.BrainID] = sig
		o.symbols[sig.BrainID] = sig.Symbol
		o.mu.Unlock()

		signalsReceivedCounter.Inc()
	})
	if err != nil {
		return fmt.Errorf("subscribe to %s failed: %w", o.cfg.NATSSubjectRaw, err)
	}

	log.Printf("✅ Subscribed to %s", o.cfg.NATSSubjectRaw)
	_ = ctx
	return nil
}

// StartAggregationLoop runs the periodic weighted aggregation and publishes
// the result to signals.aggregated.
func (o *NATSOrchestrator) StartAggregationLoop() {
	interval := time.Duration(o.cfg.AggregationIntervalSecs) * time.Second
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	log.Printf("🔄 Aggregation loop started (interval: %s)", interval)

	for range ticker.C {
		result := o.aggregate()
		if result == nil {
			continue // no signals yet
		}

		data, err := json.Marshal(result)
		if err != nil {
			log.Printf("⚠️  Marshal aggregated signal failed: %v", err)
			continue
		}

		if err := o.nc.Publish(o.cfg.NATSSubjectAggregated, data); err != nil {
			log.Printf("⚠️  Publish to %s failed: %v", o.cfg.NATSSubjectAggregated, err)
			continue
		}

		o.mu.Lock()
		o.lastResult = result
		o.mu.Unlock()

		log.Printf("📊 [%s] %s → score=%.4f (%d brains)",
			result.Symbol, result.Action, result.FinalScore, result.ActiveBrains)
	}
}

// aggregate computes the weighted score from all brain signals received
// in the current window. Only brains that have published are included.
func (o *NATSOrchestrator) aggregate() *AggregatedSignal {
	o.mu.RLock()
	defer o.mu.RUnlock()

	if len(o.signals) == 0 {
		return nil
	}

	// Build weight lookup
	weightMap := make(map[string]float64, len(o.cfg.BrainWeights))
	for _, bw := range o.cfg.BrainWeights {
		weightMap[bw.ID] = bw.Weight
	}

	// Determine the majority symbol (all brains should target the same symbol)
	symbolCounts := make(map[string]int)
	for _, sym := range o.symbols {
		symbolCounts[sym]++
	}
	dominantSymbol := "BTC/USDT"
	maxCount := 0
	for sym, cnt := range symbolCounts {
		if cnt > maxCount {
			dominantSymbol = sym
			maxCount = cnt
		}
	}

	// Weighted average
	var weightedSum, totalWeight float64
	brainScores := make(map[string]float64, len(o.signals))
	weightsUsed := make(map[string]float64, len(o.signals))
	var confidenceSum float64

	for brainID, sig := range o.signals {
		w, ok := weightMap[brainID]
		if !ok {
			w = 0.05 // tiny default weight for unknown brains
		}

		// Only count signals targeting the dominant symbol
		if sig.Symbol != dominantSymbol {
			continue
		}

		weightedSum += sig.Score * w * sig.Confidence
		totalWeight += w * sig.Confidence
		brainScores[brainID] = sig.Score
		weightsUsed[brainID] = w
		confidenceSum += sig.Confidence
	}

	if totalWeight == 0 {
		return nil
	}

	finalScore := weightedSum / totalWeight
	action := "HOLD"
	if finalScore > o.cfg.BuyThreshold {
		action = "BUY"
	} else if finalScore < o.cfg.SellThreshold {
		action = "SELL"
	}

	activeBrains := len(brainScores)
	avgConfidence := confidenceSum / float64(activeBrains)

	// Clear signals for next window (reset after aggregation)
	o.mu.RUnlock()
	o.mu.Lock()
	o.signals = make(map[string]BrainSignal)
	o.symbols = make(map[string]string)
	o.mu.Unlock()
	o.mu.RLock()

	return &AggregatedSignal{
		Action:       action,
		Symbol:       dominantSymbol,
		FinalScore:   finalScore,
		ActiveBrains: activeBrains,
		BrainScores:  brainScores,
		WeightsUsed:  weightsUsed,
		TimestampMs:  time.Now().UnixMilli(),
		Confidence:   avgConfidence,
	}
}

// GetLastResult returns the most recent aggregation result (thread-safe).
func (o *NATSOrchestrator) GetLastResult() *AggregatedSignal {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.lastResult
}

// Close drains the NATS connection.
func (o *NATSOrchestrator) Close() {
	if o.nc != nil {
		o.nc.Drain()
		o.nc.Close()
	}
}

// ── Minimal context for JetStream calls ───────────────────────

type natsContext struct{}

func (c *natsContext) Deadline() (time.Time, bool)       { return time.Time{}, false }
func (c *natsContext) Done() <-chan struct{}              { return nil }
func (c *natsContext) Err() error                         { return nil }
func (c *natsContext) Value(_ interface{}) interface{}    { return nil }
func (c *natsContext) Cancel() {}
