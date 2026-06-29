package main

import (
	"os"
	"strconv"
)

// ── Brain Weight Configuration ──────────────────────────────────

// BrainWeight defines the weight for each AI brain in the final score.
type BrainWeight struct {
	ID     string
	Weight float64
}

// Default weights for the 11 Trinity Architecture brains.
// Total = 1.00 (timesfm gets highest as lead forecaster).
var DefaultBrainWeights = []BrainWeight{
	{ID: "timesfm", Weight: 0.25},
	{ID: "freqai", Weight: 0.15},
	{ID: "llm_regime", Weight: 0.15},
	{ID: "finbert_nlp", Weight: 0.07},
	{ID: "microstructure", Weight: 0.05},
	{ID: "orderflow_nautilus", Weight: 0.05},
	{ID: "finrl_kelly", Weight: 0.05},
	{ID: "statarb_funding", Weight: 0.05},
	{ID: "onchain_whale", Weight: 0.05},
	{ID: "custom_nn", Weight: 0.05},
	{ID: "polymarket_alpha", Weight: 0.05},
	{ID: "odoo_erp", Weight: 0.05},
}

// ── Orchestrator Configuration ─────────────────────────────────

// OrchestratorConfig holds all configuration for the NATS orchestrator.
// Every field has a sensible default and can be overridden via env vars.
type OrchestratorConfig struct {
	NATSURL                 string
	NATSSubjectRaw          string
	NATSSubjectAggregated   string
	NATSStreamName          string
	AggregationIntervalSecs int

	BuyThreshold  float64
	SellThreshold float64

	BrainWeights []BrainWeight
}

// LoadOrchestratorConfig reads config from environment with fallback defaults.
func LoadOrchestratorConfig() *OrchestratorConfig {
	cfg := &OrchestratorConfig{
		NATSURL:                 envStr("NATS_URL", "nats://localhost:4222"),
		NATSSubjectRaw:          envStr("NATS_SUBJECT_RAW", "signals.raw"),
		NATSSubjectAggregated:   envStr("NATS_SUBJECT_AGGREGATED", "signals.aggregated"),
		NATSStreamName:          envStr("NATS_STREAM_SIGNALS", "signals"),
		AggregationIntervalSecs: envInt("AGGREGATION_INTERVAL_SECS", 15),
		BuyThreshold:            envFloat("BUY_THRESHOLD", 0.35),
		SellThreshold:           envFloat("SELL_THRESHOLD", -0.35),
		BrainWeights:            make([]BrainWeight, len(DefaultBrainWeights)),
	}

	copy(cfg.BrainWeights, DefaultBrainWeights)

	// Allow per-brain weight overrides: BRAIN_WEIGHT_TIMESFM=0.30
	for i, bw := range cfg.BrainWeights {
		envKey := "BRAIN_WEIGHT_" + toUnderscoreUpper(bw.ID)
		if w := os.Getenv(envKey); w != "" {
			if parsed, err := strconv.ParseFloat(w, 64); err == nil && parsed >= 0 {
				cfg.BrainWeights[i].Weight = parsed
			}
		}
	}

	return cfg
}

// ── Helpers ─────────────────────────────────────────────────────

func envStr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}

func toUnderscoreUpper(s string) string {
	result := make([]byte, 0, len(s)+4)
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'a' && c <= 'z' {
			c -= 32
		}
		if c == '_' || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
			result = append(result, c)
		}
	}
	return string(result)
}
