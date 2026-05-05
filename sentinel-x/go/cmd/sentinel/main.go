// Sentinel-X Go Service — API Gateway + OMS entry point.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.uber.org/zap"

	"github.com/sentinelx/go/internal/gateway"
)

func main() {
	log, _ := zap.NewProduction()
	defer log.Sync()

	riskUDS  := getenv("RISK_UDS_PATH",  "/tmp/sentinel-risk.sock")
	agentUDS := getenv("AGENT_UDS_PATH", "/tmp/sentinel-agent.sock")
	httpAddr := getenv("GATEWAY_ADDR",   ":8080")

	gw, err := gateway.New(gateway.GatewayConfig{
		HTTPAddr:        httpAddr,
		RiskUDSPath:     riskUDS,
		AgentUDSPath:    agentUDS,
		PortfolioEquity: 100_000.0,
	}, log)
	if err != nil {
		log.Fatal("Gateway init failed", zap.Error(err))
	}

	// Graceful shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Info("Starting Sentinel-X Gateway", zap.String("addr", httpAddr))
		if err := gw.Start(); err != nil {
			log.Error("Gateway stopped", zap.Error(err))
		}
	}()

	<-stop
	log.Info("Shutdown signal received")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = ctx
	log.Info("Sentinel-X Gateway stopped")
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
