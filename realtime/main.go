package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ── Prometheus Metrics ─────────────────────────────────────

var (
	wsClientsGauge = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "realtime_ws_clients",
		Help: "Current number of connected WebSocket clients.",
	})
	eventsPublishedCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "realtime_events_published_total",
			Help: "Total number of events published, by type.",
		},
		[]string{"type"},
	)
	signalsReceivedCounter = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "realtime_signals_received_total",
		Help: "Total number of trading signals received via HTTP.",
	})
	ticksGeneratedCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "realtime_market_ticks_total",
			Help: "Total number of market ticks generated, by symbol.",
		},
		[]string{"symbol"},
	)
)

func init() {
	prometheus.MustRegister(wsClientsGauge)
	prometheus.MustRegister(eventsPublishedCounter)
	prometheus.MustRegister(signalsReceivedCounter)
	prometheus.MustRegister(ticksGeneratedCounter)
}

// ── Event Types ──────────────────────────────────────────────

type Event struct {
	Type      string                 `json:"type"`
	Source    string                 `json:"source"`
	Timestamp int64                  `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
}

type MarketTick struct {
	Symbol    string  `json:"symbol"`
	Price     float64 `json:"price"`
	Volume    float64 `json:"volume"`
	Bid       float64 `json:"bid"`
	Ask       float64 `json:"ask"`
	Timestamp int64   `json:"timestamp"`
}

type SignalEvent struct {
	Symbol     string  `json:"symbol"`
	Direction  string  `json:"direction"`
	Confidence float64 `json:"confidence"`
	EntryPrice float64 `json:"entry_price"`
	StopLoss   float64 `json:"stop_loss"`
	Reason     string  `json:"reason"`
	Timestamp  int64   `json:"timestamp"`
}

// ── Event Bus ────────────────────────────────────────────────

type EventBus struct {
	mu          sync.RWMutex
	subscribers map[string][]chan Event
}

func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[string][]chan Event),
	}
}

func (eb *EventBus) Subscribe(eventType string, buffer int) chan Event {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	ch := make(chan Event, buffer)
	eb.subscribers[eventType] = append(eb.subscribers[eventType], ch)
	return ch
}

func (eb *EventBus) Publish(event Event) {
	eb.mu.RLock()
	defer eb.mu.RUnlock()

	eventsPublishedCounter.WithLabelValues(event.Type).Inc()

	if chans, ok := eb.subscribers[event.Type]; ok {
		for _, ch := range chans {
			select {
			case ch <- event:
			default:
				// Drop if buffer full (non-blocking)
			}
		}
	}
	// Also publish to wildcard
	if chans, ok := eb.subscribers["*"]; ok {
		for _, ch := range chans {
			select {
			case ch <- event:
			default:
			}
		}
	}
}

// ── WebSocket Hub ────────────────────────────────────────────

type WSHub struct {
	mu      sync.RWMutex
	clients map[*websocket.Conn]bool
	upgrader websocket.Upgrader
}

func NewWSHub() *WSHub {
	return &WSHub{
		clients: make(map[*websocket.Conn]bool),
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins for development
			},
		},
	}
}

func (h *WSHub) Add(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.clients[conn] = true
	wsClientsGauge.Set(float64(len(h.clients)))
}

func (h *WSHub) Remove(conn *websocket.Conn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.clients, conn)
	conn.Close()
	wsClientsGauge.Set(float64(len(h.clients)))
}

func (h *WSHub) Broadcast(msg []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	for client := range h.clients {
		err := client.WriteMessage(websocket.TextMessage, msg)
		if err != nil {
			log.Printf("WS write error: %v", err)
			client.Close()
			delete(h.clients, client)
		}
	}
}

func (h *WSHub) HandleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WS upgrade error: %v", err)
		return
	}
	h.Add(conn)
	log.Printf("🔌 WS client connected (%d total)", len(h.clients))

	// Read loop (keeps connection alive, handles pong)
	go func() {
		defer h.Remove(conn)
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				break
			}
		}
	}()
}

// ── Market Simulator ─────────────────────────────────────────

type MarketSimulator struct {
	bus     *EventBus
	symbols []string
}

func NewMarketSimulator(bus *EventBus) *MarketSimulator {
	return &MarketSimulator{
		bus:     bus,
		symbols: []string{"BTCUSDT", "ETHUSDT", "SOLUSDT"},
	}
}

func (ms *MarketSimulator) Start() {
	log.Println("📊 Market simulator started")
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	basePrices := map[string]float64{
		"BTCUSDT": 50000.0,
		"ETHUSDT": 3000.0,
		"SOLUSDT": 150.0,
	}

	for range ticker.C {
		for _, symbol := range ms.symbols {
			base := basePrices[symbol]
			// Simple random walk
			price := base + (float64(time.Now().UnixNano()%200)-100)/100*base*0.001

			tick := MarketTick{
				Symbol:    symbol,
				Price:     price,
				Volume:    float64(time.Now().UnixNano() % 100),
				Bid:       price * 0.9998,
				Ask:       price * 1.0002,
				Timestamp: time.Now().UnixMilli(),
			}

			ticksGeneratedCounter.WithLabelValues(symbol).Inc()

			data, _ := json.Marshal(tick)
			var eventData map[string]interface{}
			json.Unmarshal(data, &eventData)

			ms.bus.Publish(Event{
				Type:      "market.tick",
				Source:    "simulator",
				Timestamp: tick.Timestamp,
				Data:      eventData,
			})
		}
	}
}

// ── HTTP Handlers ────────────────────────────────────────────

type RealtimeService struct {
	bus *EventBus
	hub *WSHub
}

func (rs *RealtimeService) handleSignal(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var signal SignalEvent
	if err := json.NewDecoder(r.Body).Decode(&signal); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	signal.Timestamp = time.Now().UnixMilli()

	signalsReceivedCounter.Inc()

	eventData, _ := json.Marshal(signal)
	var data map[string]interface{}
	json.Unmarshal(eventData, &data)

	rs.bus.Publish(Event{
		Type:      "signal.generated",
		Source:    "strategy",
		Timestamp: signal.Timestamp,
		Data:      data,
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (rs *RealtimeService) handleHealth(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]interface{}{
		"service":   "quantex-realtime",
		"status":    "ok",
		"clients":   len(rs.hub.clients),
		"timestamp": time.Now().UnixMilli(),
	})
}

// ── Event Bridge: Bus → WebSocket ────────────────────────────

func bridgeEvents(bus *EventBus, hub *WSHub) {
	ch := bus.Subscribe("*", 1000)
	for event := range ch {
		data, err := json.Marshal(event)
		if err != nil {
			continue
		}
		hub.Broadcast(data)
	}
}

// ── Main ─────────────────────────────────────────────────────

func main() {
	log.Println("🚀 QUANTEX Trinity Orchestrator (Go Layer B) starting...")

	bus := NewEventBus()
	hub := NewWSHub()
	svc := &RealtimeService{bus: bus, hub: hub}

	// ── NATS JetStream Orchestrator (Trinity Architecture Layer B) ──
	orchCfg := LoadOrchestratorConfig()
	orch, err := NewNATSOrchestrator(orchCfg)
	if err != nil {
		log.Printf("⚠️  NATS orchestrator init failed (running without NATS): %v", err)
	} else {
		if err := orch.SetupStream(); err != nil {
			log.Printf("⚠️  NATS stream setup failed: %v", err)
		}
		if err := orch.SubscribeRawSignals(); err != nil {
			log.Printf("⚠️  NATS subscribe failed: %v", err)
		}
		go orch.StartAggregationLoop()
		log.Println("✅ NATS orchestrator running")
	}

	// Start market simulator (kept for dev/testing without real exchange)
	sim := NewMarketSimulator(bus)
	go sim.Start()

	// Bridge events to WebSocket
	go bridgeEvents(bus, hub)

	// ── HTTP routes ──────────────────────────────────────────────
	mux := http.NewServeMux()
	mux.HandleFunc("/ws", hub.HandleWS)
	mux.HandleFunc("/api/v1/signal", svc.handleSignal)
	mux.HandleFunc("/health", svc.handleHealth)
	mux.Handle("/metrics", promhttp.Handler())

	// NATS orchestrator status endpoint
	mux.HandleFunc("/api/v1/orchestrator/status", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if orch != nil {
			result := orch.GetLastResult()
			if result != nil {
				json.NewEncoder(w).Encode(result)
				return
			}
		}
		json.NewEncoder(w).Encode(map[string]string{"status": "no_signals_yet"})
	})

	server := &http.Server{
		Addr:         ":8082",
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
	}

	// Graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		log.Println("🛑 Shutting down...")
		if orch != nil {
			orch.Close()
		}
		server.Close()
	}()

	log.Println("✅ QUANTEX Trinity Orchestrator ready on :8082")
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}
