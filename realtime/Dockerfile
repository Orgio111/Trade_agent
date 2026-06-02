# ═══════════════════════════════════════════════════════════
# QUANTEX Realtime — Multi-stage Go Build
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Builder ────────────────────────────────────
FROM golang:1.26-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk --no-cache add ca-certificates

# Copy dependency files (layer caching)
COPY go.mod go.sum ./
RUN go mod download

# Copy and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /realtime .

# ── Stage 2: Runner ─────────────────────────────────────
FROM alpine:3.20 AS runner

RUN apk --no-cache add ca-certificates && \
    adduser -D -u 1001 appuser

USER appuser

COPY --from=builder /realtime /realtime

EXPOSE 8082

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8082/health || exit 1

CMD ["/realtime"]
