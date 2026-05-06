FROM golang:1.22-alpine AS builder

# ── Build deps ────────────────────────────────────────────────────────────────
RUN apk add --no-cache git protoc protobuf-dev

WORKDIR /build

# ── Proto stubs ───────────────────────────────────────────────────────────────
COPY proto/ /proto/
COPY go/ .
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest && \
    protoc -I/proto \
        --go_out=pkg/proto \
        --go-grpc_out=pkg/proto \
        /proto/risk.proto /proto/orders.proto /proto/agents.proto

# ── Compile ───────────────────────────────────────────────────────────────────
RUN go mod download && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /sentinel-gateway ./cmd/sentinel/...

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM gcr.io/distroless/static:nonroot
COPY --from=builder /sentinel-gateway /sentinel-gateway
EXPOSE 8080
ENTRYPOINT ["/sentinel-gateway"]
