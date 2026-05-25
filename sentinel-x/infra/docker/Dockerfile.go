FROM golang:1.23-alpine AS builder

# ── Build deps ────────────────────────────────────────────────────────────────
RUN apk add --no-cache git protoc protobuf-dev

WORKDIR /build

# ── Proto stubs + compile (single layer to handle missing go.sum) ────────────
COPY proto/ /proto/
COPY go/ .
RUN mkdir -p pkg/proto && \
    go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.35 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5 && \
    PATH="$PATH:$(go env GOPATH)/bin" protoc -I/proto \
        --go_opt=module=github.com/sentinelx/go \
        --go-grpc_opt=module=github.com/sentinelx/go \
        --go_out=. \
        --go-grpc_out=. \
        /proto/risk.proto /proto/orders.proto /proto/agents.proto && \
    go mod tidy && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /sentinel-gateway ./cmd/sentinel/...

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM gcr.io/distroless/static:nonroot
COPY --from=builder /sentinel-gateway /sentinel-gateway
EXPOSE 8080
ENTRYPOINT ["/sentinel-gateway"]
