# ═══════════════════════════════════════════════════════════
# QUANTEX Frontend — Multi-stage Next.js Build
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Dependencies ───────────────────────────────
FROM node:22-alpine@sha256:16e22a550f3863206a3f701448c45f7912c6896a62de43add43bb9c86130c3e2 AS deps

WORKDIR /app

# Switch to root for npm install (node:22-alpine defaults to 'node' user)
USER root

COPY package.json package-lock.json ./

RUN npm ci --production=false

# ── Stage 2: Builder ────────────────────────────────────
FROM node:22-alpine@sha256:16e22a550f3863206a3f701448c45f7912c6896a62de43add43bb9c86130c3e2 AS builder

WORKDIR /app

# Switch to root for build (node:22-alpine defaults to 'node' user)
USER root

COPY --from=deps /app/node_modules ./node_modules
COPY . .

ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

RUN npm run build

# ── Stage 3: Runner ─────────────────────────────────────
FROM node:22-alpine@sha256:16e22a550f3863206a3f701448c45f7912c6896a62de43add43bb9c86130c3e2 AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Copy standalone build output (includes public/ if it exists)
COPY --from=builder /app/.next/standalone ./
# Copy static assets separately (Next.js requirement)
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD pgrep -f "server.js" > /dev/null && wget -q -O /dev/null http://frontend:3000/ || exit 1

CMD ["node", "server.js"]
