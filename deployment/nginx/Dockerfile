# ═══════════════════════════════════════════════════════════
# QUANTEX — Nginx Dockerfile (with certbot fallback entrypoint)
# ═══════════════════════════════════════════════════════════

FROM nginx:1.26-alpine

# Install OpenSSL for fallback cert generation
RUN apk --no-cache add openssl

# Copy nginx config and entrypoint
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 80 443

ENTRYPOINT ["/entrypoint.sh"]
