# ═══════════════════════════════════════════════════════════
# QUANTEX — Nginx Dockerfile (with certbot fallback entrypoint)
# ═══════════════════════════════════════════════════════════

FROM nginx:1.26-alpine@sha256:1eadbb07820339e8bbfed18c771691970baee292ec4ab2558f1453d26153e22d

# Install OpenSSL for fallback cert generation
RUN apk --no-cache add openssl

# Copy nginx config and entrypoint
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 80 443

ENTRYPOINT ["/entrypoint.sh"]
