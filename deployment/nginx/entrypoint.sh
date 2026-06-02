#!/bin/sh
# ═══════════════════════════════════════════════════════════
# QUANTEX — Nginx Entrypoint
# Generates fallback self-signed certs if Let's Encrypt certs
# don't exist yet, then starts nginx.
# ═══════════════════════════════════════════════════════════

set -e

DOMAIN="${CERTBOT_DOMAIN:-quantex.local}"
LE_LIVE_DIR="/etc/letsencrypt/live/${DOMAIN}"
LE_CERT="${LE_LIVE_DIR}/fullchain.pem"
LE_KEY="${LE_LIVE_DIR}/privkey.pem"

# If Let's Encrypt certs don't exist yet, generate self-signed fallbacks
# at the LE path so nginx can start with valid certs.
if [ ! -f "${LE_CERT}" ] || [ ! -f "${LE_KEY}" ]; then
    echo "Let's Encrypt certs not found for domain '${DOMAIN}'."
    echo "Generating temporary self-signed fallback certs..."

    mkdir -p "${LE_LIVE_DIR}"

    # Create OpenSSL config inline (avoids Windows path issues)
    cat > /tmp/fallback.cnf << 'EOF'
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = ${DOMAIN}
O = QUANTEX Fallback
C = US

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${DOMAIN}
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF

    # Replace ${DOMAIN} placeholder with actual domain
    sed -i "s/\${DOMAIN}/${DOMAIN}/g" /tmp/fallback.cnf

    openssl req -x509 \
        -newkey rsa:2048 \
        -keyout "${LE_KEY}" \
        -out "${LE_CERT}" \
        -days 7 \
        -nodes \
        -config /tmp/fallback.cnf

    rm -f /tmp/fallback.cnf
    chmod 600 "${LE_KEY}"
    echo "Fallback certs generated at: ${LE_LIVE_DIR}"
fi

# Replace __DOMAIN__ placeholder in nginx config
sed -i "s/__DOMAIN__/${DOMAIN}/g" /etc/nginx/nginx.conf
echo "Nginx config: domain placeholder substituted with '${DOMAIN}'"

# Start background periodic nginx reload (every 12h) to pick up renewed certs
# Certbot runs in a separate container and can't signal nginx directly
PERIODIC_RELOAD_INTERVAL=${NGINX_RELOAD_INTERVAL:-43200}
(
    while true; do
        sleep "${PERIODIC_RELOAD_INTERVAL}"
        echo "Periodic nginx reload to pick up any renewed certificates..."
        nginx -s reload
    done
) &

# Start nginx in the foreground
echo "Starting nginx..."
exec nginx -g "daemon off;"
