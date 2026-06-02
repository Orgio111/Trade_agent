#!/bin/bash
# ═══════════════════════════════════════════════════════════
# QUANTEX — Self-Signed SSL Certificate Generator
# Usage: bash generate.sh
# Output: quantex.crt, quantex.key in the same directory
# ═══════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_FILE="${SCRIPT_DIR}/quantex.crt"
KEY_FILE="${SCRIPT_DIR}/quantex.key"
CONF_FILE="${SCRIPT_DIR}/quantex.cnf"

# Only generate if certificates don't already exist
if [ -f "${CERT_FILE}" ] && [ -f "${KEY_FILE}" ]; then
    echo "Certificates already exist at:"
    echo "  ${CERT_FILE}"
    echo "  ${KEY_FILE}"
    echo "Delete them first to regenerate."
    exit 0
fi

# Create a temporary OpenSSL config for SAN support on all platforms
cat > "${CONF_FILE}" << 'CONFEOF'
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = quantex.local
O = QUANTEX
C = US

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = quantex.local
DNS.2 = localhost
IP.1 = 127.0.0.1
CONFEOF

echo "Generating self-signed SSL certificates..."
echo "  Cert: ${CERT_FILE}"
echo "  Key:  ${KEY_FILE}"

# Generate CSR + self-sign using config file to avoid Windows path issues
openssl req -x509 \
    -newkey rsa:2048 \
    -keyout "${KEY_FILE}" \
    -out "${CERT_FILE}" \
    -days 365 \
    -nodes \
    -config "${CONF_FILE}"

# Clean up config file
rm -f "${CONF_FILE}"

# Set secure permissions
chmod 600 "${KEY_FILE}"
chmod 644 "${CERT_FILE}"

echo ""
echo "Certificates generated successfully!"
echo ""
echo "For production, replace these with real certificates from Let's Encrypt:"
echo "  certbot certonly --standalone -d yourdomain.com"
echo ""
echo "To trust this certificate locally (macOS):"
echo "  sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain ${CERT_FILE}"
echo ""
echo "To trust this certificate locally (Linux):"
echo "  sudo cp ${CERT_FILE} /usr/local/share/ca-certificates/quantex.crt && sudo update-ca-certificates"
echo ""
echo "To trust this certificate locally (Windows):"
echo "  certutil -addstore Root ${CERT_FILE}"
