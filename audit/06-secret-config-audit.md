# Secret and Configuration Audit

## Safety handling

The local environment file was inspected only for variable names, presence state,
duplicates, placeholders, permissions, and service mapping. No values are included
in this report or evidence. No secret-validation request was sent to an external
provider.

Sensitive variables were present for broker access, multiple AI providers,
Telegram, PostgreSQL, and Grafana. The database URL entry was a placeholder.
`ACCOUNT_EQUITY_USDT`, `PAPER_TRADING`, and `RISK_PCT_PER_TRADE` were duplicated.
The local environment file is ignored by Git, but its Windows ACL grants Modify to
broader local principals than the owner/administrator/system set.

## [AUD-004] Secrets are over-broadly stored, mounted, and distributed

- Severity: High
- Category: Secret management
- Component: Local environment and legacy Compose
- File: `.env` (ignored), `docker-compose.yml`
- Line: Secret definitions and service environment/mounts
- Runtime service: Legacy orchestrator and execution services
- Status: Confirmed
- Evidence: Real values are present for several external systems; the orchestrator receives many unrelated provider keys and mounts the environment file writable; the execution service receives broker credentials even in paper/testnet mode; local ACL permits Modify to additional principals.
- Impact: Compromise of one broad service or local principal can expose multiple systems; writable mounts allow tampering and persistence.
- Root cause: One shared secret file and service-wide environment injection replace per-service secret scopes.
- Reproduction: Compare variable names and Docker inspect environment/mount keys; inspect file ACL without printing values.
- Recommended fix: Rotate credentials whose exposure cannot be excluded; use per-service secret objects/read-only mounts; remove broker credentials from paper services; narrow filesystem ACLs; prefer short-lived credentials and outbound allowlists.
- Validation after fix: Each service receives only its declared secret names; paper topology has no live-broker secret; secret files are owner-restricted and read-only; logs and diagnostics remain redacted.
- Estimated effort: M
- Priority: P0

## [AUD-005] A TLS private key remains recoverable from Git history

- Severity: High
- Category: Historical secret exposure
- Component: nginx TLS
- File: Historical `deployment/nginx/ssl/quantex.key`
- Line: Removed in commit `f9b69ac`
- Runtime service: nginx
- Status: Confirmed
- Evidence: Git history records that a TLS private key was committed in the initial history and later removed as a leaked key; certificate/key rotation evidence was not found.
- Impact: If the certificate or key remains trusted, any holder of repository history can impersonate the endpoint or decrypt applicable traffic.
- Root cause: Key material was originally stored inside source control and deletion was treated as sufficient remediation.
- Reproduction: Inspect path history/commit metadata without displaying blob contents.
- Recommended fix: Revoke/reissue the certificate and key, rotate related trust material, document the incident, then consider history rewriting only as defense-in-depth.
- Validation after fix: Current certificate serial/fingerprint differs from the compromised pair, revocation/rotation is documented, and secret scanning blocks private-key patterns.
- Estimated effort: S
- Priority: P0

## Configuration quality

- Canonical Compose correctly uses fail-closed required interpolation for the
  runtime DB password, but the variable is undocumented.
- `.env.example` contains placeholders rather than real values.
- A portable scanner such as gitleaks/trufflehog/detect-secrets was not installed;
  manual pattern searching generated false positives and was not treated as proof.
- Secret rotation state is **unverified**, not assumed valid or invalid.

