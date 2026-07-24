# Security Audit

Security score: **20/100**. The canonical code reduces live-trading risk by using
paper execution and deterministic risk, but the observed network/runtime posture is
not approvable.

## [AUD-002] Unauthenticated mutation APIs are exposed through the legacy runtime

- Severity: Critical
- Category: Authentication / authorization
- Component: Legacy orchestrator and dashboard APIs
- File: `orchestrator/`, `services/dashboard/`, legacy Compose/nginx configuration
- Line: Multiple API route declarations
- Runtime service: `quantex-orchestrator`, dashboard, nginx
- Status: Confirmed
- Evidence: Orchestrator OpenAPI exposes 62 routes and no security scheme; state-changing routes include account reset, price changes, training/evolution, backtest/ML actions, memory writes, position/trade operations, and model warmup; host ports were bound on all interfaces. Dashboard CORS allows `*`.
- Impact: Any network-reachable caller can mutate state, trigger expensive work, corrupt test/account data, or combine the API with mounted secrets for wider compromise.
- Root cause: The legacy stack assumes a trusted local network but publishes administrative APIs without identity, authorization, or network isolation.
- Reproduction: Inspect OpenAPI security schemes/routes and Docker published bindings; no mutation call is required.
- Recommended fix: Immediately restrict bindings/firewall to loopback or a trusted gateway; disable legacy mutation routes; implement strong authentication, authorization, CSRF protection where applicable, rate limits, and audit events before exposure.
- Validation after fix: Anonymous mutation requests receive 401/403; only role-authorized identities can act; the paper stack remains non-public; external surface scan finds no direct admin port.
- Estimated effort: L
- Priority: P0

## [AUD-003] Chroma 1.5.9 is affected by a critical pre-auth code-injection issue

- Severity: Critical
- Category: Dependency / remote code execution
- Component: Vector database
- File: `docker-compose.yml`, dependency artifacts
- Line: Chroma image declaration
- Runtime service: `quantex-chroma`
- Status: Confirmed
- Evidence: Running/source version is 1.5.9. GHSA-f4j7-r4q5-qw2c/CVE-2026-45829 affects versions 1.0.0 through 1.5.9 and had no patched stable release at audit time. Chroma is unauthenticated inside the Docker network.
- Impact: A network-reachable attacker or compromised peer container may achieve code execution in Chroma's context.
- Root cause: An affected service is deployed without compensating authentication/network isolation and without an available upstream patch.
- Reproduction: Verify image version and advisory range; do not execute exploit payloads.
- Recommended fix: Remove Chroma from the production trading path or isolate it in a deny-by-default network with no untrusted writers; evaluate a safe replacement/fork; upgrade only after a verified patched release.
- Validation after fix: Image SBOM reports an unaffected version/replacement, network policy denies non-approved clients, and a safe security test confirms unauthorized writes are rejected.
- Estimated effort: M
- Priority: P0

## [AUD-016] Kubernetes and Terraform deployment surfaces are stale and overexposed

- Severity: Medium
- Category: Cloud/infrastructure security
- Component: Kubernetes and Hetzner Terraform
- File: `deployment/k8s/`, `deployment/terraform/`
- Line: Workload images, service specs, firewall rules
- Runtime service: Not currently deployed
- Status: Confirmed
- Evidence: K8s models the older topology, uses latest tags, lacks workload security contexts/NetworkPolicy/PDB coverage, and omits several readiness controls. Terraform permits SSH, HTTP/S, frontend, orchestrator, Prometheus, and Grafana from IPv4/IPv6 any-address CIDRs.
- Impact: If applied as written, unauthenticated administrative/monitoring surfaces can become Internet reachable and the canonical safety topology is not preserved.
- Root cause: Deployment manifests were not migrated with the application architecture and use permissive bootstrap rules.
- Reproduction: Static inspection of K8s resources and Terraform firewall blocks; no apply was performed.
- Recommended fix: Rebuild manifests from the canonical service inventory; private networking/VPN only for admin/metrics; deny-by-default NetworkPolicy; non-root/read-only/cap-drop security contexts; immutable images; secrets from a managed store.
- Validation after fix: Policy-as-code rejects public admin CIDRs, latest tags, missing probes/security contexts, and direct legacy service deployment.
- Estimated effort: L
- Priority: P1

## [AUD-019] Dependency and base-image advisories remain

- Severity: Medium
- Category: Dependency vulnerability
- Component: Python and worker image
- File: `uv.lock`, `infra/workers/Dockerfile`, `artifacts/workers-fixed.spdx.json`
- Line: Dependency/image definitions
- Runtime service: Canonical workers and Python services
- Status: Confirmed
- Evidence: Fresh `pip-audit` found four advisories across Chroma, setuptools, and torch; setuptools has a fixed release; the torch scanner range conflicts with current advisory metadata and needs manual resolution. Existing worker SARIF lists one critical and two high Debian/Perl findings with no fix.
- Impact: Known vulnerabilities or scanner ambiguity prevent a clean release attestation.
- Root cause: Dependency policy does not define accepted risk, VEX, remediation SLA, or rebuild cadence.
- Reproduction: Run locked `pip-audit`, npm audit, and image scanning; compare advisory IDs to primary records.
- Recommended fix: Upgrade setuptools; manually verify torch applicability and record VEX; rebuild on a patched/minimal base when available; fail CI on unaccepted critical/high results.
- Validation after fix: Fresh language/image scans are clean or each residual advisory has reviewed, expiring VEX.
- Estimated effort: S
- Priority: P1

## [AUD-026] Edge hardening and dashboard plugin hygiene are incomplete

- Severity: Low
- Category: Defense in depth
- Component: nginx and Grafana
- File: `deployment/nginx/`, `monitoring/`
- Line: Headers/plugin configuration
- Runtime service: `quantex-nginx`, `quantex-grafana`
- Status: Confirmed
- Evidence: HTTPS sends HSTS, nosniff, SAMEORIGIN, and Referrer-Policy, but no CSP; nginx banner is visible; Grafana logs deprecated Angular/plugin-permission errors.
- Impact: Reduced browser exploit containment and noisy monitoring that can conceal meaningful errors.
- Root cause: Baseline edge/plugin policy is incomplete.
- Reproduction: Read HTTPS response headers and Grafana logs.
- Recommended fix: Add a tested CSP, minimize server disclosure, remove deprecated plugins, pin/install plugins at image build time.
- Validation after fix: Header scan passes the approved baseline and Grafana starts without plugin errors.
- Estimated effort: S
- Priority: P2

## Security decision

```text
SECURITY APPROVED: NO
LIVE BROKER CREDENTIALS IN PAPER RUNTIME: NOT ACCEPTABLE
PUBLIC ADMIN/API EXPOSURE: NOT ACCEPTABLE
```

The canonical deterministic risk gate is a positive control, but it cannot
compensate for the deployed access-control and dependency risks.

