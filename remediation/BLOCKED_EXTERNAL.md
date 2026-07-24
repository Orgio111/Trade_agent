# Externally Blocked Remediation

Updated: 2026-07-24 (Asia/Ulaanbaatar)

Only actions requiring authority or infrastructure outside this repository/local
runtime are listed here. Missing elapsed soak time is tracked separately and is
not mislabeled as external completion.

## AUD-005 — historical TLS certificate/key rotation

Status: `BLOCKED_EXTERNAL`

Confirmed: Git history records a removed TLS private key. The key was not
displayed, copied, or restored during remediation. Local secret scanning and
profile isolation are implemented.

Required certificate-owner actions:

1. Identify the affected certificate serial/fingerprint without sharing the key.
2. Revoke it with the issuing CA.
3. Generate a new private key outside Git and issue a replacement certificate.
4. Store/mount it through the approved secret manager.
5. Provide sanitized CA/active-certificate metadata proving revocation and a
   different active serial/fingerprint.

## AUD-009 — off-host immutable backup and key custody

Status: `BLOCKED_EXTERNAL`

Confirmed: encrypted local PostgreSQL/NATS backups and isolated restores pass.
Archives and AES keys remain local; that is not disaster-recovery separation.

Required storage/key-owner actions:

1. Approve an off-host immutable/WORM destination and retention policy.
2. Approve a KMS/HSM identity and separate backup-key custodian.
3. Copy ciphertext only, never the local key file, to the approved destination.
4. Store/wrap the key in KMS and provide sanitized object version/retention/key-ID
   evidence.
5. Authorize a second isolated restore drill from the off-host copy.

## Infrastructure and alert delivery

Status: `BLOCKED_EXTERNAL` where applicable

- Rebuilding source manifests for Kubernetes/Terraform is local work; applying
  them requires the cluster/cloud owner and is not authorized.
- Proving alert delivery requires an approved receiver credential and endpoint.
- External provider or broker credential rotation requires the credential owner.
- Signed artifact storage for tracked model/runtime artifacts requires an approved
  registry, retention owner, and signing identity.
- Live/testnet broker activation is explicitly not authorized. It must not be used
  as a validation shortcut.

## Not externally blocked, but incomplete

- A 24-hour fault-injection paper soak and subsequent seven-day paper soak require
  actual elapsed monitoring time. Scripts or short runtime evidence do not count.
- Rust and Terraform gates require installing approved toolchains; absence is
  reported, not treated as a pass.
