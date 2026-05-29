"""SBE (Simple Binary Encoding) decoder for Sentinel-X Rust risk snapshots.

Matches the Rust SBE encoder in ``sentinel-x/rust/src/sbe/encoder.rs``:

Frame layout (40 bytes total):
┌──────────┬──────────┬──────────┬──────────┬──────────────────┐
│ Magic(2) │ Ver(1)   │ Flags(1) │ Len(4)   │ Body (32 bytes)  │
└──────────┴──────────┴──────────┴──────────┴──────────────────┘

Body (4 × f64 = 32 bytes, little-endian):
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ var_99 (f64) │ kelly (f64)  │ size_usd(f64)│ heat (f64)   │
└──────────────┴──────────────┴──────────────┴──────────────┘
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# ── SBE constants (must match Rust encoder) ───────────────────────────────────
_MAGIC = 0xABCD
_VERSION = 1
_FLAGS_RISK = 0x01
_BODY_LEN = 32  # 4 × f64
_HEADER_LEN = 8  # magic(2) + ver(1) + flags(1) + len(4)
_TOTAL_LEN = _HEADER_LEN + _BODY_LEN  # 40


@dataclass(frozen=True)
class RiskSnapshot:
    """Decoded SBE risk snapshot from the Rust engine."""

    var_99: float
    """Value at Risk (99% confidence) as a positive loss fraction."""

    kelly_fractional: float
    """Fractional Kelly position size as fraction of equity."""

    size_usd: float
    """Recommended position size in USD."""

    heat_score: float
    """Correlation-weighted portfolio heat (0.0 = none, 1.0 = max)."""


def decode_risk_snapshot(data: bytes) -> RiskSnapshot:
    """Decode an SBE-encoded risk snapshot from raw binary *data*.

    Parameters
    ----------
    data : bytes
        The full 40-byte SBE frame to decode.

    Returns
    -------
    RiskSnapshot
        Decoded risk metrics.

    Raises
    ------
    ValueError
        If the frame is too short, has an invalid magic number, or has an
        unexpected body length.
    """
    if len(data) < _TOTAL_LEN:
        raise ValueError(
            f"SBE frame too short: {len(data)} bytes (expected {_TOTAL_LEN})"
        )

    # ── Header validation ────────────────────────────────────────────────────
    magic, = struct.unpack_from(">H", data, 0)
    if magic != _MAGIC:
        raise ValueError(
            f"Invalid SBE magic: 0x{magic:04X} (expected 0x{_MAGIC:04X})"
        )

    _version = data[2]
    _flags = data[3]
    body_len, = struct.unpack_from(">I", data, 4)
    if body_len != _BODY_LEN:
        raise ValueError(
            f"Unexpected SBE body length: {body_len} (expected {_BODY_LEN})"
        )

    # ── Body: 4 × f64 little-endian ──────────────────────────────────────────
    var_99, kelly_frac, size_usd, heat_score = struct.unpack_from(
        "<dddd", data, _HEADER_LEN
    )

    return RiskSnapshot(
        var_99=var_99,
        kelly_fractional=kelly_frac,
        size_usd=size_usd,
        heat_score=heat_score,
    )


def decode_risk_snapshot_b64(b64_payload: str) -> RiskSnapshot:
    """Decode an SBE risk snapshot from a base64 string.

    This is the format returned by the gRPC bridge's ``sbe_payload`` field.

    Parameters
    ----------
    b64_payload : str
        Base64-encoded SBE frame (as returned by the Rust engine).

    Returns
    -------
    RiskSnapshot
        Decoded risk metrics.

    Raises
    ------
    ValueError
        If the payload is empty, invalid base64, or has an invalid SBE frame.
    """
    import base64

    if not b64_payload:
        raise ValueError("Empty SBE payload — nothing to decode")

    try:
        raw = base64.b64decode(b64_payload)
    except Exception as exc:
        raise ValueError(f"Invalid base64 in SBE payload: {exc}") from exc

    return decode_risk_snapshot(raw)
