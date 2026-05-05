//! Simple Binary Encoding (SBE) — zero-copy risk snapshot serialization.
//!
//! Frame layout (32 bytes total):
//! ┌──────────┬──────────┬──────────┬──────────┬──────────┐
//! │ Magic(2) │ Ver(1)   │ Flags(1) │ Len(4)   │ Body(N)  │
//! └──────────┴──────────┴──────────┴──────────┴──────────┘
//!
//! Risk snapshot body (fixed 32 bytes):
//! ┌──────────────┬──────────────┬──────────────┬──────────────┐
//! │ var_99 (f64) │ kelly   (f64)│ size_usd (f64│ heat    (f64)│
//! └──────────────┴──────────────┴──────────────┴──────────────┘

const MAGIC: u16       = 0xABCD;
const VERSION: u8      = 1;
const FLAGS_RISK: u8   = 0x01;
const BODY_LEN: u32    = 32; // 4 × f64

/// Encode a risk snapshot into a 40-byte SBE frame.
pub fn encode_risk_snapshot(
    var_99: f64,
    kelly_frac: f64,
    size_usd: f64,
    heat_score: f64,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + BODY_LEN as usize);

    // Header (8 bytes)
    buf.extend_from_slice(&MAGIC.to_be_bytes());
    buf.push(VERSION);
    buf.push(FLAGS_RISK);
    buf.extend_from_slice(&BODY_LEN.to_be_bytes());

    // Body (4 × f64 = 32 bytes, little-endian for CPU cache alignment)
    buf.extend_from_slice(&var_99.to_le_bytes());
    buf.extend_from_slice(&kelly_frac.to_le_bytes());
    buf.extend_from_slice(&size_usd.to_le_bytes());
    buf.extend_from_slice(&heat_score.to_le_bytes());

    debug_assert_eq!(buf.len(), 40);
    buf
}

#[derive(Debug, PartialEq)]
pub struct RiskSnapshot {
    pub var_99:     f64,
    pub kelly_frac: f64,
    pub size_usd:   f64,
    pub heat_score: f64,
}

/// Decode an SBE frame produced by `encode_risk_snapshot`.
pub fn decode_risk_snapshot(buf: &[u8]) -> Result<RiskSnapshot, &'static str> {
    if buf.len() < 40 { return Err("Buffer too short"); }

    let magic = u16::from_be_bytes([buf[0], buf[1]]);
    if magic != MAGIC { return Err("Invalid SBE magic"); }

    let _version = buf[2];
    let _flags   = buf[3];
    let body_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if body_len != BODY_LEN { return Err("Unexpected body length"); }

    let var_99     = f64::from_le_bytes(buf[8..16].try_into().unwrap());
    let kelly_frac = f64::from_le_bytes(buf[16..24].try_into().unwrap());
    let size_usd   = f64::from_le_bytes(buf[24..32].try_into().unwrap());
    let heat_score = f64::from_le_bytes(buf[32..40].try_into().unwrap());

    Ok(RiskSnapshot { var_99, kelly_frac, size_usd, heat_score })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_encoding() {
        let encoded = encode_risk_snapshot(0.035, 0.0875, 8_750.0, 0.42);
        let decoded = decode_risk_snapshot(&encoded).unwrap();
        assert!((decoded.var_99 - 0.035).abs() < 1e-12);
        assert!((decoded.kelly_frac - 0.0875).abs() < 1e-12);
        assert!((decoded.size_usd - 8_750.0).abs() < 1e-9);
        assert!((decoded.heat_score - 0.42).abs() < 1e-12);
    }

    #[test]
    fn frame_is_exactly_40_bytes() {
        let buf = encode_risk_snapshot(0.01, 0.05, 5000.0, 0.3);
        assert_eq!(buf.len(), 40);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = encode_risk_snapshot(0.01, 0.05, 5000.0, 0.3);
        buf[0] = 0xFF;
        assert!(decode_risk_snapshot(&buf).is_err());
    }
}
