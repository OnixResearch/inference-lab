//! Issue #54: Measure a bounded tufazip-style online Transformer compressor on a small text prefix.
//!
//! Default-off (`default_enabled: false`). Pinned upstream reference:
//! - `Tufalabs/tufazip@b9d678db3a0f726e117c8cab51927975830395ac` (MIT)
//! - Official README requires Linux x86-64, 2x NVIDIA B200 GPUs, CUDA 13.0 for `enwik9-v2.bin`.
//!   Hardware gate is recorded cleanly (`unavailable_hardware_runtime`) and full `enwik9` is NOT
//!   run as a first arm (`enwik9_full_run_attempted: false`).
//! - Author's `105,388,353`-byte total, 2nd-place ranking, and `631,548 s` / `645,748 s` run times
//!   are strictly unverified related work and NEVER imported as lab pass/fail thresholds.
//!
//! Provides:
//! - Genuine dependency-free pure-Rust miniature **online-learning fixed-point range codec**
//!   initialized from a serialized quantized offline prior (`prior_bytes`), updating its byte-level
//!   context predictor weights online step-by-step identically during encode AND decode WITHOUT
//!   transferring online-trained weights!
//! - Verified **bit-exact lossless round-trip** (`sha256(decoded) == sha256(input)`), accounting
//!   for `online_total_bytes = payload_bytes + offline_prior_bytes`.
//! - Real pure-Rust conventional LZ77 baseline on the exact same pinned text prefix, also verified
//!   for bit-exact SHA-256 lossless round-trip.
//! - 8 executable negative controls, including proving that a decoder which skips online weight
//!   updates fails lossless round-trip (explicit finding, never a silent pass).

use std::fmt::Write as FmtWrite;

use crate::json::{opt_f64, quote, sha256_hex};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "Tufalabs/tufazip";
pub const UPSTREAM_COMMIT: &str = "b9d678db3a0f726e117c8cab51927975830395ac";
pub const UPSTREAM_LICENSE: &str = "MIT";
pub const SOURCE_BOOKMARK: &str = "https://x.com/tufalabs/status/2123350647273631907";

const ALPHABET: usize = 256;
const NUM_CTX_BUCKETS: usize = 64;
const PRIOR_CLASSES: usize = 16;

#[derive(Debug, Clone)]
pub struct HostHardwareProfile {
    pub os_arch: String,
    pub gpu_model: String,
    pub visible_gpu_count: usize,
    pub cuda_version: String,
}

#[derive(Debug, Clone)]
pub struct OnlineEncodedArtifact {
    pub original_len: usize,
    pub offline_prior_bytes: Vec<u8>,
    pub payload_bytes: Vec<u8>,
    pub online_weights_transferred: bool,
    pub online_update_steps: usize,
}

#[derive(Debug, Clone)]
pub struct CodecMeasurement {
    pub codec_name: String,
    pub input_bytes: usize,
    pub payload_bytes: usize,
    pub offline_prior_bytes: usize,
    pub total_bytes: usize,
    pub bits_per_byte: Option<f64>,
    pub compression_ratio: Option<f64>,
    pub online_weights_transferred: bool,
    pub decoder_relearned_online: bool,
    pub roundtrip_sha256_match: bool,
    pub decoded_sha256: String,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct TufazipOnlineCodecReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub default_enabled: bool,
    pub enwik9_full_run_attempted: bool,
    pub input_prefix_bytes: usize,
    pub input_sha256: DigestRef,
    pub online_codec: CodecMeasurement,
    pub conventional_lz77: CodecMeasurement,
    pub negative_controls: Vec<NegativeControl>,
}

/// Evaluate whether a host meets official `tufazip` B200 decoding requirements.
pub fn check_tufazip_b200_hardware_gate(host: &HostHardwareProfile) -> OutcomeStatus {
    let arch_ok = host.os_arch == "linux-x86_64";
    let gpu_ok = host.gpu_model == "NVIDIA B200" && host.visible_gpu_count == 2;
    let cuda_ok = host.cuda_version == "13.0";
    if arch_ok && gpu_ok && cuda_ok {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::UnavailableHardwareRuntime
    }
}

/// Pure-Rust quantized offline prior (16 byte-class log-weights stored as 16 signed bytes).
pub fn default_quantized_offline_prior() -> Vec<u8> {
    // 16 quantized prior weights (1 byte each = 16 bytes serialized offline prior):
    // biases printable ASCII lowercase/space/punctuation higher than control bytes.
    let weights: [i8; PRIOR_CLASSES] = [
        -12, -12, 18, 10, 8, 14, 24, 22, -16, -16, -16, -16, -16, -16, -16, -16,
    ];
    weights.iter().map(|&w| w as u8).collect()
}

fn byte_class(b: u8) -> usize {
    (b as usize) >> 4
}

/// Deterministic integer fixed-point online byte predictor.
/// Initialized solely from `offline_prior_bytes` (no online weights transferred).
/// Both encoder and decoder run the exact same `predict_cdf` -> `update_online` loop.
struct OnlineBytePredictor {
    base_counts: [u32; ALPHABET],
    ctx_counts: Vec<[u16; ALPHABET]>,
    prev1: u8,
    prev2: u8,
    steps: usize,
}

impl OnlineBytePredictor {
    fn from_quantized_prior(prior_bytes: &[u8]) -> Result<Self, String> {
        if prior_bytes.len() != PRIOR_CLASSES {
            return Err(format!(
                "expected {PRIOR_CLASSES}-byte quantized offline prior, got {}",
                prior_bytes.len()
            ));
        }
        let mut base_counts = [1u32; ALPHABET];
        for (b, slot) in base_counts.iter_mut().enumerate() {
            let q = prior_bytes[byte_class(b as u8)] as i8;
            let init = if q >= 0 { 2u32 + (q as u32) } else { 1u32 };
            *slot = init;
        }
        Ok(Self {
            base_counts,
            ctx_counts: vec![[0u16; ALPHABET]; NUM_CTX_BUCKETS],
            prev1: 0,
            prev2: 0,
            steps: 0,
        })
    }

    fn context_bucket(&self) -> usize {
        let h = ((self.prev1 as usize) * 31) ^ ((self.prev2 as usize) * 13);
        h & (NUM_CTX_BUCKETS - 1)
    }

    /// Build strictly positive cumulative distribution `cdf[0..=256]` with total <= 16384.
    fn build_cdf(&self) -> ([u32; ALPHABET + 1], u32) {
        let bucket = self.context_bucket();
        let ctx_row = &self.ctx_counts[bucket];
        let mut raw = [0u32; ALPHABET];
        let mut raw_total = 0u32;
        for b in 0..ALPHABET {
            // Mix quantized offline prior base counts with online-learned context counts.
            let w = self.base_counts[b] + (ctx_row[b] as u32) * 8;
            raw[b] = w;
            raw_total += w;
        }
        // Rescale cleanly into total <= 16384 while preserving strictly positive >= 1 per symbol.
        let mut cdf = [0u32; ALPHABET + 1];
        let mut acc = 0u32;
        let max_total = 16_384u32;
        for b in 0..ALPHABET {
            let scaled = if raw_total <= max_total {
                raw[b]
            } else {
                let s =
                    ((raw[b] as u64) * ((max_total - ALPHABET as u32) as u64)) / (raw_total as u64);
                (s as u32) + 1
            };
            cdf[b] = acc;
            acc += scaled;
        }
        cdf[ALPHABET] = acc;
        (cdf, acc)
    }

    /// Online weight update executed identically after each byte by encoder and decoder.
    fn update_online(&mut self, sym: u8) {
        let bucket = self.context_bucket();
        let idx = sym as usize;
        if self.ctx_counts[bucket][idx] < 2000 {
            self.ctx_counts[bucket][idx] += 4;
        }
        if self.base_counts[idx] < 4000 {
            self.base_counts[idx] += 1;
        }
        self.prev2 = self.prev1;
        self.prev1 = sym;
        self.steps += 1;
    }
}

/// 32-bit exact integer carryless range encoder (sub-bit arithmetic precision, zero float drift).
struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    first_byte: bool,
    out: Vec<u8>,
}

impl RangeEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            first_byte: true,
            out: Vec::new(),
        }
    }

    fn shift_low(&mut self) {
        let low_hi = (self.low >> 32) as u32;
        if low_hi != 0 || (self.low as u32) < 0xFF00_0000 {
            let mut temp = self.cache;
            loop {
                if !self.first_byte {
                    self.out.push(temp.wrapping_add(low_hi as u8));
                } else {
                    self.first_byte = false;
                }
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    fn encode_symbol(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let step = self.range / total;
        self.low += (step as u64) * (cum_low as u64);
        self.range = step * (cum_high - cum_low);
        while self.range < (1u32 << 24) {
            self.range <<= 8;
            self.shift_low();
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    code: u32,
    range: u32,
}

impl<'a> RangeDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut dec = Self {
            data,
            pos: 0,
            code: 0,
            range: 0xFFFF_FFFF,
        };
        for _ in 0..4 {
            dec.code = (dec.code << 8) | (dec.read_byte() as u32);
        }
        dec
    }

    fn read_byte(&mut self) -> u8 {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }

    fn decode_symbol(&mut self, cdf: &[u32; ALPHABET + 1], total: u32) -> u8 {
        let step = self.range / total;
        let target_raw = self.code / step;
        let target = if target_raw >= total {
            total - 1
        } else {
            target_raw
        };
        // Binary search in strictly increasing cdf[0..=256]
        let mut lo = 0usize;
        let mut hi = ALPHABET;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid] <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let cum_low = cdf[lo];
        let cum_high = cdf[lo + 1];
        self.code -= step * cum_low;
        self.range = step * (cum_high - cum_low);
        while self.range < (1u32 << 24) {
            self.range <<= 8;
            self.code = (self.code << 8) | (self.read_byte() as u32);
        }
        lo as u8
    }
}

/// Compress `input` with the miniature online-learning predictor initialized from `prior_bytes`.
/// Online-trained weights are NEVER serialized into the returned artifact.
pub fn encode_online_range(
    input: &[u8],
    prior_bytes: &[u8],
) -> Result<OnlineEncodedArtifact, String> {
    let mut predictor = OnlineBytePredictor::from_quantized_prior(prior_bytes)?;
    let mut enc = RangeEncoder::new();
    for &b in input {
        let (cdf, total) = predictor.build_cdf();
        let sym = b as usize;
        enc.encode_symbol(cdf[sym], cdf[sym + 1], total);
        predictor.update_online(b);
    }
    Ok(OnlineEncodedArtifact {
        original_len: input.len(),
        offline_prior_bytes: prior_bytes.to_vec(),
        payload_bytes: enc.finish(),
        online_weights_transferred: false,
        online_update_steps: predictor.steps,
    })
}

/// Decompress `artifact` by initializing from `artifact.offline_prior_bytes` and repeating the
/// online predictor weight updates step-by-step. Setting `repeat_online_learning = false` is a
/// negative control verifying that skipping decoder-side relearning breaks lossless reconstruction.
pub fn decode_online_range(
    artifact: &OnlineEncodedArtifact,
    repeat_online_learning: bool,
) -> Result<Vec<u8>, String> {
    let mut predictor = OnlineBytePredictor::from_quantized_prior(&artifact.offline_prior_bytes)?;
    let mut dec = RangeDecoder::new(&artifact.payload_bytes);
    let mut out = Vec::with_capacity(artifact.original_len);
    for _ in 0..artifact.original_len {
        let (cdf, total) = predictor.build_cdf();
        let b = dec.decode_symbol(&cdf, total);
        out.push(b);
        if repeat_online_learning {
            predictor.update_online(b);
        }
    }
    Ok(out)
}

/// Real pure-Rust conventional LZ77 baseline with a 4096-byte sliding window:
/// - Control byte `0x00..=0x7F`: literal run of `(ctrl + 1)` raw bytes (`1..=128`).
/// - Control byte `0x80..=0xFF`: back-reference match of length `((ctrl & 0x7F) + 4)` (`4..=131`),
///   followed by a 2-byte big-endian distance (`1..=4096`).
pub fn encode_lz77_baseline(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut lit_buf: Vec<u8> = Vec::new();
    let flush_lits = |lit_buf: &mut Vec<u8>, out: &mut Vec<u8>| {
        let mut off = 0usize;
        while off < lit_buf.len() {
            let chunk = if lit_buf.len() - off > 128 {
                128
            } else {
                lit_buf.len() - off
            };
            out.push((chunk - 1) as u8);
            out.extend_from_slice(&lit_buf[off..off + chunk]);
            off += chunk;
        }
        lit_buf.clear();
    };

    let mut i = 0usize;
    while i < input.len() {
        let win_start = i.saturating_sub(4096);
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        for start in win_start..i {
            let mut l = 0usize;
            while l < 131 && i + l < input.len() && input[start + l] == input[i + l] {
                l += 1;
            }
            if l >= 4 && l > best_len {
                best_len = l;
                best_dist = i - start;
            }
        }
        if best_len >= 4 {
            flush_lits(&mut lit_buf, &mut out);
            out.push(0x80 | ((best_len - 4) as u8));
            out.push(((best_dist >> 8) & 0xFF) as u8);
            out.push((best_dist & 0xFF) as u8);
            i += best_len;
        } else {
            lit_buf.push(input[i]);
            i += 1;
        }
    }
    flush_lits(&mut lit_buf, &mut out);
    out
}

pub fn decode_lz77_baseline(encoded: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < encoded.len() {
        let ctrl = encoded[i];
        i += 1;
        if ctrl < 0x80 {
            let len = (ctrl as usize) + 1;
            if i + len > encoded.len() {
                return Err("truncated LZ77 literal run".into());
            }
            out.extend_from_slice(&encoded[i..i + len]);
            i += len;
        } else {
            let len = ((ctrl & 0x7F) as usize) + 4;
            if i + 2 > encoded.len() {
                return Err("truncated LZ77 match header".into());
            }
            let dist = ((encoded[i] as usize) << 8) | (encoded[i + 1] as usize);
            i += 2;
            if dist == 0 || dist > out.len() {
                return Err("invalid LZ77 match distance".into());
            }
            for _ in 0..len {
                let b = out[out.len() - dist];
                out.push(b);
            }
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_codec_measurement(
    codec_name: &str,
    input: &[u8],
    payload_bytes: usize,
    offline_prior_bytes: usize,
    include_prior_in_total: bool,
    online_weights_transferred: bool,
    decoder_relearned_online: bool,
    decoded: &[u8],
) -> Result<CodecMeasurement, String> {
    if offline_prior_bytes > 0 && !include_prior_in_total {
        return Err(
            "omitting required offline prior/decoder bytes from total size is forbidden".into(),
        );
    }
    let total_bytes = payload_bytes + offline_prior_bytes;
    let input_sha = sha256_hex(input);
    let decoded_sha = sha256_hex(decoded);
    let (bpb, ratio) = if input.is_empty() {
        (None, None)
    } else {
        let b = ((total_bytes as f64) * 8.0) / (input.len() as f64);
        let r = (total_bytes as f64) / (input.len() as f64);
        if !b.is_finite() || !r.is_finite() {
            return Err("non-finite compression metric".into());
        }
        (Some(b), Some(r))
    };
    Ok(CodecMeasurement {
        codec_name: codec_name.into(),
        input_bytes: input.len(),
        payload_bytes,
        offline_prior_bytes,
        total_bytes,
        bits_per_byte: bpb,
        compression_ratio: ratio,
        online_weights_transferred,
        decoder_relearned_online,
        roundtrip_sha256_match: input == decoded && input_sha == decoded_sha,
        decoded_sha256: decoded_sha,
    })
}

pub fn validate_bounded_prefix_request(
    requested_bytes: usize,
    max_bounded_prefix: usize,
) -> OutcomeStatus {
    if requested_bytes > max_bounded_prefix {
        OutcomeStatus::Unsupported
    } else {
        OutcomeStatus::Accepted
    }
}

pub fn validate_pinned_commit(commit: &str) -> OutcomeStatus {
    if commit == UPSTREAM_COMMIT {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

pub fn validate_no_author_rank_gate(imports_enwik9_rank_as_gate: bool) -> OutcomeStatus {
    if imports_enwik9_rank_as_gate {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

/// Deterministic small structured Wikipedia-style text prefix (1536 bytes).
pub fn pinned_small_text_prefix() -> Vec<u8> {
    let paragraphs = [
        "== Inference Lab Compression Fixture ==\n",
        "Online learning neural compression updates predictor weights during encoding and decoding.\n",
        "Because the decoder repeats the exact deterministic update trajectory on each decoded byte, ",
        "the online-adapted weights never need to be transmitted inside the compressed payload.\n",
        "Only the compact quantized offline prior and the arithmetic bitstream are required for exact reconstruction.\n",
        "Conventional dictionary compressors such as LZ77 exploit exact repeated substrings in a sliding window, ",
        "while adaptive statistical and neural predictors adapt symbol probabilities dynamically across context.\n",
        "Every executed arm verifies SHA-256 bit-exact round-trip parity and counts prior plus payload bytes.\n",
    ];
    let mut buf = Vec::with_capacity(2048);
    for repeat in 0..3 {
        buf.extend_from_slice(format!("[section-{repeat:02}] ").as_bytes());
        for p in &paragraphs {
            buf.extend_from_slice(p.as_bytes());
        }
    }
    buf
}

pub fn run_default_harness() -> Result<TufazipOnlineCodecReport, String> {
    let input = pinned_small_text_prefix();
    let input_sha256 = DigestRef::sha256(&input);
    let prior = default_quantized_offline_prior();

    let artifact = encode_online_range(&input, &prior)?;
    let online_decoded = decode_online_range(&artifact, true)?;
    let online_codec = compute_codec_measurement(
        "miniature_online_fixed_point_range",
        &input,
        artifact.payload_bytes.len(),
        artifact.offline_prior_bytes.len(),
        true,
        artifact.online_weights_transferred,
        true,
        &online_decoded,
    )?;
    if !online_codec.roundtrip_sha256_match {
        return Err("online codec failed bit-exact SHA-256 lossless round-trip".into());
    }

    let lz_encoded = encode_lz77_baseline(&input);
    let lz_decoded = decode_lz77_baseline(&lz_encoded)?;
    let conventional_lz77 = compute_codec_measurement(
        "conventional_lz77_baseline",
        &input,
        lz_encoded.len(),
        0,
        true,
        false,
        false,
        &lz_decoded,
    )?;
    if !conventional_lz77.roundtrip_sha256_match {
        return Err(
            "conventional LZ77 baseline failed bit-exact SHA-256 lossless round-trip".into(),
        );
    }

    let mut negative_controls = Vec::new();

    // NC1: decoder skipping online weight updates MUST fail lossless round-trip
    {
        let stale_decoded = decode_online_range(&artifact, false)?;
        let stale_match = stale_decoded == input && sha256_hex(&stale_decoded) == input_sha256.hex;
        negative_controls.push(NegativeControl {
            case_id: "nc1_decoder_skipping_online_updates_fails_roundtrip".into(),
            outcome: if !stale_match {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "disabling decoder-side online weight relearning breaks bit-exact reconstruction (finding, not silent pass)"
                .into(),
        });
    }

    // NC2: flipping 1 bit in compressed payload fails SHA-256 lossless round-trip
    {
        let mut corrupted = artifact.clone();
        let mid_idx = corrupted.payload_bytes.len() / 2;
        if let Some(mid) = corrupted.payload_bytes.get_mut(mid_idx) {
            *mid ^= 0x01;
        }
        let corrupt_decoded = decode_online_range(&corrupted, true)?;
        let corrupt_match =
            corrupt_decoded == input && sha256_hex(&corrupt_decoded) == input_sha256.hex;
        negative_controls.push(NegativeControl {
            case_id: "nc2_single_bit_payload_corruption_detected".into(),
            outcome: if !corrupt_match {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "flipping one payload bit breaks SHA-256 round-trip verification".into(),
        });
    }

    // NC3: omitting offline prior bytes from total reported size rejected
    {
        let outcome = if compute_codec_measurement(
            "missing_prior",
            &input,
            artifact.payload_bytes.len(),
            artifact.offline_prior_bytes.len(),
            false,
            false,
            true,
            &online_decoded,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc3_omitting_prior_bytes_rejected".into(),
            outcome,
            detail: "total size must include required offline prior/decoder bytes".into(),
        });
    }

    // NC4: requesting full 1 GB enwik9 run as first arm refused
    negative_controls.push(NegativeControl {
        case_id: "nc4_full_enwik9_first_arm_refused".into(),
        outcome: validate_bounded_prefix_request(1_000_000_000, 1_048_576),
        detail: "full 1 GB enwik9 job is refused by bounded-prefix policy".into(),
    });

    // NC5: non-2xB200 / non-CUDA-13.0 host triggers clean unavailable_hardware_runtime gate
    {
        let spark_host = HostHardwareProfile {
            os_arch: "linux-aarch64".into(),
            gpu_model: "NVIDIA GB10".into(),
            visible_gpu_count: 1,
            cuda_version: "12.8".into(),
        };
        negative_controls.push(NegativeControl {
            case_id: "nc5_non_2xb200_hardware_gate".into(),
            outcome: check_tufazip_b200_hardware_gate(&spark_host),
            detail: "official tufazip decoder requires Linux x86-64 + 2x NVIDIA B200 + CUDA 13.0"
                .into(),
        });
    }

    // NC6: unpinned tufazip commit rejected
    negative_controls.push(NegativeControl {
        case_id: "nc6_unpinned_tufazip_commit_rejected".into(),
        outcome: validate_pinned_commit("HEAD"),
        detail: "floating commit 'HEAD' is rejected; must pin exact tufazip SHA".into(),
    });

    // NC7: importing author's 105,388,353-byte / 2nd-place / 631,548s claims as pass gate rejected
    negative_controls.push(NegativeControl {
        case_id: "nc7_author_enwik9_rank_not_pass_gate".into(),
        outcome: validate_no_author_rank_gate(true),
        detail: "author's enwik9 total, rank, and B200 timings are unverified related work, never pass gates"
            .into(),
    });

    // NC8: zero-length input yields None bits_per_byte & ratio (no div-by-zero)
    {
        let empty_m = compute_codec_measurement("empty", b"", 0, 0, true, false, false, b"")?;
        negative_controls.push(NegativeControl {
            case_id: "nc8_empty_input_zero_divisor_guarded".into(),
            outcome: if empty_m.bits_per_byte.is_none() && empty_m.compression_ratio.is_none() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "empty input yields None bpb and compression ratio (no division by zero)"
                .into(),
        });
    }

    Ok(TufazipOnlineCodecReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Linux x86-64 + 2x NVIDIA B200 GPUs + CUDA 13.0 for official Tufalabs/tufazip enwik9-v2.bin",
            "Miniature fixed-point online-learning range codec (zero transferred online weights, bit-exact SHA-256 round-trip) and LZ77 baseline verified on CPU; official 2x B200 CUDA 13.0 tufazip runtime did not execute locally",
        ),
        synthetic_contract_fixture: true,
        default_enabled: false,
        enwik9_full_run_attempted: false,
        input_prefix_bytes: input.len(),
        input_sha256,
        online_codec,
        conventional_lz77,
        negative_controls,
    })
}

fn codec_json(m: &CodecMeasurement) -> String {
    format!(
        "{{\"codec_name\":{},\"input_bytes\":{},\"payload_bytes\":{},\"offline_prior_bytes\":{},\"total_bytes\":{},\"bits_per_byte\":{},\"compression_ratio\":{},\"online_weights_transferred\":{},\"decoder_relearned_online\":{},\"roundtrip_sha256_match\":{},\"decoded_sha256\":{}}}",
        quote(&m.codec_name),
        m.input_bytes,
        m.payload_bytes,
        m.offline_prior_bytes,
        m.total_bytes,
        opt_f64(m.bits_per_byte),
        opt_f64(m.compression_ratio),
        m.online_weights_transferred,
        m.decoder_relearned_online,
        m.roundtrip_sha256_match,
        quote(&m.decoded_sha256),
    )
}

pub fn inputs_json(report: &TufazipOnlineCodecReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.tufazip-online-codec.inputs.v1\",\n",
            "  \"issue\": 54,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"enwik9_full_run_attempted\": {},\n",
            "  \"upstream_repo\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"upstream_license\": {},\n",
            "  \"source_bookmark\": {},\n",
            "  \"input_prefix_bytes\": {},\n",
            "  \"input_sha256\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.enwik9_full_run_attempted,
        quote(UPSTREAM_REPO),
        quote(UPSTREAM_COMMIT),
        quote(UPSTREAM_LICENSE),
        quote(SOURCE_BOOKMARK),
        report.input_prefix_bytes,
        report.input_sha256.to_json(),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &TufazipOnlineCodecReport) -> String {
    let controls: Vec<String> = report
        .negative_controls
        .iter()
        .map(|c| {
            format!(
                "{{\"case_id\":{},\"outcome\":{},\"detail\":{}}}",
                quote(&c.case_id),
                quote(c.outcome.as_str()),
                quote(&c.detail),
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.tufazip-online-codec.results.v1\",\n",
            "  \"issue\": 54,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"enwik9_full_run_attempted\": {},\n",
            "  \"input_prefix_bytes\": {},\n",
            "  \"input_sha256\": {},\n",
            "  \"online_codec\": {},\n",
            "  \"conventional_lz77\": {},\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.default_enabled,
        report.enwik9_full_run_attempted,
        report.input_prefix_bytes,
        report.input_sha256.to_json(),
        codec_json(&report.online_codec),
        codec_json(&report.conventional_lz77),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &TufazipOnlineCodecReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #54: Bounded tufazip-Style Online Compressor (Small Prefix & B200 Gate)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Pin: `{}@{}` (`{}`) | `default_enabled: {}` | `enwik9_full_run_attempted: {}`",
        UPSTREAM_REPO,
        UPSTREAM_COMMIT,
        UPSTREAM_LICENSE,
        report.default_enabled,
        report.enwik9_full_run_attempted
    )
    .unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`synthetic_contract_fixture: {}`)",
        report.execution_class.as_str(),
        report.synthetic_contract_fixture
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Lossless Round-Trip Comparison (`input_bytes={}`, `sha256={}`)",
        report.input_prefix_bytes, report.input_sha256.hex
    )
    .unwrap();
    for m in [&report.online_codec, &report.conventional_lz77] {
        writeln!(
            out,
            "- `{}`: `payload={}`, `prior={}`, `total={}`, `bpb={:.3}`, `online_weights_transferred={}`, `sha256_match={}`",
            m.codec_name,
            m.payload_bytes,
            m.offline_prior_bytes,
            m.total_bytes,
            m.bits_per_byte.unwrap_or(0.0),
            m.online_weights_transferred,
            m.roundtrip_sha256_match,
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Negative Controls ({} / {} rejected)",
        report
            .negative_controls
            .iter()
            .filter(|c| c.outcome != OutcomeStatus::Accepted)
            .count(),
        report.negative_controls.len()
    )
    .unwrap();
    for c in &report.negative_controls {
        writeln!(
            out,
            "- `{}` -> `{}` ({})",
            c.case_id,
            c.outcome.as_str(),
            c.detail
        )
        .unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn online_range_codec_compresses_and_roundtrips_bit_exactly_without_transferring_weights() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert!(!report.enwik9_full_run_attempted);
        assert!(report.online_codec.roundtrip_sha256_match);
        assert!(!report.online_codec.online_weights_transferred);
        assert!(report.online_codec.decoder_relearned_online);
        assert!(report.online_codec.total_bytes < report.input_prefix_bytes);
        assert!(report.conventional_lz77.roundtrip_sha256_match);
        assert!(report.conventional_lz77.total_bytes < report.input_prefix_bytes);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }

    #[test]
    fn skipping_decoder_online_updates_fails_roundtrip() {
        let input = pinned_small_text_prefix();
        let prior = default_quantized_offline_prior();
        let art = encode_online_range(&input, &prior).unwrap();
        let ok = decode_online_range(&art, true).unwrap();
        let bad = decode_online_range(&art, false).unwrap();
        assert_eq!(ok, input);
        assert_ne!(bad, input);
    }
}
