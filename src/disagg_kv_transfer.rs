//! Issue #62: Measure default-off disaggregated Spark-CUDA prefill to Apple-Metal decode
//! prefix-cache transfer over 10GbE against single-host prefill.
//!
//! Unverified related-work reference (strictly non-gating):
//! - X post (2026-09-07, `@volatilemarkts`): `https://x.com/volatilemarkts/status/2096797538888057213`
//! - Authors' unverified claims: CUDA DGX Sparks compute finished KV prefix caches using the
//!   decoder's weights and stream ~10 KB/token over plain 10GbE into Apple Silicon Mac Studios
//!   for Metal decode (claiming 241K DeepSeek-V4-Flash cold TTFT 12 min -> 3 min, 19 s warm).
//! - Complements `#5` (KV migration vs recomputation) and `#42` (split-model inference).
//!
//! Provides:
//! - Bounded default-off KV prefix-cache wire framing codec (`disagg-kv-v1`): versioned header
//!   binding model/effective-weight/tokenizer/template digests, layer/head/dtype/endianness
//!   metadata, chunked payload framing, and SHA-256 payload digest verification.
//! - Independent miniature 2-layer causal attention model verifying exact token and logit parity
//!   (`<= 1e-12`) between single-host prefill+decode and disaggregated CUDA-prefill -> 10GbE wire
//!   serialization/deserialization -> Metal-decode on frozen prompts.
//! - Explicit wire bytes/token, cold and warm TTFT, 10GbE transfer latency, and end-to-end
//!   latency comparison against single-host prefill baselines.
//! - 8 adversarial negative controls (corrupted payload checksum, mismatched weight digest,
//!   truncated wire packet, dtype/endianness mismatch, wrong prompt prefix length, non-finite KV
//!   entry, zero-bandwidth divisor guard, and silent CPU fallback refusal).

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote, sha256_hex};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/volatilemarkts/status/2096797538888057213";
pub const WIRE_FORMAT_VERSION: u16 = 1;
pub const WIRE_MAGIC: [u8; 8] = *b"ILABKV01";
pub const COMPLEMENTED_ISSUES: [&str; 2] = ["inference-lab#5", "inference-lab#42"];

pub const N_LAYERS: usize = 2;
pub const D_MODEL: usize = 8;
pub const N_HEADS: usize = 2;
pub const HEAD_DIM: usize = 4;
pub const VOCAB_SIZE: usize = 16;
pub const PARITY_TOL: f64 = 1.0e-12;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("length mismatch in max_abs_diff".into());
    }
    let mut best = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = (check_finite(x, "diff_a")? - check_finite(y, "diff_b")?).abs();
        if dx > best {
            best = dx;
        }
    }
    Ok(best)
}

fn argmax_checked(logits: &[f64]) -> Result<usize, String> {
    if logits.is_empty() {
        return Err("empty logits".into());
    }
    let mut best_idx = 0usize;
    let mut best_val = check_finite(logits[0], "argmax[0]")?;
    for (idx, &v) in logits.iter().enumerate().skip(1) {
        let fv = check_finite(v, "argmax")?;
        if fv > best_val {
            best_val = fv;
            best_idx = idx;
        }
    }
    Ok(best_idx)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvWireHeader {
    pub version: u16,
    pub model_weight_digest: String,
    pub tokenizer_digest: String,
    pub template_digest: String,
    pub num_layers: u16,
    pub num_heads: u16,
    pub head_dim: u16,
    pub prefix_tokens: u32,
    pub dtype_tag: String,
    pub endianness_tag: String,
    pub chunk_bytes: u32,
    pub payload_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvWirePacket {
    pub header: KvWireHeader,
    pub payload_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct MiniAttentionLayer {
    pub wq: [[f64; D_MODEL]; D_MODEL],
    pub wk: [[f64; D_MODEL]; D_MODEL],
    pub wv: [[f64; D_MODEL]; D_MODEL],
    pub wo: [[f64; D_MODEL]; D_MODEL],
}

#[derive(Debug, Clone)]
pub struct MiniPrefixModel {
    pub embed: [[f64; D_MODEL]; VOCAB_SIZE],
    pub layers: Vec<MiniAttentionLayer>,
    pub lm_head: [[f64; D_MODEL]; VOCAB_SIZE],
    pub weight_digest: String,
    pub tokenizer_digest: String,
    pub template_digest: String,
}

impl MiniPrefixModel {
    pub fn build_deterministic() -> Self {
        let mut embed = [[0.0_f64; D_MODEL]; VOCAB_SIZE];
        let mut lm_head = [[0.0_f64; D_MODEL]; VOCAB_SIZE];
        let mut digest_buf = Vec::new();
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                embed[v][d] = det_weight(6201, v * D_MODEL + d, 0.50);
                lm_head[v][d] = det_weight(6202, v * D_MODEL + d, 0.45);
                digest_buf.extend_from_slice(&embed[v][d].to_le_bytes());
                digest_buf.extend_from_slice(&lm_head[v][d].to_le_bytes());
            }
        }
        let mut layers = Vec::with_capacity(N_LAYERS);
        for l in 0..N_LAYERS {
            let mut wq = [[0.0_f64; D_MODEL]; D_MODEL];
            let mut wk = [[0.0_f64; D_MODEL]; D_MODEL];
            let mut wv = [[0.0_f64; D_MODEL]; D_MODEL];
            let mut wo = [[0.0_f64; D_MODEL]; D_MODEL];
            for r in 0..D_MODEL {
                for c in 0..D_MODEL {
                    wq[r][c] = det_weight(6210 + l * 31, r * D_MODEL + c, 0.25);
                    wk[r][c] = det_weight(6211 + l * 31, r * D_MODEL + c, 0.25);
                    wv[r][c] = det_weight(6212 + l * 31, r * D_MODEL + c, 0.25);
                    wo[r][c] = det_weight(6213 + l * 31, r * D_MODEL + c, 0.22);
                    digest_buf.extend_from_slice(&wq[r][c].to_le_bytes());
                    digest_buf.extend_from_slice(&wk[r][c].to_le_bytes());
                    digest_buf.extend_from_slice(&wv[r][c].to_le_bytes());
                    digest_buf.extend_from_slice(&wo[r][c].to_le_bytes());
                }
            }
            layers.push(MiniAttentionLayer { wq, wk, wv, wo });
        }
        Self {
            embed,
            layers,
            lm_head,
            weight_digest: sha256_hex(&digest_buf),
            tokenizer_digest: sha256_hex(b"disagg-kv-tokenizer-v1"),
            template_digest: sha256_hex(b"disagg-kv-chat-template-v1"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefixKvCache {
    pub keys: Vec<Vec<[f64; D_MODEL]>>,
    pub values: Vec<Vec<[f64; D_MODEL]>>,
}

impl PrefixKvCache {
    pub fn empty(num_layers: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
        }
    }

    pub fn prefix_len(&self) -> usize {
        self.keys.first().map_or(0, Vec::len)
    }
}

fn matvec_dd(w: &[[f64; D_MODEL]; D_MODEL], x: &[f64; D_MODEL]) -> [f64; D_MODEL] {
    let mut out = [0.0_f64; D_MODEL];
    for (row, dst) in w.iter().zip(out.iter_mut()) {
        let mut s = 0.0_f64;
        for (&w_rc, &x_c) in row.iter().zip(x.iter()) {
            s += w_rc * x_c;
        }
        *dst = s;
    }
    out
}

fn rms_norm(x: &[f64; D_MODEL]) -> Result<[f64; D_MODEL], String> {
    let mut sq = 0.0_f64;
    for &v in x {
        let fv = check_finite(v, "rmsnorm")?;
        sq += fv * fv;
    }
    let denom = (sq / (D_MODEL as f64) + 1.0e-6).sqrt();
    if !denom.is_finite() || denom == 0.0 {
        return Err("invalid rmsnorm denominator".into());
    }
    let mut out = [0.0_f64; D_MODEL];
    for i in 0..D_MODEL {
        out[i] = x[i] / denom;
    }
    Ok(out)
}

fn step_token(
    model: &MiniPrefixModel,
    token: usize,
    kv: &mut PrefixKvCache,
) -> Result<[f64; VOCAB_SIZE], String> {
    if token >= VOCAB_SIZE {
        return Err(format!("token id {token} out of vocab range {VOCAB_SIZE}"));
    }
    let mut h = model.embed[token];
    let inv_scale = 1.0_f64 / (HEAD_DIM as f64).sqrt();
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let norm_h = rms_norm(&h)?;
        let q = matvec_dd(&layer.wq, &norm_h);
        let k = matvec_dd(&layer.wk, &norm_h);
        let v = matvec_dd(&layer.wv, &norm_h);
        kv.keys[layer_idx].push(k);
        kv.values[layer_idx].push(v);

        let seq_len = kv.keys[layer_idx].len();
        let mut attn_concat = [0.0_f64; D_MODEL];
        for head in 0..N_HEADS {
            let offset = head * HEAD_DIM;
            let mut scores = Vec::with_capacity(seq_len);
            let mut max_score = f64::NEG_INFINITY;
            for pos in 0..seq_len {
                let mut dot = 0.0_f64;
                for d in 0..HEAD_DIM {
                    dot += q[offset + d] * kv.keys[layer_idx][pos][offset + d];
                }
                let s = check_finite(dot * inv_scale, "attn_score")?;
                if s > max_score {
                    max_score = s;
                }
                scores.push(s);
            }
            let mut weights = Vec::with_capacity(seq_len);
            let mut sum_w = 0.0_f64;
            for &s in &scores {
                let w = (s - max_score).exp();
                sum_w += w;
                weights.push(w);
            }
            if !sum_w.is_finite() || sum_w == 0.0 {
                return Err("invalid softmax sum".into());
            }
            for d in 0..HEAD_DIM {
                let mut acc = 0.0_f64;
                for (w, v_row) in weights.iter().zip(kv.values[layer_idx].iter()) {
                    acc += (*w / sum_w) * v_row[offset + d];
                }
                attn_concat[offset + d] = check_finite(acc, "attn_val")?;
            }
        }
        let proj = matvec_dd(&layer.wo, &attn_concat);
        for d in 0..D_MODEL {
            h[d] = check_finite(h[d] + proj[d], "residual")?;
        }
    }
    let final_h = rms_norm(&h)?;
    let mut logits = [0.0_f64; VOCAB_SIZE];
    for (v_idx, row) in model.lm_head.iter().enumerate() {
        let mut s = 0.0_f64;
        for d in 0..D_MODEL {
            s += row[d] * final_h[d];
        }
        logits[v_idx] = check_finite(s, "logits")?;
    }
    Ok(logits)
}

/// Serialize finished prefix KV cache into the bounded `disagg-kv-v1` 10GbE wire packet.
pub fn encode_kv_wire_packet(
    model: &MiniPrefixModel,
    kv: &PrefixKvCache,
    chunk_bytes: u32,
) -> Result<KvWirePacket, String> {
    if kv.keys.len() != model.layers.len() || kv.values.len() != model.layers.len() {
        return Err("KV layer count mismatch".into());
    }
    if chunk_bytes == 0 {
        return Err("chunk_bytes must be > 0".into());
    }
    let prefix_len = kv.prefix_len();
    let mut payload = Vec::with_capacity(WIRE_MAGIC.len() + N_LAYERS * prefix_len * D_MODEL * 16);
    payload.extend_from_slice(&WIRE_MAGIC);
    for l in 0..kv.keys.len() {
        if kv.keys[l].len() != prefix_len || kv.values[l].len() != prefix_len {
            return Err("ragged KV prefix lengths across layers".into());
        }
        for pos in 0..prefix_len {
            for &val in &kv.keys[l][pos] {
                let fv = check_finite(val, "encode_k")?;
                payload.extend_from_slice(&fv.to_le_bytes());
            }
            for &val in &kv.values[l][pos] {
                let fv = check_finite(val, "encode_v")?;
                payload.extend_from_slice(&fv.to_le_bytes());
            }
        }
    }
    let payload_sha256 = sha256_hex(&payload);
    Ok(KvWirePacket {
        header: KvWireHeader {
            version: WIRE_FORMAT_VERSION,
            model_weight_digest: model.weight_digest.clone(),
            tokenizer_digest: model.tokenizer_digest.clone(),
            template_digest: model.template_digest.clone(),
            num_layers: model.layers.len() as u16,
            num_heads: N_HEADS as u16,
            head_dim: HEAD_DIM as u16,
            prefix_tokens: prefix_len as u32,
            dtype_tag: "f64_le".into(),
            endianness_tag: "little".into(),
            chunk_bytes,
            payload_sha256,
        },
        payload_bytes: payload,
    })
}

/// Validate wire header/digest/layout and reconstruct prefix KV cache on the Metal decode host.
pub fn decode_kv_wire_packet(
    model: &MiniPrefixModel,
    packet: &KvWirePacket,
    expected_prefix_tokens: usize,
) -> Result<PrefixKvCache, OutcomeStatus> {
    if packet.header.version != WIRE_FORMAT_VERSION {
        return Err(OutcomeStatus::Unsupported);
    }
    if packet.header.model_weight_digest != model.weight_digest
        || packet.header.tokenizer_digest != model.tokenizer_digest
        || packet.header.template_digest != model.template_digest
    {
        return Err(OutcomeStatus::Invalid);
    }
    if packet.header.dtype_tag != "f64_le" || packet.header.endianness_tag != "little" {
        return Err(OutcomeStatus::Unsupported);
    }
    if packet.header.num_layers as usize != model.layers.len()
        || packet.header.num_heads as usize != N_HEADS
        || packet.header.head_dim as usize != HEAD_DIM
    {
        return Err(OutcomeStatus::Invalid);
    }
    if packet.header.prefix_tokens as usize != expected_prefix_tokens {
        return Err(OutcomeStatus::Invalid);
    }
    if sha256_hex(&packet.payload_bytes) != packet.header.payload_sha256 {
        return Err(OutcomeStatus::Invalid);
    }
    let expected_bytes =
        WIRE_MAGIC.len() + model.layers.len() * expected_prefix_tokens * D_MODEL * 16;
    if packet.payload_bytes.len() != expected_bytes
        || !packet.payload_bytes.starts_with(&WIRE_MAGIC)
    {
        return Err(OutcomeStatus::Invalid);
    }

    let mut kv = PrefixKvCache::empty(model.layers.len());
    let mut cursor = WIRE_MAGIC.len();
    for l in 0..model.layers.len() {
        for _ in 0..expected_prefix_tokens {
            let mut k = [0.0_f64; D_MODEL];
            for slot in &mut k {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&packet.payload_bytes[cursor..cursor + 8]);
                cursor += 8;
                let v = f64::from_le_bytes(bytes);
                if !v.is_finite() {
                    return Err(OutcomeStatus::Invalid);
                }
                *slot = v;
            }
            let mut v_vec = [0.0_f64; D_MODEL];
            for slot in &mut v_vec {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&packet.payload_bytes[cursor..cursor + 8]);
                cursor += 8;
                let v = f64::from_le_bytes(bytes);
                if !v.is_finite() {
                    return Err(OutcomeStatus::Invalid);
                }
                *slot = v;
            }
            kv.keys[l].push(k);
            kv.values[l].push(v_vec);
        }
    }
    Ok(kv)
}

/// Qualify disaggregated CUDA-prefill + Metal-decode hardware target; refuses silent CPU fallback.
pub fn qualify_disagg_hosts(
    cuda_prefill_available: bool,
    metal_decode_available: bool,
    link_gbps: f64,
    allow_cpu_fallback: bool,
) -> OutcomeStatus {
    if allow_cpu_fallback {
        return OutcomeStatus::Invalid;
    }
    if !link_gbps.is_finite() || link_gbps <= 0.0 {
        return OutcomeStatus::Invalid;
    }
    if !cuda_prefill_available || !metal_decode_available {
        return OutcomeStatus::UnavailableHardwareRuntime;
    }
    OutcomeStatus::Accepted
}

#[derive(Debug, Clone)]
pub struct PromptParityResult {
    pub prompt_id: String,
    pub prefix_tokens: usize,
    pub decode_steps: usize,
    pub wire_payload_bytes: usize,
    pub wire_bytes_per_token: Option<f64>,
    pub token_exact_match: bool,
    pub max_logit_abs_err: f64,
    pub single_host_tokens: Vec<usize>,
    pub disagg_tokens: Vec<usize>,
    pub outcome: OutcomeStatus,
}

#[derive(Debug, Clone)]
pub struct TransferLatencyRow {
    pub profile_id: String,
    pub context_tokens: usize,
    pub wire_bytes_per_token: usize,
    pub link_gbps: f64,
    pub single_host_cold_ttft_s: f64,
    pub disagg_cold_prefill_s: f64,
    pub disagg_wire_transfer_s: Option<f64>,
    pub disagg_cold_ttft_s: Option<f64>,
    pub disagg_warm_ttft_s: Option<f64>,
    pub decode_phase_s: f64,
    pub single_host_e2e_s: f64,
    pub disagg_cold_e2e_s: Option<f64>,
    pub outcome: OutcomeStatus,
}

#[allow(clippy::too_many_arguments)]
pub fn compute_transfer_latency(
    profile_id: &str,
    context_tokens: usize,
    wire_bytes_per_token: usize,
    link_gbps: f64,
    single_host_cold_ttft_s: f64,
    disagg_cold_prefill_s: f64,
    warm_cache_hit_ratio: f64,
    decode_phase_s: f64,
) -> Result<TransferLatencyRow, String> {
    check_finite(link_gbps, "link_gbps")?;
    check_finite(single_host_cold_ttft_s, "single_host_cold_ttft_s")?;
    check_finite(disagg_cold_prefill_s, "disagg_cold_prefill_s")?;
    check_finite(warm_cache_hit_ratio, "warm_cache_hit_ratio")?;
    check_finite(decode_phase_s, "decode_phase_s")?;
    if link_gbps <= 0.0
        || !(0.0..=1.0).contains(&warm_cache_hit_ratio)
        || single_host_cold_ttft_s <= 0.0
        || disagg_cold_prefill_s <= 0.0
        || decode_phase_s < 0.0
    {
        return Ok(TransferLatencyRow {
            profile_id: profile_id.into(),
            context_tokens,
            wire_bytes_per_token,
            link_gbps,
            single_host_cold_ttft_s,
            disagg_cold_prefill_s,
            disagg_wire_transfer_s: None,
            disagg_cold_ttft_s: None,
            disagg_warm_ttft_s: None,
            decode_phase_s,
            single_host_e2e_s: single_host_cold_ttft_s + decode_phase_s,
            disagg_cold_e2e_s: None,
            outcome: OutcomeStatus::Invalid,
        });
    }
    let total_bits = (context_tokens as f64) * (wire_bytes_per_token as f64) * 8.0;
    let effective_bps = link_gbps * 1.0e9 * 0.90; // 10% protocol/framing overhead
    let wire_s = total_bits / effective_bps;
    let cold_ttft_s = disagg_cold_prefill_s + wire_s;
    let uncached_frac = 1.0 - warm_cache_hit_ratio;
    let warm_ttft_s = disagg_cold_prefill_s * uncached_frac + wire_s;
    Ok(TransferLatencyRow {
        profile_id: profile_id.into(),
        context_tokens,
        wire_bytes_per_token,
        link_gbps,
        single_host_cold_ttft_s,
        disagg_cold_prefill_s,
        disagg_wire_transfer_s: Some(wire_s),
        disagg_cold_ttft_s: Some(cold_ttft_s),
        disagg_warm_ttft_s: Some(warm_ttft_s),
        decode_phase_s,
        single_host_e2e_s: single_host_cold_ttft_s + decode_phase_s,
        disagg_cold_e2e_s: Some(cold_ttft_s + decode_phase_s),
        outcome: OutcomeStatus::Accepted,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct DisaggKvTransferReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub wire_format_version: u16,
    pub prompt_parity: Vec<PromptParityResult>,
    pub latency_profiles: Vec<TransferLatencyRow>,
    pub negative_controls: Vec<NegativeControlResult>,
}

fn eval_prompt_parity(
    model: &MiniPrefixModel,
    prompt_id: &str,
    prefix: &[usize],
    decode_steps: usize,
) -> Result<PromptParityResult, String> {
    if prefix.is_empty() {
        return Err("prefix must be non-empty".into());
    }
    // Path 1: single-host continuous prefill + decode
    let mut single_kv = PrefixKvCache::empty(model.layers.len());
    let mut last_single_logits = [0.0_f64; VOCAB_SIZE];
    for &tok in prefix {
        last_single_logits = step_token(model, tok, &mut single_kv)?;
    }
    let mut single_tokens = Vec::with_capacity(decode_steps);
    let mut cur_tok = argmax_checked(&last_single_logits)?;
    single_tokens.push(cur_tok);
    let mut single_logit_trace = vec![last_single_logits];
    for _ in 1..decode_steps {
        let logits = step_token(model, cur_tok, &mut single_kv)?;
        cur_tok = argmax_checked(&logits)?;
        single_tokens.push(cur_tok);
        single_logit_trace.push(logits);
    }

    // Path 2: CUDA prefill -> 10GbE wire serialize -> Metal deserialize -> decode
    let mut prefill_kv = PrefixKvCache::empty(model.layers.len());
    let mut last_prefill_logits = [0.0_f64; VOCAB_SIZE];
    for &tok in prefix {
        last_prefill_logits = step_token(model, tok, &mut prefill_kv)?;
    }
    let packet = encode_kv_wire_packet(model, &prefill_kv, 4096)?;
    let mut decode_kv = decode_kv_wire_packet(model, &packet, prefix.len())
        .map_err(|st| format!("wire decode failed unexpectedly: {}", st.as_str()))?;
    let mut disagg_tokens = Vec::with_capacity(decode_steps);
    let mut d_tok = argmax_checked(&last_prefill_logits)?;
    disagg_tokens.push(d_tok);
    let mut disagg_logit_trace = vec![last_prefill_logits];
    for _ in 1..decode_steps {
        let logits = step_token(model, d_tok, &mut decode_kv)?;
        d_tok = argmax_checked(&logits)?;
        disagg_tokens.push(d_tok);
        disagg_logit_trace.push(logits);
    }

    let mut max_err = 0.0_f64;
    for (s_log, d_log) in single_logit_trace.iter().zip(disagg_logit_trace.iter()) {
        let err = max_abs_diff(s_log, d_log)?;
        if err > max_err {
            max_err = err;
        }
    }
    let exact = single_tokens == disagg_tokens && max_err <= PARITY_TOL;
    let bytes_per_tok = Some((packet.payload_bytes.len() as f64) / (prefix.len() as f64));
    Ok(PromptParityResult {
        prompt_id: prompt_id.into(),
        prefix_tokens: prefix.len(),
        decode_steps,
        wire_payload_bytes: packet.payload_bytes.len(),
        wire_bytes_per_token: bytes_per_tok,
        token_exact_match: exact,
        max_logit_abs_err: max_err,
        single_host_tokens: single_tokens,
        disagg_tokens,
        outcome: if exact {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
    })
}

pub fn run_default_harness() -> Result<DisaggKvTransferReport, String> {
    let model = MiniPrefixModel::build_deterministic();
    let prompt_parity = vec![
        eval_prompt_parity(&model, "frozen_short_8tok", &[1, 3, 5, 7, 2, 4, 6, 8], 6)?,
        eval_prompt_parity(
            &model,
            "frozen_medium_16tok",
            &[0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15],
            6,
        )?,
        eval_prompt_parity(
            &model,
            "frozen_repeat_12tok",
            &[4, 9, 4, 9, 11, 2, 4, 9, 4, 9, 11, 2],
            6,
        )?,
    ];

    let latency_profiles = vec![
        compute_transfer_latency(
            "ctx_32k_10gbe",
            32_768,
            10_240,
            10.0,
            92.0,
            22.0,
            0.90,
            14.0,
        )?,
        compute_transfer_latency(
            "ctx_128k_10gbe",
            131_072,
            10_240,
            10.0,
            380.0,
            90.0,
            0.90,
            18.0,
        )?,
        compute_transfer_latency(
            "ctx_241k_10gbe",
            241_000,
            10_240,
            10.0,
            720.0,
            165.0,
            0.92,
            22.0,
        )?,
    ];

    // Build baseline wire packet for adversarial negative controls
    let mut base_kv = PrefixKvCache::empty(model.layers.len());
    for &tok in &[1usize, 3, 5, 7] {
        let _ = step_token(&model, tok, &mut base_kv)?;
    }
    let valid_pkt = encode_kv_wire_packet(&model, &base_kv, 4096)?;

    let mut nc = Vec::new();
    // NC1: corrupted wire payload byte fails SHA-256 verification
    let mut corrupt = valid_pkt.clone();
    let last_idx = corrupt.payload_bytes.len() - 1;
    corrupt.payload_bytes[last_idx] ^= 0xFF;
    let nc1 = decode_kv_wire_packet(&model, &corrupt, 4).unwrap_err();
    nc.push(NegativeControlResult {
        case_id: "nc1_corrupted_wire_checksum_rejected".into(),
        outcome: nc1,
        detail: "1-bit payload corruption caught by SHA-256 digest verification".into(),
    });

    // NC2: mismatched decoder weight digest rejected
    let mut bad_weight = valid_pkt.clone();
    bad_weight.header.model_weight_digest = sha256_hex(b"different-decoder-weights");
    let nc2 = decode_kv_wire_packet(&model, &bad_weight, 4).unwrap_err();
    nc.push(NegativeControlResult {
        case_id: "nc2_weight_digest_mismatch_rejected".into(),
        outcome: nc2,
        detail: "prefill KV produced with mismatched decoder weight digest refused".into(),
    });

    // NC3: truncated 10GbE packet rejected
    let mut trunc = valid_pkt.clone();
    trunc.payload_bytes.truncate(trunc.payload_bytes.len() - 16);
    trunc.header.payload_sha256 = sha256_hex(&trunc.payload_bytes);
    let nc3 = decode_kv_wire_packet(&model, &trunc, 4).unwrap_err();
    nc.push(NegativeControlResult {
        case_id: "nc3_truncated_wire_payload_rejected".into(),
        outcome: nc3,
        detail: "truncated KV byte stream refused even with recomputed payload digest".into(),
    });

    // NC4: unsupported dtype / endianness rejected
    let mut bad_dtype = valid_pkt.clone();
    bad_dtype.header.dtype_tag = "fp4_custom".into();
    let nc4 = decode_kv_wire_packet(&model, &bad_dtype, 4).unwrap_err();
    nc.push(NegativeControlResult {
        case_id: "nc4_unsupported_dtype_rejected".into(),
        outcome: nc4,
        detail: "unknown wire KV dtype tag refused with unsupported".into(),
    });

    // NC5: prefix length mismatch rejected
    let nc5 = decode_kv_wire_packet(&model, &valid_pkt, 5).unwrap_err();
    nc.push(NegativeControlResult {
        case_id: "nc5_prefix_token_count_mismatch_rejected".into(),
        outcome: nc5,
        detail: "expected 5 prefix tokens vs 4 in wire header refused".into(),
    });

    // NC6: non-finite NaN KV entry refused during encoding
    let mut nan_kv = base_kv.clone();
    nan_kv.keys[0][0][0] = f64::NAN;
    let nc6 = if encode_kv_wire_packet(&model, &nan_kv, 4096).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc6_non_finite_kv_entry_rejected".into(),
        outcome: nc6,
        detail: "NaN KV tensor element refused before wire serialization".into(),
    });

    // NC7: zero link bandwidth guarded against division by zero
    let zero_bw = compute_transfer_latency("bad_bw", 32_768, 10_240, 0.0, 92.0, 22.0, 0.9, 14.0)?;
    nc.push(NegativeControlResult {
        case_id: "nc7_zero_link_bandwidth_guarded".into(),
        outcome: zero_bw.outcome,
        detail: "0 Gbps link returned invalid with wire_transfer_s=None instead of Inf".into(),
    });

    // NC8: silent CPU fallback refused when CUDA/Metal hosts absent
    let nc8 = qualify_disagg_hosts(false, false, 10.0, true);
    nc.push(NegativeControlResult {
        case_id: "nc8_silent_cpu_fallback_refused".into(),
        outcome: nc8,
        detail: "attempt to substitute CPU execution for Spark-CUDA + Apple-Metal refused".into(),
    });

    Ok(DisaggKvTransferReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "DGX Spark CUDA prefill host + Apple Silicon Mac Studio Metal decode host over 10GbE",
            "Versioned KV prefix wire codec, exact single-host vs disaggregated token/logit parity, and 10GbE transfer accounting verified on CPU; physical CUDA/Metal hosts did not execute locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        wire_format_version: WIRE_FORMAT_VERSION,
        prompt_parity,
        latency_profiles,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &DisaggKvTransferReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.disagg-kv-transfer.inputs.v1\",\n",
            "  \"issue\": 62,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"wire_format_version\": {},\n",
            "  \"complements\": [\"inference-lab#5\",\"inference-lab#42\"],\n",
            "  \"upstream_x_post\": {},\n",
            "  \"logit_parity_tolerance\": {},\n",
            "  \"author_unverified_non_gating_notes\": \"10 KB/tok over 10GbE and 241K DeepSeek-V4-Flash 12m->3m cold / 19s warm TTFT claims are unverified X-post figures, not lab targets\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        report.wire_format_version,
        quote(UPSTREAM_X_POST),
        finite_f64(PARITY_TOL),
    )
}

pub fn results_json(report: &DisaggKvTransferReport) -> String {
    let parity_json: Vec<String> = report
        .prompt_parity
        .iter()
        .map(|p| {
            format!(
                "{{\"prompt_id\":{},\"prefix_tokens\":{},\"decode_steps\":{},\"wire_payload_bytes\":{},\"wire_bytes_per_token\":{},\"token_exact_match\":{},\"max_logit_abs_err\":{},\"outcome\":{}}}",
                quote(&p.prompt_id),
                p.prefix_tokens,
                p.decode_steps,
                p.wire_payload_bytes,
                opt_f64(p.wire_bytes_per_token),
                p.token_exact_match,
                finite_f64(p.max_logit_abs_err),
                quote(p.outcome.as_str()),
            )
        })
        .collect();
    let lat_json: Vec<String> = report
        .latency_profiles
        .iter()
        .map(|r| {
            format!(
                "{{\"profile_id\":{},\"context_tokens\":{},\"wire_bytes_per_token\":{},\"link_gbps\":{},\"single_host_cold_ttft_s\":{},\"disagg_cold_prefill_s\":{},\"disagg_wire_transfer_s\":{},\"disagg_cold_ttft_s\":{},\"disagg_warm_ttft_s\":{},\"single_host_e2e_s\":{},\"disagg_cold_e2e_s\":{},\"outcome\":{}}}",
                quote(&r.profile_id),
                r.context_tokens,
                r.wire_bytes_per_token,
                finite_f64(r.link_gbps),
                finite_f64(r.single_host_cold_ttft_s),
                finite_f64(r.disagg_cold_prefill_s),
                opt_f64(r.disagg_wire_transfer_s),
                opt_f64(r.disagg_cold_ttft_s),
                opt_f64(r.disagg_warm_ttft_s),
                finite_f64(r.single_host_e2e_s),
                opt_f64(r.disagg_cold_e2e_s),
                quote(r.outcome.as_str()),
            )
        })
        .collect();
    let nc_json: Vec<String> = report
        .negative_controls
        .iter()
        .map(|c| {
            format!(
                "{{\"case_id\":{},\"outcome\":{},\"detail\":{}}}",
                quote(&c.case_id),
                quote(c.outcome.as_str()),
                quote(&c.detail)
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.disagg-kv-transfer.results.v1\",\n",
            "  \"issue\": 62,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"prompt_parity\": [{}],\n",
            "  \"latency_profiles\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        parity_json.join(","),
        lat_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &DisaggKvTransferReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #62: Disaggregated Spark-CUDA Prefill to Apple-Metal Decode KV Prefix Transfer (10GbE)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, `synthetic_contract_fixture: {}`)",
        report.execution_class.as_str(),
        report.default_enabled,
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
    writeln!(
        out,
        "- Wire contract: `disagg-kv-v{}`",
        report.wire_format_version
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Single-Host vs Disaggregated KV Wire Parity").unwrap();
    for p in &report.prompt_parity {
        writeln!(
            out,
            "- `{}` (`prefix={}`, `steps={}`): `wire_bytes={}`, `exact={}`, `max_logit_err={:.3e}`, `outcome={}`",
            p.prompt_id,
            p.prefix_tokens,
            p.decode_steps,
            p.wire_payload_bytes,
            p.token_exact_match,
            p.max_logit_abs_err,
            p.outcome.as_str()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Synthetic 10GbE Transfer & Cold/Warm TTFT Contract Profiles"
    )
    .unwrap();
    for r in &report.latency_profiles {
        writeln!(
            out,
            "- `{}` (`ctx={}`, `B/tok={}`): `single_cold_ttft_s={:.2}`, `wire_s={:?}`, `disagg_cold_ttft_s={:?}`, `disagg_warm_ttft_s={:?}`, `outcome={}`",
            r.profile_id,
            r.context_tokens,
            r.wire_bytes_per_token,
            r.single_host_cold_ttft_s,
            r.disagg_wire_transfer_s,
            r.disagg_cold_ttft_s,
            r.disagg_warm_ttft_s,
            r.outcome.as_str()
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
    fn default_harness_passes_parity_and_rejects_all_controls() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert_eq!(report.prompt_parity.len(), 3);
        for p in &report.prompt_parity {
            assert!(p.token_exact_match);
            assert!(p.max_logit_abs_err <= PARITY_TOL);
            assert_eq!(p.outcome, OutcomeStatus::Accepted);
        }
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
