//! Issue #64: Measure two-Spark KV sharding and disk-persisted KV checkpoint resume
//! (`GLM-5.3-Flash-DGX-Sparks`) against recomputation, default-off.
//!
//! Pinned upstream reference (observed 2026-11-24, Apache-2.0):
//! - Repository: `https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104`
//! - Commit: `6c55c20da31a02d6d381a268fa42f8a9ebe0c5d2`
//! - X post (2026-11-07, `@MiaAI_lab`): `https://x.com/MiaAI_lab/status/2119016651064529304`
//! - Authors' unverified claims: ~3.1M total sharded KV cache across 2x DGX Spark and resuming a
//!   101K chat with 98K cached tokens with identical output.
//! - Complements `#5` (KV routing/migration vs recomputation) and `#11` (Spark lifecycle).
//!
//! Provides:
//! - Default-off pinned profile for `MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104@6c55c20d`.
//! - Pure-Rust 2-Spark cross-node KV sharding (head-sharded across rank 0 and rank 1) +
//!   deterministic binary disk-persisted KV checkpoint save/resume serializer with header binding
//!   repo commit, model/weight/tokenizer digest, prefix token hash, and per-shard SHA-256.
//! - Exact token and logit parity check (`<= 1e-12`) between disk-resumed 2-shard KV sessions
//!   and fresh full recomputation on frozen long-context prompts.
//! - KV capacity headroom, save/load I/O latency, and cold-restart TTFT vs recomputation.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote, sha256_hex};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104";
pub const UPSTREAM_COMMIT: &str = "6c55c20da31a02d6d381a268fa42f8a9ebe0c5d2";
pub const UPSTREAM_LICENSE: &str = "Apache-2.0";
pub const UPSTREAM_X_POST: &str = "https://x.com/MiaAI_lab/status/2119016651064529304";
pub const CKPT_MAGIC: [u8; 8] = *b"SPKKVCK1";

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

fn argmax_checked(logits: &[f64]) -> Result<usize, String> {
    if logits.is_empty() {
        return Err("empty logits".into());
    }
    let mut best_i = 0usize;
    let mut best_v = check_finite(logits[0], "argmax[0]")?;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        let fv = check_finite(v, "argmax")?;
        if fv > best_v {
            best_v = fv;
            best_i = i;
        }
    }
    Ok(best_i)
}

#[derive(Debug, Clone)]
pub struct MiniGlmModel {
    pub embed: [[f64; D_MODEL]; VOCAB_SIZE],
    pub wq: [[[f64; D_MODEL]; D_MODEL]; N_LAYERS],
    pub wk: [[[f64; D_MODEL]; D_MODEL]; N_LAYERS],
    pub wv: [[[f64; D_MODEL]; D_MODEL]; N_LAYERS],
    pub wo: [[[f64; D_MODEL]; D_MODEL]; N_LAYERS],
    pub lm_head: [[f64; D_MODEL]; VOCAB_SIZE],
    pub weight_digest: String,
    pub tokenizer_digest: String,
}

impl MiniGlmModel {
    pub fn build_deterministic() -> Self {
        let mut embed = [[0.0_f64; D_MODEL]; VOCAB_SIZE];
        let mut lm_head = [[0.0_f64; D_MODEL]; VOCAB_SIZE];
        let mut wq = [[[0.0_f64; D_MODEL]; D_MODEL]; N_LAYERS];
        let mut wk = [[[0.0_f64; D_MODEL]; D_MODEL]; N_LAYERS];
        let mut wv = [[[0.0_f64; D_MODEL]; D_MODEL]; N_LAYERS];
        let mut wo = [[[0.0_f64; D_MODEL]; D_MODEL]; N_LAYERS];
        let mut digest_bytes = Vec::new();
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                embed[v][d] = det_weight(6401, v * D_MODEL + d, 0.52);
                lm_head[v][d] = det_weight(6402, v * D_MODEL + d, 0.44);
                digest_bytes.extend_from_slice(&embed[v][d].to_le_bytes());
                digest_bytes.extend_from_slice(&lm_head[v][d].to_le_bytes());
            }
        }
        for l in 0..N_LAYERS {
            for r in 0..D_MODEL {
                for c in 0..D_MODEL {
                    wq[l][r][c] = det_weight(6410 + l * 29, r * D_MODEL + c, 0.26);
                    wk[l][r][c] = det_weight(6411 + l * 29, r * D_MODEL + c, 0.26);
                    wv[l][r][c] = det_weight(6412 + l * 29, r * D_MODEL + c, 0.26);
                    wo[l][r][c] = det_weight(6413 + l * 29, r * D_MODEL + c, 0.22);
                    digest_bytes.extend_from_slice(&wq[l][r][c].to_le_bytes());
                    digest_bytes.extend_from_slice(&wk[l][r][c].to_le_bytes());
                    digest_bytes.extend_from_slice(&wv[l][r][c].to_le_bytes());
                    digest_bytes.extend_from_slice(&wo[l][r][c].to_le_bytes());
                }
            }
        }
        Self {
            embed,
            wq,
            wk,
            wv,
            wo,
            lm_head,
            weight_digest: sha256_hex(&digest_bytes),
            tokenizer_digest: sha256_hex(b"glm53-flash-tokenizer-v1"),
        }
    }
}

/// 2-node head-sharded KV store (rank 0 holds head 0, rank 1 holds head 1).
#[derive(Debug, Clone)]
pub struct ShardedKvCache {
    pub spark0_k: [Vec<[f64; HEAD_DIM]>; N_LAYERS],
    pub spark0_v: [Vec<[f64; HEAD_DIM]>; N_LAYERS],
    pub spark1_k: [Vec<[f64; HEAD_DIM]>; N_LAYERS],
    pub spark1_v: [Vec<[f64; HEAD_DIM]>; N_LAYERS],
}

impl ShardedKvCache {
    pub fn empty() -> Self {
        Self {
            spark0_k: [Vec::new(), Vec::new()],
            spark0_v: [Vec::new(), Vec::new()],
            spark1_k: [Vec::new(), Vec::new()],
            spark1_v: [Vec::new(), Vec::new()],
        }
    }

    pub fn cached_tokens(&self) -> usize {
        self.spark0_k[0].len()
    }
}

#[derive(Debug, Clone)]
pub struct DiskKvCheckpoint {
    pub repo_commit: String,
    pub weight_digest: String,
    pub tokenizer_digest: String,
    pub prefix_tokens_sha256: String,
    pub cached_tokens: usize,
    pub shard0_sha256: String,
    pub shard1_sha256: String,
    pub shard0_bytes: Vec<u8>,
    pub shard1_bytes: Vec<u8>,
}

fn hash_tokens(tokens: &[usize]) -> String {
    let mut b = Vec::with_capacity(tokens.len() * 8);
    for &t in tokens {
        b.extend_from_slice(&(t as u64).to_le_bytes());
    }
    sha256_hex(&b)
}

fn serialize_shard(
    ks: &[Vec<[f64; HEAD_DIM]>; N_LAYERS],
    vs: &[Vec<[f64; HEAD_DIM]>; N_LAYERS],
) -> Result<Vec<u8>, String> {
    let len = ks[0].len();
    let mut out = Vec::with_capacity(CKPT_MAGIC.len() + N_LAYERS * len * HEAD_DIM * 16);
    out.extend_from_slice(&CKPT_MAGIC);
    for l in 0..N_LAYERS {
        if ks[l].len() != len || vs[l].len() != len {
            return Err("ragged shard lengths".into());
        }
        for p in 0..len {
            for &v in &ks[l][p] {
                out.extend_from_slice(&check_finite(v, "save_k")?.to_le_bytes());
            }
            for &v in &vs[l][p] {
                out.extend_from_slice(&check_finite(v, "save_v")?.to_le_bytes());
            }
        }
    }
    Ok(out)
}

type DeserializedShard = (
    [Vec<[f64; HEAD_DIM]>; N_LAYERS],
    [Vec<[f64; HEAD_DIM]>; N_LAYERS],
);

fn deserialize_shard(
    bytes: &[u8],
    expected_tokens: usize,
) -> Result<DeserializedShard, OutcomeStatus> {
    let expected_len = CKPT_MAGIC.len() + N_LAYERS * expected_tokens * HEAD_DIM * 16;
    if bytes.len() != expected_len || !bytes.starts_with(&CKPT_MAGIC) {
        return Err(OutcomeStatus::Invalid);
    }
    let mut ks = [
        Vec::with_capacity(expected_tokens),
        Vec::with_capacity(expected_tokens),
    ];
    let mut vs = [
        Vec::with_capacity(expected_tokens),
        Vec::with_capacity(expected_tokens),
    ];
    let mut c = CKPT_MAGIC.len();
    for l in 0..N_LAYERS {
        for _ in 0..expected_tokens {
            let mut k = [0.0_f64; HEAD_DIM];
            for slot in &mut k {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[c..c + 8]);
                c += 8;
                let v = f64::from_le_bytes(b);
                if !v.is_finite() {
                    return Err(OutcomeStatus::Invalid);
                }
                *slot = v;
            }
            let mut v_arr = [0.0_f64; HEAD_DIM];
            for slot in &mut v_arr {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[c..c + 8]);
                c += 8;
                let v = f64::from_le_bytes(b);
                if !v.is_finite() {
                    return Err(OutcomeStatus::Invalid);
                }
                *slot = v;
            }
            ks[l].push(k);
            vs[l].push(v_arr);
        }
    }
    Ok((ks, vs))
}

pub fn save_kv_checkpoint(
    model: &MiniGlmModel,
    kv: &ShardedKvCache,
    prefix_tokens: &[usize],
) -> Result<DiskKvCheckpoint, String> {
    if kv.cached_tokens() != prefix_tokens.len() {
        return Err("cached_tokens does not match prefix_tokens length".into());
    }
    let shard0_bytes = serialize_shard(&kv.spark0_k, &kv.spark0_v)?;
    let shard1_bytes = serialize_shard(&kv.spark1_k, &kv.spark1_v)?;
    Ok(DiskKvCheckpoint {
        repo_commit: UPSTREAM_COMMIT.into(),
        weight_digest: model.weight_digest.clone(),
        tokenizer_digest: model.tokenizer_digest.clone(),
        prefix_tokens_sha256: hash_tokens(prefix_tokens),
        cached_tokens: prefix_tokens.len(),
        shard0_sha256: sha256_hex(&shard0_bytes),
        shard1_sha256: sha256_hex(&shard1_bytes),
        shard0_bytes,
        shard1_bytes,
    })
}

pub fn resume_kv_checkpoint(
    model: &MiniGlmModel,
    ckpt: &DiskKvCheckpoint,
    expected_prefix_tokens: &[usize],
) -> Result<ShardedKvCache, OutcomeStatus> {
    if ckpt.repo_commit != UPSTREAM_COMMIT
        || ckpt.weight_digest != model.weight_digest
        || ckpt.tokenizer_digest != model.tokenizer_digest
    {
        return Err(OutcomeStatus::Invalid);
    }
    if ckpt.cached_tokens != expected_prefix_tokens.len()
        || ckpt.prefix_tokens_sha256 != hash_tokens(expected_prefix_tokens)
    {
        return Err(OutcomeStatus::Invalid);
    }
    if sha256_hex(&ckpt.shard0_bytes) != ckpt.shard0_sha256
        || sha256_hex(&ckpt.shard1_bytes) != ckpt.shard1_sha256
    {
        return Err(OutcomeStatus::Invalid);
    }
    let (s0k, s0v) = deserialize_shard(&ckpt.shard0_bytes, ckpt.cached_tokens)?;
    let (s1k, s1v) = deserialize_shard(&ckpt.shard1_bytes, ckpt.cached_tokens)?;
    Ok(ShardedKvCache {
        spark0_k: s0k,
        spark0_v: s0v,
        spark1_k: s1k,
        spark1_v: s1v,
    })
}

#[allow(clippy::needless_range_loop)]
fn matvec_dd(w: &[[f64; D_MODEL]; D_MODEL], x: &[f64; D_MODEL]) -> [f64; D_MODEL] {
    let mut out = [0.0_f64; D_MODEL];
    for r in 0..D_MODEL {
        let mut s = 0.0_f64;
        for c in 0..D_MODEL {
            s += w[r][c] * x[c];
        }
        out[r] = s;
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn step_sharded(
    model: &MiniGlmModel,
    token: usize,
    kv: &mut ShardedKvCache,
) -> Result<[f64; VOCAB_SIZE], String> {
    if token >= VOCAB_SIZE {
        return Err("token out of vocab".into());
    }
    let mut h = model.embed[token];
    let inv_scale = 1.0_f64 / (HEAD_DIM as f64).sqrt();
    for l in 0..N_LAYERS {
        let q = matvec_dd(&model.wq[l], &h);
        let k = matvec_dd(&model.wk[l], &h);
        let v = matvec_dd(&model.wv[l], &h);
        let mut k0 = [0.0_f64; HEAD_DIM];
        let mut v0 = [0.0_f64; HEAD_DIM];
        let mut k1 = [0.0_f64; HEAD_DIM];
        let mut v1 = [0.0_f64; HEAD_DIM];
        k0.copy_from_slice(&k[0..HEAD_DIM]);
        v0.copy_from_slice(&v[0..HEAD_DIM]);
        k1.copy_from_slice(&k[HEAD_DIM..D_MODEL]);
        v1.copy_from_slice(&v[HEAD_DIM..D_MODEL]);
        kv.spark0_k[l].push(k0);
        kv.spark0_v[l].push(v0);
        kv.spark1_k[l].push(k1);
        kv.spark1_v[l].push(v1);

        let t = kv.spark0_k[l].len();
        let mut concat = [0.0_f64; D_MODEL];
        for (shard_idx, (sk, sv)) in [
            (&kv.spark0_k[l], &kv.spark0_v[l]),
            (&kv.spark1_k[l], &kv.spark1_v[l]),
        ]
        .iter()
        .enumerate()
        {
            let off = shard_idx * HEAD_DIM;
            let mut scores = Vec::with_capacity(t);
            let mut m = f64::NEG_INFINITY;
            for p in 0..t {
                let mut dot = 0.0_f64;
                for d in 0..HEAD_DIM {
                    dot += q[off + d] * sk[p][d];
                }
                let s = check_finite(dot * inv_scale, "score")?;
                if s > m {
                    m = s;
                }
                scores.push(s);
            }
            let mut sum = 0.0_f64;
            let mut ws = Vec::with_capacity(t);
            for &s in &scores {
                let w = (s - m).exp();
                sum += w;
                ws.push(w);
            }
            for d in 0..HEAD_DIM {
                let mut acc = 0.0_f64;
                for p in 0..t {
                    acc += (ws[p] / sum) * sv[p][d];
                }
                concat[off + d] = acc;
            }
        }
        let proj = matvec_dd(&model.wo[l], &concat);
        for d in 0..D_MODEL {
            h[d] = check_finite(h[d] + proj[d], "res")?;
        }
    }
    let mut logits = [0.0_f64; VOCAB_SIZE];
    for v in 0..VOCAB_SIZE {
        let mut s = 0.0_f64;
        for d in 0..D_MODEL {
            s += model.lm_head[v][d] * h[d];
        }
        logits[v] = check_finite(s, "logits")?;
    }
    Ok(logits)
}

#[derive(Debug, Clone)]
pub struct ResumeParityResult {
    pub session_id: String,
    pub cached_prefix_tokens: usize,
    pub tail_prompt_tokens: usize,
    pub decode_steps: usize,
    pub checkpoint_total_bytes: usize,
    pub token_exact_match: bool,
    pub max_logit_abs_err: f64,
    pub outcome: OutcomeStatus,
}

#[derive(Debug, Clone)]
pub struct CapacityAndIoRow {
    pub profile_id: String,
    pub spark_nodes: usize,
    pub total_kv_capacity_tokens: usize,
    pub cached_tokens: usize,
    pub tail_tokens: usize,
    pub headroom_ratio: Option<f64>,
    pub save_io_s: Option<f64>,
    pub load_io_s: Option<f64>,
    pub resume_ttft_s: Option<f64>,
    pub fresh_recompute_ttft_s: f64,
    pub outcome: OutcomeStatus,
}

#[allow(clippy::too_many_arguments)]
pub fn compute_capacity_and_io(
    profile_id: &str,
    spark_nodes: usize,
    per_node_kv_gb: f64,
    kv_bytes_per_token: usize,
    cached_tokens: usize,
    tail_tokens: usize,
    nvme_gb_s: f64,
    prefill_tok_s: f64,
) -> Result<CapacityAndIoRow, String> {
    check_finite(per_node_kv_gb, "per_node_kv_gb")?;
    check_finite(nvme_gb_s, "nvme_gb_s")?;
    check_finite(prefill_tok_s, "prefill_tok_s")?;
    if spark_nodes == 0
        || kv_bytes_per_token == 0
        || per_node_kv_gb <= 0.0
        || nvme_gb_s <= 0.0
        || prefill_tok_s <= 0.0
    {
        return Ok(CapacityAndIoRow {
            profile_id: profile_id.into(),
            spark_nodes,
            total_kv_capacity_tokens: 0,
            cached_tokens,
            tail_tokens,
            headroom_ratio: None,
            save_io_s: None,
            load_io_s: None,
            resume_ttft_s: None,
            fresh_recompute_ttft_s: 0.0,
            outcome: OutcomeStatus::Invalid,
        });
    }
    let total_bytes = (spark_nodes as f64) * per_node_kv_gb * 1.0e9;
    let total_cap_tokens = (total_bytes / (kv_bytes_per_token as f64)) as usize;
    let total_used = cached_tokens.saturating_add(tail_tokens);
    let fresh_recompute_ttft_s = (total_used as f64) / prefill_tok_s;
    if total_used > total_cap_tokens {
        return Ok(CapacityAndIoRow {
            profile_id: profile_id.into(),
            spark_nodes,
            total_kv_capacity_tokens: total_cap_tokens,
            cached_tokens,
            tail_tokens,
            headroom_ratio: None,
            save_io_s: None,
            load_io_s: None,
            resume_ttft_s: None,
            fresh_recompute_ttft_s,
            outcome: OutcomeStatus::Oom,
        });
    }
    let headroom = ((total_cap_tokens - total_used) as f64) / (total_cap_tokens as f64);
    let ckpt_gb = ((cached_tokens as f64) * (kv_bytes_per_token as f64)) / 1.0e9;
    let save_s = ckpt_gb / (nvme_gb_s * 0.85);
    let load_s = ckpt_gb / nvme_gb_s;
    let tail_prefill_s = (tail_tokens as f64) / prefill_tok_s;
    let resume_ttft_s = load_s + tail_prefill_s;
    Ok(CapacityAndIoRow {
        profile_id: profile_id.into(),
        spark_nodes,
        total_kv_capacity_tokens: total_cap_tokens,
        cached_tokens,
        tail_tokens,
        headroom_ratio: Some(headroom),
        save_io_s: Some(save_s),
        load_io_s: Some(load_s),
        resume_ttft_s: Some(resume_ttft_s),
        fresh_recompute_ttft_s,
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
pub struct SparkKvCheckpointReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub upstream_repo: String,
    pub upstream_commit: String,
    pub upstream_license: String,
    pub resume_parity: Vec<ResumeParityResult>,
    pub capacity_profiles: Vec<CapacityAndIoRow>,
    pub negative_controls: Vec<NegativeControlResult>,
}

fn eval_resume_session(
    model: &MiniGlmModel,
    session_id: &str,
    cached_prefix: &[usize],
    tail_prompt: &[usize],
    decode_steps: usize,
) -> Result<ResumeParityResult, String> {
    // Path 1: Fresh full recomputation of cached_prefix + tail_prompt + decode
    let mut fresh_kv = ShardedKvCache::empty();
    let mut fresh_last = [0.0_f64; VOCAB_SIZE];
    for &tok in cached_prefix.iter().chain(tail_prompt.iter()) {
        fresh_last = step_sharded(model, tok, &mut fresh_kv)?;
    }
    let mut fresh_tokens = Vec::with_capacity(decode_steps);
    let mut fresh_logits = vec![fresh_last];
    let mut cur = argmax_checked(&fresh_last)?;
    fresh_tokens.push(cur);
    for _ in 1..decode_steps {
        let log = step_sharded(model, cur, &mut fresh_kv)?;
        cur = argmax_checked(&log)?;
        fresh_tokens.push(cur);
        fresh_logits.push(log);
    }

    // Path 2: Prefill cached_prefix -> save 2-shard disk checkpoint -> resume -> tail_prompt -> decode
    let mut pre_kv = ShardedKvCache::empty();
    for &tok in cached_prefix {
        let _ = step_sharded(model, tok, &mut pre_kv)?;
    }
    let ckpt = save_kv_checkpoint(model, &pre_kv, cached_prefix)?;
    let mut resumed_kv = resume_kv_checkpoint(model, &ckpt, cached_prefix)
        .map_err(|st| format!("unexpected resume failure: {}", st.as_str()))?;
    let mut res_last = [0.0_f64; VOCAB_SIZE];
    for &tok in tail_prompt {
        res_last = step_sharded(model, tok, &mut resumed_kv)?;
    }
    let mut res_tokens = Vec::with_capacity(decode_steps);
    let mut res_logits = vec![res_last];
    let mut rcur = argmax_checked(&res_last)?;
    res_tokens.push(rcur);
    for _ in 1..decode_steps {
        let log = step_sharded(model, rcur, &mut resumed_kv)?;
        rcur = argmax_checked(&log)?;
        res_tokens.push(rcur);
        res_logits.push(log);
    }

    let mut max_err = 0.0_f64;
    for (a, b) in fresh_logits.iter().zip(res_logits.iter()) {
        for d in 0..VOCAB_SIZE {
            let e = (a[d] - b[d]).abs();
            if e > max_err {
                max_err = e;
            }
        }
    }
    let exact = fresh_tokens == res_tokens && max_err <= PARITY_TOL;
    Ok(ResumeParityResult {
        session_id: session_id.into(),
        cached_prefix_tokens: cached_prefix.len(),
        tail_prompt_tokens: tail_prompt.len(),
        decode_steps,
        checkpoint_total_bytes: ckpt.shard0_bytes.len() + ckpt.shard1_bytes.len(),
        token_exact_match: exact,
        max_logit_abs_err: max_err,
        outcome: if exact {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
    })
}

pub fn run_default_harness() -> Result<SparkKvCheckpointReport, String> {
    let model = MiniGlmModel::build_deterministic();
    let resume_parity = vec![
        eval_resume_session(
            &model,
            "chat_resume_8_plus_3",
            &[1, 4, 7, 2, 9, 5, 11, 3],
            &[6, 8, 10],
            5,
        )?,
        eval_resume_session(
            &model,
            "chat_resume_12_plus_4",
            &[0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1],
            &[4, 7, 10, 13],
            5,
        )?,
    ];

    let capacity_profiles = vec![
        compute_capacity_and_io(
            "2spark_101k_chat_98k_cached",
            2,
            85.0,
            54_000,
            98_000,
            3_000,
            5.5,
            2400.0,
        )?,
        compute_capacity_and_io(
            "2spark_512k_session_490k_cached",
            2,
            85.0,
            54_000,
            490_000,
            22_000,
            5.5,
            2200.0,
        )?,
        compute_capacity_and_io(
            "2spark_1m_session_980k_cached",
            2,
            85.0,
            54_000,
            980_000,
            20_000,
            5.5,
            1900.0,
        )?,
    ];

    let prefix = [1usize, 4, 7, 2];
    let mut kv = ShardedKvCache::empty();
    for &t in &prefix {
        let _ = step_sharded(&model, t, &mut kv)?;
    }
    let ckpt = save_kv_checkpoint(&model, &kv, &prefix)?;

    let mut nc = Vec::new();
    // NC1: corrupted shard 1 byte rejected
    let mut bad_s1 = ckpt.clone();
    let last = bad_s1.shard1_bytes.len() - 1;
    bad_s1.shard1_bytes[last] ^= 0x01;
    nc.push(NegativeControlResult {
        case_id: "nc1_corrupted_shard1_checksum_rejected".into(),
        outcome: resume_kv_checkpoint(&model, &bad_s1, &prefix).unwrap_err(),
        detail: "1-bit corruption in Spark-1 KV shard refused by SHA-256 verification".into(),
    });

    // NC2: edited prefix token invalidates checkpoint resume
    nc.push(NegativeControlResult {
        case_id: "nc2_prefix_token_drift_rejected".into(),
        outcome: resume_kv_checkpoint(&model, &ckpt, &[1, 4, 7, 9]).unwrap_err(),
        detail: "modified prefix token failed prefix_tokens_sha256 match".into(),
    });

    // NC3: mismatched model weight digest rejected
    let mut bad_w = ckpt.clone();
    bad_w.weight_digest = sha256_hex(b"other-glm-weights");
    nc.push(NegativeControlResult {
        case_id: "nc3_weight_digest_mismatch_rejected".into(),
        outcome: resume_kv_checkpoint(&model, &bad_w, &prefix).unwrap_err(),
        detail: "checkpoint produced by different weight digest refused".into(),
    });

    // NC4: mismatched upstream repo commit rejected
    let mut bad_c = ckpt.clone();
    bad_c.repo_commit = "0000000000000000000000000000000000000000".into();
    nc.push(NegativeControlResult {
        case_id: "nc4_unpinned_commit_mismatch_rejected".into(),
        outcome: resume_kv_checkpoint(&model, &bad_c, &prefix).unwrap_err(),
        detail: "unpinned/mismatched GLM-5.3-Flash-DGX-Sparks commit refused".into(),
    });

    // NC5: truncated shard 0 checkpoint refused
    let mut trunc = ckpt.clone();
    trunc.shard0_bytes.truncate(16);
    trunc.shard0_sha256 = sha256_hex(&trunc.shard0_bytes);
    nc.push(NegativeControlResult {
        case_id: "nc5_truncated_shard0_rejected".into(),
        outcome: resume_kv_checkpoint(&model, &trunc, &prefix).unwrap_err(),
        detail: "truncated Spark-0 shard refused even with recomputed digest".into(),
    });

    // NC6: NaN entry in shard rejected on save
    let mut nan_kv = kv.clone();
    nan_kv.spark0_k[0][0][0] = f64::NAN;
    let nc6 = if save_kv_checkpoint(&model, &nan_kv, &prefix).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc6_nan_kv_entry_blocked_on_save".into(),
        outcome: nc6,
        detail: "NaN KV shard entry refused before disk checkpoint serialization".into(),
    });

    // NC7: exceeding 2-Spark KV capacity returns Oom
    let oom_row =
        compute_capacity_and_io("overflow", 2, 85.0, 54_000, 4_000_000, 100_000, 5.5, 2000.0)?;
    nc.push(NegativeControlResult {
        case_id: "nc7_kv_capacity_overflow_returns_oom".into(),
        outcome: oom_row.outcome,
        detail: "4.1M tokens > 2-Spark capacity returned explicit oom without silent eviction"
            .into(),
    });

    // NC8: zero NVMe bandwidth guarded against division by zero
    let zero_io = compute_capacity_and_io("zero_io", 2, 85.0, 54_000, 98_000, 3_000, 0.0, 2000.0)?;
    nc.push(NegativeControlResult {
        case_id: "nc8_zero_nvme_bandwidth_guarded".into(),
        outcome: zero_io.outcome,
        detail: "0 GB/s NVMe throughput returned invalid with save/load_io_s=None".into(),
    });

    Ok(SparkKvCheckpointReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "2x DGX Spark cluster serving pinned MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104@6c55c20d with NVMe KV checkpoints",
            "2-shard KV checkpoint save/resume exact parity and capacity/I/O accounting verified on CPU; physical 2x DGX Spark cluster did not run locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        upstream_repo: UPSTREAM_REPO.into(),
        upstream_commit: UPSTREAM_COMMIT.into(),
        upstream_license: UPSTREAM_LICENSE.into(),
        resume_parity,
        capacity_profiles,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &SparkKvCheckpointReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.spark-kv-checkpoint.inputs.v1\",\n",
            "  \"issue\": 64,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_repo\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"upstream_license\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"complements\": [\"inference-lab#5\",\"inference-lab#11\"],\n",
            "  \"author_unverified_non_gating_notes\": \"~3.1M total KV cache and 101K chat / 98K cached identical resume are unverified author claims, not lab targets\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_repo),
        quote(&report.upstream_commit),
        quote(&report.upstream_license),
        quote(UPSTREAM_X_POST),
    )
}

pub fn results_json(report: &SparkKvCheckpointReport) -> String {
    let par_json: Vec<String> = report
        .resume_parity
        .iter()
        .map(|p| {
            format!(
                "{{\"session_id\":{},\"cached_prefix_tokens\":{},\"tail_prompt_tokens\":{},\"decode_steps\":{},\"checkpoint_total_bytes\":{},\"token_exact_match\":{},\"max_logit_abs_err\":{},\"outcome\":{}}}",
                quote(&p.session_id),
                p.cached_prefix_tokens,
                p.tail_prompt_tokens,
                p.decode_steps,
                p.checkpoint_total_bytes,
                p.token_exact_match,
                finite_f64(p.max_logit_abs_err),
                quote(p.outcome.as_str()),
            )
        })
        .collect();
    let cap_json: Vec<String> = report
        .capacity_profiles
        .iter()
        .map(|r| {
            format!(
                "{{\"profile_id\":{},\"spark_nodes\":{},\"total_kv_capacity_tokens\":{},\"cached_tokens\":{},\"tail_tokens\":{},\"headroom_ratio\":{},\"save_io_s\":{},\"load_io_s\":{},\"resume_ttft_s\":{},\"fresh_recompute_ttft_s\":{},\"outcome\":{}}}",
                quote(&r.profile_id),
                r.spark_nodes,
                r.total_kv_capacity_tokens,
                r.cached_tokens,
                r.tail_tokens,
                opt_f64(r.headroom_ratio),
                opt_f64(r.save_io_s),
                opt_f64(r.load_io_s),
                opt_f64(r.resume_ttft_s),
                finite_f64(r.fresh_recompute_ttft_s),
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
            "  \"schema\": \"inference-lab.spark-kv-checkpoint.results.v1\",\n",
            "  \"issue\": 64,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"resume_parity\": [{}],\n",
            "  \"capacity_profiles\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_commit),
        par_json.join(","),
        cap_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &SparkKvCheckpointReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #64: Two-Spark KV Sharding & Disk-Persisted KV Checkpoint Resume (GLM-5.3-Flash)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, pinned `{}@{}` `{}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.upstream_repo,
        &report.upstream_commit[..8],
        report.upstream_license
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
    writeln!(out, "## Sharded Disk-Resume vs Fresh Recomputation Parity").unwrap();
    for p in &report.resume_parity {
        writeln!(
            out,
            "- `{}` (`cached={}`, `tail={}`, `steps={}`): `bytes={}`, `exact={}`, `max_logit_err={:.3e}`, `outcome={}`",
            p.session_id,
            p.cached_prefix_tokens,
            p.tail_prompt_tokens,
            p.decode_steps,
            p.checkpoint_total_bytes,
            p.token_exact_match,
            p.max_logit_abs_err,
            p.outcome.as_str()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "## KV Capacity Headroom & Save/Load I/O vs Recomputation"
    )
    .unwrap();
    for r in &report.capacity_profiles {
        writeln!(
            out,
            "- `{}` (`cap={}`, `cached={}`, `tail={}`): `headroom={:?}`, `load_s={:?}`, `resume_ttft_s={:?}` vs `recompute_ttft_s={:.2}`, `outcome={}`",
            r.profile_id,
            r.total_kv_capacity_tokens,
            r.cached_tokens,
            r.tail_tokens,
            r.headroom_ratio,
            r.load_io_s,
            r.resume_ttft_s,
            r.fresh_recompute_ttft_s,
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
    fn resume_is_exact_and_negative_controls_fail() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert_eq!(report.upstream_commit.len(), 40);
        for p in &report.resume_parity {
            assert!(p.token_exact_match);
            assert!(p.max_logit_abs_err <= PARITY_TOL);
        }
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
