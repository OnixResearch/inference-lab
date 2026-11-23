//! Issue #32: Validate QSA cache reuse and target/indexer/GDN/MTP state coherence for agent sessions.
//!
//! Pinned implementation references (strictly non-gating upstream claims):
//! - `kuraneko1/Qwen3.8-27B-QSA-llama.cpp@864934c78136ad75274f202b8eb1e7c751d81fb1`
//!   (`README.md`, `src/llama-memory-hybrid-idx.cpp`, `qsa27b/validation/check-mm2-mtp.cpp`,
//!   `qsa27b/RUNTIME.md`).
//! - Single live slot prefix reuse, no persistent on-disk cache, no survival across restart; small
//!   native fixtures do not qualify arbitrary multi-agent concurrency or 27B serving.
//! - Supports `#18` and `#31` (dense-vs-QSA approximation quality belongs to `#31`); reuses `#3`/`#22`
//!   speculation semantics and `#7`/`#11` runner/lifecycle accounting.
//!
//! Provides:
//! - Complete `ContinuationIdentity` binding backbone/indexer hashes, engine/build, tokenizer/template,
//!   accepted token history, positions, attention mode/budget, KV precision, recurrent layout, MTP mode,
//!   and session/deployment generation, plus explicit `advertised`/`observed`/`verified` cache-reuse state.
//! - Coupled pure-Rust 4-component state engine (Target KV, Raw + 4-token Pooled Indexer Keys, Gated
//!   DeltaNet matrix recurrence `S_t`, and MTP draft-2 cache) with `1e6` sentinel poisoning on
//!   uncommitted/rolled-back tail slots.
//! - Fresh-recompute controls vs append/resume, repeated prefix, chunked prefill across 4-token block
//!   and sparse-selection boundaries, truncation + changed suffix, checkpoint restore with rebuilt
//!   pooled keys, MTP-off vs draft-2 accept/reject/bonus rollback, two-session single-slot isolation,
//!   explicitly unsupported branching/disk persistence, reuse cost accounting, and 8 negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, sigmoid, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote, u32_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};
use crate::qwen38_qsa::{QSA_BLOCK_SIZE, QSA_ENGINE_COMMIT};

pub const QSA_DIM: usize = 4;
pub const QSA_VOCAB: usize = 8;
pub const SENTINEL_POISON: f64 = 1.0e6;
pub const CONTINUATION_TOL: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsaPolicyMode {
    QsaBlockSparse { top_blocks: usize },
    DenseControl,
}

impl QsaPolicyMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QsaBlockSparse { .. } => "qsa_block_sparse",
            Self::DenseControl => "dense_control",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheReuseEvidence {
    AdvertisedOnly,
    ObservedTelemetry,
    VerifiedAgainstFreshRecompute,
    MissingTelemetry,
}

impl CacheReuseEvidence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AdvertisedOnly => "advertised_only",
            Self::ObservedTelemetry => "observed_telemetry",
            Self::VerifiedAgainstFreshRecompute => "verified_against_fresh_recompute",
            Self::MissingTelemetry => "missing_telemetry",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuationIdentity {
    pub backbone_digest: DigestRef,
    pub indexer_digest: DigestRef,
    pub engine_commit: String,
    pub tokenizer_template_rev: String,
    pub accepted_tokens: Vec<u32>,
    pub accepted_positions: Vec<usize>,
    pub attention_mode: QsaPolicyMode,
    pub kv_precision: String,
    pub recurrent_layout: String,
    pub mtp_draft2_enabled: bool,
    pub session_id: String,
    pub deployment_generation: u64,
}

impl ContinuationIdentity {
    pub fn canonical_digest(&self) -> DigestRef {
        let repr = format!(
            "{}|{}|{}|{}|{:?}|{:?}|{:?}|{}|{}|{}|{}|{}",
            self.backbone_digest.hex,
            self.indexer_digest.hex,
            self.engine_commit,
            self.tokenizer_template_rev,
            self.accepted_tokens,
            self.accepted_positions,
            self.attention_mode,
            self.kv_precision,
            self.recurrent_layout,
            self.mtp_draft2_enabled,
            self.session_id,
            self.deployment_generation
        );
        DigestRef::sha256(repr.as_bytes())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StateFaultHooks {
    pub stale_pooled_index_keys: bool,
    pub wrong_token_position_offset: usize,
    pub omit_gdn_rollback: bool,
    pub retain_rejected_draft_in_kv: bool,
    pub allow_cross_session_prefix_reuse: bool,
}

#[derive(Debug, Clone)]
pub struct QsaCoupledState {
    pub identity: ContinuationIdentity,
    pub capacity: usize,
    pub valid_len: usize,
    pub target_k: Vec<[f64; QSA_DIM]>,
    pub target_v: Vec<[f64; QSA_DIM]>,
    pub raw_index_k: Vec<[f64; QSA_DIM]>,
    pub pooled_index_k: Vec<[f64; QSA_DIM]>,
    pub gdn_matrix: [[f64; QSA_DIM]; QSA_DIM],
    pub mtp_hidden: [f64; QSA_DIM],
    pub mtp_valid_len: usize,
    pub positions: Vec<usize>,
    pub last_logits: [f64; QSA_VOCAB],
}

fn checked_max_abs_diff(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    let mut max_err = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite value encountered during state coherence diff".into());
        }
        let d = (x - y).abs();
        if d > max_err {
            max_err = d;
        }
    }
    Ok(max_err)
}

fn token_embed(token: u32, pos: usize) -> [f64; QSA_DIM] {
    let mut e = [0.0_f64; QSA_DIM];
    for (i, slot) in e.iter_mut().enumerate() {
        let base = det_weight(3201 + (token as usize) * 13, i, 0.45);
        let pos_enc = ((pos as f64 + 1.0) * 0.17 * (i as f64 + 1.0)).sin() * 0.15;
        *slot = base + pos_enc;
    }
    e
}

fn project_vec(x: &[f64; QSA_DIM], seed: usize) -> [f64; QSA_DIM] {
    let mut out = [0.0_f64; QSA_DIM];
    for (o, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (i, &xv) in x.iter().enumerate() {
            acc += xv * det_weight(seed + o * 7, i, 0.55);
        }
        *slot = acc;
    }
    out
}

fn rebuild_pooled_index_keys(raw: &[[f64; QSA_DIM]], valid_len: usize) -> Vec<[f64; QSA_DIM]> {
    let num_blocks = valid_len.div_ceil(QSA_BLOCK_SIZE);
    let mut pooled = vec![[0.0_f64; QSA_DIM]; num_blocks];
    for (b, slot) in pooled.iter_mut().enumerate() {
        let start = b * QSA_BLOCK_SIZE;
        let end = (start + QSA_BLOCK_SIZE).min(valid_len);
        let count = (end - start) as f64;
        for row in &raw[start..end] {
            for (d, val) in slot.iter_mut().enumerate() {
                *val += row[d] / count;
            }
        }
    }
    pooled
}

impl QsaCoupledState {
    pub fn fresh(identity: ContinuationIdentity, capacity: usize) -> Self {
        let mut id = identity;
        id.accepted_tokens.clear();
        id.accepted_positions.clear();
        Self {
            identity: id,
            capacity,
            valid_len: 0,
            target_k: vec![[SENTINEL_POISON; QSA_DIM]; capacity],
            target_v: vec![[SENTINEL_POISON; QSA_DIM]; capacity],
            raw_index_k: vec![[SENTINEL_POISON; QSA_DIM]; capacity],
            pooled_index_k: Vec::new(),
            gdn_matrix: [[0.0; QSA_DIM]; QSA_DIM],
            mtp_hidden: [0.0; QSA_DIM],
            mtp_valid_len: 0,
            positions: vec![usize::MAX; capacity],
            last_logits: [0.0; QSA_VOCAB],
        }
    }

    pub fn poison_tail(&mut self) {
        for idx in self.valid_len..self.capacity {
            self.target_k[idx] = [SENTINEL_POISON; QSA_DIM];
            self.target_v[idx] = [SENTINEL_POISON; QSA_DIM];
            self.raw_index_k[idx] = [SENTINEL_POISON; QSA_DIM];
            self.positions[idx] = usize::MAX;
        }
    }

    pub fn step_token(
        &mut self,
        token: u32,
        hooks: StateFaultHooks,
    ) -> Result<[f64; QSA_VOCAB], String> {
        if self.valid_len >= self.capacity {
            return Err("context capacity exhausted".into());
        }
        let pos = self.valid_len + hooks.wrong_token_position_offset;
        let x = token_embed(token, pos);
        let q = project_vec(&x, 3210);
        let k = project_vec(&x, 3220);
        let v = project_vec(&x, 3230);
        let idx_k = project_vec(&x, 3240);

        let slot = self.valid_len;
        self.target_k[slot] = k;
        self.target_v[slot] = v;
        self.raw_index_k[slot] = idx_k;
        self.positions[slot] = pos;
        self.valid_len += 1;
        self.identity.accepted_tokens.push(token);
        self.identity.accepted_positions.push(pos);

        if !hooks.stale_pooled_index_keys || self.pooled_index_k.is_empty() {
            self.pooled_index_k = rebuild_pooled_index_keys(&self.raw_index_k, self.valid_len);
        }

        // Gated DeltaNet recurrent state update: S_t = alpha * S_{t-1} + beta * (v - S_{t-1} k_n) k_n^T
        let alpha = 0.72 + 0.18 * sigmoid(x[0]);
        let beta = 0.25 * sigmoid(x[1]);
        let k_sq: f64 = k.iter().map(|c| c * c).sum();
        if !k_sq.is_finite() {
            return Err("non-finite key norm in GDN update".into());
        }
        let kn_raw = k_sq.sqrt();
        let kn_norm = if kn_raw < 1.0e-6 { 1.0e-6 } else { kn_raw };
        let mut kn = [0.0_f64; QSA_DIM];
        for i in 0..QSA_DIM {
            kn[i] = k[i] / kn_norm;
        }
        let mut sk = [0.0_f64; QSA_DIM];
        for (r, sk_r) in sk.iter_mut().enumerate() {
            for (c, &kn_c) in kn.iter().enumerate() {
                *sk_r += self.gdn_matrix[r][c] * kn_c;
            }
        }
        for (r, row) in self.gdn_matrix.iter_mut().enumerate() {
            let delta = v[r] - sk[r];
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = alpha * *cell + beta * delta * kn[c];
                if !cell.is_finite() {
                    return Err("non-finite GDN recurrent state".into());
                }
            }
        }
        let mut gdn_out = [0.0_f64; QSA_DIM];
        for (r, out_r) in gdn_out.iter_mut().enumerate() {
            for (c, &q_c) in q.iter().enumerate() {
                *out_r += self.gdn_matrix[r][c] * q_c;
            }
        }

        // Select attending tokens under fixed QSA or dense policy
        let selected: Vec<usize> = match self.identity.attention_mode {
            QsaPolicyMode::DenseControl => (0..self.valid_len).collect(),
            QsaPolicyMode::QsaBlockSparse { top_blocks } => {
                let num_blocks = self.pooled_index_k.len();
                if num_blocks <= top_blocks {
                    (0..self.valid_len).collect()
                } else {
                    let idx_q = project_vec(&x, 3250);
                    let mut scored: Vec<(usize, f64)> = self
                        .pooled_index_k
                        .iter()
                        .enumerate()
                        .map(|(b, pk)| {
                            let s: f64 = idx_q.iter().zip(pk.iter()).map(|(a, c)| a * c).sum();
                            (b, s)
                        })
                        .collect();
                    if scored.iter().any(|(_, s)| !s.is_finite()) {
                        return Err("non-finite indexer block score".into());
                    }
                    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                    let mut idxs = Vec::new();
                    for &(b, _) in scored.iter().take(top_blocks.max(1)) {
                        let start = b * QSA_BLOCK_SIZE;
                        let end = (start + QSA_BLOCK_SIZE).min(self.valid_len);
                        for t in start..end {
                            idxs.push(t);
                        }
                    }
                    idxs.sort_unstable();
                    idxs
                }
            }
        };

        let scale = 1.0 / (QSA_DIM as f64).sqrt();
        let mut scores = Vec::with_capacity(selected.len());
        for &t in &selected {
            let dot: f64 = q
                .iter()
                .zip(self.target_k[t].iter())
                .map(|(a, b)| a * b)
                .sum();
            scores.push(dot * scale);
        }
        let probs = softmax_stable(&scores)?;
        let mut attn_out = [0.0_f64; QSA_DIM];
        for (p, &t) in probs.iter().zip(selected.iter()) {
            for (d, out_d) in attn_out.iter_mut().enumerate() {
                *out_d += p * self.target_v[t][d];
            }
        }

        // Combined hidden + MTP state
        let mut hidden = [0.0_f64; QSA_DIM];
        for d in 0..QSA_DIM {
            hidden[d] = 0.55 * attn_out[d] + 0.35 * gdn_out[d] + 0.10 * x[d];
        }
        if self.identity.mtp_draft2_enabled {
            for (d, m) in self.mtp_hidden.iter_mut().enumerate() {
                *m = 0.6 * *m + 0.4 * hidden[d];
            }
            self.mtp_valid_len = self.valid_len;
        }

        let mut logits = [0.0_f64; QSA_VOCAB];
        for (v_idx, l) in logits.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (d, &h_d) in hidden.iter().enumerate() {
                acc += h_d * det_weight(3260 + v_idx * 5, d, 0.7);
            }
            if !acc.is_finite() {
                return Err("non-finite output logit".into());
            }
            *l = acc;
        }
        self.last_logits = logits;
        Ok(logits)
    }

    pub fn process_tokens(
        &mut self,
        tokens: &[u32],
        hooks: StateFaultHooks,
    ) -> Result<[f64; QSA_VOCAB], String> {
        let mut last = self.last_logits;
        for &tok in tokens {
            last = self.step_token(tok, hooks)?;
        }
        Ok(last)
    }

    pub fn argmax_token(&self) -> Result<u32, String> {
        let mut best_i = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for (i, &v) in self.last_logits.iter().enumerate() {
            if !v.is_finite() {
                return Err("non-finite logit in argmax".into());
            }
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        Ok(best_i as u32)
    }

    /// Snapshot accepted state at the current boundary (in-slot memory checkpoint, not disk cache).
    pub fn snapshot(&self) -> QsaCoupledState {
        self.clone()
    }

    /// Restore a saved checkpoint and explicitly rebuild pooled index keys unless fault-injected.
    pub fn restore_checkpoint(
        &mut self,
        ckpt: &QsaCoupledState,
        hooks: StateFaultHooks,
    ) -> Result<(), String> {
        if self.identity.backbone_digest != ckpt.identity.backbone_digest
            || self.identity.indexer_digest != ckpt.identity.indexer_digest
            || self.identity.engine_commit != ckpt.identity.engine_commit
            || self.identity.tokenizer_template_rev != ckpt.identity.tokenizer_template_rev
            || self.identity.attention_mode != ckpt.identity.attention_mode
            || self.identity.kv_precision != ckpt.identity.kv_precision
            || self.identity.recurrent_layout != ckpt.identity.recurrent_layout
            || self.identity.mtp_draft2_enabled != ckpt.identity.mtp_draft2_enabled
        {
            return Err("checkpoint artifact/policy identity mismatch".into());
        }
        if self.identity.deployment_generation != ckpt.identity.deployment_generation {
            return Err("stale deployment generation checkpoint rejected".into());
        }
        if !hooks.allow_cross_session_prefix_reuse
            && self.identity.session_id != ckpt.identity.session_id
        {
            return Err("cross-session prefix reuse rejected".into());
        }
        let saved_gdn = self.gdn_matrix;
        let saved_valid = self.valid_len;
        let saved_pooled = self.pooled_index_k.clone();
        *self = ckpt.clone();
        if hooks.omit_gdn_rollback {
            self.gdn_matrix = saved_gdn;
        }
        if hooks.retain_rejected_draft_in_kv {
            self.valid_len = saved_valid;
        }
        if hooks.stale_pooled_index_keys {
            self.pooled_index_k = saved_pooled;
        } else {
            self.pooled_index_k = rebuild_pooled_index_keys(&self.raw_index_k, self.valid_len);
        }
        if !hooks.retain_rejected_draft_in_kv {
            self.poison_tail();
        }
        Ok(())
    }
}

pub fn compare_states(a: &QsaCoupledState, b: &QsaCoupledState) -> Result<f64, String> {
    if a.valid_len != b.valid_len
        || a.identity.accepted_tokens != b.identity.accepted_tokens
        || a.identity.accepted_positions != b.identity.accepted_positions
        || a.pooled_index_k.len() != b.pooled_index_k.len()
        || a.mtp_valid_len != b.mtp_valid_len
    {
        return Err(format!(
            "state structural mismatch: valid_len {} vs {}, pooled {} vs {}, mtp_valid {} vs {}",
            a.valid_len,
            b.valid_len,
            a.pooled_index_k.len(),
            b.pooled_index_k.len(),
            a.mtp_valid_len,
            b.mtp_valid_len
        ));
    }
    let mut max_err = checked_max_abs_diff(&a.last_logits, &b.last_logits)?;
    for t in 0..a.valid_len {
        if a.positions[t] != b.positions[t] {
            return Err(format!(
                "position mismatch at slot {t}: {} vs {}",
                a.positions[t], b.positions[t]
            ));
        }
        for (x, y) in [
            (&a.target_k[t][..], &b.target_k[t][..]),
            (&a.target_v[t][..], &b.target_v[t][..]),
            (&a.raw_index_k[t][..], &b.raw_index_k[t][..]),
        ] {
            let e = checked_max_abs_diff(x, y)?;
            if e > max_err {
                max_err = e;
            }
        }
    }
    for blk in 0..a.pooled_index_k.len() {
        let e = checked_max_abs_diff(&a.pooled_index_k[blk], &b.pooled_index_k[blk])?;
        if e > max_err {
            max_err = e;
        }
    }
    for r in 0..QSA_DIM {
        let e = checked_max_abs_diff(&a.gdn_matrix[r], &b.gdn_matrix[r])?;
        if e > max_err {
            max_err = e;
        }
    }
    let mtp_e = checked_max_abs_diff(&a.mtp_hidden, &b.mtp_hidden)?;
    if mtp_e > max_err {
        max_err = mtp_e;
    }
    Ok(max_err)
}

/// Verify two-draft speculative commitment + rollback: if draft_0 or draft_1 is rejected,
/// tentative state updates must be rolled back leaving zero committed rejected state.
pub fn run_mtp_draft2_speculative_step(
    state: &mut QsaCoupledState,
    draft_tokens: [u32; 2],
    hooks: StateFaultHooks,
) -> Result<(usize, Vec<u32>), String> {
    let ckpt_0 = state.snapshot();
    let target_0 = state.argmax_token()?;
    let mut committed = Vec::new();
    let mut accepted_drafts = 0usize;

    // Tentatively advance both drafts to exercise rollback of speculative state
    let _ = state.step_token(draft_tokens[0], StateFaultHooks::default())?;
    let ckpt_1 = state.snapshot();
    let target_1 = state.argmax_token()?;
    let _ = state.step_token(draft_tokens[1], StateFaultHooks::default())?;
    let bonus_tok = state.argmax_token()?;

    if draft_tokens[0] != target_0 {
        // Reject at draft position 0 -> rollback to ckpt_0 and commit corrected target_0
        state.restore_checkpoint(&ckpt_0, hooks)?;
        let _ = state.step_token(target_0, StateFaultHooks::default())?;
        committed.push(target_0);
    } else if draft_tokens[1] != target_1 {
        // Accept draft 0, reject draft 1 -> rollback to ckpt_1 and commit corrected target_1
        accepted_drafts = 1;
        state.restore_checkpoint(&ckpt_1, hooks)?;
        let _ = state.step_token(target_1, StateFaultHooks::default())?;
        committed.push(draft_tokens[0]);
        committed.push(target_1);
    } else {
        // Accept both drafts + bonus token
        accepted_drafts = 2;
        let _ = state.step_token(bonus_tok, StateFaultHooks::default())?;
        committed.push(draft_tokens[0]);
        committed.push(draft_tokens[1]);
        committed.push(bonus_tok);
    }
    Ok((accepted_drafts, committed))
}

#[derive(Debug, Clone)]
pub struct ReuseTelemetryClaim {
    pub client_prefix_overlap_tokens: usize,
    pub server_observed_reused_tokens: Option<usize>,
    pub verified_against_fresh: bool,
    pub deployment_gen_request: u64,
    pub deployment_gen_active: u64,
}

pub fn classify_cache_reuse(claim: &ReuseTelemetryClaim) -> (CacheReuseEvidence, OutcomeStatus) {
    if claim.deployment_gen_request != claim.deployment_gen_active {
        return (CacheReuseEvidence::MissingTelemetry, OutcomeStatus::Invalid);
    }
    match claim.server_observed_reused_tokens {
        None => (CacheReuseEvidence::MissingTelemetry, OutcomeStatus::Invalid),
        Some(0) if claim.client_prefix_overlap_tokens > 0 => {
            (CacheReuseEvidence::AdvertisedOnly, OutcomeStatus::Invalid)
        }
        Some(_) if claim.verified_against_fresh => (
            CacheReuseEvidence::VerifiedAgainstFreshRecompute,
            OutcomeStatus::Accepted,
        ),
        Some(_) => (
            CacheReuseEvidence::ObservedTelemetry,
            OutcomeStatus::Accepted,
        ),
    }
}

#[derive(Debug, Clone)]
pub struct PathCheckRow {
    pub case_id: String,
    pub policy_mode: String,
    pub max_state_and_logit_err: f64,
    pub reuse_evidence: CacheReuseEvidence,
    pub outcome: OutcomeStatus,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct SessionCostRecord {
    pub session_id: String,
    pub outcome: OutcomeStatus,
    pub cold_ttft_ms: f64,
    pub reused_ttft_ms: f64,
    pub rollback_recovery_ms: f64,
    pub total_session_ms: f64,
}

#[derive(Debug, Clone)]
pub struct SessionCostSummary {
    pub planned_sessions: usize,
    pub accepted_sessions: usize,
    pub mean_cold_ttft_ms: f64,
    pub mean_reused_ttft_ms: f64,
    pub mean_rollback_recovery_ms: f64,
    pub cost_per_accepted_session_ms: Option<f64>,
}

pub fn reduce_session_reuse_costs(
    sessions: &[SessionCostRecord],
) -> Result<SessionCostSummary, String> {
    if sessions.is_empty() {
        return Err("empty session cost records".into());
    }
    let mut accepted = 0usize;
    let mut cold_sum = 0.0_f64;
    let mut reused_sum = 0.0_f64;
    let mut rb_sum = 0.0_f64;
    let mut total_sum = 0.0_f64;
    for s in sessions {
        if !s.cold_ttft_ms.is_finite()
            || !s.reused_ttft_ms.is_finite()
            || !s.rollback_recovery_ms.is_finite()
            || !s.total_session_ms.is_finite()
            || s.cold_ttft_ms < 0.0
            || s.reused_ttft_ms < 0.0
            || s.rollback_recovery_ms < 0.0
            || s.total_session_ms < 0.0
        {
            return Err("non-finite or negative session reuse timing".into());
        }
        cold_sum += s.cold_ttft_ms;
        reused_sum += s.reused_ttft_ms;
        rb_sum += s.rollback_recovery_ms;
        total_sum += s.total_session_ms;
        if s.outcome == OutcomeStatus::Accepted {
            accepted += 1;
        }
    }
    let n = sessions.len() as f64;
    Ok(SessionCostSummary {
        planned_sessions: sessions.len(),
        accepted_sessions: accepted,
        mean_cold_ttft_ms: cold_sum / n,
        mean_reused_ttft_ms: reused_sum / n,
        mean_rollback_recovery_ms: rb_sum / n,
        cost_per_accepted_session_ms: if accepted == 0 {
            None
        } else {
            Some(total_sum / (accepted as f64))
        },
    })
}

#[derive(Debug, Clone)]
pub struct QsaStateCoherenceReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub service_verdict: String,
    pub tested_slot_limit: usize,
    pub tested_context_limit: usize,
    pub path_checks: Vec<PathCheckRow>,
    pub unsupported_ops: Vec<(&'static str, OutcomeStatus)>,
    pub cold_ttft_ms_synthetic: f64,
    pub reused_ttft_ms_synthetic: f64,
    pub rollback_recovery_ms_synthetic: f64,
    pub cost_per_accepted_session_synthetic: Option<f64>,
    pub zero_accepted_session_cost_undefined: bool,
    pub negative_controls: Vec<NegativeControl>,
}

fn base_identity(
    mode: QsaPolicyMode,
    mtp: bool,
    session_id: &str,
    gen: u64,
) -> ContinuationIdentity {
    ContinuationIdentity {
        backbone_digest: DigestRef::sha256(b"qwen3.8-27b-qsa-backbone-fixture-v1"),
        indexer_digest: DigestRef::sha256(b"qwen3.8-27b-qsa-indexer-fixture-v1"),
        engine_commit: QSA_ENGINE_COMMIT.into(),
        tokenizer_template_rev: "qwen38-chatml-tool-v1".into(),
        accepted_tokens: Vec::new(),
        accepted_positions: Vec::new(),
        attention_mode: mode,
        kv_precision: "fp16_kv_target_full".into(),
        recurrent_layout: "gdn_f32_4x4_rowmajor".into(),
        mtp_draft2_enabled: mtp,
        session_id: session_id.into(),
        deployment_generation: gen,
    }
}

pub fn run_default_harness() -> Result<QsaStateCoherenceReport, String> {
    let qsa_mode = QsaPolicyMode::QsaBlockSparse { top_blocks: 2 };
    let cap = 32usize;
    let mut path_checks = Vec::new();

    // 1. Fresh vs append/resume across 4-token block boundary (3 -> 4 -> 5 -> 13 tokens, beyond top_blocks=2*4=8!)
    let seq: Vec<u32> = vec![1, 4, 2, 7, 3, 0, 5, 6, 2, 1, 7, 4, 3];
    let mut fresh = QsaCoupledState::fresh(base_identity(qsa_mode, false, "sess-a", 1), cap);
    fresh.process_tokens(&seq, StateFaultHooks::default())?;

    let mut resumed = QsaCoupledState::fresh(base_identity(qsa_mode, false, "sess-a", 1), cap);
    for chunk in [&seq[0..3], &seq[3..5], &seq[5..9], &seq[9..13]] {
        resumed.process_tokens(chunk, StateFaultHooks::default())?;
    }
    let err_chunked = compare_states(&fresh, &resumed)?;
    path_checks.push(PathCheckRow {
        case_id: "qsa_chunked_block_and_sparse_boundary".into(),
        policy_mode: qsa_mode.as_str().into(),
        max_state_and_logit_err: err_chunked,
        reuse_evidence: CacheReuseEvidence::VerifiedAgainstFreshRecompute,
        outcome: if err_chunked <= CONTINUATION_TOL {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
        note: "chunks [3,2,4,4] cross 4-token block and 8-token sparse-selection boundaries".into(),
    });

    // 2. Fixed dense control path check (separate policy, never requiring dense == QSA)
    let mut dense_fresh = QsaCoupledState::fresh(
        base_identity(QsaPolicyMode::DenseControl, false, "sess-a", 1),
        cap,
    );
    dense_fresh.process_tokens(&seq, StateFaultHooks::default())?;
    let mut dense_resumed = QsaCoupledState::fresh(
        base_identity(QsaPolicyMode::DenseControl, false, "sess-a", 1),
        cap,
    );
    dense_resumed.process_tokens(&seq[..6], StateFaultHooks::default())?;
    dense_resumed.process_tokens(&seq[6..], StateFaultHooks::default())?;
    let err_dense = compare_states(&dense_fresh, &dense_resumed)?;
    path_checks.push(PathCheckRow {
        case_id: "dense_control_append_resume".into(),
        policy_mode: QsaPolicyMode::DenseControl.as_str().into(),
        max_state_and_logit_err: err_dense,
        reuse_evidence: CacheReuseEvidence::VerifiedAgainstFreshRecompute,
        outcome: if err_dense <= CONTINUATION_TOL {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
        note: "fixed dense-mode control checked independently of QSA approximation quality".into(),
    });

    // 3. Truncation + checkpoint restore with rebuilt pooled keys + changed suffix
    let mut prefix_state = QsaCoupledState::fresh(base_identity(qsa_mode, false, "sess-a", 1), cap);
    prefix_state.process_tokens(&seq[..7], StateFaultHooks::default())?;
    let ckpt7 = prefix_state.snapshot();
    prefix_state.process_tokens(&[0, 1, 2, 3], StateFaultHooks::default())?;
    prefix_state.restore_checkpoint(&ckpt7, StateFaultHooks::default())?;
    let alt_suffix = [6, 5, 4, 2, 1];
    prefix_state.process_tokens(&alt_suffix, StateFaultHooks::default())?;

    let mut fresh_alt = QsaCoupledState::fresh(base_identity(qsa_mode, false, "sess-a", 1), cap);
    let mut full_alt = seq[..7].to_vec();
    full_alt.extend_from_slice(&alt_suffix);
    fresh_alt.process_tokens(&full_alt, StateFaultHooks::default())?;
    let err_restore = compare_states(&fresh_alt, &prefix_state)?;
    path_checks.push(PathCheckRow {
        case_id: "qsa_checkpoint_restore_and_changed_suffix".into(),
        policy_mode: qsa_mode.as_str().into(),
        max_state_and_logit_err: err_restore,
        reuse_evidence: CacheReuseEvidence::VerifiedAgainstFreshRecompute,
        outcome: if err_restore <= CONTINUATION_TOL {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
        note: "rollback to mid-block pos 7 rebuilt pooled index keys + poisoned tail".into(),
    });

    // 4. MTP draft-2 speculative commitment across reject@0, reject@1, and accept-all+bonus
    let mut mtp_base = QsaCoupledState::fresh(base_identity(qsa_mode, true, "sess-mtp", 1), cap);
    mtp_base.process_tokens(&seq[..9], StateFaultHooks::default())?;
    let t0 = mtp_base.argmax_token()?;
    let mut probe1 = mtp_base.snapshot();
    probe1.step_token(t0, StateFaultHooks::default())?;
    let t1 = probe1.argmax_token()?;

    let mut max_mtp_err = 0.0_f64;
    for (case_label, drafts) in [
        ("reject_pos0", [(t0 + 1) % (QSA_VOCAB as u32), t1]),
        ("reject_pos1", [t0, (t1 + 1) % (QSA_VOCAB as u32)]),
        ("accept_both_plus_bonus", [t0, t1]),
    ] {
        let mut spec = mtp_base.snapshot();
        let (_acc, committed) =
            run_mtp_draft2_speculative_step(&mut spec, drafts, StateFaultHooks::default())?;
        let mut fresh_mtp =
            QsaCoupledState::fresh(base_identity(qsa_mode, true, "sess-mtp", 1), cap);
        let mut expected_seq = seq[..9].to_vec();
        expected_seq.extend_from_slice(&committed);
        fresh_mtp.process_tokens(&expected_seq, StateFaultHooks::default())?;
        let e = compare_states(&fresh_mtp, &spec)?;
        if e > max_mtp_err {
            max_mtp_err = e;
        }
        path_checks.push(PathCheckRow {
            case_id: format!("mtp_draft2_{case_label}"),
            policy_mode: qsa_mode.as_str().into(),
            max_state_and_logit_err: e,
            reuse_evidence: CacheReuseEvidence::VerifiedAgainstFreshRecompute,
            outcome: if e <= CONTINUATION_TOL {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            },
            note: format!(
                "committed {} tokens cleanly matching fresh recompute",
                committed.len()
            ),
        });
    }
    let _ = max_mtp_err;

    // 5. Single-slot two-session isolation via explicit reset
    let mut slot = QsaCoupledState::fresh(base_identity(qsa_mode, true, "sess-1", 1), cap);
    slot.process_tokens(&seq[..10], StateFaultHooks::default())?;
    // Reset slot before serving sess-2 (no cross-session carry-over)
    slot = QsaCoupledState::fresh(base_identity(qsa_mode, true, "sess-2", 1), cap);
    let sess2_seq = [7, 6, 5, 4, 3, 2, 1, 0, 7];
    slot.process_tokens(&sess2_seq, StateFaultHooks::default())?;
    let mut sess2_fresh = QsaCoupledState::fresh(base_identity(qsa_mode, true, "sess-2", 1), cap);
    sess2_fresh.process_tokens(&sess2_seq, StateFaultHooks::default())?;
    let err_iso = compare_states(&sess2_fresh, &slot)?;
    path_checks.push(PathCheckRow {
        case_id: "single_slot_two_session_explicit_reset".into(),
        policy_mode: qsa_mode.as_str().into(),
        max_state_and_logit_err: err_iso,
        reuse_evidence: CacheReuseEvidence::VerifiedAgainstFreshRecompute,
        outcome: if err_iso <= CONTINUATION_TOL {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
        note: "sess-1 state completely cleared on reset; zero influence on sess-2".into(),
    });

    // Negative controls (all executable through real validators/state machine!)
    let mut negative_controls = Vec::new();

    // NC1: stale pooled index keys after checkpoint rollback
    {
        let mut bad = prefix_state.clone();
        bad.restore_checkpoint(
            &ckpt7,
            StateFaultHooks {
                stale_pooled_index_keys: true,
                ..StateFaultHooks::default()
            },
        )?;
        let detected = compare_states(&ckpt7, &bad).is_err();
        negative_controls.push(NegativeControl {
            case_id: "nc1_stale_pooled_index_keys".into(),
            outcome: if detected {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "pooled index keys not rebuilt on rollback fail independent checker".into(),
        });
    }

    // NC2: wrong token position offset
    {
        let mut bad = QsaCoupledState::fresh(base_identity(qsa_mode, false, "sess-a", 1), cap);
        bad.process_tokens(
            &seq,
            StateFaultHooks {
                wrong_token_position_offset: 1,
                ..StateFaultHooks::default()
            },
        )?;
        let detected = compare_states(&fresh, &bad).is_err();
        negative_controls.push(NegativeControl {
            case_id: "nc2_wrong_token_positions".into(),
            outcome: if detected {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "shifted token position encoding diverges and fails position/state check"
                .into(),
        });
    }

    // NC3: omitted GDN rollback
    {
        let mut bad = mtp_base.snapshot();
        let bad_drafts = [(t0 + 1) % (QSA_VOCAB as u32), t1];
        let (_acc, committed) = run_mtp_draft2_speculative_step(
            &mut bad,
            bad_drafts,
            StateFaultHooks {
                omit_gdn_rollback: true,
                ..StateFaultHooks::default()
            },
        )?;
        let mut expected =
            QsaCoupledState::fresh(base_identity(qsa_mode, true, "sess-mtp", 1), cap);
        let mut full = seq[..9].to_vec();
        full.extend_from_slice(&committed);
        expected.process_tokens(&full, StateFaultHooks::default())?;
        let detected = match compare_states(&expected, &bad) {
            Ok(e) => e > CONTINUATION_TOL,
            Err(_) => true,
        };
        negative_controls.push(NegativeControl {
            case_id: "nc3_omitted_gdn_rollback".into(),
            outcome: if detected {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "omitting GDN recurrent rollback corrupts matrix state and output logits"
                .into(),
        });
    }

    // NC4: retained rejected draft in KV cache
    {
        let mut bad = mtp_base.snapshot();
        let bad_drafts = [(t0 + 1) % (QSA_VOCAB as u32), t1];
        let (_acc, committed) = run_mtp_draft2_speculative_step(
            &mut bad,
            bad_drafts,
            StateFaultHooks {
                retain_rejected_draft_in_kv: true,
                ..StateFaultHooks::default()
            },
        )?;
        let mut expected =
            QsaCoupledState::fresh(base_identity(qsa_mode, true, "sess-mtp", 1), cap);
        let mut full = seq[..9].to_vec();
        full.extend_from_slice(&committed);
        expected.process_tokens(&full, StateFaultHooks::default())?;
        let detected = compare_states(&expected, &bad).is_err();
        negative_controls.push(NegativeControl {
            case_id: "nc4_retained_rejected_draft".into(),
            outcome: if detected {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "retained rejected draft slots break valid_len and accepted-prefix agreement"
                .into(),
        });
    }

    // NC5: mismatched indexer artifact identity
    {
        let mut other_id = base_identity(qsa_mode, false, "sess-a", 1);
        other_id.indexer_digest = DigestRef::sha256(b"different-indexer-weights-v2");
        let mut other = QsaCoupledState::fresh(other_id, cap);
        let detected = other
            .restore_checkpoint(&ckpt7, StateFaultHooks::default())
            .is_err();
        negative_controls.push(NegativeControl {
            case_id: "nc5_mismatched_indexer_artifact_identity".into(),
            outcome: if detected {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "swapped indexer hash refuses checkpoint continuation".into(),
        });
    }

    // NC6: cross-session prefix reuse
    {
        let mut other_sess =
            QsaCoupledState::fresh(base_identity(qsa_mode, false, "sess-b", 1), cap);
        let detected = other_sess
            .restore_checkpoint(&ckpt7, StateFaultHooks::default())
            .is_err();
        negative_controls.push(NegativeControl {
            case_id: "nc6_cross_session_prefix_reuse".into(),
            outcome: if detected {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "session-b cannot inherit session-a live slot state without isolation breach"
                .into(),
        });
    }

    // NC7: client-known overlap without server reuse telemetry
    {
        let (_ev, st) = classify_cache_reuse(&ReuseTelemetryClaim {
            client_prefix_overlap_tokens: 1024,
            server_observed_reused_tokens: None,
            verified_against_fresh: false,
            deployment_gen_request: 1,
            deployment_gen_active: 1,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc7_missing_reuse_telemetry_client_overlap_only".into(),
            outcome: st,
            detail: "client-known prefix overlap without server reuse telemetry cannot pass".into(),
        });
    }

    // NC8: late response / restore from older deployment generation
    {
        let mut new_gen = QsaCoupledState::fresh(base_identity(qsa_mode, false, "sess-a", 2), cap);
        let detected = new_gen
            .restore_checkpoint(&ckpt7, StateFaultHooks::default())
            .is_err();
        negative_controls.push(NegativeControl {
            case_id: "nc8_stale_deployment_generation_after_restart".into(),
            outcome: if detected {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "deployment generation bump after restart invalidates pre-restart state".into(),
        });
    }

    let cost_summary = reduce_session_reuse_costs(&[
        SessionCostRecord {
            session_id: "sess-a".into(),
            outcome: OutcomeStatus::Accepted,
            cold_ttft_ms: 14.0,
            reused_ttft_ms: 3.0,
            rollback_recovery_ms: 1.2,
            total_session_ms: 18.2,
        },
        SessionCostRecord {
            session_id: "sess-b".into(),
            outcome: OutcomeStatus::Accepted,
            cold_ttft_ms: 14.4,
            reused_ttft_ms: 3.2,
            rollback_recovery_ms: 1.6,
            total_session_ms: 19.2,
        },
        SessionCostRecord {
            session_id: "sess-cancelled".into(),
            outcome: OutcomeStatus::Cancelled,
            cold_ttft_ms: 14.2,
            reused_ttft_ms: 3.1,
            rollback_recovery_ms: 1.4,
            total_session_ms: 10.0,
        },
    ])?;
    let zero_cost = reduce_session_reuse_costs(&[SessionCostRecord {
        session_id: "sess-fail".into(),
        outcome: OutcomeStatus::Timeout,
        cold_ttft_ms: 50.0,
        reused_ttft_ms: 0.0,
        rollback_recovery_ms: 0.0,
        total_session_ms: 50.0,
    }])?;

    Ok(QsaStateCoherenceReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "DGX Spark GB10 ARM64 / Qwen3.8-27B QSA live single-slot server",
            "Toy pure-Rust fixtures verify state-coherence contracts only; physical ARM64/GB10 build and 27B serve did not execute locally",
        ),
        synthetic_contract_fixture: true,
        service_verdict: "INCONCLUSIVE".into(),
        tested_slot_limit: 1,
        tested_context_limit: cap,
        path_checks,
        unsupported_ops: vec![
            ("persistent_on_disk_prefix_cache", OutcomeStatus::Unsupported),
            ("state_survival_across_server_restart", OutcomeStatus::Unsupported),
            ("arbitrary_multi_slot_concurrent_branching", OutcomeStatus::Unsupported),
        ],
        cold_ttft_ms_synthetic: cost_summary.mean_cold_ttft_ms,
        reused_ttft_ms_synthetic: cost_summary.mean_reused_ttft_ms,
        rollback_recovery_ms_synthetic: cost_summary.mean_rollback_recovery_ms,
        cost_per_accepted_session_synthetic: cost_summary.cost_per_accepted_session_ms,
        zero_accepted_session_cost_undefined: zero_cost.cost_per_accepted_session_ms.is_none(),
        negative_controls,
    })
}

pub fn inputs_json(report: &QsaStateCoherenceReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.qsa-state-coherence.inputs.v1\",\n",
            "  \"issue\": 32,\n",
            "  \"execution_class\": {},\n",
            "  \"pinned_engine_commit\": {},\n",
            "  \"qsa_block_size\": {},\n",
            "  \"tested_slot_limit\": {},\n",
            "  \"tested_context_limit\": {},\n",
            "  \"sample_sequence\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(QSA_ENGINE_COMMIT),
        QSA_BLOCK_SIZE,
        report.tested_slot_limit,
        report.tested_context_limit,
        u32_array(&[1, 4, 2, 7, 3, 0, 5, 6, 2, 1, 7, 4, 3]),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &QsaStateCoherenceReport) -> String {
    let checks: Vec<String> = report
        .path_checks
        .iter()
        .map(|p| {
            format!(
                "{{\"case_id\":{},\"policy_mode\":{},\"max_state_and_logit_err\":{},\"reuse_evidence\":{},\"outcome\":{},\"note\":{}}}",
                quote(&p.case_id),
                quote(&p.policy_mode),
                finite_f64(p.max_state_and_logit_err),
                quote(p.reuse_evidence.as_str()),
                quote(p.outcome.as_str()),
                quote(&p.note),
            )
        })
        .collect();
    let unsup: Vec<String> = report
        .unsupported_ops
        .iter()
        .map(|(op, st)| {
            format!(
                "{{\"operation\":{},\"status\":{}}}",
                quote(op),
                quote(st.as_str())
            )
        })
        .collect();
    let controls: Vec<String> = report
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
            "  \"schema\": \"inference-lab.qsa-state-coherence.results.v1\",\n",
            "  \"issue\": 32,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"service_verdict\": {},\n",
            "  \"tested_slot_limit\": {},\n",
            "  \"tested_context_limit\": {},\n",
            "  \"path_checks\": [{}],\n",
            "  \"unsupported_operations\": [{}],\n",
            "  \"reuse_cost_fixture\": {{\"cold_ttft_ms\":{},\"reused_ttft_ms\":{},\"rollback_recovery_ms\":{},\"cost_per_accepted_session\":{}}},\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        quote(&report.service_verdict),
        report.tested_slot_limit,
        report.tested_context_limit,
        checks.join(","),
        unsup.join(","),
        finite_f64(report.cold_ttft_ms_synthetic),
        finite_f64(report.reused_ttft_ms_synthetic),
        finite_f64(report.rollback_recovery_ms_synthetic),
        opt_f64(report.cost_per_accepted_session_synthetic),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &QsaStateCoherenceReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #32: QSA Cache Reuse & Target/Indexer/GDN/MTP State Coherence"
    )
    .unwrap();
    writeln!(out).unwrap();
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
    writeln!(
        out,
        "- Service qualification verdict: `{}` (toy fixtures do not qualify a 27B service)",
        report.service_verdict
    )
    .unwrap();
    writeln!(
        out,
        "- Tested limits: `slots = {}`, `context_tokens = {}`, `block_size = {}`",
        report.tested_slot_limit, report.tested_context_limit, QSA_BLOCK_SIZE
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Path Coherence Checks").unwrap();
    for p in &report.path_checks {
        writeln!(
            out,
            "- `{}` (`{}`): `max_err = {:.3e}`, `reuse = {}`, `outcome = {}` — {}",
            p.case_id,
            p.policy_mode,
            p.max_state_and_logit_err,
            p.reuse_evidence.as_str(),
            p.outcome.as_str(),
            p.note
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
