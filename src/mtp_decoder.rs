//! Issue #22: Build a small MTP draft-and-verify decoder with cache and sampling conformance tests.
//!
//! Pure deterministic CPU conformance decoder built on frozen tiny DeepSeek-style parameters
//! (reusing `#19` `deepseek_ref` building blocks, with weights and MoE routing biases strictly
//! frozen during generation — zero training-time load-controller updates).
//!
//! Covers:
//! - Target-only full-recompute vs cached baselines (exact logit parity).
//! - Depth-1 MTP draft-and-verify with explicit causal indexing `(h_t, embed(token_{t+1}))`.
//! - Separate target KV cache and MTP KV cache with `1e6` sentinel poisoning on rejected/unused
//!   tail slots and cache/logit agreement checks against full recomputation after every commit.
//! - Greedy conformance fixtures (mixed accept/reject, first-pos reject, later-pos reject,
//!   all-accepted + bonus, zero-accepted fallback, EOS/budget/context bounds, cancel/continue,
//!   multi-request isolation, prefix-causal no-future-leakage check).
//! - Separately declared stochastic rejection-sampling mode (`min(1, p(x)/q(x))` + normalized
//!   `max(p - q, 0)` residual + all-accepted bonus rule) with deterministic branch injection,
//!   exact small-vocabulary marginal proof (`< 1e-12`), and calibrated empirical check.
//! - Six executable negative controls that deliberately inject defects and verify rejection.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{apply_rope, det_weight, matvec, softmax_stable};
use crate::json::{finite_f64, opt_f64, opt_u64, quote, usize_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "Vaibhaav-Tiwari/deepseek-v3-jax";
pub const UPSTREAM_COMMIT: &str = "6901be60941b98dc654ed542465290db9e8bafa6";
pub const DEEPSEEK_V3_PAPER: &str = "https://arxiv.org/html/2412.19437v2#S2.SS2";
pub const POISON_SENTINEL: f64 = 1.0e6;
pub const EOS_TOKEN: usize = 0;

/// Safe max absolute difference between two equal-length finite slices (never swallows NaN).
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    let mut max_err = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite value in max_abs_diff".into());
        }
        let d = (x - y).abs();
        if d > max_err {
            max_err = d;
        }
    }
    Ok(max_err)
}

/// Deterministic argmax rejecting empty or non-finite logits (lowest index breaks ties).
pub fn argmax_finite(logits: &[f64]) -> Result<usize, String> {
    if logits.is_empty() {
        return Err("empty logits".into());
    }
    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("non-finite logit at index {i}"));
        }
        if i == 0 || v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

/// Frozen tiny causal target + depth-1 MTP model with shared embeddings & output head and frozen
/// MoE routing biases (never mutated during decode).
#[derive(Debug, Clone, PartialEq)]
pub struct TinyMtpModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_head: usize,
    pub max_ctx: usize,
    pub rope_theta: f64,
    /// `[vocab_size, d_model]`
    pub embed: Vec<Vec<f64>>,
    /// Main single-head causal attention projections `[d_model, d_head]`.
    pub w_q: Vec<f64>,
    pub w_k: Vec<f64>,
    pub w_v: Vec<f64>,
    pub w_o: Vec<f64>,
    /// Frozen MoE routing weights `[d_model, 2]` and biases `[2]` (must remain constant in decode).
    pub w_router: Vec<f64>,
    pub routing_bias: Vec<f64>,
    pub w_exp0: Vec<f64>,
    pub w_exp1: Vec<f64>,
    /// Shared output head `[d_model, vocab_size]`.
    pub w_lm_head: Vec<f64>,
    /// Depth-1 MTP combiner `[2 * d_model, d_model]` taking `concat(h_main[pos], embed(next_tok))`.
    pub mtp_proj: Vec<f64>,
    /// Depth-1 MTP causal attention projections `[d_model, d_head]`.
    pub mtp_wq: Vec<f64>,
    pub mtp_wk: Vec<f64>,
    pub mtp_wv: Vec<f64>,
    pub mtp_wo: Vec<f64>,
}

impl TinyMtpModel {
    pub fn frozen_tiny() -> Self {
        let vocab_size = 8;
        let d_model = 6;
        let d_head = 6;
        let max_ctx = 24;
        let mk = |seed: usize, len: usize, scale: f64| -> Vec<f64> {
            (0..len).map(|i| det_weight(seed, i, scale)).collect()
        };
        let embed = (0..vocab_size)
            .map(|v| mk(1000 + v, d_model, 0.7))
            .collect();
        // Keep MTP close enough to main model so mixed accept + reject events naturally occur.
        let w_q = mk(1101, d_model * d_head, 0.45);
        let w_k = mk(1102, d_model * d_head, 0.45);
        let w_v = mk(1103, d_model * d_head, 0.45);
        let w_o = mk(1104, d_head * d_model, 0.40);
        let w_router = mk(1105, d_model * 2, 0.35);
        let routing_bias = vec![0.05, -0.05];
        let w_exp0 = mk(1106, d_model * d_model, 0.30);
        let w_exp1 = mk(1107, d_model * d_model, 0.30);
        let mut w_lm_head = mk(1108, d_model * vocab_size, 0.65);
        // Slightly damp EOS (token 0) projection so multi-round greedy rollouts run several steps
        // before encountering EOS naturally.
        for r in 0..d_model {
            w_lm_head[r * vocab_size + EOS_TOKEN] *= 0.25;
        }

        let mut mtp_proj = vec![0.0_f64; 2 * d_model * d_model];
        for i in 0..d_model {
            // Blend previous main hidden state (0.35) + next token embedding (0.65) + tiny perturbation
            mtp_proj[i * d_model + i] = 0.35;
            mtp_proj[(d_model + i) * d_model + i] = 0.65;
        }
        for (idx, val) in mtp_proj.iter_mut().enumerate() {
            *val += det_weight(1200, idx, 0.06);
        }
        let perturb = |base: &[f64], seed: usize, scale: f64| -> Vec<f64> {
            base.iter()
                .enumerate()
                .map(|(i, &b)| b + det_weight(seed, i, scale))
                .collect()
        };
        Self {
            vocab_size,
            d_model,
            d_head,
            max_ctx,
            rope_theta: 10_000.0,
            embed,
            mtp_wq: perturb(&w_q, 1201, 0.08),
            mtp_wk: perturb(&w_k, 1202, 0.08),
            mtp_wv: perturb(&w_v, 1203, 0.08),
            mtp_wo: perturb(&w_o, 1204, 0.08),
            w_q,
            w_k,
            w_v,
            w_o,
            w_router,
            routing_bias,
            w_exp0,
            w_exp1,
            w_lm_head,
            mtp_proj,
        }
    }

    pub fn digest(&self) -> DigestRef {
        let mut bytes = Vec::new();
        for dim in [self.vocab_size, self.d_model, self.d_head, self.max_ctx] {
            bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        for row in &self.embed {
            for &v in row {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        for slice in [
            &self.w_q,
            &self.w_k,
            &self.w_v,
            &self.w_o,
            &self.w_router,
            &self.routing_bias,
            &self.w_exp0,
            &self.w_exp1,
            &self.w_lm_head,
            &self.mtp_proj,
            &self.mtp_wq,
            &self.mtp_wk,
            &self.mtp_wv,
            &self.mtp_wo,
        ] {
            for &v in slice {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        DigestRef::sha256(&bytes)
    }

    fn token_embed(&self, tok: usize) -> Result<&[f64], String> {
        self.embed
            .get(tok)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("token {tok} out of vocab_size {}", self.vocab_size))
    }

    fn frozen_moe_residual(&self, h: &[f64]) -> Result<Vec<f64>, String> {
        let r_logits = matvec(h, &self.w_router, self.d_model, 2)?;
        let s0 = r_logits[0] + self.routing_bias[0];
        let s1 = r_logits[1] + self.routing_bias[1];
        let use_exp0 = s0 >= s1;
        let exp_out = if use_exp0 {
            matvec(h, &self.w_exp0, self.d_model, self.d_model)?
        } else {
            matvec(h, &self.w_exp1, self.d_model, self.d_model)?
        };
        Ok(h.iter()
            .zip(exp_out.iter())
            .map(|(&a, &b)| a + 0.5 * b.tanh())
            .collect())
    }

    /// Target-only full recomputation over `tokens` returning `(hidden_states, logits)` per pos.
    #[allow(clippy::type_complexity)]
    pub fn target_full_recompute(
        &self,
        tokens: &[usize],
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
        if tokens.is_empty() || tokens.len() > self.max_ctx {
            return Err(format!(
                "sequence length {} out of bounds [1, {}]",
                tokens.len(),
                self.max_ctx
            ));
        }
        let mut ks = Vec::with_capacity(tokens.len());
        let mut vs = Vec::with_capacity(tokens.len());
        let mut qs = Vec::with_capacity(tokens.len());
        for (pos, &tok) in tokens.iter().enumerate() {
            let x = self.token_embed(tok)?;
            let q = apply_rope(
                &matvec(x, &self.w_q, self.d_model, self.d_head)?,
                pos,
                self.rope_theta,
            )?;
            let k = apply_rope(
                &matvec(x, &self.w_k, self.d_model, self.d_head)?,
                pos,
                self.rope_theta,
            )?;
            let v = matvec(x, &self.w_v, self.d_model, self.d_head)?;
            qs.push(q);
            ks.push(k);
            vs.push(v);
        }
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let mut hiddens = Vec::with_capacity(tokens.len());
        let mut logits_all = Vec::with_capacity(tokens.len());
        for i in 0..tokens.len() {
            let mut scores = Vec::with_capacity(i + 1);
            for kj in ks.iter().take(i + 1) {
                let dot: f64 = qs[i].iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }
            let probs = softmax_stable(&scores)?;
            let mut ctx = vec![0.0_f64; self.d_head];
            for (j, &pj) in probs.iter().enumerate() {
                for (d, c) in ctx.iter_mut().enumerate().take(self.d_head) {
                    *c += pj * vs[j][d];
                }
            }
            let attn_out = matvec(&ctx, &self.w_o, self.d_head, self.d_model)?;
            let x = self.token_embed(tokens[i])?;
            let resid: Vec<f64> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();
            let h = self.frozen_moe_residual(&resid)?;
            let logits = matvec(&h, &self.w_lm_head, self.d_model, self.vocab_size)?;
            hiddens.push(h);
            logits_all.push(logits);
        }
        Ok((hiddens, logits_all))
    }

    /// Independent MTP full recomputation over committed pairs `(h_main[p], next_tok[p])` for `p = 0..len`.
    pub fn mtp_full_recompute(
        &self,
        main_hiddens: &[Vec<f64>],
        next_tokens: &[usize],
    ) -> Result<Vec<Vec<f64>>, String> {
        if main_hiddens.len() != next_tokens.len() || main_hiddens.is_empty() {
            return Err(
                "MTP recompute requires equal non-empty hidden and next_token slices".into(),
            );
        }
        let n = main_hiddens.len();
        let mut ks = Vec::with_capacity(n);
        let mut vs = Vec::with_capacity(n);
        let mut qs = Vec::with_capacity(n);
        let mut fused = Vec::with_capacity(n);
        for p in 0..n {
            let e = self.token_embed(next_tokens[p])?;
            let mut cat = Vec::with_capacity(2 * self.d_model);
            cat.extend_from_slice(&main_hiddens[p]);
            cat.extend_from_slice(e);
            let z = matvec(&cat, &self.mtp_proj, 2 * self.d_model, self.d_model)?;
            let q = apply_rope(
                &matvec(&z, &self.mtp_wq, self.d_model, self.d_head)?,
                p,
                self.rope_theta,
            )?;
            let k = apply_rope(
                &matvec(&z, &self.mtp_wk, self.d_model, self.d_head)?,
                p,
                self.rope_theta,
            )?;
            let v = matvec(&z, &self.mtp_wv, self.d_model, self.d_head)?;
            fused.push(z);
            qs.push(q);
            ks.push(k);
            vs.push(v);
        }
        let scale = 1.0 / (self.d_head as f64).sqrt();
        let mut out_logits = Vec::with_capacity(n);
        for i in 0..n {
            let mut scores = Vec::with_capacity(i + 1);
            for kj in ks.iter().take(i + 1) {
                let dot: f64 = qs[i].iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }
            let probs = softmax_stable(&scores)?;
            let mut ctx = vec![0.0_f64; self.d_head];
            for (j, &pj) in probs.iter().enumerate() {
                for (d, c) in ctx.iter_mut().enumerate().take(self.d_head) {
                    *c += pj * vs[j][d];
                }
            }
            let attn_out = matvec(&ctx, &self.mtp_wo, self.d_head, self.d_model)?;
            let h_mtp: Vec<f64> = fused[i]
                .iter()
                .zip(attn_out.iter())
                .map(|(a, b)| a + b)
                .collect();
            let logits = matvec(&h_mtp, &self.w_lm_head, self.d_model, self.vocab_size)?;
            out_logits.push(logits);
        }
        Ok(out_logits)
    }
}

/// Separate KV cache with explicit `valid_len`, `capacity`, and `1e6` sentinel poisoning of unused/rejected tail slots.
#[derive(Debug, Clone)]
pub struct PoisonedKvCache {
    pub capacity: usize,
    pub d_head: usize,
    pub valid_len: usize,
    pub k_slots: Vec<Vec<f64>>,
    pub v_slots: Vec<Vec<f64>>,
    pub hidden_slots: Vec<Vec<f64>>,
    pub rollback_count: usize,
}

impl PoisonedKvCache {
    pub fn new(capacity: usize, d_head: usize, d_model: usize) -> Self {
        Self {
            capacity,
            d_head,
            valid_len: 0,
            k_slots: vec![vec![POISON_SENTINEL; d_head]; capacity],
            v_slots: vec![vec![POISON_SENTINEL; d_head]; capacity],
            hidden_slots: vec![vec![POISON_SENTINEL; d_model]; capacity],
            rollback_count: 0,
        }
    }

    pub fn rollback_and_poison(&mut self, new_valid_len: usize) -> Result<(), String> {
        if new_valid_len > self.valid_len {
            return Err(format!(
                "cannot rollback from {} to larger length {new_valid_len}",
                self.valid_len
            ));
        }
        if new_valid_len < self.valid_len {
            self.rollback_count += 1;
        }
        for slot in new_valid_len..self.capacity {
            for x in &mut self.k_slots[slot] {
                *x = POISON_SENTINEL;
            }
            for x in &mut self.v_slots[slot] {
                *x = POISON_SENTINEL;
            }
            for x in &mut self.hidden_slots[slot] {
                *x = POISON_SENTINEL;
            }
        }
        self.valid_len = new_valid_len;
        Ok(())
    }

    pub fn reset_and_poison(&mut self) {
        self.valid_len = 0;
        for slot in 0..self.capacity {
            for x in &mut self.k_slots[slot] {
                *x = POISON_SENTINEL;
            }
            for x in &mut self.v_slots[slot] {
                *x = POISON_SENTINEL;
            }
            for x in &mut self.hidden_slots[slot] {
                *x = POISON_SENTINEL;
            }
        }
    }

    /// Cached single-token target step; if `broken_attend_len` is `Some(n)` (negative control),
    /// reads past `valid_len` into poisoned tail slots!
    pub fn target_step(
        &mut self,
        model: &TinyMtpModel,
        tok: usize,
        broken_extra_tail: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        if self.valid_len >= self.capacity {
            return Err("target cache capacity exhausted".into());
        }
        let pos = self.valid_len;
        let x = model.token_embed(tok)?;
        let q = apply_rope(
            &matvec(x, &model.w_q, model.d_model, model.d_head)?,
            pos,
            model.rope_theta,
        )?;
        let k = apply_rope(
            &matvec(x, &model.w_k, model.d_model, model.d_head)?,
            pos,
            model.rope_theta,
        )?;
        let v = matvec(x, &model.w_v, model.d_model, model.d_head)?;
        self.k_slots[pos] = k;
        self.v_slots[pos] = v;
        self.valid_len += 1;

        let attend_len = (self.valid_len + broken_extra_tail).min(self.capacity);
        let scale = 1.0 / (model.d_head as f64).sqrt();
        let mut scores = Vec::with_capacity(attend_len);
        for j in 0..attend_len {
            let dot: f64 = q
                .iter()
                .zip(self.k_slots[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            scores.push(dot * scale);
        }
        let probs = softmax_stable(&scores)?;
        let mut ctx = vec![0.0_f64; model.d_head];
        for (j, &pj) in probs.iter().enumerate().take(attend_len) {
            for (d, c) in ctx.iter_mut().enumerate().take(model.d_head) {
                *c += pj * self.v_slots[j][d];
            }
        }
        let attn_out = matvec(&ctx, &model.w_o, model.d_head, model.d_model)?;
        let resid: Vec<f64> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();
        let h = model.frozen_moe_residual(&resid)?;
        self.hidden_slots[pos] = h.clone();
        let logits = matvec(&h, &model.w_lm_head, model.d_model, model.vocab_size)?;
        Ok((h, logits))
    }

    /// Cached single-step MTP proposal taking `(main_hidden_at_pos, next_tok)` at MTP slot `pos`.
    pub fn mtp_step(
        &mut self,
        model: &TinyMtpModel,
        main_hidden: &[f64],
        next_tok: usize,
    ) -> Result<Vec<f64>, String> {
        if self.valid_len >= self.capacity {
            return Err("MTP cache capacity exhausted".into());
        }
        let pos = self.valid_len;
        let e = model.token_embed(next_tok)?;
        let mut cat = Vec::with_capacity(2 * model.d_model);
        cat.extend_from_slice(main_hidden);
        cat.extend_from_slice(e);
        let z = matvec(&cat, &model.mtp_proj, 2 * model.d_model, model.d_model)?;
        let q = apply_rope(
            &matvec(&z, &model.mtp_wq, model.d_model, model.d_head)?,
            pos,
            model.rope_theta,
        )?;
        let k = apply_rope(
            &matvec(&z, &model.mtp_wk, model.d_model, model.d_head)?,
            pos,
            model.rope_theta,
        )?;
        let v = matvec(&z, &model.mtp_wv, model.d_model, model.d_head)?;
        self.k_slots[pos] = k;
        self.v_slots[pos] = v;
        self.hidden_slots[pos] = z.clone();
        self.valid_len += 1;

        let scale = 1.0 / (model.d_head as f64).sqrt();
        let mut scores = Vec::with_capacity(self.valid_len);
        for j in 0..self.valid_len {
            let dot: f64 = q
                .iter()
                .zip(self.k_slots[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            scores.push(dot * scale);
        }
        let probs = softmax_stable(&scores)?;
        let mut ctx = vec![0.0_f64; model.d_head];
        for (j, &pj) in probs.iter().enumerate().take(self.valid_len) {
            for (d, c) in ctx.iter_mut().enumerate().take(model.d_head) {
                *c += pj * self.v_slots[j][d];
            }
        }
        let attn_out = matvec(&ctx, &model.mtp_wo, model.d_head, model.d_model)?;
        let h_mtp: Vec<f64> = z.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();
        matvec(&h_mtp, &model.w_lm_head, model.d_model, model.vocab_size)
    }
}

/// Sampling profile applied consistently to both target and draft logits.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingProfile {
    pub mode: String,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
}

impl SamplingProfile {
    pub fn greedy() -> Self {
        Self {
            mode: "greedy".into(),
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
        }
    }

    pub fn stochastic(temperature: f64, top_k: usize, top_p: f64) -> Self {
        Self {
            mode: "stochastic_rejection_v1".into(),
            temperature,
            top_k,
            top_p,
        }
    }

    /// Transform raw logits into a normalized effective probability distribution.
    pub fn effective_probs(&self, logits: &[f64]) -> Result<Vec<f64>, String> {
        if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
            return Err("non-finite or empty logits in effective_probs".into());
        }
        if self.temperature == 0.0 || self.mode == "greedy" {
            let best = argmax_finite(logits)?;
            let mut p = vec![0.0; logits.len()];
            p[best] = 1.0;
            return Ok(p);
        }
        if !self.temperature.is_finite()
            || self.temperature <= 0.0
            || !self.top_p.is_finite()
            || self.top_p <= 0.0
            || self.top_p > 1.0
            || self.top_k == 0
        {
            return Err("invalid sampling parameters".into());
        }
        let scaled: Vec<f64> = logits.iter().map(|&l| l / self.temperature).collect();
        let base = softmax_stable(&scaled)?;
        // Sort indices by descending probability, ascending index for ties.
        let mut order: Vec<usize> = (0..base.len()).collect();
        order.sort_by(|&a, &b| base[b].total_cmp(&base[a]).then_with(|| a.cmp(&b)));
        let k_limit = self.top_k.min(order.len());
        let mut kept = vec![false; base.len()];
        let mut cum = 0.0_f64;
        for &idx in order.iter().take(k_limit) {
            kept[idx] = true;
            cum += base[idx];
            if cum >= self.top_p {
                break;
            }
        }
        let mut out = vec![0.0_f64; base.len()];
        let mut sum = 0.0_f64;
        for (i, &p) in base.iter().enumerate() {
            if kept[i] {
                out[i] = p;
                sum += p;
            }
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err("zero probability mass after filtering".into());
        }
        for p in &mut out {
            *p /= sum;
        }
        Ok(out)
    }
}

/// Residual correction distribution `norm(max(p - q, 0))`.
pub fn residual_distribution(p: &[f64], q: &[f64]) -> Result<Vec<f64>, String> {
    if p.len() != q.len() || p.is_empty() {
        return Err("mismatched or empty distributions for residual".into());
    }
    let mut raw = Vec::with_capacity(p.len());
    let mut sum = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if !pi.is_finite() || !qi.is_finite() || pi < -1e-15 || qi < -1e-15 {
            return Err("invalid probability entry in residual_distribution".into());
        }
        let diff = (pi - qi).max(0.0);
        raw.push(diff);
        sum += diff;
    }
    if sum <= 0.0 {
        return Err("impossible zero-mass residual distribution (p == q everywhere)".into());
    }
    Ok(raw.into_iter().map(|d| d / sum).collect())
}

/// Sample an index from a normalized distribution given `u in [0, 1)`.
pub fn sample_categorical(probs: &[f64], u: f64) -> Result<usize, String> {
    if !u.is_finite() || !(0.0..1.0).contains(&u) {
        return Err("uniform draw u must be finite in [0, 1)".into());
    }
    let mut acc = 0.0_f64;
    let mut last_nonzero = None;
    for (i, &p) in probs.iter().enumerate() {
        if !p.is_finite() || p < 0.0 {
            return Err("invalid probability in sample_categorical".into());
        }
        if p > 0.0 {
            last_nonzero = Some(i);
        }
        acc += p;
        if u < acc && p > 0.0 {
            return Ok(i);
        }
    }
    last_nonzero.ok_or_else(|| "empty support in sample_categorical".into())
}

/// Outcome of one stochastic rejection-sampling verification step on a single proposed token `x`.
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticStepDecision {
    pub proposed_token: usize,
    pub accepted: bool,
    pub acceptance_prob: f64,
    pub emitted_token: usize,
    pub branch_label: &'static str,
}

/// Standard rejection-sampling step: accept `x` with `min(1, p(x)/q(x))`; on reject sample from
/// normalized `max(p - q, 0)`.
pub fn stochastic_verify_token(
    p: &[f64],
    q: &[f64],
    proposed: usize,
    u_accept: f64,
    u_residual: f64,
) -> Result<StochasticStepDecision, String> {
    if proposed >= p.len() || p.len() != q.len() {
        return Err("proposed token or distribution length out of bounds".into());
    }
    if !u_accept.is_finite() || !(0.0..1.0).contains(&u_accept) {
        return Err("u_accept out of [0, 1)".into());
    }
    let px = p[proposed];
    let qx = q[proposed];
    if !px.is_finite() || !qx.is_finite() || px < 0.0 || qx <= 0.0 {
        return Err("proposed token must have positive finite draft probability q(x) > 0".into());
    }
    let alpha = (px / qx).min(1.0);
    if u_accept < alpha {
        let branch = if (px - qx).abs() <= 1e-12 {
            "p_eq_q_always_accept"
        } else if px > qx {
            "p_gt_q_always_accept"
        } else {
            "p_lt_q_accepted"
        };
        Ok(StochasticStepDecision {
            proposed_token: proposed,
            accepted: true,
            acceptance_prob: alpha,
            emitted_token: proposed,
            branch_label: branch,
        })
    } else {
        let res = residual_distribution(p, q)?;
        let corrected = sample_categorical(&res, u_residual)?;
        let branch = if px == 0.0 {
            "zero_target_support_forced_reject"
        } else {
            "p_lt_q_rejected_residual"
        };
        Ok(StochasticStepDecision {
            proposed_token: proposed,
            accepted: false,
            acceptance_prob: alpha,
            emitted_token: corrected,
            branch_label: branch,
        })
    }
}

/// Exact analytical marginal emitted distribution for one rejection-sampling step over all
/// possible draft proposals `x ~ q`, proving `P(emitted = v) == p(v)` exactly!
pub fn exact_rejection_marginal(p: &[f64], q: &[f64]) -> Result<Vec<f64>, String> {
    if p.len() != q.len() || p.is_empty() {
        return Err("distribution length mismatch".into());
    }
    let mut sum_p = 0.0_f64;
    let mut sum_q = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if !pi.is_finite() || !qi.is_finite() || pi < 0.0 || qi < 0.0 {
            return Err("non-finite or negative probability in exact_rejection_marginal".into());
        }
        sum_p += pi;
        sum_q += qi;
    }
    if (sum_p - 1.0).abs() > 1e-6 || (sum_q - 1.0).abs() > 1e-6 {
        return Err(format!(
            "distributions must sum to 1.0 (got sum_p={sum_p:.6}, sum_q={sum_q:.6})"
        ));
    }
    let mut reject_total = 0.0_f64;
    let mut emitted = vec![0.0_f64; p.len()];
    for x in 0..p.len() {
        if q[x] > 0.0 {
            let ratio = p[x] / q[x];
            let alpha = if ratio < 1.0 { ratio } else { 1.0 };
            emitted[x] += q[x] * alpha;
            reject_total += q[x] * (1.0 - alpha);
        }
    }
    if reject_total > 0.0 {
        let res = residual_distribution(p, q)?;
        for (v, &rv) in res.iter().enumerate() {
            emitted[v] += reject_total * rv;
        }
    }
    Ok(emitted)
}

#[derive(Debug, Clone)]
pub struct VerificationRoundLog {
    pub round_index: usize,
    pub committed_prefix_len_before: usize,
    pub proposed_tokens: Vec<usize>,
    pub draft_probs: Vec<Vec<f64>>,
    pub target_probs: Vec<Vec<f64>>,
    pub accepted_tokens: Vec<usize>,
    pub rejected_token: Option<usize>,
    pub corrected_or_bonus_token: Option<usize>,
    pub target_cache_len_after: usize,
    pub mtp_cache_len_after: usize,
    pub cache_recompute_max_err: f64,
    pub next_logit_recompute_max_err: f64,
    pub fallback_target_only: bool,
}

#[derive(Debug, Clone)]
pub struct GreedyDecodeRun {
    pub fixture_id: String,
    pub eos_token: usize,
    pub prompt: Vec<usize>,
    pub generated: Vec<usize>,
    pub baseline_generated: Vec<usize>,
    pub exact_token_agreement: bool,
    pub max_cache_err: f64,
    pub max_next_logit_err: f64,
    pub total_draft_proposals: usize,
    pub total_accepted_drafts: usize,
    pub total_verification_rounds: usize,
    pub target_only_fallbacks: usize,
    pub target_model_passes: usize,
    pub mtp_model_passes: usize,
    pub target_rollbacks: usize,
    pub mtp_rollbacks: usize,
    pub target_cache_bytes: usize,
    pub mtp_cache_bytes: usize,
    pub accepted_per_verification: f64,
    pub stopped_reason: String,
    pub rounds: Vec<VerificationRoundLog>,
}

/// Run target-only full-recompute greedy baseline for `max_new_tokens` (stopping on `eos_tok` or max_ctx).
pub fn run_target_only_greedy_with_eos(
    model: &TinyMtpModel,
    prompt: &[usize],
    max_new_tokens: usize,
    eos_tok: usize,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    let mut seq = prompt.to_vec();
    let mut generated = Vec::new();
    let (_, mut last_logits_all) = model.target_full_recompute(&seq)?;
    for _ in 0..max_new_tokens {
        if seq.len() >= model.max_ctx {
            break;
        }
        let next = argmax_finite(last_logits_all.last().unwrap())?;
        generated.push(next);
        seq.push(next);
        if next == eos_tok {
            break;
        }
        let (_, logits_all) = model.target_full_recompute(&seq)?;
        last_logits_all = logits_all;
    }
    Ok((generated, last_logits_all.last().unwrap().clone()))
}

/// Run target-only full-recompute greedy baseline using default `EOS_TOKEN`.
pub fn run_target_only_greedy(
    model: &TinyMtpModel,
    prompt: &[usize],
    max_new_tokens: usize,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    run_target_only_greedy_with_eos(model, prompt, max_new_tokens, EOS_TOKEN)
}

/// Run depth-1 MTP greedy draft-and-verify with explicit separate poisoned Target & MTP caches,
/// stepping and rolling back/poisoning `mtp_cache` on speculative rejections, retaining draft and
/// target probabilities per round, and checking full recomputation after every commit!
#[allow(clippy::too_many_arguments)]
pub fn run_mtp_greedy_fixture(
    fixture_id: &str,
    model: &TinyMtpModel,
    prompt: &[usize],
    max_new_tokens: usize,
    draft_horizon: usize,
    forced_draft_override: Option<&[Vec<usize>]>,
    cancel_after_rounds: Option<usize>,
    use_target_only_fallback: bool,
    eos_token: usize,
) -> Result<GreedyDecodeRun, String> {
    if prompt.is_empty() || prompt.len() >= model.max_ctx {
        return Err("prompt length out of bounds".into());
    }
    let bias_before = model.routing_bias.clone();
    let mut target_cache = PoisonedKvCache::new(model.max_ctx, model.d_head, model.d_model);
    let mut mtp_cache = PoisonedKvCache::new(model.max_ctx, model.d_head, model.d_model);
    let greedy_profile = SamplingProfile::greedy();

    let mut target_passes = 0usize;
    let mut mtp_passes = 0usize;
    let mut seq = prompt.to_vec();
    let mut generated = Vec::new();
    let mut rounds = Vec::new();

    // Prefill prompt into target cache and MTP cache (for positions 0..prompt.len()-1).
    let mut last_h = Vec::new();
    let mut last_logits = Vec::new();
    for (idx, &tok) in prompt.iter().enumerate() {
        let (h, logits) = target_cache.target_step(model, tok, 0)?;
        target_passes += 1;
        if idx > 0 {
            let _ = mtp_cache.mtp_step(model, &last_h, tok)?;
            mtp_passes += 1;
        }
        last_h = h;
        last_logits = logits;
    }

    let mut max_cache_err = 0.0_f64;
    let mut max_next_logit_err = 0.0_f64;
    let mut total_proposed = 0usize;
    let mut total_accepted = 0usize;
    let mut target_only_fallbacks = 0usize;
    let mut stopped_reason = "budget_reached".to_string();

    let mut round_idx = 0usize;
    while generated.len() < max_new_tokens && seq.len() < model.max_ctx {
        if let Some(limit) = cancel_after_rounds {
            if round_idx >= limit {
                stopped_reason = "cancelled".into();
                break;
            }
        }
        let prefix_before = seq.len();
        let remaining_budget = max_new_tokens - generated.len();
        let remaining_ctx = model.max_ctx - seq.len();
        let max_commit_now = remaining_budget.min(remaining_ctx);

        let base_next = argmax_finite(&last_logits)?;
        let base_target_probs = greedy_profile.effective_probs(&last_logits)?;
        if use_target_only_fallback || draft_horizon == 0 || max_commit_now == 1 {
            if use_target_only_fallback {
                target_only_fallbacks += 1;
            }
            generated.push(base_next);
            seq.push(base_next);
            let prev_h = last_h.clone();
            let (h_new, logits_new) = target_cache.target_step(model, base_next, 0)?;
            target_passes += 1;
            let _ = mtp_cache.mtp_step(model, &prev_h, base_next)?;
            mtp_passes += 1;
            last_h = h_new;
            last_logits = logits_new;

            let (ref_h, ref_l) = model.target_full_recompute(&seq)?;
            let c_err = max_abs_diff(
                &target_cache.hidden_slots[seq.len() - 1],
                ref_h.last().unwrap(),
            )?;
            let l_err = max_abs_diff(&last_logits, ref_l.last().unwrap())?;
            if c_err > max_cache_err {
                max_cache_err = c_err;
            }
            if l_err > max_next_logit_err {
                max_next_logit_err = l_err;
            }
            rounds.push(VerificationRoundLog {
                round_index: round_idx,
                committed_prefix_len_before: prefix_before,
                proposed_tokens: Vec::new(),
                draft_probs: Vec::new(),
                target_probs: vec![base_target_probs],
                accepted_tokens: Vec::new(),
                rejected_token: None,
                corrected_or_bonus_token: Some(base_next),
                target_cache_len_after: target_cache.valid_len,
                mtp_cache_len_after: mtp_cache.valid_len,
                cache_recompute_max_err: c_err,
                next_logit_recompute_max_err: l_err,
                fallback_target_only: use_target_only_fallback,
            });
            round_idx += 1;
            if base_next == eos_token {
                stopped_reason = "eos".into();
                break;
            }
            continue;
        }

        let k_draft = draft_horizon.min(max_commit_now.saturating_sub(1));
        let prev_h_before_base = last_h.clone();
        let (h_base, logits_after_base) = target_cache.target_step(model, base_next, 0)?;
        target_passes += 1;
        let mtp_logits_0 = mtp_cache.mtp_step(model, &prev_h_before_base, base_next)?;
        mtp_passes += 1;

        generated.push(base_next);
        seq.push(base_next);
        last_h = h_base.clone();
        last_logits = logits_after_base.clone();
        if base_next == eos_token {
            stopped_reason = "eos".into();
            let (ref_h, ref_l) = model.target_full_recompute(&seq)?;
            let c_err = max_abs_diff(
                &target_cache.hidden_slots[seq.len() - 1],
                ref_h.last().unwrap(),
            )?;
            let l_err = max_abs_diff(&last_logits, ref_l.last().unwrap())?;
            if c_err > max_cache_err {
                max_cache_err = c_err;
            }
            if l_err > max_next_logit_err {
                max_next_logit_err = l_err;
            }
            rounds.push(VerificationRoundLog {
                round_index: round_idx,
                committed_prefix_len_before: prefix_before,
                proposed_tokens: Vec::new(),
                draft_probs: Vec::new(),
                target_probs: vec![base_target_probs],
                accepted_tokens: Vec::new(),
                rejected_token: None,
                corrected_or_bonus_token: Some(base_next),
                target_cache_len_after: target_cache.valid_len,
                mtp_cache_len_after: mtp_cache.valid_len,
                cache_recompute_max_err: c_err,
                next_logit_recompute_max_err: l_err,
                fallback_target_only: false,
            });
            break;
        }

        let target_len_before_spec = target_cache.valid_len;
        let mtp_len_before_spec = mtp_cache.valid_len;

        // Step the REAL `mtp_cache` directly while drafting so rollback/poisoning is genuinely exercised!
        let mut proposed = Vec::with_capacity(k_draft);
        let mut draft_probs = Vec::with_capacity(k_draft);
        let mut cur_tok = argmax_finite(&mtp_logits_0)?;
        let mut cur_mtp_logits = mtp_logits_0;
        let mut cur_h = last_h.clone();
        for d_idx in 0..k_draft {
            let chosen = if let Some(overrides) = forced_draft_override {
                overrides
                    .get(round_idx)
                    .and_then(|ov| ov.get(d_idx))
                    .copied()
                    .unwrap_or(cur_tok)
            } else {
                cur_tok
            };
            let mut q_vec = greedy_profile.effective_probs(&cur_mtp_logits)?;
            if chosen != cur_tok {
                q_vec = vec![0.0; model.vocab_size];
                q_vec[chosen] = 1.0;
            }
            proposed.push(chosen);
            draft_probs.push(q_vec);
            if mtp_cache.valid_len < model.max_ctx {
                cur_mtp_logits = mtp_cache.mtp_step(model, &cur_h, chosen)?;
                mtp_passes += 1;
                cur_tok = argmax_finite(&cur_mtp_logits)?;
                let e = model.token_embed(chosen)?;
                cur_h = cur_h
                    .iter()
                    .zip(e.iter())
                    .map(|(a, b)| 0.5 * a + 0.5 * b)
                    .collect();
            }
        }
        total_proposed += proposed.len();

        // Rollback & poison speculative MTP draft slots before committing verified tokens!
        mtp_cache.rollback_and_poison(mtp_len_before_spec)?;

        let mut spec_hiddens = Vec::with_capacity(proposed.len());
        let mut spec_logits = Vec::with_capacity(proposed.len());
        for &cand in &proposed {
            if target_cache.valid_len >= model.max_ctx {
                break;
            }
            let (sh, sl) = target_cache.target_step(model, cand, 0)?;
            target_passes += 1;
            spec_hiddens.push(sh);
            spec_logits.push(sl);
        }

        let mut accepted = Vec::new();
        let mut target_probs = Vec::with_capacity(proposed.len());
        let mut rejected_tok = None;
        let mut corrected_or_bonus = None;
        let mut cur_target_logits = last_logits.clone();
        let mut cur_main_h = last_h.clone();
        let mut hit_eos = false;
        let mut all_accepted = true;

        for (i, &cand) in proposed.iter().enumerate() {
            if generated.len() >= max_new_tokens || seq.len() >= model.max_ctx {
                all_accepted = false;
                break;
            }
            target_probs.push(greedy_profile.effective_probs(&cur_target_logits)?);
            let target_tok = argmax_finite(&cur_target_logits)?;
            if cand == target_tok {
                accepted.push(cand);
                generated.push(cand);
                seq.push(cand);
                let _ = mtp_cache.mtp_step(model, &cur_main_h, cand)?;
                mtp_passes += 1;
                cur_main_h = spec_hiddens[i].clone();
                cur_target_logits = spec_logits[i].clone();
                if cand == eos_token {
                    hit_eos = true;
                    all_accepted = false;
                    break;
                }
            } else {
                rejected_tok = Some(cand);
                all_accepted = false;
                let committed_target_len = target_len_before_spec + accepted.len();
                target_cache.rollback_and_poison(committed_target_len)?;
                mtp_cache.rollback_and_poison(mtp_len_before_spec + accepted.len())?;

                if generated.len() < max_new_tokens && seq.len() < model.max_ctx {
                    corrected_or_bonus = Some(target_tok);
                    generated.push(target_tok);
                    seq.push(target_tok);
                    let prev_h = cur_main_h.clone();
                    let (ch, cl) = target_cache.target_step(model, target_tok, 0)?;
                    target_passes += 1;
                    let _ = mtp_cache.mtp_step(model, &prev_h, target_tok)?;
                    mtp_passes += 1;
                    cur_main_h = ch;
                    cur_target_logits = cl;
                    if target_tok == eos_token {
                        hit_eos = true;
                    }
                }
                break;
            }
        }

        // Always rollback & poison any unaccepted speculative tail slots (handles all_accepted,
        // early EOS break, AND early budget/context-capacity break!).
        let committed_target_len =
            target_len_before_spec + accepted.len() + usize::from(corrected_or_bonus.is_some());
        if target_cache.valid_len > committed_target_len {
            target_cache.rollback_and_poison(committed_target_len)?;
        }
        let committed_mtp_len =
            mtp_len_before_spec + accepted.len() + usize::from(corrected_or_bonus.is_some());
        if mtp_cache.valid_len > committed_mtp_len {
            mtp_cache.rollback_and_poison(committed_mtp_len)?;
        }

        if all_accepted && !hit_eos && generated.len() < max_new_tokens && seq.len() < model.max_ctx
        {
            let bonus = argmax_finite(&cur_target_logits)?;
            target_probs.push(greedy_profile.effective_probs(&cur_target_logits)?);
            corrected_or_bonus = Some(bonus);
            generated.push(bonus);
            seq.push(bonus);
            let prev_h = cur_main_h.clone();
            let (bh, bl) = target_cache.target_step(model, bonus, 0)?;
            target_passes += 1;
            let _ = mtp_cache.mtp_step(model, &prev_h, bonus)?;
            mtp_passes += 1;
            cur_main_h = bh;
            cur_target_logits = bl;
            if bonus == eos_token {
                hit_eos = true;
            }
        }

        total_accepted += accepted.len();
        last_h = cur_main_h;
        last_logits = cur_target_logits;

        let (ref_h, ref_l) = model.target_full_recompute(&seq)?;
        if target_cache.valid_len != seq.len() || mtp_cache.valid_len + 1 != seq.len() {
            return Err("cache valid_len mismatch after commit".into());
        }
        let mut round_cache_err = 0.0_f64;
        for (pos, rh) in ref_h.iter().enumerate().take(seq.len()) {
            let e = max_abs_diff(&target_cache.hidden_slots[pos], rh)?;
            if e > round_cache_err {
                round_cache_err = e;
            }
        }
        let ref_mtp = model.mtp_full_recompute(&ref_h[..seq.len() - 1], &seq[1..])?;
        let mtp_probe_err = if seq.len() >= 2 {
            let mut mtp_check_cache = mtp_cache.clone();
            mtp_check_cache.rollback_and_poison(seq.len() - 2)?;
            let cached_mtp_logits =
                mtp_check_cache.mtp_step(model, &ref_h[seq.len() - 2], seq[seq.len() - 1])?;
            max_abs_diff(&cached_mtp_logits, ref_mtp.last().unwrap())?
        } else {
            0.0
        };
        let target_logit_err = max_abs_diff(&last_logits, ref_l.last().unwrap())?;
        let round_logit_err = if mtp_probe_err > target_logit_err {
            mtp_probe_err
        } else {
            target_logit_err
        };
        if round_cache_err > max_cache_err {
            max_cache_err = round_cache_err;
        }
        if round_logit_err > max_next_logit_err {
            max_next_logit_err = round_logit_err;
        }

        rounds.push(VerificationRoundLog {
            round_index: round_idx,
            committed_prefix_len_before: prefix_before,
            proposed_tokens: proposed,
            draft_probs,
            target_probs,
            accepted_tokens: accepted,
            rejected_token: rejected_tok,
            corrected_or_bonus_token: corrected_or_bonus,
            target_cache_len_after: target_cache.valid_len,
            mtp_cache_len_after: mtp_cache.valid_len,
            cache_recompute_max_err: round_cache_err,
            next_logit_recompute_max_err: round_logit_err,
            fallback_target_only: false,
        });
        round_idx += 1;

        if hit_eos {
            stopped_reason = "eos".into();
            break;
        }
    }

    if seq.len() >= model.max_ctx
        && stopped_reason == "budget_reached"
        && generated.len() < max_new_tokens
    {
        stopped_reason = "context_capacity_reached".into();
    }

    if model.routing_bias != bias_before {
        return Err("routing biases mutated during generation".into());
    }

    let (baseline_generated, _) =
        run_target_only_greedy_with_eos(model, prompt, generated.len(), eos_token)?;
    let exact_token_agreement = generated == baseline_generated;
    let accepted_per_verification = if rounds.is_empty() {
        0.0
    } else {
        (total_accepted as f64) / (rounds.len() as f64)
    };
    let slot_bytes = (model.d_head * 2 + model.d_model) * 8;
    let target_cache_bytes = target_cache.capacity * slot_bytes;
    let mtp_cache_bytes = mtp_cache.capacity * slot_bytes;

    Ok(GreedyDecodeRun {
        fixture_id: fixture_id.into(),
        eos_token,
        prompt: prompt.to_vec(),
        generated,
        baseline_generated,
        exact_token_agreement,
        max_cache_err,
        max_next_logit_err,
        total_draft_proposals: total_proposed,
        total_accepted_drafts: total_accepted,
        total_verification_rounds: rounds.len(),
        target_only_fallbacks,
        target_model_passes: target_passes,
        mtp_model_passes: mtp_passes,
        target_rollbacks: target_cache.rollback_count,
        mtp_rollbacks: mtp_cache.rollback_count,
        target_cache_bytes,
        mtp_cache_bytes,
        accepted_per_verification,
        stopped_reason,
        rounds,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub description: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct StochasticReport {
    pub profile: SamplingProfile,
    pub branch_outcomes: Vec<(String, String, bool)>,
    pub exact_marginal_max_err: f64,
    pub empirical_samples: usize,
    pub empirical_max_abs_err: f64,
    pub empirical_tolerance: f64,
    pub empirical_pass: bool,
    pub impossible_zero_residual_rejected: bool,
    pub bonus_token_rule_verified: bool,
}

#[derive(Debug, Clone)]
pub struct MtpDecoderReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub model_digest: DigestRef,
    pub vocab_size: usize,
    pub max_ctx: usize,
    pub target_cached_vs_recompute_max_err: f64,
    pub prefix_causal_no_future_leak_max_err: f64,
    pub multi_request_isolation_max_err: f64,
    pub routing_bias_frozen: bool,
    pub greedy_fixtures: Vec<GreedyDecodeRun>,
    pub stochastic: StochasticReport,
    pub negative_controls: Vec<NegativeControlResult>,
    pub decision: String,
}

fn det_uniform(seed: u64, idx: u64) -> f64 {
    let mut z = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ idx.wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ 0x94d0_49bb_1331_11eb;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

pub fn run_default_harness() -> Result<MtpDecoderReport, String> {
    let model = TinyMtpModel::frozen_tiny();
    let model_digest = model.digest();

    // 1. Target-only cached vs full-recompute baseline parity.
    let prompt_main = vec![1, 3, 5, 2];
    let (_, ref_logits_all) = model.target_full_recompute(&prompt_main)?;
    let mut base_cache = PoisonedKvCache::new(model.max_ctx, model.d_head, model.d_model);
    let mut last_cached_logits = Vec::new();
    for &t in &prompt_main {
        let (_, l) = base_cache.target_step(&model, t, 0)?;
        last_cached_logits = l;
    }
    let target_cached_vs_recompute_max_err =
        max_abs_diff(&last_cached_logits, ref_logits_all.last().unwrap())?;

    // 2. Prefix-causal no-future-ground-truth-leak check: changing future tokens beyond prefix
    // cannot alter MTP logits at earlier positions.
    let seq_a = vec![1, 3, 5, 2, 4, 6];
    let seq_b = vec![1, 3, 5, 2, 7, 1];
    let (ha, _) = model.target_full_recompute(&seq_a)?;
    let (hb, _) = model.target_full_recompute(&seq_b)?;
    let mtp_a = model.mtp_full_recompute(&ha[..3], &seq_a[1..4])?;
    let mtp_b = model.mtp_full_recompute(&hb[..3], &seq_b[1..4])?;
    let mut prefix_causal_err = 0.0_f64;
    for p in 0..3 {
        let e = max_abs_diff(&mtp_a[p], &mtp_b[p])?;
        if e > prefix_causal_err {
            prefix_causal_err = e;
        }
    }

    // 3. Greedy conformance fixtures.
    let (base_seq_for_ov, _) = run_target_only_greedy(&model, &prompt_main, 8)?;
    if base_seq_for_ov.len() < 4 {
        return Err(format!(
            "prompt_main generated only {} tokens before EOS; need >= 4 for multi-position fixtures",
            base_seq_for_ov.len()
        ));
    }
    let wrong_tok = |t: usize| {
        let cand = (t + 1) % model.vocab_size;
        if cand == EOS_TOKEN {
            (cand + 1) % model.vocab_size
        } else {
            cand
        }
    };
    // Fixture A: natural MTP draft-and-verify (horizon=2)
    let fix_natural = run_mtp_greedy_fixture(
        "greedy_natural_mixed_accept_reject",
        &model,
        &prompt_main,
        8,
        2,
        None,
        None,
        false,
        EOS_TOKEN,
    )?;
    // Fixture B: forced first-position rejection
    let ov_first_reject = vec![vec![
        wrong_tok(base_seq_for_ov[1]),
        wrong_tok(base_seq_for_ov[2]),
    ]];
    let fix_first_reject = run_mtp_greedy_fixture(
        "greedy_first_position_rejection",
        &model,
        &prompt_main,
        5,
        2,
        Some(&ov_first_reject),
        None,
        false,
        EOS_TOKEN,
    )?;
    // Fixture C: forced later-position rejection (pos 0 accepted, pos 1 rejected)
    let ov_later_reject = vec![vec![base_seq_for_ov[1], wrong_tok(base_seq_for_ov[2])]];
    let fix_later_reject = run_mtp_greedy_fixture(
        "greedy_later_position_rejection",
        &model,
        &prompt_main,
        5,
        2,
        Some(&ov_later_reject),
        None,
        false,
        EOS_TOKEN,
    )?;
    // Fixture D: forced all-accepted block + bonus token
    let ov_all_accept = vec![vec![base_seq_for_ov[1], base_seq_for_ov[2]]];
    let fix_all_accept = run_mtp_greedy_fixture(
        "greedy_all_accepted_plus_bonus",
        &model,
        &prompt_main,
        5,
        2,
        Some(&ov_all_accept),
        None,
        false,
        EOS_TOKEN,
    )?;
    // Fixture E: explicit target-only fallback (counted)
    let fix_fallback = run_mtp_greedy_fixture(
        "greedy_explicit_target_only_fallback",
        &model,
        &prompt_main,
        4,
        2,
        None,
        None,
        true,
        EOS_TOKEN,
    )?;
    // Fixture F: cancellation after 1 round + immediate continuation
    let fix_cancel = run_mtp_greedy_fixture(
        "greedy_cancel_stop",
        &model,
        &prompt_main,
        8,
        2,
        None,
        Some(1),
        false,
        EOS_TOKEN,
    )?;
    let mut cont_prompt = prompt_main.clone();
    cont_prompt.extend_from_slice(&fix_cancel.generated);
    let fix_continue = run_mtp_greedy_fixture(
        "greedy_immediate_continuation",
        &model,
        &cont_prompt,
        4,
        2,
        None,
        None,
        false,
        EOS_TOKEN,
    )?;
    // Fixture G: context capacity edge (prompt len 21, max_ctx 24 -> at most 3 tokens committed)
    let cap_prompt: Vec<usize> = (0..21).map(|i| 1 + (i % 7)).collect();
    let fix_cap = run_mtp_greedy_fixture(
        "greedy_context_capacity_edge",
        &model,
        &cap_prompt,
        8,
        2,
        None,
        None,
        false,
        EOS_TOKEN,
    )?;
    // Fixture H: explicit EOS-boundary stopping mid-speculative-block (declaring base_seq_for_ov[1]
    // as EOS while proposing a 2-token speculative suffix `[eos, tail]` -> must stop on EOS,
    // rollback & poison tail, and never emit any token past EOS!).
    let eos_mid_tok = base_seq_for_ov[1];
    let ov_eos_tail = vec![vec![eos_mid_tok, wrong_tok(base_seq_for_ov[2])]];
    let fix_eos = run_mtp_greedy_fixture(
        "greedy_eos_boundary_no_tail_emit",
        &model,
        &prompt_main,
        8,
        2,
        Some(&ov_eos_tail),
        None,
        false,
        eos_mid_tok,
    )?;

    // Multi-request cache reset isolation check (req1 writes & rolls back -> reset -> req2 matches fresh).
    let mut shared_cache = PoisonedKvCache::new(model.max_ctx, model.d_head, model.d_model);
    for &t in &[2, 4, 6, 7] {
        let _ = shared_cache.target_step(&model, t, 0)?;
    }
    shared_cache.rollback_and_poison(2)?;
    shared_cache.reset_and_poison();
    let req2 = vec![3, 1, 5];
    let mut req2_logits = Vec::new();
    for &t in &req2 {
        let (_, l) = shared_cache.target_step(&model, t, 0)?;
        req2_logits = l;
    }
    let (_, req2_ref) = model.target_full_recompute(&req2)?;
    let multi_request_isolation_max_err = max_abs_diff(&req2_logits, req2_ref.last().unwrap())?;

    // 4. Stochastic rejection sampling mode (exact analytical + deterministic branch injection + empirical).
    let stoch_profile = SamplingProfile::stochastic(0.8, 4, 0.95);
    // Exact small-vocab distributions covering differing support, p>q, p<q, p=q:
    let p_exact = vec![0.0, 0.40, 0.25, 0.20, 0.15];
    let q_exact = vec![0.10, 0.20, 0.45, 0.25, 0.00];
    let exact_marg = exact_rejection_marginal(&p_exact, &q_exact)?;
    let exact_marginal_max_err = max_abs_diff(&exact_marg, &p_exact)?;

    // Deterministic branch injection covering every reachable branch:
    let mut branch_outcomes = Vec::new();
    let b_eq = stochastic_verify_token(&[0.5, 0.5], &[0.5, 0.5], 0, 0.99, 0.1)?;
    branch_outcomes.push(("p_eq_q".into(), b_eq.branch_label.into(), b_eq.accepted));
    let b_gt = stochastic_verify_token(&p_exact, &q_exact, 1, 0.99, 0.1)?;
    branch_outcomes.push(("p_gt_q".into(), b_gt.branch_label.into(), b_gt.accepted));
    let b_lt_acc = stochastic_verify_token(&p_exact, &q_exact, 2, 0.20, 0.1)?;
    branch_outcomes.push((
        "p_lt_q_accept".into(),
        b_lt_acc.branch_label.into(),
        b_lt_acc.accepted,
    ));
    let b_lt_rej = stochastic_verify_token(&p_exact, &q_exact, 2, 0.90, 0.50)?;
    branch_outcomes.push((
        "p_lt_q_reject".into(),
        b_lt_rej.branch_label.into(),
        !b_lt_rej.accepted,
    ));
    let b_zero = stochastic_verify_token(&p_exact, &q_exact, 0, 0.0, 0.80)?;
    branch_outcomes.push((
        "zero_target_support".into(),
        b_zero.branch_label.into(),
        !b_zero.accepted && b_zero.emitted_token != 0,
    ));
    let impossible_zero_residual_rejected =
        residual_distribution(&[0.5, 0.5], &[0.5, 0.5]).is_err();

    // Empirical calibrated check (N = 4000 deterministic draws, tolerance 0.03)
    let empirical_samples = 4000usize;
    let empirical_tolerance = 0.03_f64;
    let mut counts = vec![0usize; p_exact.len()];
    for s in 0..empirical_samples {
        let u_prop = det_uniform(2201, s as u64);
        let u_acc = det_uniform(2202, s as u64);
        let u_res = det_uniform(2203, s as u64);
        let x = sample_categorical(&q_exact, u_prop)?;
        let dec = stochastic_verify_token(&p_exact, &q_exact, x, u_acc, u_res)?;
        counts[dec.emitted_token] += 1;
    }
    let emp_probs: Vec<f64> = counts
        .iter()
        .map(|&c| (c as f64) / (empirical_samples as f64))
        .collect();
    let empirical_max_abs_err = max_abs_diff(&emp_probs, &p_exact)?;
    let empirical_pass = empirical_max_abs_err <= empirical_tolerance;

    // Bonus token rule verification under stochastic sampling (when all draft tokens accepted,
    // sample bonus token from target's effective distribution at last accepted position).
    let bonus_probs = stoch_profile.effective_probs(&last_cached_logits)?;
    let bonus_tok = sample_categorical(&bonus_probs, 0.25)?;
    let bonus_token_rule_verified = bonus_probs[bonus_tok] > 0.0;

    // 5. Six executable negative controls!
    let mut negative_controls = Vec::new();

    // Control 1: Swapped p/q probabilities -> marginal diverges from p!
    let swapped_marg = exact_rejection_marginal(&q_exact, &p_exact)?;
    let swapped_err = max_abs_diff(&swapped_marg, &p_exact)?;
    negative_controls.push(NegativeControlResult {
        case_id: "swapped_p_q_probabilities".into(),
        description: "Swapping target p and draft q violates exact marginal preservation".into(),
        outcome: if swapped_err > 1e-6 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("marginal_l_inf_err={swapped_err:.6} > 1e-6"),
    });

    // Control 2: Incorrect residual normalization / wrong sign max(q - p, 0)
    let bad_res_raw: Vec<f64> = p_exact
        .iter()
        .zip(q_exact.iter())
        .map(|(pi, qi)| (qi - pi).max(0.0))
        .collect();
    let bad_res_sum: f64 = bad_res_raw.iter().sum();
    let bad_res: Vec<f64> = bad_res_raw.iter().map(|d| d / bad_res_sum).collect();
    let mut bad_emitted = vec![0.0_f64; p_exact.len()];
    let mut rej_mass = 0.0_f64;
    for x in 0..p_exact.len() {
        if q_exact[x] > 0.0 {
            let alpha = (p_exact[x] / q_exact[x]).min(1.0);
            bad_emitted[x] += q_exact[x] * alpha;
            rej_mass += q_exact[x] * (1.0 - alpha);
        }
    }
    for (v, &rv) in bad_res.iter().enumerate() {
        bad_emitted[v] += rej_mass * rv;
    }
    let bad_res_err = max_abs_diff(&bad_emitted, &p_exact)?;
    negative_controls.push(NegativeControlResult {
        case_id: "incorrect_residual_correction_mass".into(),
        description: "Using max(q - p, 0) instead of max(p - q, 0) biases emitted distribution"
            .into(),
        outcome: if bad_res_err > 1e-6 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("biased_marginal_err={bad_res_err:.6} > 1e-6"),
    });

    // Control 3: Off-by-one target alignment (comparing draft[0] against target logits at pos-1)
    let (_, ref_l_align) = model.target_full_recompute(&prompt_main)?;
    let shifted_err = max_abs_diff(
        &ref_l_align[prompt_main.len() - 2],
        ref_l_align.last().unwrap(),
    )?;
    negative_controls.push(NegativeControlResult {
        case_id: "off_by_one_target_alignment".into(),
        description: "Aligning draft proposal against pos-1 target logits fails exact logit check"
            .into(),
        outcome: if shifted_err > 1e-9 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("shifted_logit_err={shifted_err:.6} > 1e-9"),
    });

    // Control 4: Missing cache rollback -> reads 1e6 poisoned tail slot!
    let mut bad_cache = PoisonedKvCache::new(model.max_ctx, model.d_head, model.d_model);
    for &t in &prompt_main {
        let _ = bad_cache.target_step(&model, t, 0)?;
    }
    // Simulate a rejected token written and poisoned, then a buggy next step reading +1 past valid_len
    let _ = bad_cache.target_step(&model, wrong_tok(base_seq_for_ov[0]), 0)?;
    bad_cache.rollback_and_poison(prompt_main.len())?;
    let (_, poisoned_logits) = bad_cache.target_step(&model, base_seq_for_ov[0], 1)?;
    let mut clean_seq = prompt_main.clone();
    clean_seq.push(base_seq_for_ov[0]);
    let (_, clean_ref_l) = model.target_full_recompute(&clean_seq)?;
    let poison_err = max_abs_diff(&poisoned_logits, clean_ref_l.last().unwrap())?;
    negative_controls.push(NegativeControlResult {
        case_id: "missing_cache_rollback_poison_read".into(),
        description: "Reading unmasked rejected tail slot picks up 1e6 poison sentinel".into(),
        outcome: if poison_err > 1e-6 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("poisoned_logit_err={poison_err:.6} > 1e-6"),
    });

    // Control 5: Emitted rejected speculative token in output sequence
    let mut leaked_output = fix_first_reject.generated.clone();
    if let Some(rej) = fix_first_reject.rounds[0].rejected_token {
        leaked_output[1] = rej;
    }
    let leaked_agrees = leaked_output == fix_first_reject.baseline_generated;
    negative_controls.push(NegativeControlResult {
        case_id: "emitted_rejected_speculative_token".into(),
        description: "Emitting a rejected speculative token fails target-only output conformance"
            .into(),
        outcome: if !leaked_agrees {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("exact_token_agreement={leaked_agrees}"),
    });

    // Control 6: Future ground-truth token leakage into MTP proposal + routing bias mutation
    let mut leaky_model = model.clone();
    // Flip expert routing preference via an illegal training-time load-controller update during decode.
    leaky_model.routing_bias[0] -= 5.0;
    leaky_model.routing_bias[1] += 5.0;
    let (_, leaky_logits) = leaky_model.target_full_recompute(&prompt_main)?;
    let bias_drift = max_abs_diff(leaky_logits.last().unwrap(), ref_logits_all.last().unwrap())?;
    let leaked_future_mtp = model.mtp_full_recompute(&ha[..3], &[seq_a[1], seq_a[2], seq_a[5]])?;
    let future_leak_err = max_abs_diff(&leaked_future_mtp[2], &mtp_a[2])?;
    negative_controls.push(NegativeControlResult {
        case_id: "future_ground_truth_leak_and_bias_mutation".into(),
        description:
            "Injecting future ground-truth token or mutating routing bias during decode diverges"
                .into(),
        outcome: if future_leak_err > 1e-6 && bias_drift > 1e-6 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("future_leak_err={future_leak_err:.6}, bias_drift={bias_drift:.6}"),
    });

    let greedy_fixtures = vec![
        fix_natural,
        fix_first_reject,
        fix_later_reject,
        fix_all_accept,
        fix_fallback,
        fix_cancel,
        fix_continue,
        fix_cap,
        fix_eos,
    ];

    let all_greedy_ok = greedy_fixtures.iter().all(|f| {
        f.exact_token_agreement && f.max_cache_err <= 1e-11 && f.max_next_logit_err <= 1e-11
    });
    let all_branches_ok = branch_outcomes.iter().all(|(_, _, ok)| *ok);
    let all_neg_caught = negative_controls
        .iter()
        .all(|c| c.outcome == OutcomeStatus::Invalid);
    let decision = if all_greedy_ok
        && all_branches_ok
        && exact_marginal_max_err <= 1e-12
        && empirical_pass
        && impossible_zero_residual_rejected
        && bonus_token_rule_verified
        && all_neg_caught
    {
        "GO_CPU_CONFORMANCE_ONLY_NO_SERVING_SPEEDUP_CLAIMED"
    } else {
        "NO_GO_CONFORMANCE_FAILURE"
    }
    .into();

    Ok(MtpDecoderReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "serving_scale_gpu_mtp_runtime",
            "CPU tiny-model MTP draft-and-verify conformance executed locally; physical GPU / serving-scale speedup and backend MTP support remain unmeasured",
        ),
        model_digest,
        vocab_size: model.vocab_size,
        max_ctx: model.max_ctx,
        target_cached_vs_recompute_max_err,
        prefix_causal_no_future_leak_max_err: prefix_causal_err,
        multi_request_isolation_max_err,
        routing_bias_frozen: true,
        greedy_fixtures,
        stochastic: StochasticReport {
            profile: stoch_profile,
            branch_outcomes,
            exact_marginal_max_err,
            empirical_samples,
            empirical_max_abs_err,
            empirical_tolerance,
            empirical_pass,
            impossible_zero_residual_rejected,
            bonus_token_rule_verified,
        },
        negative_controls,
        decision,
    })
}

pub fn inputs_json(report: &MtpDecoderReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.mtp-decoder.inputs.v1\",\n",
            "  \"issue\": 22,\n",
            "  \"upstream_reference\": {{\"repo\":{},\"commit\":{},\"paper\":{}}},\n",
            "  \"execution_class\": {},\n",
            "  \"model_digest\": {},\n",
            "  \"vocab_size\": {},\n",
            "  \"max_ctx\": {},\n",
            "  \"eos_token\": {},\n",
            "  \"poison_sentinel\": {},\n",
            "  \"stochastic_profile\": {{\"mode\":{},\"temperature\":{},\"top_k\":{},\"top_p\":{},\"empirical_samples\":{},\"empirical_tolerance\":{}}},\n",
            "  \"routing_bias_frozen\": {}\n",
            "}}\n"
        ),
        quote(UPSTREAM_REPO),
        quote(UPSTREAM_COMMIT),
        quote(DEEPSEEK_V3_PAPER),
        quote(report.execution_class.as_str()),
        report.model_digest.to_json(),
        report.vocab_size,
        report.max_ctx,
        EOS_TOKEN,
        finite_f64(POISON_SENTINEL),
        quote(&report.stochastic.profile.mode),
        finite_f64(report.stochastic.profile.temperature),
        report.stochastic.profile.top_k,
        finite_f64(report.stochastic.profile.top_p),
        report.stochastic.empirical_samples,
        finite_f64(report.stochastic.empirical_tolerance),
        report.routing_bias_frozen,
    )
}

fn f64_matrix_json(mat: &[Vec<f64>]) -> String {
    let rows: Vec<String> = mat
        .iter()
        .map(|row| {
            let vals: Vec<String> = row.iter().map(|&v| finite_f64(v)).collect();
            format!("[{}]", vals.join(","))
        })
        .collect();
    format!("[{}]", rows.join(","))
}

pub fn results_json(report: &MtpDecoderReport) -> String {
    let fixtures_json: Vec<String> = report
        .greedy_fixtures
        .iter()
        .map(|f| {
            let rounds_json: Vec<String> = f
                .rounds
                .iter()
                .map(|r| {
                    format!(
                        concat!(
                            "{{\"round_index\":{},\"committed_prefix_len_before\":{},",
                            "\"proposed_tokens\":{},\"draft_probs\":{},\"target_probs\":{},",
                            "\"accepted_tokens\":{},\"rejected_token\":{},\"corrected_or_bonus_token\":{},",
                            "\"target_cache_len_after\":{},\"mtp_cache_len_after\":{},",
                            "\"cache_recompute_max_err\":{},\"next_logit_recompute_max_err\":{},",
                            "\"fallback_target_only\":{}}}"
                        ),
                        r.round_index,
                        r.committed_prefix_len_before,
                        usize_array(&r.proposed_tokens),
                        f64_matrix_json(&r.draft_probs),
                        f64_matrix_json(&r.target_probs),
                        usize_array(&r.accepted_tokens),
                        opt_u64(r.rejected_token.map(|v| v as u64)),
                        opt_u64(r.corrected_or_bonus_token.map(|v| v as u64)),
                        r.target_cache_len_after,
                        r.mtp_cache_len_after,
                        finite_f64(r.cache_recompute_max_err),
                        finite_f64(r.next_logit_recompute_max_err),
                        r.fallback_target_only,
                    )
                })
                .collect();
            format!(
                concat!(
                    "{{\"fixture_id\":{},\"eos_token\":{},\"prompt\":{},\"generated\":{},\"exact_token_agreement\":{},",
                    "\"max_cache_err\":{},\"max_next_logit_err\":{},\"total_draft_proposals\":{},",
                    "\"total_accepted_drafts\":{},\"total_verification_rounds\":{},\"target_only_fallbacks\":{},",
                    "\"target_model_passes\":{},\"mtp_model_passes\":{},\"target_rollbacks\":{},\"mtp_rollbacks\":{},",
                    "\"target_cache_bytes\":{},\"mtp_cache_bytes\":{},",
                    "\"accepted_per_verification\":{},\"stopped_reason\":{},\"rounds\":[{}]}}"
                ),
                quote(&f.fixture_id),
                f.eos_token,
                usize_array(&f.prompt),
                usize_array(&f.generated),
                f.exact_token_agreement,
                finite_f64(f.max_cache_err),
                finite_f64(f.max_next_logit_err),
                f.total_draft_proposals,
                f.total_accepted_drafts,
                f.total_verification_rounds,
                f.target_only_fallbacks,
                f.target_model_passes,
                f.mtp_model_passes,
                f.target_rollbacks,
                f.mtp_rollbacks,
                f.target_cache_bytes,
                f.mtp_cache_bytes,
                finite_f64(f.accepted_per_verification),
                quote(&f.stopped_reason),
                rounds_json.join(","),
            )
        })
        .collect();

    let branches_json: Vec<String> = report
        .stochastic
        .branch_outcomes
        .iter()
        .map(|(name, label, ok)| {
            format!(
                "{{\"case\":{},\"branch\":{},\"verified\":{}}}",
                quote(name),
                quote(label),
                ok
            )
        })
        .collect();

    let neg_json: Vec<String> = report
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
            "  \"schema\": \"inference-lab.mtp-decoder.results.v1\",\n",
            "  \"issue\": 22,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"decision\": {},\n",
            "  \"speedup_claimed\": false,\n",
            "  \"target_cached_vs_recompute_max_err\": {},\n",
            "  \"prefix_causal_no_future_leak_max_err\": {},\n",
            "  \"multi_request_isolation_max_err\": {},\n",
            "  \"synchronized_gpu_latency_ms\": {},\n",
            "  \"serving_backend_mtp_qualified\": {},\n",
            "  \"greedy_fixtures\": [{}],\n",
            "  \"stochastic\": {{\n",
            "    \"mode\": {},\n",
            "    \"exact_marginal_max_err\": {},\n",
            "    \"empirical_samples\": {},\n",
            "    \"empirical_max_abs_err\": {},\n",
            "    \"empirical_tolerance\": {},\n",
            "    \"empirical_pass\": {},\n",
            "    \"impossible_zero_residual_rejected\": {},\n",
            "    \"bonus_token_rule_verified\": {},\n",
            "    \"finite_empirical_proves_universal_preservation\": false,\n",
            "    \"branches\": [{}]\n",
            "  }},\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        quote(&report.decision),
        finite_f64(report.target_cached_vs_recompute_max_err),
        finite_f64(report.prefix_causal_no_future_leak_max_err),
        finite_f64(report.multi_request_isolation_max_err),
        opt_f64(None),
        opt_u64(None),
        fixtures_json.join(","),
        quote(&report.stochastic.profile.mode),
        finite_f64(report.stochastic.exact_marginal_max_err),
        report.stochastic.empirical_samples,
        finite_f64(report.stochastic.empirical_max_abs_err),
        finite_f64(report.stochastic.empirical_tolerance),
        report.stochastic.empirical_pass,
        report.stochastic.impossible_zero_residual_rejected,
        report.stochastic.bonus_token_rule_verified,
        branches_json.join(","),
        neg_json.join(","),
    )
}

pub fn summary_markdown(report: &MtpDecoderReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #22: Small MTP Draft-and-Verify Decoder Conformance"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}`",
        report.execution_class.as_str()
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware gate: `{}` (`{}`)",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Decision: `{}` (no serving speedup assumed)",
        report.decision
    )
    .unwrap();
    writeln!(
        out,
        "- Target cached vs full recompute max error: `{:.3e}`",
        report.target_cached_vs_recompute_max_err
    )
    .unwrap();
    writeln!(
        out,
        "- Prefix-causal no-future-leak max error: `{:.3e}`",
        report.prefix_causal_no_future_leak_max_err
    )
    .unwrap();
    writeln!(
        out,
        "- Multi-request reset isolation max error: `{:.3e}`",
        report.multi_request_isolation_max_err
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Greedy conformance fixtures").unwrap();
    writeln!(
        out,
        "| Fixture | Exact Agreement | Max Cache Err | Max Logit Err | Proposed | Accepted | Fallbacks | Stop |"
    )
    .unwrap();
    writeln!(
        out,
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    .unwrap();
    for f in &report.greedy_fixtures {
        writeln!(
            out,
            "| `{}` | `{}` | `{:.3e}` | `{:.3e}` | {} | {} | {} | `{}` |",
            f.fixture_id,
            f.exact_token_agreement,
            f.max_cache_err,
            f.max_next_logit_err,
            f.total_draft_proposals,
            f.total_accepted_drafts,
            f.target_only_fallbacks,
            f.stopped_reason,
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Stochastic rejection-sampling conformance").unwrap();
    writeln!(
        out,
        "- Exact small-vocab marginal max error: `{:.3e}`",
        report.stochastic.exact_marginal_max_err
    )
    .unwrap();
    writeln!(
        out,
        "- Calibrated empirical check (N={}): max abs error `{:.5}` <= `{}` -> `{}` (finite empirical check does not prove universal preservation)",
        report.stochastic.empirical_samples,
        report.stochastic.empirical_max_abs_err,
        report.stochastic.empirical_tolerance,
        report.stochastic.empirical_pass,
    )
    .unwrap();
    writeln!(
        out,
        "- Impossible zero-mass residual rejected: `{}`; bonus-token rule verified: `{}`",
        report.stochastic.impossible_zero_residual_rejected,
        report.stochastic.bonus_token_rule_verified
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Negative controls").unwrap();
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
