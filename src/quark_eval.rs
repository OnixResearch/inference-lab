//! Issue #44: Qualify Quark Base/Zero inference with cached-decode parity and measured context costs.
//!
//! Pinned source basis (published cache figures are configuration estimates, NOT measurements):
//! - `empero-ai/Quark-150M-128K-Base` and `empero-ai/Quark-150M-128K-Zero` (~165.3M params,
//!   custom Transformers 5.23+ code, bounded local attention/convolution + 4 global attention
//!   layers retaining context-growing KV state).
//! - Reuses `#7` endpoint accounting, `#11` lifecycle, `#28` diagnostics (without requiring
//!   compression), and `#36` profile conventions; supplies profiles for RSI Lab `#98`.
//!
//! Provides:
//! - Separate pinned profiles for `Quark-150M-128K-Base` and `Quark-150M-128K-Zero` with
//!   Transformers >= 5.23 version gate and explicit Base-vs-Zero separation (quality comparison,
//!   never a numerical-parity assertion).
//! - Pure-Rust miniature Quark hybrid architecture (**2 bounded local-window/conv layers** with
//!   window `W=4` + **1 global full-KV layer**) with BOTH an **independent full-prefix recomputation
//!   reference** (`forward_full_prefix`, no rolling cache!) AND a stateful cached decoder
//!   (`QuarkHybridCache`), verifying cached decode across irregular chunks, local-window rollover
//!   past `W=4`, exact positions, and changed tool results while reporting the first divergent token.
//! - Turn rendering / assistant stop delimiter / tool-call execution verifier rejecting generated
//!   text that resembles a tool result as an executed effect.
//! - Progressive context envelope (`1K`, `8K`, `32K`, `128K`), deterministic held-out diagnostic +
//!   RSI task checker & per-family reducer (exact values, near-match distractors, corrected facts,
//!   source attribution, `cost_per_verified_success = None` at 0 successes), and 8 negative controls.

use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, opt_u64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const QUARK_DIM: usize = 4;
pub const QUARK_LOCAL_LAYERS: usize = 2;
pub const QUARK_LOCAL_WINDOW: usize = 4;
pub const QUARK_GLOBAL_LAYERS_REAL: usize = 4;
pub const QUARK_MIN_TRANSFORMERS_MAJOR: u32 = 5;
pub const QUARK_MIN_TRANSFORMERS_MINOR: u32 = 23;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarkVariant {
    Base,
    Zero,
}

impl QuarkVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Base => "Quark-150M-128K-Base",
            Self::Zero => "Quark-150M-128K-Zero",
        }
    }

    const fn seed_offset(self) -> usize {
        match self {
            Self::Base => 4400,
            Self::Zero => 4450,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuarkProfile {
    pub variant: QuarkVariant,
    pub checkpoint_digest: DigestRef,
    pub custom_modeling_digest: DigestRef,
    pub transformers_version: (u32, u32),
    pub supports_native_serialize_restore: bool,
    pub supports_native_fork: bool,
    pub recovery_mode_label: String,
}

pub fn validate_quark_profile(p: &QuarkProfile) -> Result<(), String> {
    let (maj, min) = p.transformers_version;
    if maj < QUARK_MIN_TRANSFORMERS_MAJOR
        || (maj == QUARK_MIN_TRANSFORMERS_MAJOR && min < QUARK_MIN_TRANSFORMERS_MINOR)
    {
        return Err(format!(
            "unsupported Transformers {maj}.{min}: Quark custom code requires >= {}.{}",
            QUARK_MIN_TRANSFORMERS_MAJOR, QUARK_MIN_TRANSFORMERS_MINOR
        ));
    }
    if !p.supports_native_serialize_restore && p.recovery_mode_label == "native_state_support" {
        return Err("recomputation-based recovery cannot be labeled native_state_support".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QuarkFaults {
    pub bad_local_window_rollover: bool,
    pub wrong_position_after_chunk: bool,
}

/// Compare two output sequences with explicit finiteness checks, returning `(max_err, first_div)`.
pub fn compare_quark_outputs(
    a: &[[f64; QUARK_DIM]],
    b: &[[f64; QUARK_DIM]],
    tol: f64,
) -> Result<(f64, Option<usize>), String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err("invalid tolerance".into());
    }
    let mut m = 0.0_f64;
    let mut first_div: Option<usize> = None;
    for (t, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
        for (&x, &y) in ra.iter().zip(rb.iter()) {
            if !x.is_finite() || !y.is_finite() {
                return Err("non-finite Quark output".into());
            }
            let d = (x - y).abs();
            if d > m {
                m = d;
            }
            if d > tol && first_div.is_none() {
                first_div = Some(t);
            }
        }
    }
    Ok((m, first_div))
}

fn quark_embed(variant: QuarkVariant, tok: u32, pos: usize) -> [f64; QUARK_DIM] {
    let base = variant.seed_offset();
    let mut out = [0.0_f64; QUARK_DIM];
    for (d, slot) in out.iter_mut().enumerate() {
        *slot = det_weight(base + (tok as usize) * 11, d, 0.55)
            + 0.08 * ((pos as f64 + 1.0) * (d as f64 + 1.0) * 0.19).sin();
    }
    out
}

fn proj4(x: &[f64; QUARK_DIM], seed: usize) -> [f64; QUARK_DIM] {
    let mut out = [0.0_f64; QUARK_DIM];
    for (o, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (i, &xv) in x.iter().enumerate() {
            acc += xv * det_weight(seed + o * 7, i, 0.45);
        }
        *slot = acc;
    }
    out
}

/// Independent stateless full-prefix recomputation reference (2 local sliding-window+conv layers
/// with window `W=4` followed by 1 global full-KV attention layer). Never uses `QuarkHybridCache`.
pub fn forward_full_prefix(
    variant: QuarkVariant,
    tokens: &[u32],
) -> Result<Vec<[f64; QUARK_DIM]>, String> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let base = variant.seed_offset();
    let n = tokens.len();
    let scale = 1.0 / (QUARK_DIM as f64).sqrt();
    let mut h: Vec<[f64; QUARK_DIM]> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &tok)| quark_embed(variant, tok, pos))
        .collect();

    // 2 bounded local conv + sliding-window attention layers
    for layer in 0..QUARK_LOCAL_LAYERS {
        let lseed = base + 10 + layer * 100;
        let mut lq = Vec::with_capacity(n);
        let mut lk = Vec::with_capacity(n);
        let mut lv = Vec::with_capacity(n);
        for t in 0..n {
            let prev = if t == 0 { [0.0; QUARK_DIM] } else { h[t - 1] };
            let mut conv_x = [0.0_f64; QUARK_DIM];
            for d in 0..QUARK_DIM {
                conv_x[d] = 0.72 * h[t][d] + 0.28 * prev[d];
            }
            lq.push(proj4(&conv_x, lseed));
            lk.push(proj4(&conv_x, lseed + 10));
            lv.push(proj4(&conv_x, lseed + 20));
        }
        let mut next_h = Vec::with_capacity(n);
        for t in 0..n {
            let start = (t + 1).saturating_sub(QUARK_LOCAL_WINDOW);
            let mut scores = Vec::with_capacity(t + 1 - start);
            for k in &lk[start..=t] {
                let dot: f64 = lq[t].iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }
            let probs = softmax_stable(&scores)?;
            let mut out = h[t];
            for (p, v) in probs.iter().zip(lv[start..=t].iter()) {
                for (dst, &vv) in out.iter_mut().zip(v.iter()) {
                    *dst += p * vv;
                }
            }
            next_h.push(out);
        }
        h = next_h;
    }

    // 1 global full-context attention layer
    let gseed = base + 300;
    let gq: Vec<[f64; QUARK_DIM]> = h.iter().map(|x| proj4(x, gseed)).collect();
    let gk: Vec<[f64; QUARK_DIM]> = h.iter().map(|x| proj4(x, gseed + 10)).collect();
    let gv: Vec<[f64; QUARK_DIM]> = h.iter().map(|x| proj4(x, gseed + 20)).collect();
    let mut final_h = Vec::with_capacity(n);
    for t in 0..n {
        let mut scores = Vec::with_capacity(t + 1);
        for k in &gk[0..=t] {
            let dot: f64 = gq[t].iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            scores.push(dot * scale);
        }
        let probs = softmax_stable(&scores)?;
        let mut out = h[t];
        for (p, v) in probs.iter().zip(gv[0..=t].iter()) {
            for (dst, &vv) in out.iter_mut().zip(v.iter()) {
                *dst += p * vv;
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err("non-finite full-prefix Quark output".into());
        }
        final_h.push(out);
    }
    Ok(final_h)
}

#[derive(Debug, Clone)]
pub struct QuarkHybridCache {
    pub variant: QuarkVariant,
    pub revision_gen: u64,
    pub session_id: String,
    pub pos: usize,
    pub local_k: [Vec<[f64; QUARK_DIM]>; QUARK_LOCAL_LAYERS],
    pub local_v: [Vec<[f64; QUARK_DIM]>; QUARK_LOCAL_LAYERS],
    pub conv_prev: [[f64; QUARK_DIM]; QUARK_LOCAL_LAYERS],
    pub global_k: Vec<[f64; QUARK_DIM]>,
    pub global_v: Vec<[f64; QUARK_DIM]>,
}

impl QuarkHybridCache {
    pub fn fresh(variant: QuarkVariant, revision_gen: u64, session_id: &str) -> Self {
        Self {
            variant,
            revision_gen,
            session_id: session_id.into(),
            pos: 0,
            local_k: [Vec::new(), Vec::new()],
            local_v: [Vec::new(), Vec::new()],
            conv_prev: [[0.0; QUARK_DIM]; QUARK_LOCAL_LAYERS],
            global_k: Vec::new(),
            global_v: Vec::new(),
        }
    }

    pub fn step(&mut self, tok: u32, faults: QuarkFaults) -> Result<[f64; QUARK_DIM], String> {
        let base = self.variant.seed_offset();
        let eff_pos = if faults.wrong_position_after_chunk && self.pos >= 4 {
            self.pos + 1
        } else {
            self.pos
        };
        let scale = 1.0 / (QUARK_DIM as f64).sqrt();
        let mut h = quark_embed(self.variant, tok, eff_pos);

        for layer in 0..QUARK_LOCAL_LAYERS {
            let lseed = base + 10 + layer * 100;
            let mut conv_x = [0.0_f64; QUARK_DIM];
            for d in 0..QUARK_DIM {
                conv_x[d] = 0.72 * h[d] + 0.28 * self.conv_prev[layer][d];
            }
            self.conv_prev[layer] = h;

            let lq = proj4(&conv_x, lseed);
            let lk = proj4(&conv_x, lseed + 10);
            let lv = proj4(&conv_x, lseed + 20);
            self.local_k[layer].push(lk);
            self.local_v[layer].push(lv);
            if self.local_k[layer].len() > QUARK_LOCAL_WINDOW {
                if faults.bad_local_window_rollover {
                    self.local_k[layer].remove(1);
                    self.local_v[layer].remove(1);
                } else {
                    self.local_k[layer].remove(0);
                    self.local_v[layer].remove(0);
                }
            }
            let mut lscores = Vec::with_capacity(self.local_k[layer].len());
            for k in &self.local_k[layer] {
                let dot: f64 = lq.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                lscores.push(dot * scale);
            }
            let lprobs = softmax_stable(&lscores)?;
            for (p, v) in lprobs.iter().zip(self.local_v[layer].iter()) {
                for (dst, &vv) in h.iter_mut().zip(v.iter()) {
                    *dst += p * vv;
                }
            }
        }

        // Global full-context KV layer
        let gseed = base + 300;
        let gq = proj4(&h, gseed);
        let gk = proj4(&h, gseed + 10);
        let gv = proj4(&h, gseed + 20);
        self.global_k.push(gk);
        self.global_v.push(gv);

        let mut gscores = Vec::with_capacity(self.global_k.len());
        for k in &self.global_k {
            let dot: f64 = gq.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            gscores.push(dot * scale);
        }
        let gprobs = softmax_stable(&gscores)?;
        for (p, v) in gprobs.iter().zip(self.global_v.iter()) {
            for (dst, &vv) in h.iter_mut().zip(v.iter()) {
                *dst += p * vv;
            }
        }
        if h.iter().any(|v| !v.is_finite()) {
            return Err("non-finite Quark layer output".into());
        }
        self.pos += 1;
        Ok(h)
    }

    pub fn run_tokens(
        &mut self,
        tokens: &[u32],
        faults: QuarkFaults,
    ) -> Result<Vec<[f64; QUARK_DIM]>, String> {
        let mut outs = Vec::with_capacity(tokens.len());
        for &t in tokens {
            outs.push(self.step(t, faults)?);
        }
        Ok(outs)
    }
}

#[derive(Debug, Clone)]
pub struct ToolTurnRecord {
    pub emitted_tool_call: bool,
    pub tool_actually_executed: bool,
    pub generated_text_resembles_tool_result: bool,
    pub assistant_stop_delimiter_present: bool,
}

pub fn validate_tool_turn(turn: &ToolTurnRecord) -> OutcomeStatus {
    if !turn.assistant_stop_delimiter_present {
        return OutcomeStatus::Invalid;
    }
    if turn.emitted_tool_call
        && !turn.tool_actually_executed
        && turn.generated_text_resembles_tool_result
    {
        return OutcomeStatus::Invalid;
    }
    OutcomeStatus::Accepted
}

#[derive(Debug, Clone)]
pub struct QuarkHeldoutTask {
    pub task_id: String,
    pub family: String, // exact_value | near_match_distractor | corrected_fact | source_attribution | rsi_tool_actor
    pub expected_value: String,
    pub expected_source: String,
    pub observed_value: String,
    pub observed_source: String,
    pub tool_turn: ToolTurnRecord,
    pub cost_ms: f64,
}

#[derive(Debug, Clone)]
pub struct QuarkFamilyCount {
    pub family: String,
    pub planned: usize,
    pub verified: usize,
}

pub fn evaluate_heldout_tasks(
    tasks: &[QuarkHeldoutTask],
) -> Result<(usize, usize, Option<f64>, Vec<QuarkFamilyCount>), String> {
    let mut planned = 0usize;
    let mut verified = 0usize;
    let mut total_ms = 0.0_f64;
    let mut by_fam: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for t in tasks {
        if !t.cost_ms.is_finite() || t.cost_ms < 0.0 {
            return Err("non-finite or negative task cost_ms".into());
        }
        planned += 1;
        total_ms += t.cost_ms;
        let entry = by_fam.entry(t.family.clone()).or_insert((0, 0));
        entry.0 += 1;
        let turn_ok = validate_tool_turn(&t.tool_turn) == OutcomeStatus::Accepted;
        let value_ok =
            t.observed_value == t.expected_value && t.observed_source == t.expected_source;
        if turn_ok && value_ok {
            verified += 1;
            entry.1 += 1;
        }
    }
    let cost_per = if verified == 0 {
        None
    } else {
        Some(total_ms / (verified as f64))
    };
    let breakdown = by_fam
        .into_iter()
        .map(|(family, (p, v))| QuarkFamilyCount {
            family,
            planned: p,
            verified: v,
        })
        .collect();
    Ok((planned, verified, cost_per, breakdown))
}

#[derive(Debug, Clone)]
pub struct ContextEnvelopeRow {
    pub context_tokens: usize,
    pub local_state_bytes_config_estimate: usize,
    pub global_4layer_kv_bytes_config_estimate: usize,
    pub outcome_note: String,
}

#[derive(Debug, Clone)]
pub struct QuarkVariantSummary {
    pub variant: QuarkVariant,
    pub full_vs_cached_rollover_max_err: f64,
    pub first_divergent_token_clean: Option<u64>,
    pub changed_tool_result_max_err: f64,
    pub heldout_planned: usize,
    pub heldout_verified_success: usize,
    pub cost_per_verified_success: Option<f64>,
    pub family_breakdown: Vec<QuarkFamilyCount>,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct QuarkEvalReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub variants: Vec<QuarkVariantSummary>,
    pub context_envelope: Vec<ContextEnvelopeRow>,
    pub base_vs_zero_is_numerical_parity_assertion: bool,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<QuarkEvalReport, String> {
    let base_prof = QuarkProfile {
        variant: QuarkVariant::Base,
        checkpoint_digest: DigestRef::sha256(b"empero-ai/Quark-150M-128K-Base@pinned"),
        custom_modeling_digest: DigestRef::sha256(b"quark-modeling-transformers-5.23"),
        transformers_version: (5, 23),
        supports_native_serialize_restore: true,
        supports_native_fork: false,
        recovery_mode_label: "recomputation_recovery".into(),
    };
    let zero_prof = QuarkProfile {
        variant: QuarkVariant::Zero,
        checkpoint_digest: DigestRef::sha256(b"empero-ai/Quark-150M-128K-Zero@pinned"),
        custom_modeling_digest: DigestRef::sha256(b"quark-modeling-transformers-5.23"),
        transformers_version: (5, 23),
        supports_native_serialize_restore: true,
        supports_native_fork: false,
        recovery_mode_label: "recomputation_recovery".into(),
    };
    validate_quark_profile(&base_prof)?;
    validate_quark_profile(&zero_prof)?;

    let ok_turn = ToolTurnRecord {
        emitted_tool_call: false,
        tool_actually_executed: false,
        generated_text_resembles_tool_result: false,
        assistant_stop_delimiter_present: true,
    };
    let ok_tool_turn = ToolTurnRecord {
        emitted_tool_call: true,
        tool_actually_executed: true,
        generated_text_resembles_tool_result: false,
        assistant_stop_delimiter_present: true,
    };

    // Synthetic held-out tasks (Base fails all 5 -> 0 verified -> cost_per_verified_success = None;
    // Zero passes 4 of 5 across exact_value, near_match_distractor, corrected_fact, source_attribution, rsi_tool_actor).
    let base_tasks = vec![
        QuarkHeldoutTask {
            task_id: "q-exact".into(),
            family: "exact_value".into(),
            expected_value: "K91".into(),
            expected_source: "doc-3".into(),
            observed_value: "K90".into(),
            observed_source: "doc-3".into(),
            tool_turn: ok_turn.clone(),
            cost_ms: 24.0,
        },
        QuarkHeldoutTask {
            task_id: "q-distractor".into(),
            family: "near_match_distractor".into(),
            expected_value: "v2".into(),
            expected_source: "doc-4".into(),
            observed_value: "v1".into(),
            observed_source: "doc-4".into(),
            tool_turn: ok_turn.clone(),
            cost_ms: 24.0,
        },
        QuarkHeldoutTask {
            task_id: "q-corrected".into(),
            family: "corrected_fact".into(),
            expected_value: "4096".into(),
            expected_source: "amend-2".into(),
            observed_value: "2048".into(),
            observed_source: "amend-1".into(),
            tool_turn: ok_turn.clone(),
            cost_ms: 24.0,
        },
        QuarkHeldoutTask {
            task_id: "q-source".into(),
            family: "source_attribution".into(),
            expected_value: "ok".into(),
            expected_source: "sec-7".into(),
            observed_value: "ok".into(),
            observed_source: "sec-2".into(),
            tool_turn: ok_turn.clone(),
            cost_ms: 24.0,
        },
        QuarkHeldoutTask {
            task_id: "q-rsi".into(),
            family: "rsi_tool_actor".into(),
            expected_value: "patch-ok".into(),
            expected_source: "exec-1".into(),
            observed_value: "patch-ok".into(),
            observed_source: "exec-1".into(),
            tool_turn: ToolTurnRecord {
                emitted_tool_call: true,
                tool_actually_executed: false,
                generated_text_resembles_tool_result: true,
                assistant_stop_delimiter_present: true,
            },
            cost_ms: 24.0,
        },
    ];
    let zero_tasks = vec![
        QuarkHeldoutTask {
            task_id: "q-exact".into(),
            family: "exact_value".into(),
            expected_value: "K91".into(),
            expected_source: "doc-3".into(),
            observed_value: "K91".into(),
            observed_source: "doc-3".into(),
            tool_turn: ok_turn.clone(),
            cost_ms: 25.0,
        },
        QuarkHeldoutTask {
            task_id: "q-distractor".into(),
            family: "near_match_distractor".into(),
            expected_value: "v2".into(),
            expected_source: "doc-4".into(),
            observed_value: "v2".into(),
            observed_source: "doc-4".into(),
            tool_turn: ok_turn.clone(),
            cost_ms: 25.0,
        },
        QuarkHeldoutTask {
            task_id: "q-corrected".into(),
            family: "corrected_fact".into(),
            expected_value: "4096".into(),
            expected_source: "amend-2".into(),
            observed_value: "4096".into(),
            observed_source: "amend-2".into(),
            tool_turn: ok_turn.clone(),
            cost_ms: 25.0,
        },
        QuarkHeldoutTask {
            task_id: "q-source".into(),
            family: "source_attribution".into(),
            expected_value: "ok".into(),
            expected_source: "sec-7".into(),
            observed_value: "ok".into(),
            observed_source: "sec-2".into(),
            tool_turn: ok_turn,
            cost_ms: 26.0,
        },
        QuarkHeldoutTask {
            task_id: "q-rsi".into(),
            family: "rsi_tool_actor".into(),
            expected_value: "patch-ok".into(),
            expected_source: "exec-1".into(),
            observed_value: "patch-ok".into(),
            observed_source: "exec-1".into(),
            tool_turn: ok_tool_turn,
            cost_ms: 25.0,
        },
    ];

    let seq = [2u32, 5, 1, 4, 7, 0, 3, 6, 2, 1];
    let mut variants = Vec::new();
    for (variant, tasks) in [
        (QuarkVariant::Base, &base_tasks),
        (QuarkVariant::Zero, &zero_tasks),
    ] {
        // True independent stateless full-prefix recomputation vs cached irregular chunks!
        let out_full = forward_full_prefix(variant, &seq)?;

        let mut cached = QuarkHybridCache::fresh(variant, 1, "s1");
        let mut out_cached = Vec::new();
        for ch in [&seq[0..3], &seq[3..5], &seq[5..8], &seq[8..10]] {
            out_cached.extend(cached.run_tokens(ch, QuarkFaults::default())?);
        }
        let (rollover_err, first_div) = compare_quark_outputs(&out_full, &out_cached, 1.0e-12)?;

        // Changed tool result after prefix: restore snap at pos 5 and compare vs independent full_prefix!
        let mut prefix_cache = QuarkHybridCache::fresh(variant, 1, "s1");
        let _ = prefix_cache.run_tokens(&seq[..5], QuarkFaults::default())?;
        let snap = prefix_cache.clone();
        let _ = prefix_cache.run_tokens(&[0, 1, 2], QuarkFaults::default())?;
        let mut changed = snap;
        let tail_changed = changed.run_tokens(&[6, 7, 4], QuarkFaults::default())?;

        let mut full_seq_changed = seq[..5].to_vec();
        full_seq_changed.extend_from_slice(&[6, 7, 4]);
        let fresh_out_changed = forward_full_prefix(variant, &full_seq_changed)?;
        let (changed_err, _) =
            compare_quark_outputs(&fresh_out_changed[5..], &tail_changed, 1.0e-12)?;

        let (planned, verified, cost_per, family_breakdown) = evaluate_heldout_tasks(tasks)?;
        variants.push(QuarkVariantSummary {
            variant,
            full_vs_cached_rollover_max_err: rollover_err,
            first_divergent_token_clean: first_div.map(|x| x as u64),
            changed_tool_result_max_err: changed_err,
            heldout_planned: planned,
            heldout_verified_success: verified,
            cost_per_verified_success: cost_per,
            family_breakdown,
        });
    }

    let context_envelope: Vec<ContextEnvelopeRow> = [1024usize, 8192, 32768, 131072]
        .iter()
        .map(|&ctx| ContextEnvelopeRow {
            context_tokens: ctx,
            local_state_bytes_config_estimate: 128 * 1024,
            global_4layer_kv_bytes_config_estimate: ctx * QUARK_GLOBAL_LAYERS_REAL * 2 * 512 * 2,
            outcome_note: "configuration estimate only; measured allocation requires live CUDA run"
                .into(),
        })
        .collect();

    let mut negative_controls = Vec::new();

    // NC1: unsupported Transformers version (< 5.23)
    {
        let mut bad = base_prof.clone();
        bad.transformers_version = (4, 49);
        negative_controls.push(NegativeControl {
            case_id: "nc1_transformers_version_below_5_23".into(),
            outcome: if validate_quark_profile(&bad).is_err() {
                OutcomeStatus::Unsupported
            } else {
                OutcomeStatus::Accepted
            },
            detail: "Transformers < 5.23 rejected as unsupported for Quark custom code".into(),
        });
    }

    // NC2: local-window rollover bug detected against independent full-prefix reference
    {
        let out_full = forward_full_prefix(QuarkVariant::Zero, &seq)?;
        let mut bad = QuarkHybridCache::fresh(QuarkVariant::Zero, 1, "s1");
        let out_bad = bad.run_tokens(
            &seq,
            QuarkFaults {
                bad_local_window_rollover: true,
                ..QuarkFaults::default()
            },
        )?;
        let (e, div_tok) = compare_quark_outputs(&out_full, &out_bad, 1.0e-6)?;
        negative_controls.push(NegativeControl {
            case_id: "nc2_local_window_rollover_corruption".into(),
            outcome: if e > 1.0e-6 && div_tok == Some(QUARK_LOCAL_WINDOW) {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "wrong local-window eviction diverged at token {:?} (max_err={:.3e})",
                div_tok, e
            ),
        });
    }

    // NC3: wrong position ID after irregular chunk boundary
    {
        let out_full = forward_full_prefix(QuarkVariant::Zero, &seq)?;
        let mut bad = QuarkHybridCache::fresh(QuarkVariant::Zero, 1, "s1");
        let out_bad = bad.run_tokens(
            &seq,
            QuarkFaults {
                wrong_position_after_chunk: true,
                ..QuarkFaults::default()
            },
        )?;
        let (e, div_tok) = compare_quark_outputs(&out_full, &out_bad, 1.0e-6)?;
        negative_controls.push(NegativeControl {
            case_id: "nc3_wrong_position_after_chunk".into(),
            outcome: if e > 1.0e-6 && div_tok == Some(4) {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "shifted post-chunk position ID diverged at token {:?} (max_err={:.3e})",
                div_tok, e
            ),
        });
    }

    // NC4: hallucinated text resembling tool result rejected as unexecuted effect
    {
        let st = validate_tool_turn(&ToolTurnRecord {
            emitted_tool_call: true,
            tool_actually_executed: false,
            generated_text_resembles_tool_result: true,
            assistant_stop_delimiter_present: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc4_hallucinated_tool_result_rejected".into(),
            outcome: st,
            detail: "generated text resembling a tool result is never an executed effect".into(),
        });
    }

    // NC5: missing assistant stop delimiter rejected
    {
        let st = validate_tool_turn(&ToolTurnRecord {
            emitted_tool_call: false,
            tool_actually_executed: false,
            generated_text_resembles_tool_result: false,
            assistant_stop_delimiter_present: false,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc5_missing_assistant_stop_delimiter".into(),
            outcome: st,
            detail: "unterminated assistant turn rejected by rendering checker".into(),
        });
    }

    // NC6: recomputation recovery mislabeled as native_state_support
    {
        let mut bad = base_prof.clone();
        bad.supports_native_serialize_restore = false;
        bad.recovery_mode_label = "native_state_support".into();
        negative_controls.push(NegativeControl {
            case_id: "nc6_recompute_recovery_mislabeled_native".into(),
            outcome: if validate_quark_profile(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "recomputation recovery must stay labeled separately from native state support"
                .into(),
        });
    }

    // NC7: Base vs Zero are distinct checkpoints (quality comparison, never numerical parity!)
    {
        let ob = forward_full_prefix(QuarkVariant::Base, &seq)?;
        let oz = forward_full_prefix(QuarkVariant::Zero, &seq)?;
        let (e, div_tok) = compare_quark_outputs(&ob, &oz, 1.0e-3)?;
        negative_controls.push(NegativeControl {
            case_id: "nc7_base_vs_zero_not_numerical_parity".into(),
            outcome: if e > 1.0e-3 && div_tok.is_some() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "Base vs Zero diverge by {:.3e}; comparison is quality, not parity",
                e
            ),
        });
    }

    // NC8: zero verified successes leaves cost_per_verified_success undefined (None)
    {
        let (_p, verified, cost_per, _) = evaluate_heldout_tasks(&base_tasks)?;
        negative_controls.push(NegativeControl {
            case_id: "nc8_zero_success_undefined_cost".into(),
            outcome: if verified == 0 && cost_per.is_none() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "zero verified successes keeps cost_per_verified_success explicitly None/null"
                .into(),
        });
    }

    Ok(QuarkEvalReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "empero-ai/Quark-150M-128K-Base & Zero HF checkpoints + Transformers 5.23+/CUDA",
            "Independent full-prefix vs cached 2-local+1-global parity and tool/context contracts verified on CPU; official Quark checkpoints and Transformers 5.23+ CUDA serve are absent locally",
        ),
        synthetic_contract_fixture: true,
        variants,
        context_envelope,
        base_vs_zero_is_numerical_parity_assertion: false,
        negative_controls,
    })
}

pub fn inputs_json(report: &QuarkEvalReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.quark-eval.inputs.v1\",\n",
            "  \"issue\": 44,\n",
            "  \"execution_class\": {},\n",
            "  \"min_transformers_version\": \"{}.{}\",\n",
            "  \"local_layers_fixture\": {},\n",
            "  \"local_window_fixture\": {},\n",
            "  \"global_attention_layers_real\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        QUARK_MIN_TRANSFORMERS_MAJOR,
        QUARK_MIN_TRANSFORMERS_MINOR,
        QUARK_LOCAL_LAYERS,
        QUARK_LOCAL_WINDOW,
        QUARK_GLOBAL_LAYERS_REAL,
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &QuarkEvalReport) -> String {
    let vars: Vec<String> = report
        .variants
        .iter()
        .map(|v| {
            let fams: Vec<String> = v
                .family_breakdown
                .iter()
                .map(|f| {
                    format!(
                        "{{\"family\":{},\"planned\":{},\"verified\":{}}}",
                        quote(&f.family),
                        f.planned,
                        f.verified
                    )
                })
                .collect();
            format!(
                "{{\"variant\":{},\"rollover_parity_max_err\":{},\"first_divergent_token\":{},\"changed_tool_max_err\":{},\"heldout_planned\":{},\"heldout_verified_success\":{},\"cost_per_verified_success\":{},\"family_breakdown\":[{}]}}",
                quote(v.variant.as_str()),
                finite_f64(v.full_vs_cached_rollover_max_err),
                opt_u64(v.first_divergent_token_clean),
                finite_f64(v.changed_tool_result_max_err),
                v.heldout_planned,
                v.heldout_verified_success,
                opt_f64(v.cost_per_verified_success),
                fams.join(","),
            )
        })
        .collect();
    let envs: Vec<String> = report
        .context_envelope
        .iter()
        .map(|e| {
            format!(
                "{{\"context_tokens\":{},\"local_state_bytes_config_estimate\":{},\"global_4layer_kv_bytes_config_estimate\":{},\"note\":{}}}",
                e.context_tokens,
                e.local_state_bytes_config_estimate,
                e.global_4layer_kv_bytes_config_estimate,
                quote(&e.outcome_note),
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
            "  \"schema\": \"inference-lab.quark-eval.results.v1\",\n",
            "  \"issue\": 44,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"base_vs_zero_is_numerical_parity_assertion\": {},\n",
            "  \"variants\": [{}],\n",
            "  \"context_envelope\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.base_vs_zero_is_numerical_parity_assertion,
        vars.join(","),
        envs.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &QuarkEvalReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #44: Quark Base/Zero Cached-Decode Parity & Context Costs"
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
        "- Artifact/hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Per-Checkpoint Parity & Held-Out Summary").unwrap();
    for v in &report.variants {
        writeln!(
            out,
            "- `{}`: `rollover_err={:.3e}`, `changed_tool_err={:.3e}`, `verified={}/{}`, `cost_per_success={:?}`",
            v.variant.as_str(),
            v.full_vs_cached_rollover_max_err,
            v.changed_tool_result_max_err,
            v.heldout_verified_success,
            v.heldout_planned,
            v.cost_per_verified_success
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
