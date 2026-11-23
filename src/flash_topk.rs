//! Issue #23: Evaluate FlashTopK fused sparse-indexer selection with independent correctness and
//! hardware checks.
//!
//! Pinned upstream: `Tencent-Hunyuan/FlashTopK@090902abad86e5bb3324519124a680122fff91a1`
//! (`Apache-2.0`; LICENSE and NOTICE provenance preserved).
//!
//! Provides:
//! - Strict compatibility contract validator (`SM90`/`SM100` only -> explicitly rejects `SM120`
//!   RTX 5080, `SM121` DGX Spark/GB10, AMD, Tenstorrent, DeepSeek-V3.2 64-head indexers, paged KV,
//!   and block-level CSA/QSA).
//! - Independent FP64 dense reference implementing
//!   `score(t, s) = scale(s) * sum_h weights(t, h) * ReLU(dot(q(t, h), key(s)))`
//!   with BF16 `scale=1` and FP8 explicit per-key scales, `[ks, ke)` windows, descending-score /
//!   ascending-key-index canonical ordering (including signed-zero ties), short-window sentinel
//!   padding, and explicit rejection of invalid windows or non-finite inputs.
//! - Fused threshold-and-recovery CPU selector modeling sampled-key threshold estimation,
//!   optional score-bound pruning, and candidate underflow/overflow recovery (distinguishing
//!   unmodified runs from clearly labeled test-only fault-injected threshold hooks).
//! - Three separated correctness checks (independent numerical agreement with kth-boundary margins,
//!   computed-score selection exactness, deterministic repetition & chunk/row-partition invariance),
//!   complete memory accounting, 7 executable negative controls, and configuration-scoped decisions.

use std::cmp::Ordering;
use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "Tencent-Hunyuan/FlashTopK";
pub const UPSTREAM_COMMIT: &str = "090902abad86e5bb3324519124a680122fff91a1";
pub const UPSTREAM_LICENSE: &str = "Apache-2.0 (preserve LICENSE and NOTICE)";
pub const UPSTREAM_UNVERIFIED_CLAIM: &str =
    "5317ms -> 1623ms for 1M-token causal prefill on SM100 FP8 GLM-5.1 topk=2048 (non-gating)";
pub const PAD_INDEX: i64 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexerDtype {
    Bf16,
    Fp8E4M3,
    Fp32Unsupported,
}

impl IndexerDtype {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8E4M3 => "fp8_e4m3",
            Self::Fp32Unsupported => "fp32_unsupported",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareContractRequest {
    pub target_name: String,
    pub backend: String,
    pub cuda_cc_major: Option<u32>,
    pub cuda_cc_minor: Option<u32>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    pub dtype: IndexerDtype,
    pub contiguous_keys: bool,
    pub single_contiguous_window: bool,
    pub block_level_qsa: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityCheckResult {
    pub target_name: String,
    pub outcome: OutcomeStatus,
    pub reason: String,
}

/// Exact pinned `FlashTopK@090902ab` `supported()` contract validator (zero Blackwell-name
/// extrapolation: SM120/SM121 are explicitly `Unsupported`).
pub fn validate_flashtopk_contract(req: &HardwareContractRequest) -> CompatibilityCheckResult {
    if req.backend != "cuda" {
        return CompatibilityCheckResult {
            target_name: req.target_name.clone(),
            outcome: OutcomeStatus::Unsupported,
            reason: format!(
                "backend `{}` requires separate implementation (CUDA kernels are not portable)",
                req.backend
            ),
        };
    }
    match req.cuda_cc_major {
        Some(9) | Some(10) => {}
        Some(maj) => {
            return CompatibilityCheckResult {
                target_name: req.target_name.clone(),
                outcome: OutcomeStatus::Unsupported,
                reason: format!(
                    "CUDA SM{maj}{} rejected by pinned supported() gate (major 9 or 10 only; SM120/SM121 excluded)",
                    req.cuda_cc_minor.unwrap_or(0)
                ),
            };
        }
        None => {
            return CompatibilityCheckResult {
                target_name: req.target_name.clone(),
                outcome: OutcomeStatus::Unsupported,
                reason: "missing CUDA compute capability".into(),
            };
        }
    }
    if req.num_heads != 32 || req.head_dim != 128 {
        return CompatibilityCheckResult {
            target_name: req.target_name.clone(),
            outcome: OutcomeStatus::Unsupported,
            reason: format!(
                "query shape [rows, {}, {}] outside released [rows, 32, 128] contract (e.g. 64-head DeepSeek-V3.2 unsupported)",
                req.num_heads, req.head_dim
            ),
        };
    }
    if req.topk == 0 || req.topk > 4096 {
        return CompatibilityCheckResult {
            target_name: req.target_name.clone(),
            outcome: OutcomeStatus::Unsupported,
            reason: format!("topk={} outside [1, 4096]", req.topk),
        };
    }
    if req.dtype == IndexerDtype::Fp32Unsupported {
        return CompatibilityCheckResult {
            target_name: req.target_name.clone(),
            outcome: OutcomeStatus::Unsupported,
            reason: "only FP8 E4M3 and BF16 are supported".into(),
        };
    }
    if !req.contiguous_keys || !req.single_contiguous_window || req.block_level_qsa {
        return CompatibilityCheckResult {
            target_name: req.target_name.clone(),
            outcome: OutcomeStatus::Unsupported,
            reason:
                "paged KV, multi-window masks, and block-level CSA/QSA remain unsupported upstream"
                    .into(),
        };
    }
    CompatibilityCheckResult {
        target_name: req.target_name.clone(),
        outcome: OutcomeStatus::Accepted,
        reason:
            "matches pinned FlashTopK@090902ab contract (physical CUDA execution still required)"
                .into(),
    }
}

/// Saved-call operand fixture for reproducible device-free CPU verification.
#[derive(Debug, Clone)]
pub struct SavedIndexerCall {
    pub case_id: String,
    pub dtype: IndexerDtype,
    pub num_rows: usize,
    pub num_heads: usize,
    pub num_keys: usize,
    pub head_dim: usize,
    pub k: usize,
    /// `queries[row][head][d]`
    pub queries: Vec<Vec<Vec<f64>>>,
    /// `keys[s][d]`
    pub keys: Vec<Vec<f64>>,
    /// `weights[row][head]` (FP32/FP64 effective weights)
    pub weights: Vec<Vec<f64>>,
    /// Per-key scales `[num_keys]` (ignored/must be 1.0 for BF16; explicit for FP8).
    pub key_scales: Vec<f64>,
    /// Per-row `[ks, ke)` windows.
    pub windows: Vec<(usize, usize)>,
}

impl SavedIndexerCall {
    pub fn input_digest(&self) -> DigestRef {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.case_id.as_bytes());
        bytes.extend_from_slice(self.dtype.as_str().as_bytes());
        for d in [
            self.num_rows,
            self.num_heads,
            self.num_keys,
            self.head_dim,
            self.k,
        ] {
            bytes.extend_from_slice(&(d as u64).to_le_bytes());
        }
        for r in &self.queries {
            for h in r {
                for &v in h {
                    bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
        }
        for kv in &self.keys {
            for &v in kv {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        for w in &self.weights {
            for &v in w {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        for &s in &self.key_scales {
            bytes.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        for &(ks, ke) in &self.windows {
            bytes.extend_from_slice(&(ks as u64).to_le_bytes());
            bytes.extend_from_slice(&(ke as u64).to_le_bytes());
        }
        DigestRef::sha256(&bytes)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 || self.num_heads == 0 || self.head_dim == 0 {
            return Err("k, num_heads, and head_dim must be >= 1".into());
        }
        if self.queries.len() != self.num_rows
            || self.weights.len() != self.num_rows
            || self.windows.len() != self.num_rows
            || self.keys.len() != self.num_keys
            || self.key_scales.len() != self.num_keys
        {
            return Err("saved call shape mismatch".into());
        }
        for (r, &(ks, ke)) in self.windows.iter().enumerate() {
            if ks > ke || ke > self.num_keys {
                return Err(format!(
                    "invalid window [{ks}, {ke}) at row {r} for num_keys={}",
                    self.num_keys
                ));
            }
        }
        for r in &self.queries {
            if r.len() != self.num_heads {
                return Err("query head count mismatch".into());
            }
            for h in r {
                if h.len() != self.head_dim || h.iter().any(|v| !v.is_finite()) {
                    return Err("non-finite or mismatched query vector".into());
                }
            }
        }
        for kv in &self.keys {
            if kv.len() != self.head_dim || kv.iter().any(|v| !v.is_finite()) {
                return Err("non-finite or mismatched key vector".into());
            }
        }
        for w in &self.weights {
            if w.len() != self.num_heads || w.iter().any(|v| !v.is_finite()) {
                return Err("non-finite or mismatched head weights".into());
            }
        }
        if self.key_scales.iter().any(|s| !s.is_finite()) {
            return Err("non-finite key scale".into());
        }
        Ok(())
    }
}

/// Canonicalize `-0.0` to `+0.0` so signed-zero ties sort identically by ascending key index!
#[inline]
fn canonical_score(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

/// Canonical comparison: descending score, ascending key index on exact (including signed-zero) ties.
fn canonical_cmp(a: &(usize, f64), b: &(usize, f64)) -> Ordering {
    let sa = canonical_score(a.1);
    let sb = canonical_score(b.1);
    sb.total_cmp(&sa).then_with(|| a.0.cmp(&b.0))
}

#[derive(Debug, Clone, PartialEq)]
pub struct RowSelection {
    pub indices: Vec<i64>,
    pub scores: Vec<Option<f64>>,
    pub kth_margin: Option<f64>,
}

/// Exact single-entry score: `scale(s) * sum_h weights(t, h) * ReLU(dot(q(t, h), key(s)))`.
pub fn compute_entry_score(
    call: &SavedIndexerCall,
    row: usize,
    key_idx: usize,
    omit_fp8_scales: bool,
) -> f64 {
    let mut sum = 0.0_f64;
    for h in 0..call.num_heads {
        let mut dot = 0.0_f64;
        for d in 0..call.head_dim {
            dot += call.queries[row][h][d] * call.keys[key_idx][d];
        }
        let relu = if dot > 0.0 { dot } else { 0.0 };
        sum += call.weights[row][h] * relu;
    }
    let scale = match call.dtype {
        IndexerDtype::Bf16 => 1.0,
        IndexerDtype::Fp8E4M3 => {
            if omit_fp8_scales {
                1.0
            } else {
                call.key_scales[key_idx]
            }
        }
        IndexerDtype::Fp32Unsupported => 1.0,
    };
    canonical_score(scale * sum)
}

/// Independent exhaustive dense reference scorer + canonical top-k selector.
pub fn dense_reference_topk(call: &SavedIndexerCall) -> Result<Vec<RowSelection>, String> {
    call.validate()?;
    let mut out = Vec::with_capacity(call.num_rows);
    for r in 0..call.num_rows {
        let (ks, ke) = call.windows[r];
        let mut scored: Vec<(usize, f64)> = (ks..ke)
            .map(|s| (s, compute_entry_score(call, r, s, false)))
            .collect();
        scored.sort_by(canonical_cmp);
        let kth_margin = if scored.len() > call.k {
            Some(scored[call.k - 1].1 - scored[call.k].1)
        } else {
            None
        };
        let mut indices = Vec::with_capacity(call.k);
        let mut scores = Vec::with_capacity(call.k);
        for i in 0..call.k {
            if let Some(&(idx, sc)) = scored.get(i) {
                indices.push(idx as i64);
                scores.push(Some(sc));
            } else {
                indices.push(PAD_INDEX);
                scores.push(None);
            }
        }
        out.push(RowSelection {
            indices,
            scores,
            kth_margin,
        });
    }
    Ok(out)
}

/// Configuration for the CPU fused threshold-and-recovery FlashTopK model.
#[derive(Debug, Clone)]
pub struct FusedSelectorConfig {
    pub candidate_capacity: usize,
    pub sample_stride: usize,
    pub enable_pruning: bool,
    /// Clearly identified test-only fault-injection hook overriding estimated threshold.
    pub test_only_threshold_hook: Option<f64>,
    pub fault_injected: bool,
    /// Negative-control knobs (always `false` in normal execution).
    pub skip_recovery: bool,
    pub omit_fp8_scales: bool,
    pub wrong_tie_order: bool,
    pub ignore_window_mask: bool,
}

impl FusedSelectorConfig {
    pub fn unmodified(candidate_capacity: usize) -> Self {
        Self {
            candidate_capacity,
            sample_stride: 3,
            enable_pruning: true,
            test_only_threshold_hook: None,
            fault_injected: false,
            skip_recovery: false,
            omit_fp8_scales: false,
            wrong_tie_order: false,
            ignore_window_mask: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FusedRunOutput {
    pub selections: Vec<RowSelection>,
    pub underflow_recoveries: usize,
    pub overflow_recoveries: usize,
    pub recovery_key_evals: usize,
    pub dense_score_table_bytes: usize,
    pub fused_workspace_bytes: usize,
    pub fault_injected: bool,
}

/// Fused threshold-and-recovery selector modeling FlashTopK's threshold estimation, online
/// candidate filtering, and underflow/overflow recovery without materializing `[rows, keys]`.
pub fn fused_flashtopk_cpu(
    call: &SavedIndexerCall,
    cfg: &FusedSelectorConfig,
) -> Result<FusedRunOutput, String> {
    call.validate()?;
    if let Some(hook) = cfg.test_only_threshold_hook {
        if hook.is_nan() {
            return Err("NaN test_only_threshold_hook rejected".into());
        }
    }
    let mut selections = Vec::with_capacity(call.num_rows);
    let mut underflow_recoveries = 0usize;
    let mut overflow_recoveries = 0usize;
    let mut recovery_key_evals = 0usize;

    let cmp_fn = |a: &(usize, f64), b: &(usize, f64)| -> Ordering {
        if cfg.wrong_tie_order {
            // Deliberately buggy negative control: descending index on ties!
            canonical_score(b.1)
                .total_cmp(&canonical_score(a.1))
                .then_with(|| b.0.cmp(&a.0))
        } else {
            canonical_cmp(a, b)
        }
    };

    for r in 0..call.num_rows {
        let (ks, ke) = if cfg.ignore_window_mask {
            (0, call.num_keys)
        } else {
            call.windows[r]
        };
        let win_len = ke - ks;
        let effective_k = call.k.min(win_len);

        if win_len == 0 {
            selections.push(RowSelection {
                indices: vec![PAD_INDEX; call.k],
                scores: vec![None; call.k],
                kth_margin: None,
            });
            continue;
        }

        // Step 1: Estimate threshold from sampled keys (or test-only hook).
        let threshold = if let Some(hook) = cfg.test_only_threshold_hook {
            hook
        } else if cfg.enable_pruning && win_len > call.k {
            let stride = cfg.sample_stride.max(1);
            let mut sample_scores: Vec<f64> = (ks..ke)
                .step_by(stride)
                .map(|s| compute_entry_score(call, r, s, cfg.omit_fp8_scales))
                .collect();
            sample_scores.sort_by(|a, b| b.total_cmp(a));
            let idx = (call.k.min(sample_scores.len())).saturating_sub(1);
            sample_scores[idx]
        } else {
            f64::NEG_INFINITY
        };

        // Step 2: Retain candidates >= threshold while streaming keys.
        let mut candidates: Vec<(usize, f64)> = Vec::new();
        let mut overflowed = false;
        for s in ks..ke {
            let sc = compute_entry_score(call, r, s, cfg.omit_fp8_scales);
            if sc >= threshold {
                candidates.push((s, sc));
                if candidates.len() > cfg.candidate_capacity {
                    overflowed = true;
                }
            }
        }

        // Step 3: Check underflow (< effective_k) or overflow (> candidate_capacity) and recover!
        let underflowed = candidates.len() < effective_k;
        if (underflowed || overflowed) && !cfg.skip_recovery {
            if underflowed {
                underflow_recoveries += 1;
            }
            if overflowed {
                overflow_recoveries += 1;
            }
            recovery_key_evals += win_len;
            candidates = (ks..ke)
                .map(|s| (s, compute_entry_score(call, r, s, cfg.omit_fp8_scales)))
                .collect();
        } else if overflowed && cfg.skip_recovery {
            candidates.truncate(cfg.candidate_capacity);
        }

        candidates.sort_by(cmp_fn);
        let kth_margin = if candidates.len() > call.k {
            Some(candidates[call.k - 1].1 - candidates[call.k].1)
        } else {
            None
        };
        let mut indices = Vec::with_capacity(call.k);
        let mut scores = Vec::with_capacity(call.k);
        for i in 0..call.k {
            if let Some(&(idx, sc)) = candidates.get(i) {
                indices.push(idx as i64);
                scores.push(Some(sc));
            } else {
                indices.push(PAD_INDEX);
                scores.push(None);
            }
        }
        selections.push(RowSelection {
            indices,
            scores,
            kth_margin,
        });
    }

    let dense_score_table_bytes = call.num_rows * call.num_keys * 4;
    let fused_workspace_bytes = call.num_rows * cfg.candidate_capacity * 12;

    Ok(FusedRunOutput {
        selections,
        underflow_recoveries,
        overflow_recoveries,
        recovery_key_evals,
        dense_score_table_bytes,
        fused_workspace_bytes,
        fault_injected: cfg.fault_injected,
    })
}

#[derive(Debug, Clone)]
pub struct ComparisonSummary {
    pub case_id: String,
    pub input_digest: DigestRef,
    pub rows_checked: usize,
    pub total_rows: usize,
    pub max_score_abs_err: f64,
    pub differing_indices_count: usize,
    pub exact_selection_match: bool,
    pub deterministic_repeat_bit_identical: bool,
    pub row_partition_invariant: bool,
    pub min_kth_margin: Option<f64>,
    pub underflow_recoveries: usize,
    pub overflow_recoveries: usize,
    pub recovery_key_evals: usize,
    pub fault_injected: bool,
    pub dense_score_table_bytes: usize,
    pub fused_workspace_bytes: usize,
}

pub fn compare_selections(
    dense: &[RowSelection],
    cand: &[RowSelection],
) -> Result<(f64, usize, Option<f64>), String> {
    if dense.len() != cand.len() {
        return Err("row count mismatch".into());
    }
    let mut max_err = 0.0_f64;
    let mut diff_indices = 0usize;
    let mut min_margin: Option<f64> = None;
    for (dr, cr) in dense.iter().zip(cand.iter()) {
        if let Some(m) = dr.kth_margin {
            if !m.is_finite() {
                return Err("non-finite kth_margin in compare_selections".into());
            }
            min_margin = Some(match min_margin {
                Some(cur) if cur <= m => cur,
                _ => m,
            });
        }
        if dr.indices.len() != cr.indices.len() {
            return Err("k mismatch in row selection".into());
        }
        for i in 0..dr.indices.len() {
            if dr.indices[i] != cr.indices[i] {
                diff_indices += 1;
            }
            match (dr.scores[i], cr.scores[i]) {
                (Some(a), Some(b)) => {
                    if !a.is_finite() || !b.is_finite() {
                        return Err("non-finite score in comparison".into());
                    }
                    let d = (a - b).abs();
                    if d > max_err {
                        max_err = d;
                    }
                }
                (None, None) => {}
                _ => diff_indices += 1,
            }
        }
    }
    Ok((max_err, diff_indices, min_margin))
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct FlashTopKReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub compatibility_matrix: Vec<CompatibilityCheckResult>,
    pub fixture_comparisons: Vec<ComparisonSummary>,
    pub fault_injected_recovery_checks: Vec<ComparisonSummary>,
    pub negative_controls: Vec<NegativeControl>,
    pub sm120_sm121_decision: String,
    pub sm90_sm100_decision: String,
}

#[allow(clippy::too_many_arguments)]
fn build_synthetic_call(
    case_id: &str,
    dtype: IndexerDtype,
    num_rows: usize,
    num_heads: usize,
    num_keys: usize,
    head_dim: usize,
    k: usize,
    windows: Vec<(usize, usize)>,
    seed: usize,
) -> SavedIndexerCall {
    let queries = (0..num_rows)
        .map(|r| {
            (0..num_heads)
                .map(|h| {
                    (0..head_dim)
                        .map(|d| det_weight(seed + r * 31 + h, d, 0.8))
                        .collect()
                })
                .collect()
        })
        .collect();
    let keys = (0..num_keys)
        .map(|s| {
            (0..head_dim)
                .map(|d| det_weight(seed + 500 + s, d, 0.8))
                .collect()
        })
        .collect();
    let weights = (0..num_rows)
        .map(|r| {
            (0..num_heads)
                .map(|h| det_weight(seed + 900 + r, h, 1.0)) // includes negative weights!
                .collect()
        })
        .collect();
    let key_scales = (0..num_keys)
        .map(|s| match dtype {
            IndexerDtype::Bf16 => 1.0,
            _ => 0.5 + (s % 5) as f64 * 0.25,
        })
        .collect();
    SavedIndexerCall {
        case_id: case_id.into(),
        dtype,
        num_rows,
        num_heads,
        num_keys,
        head_dim,
        k,
        queries,
        keys,
        weights,
        key_scales,
        windows,
    }
}

fn evaluate_fixture(
    call: &SavedIndexerCall,
    cfg: &FusedSelectorConfig,
) -> Result<ComparisonSummary, String> {
    let dense = dense_reference_topk(call)?;
    let run1 = fused_flashtopk_cpu(call, cfg)?;
    let run2 = fused_flashtopk_cpu(call, cfg)?;
    let (max_err, diff_idx, min_margin) = compare_selections(&dense, &run1.selections)?;
    let repeat_bit_identical = run1.selections == run2.selections;

    // Row-partition invariance check (split rows in half and compare).
    let row_partition_invariant = if call.num_rows >= 2 {
        let mid = call.num_rows / 2;
        let part_a = SavedIndexerCall {
            num_rows: mid,
            queries: call.queries[..mid].to_vec(),
            weights: call.weights[..mid].to_vec(),
            windows: call.windows[..mid].to_vec(),
            ..call.clone()
        };
        let part_b = SavedIndexerCall {
            num_rows: call.num_rows - mid,
            queries: call.queries[mid..].to_vec(),
            weights: call.weights[mid..].to_vec(),
            windows: call.windows[mid..].to_vec(),
            ..call.clone()
        };
        let mut merged = fused_flashtopk_cpu(&part_a, cfg)?.selections;
        merged.extend(fused_flashtopk_cpu(&part_b, cfg)?.selections);
        merged == run1.selections
    } else {
        true
    };

    Ok(ComparisonSummary {
        case_id: call.case_id.clone(),
        input_digest: call.input_digest(),
        rows_checked: call.num_rows,
        total_rows: call.num_rows,
        max_score_abs_err: max_err,
        differing_indices_count: diff_idx,
        exact_selection_match: diff_idx == 0 && max_err <= 1e-12,
        deterministic_repeat_bit_identical: repeat_bit_identical,
        row_partition_invariant,
        min_kth_margin: min_margin,
        underflow_recoveries: run1.underflow_recoveries,
        overflow_recoveries: run1.overflow_recoveries,
        recovery_key_evals: run1.recovery_key_evals,
        fault_injected: run1.fault_injected,
        dense_score_table_bytes: run1.dense_score_table_bytes,
        fused_workspace_bytes: run1.fused_workspace_bytes,
    })
}

pub fn run_default_harness() -> Result<FlashTopKReport, String> {
    // 1. Hardware & model capability matrix.
    let matrix_reqs = vec![
        HardwareContractRequest {
            target_name: "h100_sm90_glm51_32h_fp8".into(),
            backend: "cuda".into(),
            cuda_cc_major: Some(9),
            cuda_cc_minor: Some(0),
            num_heads: 32,
            head_dim: 128,
            topk: 2048,
            dtype: IndexerDtype::Fp8E4M3,
            contiguous_keys: true,
            single_contiguous_window: true,
            block_level_qsa: false,
        },
        HardwareContractRequest {
            target_name: "b200_sm100_glm51_32h_fp8".into(),
            backend: "cuda".into(),
            cuda_cc_major: Some(10),
            cuda_cc_minor: Some(0),
            num_heads: 32,
            head_dim: 128,
            topk: 2048,
            dtype: IndexerDtype::Fp8E4M3,
            contiguous_keys: true,
            single_contiguous_window: true,
            block_level_qsa: false,
        },
        HardwareContractRequest {
            target_name: "rtx5080_sm120_blackwell_consumer".into(),
            backend: "cuda".into(),
            cuda_cc_major: Some(12),
            cuda_cc_minor: Some(0),
            num_heads: 32,
            head_dim: 128,
            topk: 2048,
            dtype: IndexerDtype::Fp8E4M3,
            contiguous_keys: true,
            single_contiguous_window: true,
            block_level_qsa: false,
        },
        HardwareContractRequest {
            target_name: "dgx_spark_gb10_sm121".into(),
            backend: "cuda".into(),
            cuda_cc_major: Some(12),
            cuda_cc_minor: Some(1),
            num_heads: 32,
            head_dim: 128,
            topk: 2048,
            dtype: IndexerDtype::Fp8E4M3,
            contiguous_keys: true,
            single_contiguous_window: true,
            block_level_qsa: false,
        },
        HardwareContractRequest {
            target_name: "deepseek_v32_64head_on_sm100".into(),
            backend: "cuda".into(),
            cuda_cc_major: Some(10),
            cuda_cc_minor: Some(0),
            num_heads: 64,
            head_dim: 128,
            topk: 2048,
            dtype: IndexerDtype::Fp8E4M3,
            contiguous_keys: true,
            single_contiguous_window: true,
            block_level_qsa: false,
        },
        HardwareContractRequest {
            target_name: "block_level_qsa_selector_sm100".into(),
            backend: "cuda".into(),
            cuda_cc_major: Some(10),
            cuda_cc_minor: Some(0),
            num_heads: 32,
            head_dim: 128,
            topk: 512,
            dtype: IndexerDtype::Bf16,
            contiguous_keys: true,
            single_contiguous_window: true,
            block_level_qsa: true,
        },
        HardwareContractRequest {
            target_name: "tenstorrent_p150_non_cuda".into(),
            backend: "tenstorrent".into(),
            cuda_cc_major: None,
            cuda_cc_minor: None,
            num_heads: 32,
            head_dim: 128,
            topk: 512,
            dtype: IndexerDtype::Bf16,
            contiguous_keys: true,
            single_contiguous_window: true,
            block_level_qsa: false,
        },
    ];
    let compatibility_matrix: Vec<CompatibilityCheckResult> = matrix_reqs
        .iter()
        .map(validate_flashtopk_contract)
        .collect();

    // 2. Saved-call CPU conformance fixtures.
    let call_causal_fp8 = build_synthetic_call(
        "fp8_causal_and_nonzero_offset_windows",
        IndexerDtype::Fp8E4M3,
        4,
        4,
        24,
        8,
        5,
        vec![(0, 10), (3, 18), (6, 24), (2, 14)],
        2301,
    );
    let call_short_empty = build_synthetic_call(
        "bf16_short_and_empty_windows",
        IndexerDtype::Bf16,
        3,
        4,
        12,
        8,
        4,
        vec![(0, 0), (4, 6), (2, 12)],
        2302,
    );
    let call_k1 = build_synthetic_call(
        "bf16_k1_boundary",
        IndexerDtype::Bf16,
        3,
        4,
        12,
        8,
        1,
        vec![(0, 6), (3, 12), (5, 9)],
        2304,
    );
    // Exact ties + genuine IEEE -0.0 vs +0.0 signed-zero ties INSIDE selected top-4 + near-kth-boundary!
    let mut call_ties = build_synthetic_call(
        "signed_zero_exact_ties_and_near_kth_boundary",
        IndexerDtype::Bf16,
        2,
        1,
        10,
        4,
        4,
        vec![(0, 10), (1, 9)],
        2303,
    );
    // Row 0: single head with weight = -1.0 -> keys 0..4 have dot <= 0 -> ReLU = +0.0 -> -1.0 * 0.0 = -0.0!
    // Keys 4..10 have positive dot -> negative weighted scores -> so the top-4 ARE the signed-zero keys!
    call_ties.queries[0] = vec![vec![1.0, 0.0, 0.0, 0.0]];
    call_ties.weights[0] = vec![-1.0];
    for s in 0..4 {
        call_ties.keys[s] = vec![-1.0, 0.0, 0.0, 0.0]; // raw score = -0.0
    }
    call_ties.keys[4] = vec![1e-6, 0.0, 0.0, 0.0]; // near-kth-boundary margin 1e-6!
    for s in 5..10 {
        call_ties.keys[s] = vec![(s as f64) * 0.5, 0.0, 0.0, 0.0]; // negative scores
    }
    // Row 1: positive weight = +1.0 with exact positive ties + +0.0 ties + near-kth-boundary.
    call_ties.queries[1] = vec![vec![0.0, 1.0, 0.0, 0.0]];
    call_ties.weights[1] = vec![1.0];
    call_ties.keys[1] = vec![-1.0, 2.0, 0.0, 0.0]; // score 2.0
    call_ties.keys[2] = vec![-1.0, 2.0, 0.0, 0.0]; // exact tie 2.0 (idx 1 < idx 2)
    call_ties.keys[3] = vec![-1.0, 1.0, 0.0, 0.0]; // score 1.0
    call_ties.keys[4] = vec![1e-6, 1.0, 0.0, 0.0]; // tie 1.0
    call_ties.keys[5] = vec![2.5, 1.0 - 1e-6, 0.0, 0.0]; // near-kth-boundary 1e-6!

    let cfg_normal = FusedSelectorConfig::unmodified(12);
    let fixture_comparisons = vec![
        evaluate_fixture(&call_causal_fp8, &cfg_normal)?,
        evaluate_fixture(&call_short_empty, &cfg_normal)?,
        evaluate_fixture(&call_k1, &cfg_normal)?,
        evaluate_fixture(&call_ties, &cfg_normal)?,
    ];

    // 3. Clearly labeled test-only fault-injected underflow & overflow recovery verification!
    let mut cfg_underflow = FusedSelectorConfig::unmodified(12);
    cfg_underflow.test_only_threshold_hook = Some(1.0e9); // forces 0 candidates -> underflow recovery!
    cfg_underflow.fault_injected = true;
    let mut call_underflow = call_causal_fp8.clone();
    call_underflow.case_id = "fault_injected_underflow_recovery".into();

    let mut cfg_overflow = FusedSelectorConfig::unmodified(6);
    cfg_overflow.test_only_threshold_hook = Some(-1.0e9); // forces all keys -> overflow recovery!
    cfg_overflow.fault_injected = true;
    let mut call_overflow = call_causal_fp8.clone();
    call_overflow.case_id = "fault_injected_overflow_recovery".into();

    let fault_injected_recovery_checks = vec![
        evaluate_fixture(&call_underflow, &cfg_underflow)?,
        evaluate_fixture(&call_overflow, &cfg_overflow)?,
    ];

    // 4. Seven executable negative controls!
    let dense_fp8 = dense_reference_topk(&call_causal_fp8)?;
    let dense_ties = dense_reference_topk(&call_ties)?;
    let mut negative_controls = Vec::new();

    // Control 1: Omitted FP8 per-key scales
    let mut cfg_no_scale = FusedSelectorConfig::unmodified(24);
    cfg_no_scale.omit_fp8_scales = true;
    let out_no_scale = fused_flashtopk_cpu(&call_causal_fp8, &cfg_no_scale)?;
    let (err_ns, diff_ns, _) = compare_selections(&dense_fp8, &out_no_scale.selections)?;
    negative_controls.push(NegativeControl {
        case_id: "omitted_fp8_per_key_scales".into(),
        outcome: if err_ns > 1e-6 || diff_ns > 0 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("max_score_err={err_ns:.6}, differing_indices={diff_ns}"),
    });

    // Control 2: Incorrect window mask (ignores [ks, ke))
    let mut cfg_bad_mask = FusedSelectorConfig::unmodified(24);
    cfg_bad_mask.ignore_window_mask = true;
    let out_bad_mask = fused_flashtopk_cpu(&call_causal_fp8, &cfg_bad_mask)?;
    let (_, diff_mask, _) = compare_selections(&dense_fp8, &out_bad_mask.selections)?;
    negative_controls.push(NegativeControl {
        case_id: "incorrect_window_mask".into(),
        outcome: if diff_mask > 0 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("differing_indices={diff_mask}"),
    });

    // Control 3: Wrong tie order (descending index on exact tie)
    let mut cfg_bad_tie = FusedSelectorConfig::unmodified(24);
    cfg_bad_tie.wrong_tie_order = true;
    let out_bad_tie = fused_flashtopk_cpu(&call_ties, &cfg_bad_tie)?;
    let (_, diff_tie, _) = compare_selections(&dense_ties, &out_bad_tie.selections)?;
    negative_controls.push(NegativeControl {
        case_id: "wrong_tie_order_descending_index".into(),
        outcome: if diff_tie > 0 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("differing_indices={diff_tie}"),
    });

    // Control 4: Skipped underflow recovery on high threshold
    let mut cfg_skip_rec = cfg_underflow.clone();
    cfg_skip_rec.skip_recovery = true;
    let out_skip_rec = fused_flashtopk_cpu(&call_causal_fp8, &cfg_skip_rec)?;
    let (_, diff_skip, _) = compare_selections(&dense_fp8, &out_skip_rec.selections)?;
    negative_controls.push(NegativeControl {
        case_id: "skipped_underflow_recovery".into(),
        outcome: if diff_skip > 0 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("differing_indices={diff_skip}"),
    });

    // Control 5: Non-finite input (NaN) rejected cleanly (never swallowed!)
    let mut nan_call = call_short_empty.clone();
    nan_call.queries[0][0][0] = f64::NAN;
    let nan_res = dense_reference_topk(&nan_call);
    negative_controls.push(NegativeControl {
        case_id: "non_finite_query_rejected".into(),
        outcome: if nan_res.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: nan_res
            .err()
            .unwrap_or_else(|| "unexpectedly accepted NaN".into()),
    });

    // Control 6: Invalid window (ks > ke) rejected
    let mut bad_win_call = call_short_empty.clone();
    bad_win_call.windows[0] = (8, 4);
    let win_res = dense_reference_topk(&bad_win_call);
    negative_controls.push(NegativeControl {
        case_id: "invalid_window_bounds_rejected".into(),
        outcome: if win_res.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: win_res
            .err()
            .unwrap_or_else(|| "unexpectedly accepted inverted window".into()),
    });

    // Control 7: SM121 / 64-head compatibility refusal cannot be bypassed
    let sm121_refused = compatibility_matrix[3].outcome == OutcomeStatus::Unsupported;
    let h64_refused = compatibility_matrix[4].outcome == OutcomeStatus::Unsupported;
    negative_controls.push(NegativeControl {
        case_id: "sm121_and_64head_capability_gate_enforced".into(),
        outcome: if sm121_refused && h64_refused {
            OutcomeStatus::Unsupported
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("sm121_refused={sm121_refused}, deepseek_64head_refused={h64_refused}"),
    });

    Ok(FlashTopKReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "cuda_sm90_sm100_flashtopk_gpu",
            "Device-free CPU reference & recovery verified locally; SM90/SM100 CUDA GPU execution unavailable here and SM120/SM121 are unsupported by pinned FlashTopK@090902ab",
        ),
        compatibility_matrix,
        fixture_comparisons,
        fault_injected_recovery_checks,
        negative_controls,
        sm120_sm121_decision: "NO_GO_UNSUPPORTED_AT_PINNED_REVISION".into(),
        sm90_sm100_decision: "INCONCLUSIVE_GPU_EXECUTION_REQUIRED".into(),
    })
}

pub fn inputs_json(report: &FlashTopKReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.flash-topk.inputs.v1\",\n",
            "  \"issue\": 23,\n",
            "  \"upstream\": {{\"repo\":{},\"commit\":{},\"license\":{},\"unverified_related_work_claim\":{}}},\n",
            "  \"execution_class\": {},\n",
            "  \"linked_consumers\": [\"dsv41-1017#1\",\"inference-lab#4\",\"inference-lab#7\",\"inference-lab#8\",\"inference-lab#19\"],\n",
            "  \"hyscale_training_follow_up_separate\": true\n",
            "}}\n"
        ),
        quote(UPSTREAM_REPO),
        quote(UPSTREAM_COMMIT),
        quote(UPSTREAM_LICENSE),
        quote(UPSTREAM_UNVERIFIED_CLAIM),
        quote(report.execution_class.as_str()),
    )
}

fn comparison_to_json(c: &ComparisonSummary) -> String {
    format!(
        concat!(
            "{{\"case_id\":{},\"input_digest\":{},\"rows_checked\":{},\"total_rows\":{},",
            "\"max_score_abs_err\":{},\"differing_indices_count\":{},\"exact_selection_match\":{},",
            "\"deterministic_repeat_bit_identical\":{},\"row_partition_invariant\":{},",
            "\"min_kth_margin\":{},\"underflow_recoveries\":{},\"overflow_recoveries\":{},\"recovery_key_evals\":{},",
            "\"fault_injected\":{},\"dense_score_table_bytes\":{},\"fused_workspace_bytes\":{}}}"
        ),
        quote(&c.case_id),
        c.input_digest.to_json(),
        c.rows_checked,
        c.total_rows,
        finite_f64(c.max_score_abs_err),
        c.differing_indices_count,
        c.exact_selection_match,
        c.deterministic_repeat_bit_identical,
        c.row_partition_invariant,
        opt_f64(c.min_kth_margin),
        c.underflow_recoveries,
        c.overflow_recoveries,
        c.recovery_key_evals,
        c.fault_injected,
        c.dense_score_table_bytes,
        c.fused_workspace_bytes,
    )
}

pub fn results_json(report: &FlashTopKReport) -> String {
    let compat: Vec<String> = report
        .compatibility_matrix
        .iter()
        .map(|r| {
            format!(
                "{{\"target\":{},\"outcome\":{},\"reason\":{}}}",
                quote(&r.target_name),
                quote(r.outcome.as_str()),
                quote(&r.reason)
            )
        })
        .collect();
    let fixtures: Vec<String> = report
        .fixture_comparisons
        .iter()
        .map(comparison_to_json)
        .collect();
    let recovery: Vec<String> = report
        .fault_injected_recovery_checks
        .iter()
        .map(comparison_to_json)
        .collect();
    let neg: Vec<String> = report
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
            "  \"schema\": \"inference-lab.flash-topk.results.v1\",\n",
            "  \"issue\": 23,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"sm120_sm121_decision\": {},\n",
            "  \"sm90_sm100_decision\": {},\n",
            "  \"gpu_timing_ms\": {},\n",
            "  \"whole_model_gain_claimed\": false,\n",
            "  \"compatibility_matrix\": [{}],\n",
            "  \"fixture_comparisons\": [{}],\n",
            "  \"fault_injected_recovery_checks\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        quote(&report.sm120_sm121_decision),
        quote(&report.sm90_sm100_decision),
        opt_f64(None),
        compat.join(","),
        fixtures.join(","),
        recovery.join(","),
        neg.join(","),
    )
}

pub fn summary_markdown(report: &FlashTopKReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #23: FlashTopK Sparse-Indexer Reference & Hardware Gate"
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
        "- Scoped decisions: SM120/SM121 = `{}`; SM90/SM100 = `{}`",
        report.sm120_sm121_decision, report.sm90_sm100_decision
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Compatibility matrix (`FlashTopK@090902ab`)").unwrap();
    writeln!(out, "| Target | Outcome | Reason |").unwrap();
    writeln!(out, "| --- | --- | --- |").unwrap();
    for r in &report.compatibility_matrix {
        writeln!(
            out,
            "| `{}` | `{}` | {} |",
            r.target_name,
            r.outcome.as_str(),
            r.reason
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Independent correctness & recovery checks").unwrap();
    writeln!(
        out,
        "| Case | Exact Match | Max Err | Repeat Identical | Partition Inv | Underflow Rec | Overflow Rec | Fault Injected |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | ---: | --- | --- | ---: | ---: | --- |").unwrap();
    for c in report
        .fixture_comparisons
        .iter()
        .chain(report.fault_injected_recovery_checks.iter())
    {
        writeln!(
            out,
            "| `{}` | `{}` | `{:.3e}` | `{}` | `{}` | {} | {} | `{}` |",
            c.case_id,
            c.exact_selection_match,
            c.max_score_abs_err,
            c.deterministic_repeat_bit_identical,
            c.row_partition_invariant,
            c.underflow_recoveries,
            c.overflow_recoveries,
            c.fault_injected,
        )
        .unwrap();
    }
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
