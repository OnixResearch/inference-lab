//! Issue #31: Evaluate Qwen3.8-27B QSA speed and long-context task accuracy against a matched
//! dense control.
//!
//! Pinned upstream boundaries:
//! - `kuraneko1/Qwen3.8-27B-QSA-llama.cpp@864934c78136ad75274f202b8eb1e7c751d81fb1`
//!   (upstream 1.793x decode / 2.921x cold-prefill at 261,120 tokens on 2x32 GB sm70 Q8_0 remains
//!   strictly unverified non-gating related work; packaged x86-64/sm70 binary does NOT qualify
//!   ARM64/GB10 DGX Spark!).
//! - `higa2515/Qwen3.8-27B-QSA-{BF16,GGUF}` (frozen backbone + separately trained indexers).
//! - `pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF` (motivates retrieval-loss failure cases
//!   without attributing its results to the original artifact).
//! - QSA architecture paper `arXiv:2608.30320`.
//! - `#23` FlashTopK does NOT support 4-token block-level QSA (not a drop-in dependency!).
//!
//! Provides:
//! - Pure-Rust 4-token block-level QSA selector vs matched dense control (`LLAMA_QSA_DENSE=1`)
//!   across the matched `2x2` matrix: `(dense, qsa) x (mtp_off, mtp_draft2)`.
//! - Explicit memory & state accounting showing QSA retains the FULL target KV cache alongside
//!   indexer buffers, GDN state, MTP state, and workspace.
//! - Frozen long-context matrix (`8k`, `64k`, `128k`, plus bounded `261k` profile check) with
//!   decisive evidence at start, middle, and end across 4 independently checked task families
//!   (exact lookup with near-match decoys, corrected definition, dependency chain, multi-file edit).
//! - Offline rescorer preserving full task populations (dense-control failures, sparse-induced
//!   losses, undefined cost at zero accepted tasks) and 8 executable negative controls.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const QSA_ENGINE_COMMIT: &str = "864934c78136ad75274f202b8eb1e7c751d81fb1";
pub const QSA_PAPER: &str = "https://arxiv.org/abs/2608.30320";
pub const UPSTREAM_UNVERIFIED_CLAIM: &str =
    "1.793x decode and 2.921x cold-prefill at 261,120 tokens on 2x32GB sm70 Q8_0 FP16 KV MTP draft2 (non-gating)";
pub const QSA_BLOCK_SIZE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    DenseControl, // LLAMA_QSA_DENSE=1
    QsaBlockSparse,
}

impl AttentionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DenseControl => "dense_control",
            Self::QsaBlockSparse => "qsa_block_sparse",
        }
    }
}

#[derive(Debug, Clone)]
pub struct QsaAttentionResult {
    pub mode: AttentionMode,
    pub mtp_draft2: bool,
    pub context_tokens: usize,
    pub selected_tokens: usize,
    pub indexer_keys_scanned: usize,
    pub output_vec: Vec<f64>,
    pub target_kv_bytes: usize,
    pub indexer_buffer_bytes: usize,
    pub gdn_state_bytes: usize,
    pub mtp_state_bytes: usize,
    pub workspace_bytes: usize,
}

/// Pure-Rust 4-token block-level QSA attention vs matched dense attention (`LLAMA_QSA_DENSE=1`).
/// Retains the FULL target KV cache in both modes and scans all context-dependent block index keys!
pub fn run_qsa_or_dense_attention(
    mode: AttentionMode,
    mtp_draft2: bool,
    query: &[f64],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    top_blocks: usize,
    bug_ignore_dense_flag: bool,
) -> Result<QsaAttentionResult, String> {
    let n = keys.len();
    let d = query.len();
    if n == 0 || d == 0 || values.len() != n || n % QSA_BLOCK_SIZE != 0 {
        return Err(
            "context length must be a positive multiple of QSA block size 4 with d >= 1".into(),
        );
    }
    if query.iter().any(|x| !x.is_finite())
        || keys
            .iter()
            .any(|k| k.len() != d || k.iter().any(|x| !x.is_finite()))
        || values
            .iter()
            .any(|v| v.len() != d || v.iter().any(|x| !x.is_finite()))
    {
        return Err("mismatched dimension or non-finite query/key/value in QSA attention".into());
    }
    let num_blocks = n / QSA_BLOCK_SIZE;
    let effective_sparse = match mode {
        AttentionMode::DenseControl => bug_ignore_dense_flag,
        AttentionMode::QsaBlockSparse => true,
    };

    let selected_indices: Vec<usize> = if !effective_sparse {
        (0..n).collect()
    } else {
        // Compute 4-token block index keys (mean of block keys) and scan ALL blocks!
        let mut block_scores: Vec<(usize, f64)> = (0..num_blocks)
            .map(|b| {
                let mut block_key = vec![0.0_f64; d];
                for off in 0..QSA_BLOCK_SIZE {
                    for (k, bk) in block_key.iter_mut().enumerate().take(d) {
                        *bk += keys[b * QSA_BLOCK_SIZE + off][k] / (QSA_BLOCK_SIZE as f64);
                    }
                }
                let sc: f64 = query.iter().zip(block_key.iter()).map(|(a, b)| a * b).sum();
                (b, sc)
            })
            .collect();
        block_scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let keep_b = top_blocks.clamp(1, num_blocks);
        let mut toks = Vec::with_capacity(keep_b * QSA_BLOCK_SIZE);
        for &(b, _) in block_scores.iter().take(keep_b) {
            for off in 0..QSA_BLOCK_SIZE {
                toks.push(b * QSA_BLOCK_SIZE + off);
            }
        }
        toks.sort_unstable();
        toks
    };

    let scale = 1.0 / (d as f64).sqrt();
    let mut scores = Vec::with_capacity(selected_indices.len());
    for &idx in &selected_indices {
        let dot: f64 = query.iter().zip(keys[idx].iter()).map(|(a, b)| a * b).sum();
        scores.push(dot * scale);
    }
    let probs = softmax_stable(&scores)?;
    let mut out = vec![0.0_f64; d];
    for (i, &idx) in selected_indices.iter().enumerate() {
        for (k, o) in out.iter_mut().enumerate().take(d) {
            *o += probs[i] * values[idx][k];
        }
    }

    // Honest memory accounting: QSA retains the FULL target KV cache + adds indexer buffers!
    let target_kv_bytes = n * d * 2 * 2;
    let indexer_buffer_bytes = if effective_sparse {
        num_blocks * d * 2 + num_blocks * 4
    } else {
        0
    };
    let gdn_state_bytes = d * d * 4;
    let mtp_state_bytes = if mtp_draft2 { 2 * d * 4 } else { 0 };
    let workspace_bytes = selected_indices.len() * 8;

    Ok(QsaAttentionResult {
        mode,
        mtp_draft2,
        context_tokens: n,
        selected_tokens: selected_indices.len(),
        indexer_keys_scanned: if effective_sparse { num_blocks } else { 0 },
        output_vec: out,
        target_kv_bytes,
        indexer_buffer_bytes,
        gdn_state_bytes,
        mtp_state_bytes,
        workspace_bytes,
    })
}

/// Four independently checked long-context task families (hidden checks kept out of inputs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongContextTaskKind {
    ExactLookupWithDecoys,
    CorrectedDefinition,
    DependencyChain,
    MultiFileEdit,
}

impl LongContextTaskKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExactLookupWithDecoys => "exact_lookup_with_decoys",
            Self::CorrectedDefinition => "corrected_definition",
            Self::DependencyChain => "dependency_chain",
            Self::MultiFileEdit => "multi_file_edit",
        }
    }
}

pub fn check_qsa_task(kind: LongContextTaskKind, output: &str) -> OutcomeStatus {
    match kind {
        LongContextTaskKind::ExactLookupWithDecoys => {
            if output.trim() == "TOKEN-7F3A-EXACT" {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            }
        }
        LongContextTaskKind::CorrectedDefinition => {
            if output.trim() == "max_retry_budget=12 (supersedes 4)" {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            }
        }
        LongContextTaskKind::DependencyChain => {
            if output.trim() == "mod_a -> mod_b -> mod_c:port=9100" {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            }
        }
        LongContextTaskKind::MultiFileEdit => {
            // Independent syntax & cross-file invariant check!
            if output.contains("file:api.rs:pub fn limit() -> usize { 64 }")
                && output.contains("file:test.rs:assert_eq!(limit(), 64);")
            {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PairedLongContextRecord {
    pub task_id: String,
    pub kind: LongContextTaskKind,
    pub context_label: &'static str,
    pub evidence_position: &'static str,
    pub trial_order: &'static str,
    pub dense_status: OutcomeStatus,
    pub qsa_status: OutcomeStatus,
    pub dense_cost_ms: f64,
    pub qsa_cost_ms: f64,
}

#[derive(Debug, Clone)]
pub struct QsaPopulationSummary {
    pub planned_tasks: usize,
    pub dense_accepted: usize,
    pub qsa_accepted: usize,
    pub sparse_induced_losses: usize,
    pub both_failed_not_retained_capability: usize,
    pub dense_cost_per_accepted_ms: Option<f64>,
    pub qsa_cost_per_accepted_ms: Option<f64>,
}

/// Offline population-complete rescorer (rejects incomplete populations; separates sparse losses
/// from tasks dense could not solve; cost per accepted task is `None` when 0 tasks succeed!).
pub fn rescore_qsa_population(
    expected_planned: usize,
    records: &[PairedLongContextRecord],
) -> Result<QsaPopulationSummary, String> {
    if expected_planned == 0 || records.len() != expected_planned {
        return Err(format!(
            "incomplete or empty population: observed {} vs expected {expected_planned}",
            records.len()
        ));
    }
    let mut seen_ids = BTreeSet::new();
    let mut dense_ok = 0usize;
    let mut qsa_ok = 0usize;
    let mut sparse_losses = 0usize;
    let mut both_failed = 0usize;
    let mut dense_ms = 0.0_f64;
    let mut qsa_ms = 0.0_f64;
    for r in records {
        if !seen_ids.insert(r.task_id.as_str()) {
            return Err(format!(
                "duplicate task_id `{}` in QSA population",
                r.task_id
            ));
        }
        if !r.dense_cost_ms.is_finite()
            || r.dense_cost_ms < 0.0
            || !r.qsa_cost_ms.is_finite()
            || r.qsa_cost_ms < 0.0
        {
            return Err(format!(
                "non-finite or negative cost on task `{}`",
                r.task_id
            ));
        }
        dense_ms += r.dense_cost_ms;
        qsa_ms += r.qsa_cost_ms;
        let d = r.dense_status == OutcomeStatus::Accepted;
        let q = r.qsa_status == OutcomeStatus::Accepted;
        if d {
            dense_ok += 1;
        }
        if q {
            qsa_ok += 1;
        }
        if d && !q {
            sparse_losses += 1;
        }
        if !d && !q {
            both_failed += 1;
        }
    }
    Ok(QsaPopulationSummary {
        planned_tasks: records.len(),
        dense_accepted: dense_ok,
        qsa_accepted: qsa_ok,
        sparse_induced_losses: sparse_losses,
        both_failed_not_retained_capability: both_failed,
        dense_cost_per_accepted_ms: if dense_ok == 0 {
            None
        } else {
            Some(dense_ms / dense_ok as f64)
        },
        qsa_cost_per_accepted_ms: if qsa_ok == 0 {
            None
        } else {
            Some(qsa_ms / qsa_ok as f64)
        },
    })
}

/// Structured QSA execution & attribution claim validated by `validate_qsa_qualification_claim`.
#[derive(Debug, Clone)]
pub struct QsaQualificationClaim {
    pub binary_target_arch: String,
    pub host_target_arch: String,
    pub uses_flashtopk_23_for_block_qsa: bool,
    pub substitutes_bf16_ppl_for_native_task: bool,
    pub retrained_indexer_arm: bool,
    pub backbone_quant_runtime_fixed: bool,
}

pub fn validate_qsa_qualification_claim(claim: &QsaQualificationClaim) -> (OutcomeStatus, String) {
    if claim.binary_target_arch != claim.host_target_arch {
        return (
            OutcomeStatus::Unsupported,
            format!(
                "prebuilt `{}` binary does not qualify `{}` target",
                claim.binary_target_arch, claim.host_target_arch
            ),
        );
    }
    if claim.uses_flashtopk_23_for_block_qsa {
        return (
            OutcomeStatus::Unsupported,
            "released #23 FlashTopK@090902ab contract does not support 4-token block-level QSA"
                .into(),
        );
    }
    if claim.substitutes_bf16_ppl_for_native_task {
        return (
            OutcomeStatus::Invalid,
            "BF16 reference/PPL or attention-mass capture cannot substitute for native quantized task accuracy".into(),
        );
    }
    if claim.retrained_indexer_arm && !claim.backbone_quant_runtime_fixed {
        return (
            OutcomeStatus::Invalid,
            "retrained-indexer arm with changed backbone/quant/runtime must be labeled whole-configuration comparison".into(),
        );
    }
    (
        OutcomeStatus::Accepted,
        "valid QSA qualification claim".into(),
    )
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct Qwen38QsaReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub matrix_2x2: Vec<QsaAttentionResult>,
    pub population: QsaPopulationSummary,
    pub zero_accepted_cost_undefined: bool,
    pub negative_controls: Vec<NegativeControl>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<Qwen38QsaReport, String> {
    let d = 8usize;
    let ctx_len = 64usize; // 16 blocks of 4 tokens
    let query: Vec<f64> = (0..d).map(|k| det_weight(3100, k, 0.9)).collect();
    let keys: Vec<Vec<f64>> = (0..ctx_len)
        .map(|i| (0..d).map(|k| det_weight(3101 + i, k, 0.8)).collect())
        .collect();
    let values: Vec<Vec<f64>> = (0..ctx_len)
        .map(|i| (0..d).map(|k| det_weight(3201 + i, k, 0.8)).collect())
        .collect();

    // 1. Matched 2x2 matrix: (dense, qsa) x (mtp_off, mtp_draft2).
    let d_off = run_qsa_or_dense_attention(
        AttentionMode::DenseControl,
        false,
        &query,
        &keys,
        &values,
        4,
        false,
    )?;
    let d_on = run_qsa_or_dense_attention(
        AttentionMode::DenseControl,
        true,
        &query,
        &keys,
        &values,
        4,
        false,
    )?;
    let q_off = run_qsa_or_dense_attention(
        AttentionMode::QsaBlockSparse,
        false,
        &query,
        &keys,
        &values,
        4,
        false,
    )?;
    let q_on = run_qsa_or_dense_attention(
        AttentionMode::QsaBlockSparse,
        true,
        &query,
        &keys,
        &values,
        4,
        false,
    )?;

    // Verify within-target MTP parity (MTP off vs MTP on produces identical target attention outputs!)
    if d_off.output_vec != d_on.output_vec || q_off.output_vec != q_on.output_vec {
        return Err("within-target MTP off/on attention mismatch".into());
    }
    // Verify LLAMA_QSA_DENSE=1 selected all tokens and QSA retained full target KV cache!
    if d_off.selected_tokens != ctx_len || q_off.target_kv_bytes != d_off.target_kv_bytes {
        return Err("dense selection or full target KV retention invariant violated".into());
    }

    // 2. Frozen context/task matrix across start/middle/end placements (4 task families + 1 dense-failed case).
    let records = vec![
        PairedLongContextRecord {
            task_id: "8k_start_lookup".into(),
            kind: LongContextTaskKind::ExactLookupWithDecoys,
            context_label: "8k",
            evidence_position: "start",
            trial_order: "dense_then_qsa",
            dense_status: check_qsa_task(
                LongContextTaskKind::ExactLookupWithDecoys,
                "TOKEN-7F3A-EXACT",
            ),
            qsa_status: check_qsa_task(
                LongContextTaskKind::ExactLookupWithDecoys,
                "TOKEN-7F3A-EXACT",
            ),
            dense_cost_ms: 420.0,
            qsa_cost_ms: 430.0,
        },
        PairedLongContextRecord {
            task_id: "64k_middle_corrected_def".into(),
            kind: LongContextTaskKind::CorrectedDefinition,
            context_label: "64k",
            evidence_position: "middle",
            trial_order: "qsa_then_dense",
            dense_status: check_qsa_task(
                LongContextTaskKind::CorrectedDefinition,
                "max_retry_budget=12 (supersedes 4)",
            ),
            qsa_status: check_qsa_task(
                LongContextTaskKind::CorrectedDefinition,
                "max_retry_budget=4",
            ), // sparse loss!
            dense_cost_ms: 1950.0,
            qsa_cost_ms: 1380.0,
        },
        PairedLongContextRecord {
            task_id: "128k_end_dep_chain".into(),
            kind: LongContextTaskKind::DependencyChain,
            context_label: "128k",
            evidence_position: "end",
            trial_order: "dense_then_qsa",
            dense_status: check_qsa_task(
                LongContextTaskKind::DependencyChain,
                "mod_a -> mod_b -> mod_c:port=9100",
            ),
            qsa_status: check_qsa_task(
                LongContextTaskKind::DependencyChain,
                "mod_a -> mod_b -> mod_c:port=9100",
            ),
            dense_cost_ms: 4100.0,
            qsa_cost_ms: 2400.0,
        },
        PairedLongContextRecord {
            task_id: "128k_middle_multifile_edit".into(),
            kind: LongContextTaskKind::MultiFileEdit,
            context_label: "128k",
            evidence_position: "middle",
            trial_order: "qsa_then_dense",
            dense_status: check_qsa_task(
                LongContextTaskKind::MultiFileEdit,
                "file:api.rs:pub fn limit() -> usize { 64 }\nfile:test.rs:assert_eq!(limit(), 64);",
            ),
            qsa_status: check_qsa_task(
                LongContextTaskKind::MultiFileEdit,
                "file:api.rs:pub fn limit() -> usize { 64 }\nfile:test.rs:assert_eq!(limit(), 64);",
            ),
            dense_cost_ms: 4250.0,
            qsa_cost_ms: 2520.0,
        },
        PairedLongContextRecord {
            task_id: "261k_bounded_both_failed_preserved".into(),
            kind: LongContextTaskKind::ExactLookupWithDecoys,
            context_label: "261k_bounded_profile",
            evidence_position: "middle",
            trial_order: "dense_then_qsa",
            dense_status: OutcomeStatus::Timeout,
            qsa_status: OutcomeStatus::Invalid,
            dense_cost_ms: 10000.0,
            qsa_cost_ms: 5200.0,
        },
    ];
    let population = rescore_qsa_population(5, &records)?;

    let zero_pop = rescore_qsa_population(
        1,
        &[PairedLongContextRecord {
            task_id: "zero".into(),
            kind: LongContextTaskKind::ExactLookupWithDecoys,
            context_label: "8k",
            evidence_position: "start",
            trial_order: "dense_then_qsa",
            dense_status: OutcomeStatus::Invalid,
            qsa_status: OutcomeStatus::Oom,
            dense_cost_ms: 500.0,
            qsa_cost_ms: 100.0,
        }],
    )?;
    let zero_accepted_cost_undefined = zero_pop.dense_cost_per_accepted_ms.is_none()
        && zero_pop.qsa_cost_per_accepted_ms.is_none();

    // 3. Eight executable negative controls!
    let mut negative_controls = Vec::new();

    // Control 1: LLAMA_QSA_DENSE=1 ignored (dense control silently ran sparse path)
    let bad_dense = run_qsa_or_dense_attention(
        AttentionMode::DenseControl,
        false,
        &query,
        &keys,
        &values,
        4,
        true,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "llama_qsa_dense_flag_ignored_detected".into(),
        outcome: if bad_dense.selected_tokens != ctx_len {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!(
            "selected_tokens={} != context_tokens={ctx_len}",
            bad_dense.selected_tokens
        ),
    });

    // Control 2: Plausible near-match decoy & broken multi-file edit rejected by independent checkers
    let decoy_v = check_qsa_task(
        LongContextTaskKind::ExactLookupWithDecoys,
        "TOKEN-7F3A-DECOY",
    );
    let bad_edit_v = check_qsa_task(
        LongContextTaskKind::MultiFileEdit,
        "file:api.rs:pub fn limit() -> usize { 64 }\nfile:test.rs:assert_eq!(limit(), 32);",
    );
    negative_controls.push(NegativeControl {
        case_id: "plausible_decoy_and_broken_multifile_edit_rejected".into(),
        outcome: if decoy_v == OutcomeStatus::Invalid && bad_edit_v == OutcomeStatus::Invalid {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!(
            "decoy={}, bad_edit={}",
            decoy_v.as_str(),
            bad_edit_v.as_str()
        ),
    });

    // Control 3: Incomplete population / dropped dense-failed task rejected by offline rescorer
    let inc_res = rescore_qsa_population(5, &records[..4]);
    negative_controls.push(NegativeControl {
        case_id: "incomplete_population_rejected".into(),
        outcome: if inc_res.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: inc_res
            .err()
            .unwrap_or_else(|| "accepted dropped record".into()),
    });

    // Control 4: x86_64/sm70 prebuilt binary does NOT qualify ARM64/GB10 DGX Spark
    let (o4, d4) = validate_qsa_qualification_claim(&QsaQualificationClaim {
        binary_target_arch: "x86_64_sm70".into(),
        host_target_arch: "aarch64_sm121_gb10".into(),
        uses_flashtopk_23_for_block_qsa: false,
        substitutes_bf16_ppl_for_native_task: false,
        retrained_indexer_arm: false,
        backbone_quant_runtime_fixed: true,
    });
    negative_controls.push(NegativeControl {
        case_id: "x86_sm70_binary_does_not_qualify_spark_arm64_gb10".into(),
        outcome: o4,
        detail: d4,
    });

    // Control 5: #23 FlashTopK is NOT a drop-in dependency for 4-token block-level QSA
    let (o5, d5) = validate_qsa_qualification_claim(&QsaQualificationClaim {
        binary_target_arch: "aarch64_sm121_gb10".into(),
        host_target_arch: "aarch64_sm121_gb10".into(),
        uses_flashtopk_23_for_block_qsa: true,
        substitutes_bf16_ppl_for_native_task: false,
        retrained_indexer_arm: false,
        backbone_quant_runtime_fixed: true,
    });
    negative_controls.push(NegativeControl {
        case_id: "issue_23_flashtopk_not_dropin_for_block_qsa".into(),
        outcome: o5,
        detail: d5,
    });

    // Control 6: Omitted target KV or indexer buffer in memory accounting (QSA does NOT free target KV!)
    let qsa_total_kv_plus_idx = q_off.target_kv_bytes + q_off.indexer_buffer_bytes;
    negative_controls.push(NegativeControl {
        case_id: "qsa_retains_full_target_kv_plus_indexer_buffers".into(),
        outcome: if qsa_total_kv_plus_idx > d_off.target_kv_bytes {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!(
            "qsa_kv_plus_indexer={} bytes > dense_kv={} bytes (QSA does not reduce target KV capacity)",
            qsa_total_kv_plus_idx, d_off.target_kv_bytes
        ),
    });

    // Control 7: BF16 PPL / attention-mass capture cannot substitute for native quantized task accuracy
    let (o7, d7) = validate_qsa_qualification_claim(&QsaQualificationClaim {
        binary_target_arch: "aarch64_sm121_gb10".into(),
        host_target_arch: "aarch64_sm121_gb10".into(),
        uses_flashtopk_23_for_block_qsa: false,
        substitutes_bf16_ppl_for_native_task: true,
        retrained_indexer_arm: false,
        backbone_quant_runtime_fixed: true,
    });
    negative_controls.push(NegativeControl {
        case_id: "bf16_ppl_cannot_substitute_for_native_task_accuracy".into(),
        outcome: o7,
        detail: d7,
    });

    // Control 8: Retrained indexer with changed backbone/quant mislabeled as indexer-only ablation
    let (o8, d8) = validate_qsa_qualification_claim(&QsaQualificationClaim {
        binary_target_arch: "aarch64_sm121_gb10".into(),
        host_target_arch: "aarch64_sm121_gb10".into(),
        uses_flashtopk_23_for_block_qsa: false,
        substitutes_bf16_ppl_for_native_task: false,
        retrained_indexer_arm: true,
        backbone_quant_runtime_fixed: false,
    });
    negative_controls.push(NegativeControl {
        case_id: "retrained_indexer_with_changed_backbone_mislabeled".into(),
        outcome: o8,
        detail: d8,
    });

    Ok(Qwen38QsaReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical_nvidia_q8_0_qwen38_27b_qsa_vs_dense_llamacpp",
            "Device-free 4-token block QSA vs dense 2x2 reference, memory accounting, and 4 independent task checkers executed on CPU; physical 27B Q8_0 QSA serve unavailable locally",
        ),
        matrix_2x2: vec![d_off, d_on, q_off, q_on],
        population,
        zero_accepted_cost_undefined,
        negative_controls,
        decision: "INCONCLUSIVE_PHYSICAL_27B_Q8_0_PAIRED_SERVE_REQUIRED".into(),
    })
}

pub fn inputs_json(report: &Qwen38QsaReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.qwen38-qsa.inputs.v1\",\n",
            "  \"issue\": 31,\n",
            "  \"upstream\": {{\"engine_commit\":{},\"paper\":{},\"unverified_related_work_claim\":{}}},\n",
            "  \"block_size_tokens\": {},\n",
            "  \"context_matrix_targets\": [\"8k\",\"64k\",\"128k\",\"261k_bounded_optional\"],\n",
            "  \"flashtopk_issue_23_is_dropin\": false,\n",
            "  \"gsq_rco_attributed_to_original_artifact\": false,\n",
            "  \"execution_class\": {},\n",
            "  \"fixture_digest\": {}\n",
            "}}\n"
        ),
        quote(QSA_ENGINE_COMMIT),
        quote(QSA_PAPER),
        quote(UPSTREAM_UNVERIFIED_CLAIM),
        QSA_BLOCK_SIZE,
        quote(report.execution_class.as_str()),
        DigestRef::sha256(b"qwen38-qsa-2x2-4task-profile-v1").to_json(),
    )
}

pub fn results_json(report: &Qwen38QsaReport) -> String {
    let mat: Vec<String> = report
        .matrix_2x2
        .iter()
        .map(|m| {
            format!(
                concat!(
                    "{{\"mode\":{},\"mtp_draft2\":{},\"context_tokens\":{},\"selected_tokens\":{},",
                    "\"indexer_keys_scanned\":{},\"target_kv_bytes\":{},\"indexer_buffer_bytes\":{},",
                    "\"gdn_state_bytes\":{},\"mtp_state_bytes\":{},\"workspace_bytes\":{}}}"
                ),
                quote(m.mode.as_str()),
                m.mtp_draft2,
                m.context_tokens,
                m.selected_tokens,
                m.indexer_keys_scanned,
                m.target_kv_bytes,
                m.indexer_buffer_bytes,
                m.gdn_state_bytes,
                m.mtp_state_bytes,
                m.workspace_bytes,
            )
        })
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
    let p = &report.population;
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.qwen38-qsa.results.v1\",\n",
            "  \"issue\": 31,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"decision\": {},\n",
            "  \"zero_accepted_cost_undefined\": {},\n",
            "  \"physical_cold_prefill_ms\": {},\n",
            "  \"physical_decode_tok_s\": {},\n",
            "  \"matrix_2x2\": [{}],\n",
            "  \"population\": {{\"synthetic_contract_fixture\":true,\"planned_tasks\":{},\"dense_accepted\":{},\"qsa_accepted\":{},\"sparse_induced_losses\":{},\"both_failed_not_retained_capability\":{},\"dense_cost_per_accepted_ms\":{},\"qsa_cost_per_accepted_ms\":{}}},\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        quote(&report.decision),
        report.zero_accepted_cost_undefined,
        opt_f64(None),
        opt_f64(None),
        mat.join(","),
        p.planned_tasks,
        p.dense_accepted,
        p.qsa_accepted,
        p.sparse_induced_losses,
        p.both_failed_not_retained_capability,
        opt_f64(p.dense_cost_per_accepted_ms),
        opt_f64(p.qsa_cost_per_accepted_ms),
        neg.join(","),
    )
}

pub fn summary_markdown(report: &Qwen38QsaReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #31: Qwen3.8-27B QSA vs Matched Dense Control Contract"
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
    writeln!(out, "- Decision: `{}`", report.decision).unwrap();
    writeln!(
        out,
        "- Population (planned={}): dense_accepted={}, qsa_accepted={}, sparse_induced_losses={}, both_failed_not_retained_capability={}",
        report.population.planned_tasks,
        report.population.dense_accepted,
        report.population.qsa_accepted,
        report.population.sparse_induced_losses,
        report.population.both_failed_not_retained_capability,
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
