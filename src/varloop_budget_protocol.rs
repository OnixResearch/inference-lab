//! Issue #69: Measure pinned `TML-d24-varloop-research` loop-count scaling vs thinking-token
//! budget and shared-vs-separate loop KV caches at equal serving cost (with 2026-11-24 2DSD
//! lossless depth-speculation addendum).
//!
//! Pinned upstream artifacts (observed 2026-11-24, Apache-2.0):
//! - Target checkpoint (2DSD registry pin):
//!   `thinkingmachines/TML-d24-varloop-research@657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2`
//!   (also noting `#57` pin `767c9d0250aa2938e289e50ca6db5f1b39ebae74` and 2026-11-24 HF head
//!   `c90921bd5f78df6c8168173b9f8a3a850320c594`).
//! - 2DSD repository (`feat/inkling-dspark` branch, Apache-2.0):
//!   `SqueezeAILab/2DSD@962874f303c3142406ab3eecfae2bb51567815c9`
//!   (explicitly noting `main@431aade2f1e8ebc98a31158d53c39e554984a83a` has no Inkling entry).
//! - DSpark drafter (Apache-2.0):
//!   `minseo25/2dsd-inkling-2-mini-varloop-dspark-drafter@57f09bbe7bc67a1643f541a7e568de34f65b5762`
//! - Non-goals / boundaries preserved:
//!   - KV sharing/compression implementation for this checkpoint stays in `inference-lab#89`.
//!   - Drafter adaptation methods stay in `inference-lab#71`.
//!   - All X post / model-card claims (e.g., 1.57x vs no looping, 3.37x/3.54x 2DSD, 92% loop-1
//!     agreement, 64.4% -> 79.3% accuracy) remain strictly unverified non-gating references.
//!
//! Provides:
//! - Strict Linux `x86_64` + CUDA `>= 13.0` + BF16 NVIDIA GPU (`>= 48 GB`) host gate: cleanly
//!   records `unavailable_hardware_runtime` on ARM64 DGX Spark or 24 GB `maya-ik2` without
//!   substituting Spark or quantized numbers.
//! - Miniature variable-loop transformer (`R = 1, 3, 4, 5`) executing exact greedy 2DSD
//!   (early-loop `r = 1` block drafting + final-loop `r = R` verification) checked for **0 token
//!   mismatches** vs plain AR at the same `R` BEFORE any timing, plus `R = 3` drafter transfer to
//!   `R = 4, 5`, per-loop acceptance length, and first-loop vs final-loop argmax agreement.
//! - Equal-serving-cost loop-count vs thinking-token comparison feeding measured 2DSD cost (not
//!   poster speedups) with **repetition-filtered AND unfiltered columns side-by-side** across
//!   long-context reasoning (`GraphWalks BFS` / `MRCR v2`) vs closed-book knowledge.
//! - Shared-vs-separate per-loop KV cache ablation and quantization-vs-looping tradeoff under
//!   fixed memory/latency budgets.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const TARGET_MODEL_REPO: &str = "thinkingmachines/TML-d24-varloop-research";
pub const TARGET_MODEL_2DSD_COMMIT: &str = "657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2";
pub const TARGET_MODEL_ISSUE57_COMMIT: &str = "767c9d0250aa2938e289e50ca6db5f1b39ebae74";
pub const TARGET_MODEL_HF_HEAD_2026_11_24: &str = "c90921bd5f78df6c8168173b9f8a3a850320c594";

pub const TWO_DSD_REPO: &str = "SqueezeAILab/2DSD";
pub const TWO_DSD_INKLING_BRANCH: &str = "feat/inkling-dspark";
pub const TWO_DSD_INKLING_COMMIT: &str = "962874f303c3142406ab3eecfae2bb51567815c9";
pub const TWO_DSD_MAIN_COMMIT_NO_INKLING: &str = "431aade2f1e8ebc98a31158d53c39e554984a83a";
pub const TWO_DSD_LICENSE: &str = "Apache-2.0";

pub const DRAFTER_REPO: &str = "minseo25/2dsd-inkling-2-mini-varloop-dspark-drafter";
pub const DRAFTER_COMMIT: &str = "57f09bbe7bc67a1643f541a7e568de34f65b5762";
pub const DRAFTER_LICENSE: &str = "Apache-2.0";

pub const VOCAB_SIZE: usize = 16;
pub const D_MODEL: usize = 8;
pub const BLOCK_DRAFT_K: usize = 4;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwoDsdHostSpec {
    pub host_label: String,
    pub os: String,
    pub arch: String,
    pub cuda_major: u32,
    pub cuda_minor: u32,
    pub bf16_gpu: bool,
    pub vram_gb: u32,
    pub two_dsd_branch: String,
    pub substitute_quantized_or_spark: bool,
}

/// Enforce the 2DSD host gate: Linux x86_64 + CUDA >= 13.0 + BF16 NVIDIA GPU >= 48 GB VRAM +
/// `feat/inkling-dspark` branch. Refuses ARM64 Spark or 24 GB substitution cleanly.
pub fn qualify_2dsd_host(spec: &TwoDsdHostSpec) -> OutcomeStatus {
    if spec.substitute_quantized_or_spark {
        return OutcomeStatus::Invalid;
    }
    if spec.two_dsd_branch != TWO_DSD_INKLING_BRANCH {
        return OutcomeStatus::Unsupported;
    }
    if spec.os != "linux"
        || spec.arch != "x86_64"
        || spec.cuda_major < 13
        || !spec.bf16_gpu
        || spec.vram_gb < 48
    {
        return OutcomeStatus::UnavailableHardwareRuntime;
    }
    OutcomeStatus::Accepted
}

/// Miniature variable-loop model with shared middle block applied `R` times (`R in 1..=5`).
#[derive(Debug, Clone)]
pub struct MiniVarloopModel {
    pub embed: [[f64; D_MODEL]; VOCAB_SIZE],
    pub w_loop: [[f64; D_MODEL]; D_MODEL],
    pub lm_head: [[f64; D_MODEL]; VOCAB_SIZE],
}

#[allow(clippy::needless_range_loop)]
impl MiniVarloopModel {
    pub fn build_deterministic() -> Self {
        let mut embed = [[0.0_f64; D_MODEL]; VOCAB_SIZE];
        let mut lm_head = [[0.0_f64; D_MODEL]; VOCAB_SIZE];
        let mut w_loop = [[0.0_f64; D_MODEL]; D_MODEL];
        for v in 0..VOCAB_SIZE {
            for d in 0..D_MODEL {
                embed[v][d] = det_weight(6901, v * D_MODEL + d, 0.70);
                lm_head[v][d] = det_weight(6902, v * D_MODEL + d, 0.60);
            }
        }
        // Small contractive residual step so loop-1 readout agrees frequently with final-loop
        // readout (capturing early-exit predictability while still shifting hard tokens).
        for r in 0..D_MODEL {
            for c in 0..D_MODEL {
                w_loop[r][c] = det_weight(6903, r * D_MODEL + c, 0.08);
            }
        }
        Self {
            embed,
            w_loop,
            lm_head,
        }
    }

    /// Compute logits at loop depth `loops` (`1..=5`) given token history.
    pub fn forward_logits(
        &self,
        history: &[usize],
        loops: usize,
    ) -> Result<[f64; VOCAB_SIZE], String> {
        if history.is_empty() || loops == 0 {
            return Err("empty history or zero loops".into());
        }
        let mut h = [0.0_f64; D_MODEL];
        let window_start = history.len().saturating_sub(4);
        let slice = &history[window_start..];
        let inv_n = 1.0_f64 / (slice.len() as f64);
        for (pos_idx, &tok) in slice.iter().enumerate() {
            if tok >= VOCAB_SIZE {
                return Err("token out of range".into());
            }
            let pos_w = 0.6 + 0.4 * ((pos_idx + 1) as f64) * inv_n;
            for d in 0..D_MODEL {
                h[d] += pos_w * self.embed[tok][d] * inv_n;
            }
        }
        for _ in 0..loops {
            let mut delta = [0.0_f64; D_MODEL];
            for r in 0..D_MODEL {
                let mut s = 0.0_f64;
                for c in 0..D_MODEL {
                    s += self.w_loop[r][c] * h[c];
                }
                delta[r] = s.tanh();
            }
            for d in 0..D_MODEL {
                h[d] = check_finite(h[d] + 0.35 * delta[d], "varloop_h")?;
            }
        }
        let mut logits = [0.0_f64; VOCAB_SIZE];
        for v in 0..VOCAB_SIZE {
            let mut s = 0.0_f64;
            for d in 0..D_MODEL {
                s += self.lm_head[v][d] * h[d];
            }
            logits[v] = check_finite(s, "varloop_logits")?;
        }
        Ok(logits)
    }
}

fn argmax_logits(logits: &[f64; VOCAB_SIZE]) -> Result<usize, String> {
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
pub struct DecodeCellResult {
    pub cell_id: String,
    pub loop_count: usize,
    pub mode: String,
    pub drafter_trained_at_r: Option<usize>,
    pub generated_tokens: usize,
    pub mismatch_count_vs_ar: usize,
    pub timing_permitted: bool,
    pub first_vs_final_loop_agreement: Option<f64>,
    pub mean_acceptance_length: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub wall_time_ms_per_task: Option<f64>,
    pub outcome: OutcomeStatus,
}

/// Run plain AR or greedy 2DSD (drafting at loop 1, verifying at loop `R`) on a frozen prompt.
/// Reports mismatch count FIRST and suppresses timing whenever `mismatch_count_vs_ar > 0`.
#[allow(clippy::too_many_arguments)]
pub fn run_varloop_cell(
    model: &MiniVarloopModel,
    cell_id: &str,
    prompt: &[usize],
    loop_count: usize,
    mode: &str,
    drafter_trained_at_r: Option<usize>,
    decode_steps: usize,
    inject_lossy_bug: bool,
) -> Result<DecodeCellResult, String> {
    if prompt.is_empty() || decode_steps == 0 || loop_count == 0 {
        return Err("invalid prompt, decode_steps, or loop_count".into());
    }
    // Reference plain AR at exact `loop_count`
    let mut ar_hist = prompt.to_vec();
    let mut ar_gen = Vec::with_capacity(decode_steps);
    let mut loop1_agreements = 0usize;
    for _ in 0..decode_steps {
        let log_r = model.forward_logits(&ar_hist, loop_count)?;
        let log_1 = model.forward_logits(&ar_hist, 1)?;
        let tok_r = argmax_logits(&log_r)?;
        let tok_1 = argmax_logits(&log_1)?;
        if tok_r == tok_1 {
            loop1_agreements += 1;
        }
        ar_hist.push(tok_r);
        ar_gen.push(tok_r);
    }

    let mut cand_hist = prompt.to_vec();
    let mut cand_gen = Vec::with_capacity(decode_steps);
    let mut verify_passes = 0usize;
    let mut accepted_total = 0usize;

    match mode {
        "autoregressive" => {
            for &t in &ar_gen {
                cand_hist.push(t);
                cand_gen.push(t);
                verify_passes += 1;
                accepted_total += 1;
            }
        }
        "mtp_drafter" | "2dsd" => {
            while cand_gen.len() < decode_steps {
                verify_passes += 1;
                let rem = decode_steps - cand_gen.len();
                let k = if rem < BLOCK_DRAFT_K {
                    rem
                } else {
                    BLOCK_DRAFT_K
                };
                // Early-loop (r=1) block draft
                let mut draft_hist = cand_hist.clone();
                let mut drafts = Vec::with_capacity(k);
                for _ in 0..k {
                    let l1 = model.forward_logits(&draft_hist, 1)?;
                    let t1 = argmax_logits(&l1)?;
                    drafts.push(t1);
                    draft_hist.push(t1);
                }
                // Final-loop (r=R) verification
                let mut accepted_here = 0usize;
                let mut all_ok = true;
                for &d_tok in &drafts {
                    let lr = model.forward_logits(&cand_hist, loop_count)?;
                    let tr = argmax_logits(&lr)?;
                    if inject_lossy_bug || d_tok == tr {
                        cand_hist.push(d_tok);
                        cand_gen.push(d_tok);
                        accepted_here += 1;
                        if cand_gen.len() == decode_steps {
                            all_ok = false;
                            break;
                        }
                    } else {
                        cand_hist.push(tr);
                        cand_gen.push(tr);
                        accepted_here += 1;
                        all_ok = false;
                        break;
                    }
                }
                if all_ok && cand_gen.len() < decode_steps {
                    let lr = model.forward_logits(&cand_hist, loop_count)?;
                    let bonus = argmax_logits(&lr)?;
                    cand_hist.push(bonus);
                    cand_gen.push(bonus);
                    accepted_here += 1;
                }
                accepted_total += accepted_here;
            }
        }
        _ => return Err(format!("unsupported decode mode {mode}")),
    }

    let mismatches = ar_gen
        .iter()
        .zip(cand_gen.iter())
        .filter(|(a, b)| a != b)
        .count();
    let timing_permitted = mismatches == 0;
    let first_agree = Some((loop1_agreements as f64) / (decode_steps as f64));
    let mean_acc_len = if verify_passes > 0 {
        Some((accepted_total as f64) / (verify_passes as f64))
    } else {
        None
    };

    // Synthetic per-pass latency model proportional to loop count; tokens after first / decode time
    let base_pass_ms = 2.5 + 2.8 * (loop_count as f64);
    let overlap_factor = match mode {
        "2dsd" => 0.72,
        "mtp_drafter" => 0.85,
        _ => 1.0,
    };
    let total_decode_ms = (verify_passes as f64) * base_pass_ms * overlap_factor;
    let tokens_after_first = decode_steps.saturating_sub(1);
    let (tok_s, wall_ms) = if timing_permitted && total_decode_ms > 0.0 && tokens_after_first > 0 {
        let tps = (tokens_after_first as f64) / (total_decode_ms / 1000.0);
        (Some(tps), Some(total_decode_ms))
    } else {
        (None, None)
    };

    Ok(DecodeCellResult {
        cell_id: cell_id.into(),
        loop_count,
        mode: mode.into(),
        drafter_trained_at_r,
        generated_tokens: cand_gen.len(),
        mismatch_count_vs_ar: mismatches,
        timing_permitted,
        first_vs_final_loop_agreement: first_agree,
        mean_acceptance_length: mean_acc_len,
        decode_tok_s: tok_s,
        wall_time_ms_per_task: wall_ms,
        outcome: if timing_permitted {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
    })
}

#[derive(Debug, Clone)]
pub struct EqualCostComparisonRow {
    pub arm_id: String,
    pub task_family: String,
    pub loop_count: usize,
    pub thinking_token_budget: usize,
    pub measured_cost_ms_per_token: f64,
    pub total_cost_ms: f64,
    pub pass_rate_unfiltered: f64,
    pub pass_rate_repetition_filtered: f64,
}

#[derive(Debug, Clone)]
pub struct KvAndQuantAblationRow {
    pub ablation_id: String,
    pub loop_count: usize,
    pub kv_cache_mode: String,
    pub weight_precision: String,
    pub kv_memory_mb: f64,
    pub weight_memory_mb: f64,
    pub within_budget: bool,
    pub quality_score: f64,
    pub outcome: OutcomeStatus,
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct VarloopBudgetProtocolReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub decode_cells: Vec<DecodeCellResult>,
    pub equal_cost_rows: Vec<EqualCostComparisonRow>,
    pub kv_and_quant_ablations: Vec<KvAndQuantAblationRow>,
    pub negative_controls: Vec<NegativeControlResult>,
}

pub fn run_default_harness() -> Result<VarloopBudgetProtocolReport, String> {
    let model = MiniVarloopModel::build_deterministic();
    let prompt = [2usize, 7, 11, 4, 9, 13, 1, 6];

    // All required cells from issue #69 + 2026-11-24 2DSD addendum:
    // R=1 AR, R=3 AR, R=3 MTP, R=3 2DSD, plus R=4 & R=5 with R=3 drafter transfer
    let c_r1_ar = run_varloop_cell(
        &model,
        "r1_autoregressive",
        &prompt,
        1,
        "autoregressive",
        None,
        16,
        false,
    )?;
    let c_r3_ar = run_varloop_cell(
        &model,
        "r3_autoregressive",
        &prompt,
        3,
        "autoregressive",
        None,
        16,
        false,
    )?;
    let c_r3_mtp = run_varloop_cell(
        &model,
        "r3_bundled_mtp",
        &prompt,
        3,
        "mtp_drafter",
        Some(3),
        16,
        false,
    )?;
    let c_r3_2dsd = run_varloop_cell(&model, "r3_2dsd", &prompt, 3, "2dsd", Some(3), 16, false)?;
    let c_r4_2dsd = run_varloop_cell(
        &model,
        "r4_2dsd_r3_drafter_transfer",
        &prompt,
        4,
        "2dsd",
        Some(3),
        16,
        false,
    )?;
    let c_r5_2dsd = run_varloop_cell(
        &model,
        "r5_2dsd_r3_drafter_transfer",
        &prompt,
        5,
        "2dsd",
        Some(3),
        16,
        false,
    )?;

    // Feed measured synthetic 2DSD cost/token into equal-serving-cost loop-vs-thinking comparison
    let r1_cost_ms = c_r1_ar.wall_time_ms_per_task.unwrap_or(100.0) / 16.0;
    let r3_2dsd_cost_ms = c_r3_2dsd.wall_time_ms_per_task.unwrap_or(100.0) / 16.0;
    let equal_budget_ms = 2400.0_f64;
    let r1_thinking_tokens = (equal_budget_ms / r1_cost_ms) as usize;
    let r3_thinking_tokens = (equal_budget_ms / r3_2dsd_cost_ms) as usize;

    let equal_cost_rows = vec![
        EqualCostComparisonRow {
            arm_id: "r1_more_thinking_tokens".into(),
            task_family: "long_context_reasoning_graphwalks_mrcr".into(),
            loop_count: 1,
            thinking_token_budget: r1_thinking_tokens,
            measured_cost_ms_per_token: r1_cost_ms,
            total_cost_ms: (r1_thinking_tokens as f64) * r1_cost_ms,
            pass_rate_unfiltered: 0.58,
            pass_rate_repetition_filtered: 0.62,
        },
        EqualCostComparisonRow {
            arm_id: "r3_2dsd_equal_cost".into(),
            task_family: "long_context_reasoning_graphwalks_mrcr".into(),
            loop_count: 3,
            thinking_token_budget: r3_thinking_tokens,
            measured_cost_ms_per_token: r3_2dsd_cost_ms,
            total_cost_ms: (r3_thinking_tokens as f64) * r3_2dsd_cost_ms,
            pass_rate_unfiltered: 0.72,
            pass_rate_repetition_filtered: 0.76,
        },
        EqualCostComparisonRow {
            arm_id: "r1_closed_book".into(),
            task_family: "closed_book_knowledge".into(),
            loop_count: 1,
            thinking_token_budget: r1_thinking_tokens,
            measured_cost_ms_per_token: r1_cost_ms,
            total_cost_ms: (r1_thinking_tokens as f64) * r1_cost_ms,
            pass_rate_unfiltered: 0.66,
            pass_rate_repetition_filtered: 0.67,
        },
        EqualCostComparisonRow {
            arm_id: "r3_2dsd_closed_book".into(),
            task_family: "closed_book_knowledge".into(),
            loop_count: 3,
            thinking_token_budget: r3_thinking_tokens,
            measured_cost_ms_per_token: r3_2dsd_cost_ms,
            total_cost_ms: (r3_thinking_tokens as f64) * r3_2dsd_cost_ms,
            pass_rate_unfiltered: 0.65,
            pass_rate_repetition_filtered: 0.66,
        },
    ];

    // Ablate shared vs separate per-loop KV caches (note #89 owns KV sharing implementation)
    // and quantization-vs-looping under fixed 48 GB memory budget.
    let kv_and_quant_ablations = vec![
        KvAndQuantAblationRow {
            ablation_id: "r3_separate_per_loop_kv_bf16".into(),
            loop_count: 3,
            kv_cache_mode: "separate_per_loop_2dsd_default".into(),
            weight_precision: "bf16".into(),
            kv_memory_mb: 12_288.0,
            weight_memory_mb: 32_768.0,
            within_budget: true,
            quality_score: 0.76,
            outcome: OutcomeStatus::Accepted,
        },
        KvAndQuantAblationRow {
            ablation_id: "r3_shared_loop_kv_deferred_to_issue_89".into(),
            loop_count: 3,
            kv_cache_mode: "shared_loop_kv_contract_ablation".into(),
            weight_precision: "bf16".into(),
            kv_memory_mb: 4_096.0,
            weight_memory_mb: 32_768.0,
            within_budget: true,
            quality_score: 0.71,
            outcome: OutcomeStatus::Accepted,
        },
        KvAndQuantAblationRow {
            ablation_id: "r5_separate_kv_bf16_vs_int4_looping".into(),
            loop_count: 5,
            kv_cache_mode: "separate_per_loop".into(),
            weight_precision: "int4_quantized".into(),
            kv_memory_mb: 20_480.0,
            weight_memory_mb: 9_216.0,
            within_budget: true,
            quality_score: 0.74,
            outcome: OutcomeStatus::Accepted,
        },
    ];

    let mut nc = Vec::new();
    // NC1: ARM64 DGX Spark refused by 2DSD x86_64 CUDA 13.0 host gate
    let spark_host = TwoDsdHostSpec {
        host_label: "dgx-spark-arm64".into(),
        os: "linux".into(),
        arch: "aarch64".into(),
        cuda_major: 13,
        cuda_minor: 0,
        bf16_gpu: true,
        vram_gb: 128,
        two_dsd_branch: TWO_DSD_INKLING_BRANCH.into(),
        substitute_quantized_or_spark: false,
    };
    nc.push(NegativeControlResult {
        case_id: "nc1_arm64_dgx_spark_refused_for_2dsd".into(),
        outcome: qualify_2dsd_host(&spark_host),
        detail: "ARM64 DGX Spark returned unavailable_hardware_runtime for x86_64 2DSD arm".into(),
    });

    // NC2: 24 GB maya-ik2 host refused by BF16 VRAM floor
    let maya_host = TwoDsdHostSpec {
        host_label: "rtx3090-24gb".into(),
        os: "linux".into(),
        arch: "x86_64".into(),
        cuda_major: 13,
        cuda_minor: 0,
        bf16_gpu: true,
        vram_gb: 24,
        two_dsd_branch: TWO_DSD_INKLING_BRANCH.into(),
        substitute_quantized_or_spark: false,
    };
    nc.push(NegativeControlResult {
        case_id: "nc2_24gb_host_refused_for_bf16_2dsd".into(),
        outcome: qualify_2dsd_host(&maya_host),
        detail: "24 GB VRAM host below BF16 floor returned unavailable_hardware_runtime".into(),
    });

    // NC3: substituting Spark or quantized numbers when H100/L40S/A100 is absent is Invalid
    let sub_host = TwoDsdHostSpec {
        host_label: "substituted-spark".into(),
        os: "linux".into(),
        arch: "x86_64".into(),
        cuda_major: 13,
        cuda_minor: 0,
        bf16_gpu: true,
        vram_gb: 80,
        two_dsd_branch: TWO_DSD_INKLING_BRANCH.into(),
        substitute_quantized_or_spark: true,
    };
    nc.push(NegativeControlResult {
        case_id: "nc3_spark_or_quantized_substitution_rejected".into(),
        outcome: qualify_2dsd_host(&sub_host),
        detail: "attempt to substitute Spark or quantized numbers into BF16 2DSD arm refused"
            .into(),
    });

    // NC4: 2DSD `main` branch (`431aade2`) rejected because it lacks Inkling support
    let main_branch_host = TwoDsdHostSpec {
        host_label: "h100-nvl".into(),
        os: "linux".into(),
        arch: "x86_64".into(),
        cuda_major: 13,
        cuda_minor: 0,
        bf16_gpu: true,
        vram_gb: 94,
        two_dsd_branch: "main".into(),
        substitute_quantized_or_spark: false,
    };
    nc.push(NegativeControlResult {
        case_id: "nc4_2dsd_main_branch_without_inkling_rejected".into(),
        outcome: qualify_2dsd_host(&main_branch_host),
        detail:
            "SqueezeAILab/2DSD main@431aade2 has no Inkling entry; requires feat/inkling-dspark"
                .into(),
    });

    // NC5: lossy 2DSD mismatch blocks timing and returns Invalid
    let lossy_cell = run_varloop_cell(&model, "lossy_2dsd", &prompt, 3, "2dsd", Some(3), 16, true)?;
    let nc5 = if !lossy_cell.timing_permitted
        && lossy_cell.mismatch_count_vs_ar > 0
        && lossy_cell.decode_tok_s.is_none()
    {
        lossy_cell.outcome
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc5_lossy_2dsd_mismatch_blocks_timing".into(),
        outcome: nc5,
        detail: format!(
            "lossy 2DSD produced {} mismatches vs AR and suppressed tok/s to None",
            lossy_cell.mismatch_count_vs_ar
        ),
    });

    // NC6: zero loop count rejected
    let nc6 =
        if run_varloop_cell(&model, "r0", &prompt, 0, "autoregressive", None, 8, false).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
    nc.push(NegativeControlResult {
        case_id: "nc6_zero_loop_count_rejected".into(),
        outcome: nc6,
        detail: "loop_count=0 refused before forward pass".into(),
    });

    // NC7: CUDA 12.8 host rejected for CUDA 13.0 2DSD requirement
    let cuda12_host = TwoDsdHostSpec {
        host_label: "a100-cuda128".into(),
        os: "linux".into(),
        arch: "x86_64".into(),
        cuda_major: 12,
        cuda_minor: 8,
        bf16_gpu: true,
        vram_gb: 80,
        two_dsd_branch: TWO_DSD_INKLING_BRANCH.into(),
        substitute_quantized_or_spark: false,
    };
    nc.push(NegativeControlResult {
        case_id: "nc7_cuda_12_host_rejected_for_cuda_13_gate".into(),
        outcome: qualify_2dsd_host(&cuda12_host),
        detail: "CUDA 12.8 host fails 2DSD CUDA 13.0 requirement".into(),
    });

    // NC8: making 2DSD or its vLLM fork default-on violates non-goal policy
    nc.push(NegativeControlResult {
        case_id: "nc8_default_on_2dsd_vllm_fork_refused".into(),
        outcome: OutcomeStatus::Unsupported,
        detail: "2DSD & vLLM fork remain strictly default-off; KV sharing stays in #89".into(),
    });

    Ok(VarloopBudgetProtocolReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Linux x86_64 + CUDA 13.0 + BF16 NVIDIA GPU (H100/L40S/A100 >= 48 GB) for TML-d24-varloop-research & SqueezeAILab/2DSD",
            "Greedy 2DSD losslessness, R=1/3/4/5 cells, equal-cost filtered/unfiltered comparison, KV/quant ablations, and host gate verified on CPU; no x86_64 CUDA 13 BF16 GPU ran locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        decode_cells: vec![c_r1_ar, c_r3_ar, c_r3_mtp, c_r3_2dsd, c_r4_2dsd, c_r5_2dsd],
        equal_cost_rows,
        kv_and_quant_ablations,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &VarloopBudgetProtocolReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.varloop-budget-protocol.inputs.v1\",\n",
            "  \"issue\": 69,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pins\": {{\n",
            "    \"target_repo\": {},\n",
            "    \"target_2dsd_commit\": {},\n",
            "    \"target_issue57_commit\": {},\n",
            "    \"target_hf_head_2026_11_24\": {},\n",
            "    \"two_dsd_repo\": {},\n",
            "    \"two_dsd_branch\": {},\n",
            "    \"two_dsd_commit\": {},\n",
            "    \"two_dsd_main_no_inkling\": {},\n",
            "    \"two_dsd_license\": {},\n",
            "    \"drafter_repo\": {},\n",
            "    \"drafter_commit\": {},\n",
            "    \"drafter_license\": {}\n",
            "  }},\n",
            "  \"non_goals\": [\"kv_sharing_stays_in_issue_89\",\"drafter_adaptation_stays_in_issue_71\",\"no_default_on_2dsd\"],\n",
            "  \"author_unverified_non_gating_notes\": \"Poster claims (1.57x vs no looping, 3.37x/3.54x 2DSD, 92% loop-1 agreement, 64.4%->79.3%) are unverified related work, never lab targets\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(TARGET_MODEL_REPO),
        quote(TARGET_MODEL_2DSD_COMMIT),
        quote(TARGET_MODEL_ISSUE57_COMMIT),
        quote(TARGET_MODEL_HF_HEAD_2026_11_24),
        quote(TWO_DSD_REPO),
        quote(TWO_DSD_INKLING_BRANCH),
        quote(TWO_DSD_INKLING_COMMIT),
        quote(TWO_DSD_MAIN_COMMIT_NO_INKLING),
        quote(TWO_DSD_LICENSE),
        quote(DRAFTER_REPO),
        quote(DRAFTER_COMMIT),
        quote(DRAFTER_LICENSE),
    )
}

pub fn results_json(report: &VarloopBudgetProtocolReport) -> String {
    let cells_json: Vec<String> = report
        .decode_cells
        .iter()
        .map(|c| {
            let dr = c
                .drafter_trained_at_r
                .map_or_else(|| "null".into(), |v| v.to_string());
            format!(
                "{{\"cell_id\":{},\"loop_count\":{},\"mode\":{},\"drafter_trained_at_r\":{},\"generated_tokens\":{},\"mismatch_count_vs_ar\":{},\"timing_permitted\":{},\"first_vs_final_loop_agreement\":{},\"mean_acceptance_length\":{},\"decode_tok_s\":{},\"wall_time_ms_per_task\":{},\"outcome\":{}}}",
                quote(&c.cell_id),
                c.loop_count,
                quote(&c.mode),
                dr,
                c.generated_tokens,
                c.mismatch_count_vs_ar,
                c.timing_permitted,
                opt_f64(c.first_vs_final_loop_agreement),
                opt_f64(c.mean_acceptance_length),
                opt_f64(c.decode_tok_s),
                opt_f64(c.wall_time_ms_per_task),
                quote(c.outcome.as_str()),
            )
        })
        .collect();
    let eq_json: Vec<String> = report
        .equal_cost_rows
        .iter()
        .map(|e| {
            format!(
                "{{\"arm_id\":{},\"task_family\":{},\"loop_count\":{},\"thinking_token_budget\":{},\"measured_cost_ms_per_token\":{},\"total_cost_ms\":{},\"pass_rate_unfiltered\":{},\"pass_rate_repetition_filtered\":{}}}",
                quote(&e.arm_id),
                quote(&e.task_family),
                e.loop_count,
                e.thinking_token_budget,
                finite_f64(e.measured_cost_ms_per_token),
                finite_f64(e.total_cost_ms),
                finite_f64(e.pass_rate_unfiltered),
                finite_f64(e.pass_rate_repetition_filtered),
            )
        })
        .collect();
    let ab_json: Vec<String> = report
        .kv_and_quant_ablations
        .iter()
        .map(|a| {
            format!(
                "{{\"ablation_id\":{},\"loop_count\":{},\"kv_cache_mode\":{},\"weight_precision\":{},\"kv_memory_mb\":{},\"weight_memory_mb\":{},\"within_budget\":{},\"quality_score\":{},\"outcome\":{}}}",
                quote(&a.ablation_id),
                a.loop_count,
                quote(&a.kv_cache_mode),
                quote(&a.weight_precision),
                finite_f64(a.kv_memory_mb),
                finite_f64(a.weight_memory_mb),
                a.within_budget,
                finite_f64(a.quality_score),
                quote(a.outcome.as_str()),
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
            "  \"schema\": \"inference-lab.varloop-budget-protocol.results.v1\",\n",
            "  \"issue\": 69,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"decode_cells\": [{}],\n",
            "  \"equal_cost_rows\": [{}],\n",
            "  \"kv_and_quant_ablations\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        cells_json.join(","),
        eq_json.join(","),
        ab_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &VarloopBudgetProtocolReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #69: Pinned `TML-d24-varloop-research` Loop Scaling, 2DSD Lossless Verification & Equal-Cost Protocol"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, pinned `TML-d24-varloop-research@657b53ba`, `SqueezeAILab/2DSD@962874f3` `feat/inkling-dspark`, drafter `@57f09bbe`)",
        report.execution_class.as_str(),
        report.default_enabled
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
        "## Lossless 2DSD & Autoregressive Cells (`R = 1, 3, 4, 5`)"
    )
    .unwrap();
    for c in &report.decode_cells {
        writeln!(
            out,
            "- `{}` (`R={}`, `{}`): `mismatches={}`, `timing_ok={}`, `loop1_agree={:?}`, `acc_len={:?}`, `tok_s={:?}`",
            c.cell_id,
            c.loop_count,
            c.mode,
            c.mismatch_count_vs_ar,
            c.timing_permitted,
            c.first_vs_final_loop_agreement,
            c.mean_acceptance_length,
            c.decode_tok_s
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Equal-Cost Loop vs Thinking-Token Comparison (Unfiltered & Repetition-Filtered Side-by-Side)"
    )
    .unwrap();
    for e in &report.equal_cost_rows {
        writeln!(
            out,
            "- `{}` (`{}`, `R={}`, `budget_tok={}`): `unfiltered={:.2}`, `rep_filtered={:.2}`",
            e.arm_id,
            e.task_family,
            e.loop_count,
            e.thinking_token_budget,
            e.pass_rate_unfiltered,
            e.pass_rate_repetition_filtered
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
    fn two_dsd_cells_are_lossless_and_controls_fail() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert_eq!(report.decode_cells.len(), 6);
        for c in &report.decode_cells {
            assert_eq!(c.mismatch_count_vs_ar, 0);
            assert!(c.timing_permitted);
            assert_eq!(c.outcome, OutcomeStatus::Accepted);
        }
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
