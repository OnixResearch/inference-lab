//! Issue #18: Evaluate Qwen3.8-27B quantization and serving configurations by accepted tasks
//! and total cost (including MACaQ, QSA, Tenstorrent P150x2, Monoglot vocabulary pruning,
//! ternary `ttf`, QDA PTQ + KL rank correlation, and `vegaml` scope separation).
//!
//! Pure deterministic core integrating `#7` (`serving_bench::WorkloadIdentity`) and `#3`
//! (`drafting::check_greedy_token_parity`) without adding a second HTTP/SSE client or scheduler.

use std::fmt::Write as FmtWrite;

use crate::drafting::{check_greedy_token_parity, GreedyParityResult};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};
use crate::serving_bench::{default_workload_identity, WorkloadIdentity};

pub const UNSLOTH_GGUF_COMMIT: &str = "4af51b3542f775b0a8e4b310bfb3f17e01c306c9";
pub const TT_ONIX_COMMIT: &str = "5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a";
pub const MACAQ_COMMIT: &str = "3810efca0b9e819c8eb51addd93c3c87225b7476";
pub const TT_STACK_COMMIT: &str = "e55c6527eb2878c1f008b0fff14dd7f411a75b56";
pub const QDA_REPO_COMMIT: &str = "dc39a02bfd87bb2790e34001708d54ce9131e076";
pub const QDA_W4A16_COMMIT: &str = "3615263040d9aa156180f746c0909c01b79986d2";
pub const QDA_Q4_GGUF_COMMIT: &str = "2038e5f1502bdd09e15a4015491ff95b8b523e37";
pub const QDA_Q6_GGUF_COMMIT: &str = "fde6ba756b9218620518316b241a2b1a98418347";
pub const VEGAML_COMMIT: &str = "5e7ffad546b006ccc316bb46c02df40e9200713d";

/// Calculated planning values from Qwen3.8-27B `config.json` (16 full-attention layers,
/// 4 KV heads, head_dim 256, BF16/FP16 = 2 bytes). Explicitly NOT an observed runtime allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPlanningBudget {
    pub full_attention_layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub elem_bytes: usize,
    pub bytes_per_token_per_sequence: usize,
    pub bytes_at_32768_tokens: u64,
    pub bytes_at_262144_tokens: u64,
    pub is_calculated_planning_not_observed_allocation: bool,
}

pub fn qwen38_27b_kv_planning_budget() -> KvPlanningBudget {
    let layers = 16usize;
    let kv_heads = 4usize;
    let head_dim = 256usize;
    let elem_bytes = 2usize; // BF16/FP16
                             // K + V => factor 2.
    let per_tok = layers * 2 * kv_heads * head_dim * elem_bytes; // 65536 B = 64 KiB
    KvPlanningBudget {
        full_attention_layers: layers,
        kv_heads,
        head_dim,
        elem_bytes,
        bytes_per_token_per_sequence: per_tok,
        bytes_at_32768_tokens: (per_tok as u64) * 32_768,
        bytes_at_262144_tokens: (per_tok as u64) * 262_144,
        is_calculated_planning_not_observed_allocation: true,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptCategory {
    Accepted,
    Incorrect,
    Malformed,
    Unsupported,
    Rejected,
    Cancelled,
    TimedOut,
    Truncated,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RetainedTaskAttempt {
    pub task_id: String,
    pub category: AttemptCategory,
    pub wall_ms: f64,
    pub resource_cost_units: f64,
    pub prompt_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OfflineArmScore {
    pub arm_id: String,
    pub comparison_mode: String,
    pub planned_attempts: usize,
    pub accepted: usize,
    pub incorrect: usize,
    pub malformed: usize,
    pub unsupported: usize,
    pub rejected: usize,
    pub cancelled: usize,
    pub timed_out: usize,
    pub truncated: usize,
    pub incomplete: usize,
    pub acceptance_rate: Option<f64>,
    pub total_wall_ms: f64,
    pub total_resource_cost: f64,
    /// `None` (undefined) whenever `accepted == 0`; never fabricated as `0.0` or `inf`.
    pub cost_per_accepted_task: Option<f64>,
    pub denominator_exact: bool,
}

/// Offline rescoring from retained task attempts without making any model call.
pub fn score_retained_attempts(
    arm_id: &str,
    comparison_mode: &str,
    planned_attempts: usize,
    attempts: &[RetainedTaskAttempt],
) -> Result<OfflineArmScore, String> {
    let mut accepted = 0usize;
    let mut incorrect = 0usize;
    let mut malformed = 0usize;
    let mut unsupported = 0usize;
    let mut rejected = 0usize;
    let mut cancelled = 0usize;
    let mut timed_out = 0usize;
    let mut truncated = 0usize;
    let mut incomplete = 0usize;
    let mut total_wall_ms = 0.0_f64;
    let mut total_cost = 0.0_f64;

    for a in attempts {
        if !a.wall_ms.is_finite()
            || a.wall_ms < 0.0
            || !a.resource_cost_units.is_finite()
            || a.resource_cost_units < 0.0
        {
            return Err("non-finite or negative attempt timing/cost".into());
        }
        total_wall_ms += a.wall_ms;
        total_cost += a.resource_cost_units;
        match a.category {
            AttemptCategory::Accepted => accepted += 1,
            AttemptCategory::Incorrect => incorrect += 1,
            AttemptCategory::Malformed => malformed += 1,
            AttemptCategory::Unsupported => unsupported += 1,
            AttemptCategory::Rejected => rejected += 1,
            AttemptCategory::Cancelled => cancelled += 1,
            AttemptCategory::TimedOut => timed_out += 1,
            AttemptCategory::Truncated => truncated += 1,
            AttemptCategory::Incomplete => incomplete += 1,
        }
    }

    let counted = accepted
        + incorrect
        + malformed
        + unsupported
        + rejected
        + cancelled
        + timed_out
        + truncated
        + incomplete;
    let denominator_exact = planned_attempts > 0 && counted == planned_attempts;
    let acceptance_rate = if denominator_exact {
        Some((accepted as f64) / (planned_attempts as f64))
    } else {
        None
    };
    let cost_per_accepted_task = if accepted > 0 {
        Some(total_cost / (accepted as f64))
    } else {
        None
    };

    Ok(OfflineArmScore {
        arm_id: arm_id.into(),
        comparison_mode: comparison_mode.into(),
        planned_attempts,
        accepted,
        incorrect,
        malformed,
        unsupported,
        rejected,
        cancelled,
        timed_out,
        truncated,
        incomplete,
        acceptance_rate,
        total_wall_ms,
        total_resource_cost: total_cost,
        cost_per_accepted_task,
        denominator_exact,
    })
}

/// Pure-Rust average-rank Spearman correlation coefficient `rho` for QDA KL-vs-accepted-task
/// evaluation (`neg_kl` higher is better vs `accepted_rate` higher is better).
pub fn spearman_rank_correlation(xs: &[f64], ys: &[f64]) -> Result<f64, String> {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return Err("spearman requires equal-length slices with len >= 2".into());
    }
    if xs.iter().any(|x| !x.is_finite()) || ys.iter().any(|y| !y.is_finite()) {
        return Err("spearman requires finite inputs".into());
    }
    let rank_vec = |vals: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..vals.len()).collect();
        idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        let mut ranks = vec![0.0_f64; vals.len()];
        let mut i = 0;
        while i < vals.len() {
            let mut j = i + 1;
            while j < vals.len() && (vals[idx[j]] - vals[idx[i]]).abs() == 0.0 {
                j += 1;
            }
            let avg_rank = ((i + j - 1) as f64) * 0.5 + 1.0;
            for &pos in &idx[i..j] {
                ranks[pos] = avg_rank;
            }
            i = j;
        }
        ranks
    };
    let rx = rank_vec(xs);
    let ry = rank_vec(ys);
    let mx = rx.iter().sum::<f64>() / (n as f64);
    let my = ry.iter().sum::<f64>() / (n as f64);
    let mut num = 0.0_f64;
    let mut dx2 = 0.0_f64;
    let mut dy2 = 0.0_f64;
    for i in 0..n {
        let a = rx[i] - mx;
        let b = ry[i] - my;
        num += a * b;
        dx2 += a * a;
        dy2 += b * b;
    }
    if dx2 == 0.0 || dy2 == 0.0 {
        return Err("spearman undefined for constant rank vector".into());
    }
    Ok(num / (dx2 * dy2).sqrt())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonoglotPruneQualificationInput {
    pub case_id: String,
    pub source_quant_digest: DigestRef,
    pub pruned_artifact_digest: DigestRef,
    pub non_vocab_tensors_bit_identical: bool,
    pub english_code_greedy_parity_ok: bool,
    pub multilingual_limit_declared: bool,
    pub author_card_perplexity_used_as_gate: bool,
}

pub fn qualify_monoglot_prune(
    input: &MonoglotPruneQualificationInput,
) -> (OutcomeStatus, Option<String>) {
    if input.author_card_perplexity_used_as_gate {
        return (
            OutcomeStatus::Invalid,
            Some("author_card_perplexity_must_not_be_imported_as_local_gate".into()),
        );
    }
    if !input.non_vocab_tensors_bit_identical {
        return (
            OutcomeStatus::Invalid,
            Some("non_vocabulary_tensors_altered_fails_monoglot_qualification".into()),
        );
    }
    if !input.english_code_greedy_parity_ok {
        return (
            OutcomeStatus::Invalid,
            Some("misaligned_head_english_code_greedy_fixture_failed".into()),
        );
    }
    if !input.multilingual_limit_declared {
        return (
            OutcomeStatus::Invalid,
            Some("multilingual_token_inflation_coverage_limit_omitted".into()),
        );
    }
    if input.source_quant_digest == input.pruned_artifact_digest {
        return (
            OutcomeStatus::Invalid,
            Some("pruned_artifact_must_carry_distinct_file_digest_from_source".into()),
        );
    }
    (OutcomeStatus::Accepted, None)
}

/// Deterministic conformance check inputs for Qwen3.8-27B serving & extension rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenConformanceInput {
    pub expected_quant_digest: DigestRef,
    pub observed_quant_digest: DigestRef,
    pub expected_template_digest: DigestRef,
    pub observed_template_digest: DigestRef,
    pub reasoning_effort: String,
    pub usage_present: bool,
    pub coerced_missing_usage_to_zero: bool,
    pub silent_backend_fallback: bool,
    pub population_denominator_exact: bool,
    pub hidden_reasoning_injected_into_animus: bool,
    pub cross_request_state_contaminated: bool,
    pub exact_board_mesh_qualified: bool,
    pub macaq_r_on_dense_27b_requested: bool,
    pub admission_eligible: bool,
}

/// Evaluate a Qwen3.8-27B conformance fixture deterministically.
pub fn validate_qwen_conformance(input: &QwenConformanceInput) -> OutcomeStatus {
    const ALLOWED_REASONING: &[&str] = &["off", "low", "medium", "high", "xhigh"];
    if input.macaq_r_on_dense_27b_requested {
        return OutcomeStatus::Unsupported;
    }
    if !ALLOWED_REASONING.contains(&input.reasoning_effort.as_str()) {
        return OutcomeStatus::Unsupported;
    }
    if !input.admission_eligible {
        return OutcomeStatus::Unsupported;
    }
    if input.expected_quant_digest != input.observed_quant_digest
        || input.expected_template_digest != input.observed_template_digest
        || (!input.usage_present && input.coerced_missing_usage_to_zero)
        || input.silent_backend_fallback
        || !input.population_denominator_exact
        || input.hidden_reasoning_injected_into_animus
        || input.cross_request_state_contaminated
        || !input.exact_board_mesh_qualified
    {
        return OutcomeStatus::Invalid;
    }
    OutcomeStatus::Accepted
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenConformanceFixture {
    pub fixture_id: String,
    pub category: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionArmRecord {
    pub arm_id: String,
    pub pinned_ref: String,
    pub default_enabled: bool,
    pub gate_status: ExecutionClass,
    pub non_gating_boundary: String,
}

#[derive(Debug, Clone)]
pub struct Qwen38QuantReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub reused_workload_identity: WorkloadIdentity,
    pub kv_planning: KvPlanningBudget,
    pub matched_target_parity_check: GreedyParityResult,
    pub changed_quant_divergence_retained: GreedyParityResult,
    pub quality_pass_converts_failed_parity: bool,
    pub overlapping_unified_ram_summed: bool,
    pub hidden_reasoning_leaked_into_animus_canonical: bool,
    pub offline_arm_scores: Vec<OfflineArmScore>,
    pub qda_kl_spearman_aligned: f64,
    pub qda_kl_spearman_inverted_control: f64,
    pub conformance_fixtures: Vec<QwenConformanceFixture>,
    pub extension_arms: Vec<ExtensionArmRecord>,
    pub scoped_decision: String,
}

pub fn run_default_harness() -> Result<Qwen38QuantReport, String> {
    let workload = default_workload_identity();
    let kv_planning = qwen38_27b_kv_planning_budget();

    // Reuse #3's greedy parity checker: matched target passes; changed quant diverges and is retained.
    let matched_target_parity_check = check_greedy_token_parity(
        "qwen38-matched-target-greedy-control",
        &[101, 202, 303, 404],
        &[101, 202, 303, 404],
        &[2.4, 1.9, 2.1, 3.0],
        false,
    );
    let changed_quant_divergence_retained = check_greedy_token_parity(
        "qwen38-ud-q3-k-xl-vs-q8-anchor-diff",
        &[101, 202, 303, 404],
        &[101, 202, 999, 404],
        &[2.4, 1.9, 0.08, 3.0],
        true,
    );

    // Offline rescoring: anchor arm, practical dynamic quant arm, and zero-accepted arm (cost = None).
    let anchor_score = score_retained_attempts(
        "qwen38-27b-q8-approximate-anchor",
        "changed_quant_task_quality",
        6,
        &[
            RetainedTaskAttempt {
                task_id: "t1".into(),
                category: AttemptCategory::Accepted,
                wall_ms: 1400.0,
                resource_cost_units: 14.0,
                prompt_tokens: Some(512),
                output_tokens: Some(128),
                reasoning_tokens: Some(64),
            },
            RetainedTaskAttempt {
                task_id: "t2".into(),
                category: AttemptCategory::Accepted,
                wall_ms: 1500.0,
                resource_cost_units: 15.0,
                prompt_tokens: Some(512),
                output_tokens: Some(130),
                reasoning_tokens: Some(70),
            },
            RetainedTaskAttempt {
                task_id: "t3".into(),
                category: AttemptCategory::Accepted,
                wall_ms: 1450.0,
                resource_cost_units: 14.5,
                prompt_tokens: Some(512),
                output_tokens: Some(120),
                reasoning_tokens: Some(60),
            },
            RetainedTaskAttempt {
                task_id: "t4".into(),
                category: AttemptCategory::Accepted,
                wall_ms: 1600.0,
                resource_cost_units: 16.0,
                prompt_tokens: Some(512),
                output_tokens: Some(140),
                reasoning_tokens: Some(80),
            },
            RetainedTaskAttempt {
                task_id: "t5".into(),
                category: AttemptCategory::Incorrect,
                wall_ms: 1550.0,
                resource_cost_units: 15.5,
                prompt_tokens: Some(512),
                output_tokens: Some(135),
                reasoning_tokens: Some(75),
            },
            RetainedTaskAttempt {
                task_id: "t6".into(),
                category: AttemptCategory::Truncated,
                wall_ms: 2000.0,
                resource_cost_units: 20.0,
                prompt_tokens: Some(512),
                output_tokens: Some(256),
                reasoning_tokens: None,
            },
        ],
    )?;

    let zero_accepted_score = score_retained_attempts(
        "qwen38-27b-broken-quant-zero-accepted",
        "changed_quant_task_quality",
        3,
        &[
            RetainedTaskAttempt {
                task_id: "t1".into(),
                category: AttemptCategory::Malformed,
                wall_ms: 900.0,
                resource_cost_units: 9.0,
                prompt_tokens: Some(512),
                output_tokens: None,
                reasoning_tokens: None,
            },
            RetainedTaskAttempt {
                task_id: "t2".into(),
                category: AttemptCategory::TimedOut,
                wall_ms: 5000.0,
                resource_cost_units: 50.0,
                prompt_tokens: Some(512),
                output_tokens: None,
                reasoning_tokens: None,
            },
            RetainedTaskAttempt {
                task_id: "t3".into(),
                category: AttemptCategory::Cancelled,
                wall_ms: 300.0,
                resource_cost_units: 3.0,
                prompt_tokens: Some(512),
                output_tokens: None,
                reasoning_tokens: None,
            },
        ],
    )?;

    // QDA Spearman rank correlation: negated KL (lower KL = better) vs accepted-task rate.
    let qda_kl_spearman_aligned =
        spearman_rank_correlation(&[-0.02, -0.05, -0.11, -0.24], &[0.88, 0.80, 0.72, 0.55])?;
    let qda_kl_spearman_inverted_control =
        spearman_rank_correlation(&[-0.02, -0.05, -0.11, -0.24], &[0.40, 0.55, 0.70, 0.85])?;

    // Monoglot positive & negative qualification fixtures.
    let src_digest = DigestRef::sha256(b"unsloth-ud-iq2-s-4af51b35");
    let pruned_digest = DigestRef::sha256(b"monoglot-pruned-ud-iq2-s-v1");
    let (mono_pos, _) = qualify_monoglot_prune(&MonoglotPruneQualificationInput {
        case_id: "monoglot-positive".into(),
        source_quant_digest: src_digest.clone(),
        pruned_artifact_digest: pruned_digest.clone(),
        non_vocab_tensors_bit_identical: true,
        english_code_greedy_parity_ok: true,
        multilingual_limit_declared: true,
        author_card_perplexity_used_as_gate: false,
    });
    let (mono_neg, _) = qualify_monoglot_prune(&MonoglotPruneQualificationInput {
        case_id: "monoglot-negative-misaligned-head".into(),
        source_quant_digest: src_digest,
        pruned_artifact_digest: pruned_digest,
        non_vocab_tensors_bit_identical: true,
        english_code_greedy_parity_ok: false,
        multilingual_limit_declared: true,
        author_card_perplexity_used_as_gate: false,
    });

    let quant_pin = DigestRef::sha256(b"unsloth/Qwen3.8-27B-GGUF@4af51b35:UD-Q3_K_XL");
    let tmpl_pin = DigestRef::sha256(b"Qwen/Qwen3.8-27B:chat_template.jinja:v1");
    let base_conf = QwenConformanceInput {
        expected_quant_digest: quant_pin.clone(),
        observed_quant_digest: quant_pin.clone(),
        expected_template_digest: tmpl_pin.clone(),
        observed_template_digest: tmpl_pin.clone(),
        reasoning_effort: "medium".into(),
        usage_present: true,
        coerced_missing_usage_to_zero: false,
        silent_backend_fallback: false,
        population_denominator_exact: true,
        hidden_reasoning_injected_into_animus: false,
        cross_request_state_contaminated: false,
        exact_board_mesh_qualified: true,
        macaq_r_on_dense_27b_requested: false,
        admission_eligible: true,
    };

    // Exercise score_retained_attempts for incomplete-population negative control.
    let incomplete_pop = score_retained_attempts(
        "incomplete-pop-fixture",
        "changed_quant_task_quality",
        4,
        &[RetainedTaskAttempt {
            task_id: "only-1-of-4".into(),
            category: AttemptCategory::Accepted,
            wall_ms: 100.0,
            resource_cost_units: 1.0,
            prompt_tokens: Some(64),
            output_tokens: Some(16),
            reasoning_tokens: None,
        }],
    )?;

    let mut c_reason = base_conf.clone();
    c_reason.reasoning_effort = "turbo_unsupported".into();

    let mut c_usage = base_conf.clone();
    c_usage.usage_present = false;
    c_usage.coerced_missing_usage_to_zero = true;

    let mut c_stale_quant = base_conf.clone();
    c_stale_quant.observed_quant_digest = DigestRef::sha256(b"pre-4af51b35-stale-UD-Q3_K_XL");

    let mut c_tmpl = base_conf.clone();
    c_tmpl.observed_template_digest = DigestRef::sha256(b"unsloth-differing-chat_template.jinja");

    let mut c_pop = base_conf.clone();
    c_pop.population_denominator_exact = incomplete_pop.denominator_exact;
    c_pop.silent_backend_fallback = true;

    let mut c_animus = base_conf.clone();
    c_animus.hidden_reasoning_injected_into_animus = true;

    let mut c_p150 = base_conf.clone();
    c_p150.cross_request_state_contaminated = true;
    c_p150.exact_board_mesh_qualified = false;

    let mut c_macaq_r = base_conf.clone();
    c_macaq_r.macaq_r_on_dense_27b_requested = true;

    let mut c_tt078 = base_conf.clone();
    c_tt078.admission_eligible = false;

    let conformance_fixtures = vec![
        QwenConformanceFixture {
            fixture_id: "01-positive-leading-system-developer-nested-tool-history".into(),
            category: "template_and_tool_conformance".into(),
            outcome: validate_qwen_conformance(&base_conf),
            detail: "Leading system/developer role, nested JSON tool arguments, and assistant/tool history replay bound to exact chat_template.jinja digest.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "02-negative-unsupported-reasoning-effort-value".into(),
            category: "sampling_and_reasoning".into(),
            outcome: validate_qwen_conformance(&c_reason),
            detail: "Unsupported reasoning_effort value rejected explicitly rather than silently ignored.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "03-negative-missing-usage-tokens-unknown-not-zero".into(),
            category: "telemetry_honesty".into(),
            outcome: validate_qwen_conformance(&c_usage),
            detail: "Absent prompt/output/reasoning usage counters stay None/unknown, never coerced to 0.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "04-negative-stale-quant-digest-ud-q3-k-xl-replaced".into(),
            category: "artifact_identity".into(),
            outcome: validate_qwen_conformance(&c_stale_quant),
            detail: format!("Quant name alone is insufficient: commit {UNSLOTH_GGUF_COMMIT} replaced UD-Q3_K_XL; exact file SHA-256 mismatch rejected."),
        },
        QwenConformanceFixture {
            fixture_id: "05-negative-qwen-vs-unsloth-chat-template-mismatch".into(),
            category: "template_identity".into(),
            outcome: validate_qwen_conformance(&c_tmpl),
            detail: "Qwen upstream and Unsloth chat_template.jinja differ; unverified template substitution fails conformance.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "06-negative-silent-fallback-or-incomplete-population".into(),
            category: "population_and_backend".into(),
            outcome: validate_qwen_conformance(&c_pop),
            detail: "Silent backend fallback and dropped timeout/truncated attempts fail qualification.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "07-negative-preserve-thinking-leaked-into-animus-canonical".into(),
            category: "animus_boundary".into(),
            outcome: validate_qwen_conformance(&c_animus),
            detail: "Provider preserve_thinking output stays isolated in raw adapter boundary; never injected into Animus canonical messages or verdicts.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "08-negative-p150x2-cross-request-state-contamination".into(),
            category: "tenstorrent_concurrency".into(),
            outcome: validate_qwen_conformance(&c_p150),
            detail: "Concurrent P150x2 slot reuse leaking prompt/history state fails qualification regardless of throughput; generic Blackhole/P300/4-chip cannot qualify P150x2.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "09-positive-monoglot-bit-identical-non-vocab-qualification".into(),
            category: "monoglot_prune_default_off".into(),
            outcome: mono_pos,
            detail: "Default-off Monoglot prune passes pre-scoring gate when non-vocab tensors are bit-identical, English/code greedy parity passes, and multilingual inflation limit is declared.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "10-negative-monoglot-misaligned-head-rejected-before-scoring".into(),
            category: "monoglot_prune_default_off".into(),
            outcome: mono_neg,
            detail: "Misaligned vocabulary-pruned head fails pre-scoring qualification before task scoring.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "11-negative-macaq-r-dense-27b-unsupported-see-issue-24".into(),
            category: "macaq_extension".into(),
            outcome: validate_qwen_conformance(&c_macaq_r),
            detail: "Native MACaQ-R refinement targets MoE projections (#24); dense Qwen3.8-27B is unsupported by that command.".into(),
        },
        QwenConformanceFixture {
            fixture_id: "12-control-tt-metal-078-smoke-admission-eligible-false-preserved".into(),
            category: "tenstorrent_scope".into(),
            outcome: validate_qwen_conformance(&c_tt078),
            detail: format!("tt-onix@{TT_ONIX_COMMIT} TT-Metal 0.78 smoke (2 short greedy calls @ 2048 tokens) preserves admission_eligible=false."),
        },
    ];

    let extension_arms = vec![
        ExtensionArmRecord {
            arm_id: "macaq-full-qwen38-27b".into(),
            pinned_ref: format!("huawei-csl/MACaQ@{MACAQ_COMMIT} (arXiv:2611.10649v2; open PR #4 K-tail & PR #5 E8M0; vLLM 0.20.2 vs Spark 0.29.0; max_num_seqs=1)"),
            default_enabled: false,
            gate_status: ExecutionClass::UnavailableHardwareRuntime,
            non_gating_boundary: "Separates quantization creation cost from serving cost; paper 122B 64min/32.5GB is not a 27B/Spark fit estimate.".into(),
        },
        ExtensionArmRecord {
            arm_id: "qsa-sparse-attention-retrofit".into(),
            pinned_ref: "inference-lab#31 & #32 (higa2515 indexer identity bound separately; full KV remains allocated)".into(),
            default_enabled: false,
            gate_status: ExecutionClass::UnavailableHardwareRuntime,
            non_gating_boundary: "QSA approximates attention; BF16 reference never qualifies quantized native engine.".into(),
        },
        ExtensionArmRecord {
            arm_id: "tenstorrent-p150x2-matched-concurrency".into(),
            pinned_ref: format!("tt-stack@{TT_STACK_COMMIT}, tt-onix#27/#45/#47"),
            default_enabled: false,
            gate_status: ExecutionClass::UnavailableHardwareRuntime,
            non_gating_boundary: "Distinguishes submitted/admitted/queued/resident/scheduled/overlapping concurrency; RTX K8/V4 regression preserved as caution.".into(),
        },
        ExtensionArmRecord {
            arm_id: "monoglot-english-vocab-pruning".into(),
            pinned_ref: "guideboardlabs/monoglot-english-only-vocabulary-pruning (default-off after baseline)".into(),
            default_enabled: false,
            gate_status: ExecutionClass::UnavailableHardwareRuntime,
            non_gating_boundary: "Author card file-size, perplexity, and token-ratio tables are unverified and non-gating.".into(),
        },
        ExtensionArmRecord {
            arm_id: "ttf-ternary-weights".into(),
            pinned_ref: "oguzeldereli/ttf-home (thin tree; converter & digest-addressable artifact unpinned)".into(),
            default_enabled: false,
            gate_status: ExecutionClass::BlockedOnArtifacts,
            non_gating_boundary: "Poster's 6.74 GB & 'almost identical' claims are unverified and non-gating; blocked on converter/artifact pin.".into(),
        },
        ExtensionArmRecord {
            arm_id: "qda-ptq-checkpoints-and-kl-correlation".into(),
            pinned_ref: format!("humanfia/qda@{QDA_REPO_COMMIT} (arXiv:2611.23388); W4A16@{QDA_W4A16_COMMIT}; Q4-GGUF@{QDA_Q4_GGUF_COMMIT}; Q6-GGUF@{QDA_Q6_GGUF_COMMIT}"),
            default_enabled: false,
            gate_status: ExecutionClass::UnavailableHardwareRuntime,
            non_gating_boundary: "Author 11/12 & 350-990x claims unverified; KL rank correlation supplements but never replaces independent task checking.".into(),
        },
        ExtensionArmRecord {
            arm_id: "vegaml-vllm-teacher-out-of-scope".into(),
            pinned_ref: format!("NandhaKishorM/vegaml@{VEGAML_COMMIT} (owned by rsi-lab#151, not inference-lab#18)"),
            default_enabled: false,
            gate_status: ExecutionClass::BlockedOnArtifacts,
            non_gating_boundary: "Author 35%->79% intent-stream figure is unverified and non-gating; decision-learning loop stays in rsi-lab#151.".into(),
        },
    ];

    Ok(Qwen38QuantReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical_128gb_spark_strix_or_p150x2_qwen38_27b_serving",
            "Pure-Rust KV planning, offline rescoring, Spearman KL rank correlation, and 12 conformance fixtures ran locally; physical 27B GGUF/NVFP4/vLLM/P150x2 runs remain unavailable_hardware_runtime.",
        ),
        reused_workload_identity: workload,
        kv_planning,
        matched_target_parity_check,
        changed_quant_divergence_retained,
        quality_pass_converts_failed_parity: false,
        overlapping_unified_ram_summed: false,
        hidden_reasoning_leaked_into_animus_canonical: false,
        offline_arm_scores: vec![anchor_score, zero_accepted_score],
        qda_kl_spearman_aligned,
        qda_kl_spearman_inverted_control,
        conformance_fixtures,
        extension_arms,
        scoped_decision: "INCONCLUSIVE_PENDING_PHYSICAL_27B_SAME_NODE_RUN".into(),
    })
}

pub fn inputs_json(report: &Qwen38QuantReport) -> String {
    let arms: Vec<String> = report
        .extension_arms
        .iter()
        .map(|a| {
            format!(
                concat!(
                    "{{\"arm_id\":{},\"pinned_ref\":{},\"default_enabled\":{},",
                    "\"gate_status\":{},\"non_gating_boundary\":{}}}"
                ),
                quote(&a.arm_id),
                quote(&a.pinned_ref),
                a.default_enabled,
                quote(a.gate_status.as_str()),
                quote(&a.non_gating_boundary),
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-18-qwen38-quant.inputs.v1\",\n",
            "  \"issue\": 18,\n",
            "  \"execution_class\": {},\n",
            "  \"reused_issue_7_workload_schema\": {},\n",
            "  \"kv_planning_budget\": {{\"full_attention_layers\":{},\"kv_heads\":{},\"head_dim\":{},\"elem_bytes\":{},\"bytes_per_token_per_sequence\":{},\"bytes_at_32768_tokens\":{},\"bytes_at_262144_tokens\":{},\"is_calculated_planning_not_observed_allocation\":{}}},\n",
            "  \"extension_arms\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.reused_workload_identity.schema_version),
        report.kv_planning.full_attention_layers,
        report.kv_planning.kv_heads,
        report.kv_planning.head_dim,
        report.kv_planning.elem_bytes,
        report.kv_planning.bytes_per_token_per_sequence,
        report.kv_planning.bytes_at_32768_tokens,
        report.kv_planning.bytes_at_262144_tokens,
        report
            .kv_planning
            .is_calculated_planning_not_observed_allocation,
        arms.join(","),
    )
}

pub fn results_json(report: &Qwen38QuantReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-18-qwen38-quant.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 18,").unwrap();
    writeln!(
        out,
        "  \"execution_class\": {},",
        quote(report.execution_class.as_str())
    )
    .unwrap();
    writeln!(
        out,
        "  \"hardware_gate\": {},",
        report.hardware_gate.to_json()
    )
    .unwrap();
    writeln!(
        out,
        "  \"matched_target_exact_match\": {},",
        report.matched_target_parity_check.exact_match
    )
    .unwrap();
    writeln!(
        out,
        "  \"changed_quant_exact_match\": {},",
        report.changed_quant_divergence_retained.exact_match
    )
    .unwrap();
    writeln!(
        out,
        "  \"quality_pass_converts_failed_parity\": {},",
        report.quality_pass_converts_failed_parity
    )
    .unwrap();
    writeln!(
        out,
        "  \"overlapping_unified_ram_summed\": {},",
        report.overlapping_unified_ram_summed
    )
    .unwrap();
    writeln!(
        out,
        "  \"hidden_reasoning_leaked_into_animus_canonical\": {},",
        report.hidden_reasoning_leaked_into_animus_canonical
    )
    .unwrap();
    writeln!(
        out,
        "  \"qda_kl_spearman_aligned\": {},",
        finite_f64(report.qda_kl_spearman_aligned)
    )
    .unwrap();
    writeln!(
        out,
        "  \"qda_kl_spearman_inverted_control\": {},",
        finite_f64(report.qda_kl_spearman_inverted_control)
    )
    .unwrap();
    writeln!(
        out,
        "  \"scoped_decision\": {},",
        quote(&report.scoped_decision)
    )
    .unwrap();

    let scores: Vec<String> = report
        .offline_arm_scores
        .iter()
        .map(|s| {
            format!(
                concat!(
                    "{{\"arm_id\":{},\"comparison_mode\":{},\"planned_attempts\":{},",
                    "\"accepted\":{},\"incorrect\":{},\"malformed\":{},\"unsupported\":{},",
                    "\"rejected\":{},\"cancelled\":{},\"timed_out\":{},\"truncated\":{},",
                    "\"incomplete\":{},\"acceptance_rate\":{},\"total_wall_ms\":{},",
                    "\"total_resource_cost\":{},\"cost_per_accepted_task\":{},\"denominator_exact\":{}}}"
                ),
                quote(&s.arm_id),
                quote(&s.comparison_mode),
                s.planned_attempts,
                s.accepted,
                s.incorrect,
                s.malformed,
                s.unsupported,
                s.rejected,
                s.cancelled,
                s.timed_out,
                s.truncated,
                s.incomplete,
                opt_f64(s.acceptance_rate),
                finite_f64(s.total_wall_ms),
                finite_f64(s.total_resource_cost),
                opt_f64(s.cost_per_accepted_task),
                s.denominator_exact,
            )
        })
        .collect();
    writeln!(out, "  \"offline_arm_scores\": [{}],", scores.join(",")).unwrap();

    let fixs: Vec<String> = report
        .conformance_fixtures
        .iter()
        .map(|f| {
            format!(
                "{{\"fixture_id\":{},\"category\":{},\"outcome\":{},\"detail\":{}}}",
                quote(&f.fixture_id),
                quote(&f.category),
                quote(f.outcome.as_str()),
                quote(&f.detail),
            )
        })
        .collect();
    writeln!(out, "  \"conformance_fixtures\": [{}]", fixs.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &Qwen38QuantReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #18: Qwen3.8-27B Quantization & Serving Evaluation Profile"
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
        "- Calculated KV planning (non-observed): `{} B/tok` (`{} B` @ 32768, `{} B` @ 262144)",
        report.kv_planning.bytes_per_token_per_sequence,
        report.kv_planning.bytes_at_32768_tokens,
        report.kv_planning.bytes_at_262144_tokens
    )
    .unwrap();
    writeln!(
        out,
        "- QDA KL-vs-accepted-task Spearman `rho`: aligned=`{}`, inverted negative control=`{}`",
        finite_f64(report.qda_kl_spearman_aligned),
        finite_f64(report.qda_kl_spearman_inverted_control)
    )
    .unwrap();
    writeln!(out, "- Scoped decision: `{}`", report.scoped_decision).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Offline Arm Scoring (Zero Model Calls)").unwrap();
    writeln!(
        out,
        "| Arm | Planned | Accepted | Cost/Accepted | Exact Denom |"
    )
    .unwrap();
    writeln!(out, "| --- | ---: | ---: | ---: | --- |").unwrap();
    for s in &report.offline_arm_scores {
        writeln!(
            out,
            "| `{}` | {} | {} | `{}` | `{}` |",
            s.arm_id,
            s.planned_attempts,
            s.accepted,
            opt_f64(s.cost_per_accepted_task),
            s.denominator_exact
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Conformance Fixtures").unwrap();
    for f in &report.conformance_fixtures {
        writeln!(
            out,
            "- `{}` (`{}`): `{}` — {}",
            f.fixture_id,
            f.category,
            f.outcome.as_str(),
            f.detail
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> No physical 27B serving run occurred locally. Upstream Unsloth, MACaQ, Monoglot, `ttf`, QDA, `tt-stack`, and `vegaml` figures remain strictly unverified and non-gating."
    )
    .unwrap();
    out
}
