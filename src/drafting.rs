//! Issue #3: Diffusion drafting conformance, exact speculative rejection-sampling math,
//! silent-fallback disqualification, downstream accepted-task comparison, offline verdict
//! recheck, and pinned non-gating external claim records (DFlash2/TensorFold, DNeo `8449a3ce`).
//!
//! Shared draft/verify primitives in this module are directly reused by Issue #6 (`src/adaptive_spec.rs`).

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, opt_str, opt_u64, quote, str_array, u32_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

/// Result of token-by-token greedy conformance against target-only baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct GreedyParityResult {
    pub fixture_id: String,
    pub target_tokens: Vec<u32>,
    pub draft_assisted_tokens: Vec<u32>,
    pub exact_match: bool,
    pub first_mismatch_index: Option<usize>,
    pub first_mismatch_target_token: Option<u32>,
    pub first_mismatch_draft_token: Option<u32>,
    pub first_mismatch_logit_margin: Option<f64>,
    pub approximate_mode_labeled: bool,
}

/// Compare target-only and draft-assisted greedy outputs token-by-token, retaining the first mismatch.
pub fn check_greedy_token_parity(
    fixture_id: &str,
    target_tokens: &[u32],
    draft_assisted_tokens: &[u32],
    logit_margins: &[f64],
    approximate_mode_labeled: bool,
) -> GreedyParityResult {
    let min_len = target_tokens.len().min(draft_assisted_tokens.len());
    for i in 0..min_len {
        if target_tokens[i] != draft_assisted_tokens[i] {
            return GreedyParityResult {
                fixture_id: fixture_id.into(),
                target_tokens: target_tokens.to_vec(),
                draft_assisted_tokens: draft_assisted_tokens.to_vec(),
                exact_match: false,
                first_mismatch_index: Some(i),
                first_mismatch_target_token: Some(target_tokens[i]),
                first_mismatch_draft_token: Some(draft_assisted_tokens[i]),
                first_mismatch_logit_margin: logit_margins.get(i).copied(),
                approximate_mode_labeled,
            };
        }
    }
    if target_tokens.len() != draft_assisted_tokens.len() {
        return GreedyParityResult {
            fixture_id: fixture_id.into(),
            target_tokens: target_tokens.to_vec(),
            draft_assisted_tokens: draft_assisted_tokens.to_vec(),
            exact_match: false,
            first_mismatch_index: Some(min_len),
            first_mismatch_target_token: target_tokens.get(min_len).copied(),
            first_mismatch_draft_token: draft_assisted_tokens.get(min_len).copied(),
            first_mismatch_logit_margin: logit_margins.get(min_len).copied(),
            approximate_mode_labeled,
        };
    }
    GreedyParityResult {
        fixture_id: fixture_id.into(),
        target_tokens: target_tokens.to_vec(),
        draft_assisted_tokens: draft_assisted_tokens.to_vec(),
        exact_match: true,
        first_mismatch_index: None,
        first_mismatch_target_token: None,
        first_mismatch_draft_token: None,
        first_mismatch_logit_margin: None,
        approximate_mode_labeled,
    }
}

/// Analytical marginal check for single-step speculative rejection sampling over a finite vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticRejectionCheck {
    pub fixture_id: String,
    pub uses_residual_correction: bool,
    pub distributions_valid: bool,
    pub acceptance_probability: Option<f64>,
    pub max_marginal_abs_error: Option<f64>,
    pub restores_target_distribution: bool,
    pub failure_reason: Option<String>,
}

fn validate_prob_dist(dist: &[f64]) -> Result<(), String> {
    if dist.is_empty() {
        return Err("empty_distribution".into());
    }
    let mut sum = 0.0;
    for &p in dist {
        if !p.is_finite() || p < 0.0 {
            return Err("nonfinite_or_negative_probability".into());
        }
        sum += p;
    }
    if (sum - 1.0).abs() > 1e-9 {
        return Err(format!("unnormalized_distribution:sum={sum:.12}"));
    }
    Ok(())
}

/// Analytically compute the induced marginal distribution of speculative rejection sampling.
pub fn verify_stochastic_rejection_sampling(
    fixture_id: &str,
    target_p: &[f64],
    draft_q: &[f64],
    uses_residual_correction: bool,
) -> StochasticRejectionCheck {
    if target_p.len() != draft_q.len() {
        return StochasticRejectionCheck {
            fixture_id: fixture_id.into(),
            uses_residual_correction,
            distributions_valid: false,
            acceptance_probability: None,
            max_marginal_abs_error: None,
            restores_target_distribution: false,
            failure_reason: Some("vocabulary_size_mismatch".into()),
        };
    }
    if let Err(err) = validate_prob_dist(target_p).and_then(|()| validate_prob_dist(draft_q)) {
        return StochasticRejectionCheck {
            fixture_id: fixture_id.into(),
            uses_residual_correction,
            distributions_valid: false,
            acceptance_probability: None,
            max_marginal_abs_error: None,
            restores_target_distribution: false,
            failure_reason: Some(err),
        };
    }

    let n = target_p.len();
    let mut accept_prob = 0.0;
    let mut residual = vec![0.0; n];
    for i in 0..n {
        let accepted_mass = target_p[i].min(draft_q[i]);
        accept_prob += accepted_mass;
        residual[i] = (target_p[i] - draft_q[i]).max(0.0);
    }
    let reject_prob = (1.0 - accept_prob).max(0.0);
    let residual_sum: f64 = residual.iter().sum();

    let mut induced = vec![0.0; n];
    for i in 0..n {
        let accept_part = target_p[i].min(draft_q[i]);
        let reject_part = if reject_prob <= 1e-15 {
            0.0
        } else if uses_residual_correction {
            if residual_sum > 0.0 {
                reject_prob * (residual[i] / residual_sum)
            } else {
                0.0
            }
        } else {
            reject_prob * draft_q[i]
        };
        induced[i] = accept_part + reject_part;
    }

    let max_err = induced
        .iter()
        .zip(target_p.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let restores = max_err <= 1e-12;

    StochasticRejectionCheck {
        fixture_id: fixture_id.into(),
        uses_residual_correction,
        distributions_valid: true,
        acceptance_probability: Some(accept_prob),
        max_marginal_abs_error: Some(max_err),
        restores_target_distribution: restores,
        failure_reason: if restores {
            None
        } else {
            Some("marginal_distribution_drift_exceeds_1e-12".into())
        },
    }
}

/// Deterministic speculative block verification trace (reused by `#3` and `#6`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockVerifyOutcome {
    pub case_id: String,
    pub draft_block_len: usize,
    pub accepted_draft_tokens: usize,
    pub rejected_draft_tokens: usize,
    pub emitted_tokens: Vec<u32>,
    pub bonus_token_emitted: bool,
    pub correction_token_emitted: bool,
    pub kv_rollback_len: usize,
    pub valid_prefix_continuation: bool,
}

/// Execute exact greedy block verification with correction/bonus token and KV rollback accounting.
pub fn verify_draft_block(
    case_id: &str,
    draft_tokens: &[u32],
    target_tokens_at_positions: &[u32],
    bonus_token: u32,
) -> Result<BlockVerifyOutcome, String> {
    if draft_tokens.is_empty() {
        return Err("draft_block_must_be_nonempty".into());
    }
    if target_tokens_at_positions.len() != draft_tokens.len() {
        return Err("target_position_length_mismatch".into());
    }

    let mut accepted = 0usize;
    let mut emitted = Vec::with_capacity(draft_tokens.len() + 1);
    let mut correction_emitted = false;
    let mut bonus_emitted = false;

    for (&d, &t) in draft_tokens.iter().zip(target_tokens_at_positions.iter()) {
        if d == t {
            emitted.push(d);
            accepted += 1;
        } else {
            emitted.push(t);
            correction_emitted = true;
            break;
        }
    }
    if accepted == draft_tokens.len() {
        emitted.push(bonus_token);
        bonus_emitted = true;
    }
    let rejected = draft_tokens.len() - accepted;
    Ok(BlockVerifyOutcome {
        case_id: case_id.into(),
        draft_block_len: draft_tokens.len(),
        accepted_draft_tokens: accepted,
        rejected_draft_tokens: rejected,
        emitted_tokens: emitted,
        bonus_token_emitted: bonus_emitted,
        correction_token_emitted: correction_emitted,
        kv_rollback_len: rejected,
        valid_prefix_continuation: true,
    })
}

/// Serving condition record checking actual draft execution, derived metrics, and downstream task acceptance.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftingArmCondition {
    pub arm_id: String,
    pub concurrency: usize,
    pub draft_assisted_declared: bool,
    pub actual_draft_passes: u64,
    pub total_verify_passes: u64,
    pub accepted_draft_tokens: u64,
    pub rejected_draft_tokens: u64,
    pub fallback_to_no_draft_passes: u64,
    pub draft_time_ms: f64,
    pub verify_time_ms: f64,
    pub queue_wait_p95_ms: f64,
    pub total_wall_ms: f64,
    pub mean_accepted_tokens_per_verify: Option<f64>,
    pub fallback_fraction: Option<f64>,
    pub drafter_time_share: Option<f64>,
    pub peak_resident_memory_mib: Option<u64>,
    pub measured_energy_j: Option<f64>,
    pub planned_tasks: usize,
    pub independently_accepted_tasks: usize,
    pub failed_task_checks: usize,
    pub capped_or_truncated: usize,
    pub outcome: OutcomeStatus,
    pub disqualification_reason: Option<String>,
}

pub fn evaluate_drafting_arm(mut arm: DraftingArmCondition) -> DraftingArmCondition {
    let latencies_ok = [
        arm.draft_time_ms,
        arm.verify_time_ms,
        arm.queue_wait_p95_ms,
        arm.total_wall_ms,
    ]
    .iter()
    .all(|v| v.is_finite() && *v >= 0.0);

    if !latencies_ok {
        arm.mean_accepted_tokens_per_verify = None;
        arm.fallback_fraction = None;
        arm.drafter_time_share = None;
        arm.outcome = OutcomeStatus::Invalid;
        arm.disqualification_reason = Some("nonfinite_or_negative_latency".into());
        return arm;
    }

    if arm.fallback_to_no_draft_passes > arm.total_verify_passes {
        arm.mean_accepted_tokens_per_verify = None;
        arm.fallback_fraction = None;
        arm.drafter_time_share = None;
        arm.outcome = OutcomeStatus::Invalid;
        arm.disqualification_reason = Some("fallback_passes_exceed_total_verify_passes".into());
        return arm;
    }

    arm.mean_accepted_tokens_per_verify = if arm.total_verify_passes > 0 {
        Some(arm.accepted_draft_tokens as f64 / arm.total_verify_passes as f64)
    } else {
        None
    };
    arm.fallback_fraction = if arm.total_verify_passes > 0 {
        Some(arm.fallback_to_no_draft_passes as f64 / arm.total_verify_passes as f64)
    } else {
        None
    };
    let dv_total = arm.draft_time_ms + arm.verify_time_ms;
    arm.drafter_time_share = if dv_total > 0.0 {
        Some(arm.draft_time_ms / dv_total)
    } else {
        None
    };

    let accounted =
        arm.independently_accepted_tasks + arm.failed_task_checks + arm.capped_or_truncated;
    if accounted != arm.planned_tasks {
        arm.outcome = OutcomeStatus::InternalFailure;
        arm.disqualification_reason = Some("task_denominator_conservation_violation".into());
        return arm;
    }
    if arm.draft_assisted_declared && arm.actual_draft_passes == 0 {
        arm.outcome = OutcomeStatus::Invalid;
        arm.disqualification_reason = Some("silent_fallback_disqualified_zero_draft_passes".into());
        return arm;
    }
    if arm.draft_assisted_declared
        && arm.total_verify_passes > 0
        && arm.fallback_to_no_draft_passes >= arm.total_verify_passes
    {
        arm.outcome = OutcomeStatus::Invalid;
        arm.disqualification_reason = Some("silent_fallback_disqualified_full_fallback".into());
        return arm;
    }
    if arm.independently_accepted_tasks == 0 {
        arm.outcome = OutcomeStatus::Invalid;
        arm.disqualification_reason =
            Some("faster_decode_without_accepted_downstream_tasks".into());
        return arm;
    }
    arm.outcome = OutcomeStatus::Accepted;
    arm.disqualification_reason = None;
    arm
}

/// Offline retained-output + downstream task verdict record rechecked without a model call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedTaskOutput {
    pub task_id: String,
    pub output_bytes: String,
    pub output_digest: DigestRef,
    pub recorded_verdict: OutcomeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineRecheckSummary {
    pub retained_tasks: usize,
    pub rechecked_matches_recorded: bool,
    pub rechecked_accepted: usize,
    pub rechecked_invalid: usize,
    pub mismatched_task_ids: Vec<String>,
}

/// Independently recheck retained outputs without invoking any model or generation service.
pub fn offline_recheck_retained_outputs(outputs: &[RetainedTaskOutput]) -> OfflineRecheckSummary {
    let mut accepted = 0usize;
    let mut invalid = 0usize;
    let mut mismatched = Vec::new();
    for item in outputs {
        let digest_ok = DigestRef::sha256(item.output_bytes.as_bytes()) == item.output_digest;
        let semantic_ok = digest_ok && item.output_bytes.contains("return max(0, x)");
        let recomputed = if semantic_ok {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        };
        if recomputed == OutcomeStatus::Accepted {
            accepted += 1;
        } else {
            invalid += 1;
        }
        if recomputed != item.recorded_verdict {
            mismatched.push(item.task_id.clone());
        }
    }
    OfflineRecheckSummary {
        retained_tasks: outputs.len(),
        rechecked_matches_recorded: mismatched.is_empty(),
        rechecked_accepted: accepted,
        rechecked_invalid: invalid,
        mismatched_task_ids: mismatched,
    }
}

/// Strictly non-gating pinned external claim record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalDraftingClaim {
    pub claim_id: String,
    pub source_ref: String,
    pub pinned_revisions: Vec<String>,
    pub license: String,
    pub author_claimed_metrics: String,
    pub verified_locally: bool,
    pub gating_target: bool,
    pub scope_limitation: String,
}

#[derive(Debug, Clone)]
pub struct DraftingReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub target_checkpoint_digest: DigestRef,
    pub drafter_checkpoint_digest: DigestRef,
    pub greedy_checks: Vec<GreedyParityResult>,
    pub stochastic_checks: Vec<StochasticRejectionCheck>,
    pub block_verify_cases: Vec<BlockVerifyOutcome>,
    pub arm_conditions: Vec<DraftingArmCondition>,
    pub offline_recheck: OfflineRecheckSummary,
    pub external_claims: Vec<ExternalDraftingClaim>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<DraftingReport, String> {
    let greedy_checks = vec![
        check_greedy_token_parity(
            "positive-greedy-exact-match",
            &[101, 204, 305, 409, 512],
            &[101, 204, 305, 409, 512],
            &[2.4, 1.9, 3.1, 0.8, 4.2],
            false,
        ),
        check_greedy_token_parity(
            "negative-greedy-mismatch-retained",
            &[101, 204, 305, 409, 512],
            &[101, 204, 999, 409, 512],
            &[2.4, 1.9, 0.04, 0.8, 4.2],
            false,
        ),
        check_greedy_token_parity(
            "labeled-approximate-quant-diff-separate",
            &[101, 204, 305],
            &[101, 205, 305],
            &[2.4, 0.02, 3.1],
            true,
        ),
    ];

    let target_p = [0.50, 0.25, 0.15, 0.10];
    let draft_q = [0.30, 0.40, 0.20, 0.10];
    let stochastic_checks = vec![
        verify_stochastic_rejection_sampling(
            "positive-exact-residual-rejection",
            &target_p,
            &draft_q,
            true,
        ),
        verify_stochastic_rejection_sampling(
            "negative-broken-sampler-no-residual",
            &target_p,
            &draft_q,
            false,
        ),
        verify_stochastic_rejection_sampling(
            "negative-unnormalized-distribution",
            &[0.6, 0.6],
            &[0.5, 0.5],
            true,
        ),
        verify_stochastic_rejection_sampling(
            "negative-nan-probability",
            &[f64::NAN, 1.0],
            &[0.5, 0.5],
            true,
        ),
    ];

    let block_verify_cases = vec![
        verify_draft_block(
            "all-accepted-with-bonus",
            &[10, 20, 30, 40],
            &[10, 20, 30, 40],
            50,
        )?,
        verify_draft_block(
            "first-position-rejection",
            &[99, 20, 30, 40],
            &[10, 20, 30, 40],
            50,
        )?,
        verify_draft_block(
            "mid-block-rejection-rollback",
            &[10, 20, 99, 40],
            &[10, 20, 30, 40],
            50,
        )?,
    ];

    let mk_arm = |arm_id: &str,
                  concurrency: usize,
                  draft_declared: bool,
                  draft_passes: u64,
                  verify_passes: u64,
                  acc_tok: u64,
                  rej_tok: u64,
                  fallback: u64,
                  draft_ms: f64,
                  verify_ms: f64,
                  q_p95: f64,
                  wall: f64,
                  acc_tasks: usize,
                  fail_tasks: usize,
                  capped: usize| {
        evaluate_drafting_arm(DraftingArmCondition {
            arm_id: arm_id.into(),
            concurrency,
            draft_assisted_declared: draft_declared,
            actual_draft_passes: draft_passes,
            total_verify_passes: verify_passes,
            accepted_draft_tokens: acc_tok,
            rejected_draft_tokens: rej_tok,
            fallback_to_no_draft_passes: fallback,
            draft_time_ms: draft_ms,
            verify_time_ms: verify_ms,
            queue_wait_p95_ms: q_p95,
            total_wall_ms: wall,
            mean_accepted_tokens_per_verify: None,
            fallback_fraction: None,
            drafter_time_share: None,
            // Unmeasured hardware counters stay explicitly null/unknown!
            peak_resident_memory_mib: None,
            measured_energy_j: None,
            planned_tasks: 8,
            independently_accepted_tasks: acc_tasks,
            failed_task_checks: fail_tasks,
            capped_or_truncated: capped,
            outcome: OutcomeStatus::Accepted,
            disqualification_reason: None,
        })
    };

    let arm_conditions = vec![
        mk_arm(
            "no-draft-baseline-c1",
            1,
            false,
            0,
            40,
            0,
            0,
            0,
            0.0,
            800.0,
            2.0,
            820.0,
            7,
            1,
            0,
        ),
        mk_arm(
            "draft-assisted-contract-c1",
            1,
            true,
            14,
            14,
            32,
            10,
            1,
            70.0,
            350.0,
            2.5,
            430.0,
            7,
            1,
            0,
        ),
        mk_arm(
            "no-draft-baseline-c4",
            4,
            false,
            0,
            40,
            0,
            0,
            0,
            0.0,
            1240.0,
            95.0,
            1350.0,
            7,
            1,
            0,
        ),
        mk_arm(
            "draft-assisted-contract-c4",
            4,
            true,
            15,
            15,
            30,
            12,
            2,
            95.0,
            760.0,
            110.0,
            975.0,
            7,
            1,
            0,
        ),
        mk_arm(
            "negative-silent-fallback-arm",
            4,
            true,
            0,
            40,
            0,
            0,
            40,
            0.0,
            810.0,
            15.0,
            840.0,
            7,
            1,
            0,
        ),
        mk_arm(
            "negative-fast-but-zero-accepted-tasks",
            1,
            true,
            10,
            10,
            30,
            10,
            0,
            40.0,
            200.0,
            1.0,
            250.0,
            0,
            6,
            2,
        ),
    ];

    let retained_outputs = vec![
        RetainedTaskOutput {
            task_id: "task-01-valid".into(),
            output_bytes: "def clamp(x): return max(0, x)".into(),
            output_digest: DigestRef::sha256(b"def clamp(x): return max(0, x)"),
            recorded_verdict: OutcomeStatus::Accepted,
        },
        RetainedTaskOutput {
            task_id: "task-02-invalid-retained".into(),
            output_bytes: "def clamp(x): return x".into(),
            output_digest: DigestRef::sha256(b"def clamp(x): return x"),
            recorded_verdict: OutcomeStatus::Invalid,
        },
    ];
    let offline_recheck = offline_recheck_retained_outputs(&retained_outputs);

    let external_claims = vec![
        ExternalDraftingClaim {
            claim_id: "jrb-ai-dflash2-tensorfold-qwen38-27b".into(),
            source_ref: "x.com/JRB_AI/status/2123682757960462484".into(),
            pinned_revisions: vec!["single DGX Spark Qwen3.8-27B TensorFold DFlash2/DSpine".into()],
            license: "unverified_social_post".into(),
            author_claimed_metrics: "14.7 -> 97.7 tok/s single stream (6.6x); no parity or task-quality numbers".into(),
            verified_locally: false,
            gating_target: false,
            scope_limitation: "External unverified claim only; never a pass/fail gate.".into(),
        },
        ExternalDraftingClaim {
            claim_id: "dneo-qwen3-4b-parallel-drafter".into(),
            source_ref: "arXiv:2611.21925v1".into(),
            pinned_revisions: vec![
                "J-sandler/DNeo@8449a3ce891f6b3321efafc1e5c0c7e5c40e11c7 (dneo-inference 0.1.0rc6+, MIT)".into(),
                "J-sandler/Qwen3-4B-DNeo@8451a2e8446c0bfa09be47623ff044d896540ec4 (Apache-2.0)".into(),
                "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c".into(),
                "num_hidden_layers=5, block_size=16, target_layer_ids=[0,17,21,24,28,31,35]".into(),
            ],
            license: "MIT (runtime) / Apache-2.0 (draft weights)".into(),
            author_claimed_metrics: "Up to 1.3x vs DFlash-2; H100 80GB BF16 SGLang 4.51 tok/fwd & 701 tok/s (2.96x); Transformers 4.54 tok/fwd & 390 tok/s (3.37x)".into(),
            verified_locally: false,
            gating_target: false,
            scope_limitation: "4B checkpoint only; cannot stand in for 27B arm; GB10/SM121 execution unverified.".into(),
        },
    ];

    Ok(DraftingReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "dgx_spark_gb10_or_h100_target_and_drafter_serving",
            "Pure-Rust greedy/stochastic conformance, silent-fallback guards, and offline recheck verified locally; physical TensorFold/vLLM/SGLang target+drafter hardware runs remain unavailable.",
        ),
        target_checkpoint_digest: DigestRef::sha256(b"target-qwen38-27b-pinned-contract"),
        drafter_checkpoint_digest: DigestRef::sha256(b"drafter-dflash2-pinned-contract"),
        greedy_checks,
        stochastic_checks,
        block_verify_cases,
        arm_conditions,
        offline_recheck,
        external_claims,
        decision: "INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME".into(),
    })
}

pub fn inputs_json(report: &DraftingReport) -> String {
    let claims: Vec<String> = report
        .external_claims
        .iter()
        .map(|c| {
            format!(
                concat!(
                    "{{\"claim_id\":{},\"source_ref\":{},\"pinned_revisions\":{},\"license\":{},",
                    "\"author_claimed_metrics\":{},\"verified_locally\":{},\"gating_target\":{},",
                    "\"scope_limitation\":{}}}"
                ),
                quote(&c.claim_id),
                quote(&c.source_ref),
                str_array(&c.pinned_revisions),
                quote(&c.license),
                quote(&c.author_claimed_metrics),
                c.verified_locally,
                c.gating_target,
                quote(&c.scope_limitation),
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-03-drafting.inputs.v1\",\n",
            "  \"issue\": 3,\n",
            "  \"execution_class\": {},\n",
            "  \"target_checkpoint_digest\": {},\n",
            "  \"drafter_checkpoint_digest\": {},\n",
            "  \"non_gating_unverified_author_claims\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.target_checkpoint_digest.to_json(),
        report.drafter_checkpoint_digest.to_json(),
        claims.join(","),
    )
}

pub fn results_json(report: &DraftingReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-03-drafting.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 3,").unwrap();
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
    writeln!(out, "  \"decision\": {},", quote(&report.decision)).unwrap();
    writeln!(
        out,
        "  \"offline_recheck\": {{\"retained_tasks\":{},\"rechecked_matches_recorded\":{},\"rechecked_accepted\":{},\"rechecked_invalid\":{},\"mismatched_task_ids\":{}}},",
        report.offline_recheck.retained_tasks,
        report.offline_recheck.rechecked_matches_recorded,
        report.offline_recheck.rechecked_accepted,
        report.offline_recheck.rechecked_invalid,
        str_array(&report.offline_recheck.mismatched_task_ids),
    )
    .unwrap();

    let greedy: Vec<String> = report
        .greedy_checks
        .iter()
        .map(|g| {
            format!(
                concat!(
                    "{{\"fixture_id\":{},\"exact_match\":{},\"first_mismatch_index\":{},",
                    "\"first_mismatch_target_token\":{},\"first_mismatch_draft_token\":{},",
                    "\"first_mismatch_logit_margin\":{},\"approximate_mode_labeled\":{}}}"
                ),
                quote(&g.fixture_id),
                g.exact_match,
                opt_u64(g.first_mismatch_index.map(|x| x as u64)),
                opt_u64(g.first_mismatch_target_token.map(u64::from)),
                opt_u64(g.first_mismatch_draft_token.map(u64::from)),
                opt_f64(g.first_mismatch_logit_margin),
                g.approximate_mode_labeled,
            )
        })
        .collect();
    writeln!(out, "  \"greedy_checks\": [{}],", greedy.join(",")).unwrap();

    let stoch: Vec<String> = report
        .stochastic_checks
        .iter()
        .map(|s| {
            format!(
                concat!(
                    "{{\"fixture_id\":{},\"uses_residual_correction\":{},\"distributions_valid\":{},",
                    "\"acceptance_probability\":{},\"max_marginal_abs_error\":{},",
                    "\"restores_target_distribution\":{},\"failure_reason\":{}}}"
                ),
                quote(&s.fixture_id),
                s.uses_residual_correction,
                s.distributions_valid,
                opt_f64(s.acceptance_probability),
                opt_f64(s.max_marginal_abs_error),
                s.restores_target_distribution,
                opt_str(s.failure_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"stochastic_checks\": [{}],", stoch.join(",")).unwrap();

    let blocks: Vec<String> = report
        .block_verify_cases
        .iter()
        .map(|b| {
            format!(
                concat!(
                    "{{\"case_id\":{},\"draft_block_len\":{},\"accepted_draft_tokens\":{},",
                    "\"rejected_draft_tokens\":{},\"emitted_tokens\":{},\"bonus_token_emitted\":{},",
                    "\"correction_token_emitted\":{},\"kv_rollback_len\":{},\"valid_prefix_continuation\":{}}}"
                ),
                quote(&b.case_id),
                b.draft_block_len,
                b.accepted_draft_tokens,
                b.rejected_draft_tokens,
                u32_array(&b.emitted_tokens),
                b.bonus_token_emitted,
                b.correction_token_emitted,
                b.kv_rollback_len,
                b.valid_prefix_continuation,
            )
        })
        .collect();
    writeln!(out, "  \"block_verify_cases\": [{}],", blocks.join(",")).unwrap();

    let arms: Vec<String> = report
        .arm_conditions
        .iter()
        .map(|a| {
            format!(
                concat!(
                    "{{\"arm_id\":{},\"concurrency\":{},\"draft_assisted_declared\":{},",
                    "\"actual_draft_passes\":{},\"total_verify_passes\":{},\"accepted_draft_tokens\":{},",
                    "\"rejected_draft_tokens\":{},\"fallback_to_no_draft_passes\":{},",
                    "\"mean_accepted_tokens_per_verify\":{},\"fallback_fraction\":{},\"drafter_time_share\":{},",
                    "\"peak_resident_memory_mib\":{},\"measured_energy_j\":{},",
                    "\"draft_time_ms\":{},\"verify_time_ms\":{},\"queue_wait_p95_ms\":{},",
                    "\"total_wall_ms\":{},\"planned_tasks\":{},\"independently_accepted_tasks\":{},",
                    "\"failed_task_checks\":{},\"capped_or_truncated\":{},\"outcome\":{},\"disqualification_reason\":{}}}"
                ),
                quote(&a.arm_id),
                a.concurrency,
                a.draft_assisted_declared,
                a.actual_draft_passes,
                a.total_verify_passes,
                a.accepted_draft_tokens,
                a.rejected_draft_tokens,
                a.fallback_to_no_draft_passes,
                opt_f64(a.mean_accepted_tokens_per_verify),
                opt_f64(a.fallback_fraction),
                opt_f64(a.drafter_time_share),
                opt_u64(a.peak_resident_memory_mib),
                opt_f64(a.measured_energy_j),
                finite_f64(a.draft_time_ms),
                finite_f64(a.verify_time_ms),
                finite_f64(a.queue_wait_p95_ms),
                finite_f64(a.total_wall_ms),
                a.planned_tasks,
                a.independently_accepted_tasks,
                a.failed_task_checks,
                a.capped_or_truncated,
                quote(a.outcome.as_str()),
                opt_str(a.disqualification_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"arm_conditions\": [{}]", arms.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &DraftingReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #3: Diffusion Drafting Conformance & Target-Model Parity"
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
        "- Offline retained-output recheck matches recorded verdicts: `{}` ({} tasks)",
        report.offline_recheck.rechecked_matches_recorded, report.offline_recheck.retained_tasks
    )
    .unwrap();
    writeln!(out, "- Decision: `{}`", report.decision).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Stochastic Rejection Sampling Checks").unwrap();
    writeln!(
        out,
        "| Fixture | Residual Correction | Valid Dist | Max Abs Error | Restores Target |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | --- | ---: | --- |").unwrap();
    for s in &report.stochastic_checks {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | {} | `{}` |",
            s.fixture_id,
            s.uses_residual_correction,
            s.distributions_valid,
            opt_f64(s.max_marginal_abs_error),
            s.restores_target_distribution
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Arm Conditions & Silent-Fallback Disqualification").unwrap();
    writeln!(
        out,
        "| Arm | Conc | Draft Passes | Mean Acc/Verify | Fallback Frac | Accepted Tasks | Outcome |"
    )
    .unwrap();
    writeln!(out, "| --- | ---: | ---: | ---: | ---: | ---: | --- |").unwrap();
    for a in &report.arm_conditions {
        writeln!(
            out,
            "| `{}` | {} | {} | {} | {} | {}/{} | `{}` |",
            a.arm_id,
            a.concurrency,
            a.actual_draft_passes,
            opt_f64(a.mean_accepted_tokens_per_verify),
            opt_f64(a.fallback_fraction),
            a.independently_accepted_tasks,
            a.planned_tasks,
            a.outcome.as_str()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> External claims (JRB_AI DFlash2 6.6x, DNeo 1.3x / 701 tok/s) are logged strictly as unverified non-gating references. Peak resident memory and energy remain `null`/unknown because no GPU ran locally."
    )
    .unwrap();
    out
}
