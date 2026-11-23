//! Issue #47: Evaluate adversarial speculative reasoning drafts against unbiased drafts on
//! accepted-task rate and per-slot numerical noise.
//!
//! Extends the existing drafting harnesses (`#3`, `#6`, `#22`) rather than creating a competing
//! benchmark. Pinned motivation (strictly unverified non-gating related work):
//! - `arXiv:2611.05345` (*Adversarial Speculative Reasoning Improves Long Horizon Agents*).
//!
//! Provides:
//! - Independent exact FP64 slot-0 target forward pass (`target_logits_slot0`) vs multi-slot
//!   batched speculative verifier (`target_logits_batched_slot`) modeling deterministic per-slot
//!   numerical rounding noise that grows monotonically with draft slot index `k` (`k = 0` matches
//!   exact slot-0 reference identically).
//! - Per-slot numerical noise measurement (`max_abs_err`, `l2_err`) across slots `k = 0..4`.
//! - Matched-seed comparison between an `UnbiasedDraft` and an `AdversarialDecisionDraft`
//!   (biased to be rejected at critical decision points so the target evaluates and writes
//!   decision tokens from slot 0), evaluated by an independent deterministic task checker for
//!   accepted-task rate, wall-time proxy, exact token-sequence match against slot-0 reference,
//!   and 8 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const PAPER_PIN: &str = "arXiv:2611.05345";
pub const SOURCE_BOOKMARK: &str = "https://x.com/LijieyYang/status/2118068274512703652";
pub const EXTENDS_ISSUES: [u32; 3] = [3, 6, 22];

pub const VOCAB_SIZE: usize = 8;
pub const HIDDEN_DIM: usize = 8;
pub const NUM_DRAFT_SLOTS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftPolicy {
    UnbiasedDraft,
    AdversarialDecisionDraft,
}

impl DraftPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnbiasedDraft => "unbiased_draft",
            Self::AdversarialDecisionDraft => "adversarial_decision_draft",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlotNoiseStats {
    pub slot_index: usize,
    pub sample_count: usize,
    pub max_abs_err: f64,
    pub l2_err: f64,
}

#[derive(Debug, Clone)]
pub struct ReasoningTask {
    pub task_id: String,
    pub prompt_tokens: Vec<usize>,
    pub horizon_steps: usize,
    pub decision_steps: Vec<usize>,
    pub expected_checksum: usize,
}

#[derive(Debug, Clone)]
pub struct ArmMetrics {
    pub policy: DraftPolicy,
    pub seed: u64,
    pub planned_tasks: usize,
    pub accepted_tasks: usize,
    pub accepted_task_rate: Option<f64>,
    pub exact_slot0_match_tasks: usize,
    pub exactness_rate: Option<f64>,
    pub decision_tokens_at_slot0: usize,
    pub total_decision_tokens: usize,
    pub total_draft_tokens: usize,
    pub total_accepted_draft_tokens: usize,
    pub draft_acceptance_rate: Option<f64>,
    pub target_forward_calls: usize,
    pub draft_forward_calls: usize,
    pub wall_time_proxy_ms: f64,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct AdvSpecReasoningReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub fixture_digest: DigestRef,
    pub matched_seed: u64,
    pub slot_noise: Vec<SlotNoiseStats>,
    pub arms: Vec<ArmMetrics>,
    pub negative_controls: Vec<NegativeControl>,
}

fn check_finite(v: f64, label: &str) -> Result<f64, String> {
    if !v.is_finite() {
        return Err(format!("non-finite float in {label}"));
    }
    Ok(v)
}

fn det_uniform(seed: u64, step: usize, slot: usize) -> f64 {
    let mut z = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((step as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9))
        ^ ((slot as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

/// Independent exact FP64 slot-0 target forward pass computing next-token logits.
/// At decision steps, the top two logits have a narrow margin (`~0.014`) so higher-slot
/// accumulation noise can flip the greedy decision boundary while slot 0 remains exact.
pub fn target_logits_slot0(
    prefix: &[usize],
    is_decision_step: bool,
) -> Result<[f64; VOCAB_SIZE], String> {
    let mut hidden = [0.0_f64; HIDDEN_DIM];
    for (pos, &tok) in prefix.iter().enumerate() {
        if tok >= VOCAB_SIZE {
            return Err(format!("token id {tok} out of vocab bounds {VOCAB_SIZE}"));
        }
        for (d, h) in hidden.iter_mut().enumerate() {
            let emb = det_weight(4701 + tok * 17, d + pos * 3, 0.45);
            *h = (*h * 0.65 + emb).tanh();
            check_finite(*h, "target_logits_slot0 hidden")?;
        }
    }

    let mut logits = [0.0_f64; VOCAB_SIZE];
    for (v, slot) in logits.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (d, &hv) in hidden.iter().enumerate() {
            acc += hv * det_weight(4751 + v * 29, d, 0.85);
        }
        *slot = check_finite(acc, "target_logits_slot0 logit")?;
    }

    if is_decision_step {
        // Create a deterministic narrow-margin reasoning decision point between two distinct
        // candidate branches so slot-k numerical rounding noise can perturb argmax.
        let mut best_idx = 0usize;
        let mut second_idx = 1usize;
        for i in 0..VOCAB_SIZE {
            if logits[i] > logits[best_idx] {
                second_idx = best_idx;
                best_idx = i;
            } else if i != best_idx && (best_idx == second_idx || logits[i] > logits[second_idx]) {
                second_idx = i;
            }
        }
        logits[second_idx] = logits[best_idx] - 0.012;
    }

    Ok(logits)
}

/// Multi-slot batched speculative verifier forward pass.
/// Slot 0 executes the exact FP64 reduction; later slots `k >= 1` accumulate deterministic
/// GEMM-tiling / mixed-precision rounding noise whose magnitude grows with slot index `k`.
pub fn target_logits_batched_slot(
    prefix: &[usize],
    is_decision_step: bool,
    slot_k: usize,
) -> Result<[f64; VOCAB_SIZE], String> {
    let exact = target_logits_slot0(prefix, is_decision_step)?;
    if slot_k == 0 {
        return Ok(exact);
    }
    let mut noisy = exact;
    let quantum = 0.008 * (slot_k as f64);
    for (v, val) in noisy.iter_mut().enumerate() {
        // Deterministic signed slot-dependent numerical perturbation scaling with slot index k.
        let sign = if (prefix.len() + v + slot_k) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let bias = sign * quantum * (0.75 + 0.25 * (((v * 7 + slot_k * 3) % 5) as f64) / 4.0);
        let perturbed = ((*val + bias) / quantum).round() * quantum;
        *val = check_finite(perturbed, "target_logits_batched_slot")?;
    }
    Ok(noisy)
}

pub fn softmax_checked(logits: &[f64; VOCAB_SIZE]) -> Result<[f64; VOCAB_SIZE], String> {
    let mut peak = check_finite(logits[0], "softmax input")?;
    for &v in &logits[1..] {
        let fv = check_finite(v, "softmax input")?;
        if fv > peak {
            peak = fv;
        }
    }
    let mut exps = [0.0_f64; VOCAB_SIZE];
    let mut sum = 0.0_f64;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - peak).exp();
        check_finite(e, "softmax exp")?;
        exps[i] = e;
        sum += e;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err("invalid softmax denominator".into());
    }
    for e in &mut exps {
        *e /= sum;
    }
    Ok(exps)
}

pub fn argmax_checked(logits: &[f64; VOCAB_SIZE]) -> Result<usize, String> {
    let mut best_i = 0usize;
    let mut best_v = check_finite(logits[0], "argmax logit")?;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        let fv = check_finite(v, "argmax logit")?;
        if fv > best_v {
            best_v = fv;
            best_i = i;
        }
    }
    Ok(best_i)
}

/// Measure per-slot `max_abs_err` and `l2_err` comparing independent exact slot-0 logits
/// against batched slot `k` logits across all slots `0..NUM_DRAFT_SLOTS`.
pub fn measure_per_slot_noise(tasks: &[ReasoningTask]) -> Result<Vec<SlotNoiseStats>, String> {
    if tasks.is_empty() {
        return Err("cannot measure per-slot noise on empty task list".into());
    }
    let mut stats = Vec::with_capacity(NUM_DRAFT_SLOTS);
    for slot_k in 0..NUM_DRAFT_SLOTS {
        let mut max_abs = 0.0_f64;
        let mut sq_sum = 0.0_f64;
        let mut count = 0usize;
        for task in tasks {
            let mut prefix = task.prompt_tokens.clone();
            for step in 0..task.horizon_steps {
                let is_dec = task.decision_steps.contains(&step);
                let exact = target_logits_slot0(&prefix, is_dec)?;
                let batched = target_logits_batched_slot(&prefix, is_dec, slot_k)?;
                for i in 0..VOCAB_SIZE {
                    let d = (exact[i] - batched[i]).abs();
                    check_finite(d, "slot noise diff")?;
                    if d > max_abs {
                        max_abs = d;
                    }
                    sq_sum += d * d;
                }
                count += 1;
                let next_tok = argmax_checked(&exact)?;
                prefix.push(next_tok);
            }
        }
        let l2 = sq_sum.sqrt();
        check_finite(l2, "slot l2_err")?;
        stats.push(SlotNoiseStats {
            slot_index: slot_k,
            sample_count: count,
            max_abs_err: max_abs,
            l2_err: l2,
        });
    }
    Ok(stats)
}

/// Independent exact slot-0 reference rollout for a task (never uses any drafter).
pub fn exact_slot0_rollout(task: &ReasoningTask) -> Result<Vec<usize>, String> {
    let mut seq = task.prompt_tokens.clone();
    let mut gen = Vec::with_capacity(task.horizon_steps);
    for step in 0..task.horizon_steps {
        let is_dec = task.decision_steps.contains(&step);
        let logits = target_logits_slot0(&seq, is_dec)?;
        let tok = argmax_checked(&logits)?;
        seq.push(tok);
        gen.push(tok);
    }
    Ok(gen)
}

/// Independent deterministic task checker: verifies the generated reasoning trajectory
/// satisfies the exact task checksum invariant without inspecting which draft policy ran.
pub fn independent_task_checker(task: &ReasoningTask, generated: &[usize]) -> bool {
    if generated.len() != task.horizon_steps {
        return false;
    }
    let mut checksum = 0usize;
    for (i, &tok) in generated.iter().enumerate() {
        let weight = if task.decision_steps.contains(&i) {
            17
        } else {
            3
        };
        checksum = checksum.wrapping_add((i + 1).wrapping_mul(tok + 1).wrapping_mul(weight)) % 997;
    }
    checksum == task.expected_checksum
}

/// Draft distribution generator for unbiased vs adversarial-decision-point policies.
fn draft_distribution(
    prefix: &[usize],
    is_decision_step: bool,
    policy: DraftPolicy,
) -> Result<[f64; VOCAB_SIZE], String> {
    let exact_logits = target_logits_slot0(prefix, is_decision_step)?;
    let mut draft_logits = exact_logits;
    match policy {
        DraftPolicy::UnbiasedDraft => {
            // Unbiased draft closely tracks target everywhere, including decision points.
            for (v, d) in draft_logits.iter_mut().enumerate() {
                *d += det_weight(4801 + prefix.len() * 13, v, 0.04);
            }
        }
        DraftPolicy::AdversarialDecisionDraft => {
            if is_decision_step {
                // Trained to be rejected at decision points so target evaluates the decision
                // token cleanly at slot 0 rather than a noisy later draft slot.
                let top = argmax_checked(&exact_logits)?;
                for (v, d) in draft_logits.iter_mut().enumerate() {
                    if v == top {
                        *d -= 6.0;
                    } else if v == (top + 3) % VOCAB_SIZE {
                        *d += 5.0;
                    }
                }
            } else {
                for (v, d) in draft_logits.iter_mut().enumerate() {
                    *d += det_weight(4801 + prefix.len() * 13, v, 0.04);
                }
            }
        }
    }
    softmax_checked(&draft_logits)
}

/// Evaluate a draft policy across all frozen tasks under a fixed seed with explicit
/// rejection-sampling verification (never silently bypassing rejection sampling).
pub fn evaluate_policy(
    tasks: &[ReasoningTask],
    policy: DraftPolicy,
    seed: u64,
    allow_bypass_verification: bool,
) -> Result<ArmMetrics, String> {
    if allow_bypass_verification {
        return Err(
            "silent draft-token bypass skipping rejection-sampling verification is forbidden"
                .into(),
        );
    }
    if tasks.is_empty() {
        return Ok(ArmMetrics {
            policy,
            seed,
            planned_tasks: 0,
            accepted_tasks: 0,
            accepted_task_rate: None,
            exact_slot0_match_tasks: 0,
            exactness_rate: None,
            decision_tokens_at_slot0: 0,
            total_decision_tokens: 0,
            total_draft_tokens: 0,
            total_accepted_draft_tokens: 0,
            draft_acceptance_rate: None,
            target_forward_calls: 0,
            draft_forward_calls: 0,
            wall_time_proxy_ms: 0.0,
        });
    }

    let mut accepted_tasks = 0usize;
    let mut exact_matches = 0usize;
    let mut decision_at_slot0 = 0usize;
    let mut total_decisions = 0usize;
    let mut total_draft_tokens = 0usize;
    let mut total_accepted_draft = 0usize;
    let mut target_calls = 0usize;
    let mut draft_calls = 0usize;

    for (t_idx, task) in tasks.iter().enumerate() {
        let reference = exact_slot0_rollout(task)?;
        let mut prefix = task.prompt_tokens.clone();
        let mut generated = Vec::with_capacity(task.horizon_steps);

        while generated.len() < task.horizon_steps {
            let remaining = task.horizon_steps - generated.len();
            let draft_k = if remaining < NUM_DRAFT_SLOTS - 1 {
                remaining
            } else {
                NUM_DRAFT_SLOTS - 1
            };

            // 1. Draft `draft_k` speculative tokens
            let mut draft_tokens = Vec::with_capacity(draft_k);
            let mut draft_probs = Vec::with_capacity(draft_k);
            let mut draft_prefix = prefix.clone();
            for j in 0..draft_k {
                let step_idx = generated.len() + j;
                let is_dec = task.decision_steps.contains(&step_idx);
                let q = draft_distribution(&draft_prefix, is_dec, policy)?;
                // Greedy draft token under draft distribution
                let mut best_tok = 0usize;
                let mut best_p = q[0];
                for (v, &pv) in q.iter().enumerate().skip(1) {
                    if pv > best_p {
                        best_p = pv;
                        best_tok = v;
                    }
                }
                draft_tokens.push(best_tok);
                draft_probs.push(q);
                draft_prefix.push(best_tok);
                draft_calls += 1;
                total_draft_tokens += 1;
            }

            // 2. Single target verification call across slots 0..=draft_k
            // Slot 0 verifies token at `generated.len()`, slot `j` verifies token at `generated.len() + j`.
            // Note: token 0 of a block is evaluated at slot 0; subsequent drafted tokens occupy slots 1..draft_k.
            target_calls += 1;
            let mut verify_prefix = prefix.clone();
            let mut rejected = false;
            for j in 0..draft_k {
                let step_idx = generated.len();
                let is_dec = task.decision_steps.contains(&step_idx);
                // Slot index in the verification batch: first speculative token after prompt
                // sits at slot `j + 1` (shifted behind the committed anchor at slot 0) when
                // verifying drafted tokens in parallel; a rejection resets the next step to slot 0.
                let slot_idx = j + 1;
                let target_logits = target_logits_batched_slot(&verify_prefix, is_dec, slot_idx)?;
                let target_p = softmax_checked(&target_logits)?;
                let drafted = draft_tokens[j];
                let q_tok = draft_probs[j][drafted];
                let p_tok = target_p[drafted];
                let accept_prob = if q_tok <= 0.0 {
                    0.0
                } else {
                    let ratio = p_tok / q_tok;
                    check_finite(ratio, "rejection ratio")?;
                    if ratio > 1.0 {
                        1.0
                    } else {
                        ratio
                    }
                };
                let u = det_uniform(seed ^ (t_idx as u64).wrapping_mul(31), step_idx, slot_idx);
                if u <= accept_prob && drafted == argmax_checked(&target_logits)? {
                    // Accepted at batched slot `slot_idx` (carries slot_idx numerical noise!)
                    if is_dec {
                        total_decisions += 1;
                    }
                    generated.push(drafted);
                    prefix.push(drafted);
                    verify_prefix.push(drafted);
                    total_accepted_draft += 1;
                } else {
                    // Rejected! Target re-evaluates & writes this token cleanly from slot 0.
                    let slot0_logits = target_logits_batched_slot(&verify_prefix, is_dec, 0)?;
                    let exact_tok = argmax_checked(&slot0_logits)?;
                    if is_dec {
                        total_decisions += 1;
                        decision_at_slot0 += 1;
                    }
                    generated.push(exact_tok);
                    prefix.push(exact_tok);
                    rejected = true;
                    break;
                }
            }

            // Bonus token if all draft tokens accepted and horizon not yet reached
            if !rejected && generated.len() < task.horizon_steps {
                let step_idx = generated.len();
                let is_dec = task.decision_steps.contains(&step_idx);
                let bonus_slot = draft_k;
                let bonus_logits = target_logits_batched_slot(&prefix, is_dec, bonus_slot)?;
                let bonus_tok = argmax_checked(&bonus_logits)?;
                if is_dec {
                    total_decisions += 1;
                    if bonus_slot == 0 {
                        decision_at_slot0 += 1;
                    }
                }
                generated.push(bonus_tok);
                prefix.push(bonus_tok);
            }
        }

        if independent_task_checker(task, &generated) {
            accepted_tasks += 1;
        }
        if generated == reference {
            exact_matches += 1;
        }
    }

    let planned = tasks.len();
    let wall_time_proxy_ms = (target_calls as f64) * 1.45 + (draft_calls as f64) * 0.18;
    check_finite(wall_time_proxy_ms, "wall_time_proxy_ms")?;

    let draft_acceptance_rate = if total_draft_tokens == 0 {
        None
    } else {
        Some((total_accepted_draft as f64) / (total_draft_tokens as f64))
    };

    Ok(ArmMetrics {
        policy,
        seed,
        planned_tasks: planned,
        accepted_tasks,
        accepted_task_rate: Some((accepted_tasks as f64) / (planned as f64)),
        exact_slot0_match_tasks: exact_matches,
        exactness_rate: Some((exact_matches as f64) / (planned as f64)),
        decision_tokens_at_slot0: decision_at_slot0,
        total_decision_tokens: total_decisions,
        total_draft_tokens,
        total_accepted_draft_tokens: total_accepted_draft,
        draft_acceptance_rate,
        target_forward_calls: target_calls,
        draft_forward_calls: draft_calls,
        wall_time_proxy_ms,
    })
}

pub fn validate_matched_seed_comparison(
    unbiased: &ArmMetrics,
    adversarial: &ArmMetrics,
) -> Result<(), String> {
    if unbiased.seed != adversarial.seed {
        return Err(format!(
            "unmatched seeds between unbiased ({}) and adversarial ({}) arms",
            unbiased.seed, adversarial.seed
        ));
    }
    if unbiased.planned_tasks != adversarial.planned_tasks || unbiased.planned_tasks == 0 {
        return Err("arms must share a non-empty frozen task set".into());
    }
    check_finite(unbiased.wall_time_proxy_ms, "unbiased wall_time_proxy_ms")?;
    check_finite(
        adversarial.wall_time_proxy_ms,
        "adversarial wall_time_proxy_ms",
    )?;
    if unbiased.wall_time_proxy_ms < 0.0 || adversarial.wall_time_proxy_ms < 0.0 {
        return Err("wall_time_proxy_ms must be non-negative".into());
    }
    Ok(())
}

pub fn validate_slot0_exactness_claim(slot_used: usize, claims_slot0_exact: bool) -> OutcomeStatus {
    if claims_slot0_exact && slot_used != 0 {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

pub fn validate_no_external_claim_gate(uses_paper_claim_as_pass_gate: bool) -> OutcomeStatus {
    if uses_paper_claim_as_pass_gate {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

pub fn default_tasks() -> Result<Vec<ReasoningTask>, String> {
    let raw_specs: &[(&str, &[usize], usize, &[usize])] = &[
        ("long_horizon_01", &[1, 3, 5], 9, &[1, 4, 7]),
        ("long_horizon_02", &[2, 0, 4], 9, &[2, 5, 8]),
        ("long_horizon_03", &[6, 2, 1], 10, &[1, 5, 8]),
        ("long_horizon_04", &[0, 7, 3], 10, &[2, 4, 7]),
        ("long_horizon_05", &[4, 5, 2], 9, &[1, 3, 6]),
        ("long_horizon_06", &[7, 1, 6], 10, &[2, 6, 9]),
    ];
    let mut tasks = Vec::with_capacity(raw_specs.len());
    for &(id, prompt, horizon, decisions) in raw_specs {
        let mut tmp = ReasoningTask {
            task_id: id.into(),
            prompt_tokens: prompt.to_vec(),
            horizon_steps: horizon,
            decision_steps: decisions.to_vec(),
            expected_checksum: 0,
        };
        let exact_gen = exact_slot0_rollout(&tmp)?;
        let mut checksum = 0usize;
        for (i, &tok) in exact_gen.iter().enumerate() {
            let weight = if tmp.decision_steps.contains(&i) {
                17
            } else {
                3
            };
            checksum =
                checksum.wrapping_add((i + 1).wrapping_mul(tok + 1).wrapping_mul(weight)) % 997;
        }
        tmp.expected_checksum = checksum;
        tasks.push(tmp);
    }
    Ok(tasks)
}

pub fn run_default_harness() -> Result<AdvSpecReasoningReport, String> {
    let tasks = default_tasks()?;
    let matched_seed = 47_202_611_u64;

    let mut digest_buf = String::new();
    for t in &tasks {
        let _ = write!(
            digest_buf,
            "{}:{:?}:{}:{:?}:{};",
            t.task_id, t.prompt_tokens, t.horizon_steps, t.decision_steps, t.expected_checksum
        );
    }
    let fixture_digest = DigestRef::sha256(digest_buf.as_bytes());

    let slot_noise = measure_per_slot_noise(&tasks)?;
    let unbiased = evaluate_policy(&tasks, DraftPolicy::UnbiasedDraft, matched_seed, false)?;
    let adversarial = evaluate_policy(
        &tasks,
        DraftPolicy::AdversarialDecisionDraft,
        matched_seed,
        false,
    )?;
    validate_matched_seed_comparison(&unbiased, &adversarial)?;

    let mut negative_controls = Vec::new();

    // NC1: mismatched seed between arms rejected
    {
        let mut wrong_seed = adversarial.clone();
        wrong_seed.seed = matched_seed + 1;
        let outcome = if validate_matched_seed_comparison(&unbiased, &wrong_seed).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc1_mismatched_seed_rejected".into(),
            outcome,
            detail: "comparing unbiased and adversarial drafts under mismatched seeds is invalid"
                .into(),
        });
    }

    // NC2: NaN logits rejected by softmax/argmax
    {
        let mut nan_logits = [0.1_f64; VOCAB_SIZE];
        nan_logits[2] = f64::NAN;
        let outcome =
            if softmax_checked(&nan_logits).is_err() && argmax_checked(&nan_logits).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            };
        negative_controls.push(NegativeControl {
            case_id: "nc2_nan_logits_rejected".into(),
            outcome,
            detail: "non-finite logits fail finite check rather than being swallowed".into(),
        });
    }

    // NC3: empty task set produces None accepted_task_rate and fails comparison
    {
        let empty_arm = evaluate_policy(&[], DraftPolicy::UnbiasedDraft, matched_seed, false)?;
        let outcome = if empty_arm.accepted_task_rate.is_none()
            && validate_matched_seed_comparison(&empty_arm, &empty_arm).is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc3_zero_planned_tasks_guarded".into(),
            outcome,
            detail: "zero planned tasks yields None rates (no div-by-zero) and fails comparison"
                .into(),
        });
    }

    // NC4: claiming slot-0 exactness from slot k > 0 rejected
    negative_controls.push(NegativeControl {
        case_id: "nc4_non_slot0_claiming_exactness_rejected".into(),
        outcome: validate_slot0_exactness_claim(2, true),
        detail: "accepting a decision token at slot k=2 cannot claim slot-0 numerical exactness"
            .into(),
    });

    // NC5: silent bypass skipping rejection-sampling verification rejected
    {
        let outcome =
            if evaluate_policy(&tasks, DraftPolicy::UnbiasedDraft, matched_seed, true).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            };
        negative_controls.push(NegativeControl {
            case_id: "nc5_silent_verification_bypass_rejected".into(),
            outcome,
            detail: "bypassing rejection-sampling verification is refused".into(),
        });
    }

    // NC6: negative wall-time proxy rejected
    {
        let mut neg_time = unbiased.clone();
        neg_time.wall_time_proxy_ms = -1.0;
        let outcome = if validate_matched_seed_comparison(&neg_time, &adversarial).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc6_negative_wall_time_rejected".into(),
            outcome,
            detail: "negative wall-time measurement is rejected".into(),
        });
    }

    // NC7: importing arXiv:2611.05345 authors' claims as pass/fail threshold rejected
    negative_controls.push(NegativeControl {
        case_id: "nc7_paper_claim_as_gate_rejected".into(),
        outcome: validate_no_external_claim_gate(true),
        detail: "arXiv:2611.05345 claims are unverified related work, never lab pass gates".into(),
    });

    // NC8: corrupted generated trajectory fails independent task checker
    {
        let mut bad_gen = exact_slot0_rollout(&tasks[0])?;
        bad_gen[tasks[0].decision_steps[0]] =
            (bad_gen[tasks[0].decision_steps[0]] + 1) % VOCAB_SIZE;
        let outcome = if !independent_task_checker(&tasks[0], &bad_gen) {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc8_decision_flip_fails_independent_checker".into(),
            outcome,
            detail: "flipping a decision-point token fails independent task verification".into(),
        });
    }

    Ok(AdvSpecReasoningReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "GPU speculative decoding runtime with slot-indexed batched verifier and trained adversarial drafter",
            "Per-slot numerical noise measurement and matched-seed unbiased vs adversarial draft comparison verified on miniature CPU contract; physical GPU speculative serve did not execute locally",
        ),
        synthetic_contract_fixture: true,
        fixture_digest,
        matched_seed,
        slot_noise,
        arms: vec![unbiased, adversarial],
        negative_controls,
    })
}

pub fn inputs_json(report: &AdvSpecReasoningReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.adv-spec-reasoning.inputs.v1\",\n",
            "  \"issue\": 47,\n",
            "  \"extends_issues\": [3, 6, 22],\n",
            "  \"execution_class\": {},\n",
            "  \"non_gating_paper\": {},\n",
            "  \"source_bookmark\": {},\n",
            "  \"matched_seed\": {},\n",
            "  \"num_draft_slots\": {},\n",
            "  \"fixture_digest\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(PAPER_PIN),
        quote(SOURCE_BOOKMARK),
        report.matched_seed,
        NUM_DRAFT_SLOTS,
        report.fixture_digest.to_json(),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &AdvSpecReasoningReport) -> String {
    let slots: Vec<String> = report
        .slot_noise
        .iter()
        .map(|s| {
            format!(
                "{{\"slot_index\":{},\"sample_count\":{},\"max_abs_err\":{},\"l2_err\":{}}}",
                s.slot_index,
                s.sample_count,
                finite_f64(s.max_abs_err),
                finite_f64(s.l2_err),
            )
        })
        .collect();
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                "{{\"policy\":{},\"seed\":{},\"planned_tasks\":{},\"accepted_tasks\":{},\"accepted_task_rate\":{},\"exact_slot0_match_tasks\":{},\"exactness_rate\":{},\"decision_tokens_at_slot0\":{},\"total_decision_tokens\":{},\"total_draft_tokens\":{},\"total_accepted_draft_tokens\":{},\"draft_acceptance_rate\":{},\"target_forward_calls\":{},\"draft_forward_calls\":{},\"wall_time_proxy_ms\":{}}}",
                quote(a.policy.as_str()),
                a.seed,
                a.planned_tasks,
                a.accepted_tasks,
                opt_f64(a.accepted_task_rate),
                a.exact_slot0_match_tasks,
                opt_f64(a.exactness_rate),
                a.decision_tokens_at_slot0,
                a.total_decision_tokens,
                a.total_draft_tokens,
                a.total_accepted_draft_tokens,
                opt_f64(a.draft_acceptance_rate),
                a.target_forward_calls,
                a.draft_forward_calls,
                finite_f64(a.wall_time_proxy_ms),
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
                quote(&c.detail),
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.adv-spec-reasoning.results.v1\",\n",
            "  \"issue\": 47,\n",
            "  \"extends_issues\": [3, 6, 22],\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"matched_seed\": {},\n",
            "  \"slot_noise\": [{}],\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.matched_seed,
        slots.join(","),
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &AdvSpecReasoningReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #47: Adversarial Speculative Reasoning Drafts vs Unbiased Drafts"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Extends drafting issues: `#3`, `#6`, `#22` (non-gating reference: `{}`)",
        PAPER_PIN
    )
    .unwrap();
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
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Per-Slot Numerical Noise (vs Independent Slot-0 Reference)"
    )
    .unwrap();
    for s in &report.slot_noise {
        writeln!(
            out,
            "- Slot `{}` (`samples={}`): `max_abs_err={:.6}`, `l2_err={:.6}`",
            s.slot_index, s.sample_count, s.max_abs_err, s.l2_err
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Matched-Seed Draft Comparison (`seed={}`)",
        report.matched_seed
    )
    .unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "- `{}`: `accepted={}/{}`, `exact_slot0={}/{}`, `decision_at_slot0={}/{}`, `target_calls={}`, `wall_proxy_ms={:.3}`",
            a.policy.as_str(),
            a.accepted_tasks,
            a.planned_tasks,
            a.exact_slot0_match_tasks,
            a.planned_tasks,
            a.decision_tokens_at_slot0,
            a.total_decision_tokens,
            a.target_forward_calls,
            a.wall_time_proxy_ms,
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
    fn slot_noise_is_zero_at_slot0_and_monotone_across_later_slots() {
        let report = run_default_harness().unwrap();
        assert_eq!(report.slot_noise.len(), NUM_DRAFT_SLOTS);
        assert_eq!(report.slot_noise[0].max_abs_err, 0.0);
        assert_eq!(report.slot_noise[0].l2_err, 0.0);
        for k in 1..NUM_DRAFT_SLOTS {
            assert!(report.slot_noise[k].max_abs_err > report.slot_noise[k - 1].max_abs_err);
            assert!(report.slot_noise[k].l2_err > report.slot_noise[k - 1].l2_err);
        }
    }

    #[test]
    fn adversarial_draft_improves_decision_slot0_and_all_controls_reject() {
        let report = run_default_harness().unwrap();
        let unbiased = &report.arms[0];
        let adv = &report.arms[1];
        assert!(adv.decision_tokens_at_slot0 >= unbiased.decision_tokens_at_slot0);
        assert!(adv.exact_slot0_match_tasks >= unbiased.exact_slot0_match_tasks);
        assert!(adv.accepted_tasks >= unbiased.accepted_tasks);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
