//! Issue #25: Qualify heterogeneous rollout serving and versioned HTTP LoRA updates on a bounded
//! two-backend pool.
//!
//! Pinned upstream mechanisms (reviewed 2026-11-14):
//! - Miles `#3690` (`radixark/miles@c4d5aa11a637debd0af63c3d0d7bbbf01831a3c3`, `http_lora.py`,
//!   `tensors` mode; `path` mode separately scoped).
//! - SGLang `#48241` (`sglang-miles` `f8d0201`: single serialized CPU-tensor payload per engine).
//! - SGLang `#47135` (adapter path upsert; generation & cache coordination remain caller duties).
//! - Router `#30` (motivates separate `systems-lab#4` dispatch policy comparisons).
//!
//! Preserves `#11`'s sequential `A -> B -> A` baseline, `#5`'s cache compatibility, `#7`'s runner
//! identities, and upstream exclusions (multi-LoRA, disaggregated prefill/decode). Explicitly keeps
//! causal token-rollout qualification distinct from RSI `#87` stateful recurrence qualification.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const MILES_COMMIT: &str = "c4d5aa11a637debd0af63c3d0d7bbbf01831a3c3";
pub const SGLANG_MILES_REV: &str = "f8d0201";
pub const REVIEW_DATE: &str = "2026-11-14";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaHealth {
    Ready,
    RejectedInstallOldIntact,
    UncertainIneligible,
    Unavailable,
}

impl ReplicaHealth {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::RejectedInstallOldIntact => "rejected_install_old_intact",
            Self::UncertainIneligible => "uncertain_ineligible",
            Self::Unavailable => "unavailable",
        }
    }

    pub const fn is_eligible(self) -> bool {
        matches!(self, Self::Ready)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrozenPolicyQualResult {
    pub backend_id: String,
    pub max_abs_logprob_residual: f64,
    pub exact_full_vocab_kl: f64,
    pub sampled_token_logprob_diff: f64,
    pub sampled_is_not_full_kl_verified: bool,
    pub tokenizer_parity_ok: bool,
    pub outcome: OutcomeStatus,
}

/// Compare candidate backend logits against independent reference scorer on identical teacher-forced
/// tokens, computing both exact full-vocabulary KL and sampled-token logprob residual separately.
#[allow(clippy::too_many_arguments)]
pub fn qualify_frozen_policy(
    backend_id: &str,
    ref_logits_seq: &[Vec<f64>],
    cand_logits_seq: &[Vec<f64>],
    sampled_tokens: &[usize],
    ref_token_ids: &[usize],
    cand_token_ids: &[usize],
    logprob_tol: f64,
    kl_tol: f64,
) -> Result<FrozenPolicyQualResult, String> {
    if ref_logits_seq.is_empty()
        || ref_logits_seq.len() != cand_logits_seq.len()
        || ref_logits_seq.len() != sampled_tokens.len()
    {
        return Err("sequence length mismatch in qualify_frozen_policy".into());
    }
    if !logprob_tol.is_finite() || logprob_tol < 0.0 || !kl_tol.is_finite() || kl_tol < 0.0 {
        return Err("logprob_tol and kl_tol must be finite and non-negative".into());
    }
    let tokenizer_parity_ok = ref_token_ids == cand_token_ids;
    let mut max_lp_res = 0.0_f64;
    let mut total_kl = 0.0_f64;
    let mut total_sampled_diff = 0.0_f64;

    for t in 0..ref_logits_seq.len() {
        if ref_logits_seq[t].len() != cand_logits_seq[t].len() {
            return Err("vocab size mismatch in qualify_frozen_policy".into());
        }
        let p = softmax_stable(&ref_logits_seq[t])?;
        let q = softmax_stable(&cand_logits_seq[t])?;
        let tok = sampled_tokens[t];
        if tok >= p.len() {
            return Err("sampled token out of range".into());
        }
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi <= 0.0 || qi <= 0.0 {
                return Err("zero probability in log-prob residual".into());
            }
            let diff = (pi.ln() - qi.ln()).abs();
            if diff > max_lp_res {
                max_lp_res = diff;
            }
            total_kl += pi * (pi / qi).ln();
        }
        total_sampled_diff += (p[tok].ln() - q[tok].ln()).abs();
    }
    if !total_kl.is_finite() || !total_sampled_diff.is_finite() {
        return Err("non-finite KL or sampled residual in qualify_frozen_policy".into());
    }
    let mean_kl_raw = total_kl / (ref_logits_seq.len() as f64);
    let mean_kl = if mean_kl_raw < 0.0 { 0.0 } else { mean_kl_raw };
    let mean_sampled_diff = total_sampled_diff / (ref_logits_seq.len() as f64);
    let sampled_differs_from_kl = (mean_kl - mean_sampled_diff).abs() > 1e-15 || mean_kl == 0.0;
    let outcome = if tokenizer_parity_ok && max_lp_res <= logprob_tol && mean_kl <= kl_tol {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    };
    Ok(FrozenPolicyQualResult {
        backend_id: backend_id.into(),
        max_abs_logprob_residual: max_lp_res,
        exact_full_vocab_kl: mean_kl,
        sampled_token_logprob_diff: mean_sampled_diff,
        sampled_is_not_full_kl_verified: sampled_differs_from_kl,
        tokenizer_parity_ok,
        outcome,
    })
}

/// Serialized `http-lora` `tensors`-mode update payload.
#[derive(Debug, Clone)]
pub struct HttpLoraPayload {
    pub base_model_digest: String,
    pub adapter_version: String,
    pub deployment_generation: u64,
    pub delivery_id: String,
    pub multi_lora_requested: bool,
    pub disaggregated_prefill_decode: bool,
    pub payload_bytes: Vec<u8>,
    pub declared_sha256: String,
    /// Expected deterministic known-answer probe signature for this adapter version.
    pub expected_probe_sig: String,
}

impl HttpLoraPayload {
    pub fn valid(base: &str, version: &str, gen: u64, delivery_id: &str, probe_sig: &str) -> Self {
        let payload_bytes = format!("tensors:{base}:{version}:gen={gen}").into_bytes();
        let declared_sha256 = DigestRef::sha256(&payload_bytes).hex;
        Self {
            base_model_digest: base.into(),
            adapter_version: version.into(),
            deployment_generation: gen,
            delivery_id: delivery_id.into(),
            multi_lora_requested: false,
            disaggregated_prefill_decode: false,
            payload_bytes,
            declared_sha256,
            expected_probe_sig: probe_sig.into(),
        }
    }
}

/// Simulated disposable rollout replica tracking claimed vs probe-observed adapter identity,
/// delivery deduplication, activation counts, and failure/reconciliation states.
#[derive(Debug, Clone)]
pub struct RolloutReplica {
    pub replica_id: String,
    pub base_model_digest: String,
    pub claimed_version: String,
    pub observed_probe_sig: String,
    pub deployment_generation: u64,
    pub health: ReplicaHealth,
    pub in_flight_episode: bool,
    pub seen_deliveries: Vec<String>,
    pub activation_count: usize,
    pub delivery_count: usize,
    pub max_retries: usize,
}

impl RolloutReplica {
    pub fn new(
        replica_id: &str,
        base_model_digest: &str,
        init_version: &str,
        init_probe: &str,
    ) -> Self {
        Self {
            replica_id: replica_id.into(),
            base_model_digest: base_model_digest.into(),
            claimed_version: init_version.into(),
            observed_probe_sig: init_probe.into(),
            deployment_generation: 0,
            health: ReplicaHealth::Ready,
            in_flight_episode: false,
            seen_deliveries: Vec::new(),
            activation_count: 0,
            delivery_count: 0,
            max_retries: 2,
        }
    }

    /// Apply a versioned `http-lora` update with explicit fault injection hooks.
    pub fn apply_http_lora(
        &mut self,
        payload: &HttpLoraPayload,
        simulate_unavailable: bool,
        simulate_interrupted_activation: bool,
        simulate_echo_spoof: bool,
        simulate_restore_failure: bool,
    ) -> Result<OutcomeStatus, String> {
        if payload.multi_lora_requested || payload.disaggregated_prefill_decode {
            return Ok(OutcomeStatus::Unsupported);
        }
        if self.in_flight_episode {
            return Err(
                "cannot switch policy mid-episode without declared pause/resume semantics".into(),
            );
        }
        if simulate_unavailable {
            self.health = ReplicaHealth::Unavailable;
            return Ok(OutcomeStatus::InternalFailure);
        }
        if payload.base_model_digest != self.base_model_digest {
            self.health = ReplicaHealth::RejectedInstallOldIntact;
            return Ok(OutcomeStatus::Invalid);
        }
        let actual_sha = DigestRef::sha256(&payload.payload_bytes).hex;
        if actual_sha != payload.declared_sha256 {
            self.health = ReplicaHealth::RejectedInstallOldIntact;
            return Ok(OutcomeStatus::Invalid);
        }
        // Stale generation check (strictly monotonic unless identical delivery retry on a Ready replica).
        let is_duplicate = self.seen_deliveries.contains(&payload.delivery_id);
        if is_duplicate {
            self.delivery_count += 1;
            if !self.health.is_eligible() {
                // Never make an UncertainIneligible or Unavailable replica look Accepted/Ready on duplicate retry!
                return Ok(OutcomeStatus::Invalid);
            }
            if self.delivery_count > self.activation_count + self.max_retries + 2 {
                return Ok(OutcomeStatus::Exhausted);
            }
            // Repeated delivery must NOT trigger repeated activation!
            return Ok(OutcomeStatus::Accepted);
        }
        if payload.deployment_generation <= self.deployment_generation {
            self.health = ReplicaHealth::RejectedInstallOldIntact;
            return Ok(OutcomeStatus::Invalid);
        }
        self.delivery_count += 1;
        self.seen_deliveries.push(payload.delivery_id.clone());

        if simulate_interrupted_activation || simulate_restore_failure {
            // Interrupted activation / failed restore CANNOT assume rollback -> uncertain & ineligible!
            self.health = ReplicaHealth::UncertainIneligible;
            return Ok(OutcomeStatus::InternalFailure);
        }

        self.claimed_version = payload.adapter_version.clone();
        self.deployment_generation = payload.deployment_generation;
        self.activation_count += 1;

        // Independent known-answer probe: echoed version string alone is insufficient!
        if !simulate_echo_spoof {
            self.observed_probe_sig = payload.expected_probe_sig.clone();
        }
        if self.observed_probe_sig != payload.expected_probe_sig {
            self.health = ReplicaHealth::UncertainIneligible;
            return Ok(OutcomeStatus::Invalid);
        }
        self.health = ReplicaHealth::Ready;
        Ok(OutcomeStatus::Accepted)
    }

    /// Validate a completed rollout response against current deployment generation and eligibility.
    pub fn validate_rollout_response(
        &self,
        response_gen: u64,
        mixed_versions: bool,
    ) -> OutcomeStatus {
        if !self.health.is_eligible() {
            return OutcomeStatus::Invalid;
        }
        if mixed_versions || response_gen != self.deployment_generation {
            return OutcomeStatus::Invalid;
        }
        OutcomeStatus::Accepted
    }
}

#[derive(Debug, Clone)]
pub struct ActivationStageRecord {
    pub transition: String,
    pub target_version: String,
    pub deployment_generation: u64,
    pub payload_bytes: usize,
    pub probe_verified: bool,
    pub replica_a_health: ReplicaHealth,
    pub replica_b_health: ReplicaHealth,
}

#[derive(Debug, Clone)]
pub struct RolloutEpisodeRecord {
    pub episode_id: String,
    pub backend_id: String,
    pub policy_generation: u64,
    pub protocol_status: OutcomeStatus,
    pub independent_task_success: bool,
    pub learner_accepted: bool,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct PoolSliceSummary {
    pub pool_name: String,
    pub node_count: usize,
    pub planned_episodes: usize,
    pub protocol_valid_rollouts: usize,
    pub valid_low_reward_retained: usize,
    pub independent_task_successes: usize,
    pub learner_accepted_rollouts: usize,
    pub failed_or_rejected_episodes: usize,
    pub total_cost_ms: f64,
}

pub fn reduce_pool(
    pool_name: &str,
    node_count: usize,
    eps: &[RolloutEpisodeRecord],
) -> Result<PoolSliceSummary, String> {
    let mut valid = 0usize;
    let mut low_reward_valid = 0usize;
    let mut task_ok = 0usize;
    let mut learner_ok = 0usize;
    let mut failed = 0usize;
    let mut total_ms = 0.0_f64;
    for e in eps {
        if !e.latency_ms.is_finite() || e.latency_ms < 0.0 {
            return Err(format!(
                "invalid non-finite or negative latency on episode `{}`",
                e.episode_id
            ));
        }
        total_ms += e.latency_ms;
        if e.protocol_status == OutcomeStatus::Accepted {
            valid += 1;
            if !e.independent_task_success {
                low_reward_valid += 1;
            }
        } else {
            failed += 1;
        }
        if e.independent_task_success {
            task_ok += 1;
        }
        if e.learner_accepted {
            learner_ok += 1;
        }
    }
    Ok(PoolSliceSummary {
        pool_name: pool_name.into(),
        node_count,
        planned_episodes: eps.len(),
        protocol_valid_rollouts: valid,
        valid_low_reward_retained: low_reward_valid,
        independent_task_successes: task_ok,
        learner_accepted_rollouts: learner_ok,
        failed_or_rejected_episodes: failed,
        total_cost_ms: total_ms,
    })
}

#[derive(Debug, Clone)]
pub struct FaultControlRecord {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub replica_eligible_after: bool,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct HeteroRolloutReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub frozen_policy_checks: Vec<FrozenPolicyQualResult>,
    pub activation_transitions: Vec<ActivationStageRecord>,
    pub pool_summaries: Vec<PoolSliceSummary>,
    pub fault_controls: Vec<FaultControlRecord>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<HeteroRolloutReport, String> {
    // 1. Frozen-policy numerical qualification (Backend A vs Reference, Backend B vs Reference).
    let ref_logits: Vec<Vec<f64>> = (0..6)
        .map(|t| (0..8).map(|v| det_weight(2500 + t, v, 1.5)).collect())
        .collect();
    let cand_a_logits: Vec<Vec<f64>> = ref_logits
        .iter()
        .enumerate()
        .map(|(t, row)| {
            row.iter()
                .enumerate()
                .map(|(v, &x)| x + det_weight(2510 + t, v, 1e-4))
                .collect()
        })
        .collect();
    let cand_b_logits: Vec<Vec<f64>> = ref_logits
        .iter()
        .enumerate()
        .map(|(t, row)| {
            row.iter()
                .enumerate()
                .map(|(v, &x)| x + det_weight(2520 + t, v, 2e-4))
                .collect()
        })
        .collect();
    let sampled = vec![1, 3, 2, 5, 0, 4];
    let tok_ids = vec![10, 21, 32, 43, 54, 65];

    let qual_a = qualify_frozen_policy(
        "backend_a_nvidia_synthetic",
        &ref_logits,
        &cand_a_logits,
        &sampled,
        &tok_ids,
        &tok_ids,
        1e-2,
        1e-4,
    )?;
    let qual_b = qualify_frozen_policy(
        "backend_b_amd_synthetic",
        &ref_logits,
        &cand_b_logits,
        &sampled,
        &tok_ids,
        &tok_ids,
        1e-2,
        1e-4,
    )?;

    // 2. Bounded versioned HTTP LoRA transition sequence: parent(v0) -> v1 -> v2 -> restore parent(v0).
    let base_digest = DigestRef::sha256(b"small-causal-base-v1").hex;
    let mut rep_a = RolloutReplica::new("backend_a", &base_digest, "parent_v0", "probe_sig_v0");
    let mut rep_b = RolloutReplica::new("backend_b", &base_digest, "parent_v0", "probe_sig_v0");

    let mut activation_transitions = Vec::new();
    for (label, ver, gen, deliv, probe) in [
        ("parent_to_v1", "lora_v1", 1u64, "deliv-1", "probe_sig_v1"),
        ("v1_to_v2", "lora_v2", 2u64, "deliv-2", "probe_sig_v2"),
        (
            "v2_restore_parent",
            "parent_v0",
            3u64,
            "deliv-3",
            "probe_sig_v0",
        ),
    ] {
        let p = HttpLoraPayload::valid(&base_digest, ver, gen, deliv, probe);
        let oa = rep_a.apply_http_lora(&p, false, false, false, false)?;
        let ob = rep_b.apply_http_lora(&p, false, false, false, false)?;
        if oa != OutcomeStatus::Accepted || ob != OutcomeStatus::Accepted {
            return Err(format!("expected valid transition `{label}` to succeed"));
        }
        activation_transitions.push(ActivationStageRecord {
            transition: label.into(),
            target_version: ver.into(),
            deployment_generation: gen,
            payload_bytes: p.payload_bytes.len(),
            probe_verified: rep_a.observed_probe_sig == probe && rep_b.observed_probe_sig == probe,
            replica_a_health: rep_a.health,
            replica_b_health: rep_b.health,
        });
    }

    // 3. Ten executable fault & fleet-reconciliation controls!
    let mut fault_controls = Vec::new();

    // Control 1: Wrong base/adapter metadata
    let mut r_meta = rep_a.clone();
    let p_wrong_base =
        HttpLoraPayload::valid("wrong-base-digest", "lora_v4", 4, "d-wrong-base", "sig4");
    let o1 = r_meta.apply_http_lora(&p_wrong_base, false, false, false, false)?;
    fault_controls.push(FaultControlRecord {
        case_id: "wrong_base_model_metadata".into(),
        outcome: o1,
        replica_eligible_after: r_meta.health.is_eligible(),
        detail: format!("health={}", r_meta.health.as_str()),
    });

    // Control 2: Corrupt / truncated tensor payload
    let mut r_corrupt = rep_a.clone();
    let mut p_corrupt = HttpLoraPayload::valid(&base_digest, "lora_v4", 4, "d-corrupt", "sig4");
    p_corrupt.payload_bytes.truncate(5);
    let o2 = r_corrupt.apply_http_lora(&p_corrupt, false, false, false, false)?;
    fault_controls.push(FaultControlRecord {
        case_id: "corrupt_or_truncated_payload".into(),
        outcome: o2,
        replica_eligible_after: r_corrupt.health.is_eligible(),
        detail: format!("health={}", r_corrupt.health.as_str()),
    });

    // Control 3: Duplicate delivery (repeated delivery must NOT trigger repeated activation!)
    let mut r_dup = rep_a.clone();
    let act_before = r_dup.activation_count;
    let p_dup = HttpLoraPayload::valid(&base_digest, "parent_v0", 3, "deliv-3", "probe_sig_v0");
    let o3 = r_dup.apply_http_lora(&p_dup, false, false, false, false)?;
    let no_dup_activation =
        r_dup.activation_count == act_before && r_dup.delivery_count > act_before;
    fault_controls.push(FaultControlRecord {
        case_id: "duplicate_delivery_idempotent_no_reactivation".into(),
        outcome: if o3 == OutcomeStatus::Accepted && no_dup_activation {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
        replica_eligible_after: r_dup.health.is_eligible(),
        detail: format!(
            "deliveries={}, activations={}",
            r_dup.delivery_count, r_dup.activation_count
        ),
    });

    // Control 4: Stale deployment generation
    let mut r_stale = rep_a.clone();
    let p_stale = HttpLoraPayload::valid(&base_digest, "lora_v1", 1, "d-stale-gen", "probe_sig_v1");
    let o4 = r_stale.apply_http_lora(&p_stale, false, false, false, false)?;
    fault_controls.push(FaultControlRecord {
        case_id: "stale_deployment_generation".into(),
        outcome: o4,
        replica_eligible_after: r_stale.health.is_eligible(),
        detail: format!("health={}", r_stale.health.as_str()),
    });

    // Control 5: Unavailable engine during partial fleet update
    let mut r_unavail = rep_b.clone();
    let p_v4 = HttpLoraPayload::valid(&base_digest, "lora_v4", 4, "d-v4", "probe_sig_v4");
    let o5 = r_unavail.apply_http_lora(&p_v4, true, false, false, false)?;
    fault_controls.push(FaultControlRecord {
        case_id: "unavailable_engine_partial_fleet_update".into(),
        outcome: o5,
        replica_eligible_after: r_unavail.health.is_eligible(),
        detail: format!("health={}", r_unavail.health.as_str()),
    });

    // Control 6: Interrupted activation -> uncertain & ineligible (never assumed rolled back!)
    let mut r_intr = rep_b.clone();
    let o6 = r_intr.apply_http_lora(&p_v4, false, true, false, false)?;
    fault_controls.push(FaultControlRecord {
        case_id: "interrupted_activation_uncertain_ineligible".into(),
        outcome: o6,
        replica_eligible_after: r_intr.health.is_eligible(),
        detail: format!("health={}", r_intr.health.as_str()),
    });

    // Control 7: Late old-generation response after cutover
    let o7 = rep_a.validate_rollout_response(2, false);
    fault_controls.push(FaultControlRecord {
        case_id: "late_old_generation_response_rejected".into(),
        outcome: o7,
        replica_eligible_after: rep_a.health.is_eligible(),
        detail: format!(
            "response_gen=2 vs current_gen={}",
            rep_a.deployment_generation
        ),
    });

    // Control 8: Failed parent restoration leaves replica ineligible
    let mut r_rest = rep_a.clone();
    let p_rest =
        HttpLoraPayload::valid(&base_digest, "parent_v0", 5, "d-rest-fail", "probe_sig_v0");
    let o8 = r_rest.apply_http_lora(&p_rest, false, false, false, true)?;
    fault_controls.push(FaultControlRecord {
        case_id: "failed_parent_restoration_ineligible".into(),
        outcome: o8,
        replica_eligible_after: r_rest.health.is_eligible(),
        detail: format!("health={}", r_rest.health.as_str()),
    });

    // Control 9: Unsupported multi-LoRA AND disaggregated prefill/decode BOTH refused
    let mut r_unsup = rep_a.clone();
    let mut p_ml = HttpLoraPayload::valid(&base_digest, "lora_multi", 6, "d-multi", "sigm");
    p_ml.multi_lora_requested = true;
    let o9a = r_unsup.apply_http_lora(&p_ml, false, false, false, false)?;
    let mut p_disagg = HttpLoraPayload::valid(&base_digest, "lora_disagg", 7, "d-disagg", "sigd");
    p_disagg.disaggregated_prefill_decode = true;
    let o9b = r_unsup.apply_http_lora(&p_disagg, false, false, false, false)?;
    fault_controls.push(FaultControlRecord {
        case_id: "unsupported_multi_lora_and_disaggregation".into(),
        outcome: if o9a == OutcomeStatus::Unsupported && o9b == OutcomeStatus::Unsupported {
            OutcomeStatus::Unsupported
        } else {
            OutcomeStatus::Accepted
        },
        replica_eligible_after: r_unsup.health.is_eligible(),
        detail: format!(
            "multi_lora={}, disaggregated_prefill_decode={}",
            o9a.as_str(),
            o9b.as_str()
        ),
    });

    // Control 10: Echoed version string spoof (probe disagrees) + token/logprob misalignment
    let mut r_spoof = rep_a.clone();
    let o10a = r_spoof.apply_http_lora(&p_v4, false, false, true, false)?;
    let bad_tok_qual = qualify_frozen_policy(
        "misaligned_tok_backend",
        &ref_logits,
        &cand_a_logits,
        &sampled,
        &tok_ids,
        &[10, 21, 999, 43, 54, 65],
        1e-2,
        1e-4,
    )?;
    fault_controls.push(FaultControlRecord {
        case_id: "echoed_version_spoof_and_token_misalignment".into(),
        outcome: if o10a == OutcomeStatus::Invalid && bad_tok_qual.outcome == OutcomeStatus::Invalid
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        replica_eligible_after: r_spoof.health.is_eligible(),
        detail: format!(
            "probe_spoof_outcome={}, tok_mismatch_outcome={}",
            o10a.as_str(),
            bad_tok_qual.outcome.as_str()
        ),
    });

    // 4. Bounded mixed-pool reduction (retaining valid low-reward RL trajectories + failures!).
    let eps_a = vec![
        RolloutEpisodeRecord {
            episode_id: "e1".into(),
            backend_id: "backend_a".into(),
            policy_generation: 3,
            protocol_status: OutcomeStatus::Accepted,
            independent_task_success: true,
            learner_accepted: true,
            latency_ms: 120.0,
        },
        RolloutEpisodeRecord {
            episode_id: "e2".into(),
            backend_id: "backend_a".into(),
            policy_generation: 3,
            protocol_status: OutcomeStatus::Accepted,
            independent_task_success: false,
            learner_accepted: true,
            latency_ms: 115.0,
        },
        RolloutEpisodeRecord {
            episode_id: "e3".into(),
            backend_id: "backend_a".into(),
            policy_generation: 3,
            protocol_status: OutcomeStatus::Timeout,
            independent_task_success: false,
            learner_accepted: false,
            latency_ms: 500.0,
        },
    ];
    let eps_b = vec![
        RolloutEpisodeRecord {
            episode_id: "e1".into(),
            backend_id: "backend_b".into(),
            policy_generation: 3,
            protocol_status: OutcomeStatus::Accepted,
            independent_task_success: true,
            learner_accepted: true,
            latency_ms: 145.0,
        },
        RolloutEpisodeRecord {
            episode_id: "e2".into(),
            backend_id: "backend_b".into(),
            policy_generation: 3,
            protocol_status: OutcomeStatus::Accepted,
            independent_task_success: false,
            learner_accepted: true,
            latency_ms: 140.0,
        },
        RolloutEpisodeRecord {
            episode_id: "e3".into(),
            backend_id: "backend_b".into(),
            policy_generation: 3,
            protocol_status: OutcomeStatus::Cancelled,
            independent_task_success: false,
            learner_accepted: false,
            latency_ms: 60.0,
        },
    ];
    let mut eps_mixed = Vec::new();
    eps_mixed.extend_from_slice(&eps_a);
    eps_mixed.extend_from_slice(&eps_b);

    let pool_summaries = vec![
        reduce_pool("endpoint_a_only", 1, &eps_a)?,
        reduce_pool("endpoint_b_only", 1, &eps_b)?,
        reduce_pool("heterogeneous_pool_a_plus_b", 2, &eps_mixed)?,
    ];

    Ok(HeteroRolloutReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical_two_backend_nvidia_amd_rollout_pool",
            "Model-free frozen-policy logprob/KL, HTTP LoRA transition/reconciliation, and mixed-pool reducer verified locally; physical two-backend serving remains unqualified",
        ),
        frozen_policy_checks: vec![qual_a, qual_b],
        activation_transitions,
        pool_summaries,
        fault_controls,
        decision: "INCONCLUSIVE_PHYSICAL_TWO_BACKEND_QUALIFICATION_REQUIRED".into(),
    })
}

pub fn inputs_json(report: &HeteroRolloutReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.hetero-rollout.inputs.v1\",\n",
            "  \"issue\": 25,\n",
            "  \"upstream_pins\": {{\"miles_pr3690_commit\":{},\"sglang_miles_rev\":{},\"review_date\":{},\"transport_mode\":\"tensors\"}},\n",
            "  \"preserves_issue_11_sequential_baseline\": true,\n",
            "  \"substitutes_for_rsi_87_stateful_qualification\": false,\n",
            "  \"execution_class\": {}\n",
            "}}\n"
        ),
        quote(MILES_COMMIT),
        quote(SGLANG_MILES_REV),
        quote(REVIEW_DATE),
        quote(report.execution_class.as_str()),
    )
}

pub fn results_json(report: &HeteroRolloutReport) -> String {
    let qual: Vec<String> = report
        .frozen_policy_checks
        .iter()
        .map(|q| {
            format!(
                "{{\"backend_id\":{},\"max_abs_logprob_residual\":{},\"exact_full_vocab_kl\":{},\"sampled_token_logprob_diff\":{},\"sampled_is_not_full_kl_verified\":{},\"tokenizer_parity_ok\":{},\"outcome\":{}}}",
                quote(&q.backend_id),
                finite_f64(q.max_abs_logprob_residual),
                finite_f64(q.exact_full_vocab_kl),
                finite_f64(q.sampled_token_logprob_diff),
                q.sampled_is_not_full_kl_verified,
                q.tokenizer_parity_ok,
                quote(q.outcome.as_str()),
            )
        })
        .collect();
    let trans: Vec<String> = report
        .activation_transitions
        .iter()
        .map(|t| {
            format!(
                "{{\"transition\":{},\"target_version\":{},\"deployment_generation\":{},\"payload_bytes\":{},\"probe_verified\":{},\"replica_a_health\":{},\"replica_b_health\":{}}}",
                quote(&t.transition),
                quote(&t.target_version),
                t.deployment_generation,
                t.payload_bytes,
                t.probe_verified,
                quote(t.replica_a_health.as_str()),
                quote(t.replica_b_health.as_str()),
            )
        })
        .collect();
    let pools: Vec<String> = report
        .pool_summaries
        .iter()
        .map(|p| {
            format!(
                "{{\"synthetic_contract_fixture\":true,\"pool_name\":{},\"node_count\":{},\"planned_episodes\":{},\"protocol_valid_rollouts\":{},\"valid_low_reward_retained\":{},\"independent_task_successes\":{},\"learner_accepted_rollouts\":{},\"failed_or_rejected_episodes\":{},\"total_cost_ms\":{}}}",
                quote(&p.pool_name),
                p.node_count,
                p.planned_episodes,
                p.protocol_valid_rollouts,
                p.valid_low_reward_retained,
                p.independent_task_successes,
                p.learner_accepted_rollouts,
                p.failed_or_rejected_episodes,
                finite_f64(p.total_cost_ms),
            )
        })
        .collect();
    let faults: Vec<String> = report
        .fault_controls
        .iter()
        .map(|f| {
            format!(
                "{{\"case_id\":{},\"outcome\":{},\"replica_eligible_after\":{},\"detail\":{}}}",
                quote(&f.case_id),
                quote(f.outcome.as_str()),
                f.replica_eligible_after,
                quote(&f.detail),
            )
        })
        .collect();

    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.hetero-rollout.results.v1\",\n",
            "  \"issue\": 25,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"decision\": {},\n",
            "  \"two_node_is_equal_resource_speedup\": false,\n",
            "  \"establishes_learner_improvement\": false,\n",
            "  \"physical_update_stall_ms\": {},\n",
            "  \"frozen_policy_checks\": [{}],\n",
            "  \"activation_transitions\": [{}],\n",
            "  \"pool_summaries\": [{}],\n",
            "  \"fault_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        quote(&report.decision),
        opt_f64(None),
        qual.join(","),
        trans.join(","),
        pools.join(","),
        faults.join(","),
    )
}

pub fn summary_markdown(report: &HeteroRolloutReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #25: Heterogeneous Rollout Serving & Versioned HTTP LoRA Qualification"
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
    writeln!(out).unwrap();
    writeln!(out, "## Fault & fleet-reconciliation controls").unwrap();
    for f in &report.fault_controls {
        writeln!(
            out,
            "- `{}` -> `{}` (eligible_after=`{}`, {})",
            f.case_id,
            f.outcome.as_str(),
            f.replica_eligible_after,
            f.detail
        )
        .unwrap();
    }
    out
}
