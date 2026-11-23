//! Issue #29: Evaluate RustingStrata expert placement and host residency against llama.cpp.
//!
//! Pinned source-review commit: `RustingStudio/RustingStrata@a39d9e57e79d165201864f0de13304117d4e2129`
//! (kept strictly distinct from upstream 1.2.1/1.2.2 benchmark notes; custom `LICENSE.md` terms
//! recorded: commercial applications permitted, engine-resale restrictions, attribution, and
//! same-license requirements for substantial distributed portions).
//!
//! Upstream `SUPPORTED_GPUS.md` does NOT establish support for RTX 5080, Strix Halo, or DGX Spark;
//! upstream 64.6 vs 48.6 tok/s (warm) and 24.3 vs 47.5 tok/s (first 4K) notes remain strictly
//! unverified non-gating related work.
//!
//! Provides:
//! - One-factor-at-a-time MoE expert placement & host-page simulator comparing `llama.cpp` fixed
//!   placement, `STRATA_FIXED_EXPERTS`, adaptive response-derived placement, and `STRATA_COLD_KEEP`.
//! - Explicit 5-layer warm-state tracking (OS page cache, prompt/KV cache, saved routing profiles,
//!   in-process adaptive placement, kernel-compilation cache).
//! - Teacher-forced controlled continuation vs free-generation separation with routing-equivalence
//!   observability, >= 3 counterbalanced paired trials, unified-memory non-freeing guard, Linux
//!   `madvise` non-guarantee guard, and 8 executable negative controls.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const RUSTING_STRATA_COMMIT: &str = "a39d9e57e79d165201864f0de13304117d4e2129";
pub const TARGET_GGUF_NAME: &str = "flint-UD-Q5_K_XL.gguf";
pub const LICENSE_SUMMARY: &str =
    "Custom LICENSE.md: commercial apps allowed; engine-resale restriction, attribution, and same-license terms for substantial portions (evaluate externally before vendoring)";
pub const UPSTREAM_UNVERIFIED_CLAIM: &str =
    "64.6 vs 48.6 tok/s warm (41-46 alt continuation; 24.3 vs 47.5 first 4K) on RTX 3060 12GB (1.2.1 notes; non-gating)";

/// Five independent warm-state layers that must be tracked separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmStateVector {
    pub os_page_cache_state: &'static str,
    pub prompt_kv_cache_state: &'static str,
    pub saved_routing_profile_state: &'static str,
    pub in_process_adaptive_state: &'static str,
    pub kernel_compile_cache_state: &'static str,
}

impl WarmStateVector {
    pub fn validate_claimed_cold(&self) -> Result<(), String> {
        if self.saved_routing_profile_state != "reset_empty"
            || self.prompt_kv_cache_state != "reset_empty"
            || self.in_process_adaptive_state != "reset_empty"
        {
            return Err(format!(
                "mislabeled cold state: routing_profile={}, kv={}, adaptive={}",
                self.saved_routing_profile_state,
                self.prompt_kv_cache_state,
                self.in_process_adaptive_state
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementPolicy {
    LlamaCppFixed,
    StrataFixedExperts,
    StrataAdaptivePlacement,
    StrataAdaptiveWithColdKeep,
}

impl PlacementPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LlamaCppFixed => "llamacpp_fixed_baseline",
            Self::StrataFixedExperts => "strata_fixed_experts",
            Self::StrataAdaptivePlacement => "strata_adaptive_placement",
            Self::StrataAdaptiveWithColdKeep => "strata_adaptive_plus_cold_keep",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlacementSimulationResult {
    pub policy: PlacementPolicy,
    pub continuation_id: String,
    pub gpu_expert_slots: usize,
    pub expert_hits: usize,
    pub expert_misses: usize,
    pub expert_swaps: usize,
    pub hit_rate: f64,
    pub adaptation_overhead_units: f64,
    pub total_work_units: f64,
    pub routing_equivalence: &'static str,
}

/// Simulate deterministic MoE expert residency over a teacher-forced expert-activation trace.
pub fn simulate_expert_placement(
    policy: PlacementPolicy,
    continuation_id: &str,
    expert_trace: &[usize],
    gpu_slots: usize,
    router_telemetry_observed: bool,
) -> PlacementSimulationResult {
    let mut resident: Vec<usize> = (0..gpu_slots).collect();
    let max_exp = expert_trace
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(gpu_slots);
    let mut freq = vec![0usize; max_exp + 1];
    let mut hits = 0usize;
    let mut misses = 0usize;
    let mut swaps = 0usize;
    let mut adapt_overhead = 0.0_f64;

    for &exp in expert_trace {
        freq[exp] += 1;
        if resident.contains(&exp) {
            hits += 1;
        } else {
            misses += 1;
            if matches!(
                policy,
                PlacementPolicy::StrataAdaptivePlacement
                    | PlacementPolicy::StrataAdaptiveWithColdKeep
            ) && !resident.is_empty()
            {
                // Replace lowest-frequency resident expert when adaptive placement is enabled.
                let mut evict_pos = 0usize;
                for i in 1..resident.len() {
                    if freq[resident[i]] < freq[resident[evict_pos]] {
                        evict_pos = i;
                    }
                }
                resident[evict_pos] = exp;
                swaps += 1;
                adapt_overhead += 1.5;
            }
        }
    }
    let miss_cost = if policy == PlacementPolicy::StrataAdaptiveWithColdKeep {
        3.2 // cold-keep hint may reduce host-side re-read latency when verified, charged honestly
    } else {
        4.0
    };
    let total_work = (hits as f64) * 1.0 + (misses as f64) * miss_cost + adapt_overhead;
    let hit_rate = if expert_trace.is_empty() {
        0.0
    } else {
        (hits as f64) / (expert_trace.len() as f64)
    };
    PlacementSimulationResult {
        policy,
        continuation_id: continuation_id.into(),
        gpu_expert_slots: gpu_slots,
        expert_hits: hits,
        expert_misses: misses,
        expert_swaps: swaps,
        hit_rate,
        adaptation_overhead_units: adapt_overhead,
        total_work_units: total_work,
        routing_equivalence: if router_telemetry_observed {
            "observed_matched"
        } else {
            "unverified_router_telemetry_absent"
        },
    }
}

#[derive(Debug, Clone)]
pub struct PairedEngineTrial {
    pub trial_id: String,
    pub order: &'static str, // "A_then_B" or "B_then_A"
    pub gguf_digest: String,
    pub token_history_digest: String,
    pub engine_a_status: OutcomeStatus,
    pub engine_b_status: OutcomeStatus,
    pub engine_a_ms: f64,
    pub engine_b_ms: f64,
}

#[derive(Debug, Clone)]
pub struct OfflinePairedReduction {
    pub planned_pairs: usize,
    pub both_accepted_pairs: usize,
    pub failed_or_excluded_pairs: usize,
    pub counterbalanced: bool,
    pub mean_delta_b_minus_a_ms: Option<f64>,
}

/// Offline paired reducer rejecting mismatched GGUF/token digests, duplicate trial IDs, and
/// preserving failed/OOM/timeout trials in denominators!
pub fn reduce_paired_trials(
    expected_gguf_digest: &str,
    expected_tokens_digest: &str,
    trials: &[PairedEngineTrial],
) -> Result<OfflinePairedReduction, String> {
    if trials.len() < 3 {
        return Err("at least 3 paired repetitions are required".into());
    }
    let mut seen = BTreeSet::new();
    let mut has_ab = false;
    let mut has_ba = false;
    let mut deltas = Vec::new();
    let mut failed = 0usize;

    for t in trials {
        if !seen.insert(t.trial_id.as_str()) {
            return Err(format!("duplicate trial_id `{}` detected", t.trial_id));
        }
        if t.gguf_digest != expected_gguf_digest {
            return Err(format!("mismatched GGUF digest on trial `{}`", t.trial_id));
        }
        if t.token_history_digest != expected_tokens_digest {
            return Err(format!(
                "mismatched token history on trial `{}`",
                t.trial_id
            ));
        }
        if !t.engine_a_ms.is_finite()
            || t.engine_a_ms < 0.0
            || !t.engine_b_ms.is_finite()
            || t.engine_b_ms < 0.0
        {
            return Err(format!(
                "invalid non-finite or negative latency on trial `{}`",
                t.trial_id
            ));
        }
        if t.order == "A_then_B" {
            has_ab = true;
        } else if t.order == "B_then_A" {
            has_ba = true;
        } else {
            return Err(format!(
                "invalid trial order `{}` on trial `{}`",
                t.order, t.trial_id
            ));
        }
        if t.engine_a_status == OutcomeStatus::Accepted
            && t.engine_b_status == OutcomeStatus::Accepted
        {
            deltas.push(t.engine_b_ms - t.engine_a_ms);
        } else {
            failed += 1;
        }
    }
    let mean_delta = if deltas.is_empty() {
        None
    } else {
        Some(deltas.iter().sum::<f64>() / (deltas.len() as f64))
    };
    Ok(OfflinePairedReduction {
        planned_pairs: trials.len(),
        both_accepted_pairs: deltas.len(),
        failed_or_excluded_pairs: failed,
        counterbalanced: has_ab && has_ba,
        mean_delta_b_minus_a_ms: mean_delta,
    })
}

/// Memory & license claim validators enforcing Linux madvise, unified-memory, and custom license rules.
#[derive(Debug, Clone)]
pub struct HostMemoryClaim {
    pub claims_madvise_guarantees_reclamation: bool,
    pub unified_memory_host: bool,
    pub claims_cpu_gpu_swap_frees_physical_capacity: bool,
}

pub fn validate_host_memory_claim(claim: &HostMemoryClaim) -> Result<(), String> {
    if claim.claims_madvise_guarantees_reclamation {
        return Err(
            "Linux page-advice (madvise) is not evidence of immediate physical reclamation".into(),
        );
    }
    if claim.unified_memory_host && claim.claims_cpu_gpu_swap_frees_physical_capacity {
        return Err("on shared-memory hardware, moving an expert between CPU and GPU does not free physical capacity".into());
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct VendoringLicenseClaim {
    pub vendoring_requested: bool,
    pub has_recorded_compatibility_decision: bool,
    pub has_engine_resale_restriction: bool,
}

pub fn validate_vendoring_license_gate(claim: &VendoringLicenseClaim) -> OutcomeStatus {
    if claim.vendoring_requested
        && claim.has_engine_resale_restriction
        && !claim.has_recorded_compatibility_decision
    {
        OutcomeStatus::Unsupported
    } else {
        OutcomeStatus::Accepted
    }
}

/// Independent held-out free-generation coding/tool-use task checker.
pub fn check_free_gen_task(expected_signature: &str, generated_output: &str) -> OutcomeStatus {
    if generated_output.trim() == expected_signature {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ExpertPlacementReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub gguf_digest: DigestRef,
    pub warm_state_first_req: WarmStateVector,
    pub simulations: Vec<PlacementSimulationResult>,
    pub paired_reduction: OfflinePairedReduction,
    pub negative_controls: Vec<NegativeControl>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<ExpertPlacementReport, String> {
    let gguf_digest = DigestRef::sha256(b"synthetic-flint-UD-Q5_K_XL.gguf-bytes-v1");
    let tok_digest = DigestRef::sha256(b"frozen-teacher-forced-code-continuation-v1");

    let warm_state_first_req = WarmStateVector {
        os_page_cache_state: "unknown_no_host_wide_drop",
        prompt_kv_cache_state: "reset_empty",
        saved_routing_profile_state: "reset_empty",
        in_process_adaptive_state: "reset_empty",
        kernel_compile_cache_state: "warm_pinned_build",
    };
    warm_state_first_req.validate_claimed_cold()?;

    // Favorable (locality-heavy) AND unfavorable (thrashing) continuations both preserved!
    let favorable_trace = vec![4, 5, 4, 5, 6, 4, 5, 4, 5, 6, 4, 5, 4, 5, 6, 4];
    let unfavorable_trace = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];

    let mut simulations = Vec::new();
    for policy in [
        PlacementPolicy::LlamaCppFixed,
        PlacementPolicy::StrataFixedExperts,
        PlacementPolicy::StrataAdaptivePlacement,
        PlacementPolicy::StrataAdaptiveWithColdKeep,
    ] {
        simulations.push(simulate_expert_placement(
            policy,
            "favorable_locality_continuation",
            &favorable_trace,
            3,
            true,
        ));
        simulations.push(simulate_expert_placement(
            policy,
            "unfavorable_thrash_continuation",
            &unfavorable_trace,
            3,
            false,
        ));
    }

    // Counterbalanced paired trials (4 planned: 3 accepted + 1 retained timeout failure!).
    let trials = vec![
        PairedEngineTrial {
            trial_id: "rep-1".into(),
            order: "A_then_B",
            gguf_digest: gguf_digest.hex.clone(),
            token_history_digest: tok_digest.hex.clone(),
            engine_a_status: OutcomeStatus::Accepted,
            engine_b_status: OutcomeStatus::Accepted,
            engine_a_ms: 820.0,
            engine_b_ms: 760.0,
        },
        PairedEngineTrial {
            trial_id: "rep-2".into(),
            order: "B_then_A",
            gguf_digest: gguf_digest.hex.clone(),
            token_history_digest: tok_digest.hex.clone(),
            engine_a_status: OutcomeStatus::Accepted,
            engine_b_status: OutcomeStatus::Accepted,
            engine_a_ms: 815.0,
            engine_b_ms: 775.0,
        },
        PairedEngineTrial {
            trial_id: "rep-3".into(),
            order: "A_then_B",
            gguf_digest: gguf_digest.hex.clone(),
            token_history_digest: tok_digest.hex.clone(),
            engine_a_status: OutcomeStatus::Accepted,
            engine_b_status: OutcomeStatus::Accepted,
            engine_a_ms: 830.0,
            engine_b_ms: 890.0,
        }, // unfavorable continuation included!
        PairedEngineTrial {
            trial_id: "rep-4".into(),
            order: "B_then_A",
            gguf_digest: gguf_digest.hex.clone(),
            token_history_digest: tok_digest.hex.clone(),
            engine_a_status: OutcomeStatus::Accepted,
            engine_b_status: OutcomeStatus::Timeout,
            engine_a_ms: 825.0,
            engine_b_ms: 5000.0,
        },
    ];
    let paired_reduction = reduce_paired_trials(&gguf_digest.hex, &tok_digest.hex, &trials)?;

    // Eight executable negative controls!
    let mut negative_controls = Vec::new();

    // 1. Mismatched GGUF or token identity rejected
    let mut bad_gguf = trials.clone();
    bad_gguf[0].gguf_digest = "different-gguf-sha".into();
    let r1 = reduce_paired_trials(&gguf_digest.hex, &tok_digest.hex, &bad_gguf);
    negative_controls.push(NegativeControl {
        case_id: "mismatched_gguf_identity_rejected".into(),
        outcome: if r1.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r1.err().unwrap_or_else(|| "missed GGUF mismatch".into()),
    });

    // 2. Mislabeled cold state (process restarted without clearing persisted routing profile)
    let bad_cold = WarmStateVector {
        saved_routing_profile_state: "persisted_from_prior_run",
        ..warm_state_first_req.clone()
    };
    let r2 = bad_cold.validate_claimed_cold();
    negative_controls.push(NegativeControl {
        case_id: "mislabeled_cold_state_persisted_profile".into(),
        outcome: if r2.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r2
            .err()
            .unwrap_or_else(|| "missed persisted routing profile".into()),
    });

    // 3. Duplicate paired trial ID rejected
    let mut dup_trials = trials.clone();
    dup_trials[1].trial_id = "rep-1".into();
    let r3 = reduce_paired_trials(&gguf_digest.hex, &tok_digest.hex, &dup_trials);
    negative_controls.push(NegativeControl {
        case_id: "duplicate_paired_trial_rejected".into(),
        outcome: if r3.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r3.err().unwrap_or_else(|| "missed duplicate trial".into()),
    });

    // 4. Fewer than 3 repetitions (missing runs) rejected
    let r4 = reduce_paired_trials(&gguf_digest.hex, &tok_digest.hex, &trials[..2]);
    negative_controls.push(NegativeControl {
        case_id: "missing_paired_repetitions_rejected".into(),
        outcome: if r4.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r4
            .err()
            .unwrap_or_else(|| "missed insufficient reps".into()),
    });

    // 5. Deliberately incorrect free-generation task output rejected by independent checker
    let r5 = check_free_gen_task(
        "fn add(a: i32, b: i32) -> i32 { a + b }",
        "fn add(a: i32, b: i32) -> i32 { a - b }",
    );
    negative_controls.push(NegativeControl {
        case_id: "incorrect_free_gen_task_output_rejected".into(),
        outcome: r5,
        detail: "independent checker rejects subtly wrong subtraction implementation".into(),
    });

    // 6. Unverified router telemetry explicitly labeled unverified (never assumed equivalent!)
    let unverified_routing =
        simulations[1].routing_equivalence == "unverified_router_telemetry_absent";
    negative_controls.push(NegativeControl {
        case_id: "unobserved_router_equivalence_stays_unverified".into(),
        outcome: if unverified_routing {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("routing_equivalence={}", simulations[1].routing_equivalence),
    });

    // 7. Linux madvise & unified-memory CPU-GPU move guards via real validator
    let r7a = validate_host_memory_claim(&HostMemoryClaim {
        claims_madvise_guarantees_reclamation: true,
        unified_memory_host: false,
        claims_cpu_gpu_swap_frees_physical_capacity: false,
    });
    let r7b = validate_host_memory_claim(&HostMemoryClaim {
        claims_madvise_guarantees_reclamation: false,
        unified_memory_host: true,
        claims_cpu_gpu_swap_frees_physical_capacity: true,
    });
    negative_controls.push(NegativeControl {
        case_id: "madvise_and_unified_memory_non_freeing_guards".into(),
        outcome: if r7a.is_err() && r7b.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!(
            "madvise_rejected={}, unified_swap_rejected={}",
            r7a.is_err(),
            r7b.is_err()
        ),
    });

    // 8. Custom license vendoring gate via real validator
    let r8 = validate_vendoring_license_gate(&VendoringLicenseClaim {
        vendoring_requested: true,
        has_recorded_compatibility_decision: false,
        has_engine_resale_restriction: true,
    });
    negative_controls.push(NegativeControl {
        case_id: "custom_license_vendoring_gate_enforced".into(),
        outcome: r8,
        detail: LICENSE_SUMMARY.into(),
    });

    Ok(ExpertPlacementReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "discrete_nvidia_node_flint_ud_q5_k_xl_rustingstrata_vs_llamacpp",
            "Device-free placement/residency simulator, 5-layer warm-state checks, and offline paired reducer verified locally; physical RTX 5080 / NVIDIA node execution of flint-UD-Q5_K_XL.gguf required",
        ),
        gguf_digest,
        warm_state_first_req,
        simulations,
        paired_reduction,
        negative_controls,
        decision: "INCONCLUSIVE_PHYSICAL_DISCRETE_NVIDIA_EXECUTION_REQUIRED".into(),
    })
}

pub fn inputs_json(report: &ExpertPlacementReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.expert-placement.inputs.v1\",\n",
            "  \"issue\": 29,\n",
            "  \"upstream_source_review_commit\": {},\n",
            "  \"upstream_unverified_benchmark_note\": {},\n",
            "  \"custom_license_summary\": {},\n",
            "  \"rtx5080_strix_spark_supported_in_upstream_table\": false,\n",
            "  \"target_gguf_name\": {},\n",
            "  \"synthetic_gguf_digest\": {},\n",
            "  \"execution_class\": {}\n",
            "}}\n"
        ),
        quote(RUSTING_STRATA_COMMIT),
        quote(UPSTREAM_UNVERIFIED_CLAIM),
        quote(LICENSE_SUMMARY),
        quote(TARGET_GGUF_NAME),
        report.gguf_digest.to_json(),
        quote(report.execution_class.as_str()),
    )
}

pub fn results_json(report: &ExpertPlacementReport) -> String {
    let sims: Vec<String> = report
        .simulations
        .iter()
        .map(|s| {
            format!(
                "{{\"policy\":{},\"continuation_id\":{},\"gpu_expert_slots\":{},\"expert_hits\":{},\"expert_misses\":{},\"expert_swaps\":{},\"hit_rate\":{},\"adaptation_overhead_units\":{},\"total_work_units\":{},\"routing_equivalence\":{}}}",
                quote(s.policy.as_str()),
                quote(&s.continuation_id),
                s.gpu_expert_slots,
                s.expert_hits,
                s.expert_misses,
                s.expert_swaps,
                finite_f64(s.hit_rate),
                finite_f64(s.adaptation_overhead_units),
                finite_f64(s.total_work_units),
                quote(s.routing_equivalence),
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
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.expert-placement.results.v1\",\n",
            "  \"issue\": 29,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"decision\": {},\n",
            "  \"physical_decode_tok_s\": {},\n",
            "  \"first_request_wall_ms\": {},\n",
            "  \"paired_reduction\": {{\"synthetic_contract_fixture\":true,\"planned_pairs\":{},\"both_accepted_pairs\":{},\"failed_or_excluded_pairs\":{},\"counterbalanced\":{},\"mean_delta_b_minus_a_ms\":{}}},\n",
            "  \"simulations\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        quote(&report.decision),
        opt_f64(None),
        opt_f64(None),
        report.paired_reduction.planned_pairs,
        report.paired_reduction.both_accepted_pairs,
        report.paired_reduction.failed_or_excluded_pairs,
        report.paired_reduction.counterbalanced,
        opt_f64(report.paired_reduction.mean_delta_b_minus_a_ms),
        sims.join(","),
        neg.join(","),
    )
}

pub fn summary_markdown(report: &ExpertPlacementReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #29: RustingStrata Expert Placement & Host Residency vs llama.cpp"
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
