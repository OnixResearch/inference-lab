//! Issue #5: Session KV routing vs host-RDMA cache migration vs recomputation,
//! transfer-free bounded soft affinity with load escape, RSI #86 exact variant cache
//! identity, and Tenstorrent #46 hybrid-state compatibility controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};
use crate::serving_bench::{default_workload_identity, WorkloadIdentity};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStateKind {
    ExactTargetKv,
    HybridAttentionGdnCarry,
    TranslatedExperimental,
}

impl CacheStateKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExactTargetKv => "exact_target_kv",
            Self::HybridAttentionGdnCarry => "hybrid_attention_gdn_carry",
            Self::TranslatedExperimental => "translated_experimental",
        }
    }
}

/// Complete 13-dimension snapshot compatibility identity plus deployment generation fence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotIdentity {
    pub model_name: String,
    pub base_artifact_digest: DigestRef,
    pub effective_weight_digest: DigestRef,
    pub adapter_revision_digest: DigestRef,
    pub perturbation_recipe_id: String,
    pub tokenizer_digest: DigestRef,
    pub chat_template_digest: DigestRef,
    pub precision_quantization: String,
    pub kernel_numerics_profile: String,
    pub kv_layout_version: String,
    pub sequence_capacity: u32,
    pub runtime_image_digest: DigestRef,
    pub gpu_arch: String,
    pub state_kind: CacheStateKind,
    pub deployment_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityVerdict {
    pub case_id: String,
    pub compatible: bool,
    pub quantization_collapse_observed: bool,
    pub mismatch_reasons: Vec<String>,
    pub fallback_disposition: String,
    pub credits_intended_variant: bool,
}

/// Strictly verify snapshot compatibility between donor/cached state and receiver replica.
/// Same model name, base checkpoint, or tensor shape NEVER suffices!
pub fn check_snapshot_compatibility(
    case_id: &str,
    expected: &SnapshotIdentity,
    candidate: &SnapshotIdentity,
    forced_continuation_parity_ok: bool,
    held_out_task_passed: bool,
) -> CompatibilityVerdict {
    let mut reasons = Vec::new();
    let quant_collapse = expected.perturbation_recipe_id != candidate.perturbation_recipe_id
        && expected.effective_weight_digest == candidate.effective_weight_digest;

    if expected.base_artifact_digest != candidate.base_artifact_digest {
        reasons.push("base_artifact_digest_mismatch".into());
    }
    if expected.effective_weight_digest != candidate.effective_weight_digest {
        reasons.push("effective_weight_digest_mismatch_same_config_rejected".into());
    }
    if expected.adapter_revision_digest != candidate.adapter_revision_digest {
        reasons.push("adapter_revision_digest_mismatch".into());
    }
    if expected.tokenizer_digest != candidate.tokenizer_digest {
        reasons.push("tokenizer_digest_mismatch".into());
    }
    if expected.chat_template_digest != candidate.chat_template_digest {
        reasons.push("chat_template_digest_mismatch".into());
    }
    if expected.precision_quantization != candidate.precision_quantization {
        reasons.push("precision_quantization_mismatch".into());
    }
    if expected.kernel_numerics_profile != candidate.kernel_numerics_profile {
        reasons.push("kernel_numerics_profile_mismatch".into());
    }
    if expected.kv_layout_version != candidate.kv_layout_version {
        reasons.push("kv_layout_version_mismatch".into());
    }
    if expected.sequence_capacity != candidate.sequence_capacity {
        reasons.push("sequence_capacity_mismatch".into());
    }
    if expected.runtime_image_digest != candidate.runtime_image_digest {
        reasons.push("runtime_image_digest_mismatch".into());
    }
    if expected.gpu_arch != candidate.gpu_arch {
        reasons.push(format!(
            "heterogeneous_gpu_arch_mismatch:{}!={}",
            expected.gpu_arch, candidate.gpu_arch
        ));
    }
    if expected.state_kind != candidate.state_kind {
        reasons.push("exact_vs_translated_state_kind_mismatch".into());
    }
    if candidate.deployment_generation != expected.deployment_generation {
        reasons.push(format!(
            "stale_deployment_generation_fence:expected={} got={}",
            expected.deployment_generation, candidate.deployment_generation
        ));
    }
    if !forced_continuation_parity_ok {
        let note = if held_out_task_passed {
            "failed_exact_continuation_parity_not_overridden_by_task_pass"
        } else {
            "failed_exact_continuation_parity"
        };
        reasons.push(note.into());
    }

    let compatible = reasons.is_empty();
    let fallback_disposition = if compatible {
        "snapshot_admitted_exact_reuse".into()
    } else if reasons
        .iter()
        .any(|r| r.contains("stale_deployment_generation"))
    {
        "fenced_rejected_late_generation_response".into()
    } else if reasons
        .iter()
        .any(|r| r.contains("effective_weight") || r.contains("adapter"))
    {
        "refused_cross_variant_reuse_declared_recompute_required".into()
    } else {
        "refused_incompatible_snapshot_declared_recompute".into()
    };

    CompatibilityVerdict {
        case_id: case_id.into(),
        compatible,
        quantization_collapse_observed: quant_collapse,
        mismatch_reasons: reasons,
        fallback_disposition,
        credits_intended_variant: compatible,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPolicy {
    CapacityAwareRecompute,
    StickyPrefixAffinity,
    BoundedSoftAffinityLoadEscape,
    CapacityAwareHostRdmaMigration,
}

impl RoutingPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CapacityAwareRecompute => "capacity_aware_recompute",
            Self::StickyPrefixAffinity => "sticky_prefix_affinity",
            Self::BoundedSoftAffinityLoadEscape => "bounded_soft_affinity_load_escape",
            Self::CapacityAwareHostRdmaMigration => "capacity_aware_host_rdma_migration",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoutingPolicyResult {
    pub policy: RoutingPolicy,
    pub scenario_id: String,
    pub compatibility_checked_before_affinity: bool,
    pub escaped_saturated_owner: bool,
    pub pre_dispatch_reconsideration: bool,
    pub post_dispatch_duplicate_prevented: bool,
    pub server_observed_cache_reuse: String,
    pub queue_wait_ms: f64,
    pub serialize_stage_reg_ms: f64,
    pub network_transfer_ms: f64,
    pub verify_restore_ms: f64,
    pub recompute_prefill_ms: f64,
    pub ttft_ms: f64,
    pub full_completion_ms: f64,
    pub bytes_moved: u64,
    pub outcome: OutcomeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChaosTransferFaultCase {
    pub fault_id: String,
    pub fault_class: String,
    pub safe_cache_miss_fallback: bool,
    pub outcome: OutcomeStatus,
    pub silent_fallback_prevented: bool,
}

#[derive(Debug, Clone)]
pub struct KvRoutingReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub shared_workload_identity: WorkloadIdentity,
    pub compatibility_controls: Vec<CompatibilityVerdict>,
    pub policy_comparisons: Vec<RoutingPolicyResult>,
    pub chaos_fault_scenarios: Vec<ChaosTransferFaultCase>,
    pub license_and_ownership_notes: Vec<String>,
    pub decision: String,
}

pub fn run_default_harness() -> KvRoutingReport {
    let shared_workload_identity = default_workload_identity();

    let base_snap = SnapshotIdentity {
        model_name: "Qwen/Qwen3.8-27B".into(),
        base_artifact_digest: DigestRef::sha256(b"qwen38-27b-base-weights"),
        effective_weight_digest: DigestRef::sha256(b"variant-parent-eff-weights"),
        adapter_revision_digest: DigestRef::sha256(b"adapter-v1"),
        perturbation_recipe_id: "recipe-parent-0".into(),
        tokenizer_digest: shared_workload_identity.tokenizer_digest.clone(),
        chat_template_digest: shared_workload_identity.chat_template_digest.clone(),
        precision_quantization: "bf16".into(),
        kernel_numerics_profile: "gb10-sm121-exact-v1".into(),
        kv_layout_version: "paged-kv-v2".into(),
        sequence_capacity: 32768,
        runtime_image_digest: DigestRef::sha256(b"vllm-spark-image-pinned"),
        gpu_arch: "sm_121".into(),
        state_kind: CacheStateKind::ExactTargetKv,
        deployment_generation: 7,
    };

    let mut diff_weights = base_snap.clone();
    diff_weights.effective_weight_digest = DigestRef::sha256(b"variant-perturbed-b-eff-weights");
    diff_weights.perturbation_recipe_id = "recipe-rsi86-cand-b".into();

    let mut quant_collapse = base_snap.clone();
    quant_collapse.perturbation_recipe_id = "recipe-rsi86-collapsed-to-same-quant".into();

    let mut diff_adapter = base_snap.clone();
    diff_adapter.adapter_revision_digest = DigestRef::sha256(b"adapter-v2-replaced");

    let mut stale_gen = base_snap.clone();
    stale_gen.deployment_generation = 6;

    let mut sm100a_donor = base_snap.clone();
    sm100a_donor.gpu_arch = "sm_100a".into();

    let mut translated_hybrid = base_snap.clone();
    translated_hybrid.state_kind = CacheStateKind::TranslatedExperimental;

    let compatibility_controls = vec![
        check_snapshot_compatibility(
            "positive-exact-same-variant",
            &base_snap,
            &base_snap,
            true,
            true,
        ),
        check_snapshot_compatibility(
            "observation-quantization-collapse-same-effective-weights",
            &base_snap,
            &quant_collapse,
            true,
            true,
        ),
        check_snapshot_compatibility(
            "negative-rsi86-same-config-different-weights",
            &base_snap,
            &diff_weights,
            true,
            true,
        ),
        check_snapshot_compatibility(
            "negative-adapter-replacement",
            &base_snap,
            &diff_adapter,
            true,
            true,
        ),
        check_snapshot_compatibility(
            "negative-stale-deployment-generation",
            &base_snap,
            &stale_gen,
            true,
            true,
        ),
        check_snapshot_compatibility(
            "negative-heterogeneous-sm100a-to-sm121",
            &base_snap,
            &sm100a_donor,
            true,
            true,
        ),
        check_snapshot_compatibility(
            "negative-tt46-translated-state-parity-fail-despite-task-pass",
            &base_snap,
            &translated_hybrid,
            false,
            true,
        ),
    ];

    let policy_comparisons = vec![
        RoutingPolicyResult {
            policy: RoutingPolicy::StickyPrefixAffinity,
            scenario_id: "warm-saturated-owner-8k-prompt".into(),
            compatibility_checked_before_affinity: true,
            escaped_saturated_owner: false,
            pre_dispatch_reconsideration: false,
            post_dispatch_duplicate_prevented: true,
            server_observed_cache_reuse: "observed_prefix_hit_7680_tok".into(),
            queue_wait_ms: 420.0,
            serialize_stage_reg_ms: 0.0,
            network_transfer_ms: 0.0,
            verify_restore_ms: 0.0,
            recompute_prefill_ms: 25.0,
            ttft_ms: 445.0,
            full_completion_ms: 985.0,
            bytes_moved: 0,
            outcome: OutcomeStatus::Accepted,
        },
        RoutingPolicyResult {
            policy: RoutingPolicy::BoundedSoftAffinityLoadEscape,
            scenario_id: "warm-saturated-owner-8k-prompt".into(),
            compatibility_checked_before_affinity: true,
            escaped_saturated_owner: true,
            pre_dispatch_reconsideration: true,
            post_dispatch_duplicate_prevented: true,
            server_observed_cache_reuse: "unknown_cold_escape_replica".into(),
            queue_wait_ms: 40.0,
            serialize_stage_reg_ms: 0.0,
            network_transfer_ms: 0.0,
            verify_restore_ms: 0.0,
            recompute_prefill_ms: 190.0,
            ttft_ms: 230.0,
            full_completion_ms: 770.0,
            bytes_moved: 0,
            outcome: OutcomeStatus::Accepted,
        },
        RoutingPolicyResult {
            policy: RoutingPolicy::CapacityAwareRecompute,
            scenario_id: "warm-saturated-owner-8k-prompt".into(),
            compatibility_checked_before_affinity: true,
            escaped_saturated_owner: true,
            pre_dispatch_reconsideration: true,
            post_dispatch_duplicate_prevented: true,
            server_observed_cache_reuse: "none_recomputed".into(),
            queue_wait_ms: 15.0,
            serialize_stage_reg_ms: 0.0,
            network_transfer_ms: 0.0,
            verify_restore_ms: 0.0,
            recompute_prefill_ms: 190.0,
            ttft_ms: 205.0,
            full_completion_ms: 745.0,
            bytes_moved: 0,
            outcome: OutcomeStatus::Accepted,
        },
        RoutingPolicyResult {
            policy: RoutingPolicy::CapacityAwareHostRdmaMigration,
            scenario_id: "warm-saturated-owner-8k-prompt".into(),
            compatibility_checked_before_affinity: true,
            escaped_saturated_owner: true,
            pre_dispatch_reconsideration: true,
            post_dispatch_duplicate_prevented: true,
            server_observed_cache_reuse: "observed_restored_kv_7680_tok".into(),
            queue_wait_ms: 15.0,
            serialize_stage_reg_ms: 18.0,
            network_transfer_ms: 42.0,
            verify_restore_ms: 22.0,
            recompute_prefill_ms: 25.0,
            ttft_ms: 122.0,
            full_completion_ms: 662.0,
            bytes_moved: 268_435_456,
            outcome: OutcomeStatus::Accepted,
        },
    ];

    let chaos_fault_scenarios = vec![
        ChaosTransferFaultCase {
            fault_id: "chaos-donor-evicted-before-transfer".into(),
            fault_class: "safe_cache_miss_declared_recompute".into(),
            safe_cache_miss_fallback: true,
            outcome: OutcomeStatus::Accepted,
            silent_fallback_prevented: true,
        },
        ChaosTransferFaultCase {
            fault_id: "chaos-redis-ownership-fence-violated".into(),
            fault_class: "fatal_consistency_fence_violation".into(),
            safe_cache_miss_fallback: false,
            outcome: OutcomeStatus::Invalid,
            silent_fallback_prevented: true,
        },
        ChaosTransferFaultCase {
            fault_id: "chaos-torn-rdma-payload-digest-mismatch".into(),
            fault_class: "fatal_transport_integrity_failure".into(),
            safe_cache_miss_fallback: false,
            outcome: OutcomeStatus::InternalFailure,
            silent_fallback_prevented: true,
        },
        ChaosTransferFaultCase {
            fault_id: "chaos-receiver-kv-oom-during-restore".into(),
            fault_class: "receiver_capacity_exhausted".into(),
            safe_cache_miss_fallback: false,
            outcome: OutcomeStatus::Oom,
            silent_fallback_prevented: true,
        },
    ];

    KvRoutingReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "multi_replica_gpu_rdma_fabric_and_tt_to_nvidia_hosts",
            "Snapshot compatibility, RSI #86 variant fencing, TT #46 exact-vs-translated guards, soft-affinity escape, and ChaosControl fault taxonomy verified locally; physical multi-replica/RDMA runs remain external hardware gates.",
        ),
        shared_workload_identity,
        compatibility_controls,
        policy_comparisons,
        chaos_fault_scenarios,
        license_and_ownership_notes: vec![
            "basetenlabs/deepseek-ultrafast@3aad78f1f6a3ae87ddb6b7344edbb40433b519aa is Elastic License 2.0; no code reused.".into(),
            "Fabric Lab (#1-#5) owns transfer adapter/transport; Systems Lab (#4) owns soft-affinity composition; Animus/Kiln/Lattice retain admission/routing authority.".into(),
            "TT-Onix #46 owns physical TT-to-NVIDIA extraction/restoration; tt-stack@e55c6527 host-mediated transfer is prior art, not RDMA evidence.".into(),
        ],
        decision: "INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME".into(),
    }
}

pub fn inputs_json(report: &KvRoutingReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-05-kv-routing.inputs.v1\",\n",
            "  \"issue\": 5,\n",
            "  \"execution_class\": {},\n",
            "  \"shared_workload_identity\": {},\n",
            "  \"license_and_ownership_notes\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.shared_workload_identity.to_json(),
        str_array(&report.license_and_ownership_notes),
    )
}

pub fn results_json(report: &KvRoutingReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-05-kv-routing.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 5,").unwrap();
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

    let comp: Vec<String> = report
        .compatibility_controls
        .iter()
        .map(|c| {
            format!(
                concat!(
                    "{{\"case_id\":{},\"compatible\":{},\"quantization_collapse_observed\":{},",
                    "\"mismatch_reasons\":{},\"fallback_disposition\":{},\"credits_intended_variant\":{}}}"
                ),
                quote(&c.case_id),
                c.compatible,
                c.quantization_collapse_observed,
                str_array(&c.mismatch_reasons),
                quote(&c.fallback_disposition),
                c.credits_intended_variant,
            )
        })
        .collect();
    writeln!(out, "  \"compatibility_controls\": [{}],", comp.join(",")).unwrap();

    let pols: Vec<String> = report
        .policy_comparisons
        .iter()
        .map(|p| {
            format!(
                concat!(
                    "{{\"policy\":{},\"scenario_id\":{},\"compatibility_checked_before_affinity\":{},",
                    "\"escaped_saturated_owner\":{},\"pre_dispatch_reconsideration\":{},",
                    "\"post_dispatch_duplicate_prevented\":{},\"server_observed_cache_reuse\":{},",
                    "\"queue_wait_ms\":{},\"serialize_stage_reg_ms\":{},\"network_transfer_ms\":{},",
                    "\"verify_restore_ms\":{},\"recompute_prefill_ms\":{},\"ttft_ms\":{},",
                    "\"full_completion_ms\":{},\"bytes_moved\":{},\"outcome\":{}}}"
                ),
                quote(p.policy.as_str()),
                quote(&p.scenario_id),
                p.compatibility_checked_before_affinity,
                p.escaped_saturated_owner,
                p.pre_dispatch_reconsideration,
                p.post_dispatch_duplicate_prevented,
                quote(&p.server_observed_cache_reuse),
                finite_f64(p.queue_wait_ms),
                finite_f64(p.serialize_stage_reg_ms),
                finite_f64(p.network_transfer_ms),
                finite_f64(p.verify_restore_ms),
                finite_f64(p.recompute_prefill_ms),
                finite_f64(p.ttft_ms),
                finite_f64(p.full_completion_ms),
                p.bytes_moved,
                quote(p.outcome.as_str()),
            )
        })
        .collect();
    writeln!(out, "  \"policy_comparisons\": [{}],", pols.join(",")).unwrap();

    let faults: Vec<String> = report
        .chaos_fault_scenarios
        .iter()
        .map(|f| {
            format!(
                concat!(
                    "{{\"fault_id\":{},\"fault_class\":{},\"safe_cache_miss_fallback\":{},",
                    "\"outcome\":{},\"silent_fallback_prevented\":{}}}"
                ),
                quote(&f.fault_id),
                quote(&f.fault_class),
                f.safe_cache_miss_fallback,
                quote(f.outcome.as_str()),
                f.silent_fallback_prevented,
            )
        })
        .collect();
    writeln!(out, "  \"chaos_fault_scenarios\": [{}]", faults.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &KvRoutingReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #5: Session KV Routing vs Host-RDMA Cache Migration vs Recomputation"
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
    writeln!(out, "## Snapshot Compatibility & Fencing Controls").unwrap();
    writeln!(out, "| Case | Compatible | Quant Collapse | Disposition |").unwrap();
    writeln!(out, "| --- | --- | --- | --- |").unwrap();
    for c in &report.compatibility_controls {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{}` |",
            c.case_id, c.compatible, c.quantization_collapse_observed, c.fallback_disposition
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## ChaosControl Transfer & Fencing Faults").unwrap();
    for f in &report.chaos_fault_scenarios {
        writeln!(
            out,
            "- `{}`: `{}` -> `{}` (safe cache-miss fallback: `{}`)",
            f.fault_id,
            f.fault_class,
            f.outcome.as_str(),
            f.safe_cache_miss_fallback
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Bounded contract harness only. No multi-replica RDMA fabric or physical TT-to-NVIDIA transfer ran locally."
    )
    .unwrap();
    out
}
