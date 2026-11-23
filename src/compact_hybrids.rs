//! Issue #36: Qualify compact hybrid recurrent language models by accepted tasks and session cost.
//!
//! Pinned candidate cohort (upstream benchmark/context/memory claims strictly non-gating):
//! - IBM Granite 4.0 H-Tiny (Mamba-2/attention MoE)
//! - AI21 Jamba2 3B (compact dense SSM/Transformer)
//! - TII Falcon H1 3B (in-layer Mamba-2 + attention hybrid)
//! - Compared against qualified local controls (Transformer, Nemotron Lightning `#35`, Qwen3.8 `#18`).
//!
//! Provides:
//! - Exact common manifest schema extension (`ArchitectureFamily`, `StateComponent`,
//!   `StateVisibility`, `StateOperation`) enforcing evidence-bound operation declarations (refusing
//!   theoretical state operations when `state_visibility == OpaqueEndpoint`).
//! - 3 frozen workload classes (Class A short structured agent tasks, Class B context/correction
//!   tasks across 3 per-tokenizer length buckets with dev/held-out split disjointness, Class C
//!   separately labeled `serving_mechanics_only` trace replay) with per-family accepted/planned and
//!   actual token-length bucket breakdown.
//! - Concurrency-1 vs bounded-concurrent session-density and unified-memory provenance reducer
//!   (enforcing `queued + admitted <= submitted`, `resident <= admitted`, non-collapsing denominators,
//!   `cost_per_accepted_task = None` at zero accepted tasks, never adding overlapping unified-memory
//!   counters or reporting weight file size as resident memory).
//! - Complete preregistered GO/NO-GO/INCONCLUSIVE decision rule requiring EITHER strictly better
//!   accepted rate within latency/memory limits OR noninferiority + material cost/density improvement,
//!   complete-profile vs isolated-architecture attribution guard, and 9 executable negative controls.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureFamily {
    Transformer,
    Rwkv,
    Mamba2,
    Mamba3,
    GatedDeltanet,
    Xlstm,
    HybridMambaAttention,
    HybridDeltanetAttention,
    Other,
}

impl ArchitectureFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::Rwkv => "rwkv",
            Self::Mamba2 => "mamba2",
            Self::Mamba3 => "mamba3",
            Self::GatedDeltanet => "gated_deltanet",
            Self::Xlstm => "xlstm",
            Self::HybridMambaAttention => "hybrid_mamba_attention",
            Self::HybridDeltanetAttention => "hybrid_deltanet_attention",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StateComponent {
    RecurrentState,
    ConvolutionState,
    AttentionKv,
    PositionOrClock,
    RouterOrExpertState,
    SpeculativeState,
}

impl StateComponent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RecurrentState => "recurrent_state",
            Self::ConvolutionState => "convolution_state",
            Self::AttentionKv => "attention_kv",
            Self::PositionOrClock => "position_or_clock",
            Self::RouterOrExpertState => "router_or_expert_state",
            Self::SpeculativeState => "speculative_state",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateVisibility {
    OpaqueEndpoint,
    ObservableMetrics,
    Exportable,
}

impl StateVisibility {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpaqueEndpoint => "opaque_endpoint",
            Self::ObservableMetrics => "observable_metrics",
            Self::Exportable => "exportable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StateOperation {
    Append,
    Reset,
    Serialize,
    Restore,
    Fork,
    Rollback,
}

impl StateOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Append => "append",
            Self::Reset => "reset",
            Self::Serialize => "serialize",
            Self::Restore => "restore",
            Self::Fork => "fork",
            Self::Rollback => "rollback",
        }
    }
}

#[derive(Debug, Clone)]
pub struct HybridModelProfile {
    pub profile_id: String,
    pub model_name: String,
    pub checkpoint_digest: DigestRef,
    pub tokenizer_template_rev: String,
    pub architecture_family: ArchitectureFamily,
    pub state_components: Vec<StateComponent>,
    pub state_visibility: StateVisibility,
    pub state_operations: Vec<StateOperation>,
    pub engine_quant_hardware: String,
    pub license: String,
}

pub fn validate_hybrid_profile(p: &HybridModelProfile) -> Result<(), String> {
    if p.profile_id.is_empty() || p.tokenizer_template_rev.is_empty() || p.license.is_empty() {
        return Err("missing profile_id, tokenizer_template_rev, or license".into());
    }
    if p.state_components.is_empty() {
        return Err("state_components must be explicitly declared".into());
    }
    if p.state_visibility == StateVisibility::OpaqueEndpoint {
        for op in &p.state_operations {
            if matches!(
                op,
                StateOperation::Serialize
                    | StateOperation::Restore
                    | StateOperation::Fork
                    | StateOperation::Rollback
            ) {
                return Err(format!(
                    "opaque_endpoint cannot declare `{}` supported without exportable state evidence",
                    op.as_str()
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadClass {
    ShortStructuredAgent,
    ContextUseAndCorrections,
    FrozenAgentTraceReplay,
}

impl WorkloadClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ShortStructuredAgent => "class_a_short_structured_agent",
            Self::ContextUseAndCorrections => "class_b_context_use_and_corrections",
            Self::FrozenAgentTraceReplay => "class_c_frozen_agent_trace_replay_mechanics_only",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskOutcomeRecord {
    pub task_id: String,
    pub logical_source_id: String,
    pub split: String,
    pub family: String,
    pub class: WorkloadClass,
    pub actual_prompt_tokens: usize,
    pub outcome: OutcomeStatus,
    pub wall_time_ms: f64,
    pub consumer_validated_tool_call: bool,
    pub model_emitted_tool_call: bool,
}

#[derive(Debug, Clone)]
pub struct ConcurrencyMemoryRecord {
    pub submitted: usize,
    pub queued: usize,
    pub admitted: usize,
    pub resident_slots: usize,
    pub weight_file_bytes: u64,
    pub observed_resident_rss_bytes: u64,
    pub unified_memory_host: bool,
    pub summed_overlapping_cpu_gpu_bytes: bool,
    pub reported_weight_file_as_resident: bool,
}

pub fn validate_memory_and_concurrency(m: &ConcurrencyMemoryRecord) -> Result<(), String> {
    if m.admitted > m.submitted
        || m.resident_slots > m.admitted
        || m.queued.saturating_add(m.admitted) > m.submitted
    {
        return Err(
            "impossible concurrency counts: queued+admitted > submitted or resident > admitted"
                .into(),
        );
    }
    if m.reported_weight_file_as_resident {
        return Err("weight file size cannot be reported as resident memory".into());
    }
    if m.unified_memory_host && m.summed_overlapping_cpu_gpu_bytes {
        return Err("unified-memory host must not sum overlapping CPU/GPU allocations".into());
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct FamilyBreakdownRow {
    pub family: String,
    pub planned: usize,
    pub accepted: usize,
    pub mean_prompt_tokens: f64,
}

#[derive(Debug, Clone)]
pub struct DecisionThresholds {
    pub control_accepted_rate: f64,
    pub noninferiority_margin: f64,
    pub control_cost_per_accepted_ms: f64,
    pub required_cost_improvement_ratio: f64, // e.g. 0.10 => >=10% lower cost per accepted task
    pub within_latency_memory_limits: bool,
    pub hardware_executed: bool,
}

#[derive(Debug, Clone)]
pub struct ProfileReduction {
    pub profile_id: String,
    pub planned_tasks: usize,
    pub accepted_tasks: usize,
    pub total_wall_ms: f64,
    pub cost_per_accepted_task_ms: Option<f64>,
    pub trace_replay_items_excluded_from_quality: usize,
    pub family_breakdown: Vec<FamilyBreakdownRow>,
    pub token_length_buckets_observed: Vec<usize>,
    pub decision: String,
}

pub fn reduce_profile_tasks(
    profile_id: &str,
    records: &[TaskOutcomeRecord],
    thresholds: &DecisionThresholds,
) -> Result<ProfileReduction, String> {
    if !thresholds.control_accepted_rate.is_finite()
        || !thresholds.noninferiority_margin.is_finite()
        || !thresholds.control_cost_per_accepted_ms.is_finite()
        || !thresholds.required_cost_improvement_ratio.is_finite()
        || thresholds.control_accepted_rate < 0.0
        || thresholds.control_accepted_rate > 1.0
        || thresholds.noninferiority_margin < 0.0
        || thresholds.control_cost_per_accepted_ms <= 0.0
        || thresholds.required_cost_improvement_ratio < 0.0
    {
        return Err("non-finite or invalid decision thresholds".into());
    }

    let mut dev_sources = BTreeSet::new();
    let mut held_sources = BTreeSet::new();
    for r in records {
        if !r.wall_time_ms.is_finite() || r.wall_time_ms < 0.0 {
            return Err("non-finite or negative wall_time_ms".into());
        }
        if r.split == "dev" {
            dev_sources.insert(r.logical_source_id.clone());
        } else if r.split == "held_out" {
            held_sources.insert(r.logical_source_id.clone());
        }
    }
    if dev_sources.intersection(&held_sources).next().is_some() {
        return Err("logical source item leaked between dev and held_out splits".into());
    }

    let mut planned = 0usize;
    let mut accepted = 0usize;
    let mut total_wall_ms = 0.0_f64;
    let mut trace_replay_count = 0usize;
    let mut by_fam: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut buckets: BTreeSet<usize> = BTreeSet::new();

    for r in records {
        if r.class == WorkloadClass::FrozenAgentTraceReplay {
            trace_replay_count += 1;
            continue;
        }
        if r.split != "held_out" {
            continue;
        }
        planned += 1;
        total_wall_ms += r.wall_time_ms;
        buckets.insert(r.actual_prompt_tokens);
        let entry = by_fam.entry(r.family.clone()).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.2 += r.actual_prompt_tokens;
        let valid_tool = !r.model_emitted_tool_call || r.consumer_validated_tool_call;
        if r.outcome == OutcomeStatus::Accepted && valid_tool {
            accepted += 1;
            entry.1 += 1;
        }
    }

    let cost_per_accepted = if accepted == 0 {
        None
    } else {
        Some(total_wall_ms / (accepted as f64))
    };

    // #36 Decision rule: GO requires hardware_executed && planned > 0 && within limits AND EITHER
    // (1) strictly better accepted task rate than control, OR
    // (2) noninferior accepted task rate within margin AND material cost/density improvement!
    let decision = if !thresholds.hardware_executed || planned == 0 {
        "INCONCLUSIVE".to_string()
    } else if !thresholds.within_latency_memory_limits {
        "NO-GO".to_string()
    } else {
        let rate = (accepted as f64) / (planned as f64);
        let strictly_better = rate > thresholds.control_accepted_rate;
        let noninferior =
            rate + thresholds.noninferiority_margin >= thresholds.control_accepted_rate;
        let material_cost_gain = match cost_per_accepted {
            Some(c) => {
                c <= thresholds.control_cost_per_accepted_ms
                    * (1.0 - thresholds.required_cost_improvement_ratio)
            }
            None => false,
        };
        if strictly_better || (noninferior && material_cost_gain) {
            "GO".to_string()
        } else {
            "NO-GO".to_string()
        }
    };

    let family_breakdown = by_fam
        .into_iter()
        .map(|(family, (p, a, tok_sum))| FamilyBreakdownRow {
            family,
            planned: p,
            accepted: a,
            mean_prompt_tokens: (tok_sum as f64) / (p as f64),
        })
        .collect();

    Ok(ProfileReduction {
        profile_id: profile_id.into(),
        planned_tasks: planned,
        accepted_tasks: accepted,
        total_wall_ms,
        cost_per_accepted_task_ms: cost_per_accepted,
        trace_replay_items_excluded_from_quality: trace_replay_count,
        family_breakdown,
        token_length_buckets_observed: buckets.into_iter().collect(),
        decision,
    })
}

#[derive(Debug, Clone)]
pub struct AttributionClaim {
    pub unmatched_quant_or_engine: bool,
    pub claims_isolated_architecture_speedup: bool,
}

pub fn validate_attribution_scope(claim: &AttributionClaim) -> OutcomeStatus {
    if claim.unmatched_quant_or_engine && claim.claims_isolated_architecture_speedup {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct CompactHybridsReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub candidate_profiles: Vec<HybridModelProfile>,
    pub positive_concurrency_memory_valid: bool,
    pub reductions: Vec<ProfileReduction>,
    pub zero_accepted_cost_undefined: bool,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<CompactHybridsReport, String> {
    let granite = HybridModelProfile {
        profile_id: "ibm-granite-4.0-h-tiny-spark-v1".into(),
        model_name: "IBM Granite 4.0 H-Tiny".into(),
        checkpoint_digest: DigestRef::sha256(b"granite-4.0-h-tiny-pinned-placeholder"),
        tokenizer_template_rev: "granite4-chat-tool-v1".into(),
        architecture_family: ArchitectureFamily::HybridMambaAttention,
        state_components: vec![
            StateComponent::RecurrentState,
            StateComponent::ConvolutionState,
            StateComponent::AttentionKv,
            StateComponent::RouterOrExpertState,
            StateComponent::PositionOrClock,
        ],
        state_visibility: StateVisibility::OpaqueEndpoint,
        state_operations: vec![StateOperation::Append, StateOperation::Reset],
        engine_quant_hardware: "vllm-bf16-dgx-spark-gb10".into(),
        license: "Apache-2.0".into(),
    };
    let jamba = HybridModelProfile {
        profile_id: "ai21-jamba2-3b-spark-v1".into(),
        model_name: "AI21 Jamba2 3B".into(),
        checkpoint_digest: DigestRef::sha256(b"jamba2-3b-pinned-placeholder"),
        tokenizer_template_rev: "jamba2-chat-tool-v1".into(),
        architecture_family: ArchitectureFamily::HybridMambaAttention,
        state_components: vec![
            StateComponent::RecurrentState,
            StateComponent::ConvolutionState,
            StateComponent::AttentionKv,
            StateComponent::PositionOrClock,
        ],
        state_visibility: StateVisibility::OpaqueEndpoint,
        state_operations: vec![StateOperation::Append, StateOperation::Reset],
        engine_quant_hardware: "vllm-bf16-dgx-spark-gb10".into(),
        license: "Apache-2.0".into(),
    };
    let falcon = HybridModelProfile {
        profile_id: "tii-falcon-h1-3b-spark-v1".into(),
        model_name: "TII Falcon H1 3B".into(),
        checkpoint_digest: DigestRef::sha256(b"falcon-h1-3b-pinned-placeholder"),
        tokenizer_template_rev: "falcon-h1-chat-tool-v1".into(),
        architecture_family: ArchitectureFamily::HybridMambaAttention,
        state_components: vec![
            StateComponent::RecurrentState,
            StateComponent::ConvolutionState,
            StateComponent::AttentionKv,
            StateComponent::PositionOrClock,
        ],
        state_visibility: StateVisibility::OpaqueEndpoint,
        state_operations: vec![StateOperation::Append, StateOperation::Reset],
        engine_quant_hardware: "vllm-bf16-dgx-spark-gb10".into(),
        license: "Falcon-LLM-License".into(),
    };

    for p in [&granite, &jamba, &falcon] {
        validate_hybrid_profile(p)?;
    }

    // Positive concurrency & unified-memory check
    let valid_mem = ConcurrencyMemoryRecord {
        submitted: 4,
        queued: 1,
        admitted: 3,
        resident_slots: 2,
        weight_file_bytes: 2_000_000_000,
        observed_resident_rss_bytes: 3_150_000_000,
        unified_memory_host: true,
        summed_overlapping_cpu_gpu_bytes: false,
        reported_weight_file_as_resident: false,
    };
    validate_memory_and_concurrency(&valid_mem)?;

    let sample_records = vec![
        TaskOutcomeRecord {
            task_id: "a-json-1".into(),
            logical_source_id: "src-1".into(),
            split: "held_out".into(),
            family: "short_json".into(),
            class: WorkloadClass::ShortStructuredAgent,
            actual_prompt_tokens: 312,
            outcome: OutcomeStatus::Accepted,
            wall_time_ms: 42.0,
            consumer_validated_tool_call: true,
            model_emitted_tool_call: true,
        },
        TaskOutcomeRecord {
            task_id: "b-ctx-short".into(),
            logical_source_id: "src-2".into(),
            split: "held_out".into(),
            family: "superseded_fact".into(),
            class: WorkloadClass::ContextUseAndCorrections,
            actual_prompt_tokens: 1024,
            outcome: OutcomeStatus::Accepted,
            wall_time_ms: 88.0,
            consumer_validated_tool_call: false,
            model_emitted_tool_call: false,
        },
        TaskOutcomeRecord {
            task_id: "b-ctx-mid".into(),
            logical_source_id: "src-3".into(),
            split: "held_out".into(),
            family: "delayed_tool".into(),
            class: WorkloadClass::ContextUseAndCorrections,
            actual_prompt_tokens: 4096,
            outcome: OutcomeStatus::Timeout,
            wall_time_ms: 250.0,
            consumer_validated_tool_call: false,
            model_emitted_tool_call: false,
        },
        TaskOutcomeRecord {
            task_id: "b-ctx-long".into(),
            logical_source_id: "src-4".into(),
            split: "held_out".into(),
            family: "distractor_id".into(),
            class: WorkloadClass::ContextUseAndCorrections,
            actual_prompt_tokens: 16384,
            outcome: OutcomeStatus::Accepted,
            wall_time_ms: 195.0,
            consumer_validated_tool_call: false,
            model_emitted_tool_call: false,
        },
        TaskOutcomeRecord {
            task_id: "c-trace-serial".into(),
            logical_source_id: "src-5".into(),
            split: "held_out".into(),
            family: "trace_replay".into(),
            class: WorkloadClass::FrozenAgentTraceReplay,
            actual_prompt_tokens: 2048,
            outcome: OutcomeStatus::Accepted,
            wall_time_ms: 60.0,
            consumer_validated_tool_call: false,
            model_emitted_tool_call: false,
        },
    ];

    let default_thresholds = DecisionThresholds {
        control_accepted_rate: 0.75,
        noninferiority_margin: 0.05,
        control_cost_per_accepted_ms: 180.0,
        required_cost_improvement_ratio: 0.15,
        within_latency_memory_limits: true,
        hardware_executed: false,
    };

    let red_granite =
        reduce_profile_tasks(&granite.profile_id, &sample_records, &default_thresholds)?;
    let zero_records = vec![TaskOutcomeRecord {
        task_id: "z-1".into(),
        logical_source_id: "src-z".into(),
        split: "held_out".into(),
        family: "short_json".into(),
        class: WorkloadClass::ShortStructuredAgent,
        actual_prompt_tokens: 256,
        outcome: OutcomeStatus::Invalid,
        wall_time_ms: 35.0,
        consumer_validated_tool_call: false,
        model_emitted_tool_call: false,
    }];
    let red_zero =
        reduce_profile_tasks("zero-accepted-control", &zero_records, &default_thresholds)?;
    let zero_accepted_cost_undefined = red_zero.cost_per_accepted_task_ms.is_none();

    let mut negative_controls = Vec::new();

    {
        let mut bad = granite.clone();
        bad.state_operations.push(StateOperation::Serialize);
        negative_controls.push(NegativeControl {
            case_id: "nc1_opaque_endpoint_claims_serialize".into(),
            outcome: if validate_hybrid_profile(&bad).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "opaque_endpoint refused from claiming serialize/fork support from theory"
                .into(),
        });
    }
    {
        let mut leaked = sample_records.clone();
        leaked.push(TaskOutcomeRecord {
            task_id: "dev-leak".into(),
            logical_source_id: "src-1".into(),
            split: "dev".into(),
            family: "short_json".into(),
            class: WorkloadClass::ShortStructuredAgent,
            actual_prompt_tokens: 312,
            outcome: OutcomeStatus::Accepted,
            wall_time_ms: 40.0,
            consumer_validated_tool_call: true,
            model_emitted_tool_call: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc2_dev_heldout_source_leakage".into(),
            outcome: if reduce_profile_tasks("leak", &leaked, &default_thresholds).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "shared logical_source_id across dev and held_out rejected".into(),
        });
    }
    {
        let unval = vec![TaskOutcomeRecord {
            task_id: "unval-tool".into(),
            logical_source_id: "src-u".into(),
            split: "held_out".into(),
            family: "tool".into(),
            class: WorkloadClass::ShortStructuredAgent,
            actual_prompt_tokens: 300,
            outcome: OutcomeStatus::Accepted,
            wall_time_ms: 20.0,
            consumer_validated_tool_call: false,
            model_emitted_tool_call: true,
        }];
        let r = reduce_profile_tasks("unval", &unval, &default_thresholds)?;
        negative_controls.push(NegativeControl {
            case_id: "nc3_unvalidated_tool_call_not_accepted".into(),
            outcome: if r.accepted_tasks == 0 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "model-emitted tool call without consumer validation yields 0 accepted tasks"
                .into(),
        });
    }
    {
        let trace_only = vec![sample_records[4].clone()];
        let mut hw_th = default_thresholds.clone();
        hw_th.hardware_executed = true;
        let r = reduce_profile_tasks("trace-only", &trace_only, &hw_th)?;
        negative_controls.push(NegativeControl {
            case_id: "nc4_trace_replay_not_quality_pass".into(),
            outcome: if r.planned_tasks == 0 && r.decision == "INCONCLUSIVE" {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "serving-mechanics trace replay cannot pass quality decision rule".into(),
        });
    }
    {
        let mut bad_mem = valid_mem.clone();
        bad_mem.reported_weight_file_as_resident = true;
        negative_controls.push(NegativeControl {
            case_id: "nc5_weight_file_as_resident_memory".into(),
            outcome: if validate_memory_and_concurrency(&bad_mem).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "reporting weight file size as resident memory rejected".into(),
        });
    }
    {
        let mut bad_mem = valid_mem.clone();
        bad_mem.summed_overlapping_cpu_gpu_bytes = true;
        negative_controls.push(NegativeControl {
            case_id: "nc6_unified_memory_overlap_summed".into(),
            outcome: if validate_memory_and_concurrency(&bad_mem).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "summing overlapping CPU/GPU allocations on unified-memory host rejected"
                .into(),
        });
    }
    {
        let st = validate_attribution_scope(&AttributionClaim {
            unmatched_quant_or_engine: true,
            claims_isolated_architecture_speedup: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc7_unmatched_profile_claims_arch_speedup".into(),
            outcome: st,
            detail: "unmatched quant/engine supports complete-profile comparison only".into(),
        });
    }
    {
        let mut bad_rec = sample_records.clone();
        bad_rec[0].wall_time_ms = f64::NAN;
        negative_controls.push(NegativeControl {
            case_id: "nc8_nonfinite_wall_time_rejected".into(),
            outcome: if reduce_profile_tasks("nan", &bad_rec, &default_thresholds).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN wall_time_ms fails explicit finiteness check".into(),
        });
    }
    // NC9: noninferiority WITHOUT material cost/density improvement returns NO-GO (not GO!)
    {
        let mut hw_th = default_thresholds.clone();
        hw_th.hardware_executed = true;
        // sample_records has accepted=3/4=0.75 (equal to control=0.75), cost=575/3=191.67 > control=180
        let r = reduce_profile_tasks("no-cost-gain", &sample_records, &hw_th)?;
        negative_controls.push(NegativeControl {
            case_id: "nc9_noninferior_without_cost_improvement_nogo".into(),
            outcome: if r.decision == "NO-GO" {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "noninferior rate without material cost/density improvement yields NO-GO"
                .into(),
        });
    }

    Ok(CompactHybridsReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "DGX Spark / RTX 5080 / Strix Halo serving of Granite 4.0 H-Tiny, Jamba2 3B, Falcon H1 3B",
            "Manifest schema, dialect/workload/split hygiene, unified-memory guards, and decision rules verified on CPU; live model execution did not run locally",
        ),
        synthetic_contract_fixture: true,
        candidate_profiles: vec![granite, jamba, falcon],
        positive_concurrency_memory_valid: true,
        reductions: vec![red_granite, red_zero],
        zero_accepted_cost_undefined,
        negative_controls,
    })
}

pub fn inputs_json(report: &CompactHybridsReport) -> String {
    let profiles: Vec<String> = report
        .candidate_profiles
        .iter()
        .map(|p| {
            format!(
                "{{\"profile_id\":{},\"model\":{},\"architecture_family\":{},\"state_visibility\":{},\"license\":{}}}",
                quote(&p.profile_id),
                quote(&p.model_name),
                quote(p.architecture_family.as_str()),
                quote(p.state_visibility.as_str()),
                quote(&p.license),
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.compact-hybrids.inputs.v1\",\n",
            "  \"issue\": 36,\n",
            "  \"execution_class\": {},\n",
            "  \"candidate_profiles\": [{}],\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        profiles.join(","),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &CompactHybridsReport) -> String {
    let reds: Vec<String> = report
        .reductions
        .iter()
        .map(|r| {
            let fams: Vec<String> = r
                .family_breakdown
                .iter()
                .map(|f| {
                    format!(
                        "{{\"family\":{},\"planned\":{},\"accepted\":{},\"mean_prompt_tokens\":{}}}",
                        quote(&f.family),
                        f.planned,
                        f.accepted,
                        finite_f64(f.mean_prompt_tokens),
                    )
                })
                .collect();
            let buckets: Vec<String> = r
                .token_length_buckets_observed
                .iter()
                .map(ToString::to_string)
                .collect();
            format!(
                "{{\"profile_id\":{},\"planned_tasks\":{},\"accepted_tasks\":{},\"total_wall_ms\":{},\"cost_per_accepted_task_ms\":{},\"trace_replay_excluded\":{},\"token_length_buckets_observed\":[{}],\"family_breakdown\":[{}],\"decision\":{}}}",
                quote(&r.profile_id),
                r.planned_tasks,
                r.accepted_tasks,
                finite_f64(r.total_wall_ms),
                opt_f64(r.cost_per_accepted_task_ms),
                r.trace_replay_items_excluded_from_quality,
                buckets.join(","),
                fams.join(","),
                quote(&r.decision),
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
            "  \"schema\": \"inference-lab.compact-hybrids.results.v1\",\n",
            "  \"issue\": 36,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"positive_concurrency_memory_valid\": {},\n",
            "  \"zero_accepted_cost_undefined\": {},\n",
            "  \"profile_reductions\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.positive_concurrency_memory_valid,
        report.zero_accepted_cost_undefined,
        reds.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &CompactHybridsReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #36: Compact Hybrid Recurrent Language Models Comparison Profile"
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
        "- Hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Zero-accepted cost undefined (`None`): `{}`",
        report.zero_accepted_cost_undefined
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Candidate Profiles").unwrap();
    for p in &report.candidate_profiles {
        writeln!(
            out,
            "- `{}` (`{}`): `arch = {}`, `visibility = {}`, `components = {}`, `ops = {}`",
            p.profile_id,
            p.model_name,
            p.architecture_family.as_str(),
            p.state_visibility.as_str(),
            p.state_components.len(),
            p.state_operations.len()
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
