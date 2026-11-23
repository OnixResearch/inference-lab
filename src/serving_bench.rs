//! Issue #7: Portable serving benchmark extraction & compatibility contract
//! (tt-onix `qwen38_benchmark.py`, LGPL-2.1-only provenance, `onix.qwen38-benchmark.v1`),
//! offline artifact reducer & tamper verification, branch-aware agent trace replay
//! (TensorRT-LLM blog27), Eunomia queue/small-write transport controls, and OASYS RLM
//! recursive-call trace profile.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, opt_u64, quote, static_str_array, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

/// Shared workload & endpoint provenance identity reused by `#5`, `#6`, `#8`, and `#11`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkloadIdentity {
    pub schema_version: String,
    pub profile_id: String,
    pub workload_digest: DigestRef,
    pub model_id: String,
    pub checkpoint_digest: DigestRef,
    pub tokenizer_digest: DigestRef,
    pub chat_template_digest: DigestRef,
    pub precision: String,
    pub backend_revision: String,
    pub hardware_label: String,
    pub cache_policy: String,
}

impl WorkloadIdentity {
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\"schema_version\":{},\"profile_id\":{},\"workload_digest\":{},",
                "\"model_id\":{},\"checkpoint_digest\":{},\"tokenizer_digest\":{},",
                "\"chat_template_digest\":{},\"precision\":{},\"backend_revision\":{},",
                "\"hardware_label\":{},\"cache_policy\":{}}}"
            ),
            quote(&self.schema_version),
            quote(&self.profile_id),
            self.workload_digest.to_json(),
            quote(&self.model_id),
            self.checkpoint_digest.to_json(),
            self.tokenizer_digest.to_json(),
            self.chat_template_digest.to_json(),
            quote(&self.precision),
            quote(&self.backend_revision),
            quote(&self.hardware_label),
            quote(&self.cache_policy),
        )
    }
}

/// Donor provenance record preserving `tt-onix` commit, file list, and `LGPL-2.1-only` notice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DonorProvenance {
    pub donor_repo: String,
    pub donor_commit: String,
    pub license: String,
    pub legacy_schema: String,
    pub imported_files: Vec<String>,
    pub compatibility_package: String,
    pub rollback_pin_required: bool,
    pub metric_semantics_notes: Vec<String>,
}

/// Single retained client call observation under `onix.qwen38-benchmark.v1`.
#[derive(Debug, Clone, PartialEq)]
pub struct RetainedCallRecord {
    pub request_id: String,
    pub endpoint_url: String,
    pub model_id: String,
    pub status: OutcomeStatus,
    pub ttft_ms: Option<f64>,
    pub total_latency_ms: Option<f64>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub sse_chunk_count: u64,
    pub greedy_text_matches_baseline: Option<bool>,
    pub task_quality_passed: Option<bool>,
    pub authorization_header_redacted: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClaimedSummary {
    pub planned_requests: usize,
    pub accepted_requests: usize,
    pub total_completion_tokens: u64,
    pub p50_call_mean_itl_ms: Option<f64>,
    pub p95_call_mean_itl_ms: Option<f64>,
    pub claimed_greedy_parity_pass: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OfflineReductionVerdict {
    pub fixture_id: String,
    pub valid: bool,
    pub rejection_reasons: Vec<String>,
    pub planned: usize,
    pub accepted: usize,
    pub failed_or_unfinished: usize,
    pub recomputed_completion_tokens: Option<u64>,
    pub recomputed_p50_call_mean_itl_ms: Option<f64>,
    pub recomputed_p95_call_mean_itl_ms: Option<f64>,
    pub parity_passed: bool,
    pub throughput_claim_eligible: bool,
}

fn nearest_rank_percentile(sorted: &[f64], p_num: usize, p_den: usize) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (sorted.len() * p_num).div_ceil(p_den);
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    Some(sorted[idx])
}

fn is_safe_endpoint_url(url: &str) -> bool {
    let valid_scheme = url.starts_with("http://") || url.starts_with("https://");
    let has_userinfo = url
        .split_once("://")
        .map(|(_, rest)| rest.split('/').next().unwrap_or("").contains('@'))
        .unwrap_or(true);
    valid_scheme && !has_userinfo && !url.contains("token=") && !url.contains("api_key=")
}

/// Deterministically reduce and tamper-check a retained artifact without contacting any model.
pub fn reduce_and_verify_artifact(
    fixture_id: &str,
    expected_model_id: &str,
    records: &[RetainedCallRecord],
    claimed: &ClaimedSummary,
) -> OfflineReductionVerdict {
    let mut reasons = Vec::new();
    let planned = records.len();
    if planned != claimed.planned_requests {
        reasons.push(format!(
            "planned_count_mismatch: observed={planned} claimed={}",
            claimed.planned_requests
        ));
    }

    let mut accepted = 0usize;
    let mut failed_or_unfinished = 0usize;
    let mut total_tokens = 0u64;
    let mut missing_usage = false;
    let mut call_mean_itls = Vec::new();
    let mut parity_passed = true;

    for rec in records {
        if !rec.authorization_header_redacted || !is_safe_endpoint_url(&rec.endpoint_url) {
            reasons.push(format!(
                "credential_or_url_validation_failed:{}",
                rec.request_id
            ));
        }
        if rec.model_id != expected_model_id {
            reasons.push(format!("model_identity_mismatch:{}", rec.request_id));
        }
        if rec.status == OutcomeStatus::Accepted {
            accepted += 1;
            match (rec.completion_tokens, rec.ttft_ms, rec.total_latency_ms) {
                (Some(toks), Some(ttft), Some(total))
                    if toks > 0 && ttft.is_finite() && total.is_finite() && total >= ttft =>
                {
                    total_tokens = total_tokens.saturating_add(toks);
                    let gen_ms = total - ttft;
                    let denom = toks.saturating_sub(1).max(1) as f64;
                    call_mean_itls.push(gen_ms / denom);
                }
                _ => {
                    missing_usage = true;
                    reasons.push(format!(
                        "missing_usage_or_invalid_timing:{}",
                        rec.request_id
                    ));
                }
            }
        } else {
            failed_or_unfinished += 1;
        }
        if rec.greedy_text_matches_baseline == Some(false) {
            parity_passed = false;
        }
    }

    if accepted != claimed.accepted_requests {
        reasons.push(format!(
            "tampered_accepted_count: observed={accepted} claimed={}",
            claimed.accepted_requests
        ));
    }
    if !missing_usage && total_tokens != claimed.total_completion_tokens {
        reasons.push(format!(
            "tampered_completion_tokens: observed={total_tokens} claimed={}",
            claimed.total_completion_tokens
        ));
    }
    if claimed.claimed_greedy_parity_pass && !parity_passed {
        reasons.push("task_pass_cannot_override_failed_greedy_parity".into());
    }

    call_mean_itls.sort_by(f64::total_cmp);
    let p50 = nearest_rank_percentile(&call_mean_itls, 50, 100);
    let p95 = nearest_rank_percentile(&call_mean_itls, 95, 100);

    let check_stat =
        |label: &str, obs: Option<f64>, clm: Option<f64>, out: &mut Vec<String>| match (obs, clm) {
            (Some(o), Some(c)) => {
                if !c.is_finite() || (o - c).abs() > 1e-6 {
                    out.push(format!("tampered_{label}"));
                }
            }
            (None, None) => {}
            _ => out.push(format!("tampered_{label}_presence_mismatch")),
        };
    check_stat(
        "p50_call_mean_itl",
        p50,
        claimed.p50_call_mean_itl_ms,
        &mut reasons,
    );
    check_stat(
        "p95_call_mean_itl",
        p95,
        claimed.p95_call_mean_itl_ms,
        &mut reasons,
    );

    let throughput_claim_eligible = !missing_usage && accepted > 0 && reasons.is_empty();
    OfflineReductionVerdict {
        fixture_id: fixture_id.into(),
        valid: reasons.is_empty(),
        rejection_reasons: reasons,
        planned,
        accepted,
        failed_or_unfinished,
        recomputed_completion_tokens: (!missing_usage).then_some(total_tokens),
        recomputed_p50_call_mean_itl_ms: p50,
        recomputed_p95_call_mean_itl_ms: p95,
        parity_passed,
        throughput_claim_eligible,
    }
}

/// Branch-aware agent trace replay node (TensorRT-LLM blog27 profile).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentTraceEvent {
    pub event_id: String,
    pub job_id: String,
    pub parent_ids: Vec<String>,
    pub branch_group: Option<String>,
    pub prefix_digest: String,
    pub expected_prefix_digest: String,
    pub reasoning_tokens_stripped: bool,
    pub declared_tool_delay_ms: u64,
    pub observed_tool_delay_ms: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub model_service_ms: u64,
    pub status: OutcomeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchTraceVerdict {
    pub control_id: String,
    pub valid: bool,
    pub violations: Vec<String>,
    pub planned_jobs: usize,
    pub completed_jobs: usize,
    pub tool_delay_inclusive_wall_ms: u64,
    pub total_model_service_ms: u64,
}

pub fn validate_branch_trace(
    control_id: &str,
    expected_event_ids: &[&str],
    events: &[AgentTraceEvent],
    planned_jobs: usize,
) -> BranchTraceVerdict {
    let mut violations = Vec::new();
    let mut by_id: BTreeMap<&str, &AgentTraceEvent> = BTreeMap::new();

    for ev in events {
        if by_id.insert(ev.event_id.as_str(), ev).is_some() {
            violations.push(format!("duplicate_event:{}", ev.event_id));
        }
        if ev.prefix_digest != ev.expected_prefix_digest {
            violations.push(format!("randomized_shared_prefix:{}", ev.event_id));
        }
        if !ev.reasoning_tokens_stripped {
            violations.push(format!("retained_thinking_tokens:{}", ev.event_id));
        }
        if ev.declared_tool_delay_ms > 0 && ev.observed_tool_delay_ms < ev.declared_tool_delay_ms {
            violations.push(format!("skipped_tool_delay:{}", ev.event_id));
        }
    }

    let mut any_missing_expected = false;
    for &exp in expected_event_ids {
        if !by_id.contains_key(exp) {
            any_missing_expected = true;
            violations.push(format!("missing_event:{exp}"));
        }
    }

    // Check join dependencies (no early join before all parents finish).
    for ev in events {
        for pid in &ev.parent_ids {
            match by_id.get(pid.as_str()) {
                Some(parent) => {
                    if ev.start_ms < parent.end_ms {
                        violations.push(format!("early_join:{}<-{}", ev.event_id, pid));
                    }
                }
                None => violations.push(format!("missing_parent:{}<-{}", ev.event_id, pid)),
            }
        }
    }

    // Check parallel branch groups actually overlap rather than being serialized.
    let mut groups: BTreeMap<&str, Vec<&AgentTraceEvent>> = BTreeMap::new();
    for ev in events {
        if let Some(ref g) = ev.branch_group {
            groups.entry(g.as_str()).or_default().push(ev);
        }
    }
    for (group, members) in groups {
        if members.len() >= 2 {
            let mut any_overlap = false;
            for i in 0..members.len() {
                for j in (i + 1)..members.len() {
                    if members[i].start_ms < members[j].end_ms
                        && members[j].start_ms < members[i].end_ms
                    {
                        any_overlap = true;
                    }
                }
            }
            if !any_overlap {
                violations.push(format!("serialized_parallel_branches:{group}"));
            }
        }
    }

    let mut job_all_accepted: BTreeMap<&str, bool> = BTreeMap::new();
    for ev in events {
        let entry = job_all_accepted.entry(ev.job_id.as_str()).or_insert(true);
        if ev.status != OutcomeStatus::Accepted || any_missing_expected {
            *entry = false;
        }
    }
    let completed_jobs = job_all_accepted.values().filter(|&&ok| ok).count();
    let min_start = events.iter().map(|e| e.start_ms).min().unwrap_or(0);
    let max_end = events.iter().map(|e| e.end_ms).max().unwrap_or(0);
    let total_model_service_ms: u64 = events.iter().map(|e| e.model_service_ms).sum();

    BranchTraceVerdict {
        control_id: control_id.into(),
        valid: violations.is_empty(),
        violations,
        planned_jobs,
        completed_jobs,
        tool_delay_inclusive_wall_ms: max_end.saturating_sub(min_start),
        total_model_service_ms,
    }
}

/// Eunomia queue & small-write transport control record.
#[derive(Debug, Clone, PartialEq)]
pub struct EunomiaTransportControl {
    pub control_id: String,
    pub connection_close_preserved: bool,
    pub tcp_nodelay: bool,
    pub queue_capacity: usize,
    pub endpoint_instrumented: bool,
    pub per_process_cpu_pct: f64,
    pub whole_host_cpu_pct: f64,
    pub overlapping_cpu_summed: bool,
    pub cancelled_connections_reclaimed: bool,
    pub bpf_kernel_loop_justified: bool,
    pub causal_phase_attribution_allowed: bool,
}

/// OASYS RLM recursive-call trace node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlmTraceCall {
    pub call_id: String,
    pub parent_call_id: Option<String>,
    pub expected_parent_id: Option<String>,
    pub is_recursive_subcall: bool,
    pub root_context_tokens: u64,
    pub call_total_tokens: Option<u64>,
    pub start_ms: u64,
    pub end_ms: u64,
    pub replay_output_matches_recorded: bool,
    pub status: OutcomeStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlmTraceVerdict {
    pub control_id: String,
    pub trace_shape: String,
    pub valid: bool,
    pub violations: Vec<String>,
    pub root_context_peak_tokens: u64,
    pub aggregate_root_plus_leaf_tokens: Option<u64>,
    pub plain_subcalls: usize,
    pub recursive_subcalls: usize,
}

pub fn validate_rlm_trace(
    control_id: &str,
    trace_shape: &str,
    expected_calls: &[&str],
    calls: &[RlmTraceCall],
) -> RlmTraceVerdict {
    let mut violations = Vec::new();
    let mut seen: BTreeMap<&str, &RlmTraceCall> = BTreeMap::new();
    let mut root_peak = 0u64;
    let mut agg_tokens = 0u64;
    let mut any_missing_usage = false;
    let mut plain_subcalls = 0usize;
    let mut recursive_subcalls = 0usize;

    for call in calls {
        if seen.insert(call.call_id.as_str(), call).is_some() {
            violations.push(format!("duplicate_child_event:{}", call.call_id));
        }
        if call.parent_call_id != call.expected_parent_id {
            violations.push(format!("wrong_parent_attribution:{}", call.call_id));
        }
        if !call.replay_output_matches_recorded {
            violations.push(format!(
                "incompatible_replay_after_changed_output:{}",
                call.call_id
            ));
        }
        if let Some(toks) = call.call_total_tokens {
            agg_tokens = agg_tokens.saturating_add(toks);
        } else {
            any_missing_usage = true;
            violations.push(format!("absent_usage_accounting:{}", call.call_id));
        }
        if call.parent_call_id.is_none() {
            root_peak = root_peak.max(call.root_context_tokens);
        } else if call.is_recursive_subcall {
            recursive_subcalls += 1;
        } else {
            plain_subcalls += 1;
        }
    }

    for &exp in expected_calls {
        if !seen.contains_key(exp) {
            violations.push(format!("dropped_child_event:{exp}"));
        }
    }

    // Root/parent ordering: parent must exist, start at or before child, and end at or after child.
    for call in calls {
        if let Some(ref pid) = call.parent_call_id {
            match seen.get(pid.as_str()) {
                Some(parent) => {
                    if call.start_ms < parent.start_ms {
                        violations.push(format!(
                            "child_started_before_parent:{}->{}",
                            pid, call.call_id
                        ));
                    }
                    if parent.end_ms < call.end_ms {
                        violations.push(format!("premature_join:{}->{}", pid, call.call_id));
                    }
                }
                None => {
                    violations.push(format!(
                        "missing_parent_reference:{}->{}",
                        pid, call.call_id
                    ));
                }
            }
        }
    }

    RlmTraceVerdict {
        control_id: control_id.into(),
        trace_shape: trace_shape.into(),
        valid: violations.is_empty(),
        violations,
        root_context_peak_tokens: root_peak,
        aggregate_root_plus_leaf_tokens: (!any_missing_usage).then_some(agg_tokens),
        plain_subcalls,
        recursive_subcalls,
    }
}

#[derive(Debug, Clone)]
pub struct ServingBenchReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub donor: DonorProvenance,
    pub workload_identity: WorkloadIdentity,
    pub offline_reductions: Vec<OfflineReductionVerdict>,
    pub branch_trace_controls: Vec<BranchTraceVerdict>,
    pub eunomia_controls: Vec<EunomiaTransportControl>,
    pub rlm_trace_controls: Vec<RlmTraceVerdict>,
}

pub fn default_workload_identity() -> WorkloadIdentity {
    WorkloadIdentity {
        schema_version: "onix.qwen38-benchmark.v1".into(),
        profile_id: "qwen38-p150x2-compat-v1".into(),
        workload_digest: DigestRef::sha256(b"qwen38-corpus-8prompts+agentic-trace-v1"),
        model_id: "Qwen/Qwen3.8-27B-pinned".into(),
        checkpoint_digest: DigestRef::sha256(b"qwen38-27b-checkpoint-pinned-placeholder"),
        tokenizer_digest: DigestRef::sha256(b"qwen38-tokenizer-v1"),
        chat_template_digest: DigestRef::sha256(b"qwen-thinking-v1"),
        precision: "bf16-reference-pinned".into(),
        backend_revision: "tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a".into(),
        hardware_label: "device_free_contract_fixture".into(),
        cache_policy: "server_cache_unknown_client_only".into(),
    }
}

pub fn run_default_harness() -> ServingBenchReport {
    let donor = DonorProvenance {
        donor_repo: "OnixResearch/tt-onix".into(),
        donor_commit: "5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a".into(),
        license: "LGPL-2.1-only".into(),
        legacy_schema: "onix.qwen38-benchmark.v1".into(),
        imported_files: vec![
            "pkgs/qwen38-benchmark/qwen38_benchmark.py".into(),
            "pkgs/qwen38-benchmark/corpus.json".into(),
            "pkgs/qwen38-benchmark/agentic_trace.json".into(),
            "pkgs/qwen38-benchmark/default.nix".into(),
            "docs/qwen38-benchmark.md".into(),
        ],
        compatibility_package: "tt-onix:qwen38-benchmark (pinned rollback)".into(),
        rollback_pin_required: true,
        metric_semantics_notes: vec![
            "ttft is first nonempty content delivery, not direct device generation time".into(),
            "itl is a per-call mean; p95 is over per-call means, not individual token gaps".into(),
            "agentic mode replays frozen turns; it is not live task accuracy".into(),
            "client prefix overlap is not a server-confirmed cache hit".into(),
            "constant-watt accounting is not measured task-energy integration".into(),
        ],
    };

    let workload_identity = default_workload_identity();
    let model = workload_identity.model_id.as_str();

    // 1. Offline artifact reducer positive & negative controls.
    let good_records = vec![
        RetainedCallRecord {
            request_id: "call-01".into(),
            endpoint_url: "http://127.0.0.1:8000/v1/chat/completions".into(),
            model_id: model.into(),
            status: OutcomeStatus::Accepted,
            ttft_ms: Some(120.0),
            total_latency_ms: Some(620.0),
            prompt_tokens: Some(256),
            completion_tokens: Some(51),
            sse_chunk_count: 34,
            greedy_text_matches_baseline: Some(true),
            task_quality_passed: Some(true),
            authorization_header_redacted: true,
        },
        RetainedCallRecord {
            request_id: "call-02".into(),
            endpoint_url: "http://127.0.0.1:8000/v1/chat/completions".into(),
            model_id: model.into(),
            status: OutcomeStatus::Accepted,
            ttft_ms: Some(140.0),
            total_latency_ms: Some(740.0),
            prompt_tokens: Some(300),
            completion_tokens: Some(51),
            sse_chunk_count: 40,
            greedy_text_matches_baseline: Some(true),
            task_quality_passed: Some(true),
            authorization_header_redacted: true,
        },
        RetainedCallRecord {
            request_id: "call-03-timeout-retained".into(),
            endpoint_url: "http://127.0.0.1:8000/v1/chat/completions".into(),
            model_id: model.into(),
            status: OutcomeStatus::Timeout,
            ttft_ms: None,
            total_latency_ms: Some(5000.0),
            prompt_tokens: Some(512),
            completion_tokens: None,
            sse_chunk_count: 0,
            greedy_text_matches_baseline: None,
            task_quality_passed: None,
            authorization_header_redacted: true,
        },
    ];
    let good_summary = ClaimedSummary {
        planned_requests: 3,
        accepted_requests: 2,
        total_completion_tokens: 102,
        p50_call_mean_itl_ms: Some(10.0),
        p95_call_mean_itl_ms: Some(12.0),
        claimed_greedy_parity_pass: true,
    };

    let tampered_summary = ClaimedSummary {
        planned_requests: 3,
        accepted_requests: 3, // Dropped timeout into accepted!
        total_completion_tokens: 150,
        p50_call_mean_itl_ms: Some(10.0),
        p95_call_mean_itl_ms: Some(8.0),
        claimed_greedy_parity_pass: true,
    };

    let mut missing_usage_records = good_records.clone();
    missing_usage_records[0].completion_tokens = None;

    let mut parity_override_records = good_records.clone();
    parity_override_records[1].greedy_text_matches_baseline = Some(false);
    parity_override_records[1].task_quality_passed = Some(true);

    let mut cred_leak_records = good_records.clone();
    cred_leak_records[0].endpoint_url = "https://user:secret@host.invalid/v1/chat".into();
    cred_leak_records[0].authorization_header_redacted = false;

    let offline_reductions = vec![
        reduce_and_verify_artifact(
            "positive-legacy-v1-replay",
            model,
            &good_records,
            &good_summary,
        ),
        reduce_and_verify_artifact(
            "negative-tampered-summary-and-dropped-timeout",
            model,
            &good_records,
            &tampered_summary,
        ),
        reduce_and_verify_artifact(
            "negative-missing-usage-refuses-throughput",
            model,
            &missing_usage_records,
            &good_summary,
        ),
        reduce_and_verify_artifact(
            "negative-qwen18-task-pass-cannot-override-parity-fail",
            model,
            &parity_override_records,
            &good_summary,
        ),
        reduce_and_verify_artifact(
            "negative-credential-and-url-leak",
            model,
            &cred_leak_records,
            &good_summary,
        ),
    ];

    // 2. Branch-aware agent trace replay (positive + all 6 required negative controls).
    let base_branch_trace = vec![
        AgentTraceEvent {
            event_id: "root-turn".into(),
            job_id: "job-1".into(),
            parent_ids: vec![],
            branch_group: None,
            prefix_digest: "pfx-root".into(),
            expected_prefix_digest: "pfx-root".into(),
            reasoning_tokens_stripped: true,
            declared_tool_delay_ms: 0,
            observed_tool_delay_ms: 0,
            start_ms: 0,
            end_ms: 100,
            model_service_ms: 100,
            status: OutcomeStatus::Accepted,
        },
        AgentTraceEvent {
            event_id: "branch-a".into(),
            job_id: "job-1".into(),
            parent_ids: vec!["root-turn".into()],
            branch_group: Some("fanout-1".into()),
            prefix_digest: "pfx-shared-ab".into(),
            expected_prefix_digest: "pfx-shared-ab".into(),
            reasoning_tokens_stripped: true,
            declared_tool_delay_ms: 40,
            observed_tool_delay_ms: 40,
            start_ms: 140,
            end_ms: 260,
            model_service_ms: 120,
            status: OutcomeStatus::Accepted,
        },
        AgentTraceEvent {
            event_id: "branch-b".into(),
            job_id: "job-1".into(),
            parent_ids: vec!["root-turn".into()],
            branch_group: Some("fanout-1".into()),
            prefix_digest: "pfx-shared-ab".into(),
            expected_prefix_digest: "pfx-shared-ab".into(),
            reasoning_tokens_stripped: true,
            declared_tool_delay_ms: 50,
            observed_tool_delay_ms: 50,
            start_ms: 150,
            end_ms: 290,
            model_service_ms: 140,
            status: OutcomeStatus::Accepted,
        },
        AgentTraceEvent {
            event_id: "join-turn".into(),
            job_id: "job-1".into(),
            parent_ids: vec!["branch-a".into(), "branch-b".into()],
            branch_group: None,
            prefix_digest: "pfx-join".into(),
            expected_prefix_digest: "pfx-join".into(),
            reasoning_tokens_stripped: true,
            declared_tool_delay_ms: 10,
            observed_tool_delay_ms: 10,
            start_ms: 300,
            end_ms: 380,
            model_service_ms: 80,
            status: OutcomeStatus::Accepted,
        },
    ];
    let expected_ids = ["root-turn", "branch-a", "branch-b", "join-turn"];

    let mut rand_prefix = base_branch_trace.clone();
    rand_prefix[2].prefix_digest = "pfx-accidental-random-nonce".into();

    let mut retained_think = base_branch_trace.clone();
    retained_think[1].reasoning_tokens_stripped = false;

    let mut skipped_delay = base_branch_trace.clone();
    skipped_delay[1].observed_tool_delay_ms = 0;

    let mut serialized_branches = base_branch_trace.clone();
    serialized_branches[2].start_ms = 260;
    serialized_branches[2].end_ms = 400;
    serialized_branches[3].start_ms = 410;
    serialized_branches[3].end_ms = 490;

    let mut early_join = base_branch_trace.clone();
    early_join[3].start_ms = 270; // Before branch-b ends at 290!

    let mut dup_missing = base_branch_trace[0..3].to_vec();
    dup_missing.push(base_branch_trace[1].clone());

    let branch_trace_controls = vec![
        validate_branch_trace(
            "positive-two-branch-fork-join",
            &expected_ids,
            &base_branch_trace,
            1,
        ),
        validate_branch_trace(
            "negative-randomized-shared-prefix",
            &expected_ids,
            &rand_prefix,
            1,
        ),
        validate_branch_trace(
            "negative-retained-thinking-tokens",
            &expected_ids,
            &retained_think,
            1,
        ),
        validate_branch_trace(
            "negative-skipped-tool-delay",
            &expected_ids,
            &skipped_delay,
            1,
        ),
        validate_branch_trace(
            "negative-serialized-parallel-branches",
            &expected_ids,
            &serialized_branches,
            1,
        ),
        validate_branch_trace("negative-early-join", &expected_ids, &early_join, 1),
        validate_branch_trace(
            "negative-missing-and-duplicate-events",
            &expected_ids,
            &dup_missing,
            1,
        ),
    ];

    // 3. Eunomia queue & small-write transport controls.
    let eunomia_controls = vec![
        EunomiaTransportControl {
            control_id: "baseline-connection-close-userspace-queue".into(),
            connection_close_preserved: true,
            tcp_nodelay: true,
            queue_capacity: 16,
            endpoint_instrumented: true,
            per_process_cpu_pct: 42.0,
            whole_host_cpu_pct: 58.0,
            overlapping_cpu_summed: false,
            cancelled_connections_reclaimed: true,
            bpf_kernel_loop_justified: false,
            causal_phase_attribution_allowed: true,
        },
        EunomiaTransportControl {
            control_id: "negative-client-only-cannot-assign-server-phase".into(),
            connection_close_preserved: true,
            tcp_nodelay: false,
            queue_capacity: 32,
            endpoint_instrumented: false,
            per_process_cpu_pct: 38.0,
            whole_host_cpu_pct: 55.0,
            overlapping_cpu_summed: false,
            cancelled_connections_reclaimed: true,
            bpf_kernel_loop_justified: false,
            causal_phase_attribution_allowed: false,
        },
    ];

    // 4. OASYS RLM recursive-call trace profile (2 shapes + 5 negative controls).
    let rlm_external = vec![
        RlmTraceCall {
            call_id: "root".into(),
            parent_call_id: None,
            expected_parent_id: None,
            is_recursive_subcall: false,
            root_context_tokens: 1800,
            call_total_tokens: Some(2100),
            start_ms: 0,
            end_ms: 600,
            replay_output_matches_recorded: true,
            status: OutcomeStatus::Accepted,
        },
        RlmTraceCall {
            call_id: "child-plain-1".into(),
            parent_call_id: Some("root".into()),
            expected_parent_id: Some("root".into()),
            is_recursive_subcall: false,
            root_context_tokens: 0,
            call_total_tokens: Some(1400),
            start_ms: 80,
            end_ms: 260,
            replay_output_matches_recorded: true,
            status: OutcomeStatus::Accepted,
        },
        RlmTraceCall {
            call_id: "child-recursive-2".into(),
            parent_call_id: Some("root".into()),
            expected_parent_id: Some("root".into()),
            is_recursive_subcall: true,
            root_context_tokens: 0,
            call_total_tokens: Some(2400),
            start_ms: 120,
            end_ms: 480,
            replay_output_matches_recorded: true,
            status: OutcomeStatus::Accepted,
        },
    ];
    let mut rlm_appended = rlm_external.clone();
    rlm_appended[0].root_context_tokens = 4200;
    rlm_appended[0].call_total_tokens = Some(4600);

    let rlm_expected = ["root", "child-plain-1", "child-recursive-2"];
    let mut rlm_drop_dup = vec![
        rlm_external[0].clone(),
        rlm_external[1].clone(),
        rlm_external[1].clone(),
    ];
    rlm_drop_dup[2].start_ms = 90;

    let mut rlm_premature = rlm_external.clone();
    rlm_premature[0].end_ms = 400; // Root finished before child-recursive-2 (480ms).

    let mut rlm_wrong_parent = rlm_external.clone();
    rlm_wrong_parent[2].parent_call_id = Some("child-plain-1".into());

    let mut rlm_changed_out = rlm_external.clone();
    rlm_changed_out[1].replay_output_matches_recorded = false;

    let mut rlm_no_usage = rlm_external.clone();
    rlm_no_usage[2].call_total_tokens = None;

    let rlm_trace_controls = vec![
        validate_rlm_trace(
            "positive-rlm-externally-retained",
            "externally_retained_results",
            &rlm_expected,
            &rlm_external,
        ),
        validate_rlm_trace(
            "positive-rlm-appended-children",
            "appended_child_results",
            &rlm_expected,
            &rlm_appended,
        ),
        validate_rlm_trace(
            "negative-rlm-dropped-and-duplicate-child",
            "externally_retained_results",
            &rlm_expected,
            &rlm_drop_dup,
        ),
        validate_rlm_trace(
            "negative-rlm-premature-join",
            "externally_retained_results",
            &rlm_expected,
            &rlm_premature,
        ),
        validate_rlm_trace(
            "negative-rlm-wrong-parent",
            "externally_retained_results",
            &rlm_expected,
            &rlm_wrong_parent,
        ),
        validate_rlm_trace(
            "negative-rlm-changed-output",
            "externally_retained_results",
            &rlm_expected,
            &rlm_changed_out,
        ),
        validate_rlm_trace(
            "negative-rlm-absent-usage",
            "externally_retained_results",
            &rlm_expected,
            &rlm_no_usage,
        ),
    ];

    ServingBenchReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "live_http_sse_serving_endpoint_and_tt_p150x2",
            "Device-free reducer, tamper, branch-trace, transport, and RLM fixtures verified locally; live endpoint & tt-onix compatibility package execution remain external hardware gates.",
        ),
        donor,
        workload_identity,
        offline_reductions,
        branch_trace_controls,
        eunomia_controls,
        rlm_trace_controls,
    }
}

pub fn inputs_json(report: &ServingBenchReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-07-serving-bench.inputs.v1\",\n",
            "  \"issue\": 7,\n",
            "  \"execution_class\": {},\n",
            "  \"donor\": {{\"repo\":{},\"commit\":{},\"license\":{},\"legacy_schema\":{},\"imported_files\":{},\"compatibility_package\":{},\"rollback_pin_required\":{}}},\n",
            "  \"workload_identity\": {},\n",
            "  \"metric_semantics_notes\": {},\n",
            "  \"non_gating_unverified_author_claims\": [\n",
            "    {{\"source\":\"eunomia.dev/blog/2026/10/17/bpf-gpu-server\",\"status\":\"negative_evidence_non_gating\",\"note\":\"Single RTX 5090 binary protocol without TLS; BPF request loop failed CPU/tail-latency win criteria\"}},\n",
            "    {{\"source\":\"TensorRT-LLM@8070e5d78aab872abf71297ae06b1cc6af2d7151 blog27\",\"status\":\"design_reference_non_gating\",\"note\":\"Upstream DGX Spark support doc does not qualify local build or host\"}},\n",
            "    {{\"source\":\"OASYS RLMs arXiv:2512.24601v3\",\"status\":\"workload_shape_reference_non_gating\",\"note\":\"Serving trace shape only; live task-solving owned by systems-lab#24\"}}\n",
            "  ]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.donor.donor_repo),
        quote(&report.donor.donor_commit),
        quote(&report.donor.license),
        quote(&report.donor.legacy_schema),
        str_array(&report.donor.imported_files),
        quote(&report.donor.compatibility_package),
        report.donor.rollback_pin_required,
        report.workload_identity.to_json(),
        str_array(&report.donor.metric_semantics_notes),
    )
}

pub fn results_json(report: &ServingBenchReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-07-serving-bench.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 7,").unwrap();
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

    let red_items: Vec<String> = report
        .offline_reductions
        .iter()
        .map(|r| {
            format!(
                concat!(
                    "{{\"fixture_id\":{},\"valid\":{},\"planned\":{},\"accepted\":{},",
                    "\"failed_or_unfinished\":{},\"recomputed_completion_tokens\":{},",
                    "\"recomputed_p50_call_mean_itl_ms\":{},\"recomputed_p95_call_mean_itl_ms\":{},",
                    "\"parity_passed\":{},\"throughput_claim_eligible\":{},\"rejection_reasons\":{}}}"
                ),
                quote(&r.fixture_id),
                r.valid,
                r.planned,
                r.accepted,
                r.failed_or_unfinished,
                opt_u64(r.recomputed_completion_tokens),
                opt_f64(r.recomputed_p50_call_mean_itl_ms),
                opt_f64(r.recomputed_p95_call_mean_itl_ms),
                r.parity_passed,
                r.throughput_claim_eligible,
                str_array(&r.rejection_reasons),
            )
        })
        .collect();
    writeln!(out, "  \"offline_reductions\": [{}],", red_items.join(",")).unwrap();

    let branch_items: Vec<String> = report
        .branch_trace_controls
        .iter()
        .map(|b| {
            format!(
                concat!(
                    "{{\"control_id\":{},\"valid\":{},\"planned_jobs\":{},\"completed_jobs\":{},",
                    "\"tool_delay_inclusive_wall_ms\":{},\"total_model_service_ms\":{},\"violations\":{}}}"
                ),
                quote(&b.control_id),
                b.valid,
                b.planned_jobs,
                b.completed_jobs,
                b.tool_delay_inclusive_wall_ms,
                b.total_model_service_ms,
                str_array(&b.violations),
            )
        })
        .collect();
    writeln!(
        out,
        "  \"branch_trace_controls\": [{}],",
        branch_items.join(",")
    )
    .unwrap();

    let eunomia_items: Vec<String> = report
        .eunomia_controls
        .iter()
        .map(|e| {
            format!(
                concat!(
                    "{{\"control_id\":{},\"connection_close_preserved\":{},\"tcp_nodelay\":{},",
                    "\"queue_capacity\":{},\"endpoint_instrumented\":{},\"per_process_cpu_pct\":{},",
                    "\"whole_host_cpu_pct\":{},\"overlapping_cpu_summed\":{},",
                    "\"cancelled_connections_reclaimed\":{},\"bpf_kernel_loop_justified\":{},",
                    "\"causal_phase_attribution_allowed\":{}}}"
                ),
                quote(&e.control_id),
                e.connection_close_preserved,
                e.tcp_nodelay,
                e.queue_capacity,
                e.endpoint_instrumented,
                finite_f64(e.per_process_cpu_pct),
                finite_f64(e.whole_host_cpu_pct),
                e.overlapping_cpu_summed,
                e.cancelled_connections_reclaimed,
                e.bpf_kernel_loop_justified,
                e.causal_phase_attribution_allowed,
            )
        })
        .collect();
    writeln!(
        out,
        "  \"eunomia_transport_controls\": [{}],",
        eunomia_items.join(",")
    )
    .unwrap();

    let rlm_items: Vec<String> = report
        .rlm_trace_controls
        .iter()
        .map(|r| {
            format!(
                concat!(
                    "{{\"control_id\":{},\"trace_shape\":{},\"valid\":{},",
                    "\"root_context_peak_tokens\":{},\"aggregate_root_plus_leaf_tokens\":{},",
                    "\"plain_subcalls\":{},\"recursive_subcalls\":{},\"violations\":{}}}"
                ),
                quote(&r.control_id),
                quote(&r.trace_shape),
                r.valid,
                r.root_context_peak_tokens,
                opt_u64(r.aggregate_root_plus_leaf_tokens),
                r.plain_subcalls,
                r.recursive_subcalls,
                str_array(&r.violations),
            )
        })
        .collect();
    writeln!(out, "  \"rlm_trace_controls\": [{}],", rlm_items.join(",")).unwrap();

    let distinct_violation_tags: BTreeSet<&str> = report
        .branch_trace_controls
        .iter()
        .flat_map(|b| {
            b.violations
                .iter()
                .map(|s| s.split(':').next().unwrap_or(""))
        })
        .chain(report.rlm_trace_controls.iter().flat_map(|r| {
            r.violations
                .iter()
                .map(|s| s.split(':').next().unwrap_or(""))
        }))
        .collect();
    let tags: Vec<&str> = distinct_violation_tags.into_iter().collect();
    writeln!(
        out,
        "  \"verified_negative_control_classes\": {}",
        static_str_array(&tags)
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &ServingBenchReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #7: Portable Serving Benchmark Extraction & Trace Profiles"
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
        "- Donor pin: `{}@{}` (`{}`, schema `{}`)",
        report.donor.donor_repo,
        report.donor.donor_commit,
        report.donor.license,
        report.donor.legacy_schema
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Offline Reducer & Tamper Controls").unwrap();
    writeln!(
        out,
        "| Fixture | Valid | Planned | Accepted | Failed/Unfinished | Reasons |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | ---: | ---: | ---: | --- |").unwrap();
    for r in &report.offline_reductions {
        writeln!(
            out,
            "| `{}` | `{}` | {} | {} | {} | {} |",
            r.fixture_id,
            r.valid,
            r.planned,
            r.accepted,
            r.failed_or_unfinished,
            if r.rejection_reasons.is_empty() {
                "none".to_string()
            } else {
                r.rejection_reasons.join(", ")
            }
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Branch-Aware Agent Trace & OASYS RLM Controls").unwrap();
    writeln!(
        out,
        "- Branch-trace controls evaluated: {} (1 positive, 6 negative).",
        report.branch_trace_controls.len()
    )
    .unwrap();
    writeln!(
        out,
        "- Eunomia transport controls evaluated: {} (`Connection: close` preserved, BPF kernel loop not justified).",
        report.eunomia_controls.len()
    )
    .unwrap();
    writeln!(
        out,
        "- OASYS RLM trace controls evaluated: {} (2 shapes: `externally_retained_results` vs `appended_child_results`, 5 negative controls).",
        report.rlm_trace_controls.len()
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Device-free contract verification only. No live HTTP/SSE model server or Tenstorrent P150 hardware was invoked."
    )
    .unwrap();
    out
}
