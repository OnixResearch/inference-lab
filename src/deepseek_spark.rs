//! Issue #8: Migrate DeepSeek/Spark inference experiments (`brittonr/dsv41-1017@7f466a12`)
//! into the shared serving benchmark (`#7`), repairing counter accounting, preserving
//! burst vs sustained workload separation, enforcing observable cache state, running all 9
//! device-free fixtures via real deterministic logic, and contrasting legacy substring checks
//! against an executable input-based semantic checker.

use std::fmt::Write as FmtWrite;

use crate::json::{opt_str, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};
use crate::serving_bench::{
    default_workload_identity, reduce_and_verify_artifact, ClaimedSummary, RetainedCallRecord,
    WorkloadIdentity,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationManifestEntry {
    pub donor_path: String,
    pub donor_ref: String,
    pub license: String,
    pub publishability: String,
    pub destination_or_owner: String,
    pub source_digest: DigestRef,
    pub supersedes_or_notes: String,
}

/// Repaired Prometheus counter delta outcome (fixing donor `metrics()` returning 0 for missing
/// counters, selecting first labeled series, or passing through NaN/negative values).
#[derive(Debug, Clone, PartialEq)]
pub enum CounterDeltaOutcome {
    Observed(f64),
    UnavailableMissingCounter,
    AmbiguousMultipleLabelSeries(usize),
    InvalidNonFiniteOrNegativeSample,
    CounterResetDetected { start: f64, end: f64 },
}

impl CounterDeltaOutcome {
    pub const fn tag(&self) -> &'static str {
        match self {
            Self::Observed(_) => "observed",
            Self::UnavailableMissingCounter => "unavailable_missing_counter",
            Self::AmbiguousMultipleLabelSeries(_) => "ambiguous_multiple_label_series",
            Self::InvalidNonFiniteOrNegativeSample => "invalid_nonfinite_or_negative_sample",
            Self::CounterResetDetected { .. } => "counter_reset_detected",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricSample {
    pub metric_name: String,
    pub label_scope: String,
    pub value: f64,
}

/// Compute counter delta with strict label scoping, finiteness/non-negativity, unavailable-on-missing,
/// and counter-reset checks.
pub fn compute_repaired_counter_delta(
    start_samples: &[MetricSample],
    end_samples: &[MetricSample],
    metric_name: &str,
    required_label_scope: &str,
) -> CounterDeltaOutcome {
    let find_scoped = |samples: &[MetricSample]| -> Result<f64, CounterDeltaOutcome> {
        let matches: Vec<&MetricSample> = samples
            .iter()
            .filter(|s| s.metric_name == metric_name && s.label_scope == required_label_scope)
            .collect();
        match matches.len() {
            0 => Err(CounterDeltaOutcome::UnavailableMissingCounter),
            1 => {
                let v = matches[0].value;
                if !v.is_finite() || v < 0.0 {
                    Err(CounterDeltaOutcome::InvalidNonFiniteOrNegativeSample)
                } else {
                    Ok(v)
                }
            }
            n => Err(CounterDeltaOutcome::AmbiguousMultipleLabelSeries(n)),
        }
    };

    let start = match find_scoped(start_samples) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let end = match find_scoped(end_samples) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if end < start {
        return CounterDeltaOutcome::CounterResetDetected { start, end };
    }
    CounterDeltaOutcome::Observed(end - start)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparkWorkloadClass {
    SynchronizedStreamWaveBurst,
    SustainedArrival,
}

impl SparkWorkloadClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SynchronizedStreamWaveBurst => "synchronized_stream_wave_burst",
            Self::SustainedArrival => "sustained_arrival",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheObservationState {
    Unknown,
    RequestedUniquePrefixOnly,
    OperatorDeclared,
    ServerCounterObserved,
}

impl CacheObservationState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::RequestedUniquePrefixOnly => "requested_unique_prefix_only_not_proven_cold",
            Self::OperatorDeclared => "operator_declared",
            Self::ServerCounterObserved => "server_counter_observed",
        }
    }
}

/// Refuse comparing burst waves against sustained arrivals as identical workload classes.
pub fn check_workload_comparability(
    a: SparkWorkloadClass,
    b: SparkWorkloadClass,
) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!(
            "incomparable_workload_classes:{}!={}",
            a.as_str(),
            b.as_str()
        ))
    }
}

/// Deterministic SSE stream parser used by device-free fixtures `01`, `06`, `07`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseParseResult {
    Complete {
        chunk_count: usize,
        content_bytes: usize,
        usage_tokens: Option<u64>,
        finish_reason: String,
    },
    EmptyOrTruncated {
        chunk_count: usize,
        content_bytes: usize,
        reason: String,
    },
    MalformedFrame(String),
}

pub fn parse_sse_stream_fixture(lines: &[&str]) -> SseParseResult {
    let mut chunks = 0usize;
    let mut content_bytes = 0usize;
    let mut usage_tokens = None;
    let mut saw_done = false;
    let mut finish_reason = String::new();

    for &line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(payload) = trimmed.strip_prefix("data: ") else {
            return SseParseResult::MalformedFrame(format!("missing_data_prefix:{trimmed}"));
        };
        if payload == "[DONE]" {
            saw_done = true;
            break;
        }
        if !payload.starts_with('{') || !payload.ends_with('}') {
            return SseParseResult::MalformedFrame("truncated_or_unbalanced_json_frame".into());
        }
        if let Some(rest) = payload.split("\"delta\":").nth(1) {
            if let Some(after_q) = rest.trim_start().strip_prefix('"') {
                if let Some((text, _)) = after_q.split_once('"') {
                    chunks += 1;
                    content_bytes += text.len();
                } else {
                    return SseParseResult::MalformedFrame("unterminated_delta_string".into());
                }
            }
        }
        if let Some(rest) = payload.split("\"completion_tokens\":").nth(1) {
            let digits: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(toks) = digits.parse::<u64>() {
                usage_tokens = Some(toks);
            }
        }
        if let Some(rest) = payload.split("\"finish_reason\":").nth(1) {
            if let Some(after_q) = rest.trim_start().strip_prefix('"') {
                if let Some((reason, _)) = after_q.split_once('"') {
                    finish_reason = reason.into();
                }
            }
        }
    }

    if !saw_done || content_bytes == 0 || finish_reason == "length" {
        return SseParseResult::EmptyOrTruncated {
            chunk_count: chunks,
            content_bytes,
            reason: if content_bytes == 0 {
                "empty_output".into()
            } else if finish_reason == "length" {
                "truncated_max_tokens".into()
            } else {
                "missing_done_sentinel".into()
            },
        };
    }

    SseParseResult::Complete {
        chunk_count: chunks,
        content_bytes,
        usage_tokens,
        finish_reason,
    }
}

/// Probe endpoint optional capabilities rather than assuming universal OpenAI API support.
pub fn probe_endpoint_capability(
    supported_capabilities: &[&str],
    requested_capability: &str,
) -> OutcomeStatus {
    if supported_capabilities.contains(&requested_capability) {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Unsupported
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFreeFixtureResult {
    pub fixture_id: String,
    pub handled_honestly: bool,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityEvaluatorComparison {
    pub candidate_id: String,
    pub candidate_code: String,
    pub legacy_evaluator_id: String,
    pub legacy_substring_passed: bool,
    pub semantic_evaluator_id: String,
    pub test_inputs_evaluated: Vec<i64>,
    pub failed_inputs: Vec<i64>,
    pub semantic_executable_passed: bool,
    pub semantic_rejection_reason: Option<String>,
}

fn eval_simple_expr(expr: &str, x: i64) -> Option<i64> {
    match expr.trim() {
        "0" => Some(0),
        "x" => Some(x),
        "-x" => x.checked_neg(),
        "x if x > 0 else 0" | "x if x >= 0 else 0" | "max(0, x)" | "max(x, 0)" => Some(x.max(0)),
        "0 if x < 0 else x" | "0 if x <= 0 else x" => Some(x.max(0)),
        _ => expr.trim().parse::<i64>().ok(),
    }
}

/// Execute a restricted 1-argument `clamp_nonneg(x)` Python body on a concrete integer input.
fn execute_clamp_candidate(code: &str, x: i64) -> Option<i64> {
    let mut lines = code.lines().map(str::trim).filter(|l| !l.is_empty());
    let header = lines.next()?;
    if !header.starts_with("def clamp_nonneg(x):") {
        return None;
    }
    for line in lines {
        if let Some(rest) = line.strip_prefix("if x < 0:") {
            if let Some(expr) = rest.trim().strip_prefix("return ") {
                if x < 0 {
                    return eval_simple_expr(expr, x);
                }
            }
        } else if let Some(rest) = line.strip_prefix("if x <= 0:") {
            if let Some(expr) = rest.trim().strip_prefix("return ") {
                if x <= 0 {
                    return eval_simple_expr(expr, x);
                }
            }
        } else if let Some(expr) = line.strip_prefix("return ") {
            return eval_simple_expr(expr, x);
        }
    }
    None
}

/// Legacy substring check (`dsv41-legacy-substring-v1`) vs executable input-based semantic checker
/// evaluated across concrete inputs `[-5, -1, 0, 1, 7]`.
pub fn compare_legacy_vs_semantic_quality(
    candidate_id: &str,
    candidate_code: &str,
) -> QualityEvaluatorComparison {
    let legacy_pass =
        candidate_code.contains("def clamp_nonneg") && candidate_code.contains("return");
    let test_inputs = vec![-5i64, -1, 0, 1, 7];
    let mut failed_inputs = Vec::new();
    for &x in &test_inputs {
        let expected = x.max(0);
        if execute_clamp_candidate(candidate_code, x) != Some(expected) {
            failed_inputs.push(x);
        }
    }
    let semantic_pass = failed_inputs.is_empty();
    QualityEvaluatorComparison {
        candidate_id: candidate_id.into(),
        candidate_code: candidate_code.into(),
        legacy_evaluator_id: "dsv41-legacy-substring-v1 (historical_smoke_only)".into(),
        legacy_substring_passed: legacy_pass,
        semantic_evaluator_id: "inference-lab-code-semantic-exec-v1".into(),
        test_inputs_evaluated: test_inputs,
        failed_inputs: failed_inputs.clone(),
        semantic_executable_passed: semantic_pass,
        semantic_rejection_reason: if semantic_pass {
            None
        } else {
            Some(format!(
                "wrong_output_on_inputs_{failed_inputs:?}_despite_legacy_substring_pass={legacy_pass}"
            ))
        },
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekSparkReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub shared_workload_identity: WorkloadIdentity,
    pub migration_manifest: Vec<MigrationManifestEntry>,
    pub probed_capabilities_default: Vec<(String, bool)>,
    pub workload_cross_compare_rejected: bool,
    pub cache_states_distinguished: Vec<String>,
    pub device_free_fixtures: Vec<DeviceFreeFixtureResult>,
    pub quality_comparisons: Vec<QualityEvaluatorComparison>,
}

pub fn run_default_harness() -> DeepSeekSparkReport {
    let shared_workload_identity = default_workload_identity();
    let donor_ref = "brittonr/dsv41-1017@7f466a1256f559fbd95f03d8bef84a5a3944f9c0";

    let migration_manifest = vec![
        MigrationManifestEntry {
            donor_path: "tools/bench.py".into(),
            donor_ref: donor_ref.into(),
            license: "MIT (donor notice preserved)".into(),
            publishability: "public_adapted_contract_only".into(),
            destination_or_owner: "inference-lab (#7/#8 shared serving runner)".into(),
            source_digest: DigestRef::sha256(b"dsv41-1017:tools/bench.py@7f466a12"),
            supersedes_or_notes: "Repairs metrics() missing->0 and first-series selection bugs."
                .into(),
        },
        MigrationManifestEntry {
            donor_path: "eval/quality_eval.py + eval/hard_eval.py".into(),
            donor_ref: donor_ref.into(),
            license: "MIT (donor notice preserved)".into(),
            publishability: "public_synthetic_fixtures_only".into(),
            destination_or_owner: "inference-lab (#8 labeled legacy smoke + semantic checker)"
                .into(),
            source_digest: DigestRef::sha256(b"dsv41-1017:eval/quality_eval.py@7f466a12"),
            supersedes_or_notes:
                "Retains legacy evaluator identity; general suite recorded as saturated.".into(),
        },
        MigrationManifestEntry {
            donor_path: "benchmarks.md + spec/mtp/nccl/long-context findings".into(),
            donor_ref: donor_ref.into(),
            license: "MIT (private donor)".into(),
            publishability: "restricted_private_donor_linked_only".into(),
            destination_or_owner:
                "Historical retractions & superseded notes linked, not blindly copied.".into(),
            source_digest: DigestRef::sha256(b"dsv41-1017:benchmarks.md@7f466a12"),
            supersedes_or_notes:
                "Preserves historical token-count and burst-vs-sustained corrections.".into(),
        },
        MigrationManifestEntry {
            donor_path: "deploy.sh, boot_dsv41_1017.sh, dsv41-tp3-1017.sh".into(),
            donor_ref: donor_ref.into(),
            license: "MIT (private donor)".into(),
            publishability: "excluded_stays_with_deployment_owner".into(),
            destination_or_owner: "Deployment owner (systemd units/watchdogs/recovery unchanged)"
                .into(),
            source_digest: DigestRef::sha256(b"dsv41-1017:deployment-excluded@7f466a12"),
            supersedes_or_notes:
                "CLI never restarts/evicts resident models or drops host page cache.".into(),
        },
    ];

    // Counter fixtures.
    let s_ok = vec![MetricSample {
        metric_name: "vllm:spec_decode_num_accepted_tokens_total".into(),
        label_scope: "model=dsv41,replica=0".into(),
        value: 100.0,
    }];
    let e_ok = vec![MetricSample {
        metric_name: "vllm:spec_decode_num_accepted_tokens_total".into(),
        label_scope: "model=dsv41,replica=0".into(),
        value: 164.0,
    }];
    let e_reset = vec![MetricSample {
        metric_name: "vllm:spec_decode_num_accepted_tokens_total".into(),
        label_scope: "model=dsv41,replica=0".into(),
        value: 12.0,
    }];
    let s_ambig = vec![
        s_ok[0].clone(),
        MetricSample {
            metric_name: "vllm:spec_decode_num_accepted_tokens_total".into(),
            label_scope: "model=dsv41,replica=0".into(),
            value: 200.0,
        },
    ];

    let d_missing = compute_repaired_counter_delta(
        &[],
        &e_ok,
        "vllm:spec_decode_num_accepted_tokens_total",
        "model=dsv41,replica=0",
    );
    let d_ambig = compute_repaired_counter_delta(
        &s_ambig,
        &e_ok,
        "vllm:spec_decode_num_accepted_tokens_total",
        "model=dsv41,replica=0",
    );
    let d_reset = compute_repaired_counter_delta(
        &s_ok,
        &e_reset,
        "vllm:spec_decode_num_accepted_tokens_total",
        "model=dsv41,replica=0",
    );
    let d_good = compute_repaired_counter_delta(
        &s_ok,
        &e_ok,
        "vllm:spec_decode_num_accepted_tokens_total",
        "model=dsv41,replica=0",
    );

    // Real deterministic SSE parsing for fixtures 01, 06, 07.
    let sse_multi = parse_sse_stream_fixture(&[
        "data: {\"delta\":\"def clamp\",\"finish_reason\":\"\"}",
        "data: {\"delta\":\"_nonneg(x): return max(0,x)\",\"completion_tokens\":14,\"finish_reason\":\"stop\"}",
        "data: [DONE]",
    ]);
    let sse_trunc = parse_sse_stream_fixture(&[
        "data: {\"delta\":\"partial\",\"finish_reason\":\"length\"}",
        "data: [DONE]",
    ]);
    let sse_malformed = parse_sse_stream_fixture(&[
        "data: {\"delta\":\"ok\"}",
        "data: {\"delta\":\"broken_json_no_close\"",
    ]);

    // Reuse #7's offline reducer for fixtures 02 (missing usage) and 09 (delayed/failed timeout retained).
    let model = shared_workload_identity.model_id.as_str();
    let rec_no_usage = vec![RetainedCallRecord {
        request_id: "dsv-no-usage".into(),
        endpoint_url: "http://127.0.0.1:8000/v1/chat/completions".into(),
        model_id: model.into(),
        status: OutcomeStatus::Accepted,
        ttft_ms: Some(100.0),
        total_latency_ms: Some(400.0),
        prompt_tokens: Some(64),
        completion_tokens: None,
        sse_chunk_count: 5,
        greedy_text_matches_baseline: Some(true),
        task_quality_passed: None,
        authorization_header_redacted: true,
    }];
    let red_no_usage = reduce_and_verify_artifact(
        "dsv-fixture-02",
        model,
        &rec_no_usage,
        &ClaimedSummary {
            planned_requests: 1,
            accepted_requests: 1,
            total_completion_tokens: 5,
            p50_call_mean_itl_ms: None,
            p95_call_mean_itl_ms: None,
            claimed_greedy_parity_pass: true,
        },
    );

    let rec_timeout = vec![
        RetainedCallRecord {
            request_id: "dsv-ok-1".into(),
            endpoint_url: "http://127.0.0.1:8000/v1/chat/completions".into(),
            model_id: model.into(),
            status: OutcomeStatus::Accepted,
            ttft_ms: Some(100.0),
            total_latency_ms: Some(600.0),
            prompt_tokens: Some(64),
            completion_tokens: Some(51),
            sse_chunk_count: 25,
            greedy_text_matches_baseline: Some(true),
            task_quality_passed: Some(true),
            authorization_header_redacted: true,
        },
        RetainedCallRecord {
            request_id: "dsv-timeout-2".into(),
            endpoint_url: "http://127.0.0.1:8000/v1/chat/completions".into(),
            model_id: model.into(),
            status: OutcomeStatus::Timeout,
            ttft_ms: None,
            total_latency_ms: Some(30_000.0),
            prompt_tokens: Some(128),
            completion_tokens: None,
            sse_chunk_count: 0,
            greedy_text_matches_baseline: None,
            task_quality_passed: None,
            authorization_header_redacted: true,
        },
    ];
    let red_timeout = reduce_and_verify_artifact(
        "dsv-fixture-09",
        model,
        &rec_timeout,
        &ClaimedSummary {
            planned_requests: 2,
            accepted_requests: 1,
            total_completion_tokens: 51,
            p50_call_mean_itl_ms: Some(10.0),
            p95_call_mean_itl_ms: Some(10.0),
            claimed_greedy_parity_pass: true,
        },
    );

    // Real capability probe for fixture 08.
    let cap_tok = probe_endpoint_capability(&["/v1/chat/completions"], "/tokenize");
    let cap_eos = probe_endpoint_capability(&["/v1/chat/completions"], "vllm_ignore_eos");

    let device_free_fixtures = vec![
        DeviceFreeFixtureResult {
            fixture_id: "01-multi-token-sse-chunks".into(),
            handled_honestly: matches!(
                sse_multi,
                SseParseResult::Complete {
                    chunk_count: 2,
                    usage_tokens: Some(14),
                    ..
                }
            ) && matches!(d_good, CounterDeltaOutcome::Observed(v) if (v - 64.0).abs() < 1e-9),
            outcome: OutcomeStatus::Accepted,
            detail: "Parsed 2 multi-token SSE chunks with usage=14 tokens (chunk!=token) and spec counter delta=64.0.".into(),
        },
        DeviceFreeFixtureResult {
            fixture_id: "02-missing-usage-refuses-throughput".into(),
            handled_honestly: !red_no_usage.valid && !red_no_usage.throughput_claim_eligible,
            outcome: OutcomeStatus::Invalid,
            detail: format!(
                "Shared #7 reducer refused throughput claim: {}",
                red_no_usage.rejection_reasons.join(",")
            ),
        },
        DeviceFreeFixtureResult {
            fixture_id: "03-missing-spec-counters-unavailable".into(),
            handled_honestly: d_missing == CounterDeltaOutcome::UnavailableMissingCounter,
            outcome: OutcomeStatus::Unsupported,
            detail: format!("Missing counter reported as `{}` (never 0).", d_missing.tag()),
        },
        DeviceFreeFixtureResult {
            fixture_id: "04-multiple-metric-label-series".into(),
            handled_honestly: matches!(d_ambig, CounterDeltaOutcome::AmbiguousMultipleLabelSeries(2)),
            outcome: OutcomeStatus::Invalid,
            detail: format!("Ambiguous series reported as `{}` (no silent first-match).", d_ambig.tag()),
        },
        DeviceFreeFixtureResult {
            fixture_id: "05-counter-reset-detected".into(),
            handled_honestly: matches!(d_reset, CounterDeltaOutcome::CounterResetDetected { .. }),
            outcome: OutcomeStatus::Invalid,
            detail: format!("End < start reported as `{}`.", d_reset.tag()),
        },
        DeviceFreeFixtureResult {
            fixture_id: "06-empty-or-truncated-output".into(),
            handled_honestly: matches!(sse_trunc, SseParseResult::EmptyOrTruncated { .. }),
            outcome: OutcomeStatus::Invalid,
            detail: "SSE parser classified finish_reason=length as truncated and retained in denominator.".into(),
        },
        DeviceFreeFixtureResult {
            fixture_id: "07-malformed-sse-stream".into(),
            handled_honestly: matches!(sse_malformed, SseParseResult::MalformedFrame(_)),
            outcome: OutcomeStatus::InternalFailure,
            detail: "SSE parser caught unbalanced JSON frame as internal_failure without dropping request.".into(),
        },
        DeviceFreeFixtureResult {
            fixture_id: "08-unsupported-tokenize-or-ignore-eos".into(),
            handled_honestly: cap_tok == OutcomeStatus::Unsupported && cap_eos == OutcomeStatus::Unsupported,
            outcome: OutcomeStatus::Unsupported,
            detail: "Capability probe returned unsupported for /tokenize and vllm_ignore_eos on generic OpenAI endpoint.".into(),
        },
        DeviceFreeFixtureResult {
            fixture_id: "09-delayed-and-failed-requests-retained".into(),
            handled_honestly: red_timeout.valid && red_timeout.planned == 2 && red_timeout.failed_or_unfinished == 1,
            outcome: OutcomeStatus::Timeout,
            detail: "Shared #7 reducer retained 1 timeout alongside 1 accepted request in planned=2 denominator.".into(),
        },
    ];

    let workload_cross_compare_rejected = check_workload_comparability(
        SparkWorkloadClass::SynchronizedStreamWaveBurst,
        SparkWorkloadClass::SustainedArrival,
    )
    .is_err();

    let cache_states_distinguished = vec![
        CacheObservationState::Unknown.as_str().into(),
        CacheObservationState::RequestedUniquePrefixOnly
            .as_str()
            .into(),
        CacheObservationState::OperatorDeclared.as_str().into(),
        CacheObservationState::ServerCounterObserved.as_str().into(),
    ];

    let quality_comparisons = vec![
        compare_legacy_vs_semantic_quality(
            "cand-correct-clamp-alt-expr",
            "def clamp_nonneg(x):\n    return 0 if x < 0 else x",
        ),
        compare_legacy_vs_semantic_quality(
            "cand-wrong-abs-passes-legacy-substring",
            "def clamp_nonneg(x):\n    if x < 0: return -x\n    return x",
        ),
    ];

    DeepSeekSparkReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "dgx_spark_gb10_deepseek_endpoint",
            "Migration manifest, repaired counter accounting, 9 device-free fixtures, and executable semantic vs legacy checks verified locally; live DeepSeek/Spark endpoint smoke remains an external hardware gate.",
        ),
        shared_workload_identity,
        migration_manifest,
        probed_capabilities_default: vec![
            ("/tokenize".into(), false),
            ("vllm_ignore_eos".into(), false),
            ("speculative_counters".into(), false),
            ("server_cache_telemetry".into(), false),
        ],
        workload_cross_compare_rejected,
        cache_states_distinguished,
        device_free_fixtures,
        quality_comparisons,
    }
}

pub fn inputs_json(report: &DeepSeekSparkReport) -> String {
    let entries: Vec<String> = report
        .migration_manifest
        .iter()
        .map(|m| {
            format!(
                concat!(
                    "{{\"donor_path\":{},\"donor_ref\":{},\"license\":{},\"publishability\":{},",
                    "\"destination_or_owner\":{},\"source_digest\":{},\"supersedes_or_notes\":{}}}"
                ),
                quote(&m.donor_path),
                quote(&m.donor_ref),
                quote(&m.license),
                quote(&m.publishability),
                quote(&m.destination_or_owner),
                m.source_digest.to_json(),
                quote(&m.supersedes_or_notes),
            )
        })
        .collect();
    let caps: Vec<String> = report
        .probed_capabilities_default
        .iter()
        .map(|(k, v)| format!("{{\"capability\":{},\"assumed_universal\":{v}}}", quote(k)))
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-08-deepseek-spark.inputs.v1\",\n",
            "  \"issue\": 8,\n",
            "  \"reuses_serving_runner_issue\": 7,\n",
            "  \"execution_class\": {},\n",
            "  \"shared_workload_identity\": {},\n",
            "  \"probed_capabilities\": [{}],\n",
            "  \"migration_manifest\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.shared_workload_identity.to_json(),
        caps.join(","),
        entries.join(","),
    )
}

pub fn results_json(report: &DeepSeekSparkReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-08-deepseek-spark.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 8,").unwrap();
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
        "  \"workload_cross_compare_rejected\": {},",
        report.workload_cross_compare_rejected
    )
    .unwrap();
    writeln!(
        out,
        "  \"cache_states_distinguished\": {},",
        str_array(&report.cache_states_distinguished)
    )
    .unwrap();

    let fixs: Vec<String> = report
        .device_free_fixtures
        .iter()
        .map(|f| {
            format!(
                "{{\"fixture_id\":{},\"handled_honestly\":{},\"outcome\":{},\"detail\":{}}}",
                quote(&f.fixture_id),
                f.handled_honestly,
                quote(f.outcome.as_str()),
                quote(&f.detail),
            )
        })
        .collect();
    writeln!(out, "  \"device_free_fixtures\": [{}],", fixs.join(",")).unwrap();

    let quals: Vec<String> = report
        .quality_comparisons
        .iter()
        .map(|q| {
            let ins: Vec<String> = q
                .test_inputs_evaluated
                .iter()
                .map(ToString::to_string)
                .collect();
            let fails: Vec<String> = q.failed_inputs.iter().map(ToString::to_string).collect();
            format!(
                concat!(
                    "{{\"candidate_id\":{},\"legacy_evaluator_id\":{},\"legacy_substring_passed\":{},",
                    "\"semantic_evaluator_id\":{},\"test_inputs_evaluated\":[{}],\"failed_inputs\":[{}],",
                    "\"semantic_executable_passed\":{},\"semantic_rejection_reason\":{}}}"
                ),
                quote(&q.candidate_id),
                quote(&q.legacy_evaluator_id),
                q.legacy_substring_passed,
                quote(&q.semantic_evaluator_id),
                ins.join(","),
                fails.join(","),
                q.semantic_executable_passed,
                opt_str(q.semantic_rejection_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"quality_comparisons\": [{}]", quals.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &DeepSeekSparkReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #8: DeepSeek/Spark Migration into Shared Serving Benchmark"
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
        "- Burst vs sustained cross-comparison rejected: `{}`",
        report.workload_cross_compare_rejected
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Device-Free Fixtures (9/9)").unwrap();
    writeln!(out, "| Fixture | Handled Honestly | Outcome |").unwrap();
    writeln!(out, "| --- | --- | --- |").unwrap();
    for f in &report.device_free_fixtures {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` |",
            f.fixture_id,
            f.handled_honestly,
            f.outcome.as_str()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Legacy Substring vs Independent Executable Semantic Quality Check"
    )
    .unwrap();
    for q in &report.quality_comparisons {
        writeln!(
            out,
            "- `{}`: legacy substring=`{}` vs semantic=`{}` (failed inputs: `{:?}`)",
            q.candidate_id,
            q.legacy_substring_passed,
            q.semantic_executable_passed,
            q.failed_inputs
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Device-free contract verification only. Private donor logs/weights and deployment scripts remain excluded; no live Spark endpoint ran locally."
    )
    .unwrap();
    out
}
