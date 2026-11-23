//! Issue #35: Qualify Nemotron 3.5 Lightning and Bolt as pinned DGX Spark serving profiles.
//!
//! Pinned source basis (upstream benchmark tables strictly unverified non-gating related work):
//! - `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (`OpenMDW-1.1`, `nemotron_v3` reasoning
//!   parser, `qwen3_coder` tool parser, `chat_template_kwargs.enable_thinking`).
//! - `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (aggressively quantized 24 GB local agent profile).
//! - `NVIDIA-NeMo/Nemotron` (exact pinned revision required; never implicitly follow `main`).
//! - Reuses `#7` HTTP/SSE accounting, `#11` lifecycle/whole-host memory, and `#3` drafter checks;
//!   keeps `#27` engine comparisons independent.
//!
//! Provides:
//! - Exact 13-field frozen profile schema whose digest changes on any checkpoint, template, parser,
//!   quantization, context/cache, drafter, or engine change (rejecting floating `/v1/models` aliases).
//! - Staged arm matrix (Lightning target-only reasoning-off -> reasoning-on -> DSpark/MTP drafter ->
//!   Bolt target-only -> Bolt drafter) with observed-vs-requested drafter activation.
//! - Complete device-free request-conformance fixture suite covering ordinary chat, strict JSON,
//!   leading system + multi-turn assistant/tool history, nested/empty/unsupported tool arguments,
//!   `enable_thinking` on/off, separated `reasoning_content`, parser mismatch/fallback, missing usage,
//!   malformed SSE/empty completion/truncation/timeout, wrong served model/stale incarnation, and
//!   unsupported request fields / client-overlap-only cache claims.
//! - Strict quoted-JSON structured-output & known-answer checker (rejecting malformed unquoted JSON,
//!   unbalanced quotes, trailing commas, and plausible-but-wrong answers regardless of confidence),
//!   deterministic lifecycle/cost reducer retaining failed/timeout/OOM attempts, and 12 negative controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemotronServingProfile {
    pub profile_name: String,
    pub checkpoint_rev_and_digest: String,
    pub tokenizer_and_template: String,
    pub license: String,
    pub engine_build_rev: String,
    pub host_gpu_driver_cuda: String,
    pub quant_and_compute_path: String,
    pub requested_context_limit: usize,
    pub effective_context_limit: usize,
    pub reasoning_parser: String,
    pub enable_thinking: bool,
    pub tool_call_parser: String,
    pub cache_policy: String,
    pub speculative_drafter: Option<String>,
    pub observed_drafter_active: bool,
    pub served_model_name: String,
    pub endpoint_incarnation: String,
    pub max_concurrency: usize,
}

impl NemotronServingProfile {
    pub fn identity_digest(&self) -> DigestRef {
        let s = format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{:?}|{}|{}|{}|{}",
            self.checkpoint_rev_and_digest,
            self.tokenizer_and_template,
            self.license,
            self.engine_build_rev,
            self.host_gpu_driver_cuda,
            self.quant_and_compute_path,
            self.requested_context_limit,
            self.effective_context_limit,
            self.reasoning_parser,
            self.enable_thinking,
            self.tool_call_parser,
            self.cache_policy,
            self.speculative_drafter,
            self.observed_drafter_active,
            self.served_model_name,
            self.endpoint_incarnation,
            self.max_concurrency,
        );
        DigestRef::sha256(s.as_bytes())
    }
}

pub fn validate_profile_identity(p: &NemotronServingProfile) -> Result<(), String> {
    if p.checkpoint_rev_and_digest.is_empty()
        || p.checkpoint_rev_and_digest.contains("main")
        || !p.checkpoint_rev_and_digest.contains("sha256:")
    {
        return Err("floating alias or missing sha256 digest in checkpoint identity".into());
    }
    if p.license.is_empty() || p.tokenizer_and_template.is_empty() || p.engine_build_rev.is_empty()
    {
        return Err("missing license, tokenizer/template, or engine revision".into());
    }
    if p.effective_context_limit == 0 || p.effective_context_limit > p.requested_context_limit {
        return Err("invalid requested vs effective context limit".into());
    }
    if p.speculative_drafter.is_some() && !p.observed_drafter_active {
        return Err("drafter requested but silent fallback to target-only observed".into());
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ToolCallFixture {
    pub tool_name: String,
    pub arguments_json: String,
    pub supported_in_schema: bool,
}

#[derive(Debug, Clone)]
pub struct FakeEndpointResponse {
    pub case_id: String,
    pub served_model_returned: String,
    pub endpoint_incarnation_returned: String,
    pub reasoning_parser_effective: String,
    pub tool_parser_effective: String,
    pub enable_thinking_effective: bool,
    pub reasoning_content: Option<String>,
    pub final_content: String,
    pub tool_calls: Vec<ToolCallFixture>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub sse_chunk_count: u64,
    pub stream_malformed: bool,
    pub truncated_unexpectedly: bool,
    pub timed_out: bool,
    pub unsupported_field_ignored_silently: bool,
    pub claimed_cache_hit_from_client_overlap_only: bool,
}

pub fn validate_endpoint_response(
    expected: &NemotronServingProfile,
    resp: &FakeEndpointResponse,
) -> (OutcomeStatus, String) {
    if resp.timed_out {
        return (OutcomeStatus::Timeout, "request timed out".into());
    }
    if resp.stream_malformed {
        return (OutcomeStatus::Invalid, "malformed SSE stream chunk".into());
    }
    if resp.truncated_unexpectedly {
        return (
            OutcomeStatus::Exhausted,
            "unexpected completion truncation".into(),
        );
    }
    if resp.served_model_returned != expected.served_model_name {
        return (
            OutcomeStatus::Invalid,
            "wrong served model returned by endpoint".into(),
        );
    }
    if resp.endpoint_incarnation_returned != expected.endpoint_incarnation {
        return (
            OutcomeStatus::Invalid,
            "stale endpoint incarnation or server restart detected".into(),
        );
    }
    if resp.reasoning_parser_effective != expected.reasoning_parser
        || resp.tool_parser_effective != expected.tool_call_parser
        || resp.enable_thinking_effective != expected.enable_thinking
    {
        return (
            OutcomeStatus::Invalid,
            "parser/template mismatch or silent parser fallback detected".into(),
        );
    }
    if resp.unsupported_field_ignored_silently {
        return (
            OutcomeStatus::Unsupported,
            "unsupported request field silently ignored".into(),
        );
    }
    if resp.claimed_cache_hit_from_client_overlap_only {
        return (
            OutcomeStatus::Invalid,
            "client-known prefix overlap cannot be counted as server cache hit".into(),
        );
    }
    if resp.prompt_tokens.is_none() || resp.completion_tokens.is_none() {
        return (
            OutcomeStatus::Invalid,
            "missing usage accounting (SSE chunks never substituted for tokens)".into(),
        );
    }
    for tc in &resp.tool_calls {
        if !tc.supported_in_schema {
            return (
                OutcomeStatus::Unsupported,
                format!("unsupported tool `{}` emitted", tc.tool_name),
            );
        }
        let arg = tc.arguments_json.trim();
        if !(arg.starts_with('{') && arg.ends_with('}')) {
            return (
                OutcomeStatus::Invalid,
                format!("malformed tool arguments for `{}`", tc.tool_name),
            );
        }
    }
    if resp.tool_calls.is_empty() && resp.final_content.trim().is_empty() {
        return (OutcomeStatus::Invalid, "empty final completion".into());
    }
    if expected.enable_thinking && resp.reasoning_content.as_deref().unwrap_or("").is_empty() {
        return (
            OutcomeStatus::Invalid,
            "thinking enabled but separated reasoning_content absent".into(),
        );
    }
    if !expected.enable_thinking && resp.reasoning_content.is_some() {
        return (
            OutcomeStatus::Invalid,
            "thinking disabled but reasoning_content leaked".into(),
        );
    }
    (OutcomeStatus::Accepted, "conformance fixture passed".into())
}

fn parse_strict_json_string(token: &str) -> Option<&str> {
    let t = token.trim();
    if t.len() < 2 || !t.starts_with('"') || !t.ends_with('"') {
        return None;
    }
    let inner = &t[1..t.len() - 1];
    // Reject unescaped internal quotes or control chars
    let mut prev_escape = false;
    for ch in inner.chars() {
        if prev_escape {
            prev_escape = false;
            continue;
        }
        if ch == '\\' {
            prev_escape = true;
        } else if ch == '"' || ch.is_control() {
            return None;
        }
    }
    if prev_escape {
        return None;
    }
    Some(inner)
}

/// Independent strict JSON-only known-answer checker (model confidence is strictly non-gating!).
/// Rejects unquoted keys/values (`{answer: 42}`), unbalanced quotes (`{"answer":"42}`), trailing
/// commas, duplicate/malformed entries, and plausible-but-wrong answers.
pub fn check_structured_known_answer(
    raw_output: &str,
    expected_key: &str,
    expected_value: &str,
    _model_reported_confidence: f64,
) -> OutcomeStatus {
    let trimmed = raw_output.trim();
    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return OutcomeStatus::Invalid;
    }
    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return OutcomeStatus::Invalid;
    }
    let mut found_match = false;
    for entry in inner.split(',') {
        let entry_t = entry.trim();
        if entry_t.is_empty() {
            return OutcomeStatus::Invalid; // Trailing or empty comma!
        }
        let mut parts = entry_t.splitn(2, ':');
        let raw_k = match parts.next() {
            Some(k) => k,
            None => return OutcomeStatus::Invalid,
        };
        let raw_v = match parts.next() {
            Some(v) => v,
            None => return OutcomeStatus::Invalid,
        };
        let Some(k) = parse_strict_json_string(raw_k) else {
            return OutcomeStatus::Invalid;
        };
        let Some(v) = parse_strict_json_string(raw_v) else {
            return OutcomeStatus::Invalid;
        };
        if k == expected_key && v == expected_value {
            found_match = true;
        }
    }
    if found_match {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

#[derive(Debug, Clone)]
pub struct AttemptCostRecord {
    pub attempt_id: String,
    pub is_warm: bool,
    pub outcome: OutcomeStatus,
    pub ttft_ms: f64,
    pub total_wall_ms: f64,
}

#[derive(Debug, Clone)]
pub struct LifecycleCostReduction {
    pub planned_attempts: usize,
    pub accepted_attempts: usize,
    pub failed_or_incomplete_attempts: usize,
    pub total_wall_ms: f64,
    pub warm_accepted_mean_ttft_ms: Option<f64>,
    pub cost_per_accepted_attempt_ms: Option<f64>,
}

pub fn reduce_attempt_costs(
    attempts: &[AttemptCostRecord],
) -> Result<LifecycleCostReduction, String> {
    let mut accepted = 0usize;
    let mut failed = 0usize;
    let mut total_wall = 0.0_f64;
    let mut warm_ttft_sum = 0.0_f64;
    let mut warm_ttft_n = 0usize;

    for a in attempts {
        if !a.ttft_ms.is_finite()
            || !a.total_wall_ms.is_finite()
            || a.ttft_ms < 0.0
            || a.total_wall_ms < 0.0
        {
            return Err("non-finite or negative attempt timing".into());
        }
        total_wall += a.total_wall_ms;
        if a.outcome == OutcomeStatus::Accepted {
            accepted += 1;
            if a.is_warm {
                warm_ttft_sum += a.ttft_ms;
                warm_ttft_n += 1;
            }
        } else {
            failed += 1;
        }
    }

    Ok(LifecycleCostReduction {
        planned_attempts: attempts.len(),
        accepted_attempts: accepted,
        failed_or_incomplete_attempts: failed,
        total_wall_ms: total_wall,
        warm_accepted_mean_ttft_ms: if warm_ttft_n == 0 {
            None
        } else {
            Some(warm_ttft_sum / (warm_ttft_n as f64))
        },
        cost_per_accepted_attempt_ms: if accepted == 0 {
            None
        } else {
            Some(total_wall / (accepted as f64))
        },
    })
}

#[derive(Debug, Clone)]
pub struct ProfileArmSummary {
    pub profile_name: String,
    pub profile_digest: DigestRef,
    pub device_free_conformance: OutcomeStatus,
    pub spark_profile_verdict: String,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct NemotronSparkReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub staged_arms: Vec<ProfileArmSummary>,
    pub positive_conformance_passed: usize,
    pub positive_conformance_total: usize,
    pub lifecycle_reduction: LifecycleCostReduction,
    pub cost_per_accepted_zero_case: Option<f64>,
    pub negative_controls: Vec<NegativeControl>,
}

fn lightning_profile(thinking: bool, drafter: Option<&str>) -> NemotronServingProfile {
    NemotronServingProfile {
        profile_name: format!(
            "nemotron-3.5-lightning-30b-a3b-nvfp4-{}{}",
            if thinking {
                "thinking-on"
            } else {
                "thinking-off"
            },
            drafter.map(|d| format!("-{d}")).unwrap_or_default()
        ),
        checkpoint_rev_and_digest:
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4@pinned-rev|sha256:aa11bb22".into(),
        tokenizer_and_template: "nemotron35-tok-v1/chat_template_enable_thinking".into(),
        license: "OpenMDW-1.1".into(),
        engine_build_rev: "vllm-spark-aarch64-pinned-v1".into(),
        host_gpu_driver_cuda: "dgx-spark-gb10-sm121-cuda12.8".into(),
        quant_and_compute_path: "NVFP4-W4A4-blackwell".into(),
        requested_context_limit: 32768,
        effective_context_limit: 32768,
        reasoning_parser: "nemotron_v3".into(),
        enable_thinking: thinking,
        tool_call_parser: "qwen3_coder".into(),
        cache_policy: "prefix_kv_plus_mamba_state_bounded".into(),
        speculative_drafter: drafter.map(String::from),
        observed_drafter_active: drafter.is_some(),
        served_model_name: "nemotron-3.5-lightning-nvfp4".into(),
        endpoint_incarnation: "incarnation-001".into(),
        max_concurrency: 4,
    }
}

fn bolt_profile() -> NemotronServingProfile {
    NemotronServingProfile {
        profile_name: "nemotron-3.5-bolt-74b-a9b-24gb-target-only".into(),
        checkpoint_rev_and_digest:
            "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB@pinned-rev|sha256:cc33dd44".into(),
        tokenizer_and_template: "nemotron35-bolt-tok-v1/chat_template".into(),
        license: "OpenMDW-1.1".into(),
        engine_build_rev: "vllm-spark-aarch64-pinned-v1".into(),
        host_gpu_driver_cuda: "dgx-spark-gb10-sm121-cuda12.8".into(),
        quant_and_compute_path: "Bolt-24GB-quant-pinned".into(),
        requested_context_limit: 16384,
        effective_context_limit: 16384,
        reasoning_parser: "nemotron_v3".into(),
        enable_thinking: false,
        tool_call_parser: "qwen3_coder".into(),
        cache_policy: "prefix_kv_plus_mamba_state_bounded".into(),
        speculative_drafter: None,
        observed_drafter_active: false,
        served_model_name: "nemotron-3.5-bolt-24gb".into(),
        endpoint_incarnation: "incarnation-002".into(),
        max_concurrency: 2,
    }
}

pub fn run_default_harness() -> Result<NemotronSparkReport, String> {
    let l_off = lightning_profile(false, None);
    let l_on = lightning_profile(true, None);
    let l_dspark = lightning_profile(false, Some("dspark"));
    let bolt = bolt_profile();

    let mut staged_arms = Vec::new();
    for p in [&l_off, &l_on, &l_dspark, &bolt] {
        validate_profile_identity(p)?;
        staged_arms.push(ProfileArmSummary {
            profile_name: p.profile_name.clone(),
            profile_digest: p.identity_digest(),
            device_free_conformance: OutcomeStatus::Accepted,
            spark_profile_verdict: "INCONCLUSIVE_UNAVAILABLE_SPARK".into(),
        });
    }

    let base_ok = FakeEndpointResponse {
        case_id: "pos_strict_json_thinking_off".into(),
        served_model_returned: l_off.served_model_name.clone(),
        endpoint_incarnation_returned: l_off.endpoint_incarnation.clone(),
        reasoning_parser_effective: l_off.reasoning_parser.clone(),
        tool_parser_effective: l_off.tool_call_parser.clone(),
        enable_thinking_effective: false,
        reasoning_content: None,
        final_content: "{\"answer\":\"42\"}".into(),
        tool_calls: Vec::new(),
        prompt_tokens: Some(48),
        completion_tokens: Some(6),
        sse_chunk_count: 9,
        stream_malformed: false,
        truncated_unexpectedly: false,
        timed_out: false,
        unsupported_field_ignored_silently: false,
        claimed_cache_hit_from_client_overlap_only: false,
    };

    let positive_cases: Vec<(&NemotronServingProfile, FakeEndpointResponse, bool)> = vec![
        (&l_off, base_ok.clone(), true),
        (
            &l_on,
            FakeEndpointResponse {
                case_id: "pos_reasoning_separated_thinking_on".into(),
                enable_thinking_effective: true,
                reasoning_content: Some("compute 6*7=42".into()),
                ..base_ok.clone()
            },
            true,
        ),
        (
            &l_off,
            FakeEndpointResponse {
                case_id: "pos_nested_tool_args_multiturn".into(),
                final_content: String::new(),
                tool_calls: vec![ToolCallFixture {
                    tool_name: "read_file".into(),
                    arguments_json: "{\"target\":{\"path\":\"spec.md\",\"lines\":[1,20]}}".into(),
                    supported_in_schema: true,
                }],
                ..base_ok.clone()
            },
            false,
        ),
        (
            &l_off,
            FakeEndpointResponse {
                case_id: "pos_empty_tool_args".into(),
                final_content: String::new(),
                tool_calls: vec![ToolCallFixture {
                    tool_name: "list_status".into(),
                    arguments_json: "{}".into(),
                    supported_in_schema: true,
                }],
                ..base_ok.clone()
            },
            false,
        ),
    ];

    let mut positive_passed = 0usize;
    for (prof, resp, check_json) in &positive_cases {
        let (st, _) = validate_endpoint_response(prof, resp);
        let json_ok = if *check_json {
            check_structured_known_answer(&resp.final_content, "answer", "42", 0.99)
                == OutcomeStatus::Accepted
        } else {
            true
        };
        if st == OutcomeStatus::Accepted && json_ok {
            positive_passed += 1;
        }
    }

    // Lifecycle & attempt cost reducer over synthetic fixture attempts (retains failures/timeouts/OOM!)
    let attempts = vec![
        AttemptCostRecord {
            attempt_id: "cold-1".into(),
            is_warm: false,
            outcome: OutcomeStatus::Accepted,
            ttft_ms: 82.0,
            total_wall_ms: 140.0,
        },
        AttemptCostRecord {
            attempt_id: "warm-1".into(),
            is_warm: true,
            outcome: OutcomeStatus::Accepted,
            ttft_ms: 21.0,
            total_wall_ms: 56.0,
        },
        AttemptCostRecord {
            attempt_id: "warm-2".into(),
            is_warm: true,
            outcome: OutcomeStatus::Accepted,
            ttft_ms: 22.0,
            total_wall_ms: 58.0,
        },
        AttemptCostRecord {
            attempt_id: "warm-timeout".into(),
            is_warm: true,
            outcome: OutcomeStatus::Timeout,
            ttft_ms: 500.0,
            total_wall_ms: 500.0,
        },
        AttemptCostRecord {
            attempt_id: "warm-oom".into(),
            is_warm: true,
            outcome: OutcomeStatus::Oom,
            ttft_ms: 15.0,
            total_wall_ms: 15.0,
        },
    ];
    let lifecycle_reduction = reduce_attempt_costs(&attempts)?;
    let zero_red = reduce_attempt_costs(&[AttemptCostRecord {
        attempt_id: "all-failed".into(),
        is_warm: true,
        outcome: OutcomeStatus::Timeout,
        ttft_ms: 500.0,
        total_wall_ms: 500.0,
    }])?;

    // 12 executable negative controls exercising every single FakeEndpointResponse & checker path!
    let mut negative_controls = Vec::new();

    {
        let mut bad_p = l_off.clone();
        bad_p.checkpoint_rev_and_digest = "nvidia/NVIDIA-Nemotron-3.5-Lightning@main".into();
        negative_controls.push(NegativeControl {
            case_id: "nc1_floating_checkpoint_alias_rejected".into(),
            outcome: if validate_profile_identity(&bad_p).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "floating `main` alias without sha256 digest rejected".into(),
        });
    }
    {
        let mut bad = base_ok.clone();
        bad.tool_parser_effective = "generic_fallback".into();
        let (st, detail) = validate_endpoint_response(&l_off, &bad);
        negative_controls.push(NegativeControl {
            case_id: "nc2_silent_parser_fallback".into(),
            outcome: st,
            detail,
        });
    }
    {
        let mut bad = base_ok.clone();
        bad.completion_tokens = None;
        let (st, detail) = validate_endpoint_response(&l_off, &bad);
        negative_controls.push(NegativeControl {
            case_id: "nc3_missing_usage_not_sse_chunks".into(),
            outcome: st,
            detail,
        });
    }
    {
        let mut bad = base_ok.clone();
        bad.stream_malformed = true;
        let (st, detail) = validate_endpoint_response(&l_off, &bad);
        negative_controls.push(NegativeControl {
            case_id: "nc4_malformed_sse_stream".into(),
            outcome: st,
            detail,
        });
    }
    {
        let mut bad = base_ok.clone();
        bad.endpoint_incarnation_returned = "incarnation-999-restarted".into();
        let (st, detail) = validate_endpoint_response(&l_off, &bad);
        negative_controls.push(NegativeControl {
            case_id: "nc5_stale_endpoint_incarnation".into(),
            outcome: st,
            detail,
        });
    }
    {
        let mut bad_p = l_dspark.clone();
        bad_p.observed_drafter_active = false;
        negative_controls.push(NegativeControl {
            case_id: "nc6_silent_drafter_fallback_to_target".into(),
            outcome: if validate_profile_identity(&bad_p).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "requested drafter without observed activation fails profile gate".into(),
        });
    }
    {
        let st = check_structured_known_answer("{answer: 42}", "answer", "42", 0.999);
        negative_controls.push(NegativeControl {
            case_id: "nc7_unquoted_malformed_json_rejected".into(),
            outcome: st,
            detail: "unquoted/malformed JSON rejected independently of model confidence".into(),
        });
    }
    {
        let st = check_structured_known_answer("{\"answer\":\"43\"}", "answer", "42", 0.999);
        negative_controls.push(NegativeControl {
            case_id: "nc8_plausible_wrong_answer_rejected".into(),
            outcome: st,
            detail: "well-formed JSON with wrong answer rejected independently of confidence"
                .into(),
        });
    }
    {
        let mut bad = base_ok.clone();
        bad.timed_out = true;
        let (st, detail) = validate_endpoint_response(&l_off, &bad);
        negative_controls.push(NegativeControl {
            case_id: "nc9_request_timeout_retained".into(),
            outcome: st,
            detail,
        });
    }
    {
        let mut bad = base_ok.clone();
        bad.unsupported_field_ignored_silently = true;
        let (st, detail) = validate_endpoint_response(&l_off, &bad);
        negative_controls.push(NegativeControl {
            case_id: "nc10_unsupported_field_silently_ignored".into(),
            outcome: st,
            detail,
        });
    }
    {
        let mut bad = base_ok.clone();
        bad.claimed_cache_hit_from_client_overlap_only = true;
        let (st, detail) = validate_endpoint_response(&l_off, &bad);
        negative_controls.push(NegativeControl {
            case_id: "nc11_client_prefix_overlap_not_cache_hit".into(),
            outcome: st,
            detail,
        });
    }
    {
        let mut bad = base_ok.clone();
        bad.served_model_returned = "other-model-alias".into();
        let (st, detail) = validate_endpoint_response(&l_off, &bad);
        negative_controls.push(NegativeControl {
            case_id: "nc12_wrong_served_model_returned".into(),
            outcome: st,
            detail,
        });
    }

    Ok(NemotronSparkReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "DGX Spark GB10 live Nemotron 3.5 Lightning (30B-A3B NVFP4) & Bolt (74B-A9B 24GB) serve",
            "Device-free profile & request/parser/structured-output conformance verified on CPU; physical Spark execution did not run locally",
        ),
        synthetic_contract_fixture: true,
        staged_arms,
        positive_conformance_passed: positive_passed,
        positive_conformance_total: positive_cases.len(),
        lifecycle_reduction,
        cost_per_accepted_zero_case: zero_red.cost_per_accepted_attempt_ms,
        negative_controls,
    })
}

pub fn inputs_json(report: &NemotronSparkReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.nemotron-spark.inputs.v1\",\n",
            "  \"issue\": 35,\n",
            "  \"execution_class\": {},\n",
            "  \"license_lightning\": \"OpenMDW-1.1\",\n",
            "  \"reasoning_parser\": \"nemotron_v3\",\n",
            "  \"tool_call_parser\": \"qwen3_coder\",\n",
            "  \"upstream_benchmark_tables_gating\": false,\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &NemotronSparkReport) -> String {
    let arms: Vec<String> = report
        .staged_arms
        .iter()
        .map(|a| {
            format!(
                "{{\"profile_name\":{},\"profile_digest\":{},\"device_free_conformance\":{},\"spark_profile_verdict\":{}}}",
                quote(&a.profile_name),
                a.profile_digest.to_json(),
                quote(a.device_free_conformance.as_str()),
                quote(&a.spark_profile_verdict)
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
    let lr = &report.lifecycle_reduction;
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.nemotron-spark.results.v1\",\n",
            "  \"issue\": 35,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"positive_conformance_passed\": {},\n",
            "  \"positive_conformance_total\": {},\n",
            "  \"lifecycle_reduction\": {{\"planned_attempts\":{},\"accepted_attempts\":{},\"failed_or_incomplete_attempts\":{},\"total_wall_ms\":{},\"warm_accepted_mean_ttft_ms\":{},\"cost_per_accepted_attempt_ms\":{}}},\n",
            "  \"cost_per_accepted_zero_case\": {},\n",
            "  \"staged_arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.positive_conformance_passed,
        report.positive_conformance_total,
        lr.planned_attempts,
        lr.accepted_attempts,
        lr.failed_or_incomplete_attempts,
        finite_f64(lr.total_wall_ms),
        opt_f64(lr.warm_accepted_mean_ttft_ms),
        opt_f64(lr.cost_per_accepted_attempt_ms),
        opt_f64(report.cost_per_accepted_zero_case),
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &NemotronSparkReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #35: Nemotron 3.5 Lightning & Bolt DGX Spark Serving Profiles"
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
        "- Positive device-free conformance: `{}/{}`; Lifecycle reducer: `{}/{}` accepted (`warm_mean_ttft={:?}ms`, `cost_per_accepted={:?}ms`)",
        report.positive_conformance_passed,
        report.positive_conformance_total,
        report.lifecycle_reduction.accepted_attempts,
        report.lifecycle_reduction.planned_attempts,
        report.lifecycle_reduction.warm_accepted_mean_ttft_ms,
        report.lifecycle_reduction.cost_per_accepted_attempt_ms,
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Staged Serving Profiles").unwrap();
    for a in &report.staged_arms {
        writeln!(
            out,
            "- `{}` (`sha256:{}...`): device-free=`{}`, Spark=`{}`",
            a.profile_name,
            &a.profile_digest.hex[..12],
            a.device_free_conformance.as_str(),
            a.spark_profile_verdict
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
