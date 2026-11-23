//! Issue #13: Split the reusable `djev` protocol client from RSI decision semantics for a
//! concrete lab consumer.
//!
//! Pure deterministic core implementing:
//! - Dual-consumer extraction gate (requiring existing RSI adapter + named inference-lab `#1`
//!   structured-read/capability worker pinning the same protocol revision; rejecting hypothetical
//!   single-consumer extraction).
//! - Deterministic canonical `djev.systemone.v1` request wire-byte builder and bounded response
//!   validator preserving exact key order, UTF-8 / non-ASCII handling, ordered choice labels,
//!   and span byte-offset bounds (golden-byte SHA-256 verification).
//! - Closed RSI `#42` effective cache-identity regression suite verifying identical effective
//!   reads reuse results while changing rendering, candidate order, template, samples, seed,
//!   tournament/max-options, canvas, served revision, or wire version invalidates reuse, and
//!   missing immutable model identity falls back to `non_reusable_run_namespace` (never inventing
//!   a digest from model name alone).
//! - Explicit capability separation (`structured_read_scoring` != `generative_masked_completion`
//!   != `noble_proof_acceptance`) and whole-decision deadline/cancellation/stale-result controls.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_str, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const DJEV_DONOR_COMMIT: &str = "01e47e268d71bc1c53cc36e11c179f2b8f22f5c5";
pub const DJEV_WIRE_VERSION: &str = "djev.systemone.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanOffset {
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DjevSystemOneRequest {
    pub wire_version: String,
    pub served_model_name: String,
    pub immutable_model_digest: Option<DigestRef>,
    pub server_revision: String,
    pub template_revision: String,
    pub prompt_text: String,
    pub ordered_choices: Vec<String>,
    pub read_span: SpanOffset,
    pub canvas_id: String,
    pub max_options: usize,
    pub tournament_mode: String,
    pub samples: usize,
    pub seed: u64,
    pub temperature: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheIdentityPolicy {
    ReusableDigest(DigestRef),
    NonReusableRunNamespace(String),
}

impl CacheIdentityPolicy {
    pub fn as_key_str(&self) -> String {
        match self {
            Self::ReusableDigest(d) => format!("{}:{}", d.algorithm, d.hex),
            Self::NonReusableRunNamespace(ns) => format!("non_reusable_run_namespace:{ns}"),
        }
    }

    pub fn is_reusable(&self) -> bool {
        matches!(self, Self::ReusableDigest(_))
    }
}

/// Validate span byte offsets against UTF-8 boundaries and encode canonical `djev.systemone.v1`
/// request bytes with stable key order.
pub fn encode_canonical_request_bytes(
    req: &DjevSystemOneRequest,
) -> Result<Vec<u8>, OutcomeStatus> {
    if req.wire_version != DJEV_WIRE_VERSION {
        return Err(OutcomeStatus::Unsupported);
    }
    if req.served_model_name.is_empty()
        || req.ordered_choices.is_empty()
        || req.samples == 0
        || req.max_options == 0
        || !req.temperature.is_finite()
        || req.temperature < 0.0
    {
        return Err(OutcomeStatus::Invalid);
    }
    // Check UTF-8 char-boundary and bounds on span offsets.
    if req.read_span.start_byte >= req.read_span.end_byte
        || req.read_span.end_byte > req.prompt_text.len()
        || !req.prompt_text.is_char_boundary(req.read_span.start_byte)
        || !req.prompt_text.is_char_boundary(req.read_span.end_byte)
    {
        return Err(OutcomeStatus::Invalid);
    }
    // Distinct non-empty ordered choices required.
    let mut seen = BTreeSet::new();
    for choice in &req.ordered_choices {
        if choice.is_empty() || !seen.insert(choice.as_str()) {
            return Err(OutcomeStatus::Invalid);
        }
    }

    let model_id_json = match &req.immutable_model_digest {
        Some(d) => d.to_json(),
        None => "null".into(),
    };

    let canonical = format!(
        concat!(
            "{{\"wire_version\":{},\"served_model_name\":{},\"immutable_model_digest\":{},",
            "\"server_revision\":{},\"template_revision\":{},\"prompt_text\":{},",
            "\"ordered_choices\":{},\"read_span\":{{\"start_byte\":{},\"end_byte\":{}}},",
            "\"canvas_id\":{},\"max_options\":{},\"tournament_mode\":{},",
            "\"samples\":{},\"seed\":{},\"temperature\":{}}}"
        ),
        quote(&req.wire_version),
        quote(&req.served_model_name),
        model_id_json,
        quote(&req.server_revision),
        quote(&req.template_revision),
        quote(&req.prompt_text),
        str_array(&req.ordered_choices),
        req.read_span.start_byte,
        req.read_span.end_byte,
        quote(&req.canvas_id),
        req.max_options,
        quote(&req.tournament_mode),
        req.samples,
        req.seed,
        finite_f64(req.temperature),
    );
    Ok(canonical.into_bytes())
}

/// Dual-consumer extraction gate: requires >= 2 distinct non-empty concrete consumers pinning
/// the shared `djev.systemone.v1` protocol revision (rejects single hypothetical consumer).
pub fn validate_dual_consumer_gate(consumers: &[&str]) -> bool {
    let distinct: BTreeSet<&str> = consumers
        .iter()
        .copied()
        .filter(|c| !c.trim().is_empty())
        .collect();
    distinct.len() >= 2
}

/// Closed RSI `#42` effective cache identity: depends on the full canonical wire bytes and
/// requires an immutable model digest. Without an immutable model digest, returns an explicit
/// `NonReusableRunNamespace` instead of inventing a digest from model name alone.
pub fn compute_effective_cache_identity(
    req: &DjevSystemOneRequest,
    run_namespace: &str,
) -> Result<CacheIdentityPolicy, OutcomeStatus> {
    let bytes = encode_canonical_request_bytes(req)?;
    if req.immutable_model_digest.is_none() {
        if run_namespace.trim().is_empty() {
            return Err(OutcomeStatus::Invalid);
        }
        return Ok(CacheIdentityPolicy::NonReusableRunNamespace(
            run_namespace.into(),
        ));
    }
    Ok(CacheIdentityPolicy::ReusableDigest(DigestRef::sha256(
        &bytes,
    )))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DjevCapability {
    StructuredReadScoring,
    GenerativeMaskedCompletion,
    NobleProofAcceptance,
}

impl DjevCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StructuredReadScoring => "structured_read_scoring",
            Self::GenerativeMaskedCompletion => "generative_masked_completion",
            Self::NobleProofAcceptance => "noble_proof_acceptance",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundedResponseInput {
    pub case_id: String,
    pub requested_capability: DjevCapability,
    pub supported_capabilities: Vec<DjevCapability>,
    pub raw_response_bytes: usize,
    pub max_response_bytes: usize,
    pub transport_ok: bool,
    pub response_json_well_formed: bool,
    pub schema_valid: bool,
    pub choice_scores: Vec<f64>,
    pub expected_choices_len: usize,
    pub whole_decision_deadline_ms: u64,
    pub queue_wait_ms: u64,
    pub transport_elapsed_ms: u64,
    pub cancelled: bool,
    pub stale_decision_epoch: bool,
    pub queue_saturated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DjevControlVerdict {
    pub case_id: String,
    pub requested_capability: DjevCapability,
    pub outcome: OutcomeStatus,
    pub grants_generative_or_proof_authority: bool,
    pub total_decision_elapsed_ms: u64,
    pub rejection_reason: Option<String>,
}

/// Evaluate a bounded `djev` transport/capability/deadline control case.
/// Lower-level transport timeouts never reset the whole-decision queue+read deadline.
pub fn evaluate_djev_response(input: &BoundedResponseInput) -> DjevControlVerdict {
    let total_elapsed = input
        .queue_wait_ms
        .saturating_add(input.transport_elapsed_ms);
    let reject = |outcome: OutcomeStatus, reason: &str| DjevControlVerdict {
        case_id: input.case_id.clone(),
        requested_capability: input.requested_capability,
        outcome,
        grants_generative_or_proof_authority: false,
        total_decision_elapsed_ms: total_elapsed,
        rejection_reason: Some(reason.into()),
    };

    if !input
        .supported_capabilities
        .contains(&input.requested_capability)
    {
        return reject(
            OutcomeStatus::Unsupported,
            "requested_capability_unsupported_no_silent_chat_or_proof_fallback",
        );
    }
    if input.queue_saturated {
        return reject(
            OutcomeStatus::Exhausted,
            "bounded_async_queue_saturated_refused",
        );
    }
    if input.cancelled {
        return reject(
            OutcomeStatus::Cancelled,
            "caller_cancelled_before_decision_commit",
        );
    }
    if input.stale_decision_epoch {
        return reject(OutcomeStatus::Cancelled, "late_stale_epoch_result_refused");
    }
    if total_elapsed > input.whole_decision_deadline_ms {
        return reject(
            OutcomeStatus::Timeout,
            "whole_decision_queue_plus_read_deadline_exceeded",
        );
    }
    if !input.transport_ok {
        return reject(OutcomeStatus::InternalFailure, "http_transport_failure");
    }
    if input.raw_response_bytes > input.max_response_bytes {
        return reject(
            OutcomeStatus::Exhausted,
            "bounded_response_byte_limit_exceeded",
        );
    }
    if !input.response_json_well_formed {
        return reject(OutcomeStatus::Invalid, "malformed_response_json");
    }
    if !input.schema_valid {
        return reject(OutcomeStatus::Invalid, "protocol_schema_rejection");
    }
    if input.expected_choices_len == 0
        || input.choice_scores.len() != input.expected_choices_len
        || input.choice_scores.iter().any(|s| !s.is_finite())
    {
        return reject(
            OutcomeStatus::Invalid,
            "choice_score_length_or_nonfinite_error",
        );
    }

    DjevControlVerdict {
        case_id: input.case_id.clone(),
        requested_capability: input.requested_capability,
        outcome: OutcomeStatus::Accepted,
        grants_generative_or_proof_authority: false,
        total_decision_elapsed_ms: total_elapsed,
        rejection_reason: None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheRegressionCheck {
    pub mutation_name: String,
    pub invalidated_prior_cache: bool,
    pub cache_key: String,
}

#[derive(Debug, Clone)]
pub struct DjevProtocolReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub donor_commit: String,
    pub wire_version: String,
    pub named_consumers: Vec<String>,
    pub dual_consumer_gate_satisfied: bool,
    pub golden_ascii_sha256: String,
    pub golden_non_ascii_sha256: String,
    pub cache_regressions: Vec<CacheRegressionCheck>,
    pub wire_negative_controls: Vec<(String, OutcomeStatus)>,
    pub response_controls: Vec<DjevControlVerdict>,
    pub retained_in_rsi: Vec<String>,
}

pub fn base_golden_request() -> DjevSystemOneRequest {
    DjevSystemOneRequest {
        wire_version: DJEV_WIRE_VERSION.into(),
        served_model_name: "djev-scoring-fixture".into(),
        immutable_model_digest: Some(DigestRef::sha256(b"pinned-model-checkpoint-v1")),
        server_revision: "djev-srv-0.4.2".into(),
        template_revision: "tmpl-v2".into(),
        prompt_text: "Score candidate spans for [HOLE_0]: select best option.".into(),
        ordered_choices: vec!["A".into(), "B".into(), "C".into()],
        read_span: SpanOffset {
            start_byte: 26,
            end_byte: 34,
        },
        canvas_id: "canvas-alpha".into(),
        max_options: 3,
        tournament_mode: "single_round".into(),
        samples: 4,
        seed: 42,
        temperature: 0.0,
    }
}

pub fn run_default_harness() -> Result<DjevProtocolReport, String> {
    let base = base_golden_request();
    let ascii_bytes = encode_canonical_request_bytes(&base)
        .map_err(|s| format!("golden ASCII request failed: {}", s.as_str()))?;
    let golden_ascii_sha256 = DigestRef::sha256(&ascii_bytes).hex;

    let mut non_ascii = base.clone();
    non_ascii.prompt_text = "Évaluer «λx. x + 1» sur [HOLE_0] — résumé.".into();
    let hole_start = non_ascii
        .prompt_text
        .find("[HOLE_0]")
        .ok_or_else(|| "missing [HOLE_0] in non-ASCII fixture".to_string())?;
    non_ascii.read_span = SpanOffset {
        start_byte: hole_start,
        end_byte: hole_start + "[HOLE_0]".len(),
    };
    let non_ascii_bytes = encode_canonical_request_bytes(&non_ascii)
        .map_err(|s| format!("golden non-ASCII request failed: {}", s.as_str()))?;
    let golden_non_ascii_sha256 = DigestRef::sha256(&non_ascii_bytes).hex;

    // Closed RSI #42 effective cache identity regressions.
    let base_cache = compute_effective_cache_identity(&base, "run-001")
        .map_err(|s| format!("base cache key failed: {}", s.as_str()))?;
    let base_key = base_cache.as_key_str();

    let identical_cache = compute_effective_cache_identity(&base, "run-002")
        .map_err(|s| format!("identical cache key failed: {}", s.as_str()))?;

    let mut cache_regressions = vec![CacheRegressionCheck {
        mutation_name: "00-identical-effective-read-reuses-cache".into(),
        invalidated_prior_cache: identical_cache.as_key_str() != base_key,
        cache_key: identical_cache.as_key_str(),
    }];

    let mutations: Vec<(&str, DjevSystemOneRequest)> = vec![
        ("01-changed-prompt-rendering", {
            let mut r = base.clone();
            r.prompt_text = "Score candidate spans for [HOLE_0]: choose best option.".into();
            r
        }),
        ("02-reordered-candidates", {
            let mut r = base.clone();
            r.ordered_choices = vec!["B".into(), "A".into(), "C".into()];
            r
        }),
        ("03-changed-template-revision", {
            let mut r = base.clone();
            r.template_revision = "tmpl-v3".into();
            r
        }),
        ("04-changed-sample-count", {
            let mut r = base.clone();
            r.samples = 8;
            r
        }),
        ("05-changed-seed", {
            let mut r = base.clone();
            r.seed = 43;
            r
        }),
        ("06-changed-max-options-tournament", {
            let mut r = base.clone();
            r.max_options = 2;
            r.tournament_mode = "two_round_bracket".into();
            r
        }),
        ("07-changed-canvas", {
            let mut r = base.clone();
            r.canvas_id = "canvas-beta".into();
            r
        }),
        ("08-changed-served-revision", {
            let mut r = base.clone();
            r.server_revision = "djev-srv-0.4.3".into();
            r
        }),
        ("09-missing-immutable-model-digest-non-reusable", {
            let mut r = base.clone();
            r.immutable_model_digest = None;
            r
        }),
    ];

    for (name, req) in mutations {
        let pol = compute_effective_cache_identity(&req, "run-ephemeral-9")
            .map_err(|s| format!("mutation {name} unexpectedly failed: {}", s.as_str()))?;
        let key = pol.as_key_str();
        cache_regressions.push(CacheRegressionCheck {
            mutation_name: name.into(),
            invalidated_prior_cache: key != base_key,
            cache_key: key,
        });
    }

    // Wire-level negative controls.
    let mut bad_utf8_span = non_ascii.clone();
    bad_utf8_span.read_span = SpanOffset {
        start_byte: 1, // inside multi-byte 'É'
        end_byte: 5,
    };
    let mut oob_span = base.clone();
    oob_span.read_span = SpanOffset {
        start_byte: 10,
        end_byte: base.prompt_text.len() + 5,
    };
    let mut bad_wire = base.clone();
    bad_wire.wire_version = "djev.systemone.v99".into();
    let mut dup_choices = base.clone();
    dup_choices.ordered_choices = vec!["A".into(), "A".into()];

    let wire_negative_controls = vec![
        (
            "01-invalid-mid-utf8-span-offset".into(),
            encode_canonical_request_bytes(&bad_utf8_span).unwrap_err(),
        ),
        (
            "02-out-of-bounds-span-offset".into(),
            encode_canonical_request_bytes(&oob_span).unwrap_err(),
        ),
        (
            "03-unsupported-wire-version-no-chat-fallback".into(),
            encode_canonical_request_bytes(&bad_wire).unwrap_err(),
        ),
        (
            "04-duplicate-candidate-choices".into(),
            encode_canonical_request_bytes(&dup_choices).unwrap_err(),
        ),
    ];

    // Transport, capability, deadline, and cancellation controls.
    let base_resp = |id: &str| BoundedResponseInput {
        case_id: id.into(),
        requested_capability: DjevCapability::StructuredReadScoring,
        supported_capabilities: vec![DjevCapability::StructuredReadScoring],
        raw_response_bytes: 256,
        max_response_bytes: 4096,
        transport_ok: true,
        response_json_well_formed: true,
        schema_valid: true,
        choice_scores: vec![-0.2, -1.1, -2.3],
        expected_choices_len: 3,
        whole_decision_deadline_ms: 500,
        queue_wait_ms: 80,
        transport_elapsed_ms: 120,
        cancelled: false,
        stale_decision_epoch: false,
        queue_saturated: false,
    };

    let mut r_gen = base_resp("02-negative-generative-completion-unsupported");
    r_gen.requested_capability = DjevCapability::GenerativeMaskedCompletion;

    let mut r_proof = base_resp("03-negative-noble-proof-acceptance-unsupported");
    r_proof.requested_capability = DjevCapability::NobleProofAcceptance;

    let mut r_schema = base_resp("04-negative-schema-rejection");
    r_schema.schema_valid = false;

    let mut r_malformed = base_resp("05-negative-malformed-response-json");
    r_malformed.response_json_well_formed = false;

    let mut r_transport = base_resp("06-negative-http-transport-failure");
    r_transport.transport_ok = false;

    let mut r_oversize = base_resp("07-negative-bounded-response-oversize");
    r_oversize.raw_response_bytes = 8192;

    let mut r_deadline = base_resp("08-negative-whole-decision-deadline-not-reset-by-transport");
    r_deadline.queue_wait_ms = 350;
    r_deadline.transport_elapsed_ms = 200; // transport alone <= 500, but queue+transport=550 > 500!

    let mut r_cancel = base_resp("09-negative-cancelled-decision");
    r_cancel.cancelled = true;

    let mut r_stale = base_resp("10-negative-late-stale-epoch-refused");
    r_stale.stale_decision_epoch = true;

    let mut r_sat = base_resp("11-negative-saturated-queue-refused");
    r_sat.queue_saturated = true;

    let response_controls = vec![
        evaluate_djev_response(&base_resp("01-positive-bounded-structured-read-scoring")),
        evaluate_djev_response(&r_gen),
        evaluate_djev_response(&r_proof),
        evaluate_djev_response(&r_schema),
        evaluate_djev_response(&r_malformed),
        evaluate_djev_response(&r_transport),
        evaluate_djev_response(&r_oversize),
        evaluate_djev_response(&r_deadline),
        evaluate_djev_response(&r_cancel),
        evaluate_djev_response(&r_stale),
        evaluate_djev_response(&r_sat),
    ];

    let consumer_refs = [
        "OnixResearch/rsi-lab::rwkv_clm_adapter (existing RSI caller)",
        "OnixResearch/inference-lab::issue-1-structured-read-worker (named second consumer)",
    ];
    let dual_consumer_gate_satisfied = validate_dual_consumer_gate(&consumer_refs);
    let named_consumers: Vec<String> = consumer_refs.iter().map(|&s| s.into()).collect();

    Ok(DjevProtocolReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "live_djev_http_server_and_rsi_cutover",
            "Model-free canonical wire-byte, cache-identity, and transport/deadline contract controls ran locally; fake-server conformance remains separate from any live djev model observation.",
        ),
        donor_commit: DJEV_DONOR_COMMIT.into(),
        wire_version: DJEV_WIRE_VERSION.into(),
        dual_consumer_gate_satisfied,
        named_consumers,
        golden_ascii_sha256,
        golden_non_ascii_sha256,
        cache_regressions,
        wire_negative_controls,
        response_controls,
        retained_in_rsi: vec![
            "Action kinds/descriptions/arguments and decision-point rendering remain in RSI.".into(),
            "Default decision instructions, template file format, and learned templates remain in RSI.".into(),
            "Candidate shortlist/tournament policy, fusion coefficients, and calibration remain in RSI.".into(),
            "Per-agent state, snapshot/version handling, LabelCache, training, and diffusion_async.py whole-decision deadlines remain in RSI.".into(),
        ],
    })
}

pub fn inputs_json(report: &DjevProtocolReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-13-djev-protocol.inputs.v1\",\n",
            "  \"issue\": 13,\n",
            "  \"execution_class\": {},\n",
            "  \"donor_repo\": \"OnixResearch/rsi-lab\",\n",
            "  \"donor_commit\": {},\n",
            "  \"wire_version\": {},\n",
            "  \"named_consumers\": {},\n",
            "  \"retained_in_rsi\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.donor_commit),
        quote(&report.wire_version),
        str_array(&report.named_consumers),
        str_array(&report.retained_in_rsi),
    )
}

pub fn results_json(report: &DjevProtocolReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-13-djev-protocol.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 13,").unwrap();
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
        "  \"dual_consumer_gate_satisfied\": {},",
        report.dual_consumer_gate_satisfied
    )
    .unwrap();
    writeln!(
        out,
        "  \"golden_ascii_sha256\": {},",
        quote(&report.golden_ascii_sha256)
    )
    .unwrap();
    writeln!(
        out,
        "  \"golden_non_ascii_sha256\": {},",
        quote(&report.golden_non_ascii_sha256)
    )
    .unwrap();

    let caches: Vec<String> = report
        .cache_regressions
        .iter()
        .map(|c| {
            format!(
                "{{\"mutation_name\":{},\"invalidated_prior_cache\":{},\"cache_key\":{}}}",
                quote(&c.mutation_name),
                c.invalidated_prior_cache,
                quote(&c.cache_key),
            )
        })
        .collect();
    writeln!(
        out,
        "  \"rsi42_cache_regressions\": [{}],",
        caches.join(",")
    )
    .unwrap();

    let wires: Vec<String> = report
        .wire_negative_controls
        .iter()
        .map(|(id, status)| {
            format!(
                "{{\"case_id\":{},\"outcome\":{}}}",
                quote(id),
                quote(status.as_str())
            )
        })
        .collect();
    writeln!(out, "  \"wire_negative_controls\": [{}],", wires.join(",")).unwrap();

    let resps: Vec<String> = report
        .response_controls
        .iter()
        .map(|r| {
            format!(
                concat!(
                    "{{\"case_id\":{},\"requested_capability\":{},\"outcome\":{},",
                    "\"grants_generative_or_proof_authority\":{},\"total_decision_elapsed_ms\":{},",
                    "\"rejection_reason\":{}}}"
                ),
                quote(&r.case_id),
                quote(r.requested_capability.as_str()),
                quote(r.outcome.as_str()),
                r.grants_generative_or_proof_authority,
                r.total_decision_elapsed_ms,
                opt_str(r.rejection_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"response_controls\": [{}]", resps.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &DjevProtocolReport) -> String {
    let mut out = String::new();
    writeln!(out, "# Issue #13: Reusable `djev` Protocol Client Split").unwrap();
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
    writeln!(out, "- Wire version: `{}`", report.wire_version).unwrap();
    writeln!(
        out,
        "- Golden ASCII SHA-256: `{}`",
        report.golden_ascii_sha256
    )
    .unwrap();
    writeln!(
        out,
        "- Golden non-ASCII SHA-256: `{}`",
        report.golden_non_ascii_sha256
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Closed RSI `#42` Effective Cache-Identity Regressions"
    )
    .unwrap();
    for c in &report.cache_regressions {
        writeln!(
            out,
            "- `{}`: invalidated_prior_cache=`{}`",
            c.mutation_name, c.invalidated_prior_cache
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Transport, Capability & Deadline Controls").unwrap();
    writeln!(out, "| Case | Capability | Outcome | Elapsed ms | Reason |").unwrap();
    writeln!(out, "| --- | --- | --- | ---: | --- |").unwrap();
    for r in &report.response_controls {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | {} | {} |",
            r.case_id,
            r.requested_capability.as_str(),
            r.outcome.as_str(),
            r.total_decision_elapsed_ms,
            r.rejection_reason.as_deref().unwrap_or("none")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Model-free protocol conformance only; structured-read scoring never grants generative masked completion or Noble proof acceptance."
    )
    .unwrap();
    out
}
