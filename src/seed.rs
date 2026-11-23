//! Bootstrap seed: deterministic serving/KV-budget contract benchmark (`serving-contract-v1`).
//!
//! Runs purely as a `bounded_contract_harness` to validate non-collapsing outcomes
//! (`accepted`, `invalid`, `oom`, `timeout`, `unsupported`), explicit provenance,
//! and byte-identical artifact publication. No physical GPU or model server runs here.

use std::fmt::Write as FmtWrite;

use crate::json::{opt_u64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServingRequest {
    pub id: String,
    pub prompt_tokens: u64,
    pub max_new_tokens: u64,
    pub template_valid: bool,
    pub requires_unsupported_feature: bool,
    pub deadline_logical_ticks: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServingScenario {
    pub id: String,
    pub model_ref: String,
    pub checkpoint_digest: DigestRef,
    pub tokenizer_digest: DigestRef,
    pub quantization: String,
    pub kv_cache_token_limit: u64,
    pub max_batch_tokens: u64,
    pub requests: Vec<ServingRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractProfile {
    UnboundedNaiveRejected,
    StaticBatchBounded,
    PrefillDecodeSplitBounded,
}

impl ContractProfile {
    pub const ALL: [Self; 3] = [
        Self::UnboundedNaiveRejected,
        Self::StaticBatchBounded,
        Self::PrefillDecodeSplitBounded,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnboundedNaiveRejected => "unbounded-naive-rejected",
            Self::StaticBatchBounded => "static-batch-bounded",
            Self::PrefillDecodeSplitBounded => "prefill-decode-split-bounded",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestOutcome {
    pub request_id: String,
    pub status: OutcomeStatus,
    pub total_tokens_requested: u64,
    pub logical_completion_tick: Option<u64>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileSummary {
    pub profile: ContractProfile,
    pub planned: usize,
    pub accepted: usize,
    pub invalid: usize,
    pub oom: usize,
    pub timeout: usize,
    pub unsupported: usize,
    pub peak_kv_tokens: u64,
    pub outcomes: Vec<RequestOutcome>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedReport {
    pub scenario: ServingScenario,
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub profiles: Vec<ProfileSummary>,
}

/// Frozen authored bootstrap fixture exercising every non-collapsing outcome class.
pub fn default_scenario() -> ServingScenario {
    ServingScenario {
        id: "serving-contract-v1".into(),
        model_ref: "reference-dense-toy-0.1".into(),
        checkpoint_digest: DigestRef::sha256(b"inference-lab:seed:checkpoint:v1"),
        tokenizer_digest: DigestRef::sha256(b"inference-lab:seed:tokenizer:v1"),
        quantization: "f64-contract-reference".into(),
        kv_cache_token_limit: 4096,
        max_batch_tokens: 2048,
        requests: vec![
            ServingRequest {
                id: "req-01-valid-short".into(),
                prompt_tokens: 128,
                max_new_tokens: 64,
                template_valid: true,
                requires_unsupported_feature: false,
                deadline_logical_ticks: 20,
            },
            ServingRequest {
                id: "req-02-valid-medium".into(),
                prompt_tokens: 512,
                max_new_tokens: 128,
                template_valid: true,
                requires_unsupported_feature: false,
                deadline_logical_ticks: 40,
            },
            ServingRequest {
                id: "req-03-invalid-template".into(),
                prompt_tokens: 64,
                max_new_tokens: 32,
                template_valid: false,
                requires_unsupported_feature: false,
                deadline_logical_ticks: 20,
            },
            ServingRequest {
                id: "req-04-oom-kv".into(),
                prompt_tokens: 4000,
                max_new_tokens: 512,
                template_valid: true,
                requires_unsupported_feature: false,
                deadline_logical_ticks: 80,
            },
            ServingRequest {
                id: "req-05-tight-deadline".into(),
                prompt_tokens: 768,
                max_new_tokens: 256,
                template_valid: true,
                requires_unsupported_feature: false,
                deadline_logical_ticks: 12,
            },
            ServingRequest {
                id: "req-06-unsupported-backend-op".into(),
                prompt_tokens: 128,
                max_new_tokens: 64,
                template_valid: true,
                requires_unsupported_feature: true,
                deadline_logical_ticks: 30,
            },
        ],
    }
}

/// Evaluate the frozen seed scenario deterministically across all three contract profiles.
pub fn evaluate(scenario: &ServingScenario) -> Result<SeedReport, String> {
    if scenario.requests.is_empty() {
        return Err("serving scenario must include at least one request".into());
    }
    if scenario.kv_cache_token_limit == 0 || scenario.max_batch_tokens == 0 {
        return Err("kv_cache_token_limit and max_batch_tokens must be positive".into());
    }

    let mut profiles = Vec::with_capacity(ContractProfile::ALL.len());
    for profile in ContractProfile::ALL {
        profiles.push(run_profile(scenario, profile));
    }

    Ok(SeedReport {
        scenario: scenario.clone(),
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical-gpu-or-accelerator-serve",
            "Bootstrap seed verifies deterministic contract accounting only; no physical model server or GPU executed.",
        ),
        profiles,
    })
}

fn run_profile(scenario: &ServingScenario, profile: ContractProfile) -> ProfileSummary {
    let mut outcomes = Vec::with_capacity(scenario.requests.len());
    let mut accepted = 0usize;
    let mut invalid = 0usize;
    let mut oom = 0usize;
    let mut timeout = 0usize;
    let mut unsupported = 0usize;
    let mut peak_kv_tokens = 0u64;

    for req in &scenario.requests {
        let total_tokens = req.prompt_tokens.saturating_add(req.max_new_tokens);
        if !req.template_valid {
            invalid += 1;
            outcomes.push(RequestOutcome {
                request_id: req.id.clone(),
                status: OutcomeStatus::Invalid,
                total_tokens_requested: total_tokens,
                logical_completion_tick: None,
                detail: "disqualifying chat-template/input contract violation".into(),
            });
            continue;
        }
        if req.requires_unsupported_feature {
            unsupported += 1;
            outcomes.push(RequestOutcome {
                request_id: req.id.clone(),
                status: OutcomeStatus::Unsupported,
                total_tokens_requested: total_tokens,
                logical_completion_tick: None,
                detail: "requested backend capability is unavailable in this profile".into(),
            });
            continue;
        }

        let effective_kv_limit = match profile {
            ContractProfile::UnboundedNaiveRejected => scenario.kv_cache_token_limit / 2,
            ContractProfile::StaticBatchBounded | ContractProfile::PrefillDecodeSplitBounded => {
                scenario.kv_cache_token_limit
            }
        };

        if total_tokens > effective_kv_limit {
            oom += 1;
            outcomes.push(RequestOutcome {
                request_id: req.id.clone(),
                status: OutcomeStatus::Oom,
                total_tokens_requested: total_tokens,
                logical_completion_tick: None,
                detail: format!(
                    "requested {total_tokens} KV tokens exceeds profile limit {effective_kv_limit}"
                ),
            });
            continue;
        }

        // Deterministic logical tick model (NOT physical latency).
        let prefill_ticks = req.prompt_tokens.div_ceil(128);
        let decode_ticks = req.max_new_tokens.div_ceil(32);
        let logical_ticks = match profile {
            ContractProfile::UnboundedNaiveRejected => (prefill_ticks + decode_ticks) * 2,
            ContractProfile::StaticBatchBounded => prefill_ticks + decode_ticks + 2,
            ContractProfile::PrefillDecodeSplitBounded => prefill_ticks.max(decode_ticks) + 2,
        };

        if logical_ticks > req.deadline_logical_ticks {
            timeout += 1;
            outcomes.push(RequestOutcome {
                request_id: req.id.clone(),
                status: OutcomeStatus::Timeout,
                total_tokens_requested: total_tokens,
                logical_completion_tick: None,
                detail: format!(
                    "logical work {logical_ticks} exceeded deadline {}",
                    req.deadline_logical_ticks
                ),
            });
            continue;
        }

        accepted += 1;
        peak_kv_tokens = peak_kv_tokens.max(total_tokens);
        outcomes.push(RequestOutcome {
            request_id: req.id.clone(),
            status: OutcomeStatus::Accepted,
            total_tokens_requested: total_tokens,
            logical_completion_tick: Some(logical_ticks),
            detail: "bounded contract checks satisfied".into(),
        });
    }

    ProfileSummary {
        profile,
        planned: scenario.requests.len(),
        accepted,
        invalid,
        oom,
        timeout,
        unsupported,
        peak_kv_tokens,
        outcomes,
    }
}

pub fn inputs_json(scenario: &ServingScenario) -> String {
    let reqs: Vec<String> = scenario
        .requests
        .iter()
        .map(|r| {
            format!(
                "{{\"id\":{},\"prompt_tokens\":{},\"max_new_tokens\":{},\"template_valid\":{},\"requires_unsupported_feature\":{},\"deadline_logical_ticks\":{}}}",
                quote(&r.id),
                r.prompt_tokens,
                r.max_new_tokens,
                r.template_valid,
                r.requires_unsupported_feature,
                r.deadline_logical_ticks
            )
        })
        .collect();
    format!(
        "{{\"schema\":\"inference-lab.seed-inputs.v1\",\"scenario_id\":{},\"model_ref\":{},\"checkpoint_digest\":{},\"tokenizer_digest\":{},\"quantization\":{},\"kv_cache_token_limit\":{},\"max_batch_tokens\":{},\"requests\":[{}]}}\n",
        quote(&scenario.id),
        quote(&scenario.model_ref),
        scenario.checkpoint_digest.to_json(),
        scenario.tokenizer_digest.to_json(),
        quote(&scenario.quantization),
        scenario.kv_cache_token_limit,
        scenario.max_batch_tokens,
        reqs.join(",\n")
    )
}

pub fn results_json(report: &SeedReport) -> String {
    let profiles: Vec<String> = report
        .profiles
        .iter()
        .map(|p| {
            let outcomes: Vec<String> = p
                .outcomes
                .iter()
                .map(|o| {
                    format!(
                        "{{\"request_id\":{},\"status\":{},\"total_tokens_requested\":{},\"logical_completion_tick\":{},\"detail\":{}}}",
                        quote(&o.request_id),
                        quote(o.status.as_str()),
                        o.total_tokens_requested,
                        opt_u64(o.logical_completion_tick),
                        quote(&o.detail)
                    )
                })
                .collect();
            format!(
                "{{\"profile\":{},\"planned\":{},\"accepted\":{},\"invalid\":{},\"oom\":{},\"timeout\":{},\"unsupported\":{},\"peak_kv_tokens\":{},\"outcomes\":[{}]}}",
                quote(p.profile.as_str()),
                p.planned,
                p.accepted,
                p.invalid,
                p.oom,
                p.timeout,
                p.unsupported,
                p.peak_kv_tokens,
                outcomes.join(",\n")
            )
        })
        .collect();
    format!(
        "{{\"schema\":\"inference-lab.seed-report.v1\",\"package_version\":{},\"scenario_id\":{},\"execution_class\":{},\"hardware_gate\":{},\"source_revision\":null,\"source_revision_status\":\"not_recorded\",\"profiles\":[{}]}}\n",
        quote(env!("CARGO_PKG_VERSION")),
        quote(&report.scenario.id),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        profiles.join(",\n")
    )
}

pub fn summary_markdown(report: &SeedReport) -> String {
    let mut out = String::from(
        "# Inference Lab bootstrap seed report\n\n\
         - **Execution class**: `bounded_contract_harness`\n\
         - **Physical hardware gate**: `unavailable_hardware_runtime` (no physical GPU or model server executed)\n\
         - **Scenario**: `serving-contract-v1`\n\n\
         | Profile | Accepted/planned | Invalid | OOM | Timeout | Unsupported | Peak KV tokens |\n\
         | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    for p in &report.profiles {
        writeln!(
            out,
            "| `{}` | {}/{} | {} | {} | {} | {} | {} |",
            p.profile.as_str(),
            p.accepted,
            p.planned,
            p.invalid,
            p.oom,
            p.timeout,
            p.unsupported,
            p.peak_kv_tokens
        )
        .expect("writing to String cannot fail");
    }
    out.push_str(
        "\nEvery planned request remains represented with its explicit non-collapsing outcome. \
         Logical ticks are deterministic contract units, never wall-clock latency or GPU throughput. \
         Presence of `COMPLETE` marks completed artifact publication only.\n",
    );
    out
}
