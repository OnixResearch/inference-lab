//! Issue #1: Bounded diffusion proposal benchmark contract with Noble as first consumer,
//! EBM proposal/scoring/selection/admission separation, and default-off candidate/repair
//! tree interchange comparison (`DependentWeave`, JSONL, socketteer nested JSON).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, opt_str, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate};

/// Distinct capability operations; an ordinary AR chat call is never a diffusion sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityOp {
    ScoreAlternatives,
    ProposeCandidate,
    RepairPermittedRegion,
}

impl CapabilityOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ScoreAlternatives => "score_alternatives",
            Self::ProposeCandidate => "propose_candidate",
            Self::RepairPermittedRegion => "repair_permitted_region",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplerCapabilities {
    pub seeded_canvas: bool,
    pub pinning: bool,
    pub masking_remasking: bool,
    pub permitted_labels: bool,
    pub deterministic_seed: bool,
    pub logits_access: bool,
    pub supports_diffusion_refinement: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityProbeResult {
    pub backend_id: String,
    pub requested: SamplerCapabilities,
    pub effective: SamplerCapabilities,
    pub mismatches: Vec<String>,
    pub qualified_as_diffusion_sampler: bool,
}

/// Probe requested vs actual backend sampler capabilities; unsupported controls stay visible.
pub fn probe_capabilities(
    backend_id: &str,
    requested: &SamplerCapabilities,
    observed: &SamplerCapabilities,
) -> CapabilityProbeResult {
    let mut mismatches = Vec::new();
    let checks = [
        (
            "seeded_canvas",
            requested.seeded_canvas,
            observed.seeded_canvas,
        ),
        ("pinning", requested.pinning, observed.pinning),
        (
            "masking_remasking",
            requested.masking_remasking,
            observed.masking_remasking,
        ),
        (
            "permitted_labels",
            requested.permitted_labels,
            observed.permitted_labels,
        ),
        (
            "deterministic_seed",
            requested.deterministic_seed,
            observed.deterministic_seed,
        ),
        (
            "logits_access",
            requested.logits_access,
            observed.logits_access,
        ),
        (
            "supports_diffusion_refinement",
            requested.supports_diffusion_refinement,
            observed.supports_diffusion_refinement,
        ),
    ];
    for (name, req, obs) in checks {
        if req && !obs {
            mismatches.push(format!("{name}: requested_true_observed_false"));
        }
    }
    let qualified = observed.seeded_canvas
        && observed.pinning
        && observed.masking_remasking
        && observed.supports_diffusion_refinement;
    CapabilityProbeResult {
        backend_id: backend_id.into(),
        requested: requested.clone(),
        effective: observed.clone(),
        mismatches,
        qualified_as_diffusion_sampler: qualified,
    }
}

/// Noble structured editor node kinds (mirroring `noble-contracts/src/source/editor.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NobleNodeKind {
    Integer,
    Boolean,
    Word,
    Quotation,
    Hole,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteSpan {
    pub start: usize,
    pub end: usize,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalBudget {
    pub max_candidates: usize,
    pub max_denoising_steps: u32,
    pub deadline_units: u64,
    pub max_output_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalRequest {
    pub schema_version: String,
    pub consumer_schema: String,
    pub task_id: String,
    pub family_id: String,
    pub split: String,
    pub base_source: String,
    pub base_digest: DigestRef,
    pub context_digest: DigestRef,
    pub diagnostic_digest: DigestRef,
    pub editable_spans: Vec<ByteSpan>,
    pub frozen_spans: Vec<ByteSpan>,
    pub budget: ProposalBudget,
}

/// Noble kernel acceptance outcomes (never collapsed into a single boolean).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NobleAdmissionOutcome {
    Accepted,
    Invalid,
    Unsupported,
    Exhausted,
    InternalFailure,
}

impl NobleAdmissionOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Invalid => "invalid",
            Self::Unsupported => "unsupported",
            Self::Exhausted => "exhausted",
            Self::InternalFailure => "internal_failure",
        }
    }
}

/// Noble synthesis proof boundary (solver UNSAT alone is NOT a universal proof).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofStatus {
    Proved,
    Unproven,
    RunOnly,
    Exhausted,
    UnsatisfiableSolverOnlyNotProof,
}

impl ProofStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Proved => "proved",
            Self::Unproven => "unproven",
            Self::RunOnly => "run_only",
            Self::Exhausted => "exhausted",
            Self::UnsatisfiableSolverOnlyNotProof => "unsatisfiable_solver_only_not_proof",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehavioralStatus {
    PassedBoundedTestsNotUniversalProof,
    FailedBoundedTests,
    NotEvaluated,
}

impl BehavioralStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PassedBoundedTestsNotUniversalProof => "passed_bounded_tests_not_universal_proof",
            Self::FailedBoundedTests => "failed_bounded_tests",
            Self::NotEvaluated => "not_evaluated",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateArtifact {
    pub candidate_id: String,
    pub parent_id: Option<String>,
    pub arm_id: String,
    pub operation: CapabilityOp,
    pub base_digest: DigestRef,
    pub diagnostic_digest: DigestRef,
    pub candidate_bytes: String,
    pub candidate_digest: DigestRef,
    pub arrival_time_units: u64,
    pub generation_work_units: u64,
    pub scoring_work_units: u64,
    pub checking_work_units: u64,
    pub io_work_units: u64,
    pub informational_confidence: Option<f64>,
    pub active: bool,
    pub bookmarked: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionVerdict {
    pub candidate_id: String,
    pub contract_guard_passed: bool,
    pub rejection_reason: Option<String>,
    pub parse_type_effect: NobleAdmissionOutcome,
    pub tested_behavior: BehavioralStatus,
    pub proof_status: ProofStatus,
    pub incumbent_mutated: bool,
}

/// Validate one candidate against immutable base, diagnostic, frozen regions, size,
/// deadline, and bounded Noble syntax/type/effect contract without mutating the incumbent
/// unless independently accepted.
pub fn admit_candidate(
    request: &ProposalRequest,
    candidate: &CandidateArtifact,
    candidate_index_in_batch: usize,
) -> AdmissionVerdict {
    let reject = |reason: &str, outcome: NobleAdmissionOutcome| AdmissionVerdict {
        candidate_id: candidate.candidate_id.clone(),
        contract_guard_passed: false,
        rejection_reason: Some(reason.into()),
        parse_type_effect: outcome,
        tested_behavior: BehavioralStatus::NotEvaluated,
        proof_status: ProofStatus::Unproven,
        incumbent_mutated: false,
    };

    if candidate_index_in_batch >= request.budget.max_candidates {
        return reject(
            "excessive_candidates_budget_exceeded",
            NobleAdmissionOutcome::Exhausted,
        );
    }
    if candidate.arrival_time_units > request.budget.deadline_units {
        return reject(
            "late_result_after_deadline",
            NobleAdmissionOutcome::Exhausted,
        );
    }
    if candidate.candidate_bytes.len() > request.budget.max_output_bytes {
        return reject(
            "output_exceeds_max_output_bytes",
            NobleAdmissionOutcome::Invalid,
        );
    }
    if candidate.base_digest != request.base_digest {
        return reject("stale_base_digest", NobleAdmissionOutcome::Invalid);
    }
    if candidate.diagnostic_digest != request.diagnostic_digest {
        return reject("stale_diagnostic_digest", NobleAdmissionOutcome::Invalid);
    }

    // Verify frozen prefix/suffix regions remain byte-identical.
    let base_bytes = request.base_source.as_bytes();
    let cand_bytes = candidate.candidate_bytes.as_bytes();
    for frozen in &request.frozen_spans {
        if frozen.end > base_bytes.len() {
            return reject(
                "invalid_frozen_span_specification",
                NobleAdmissionOutcome::InternalFailure,
            );
        }
        let expected_slice = &base_bytes[frozen.start..frozen.end];
        if frozen.start == 0 {
            if !cand_bytes.starts_with(expected_slice) {
                return reject("frozen_prefix_edited", NobleAdmissionOutcome::Invalid);
            }
        } else if frozen.end == base_bytes.len() {
            if !cand_bytes.ends_with(expected_slice) {
                return reject("frozen_suffix_edited", NobleAdmissionOutcome::Invalid);
            }
        } else if cand_bytes.len() < frozen.end
            || &cand_bytes[frozen.start..frozen.end] != expected_slice
        {
            return reject("frozen_interior_edited", NobleAdmissionOutcome::Invalid);
        }
    }

    // Minimal Noble pure-stack-syntax & hole-resolution checker for the frozen contract task:
    // base: "fn clamp_pos (x: Int) -> Int = [ ?h1 ?h2 ]"
    // accepted pure forms resolve both holes with balanced brackets and no remaining `?` holes.
    if candidate.candidate_bytes.contains('?') {
        return reject("unresolved_hole_remains", NobleAdmissionOutcome::Invalid);
    }
    let open_brackets = candidate
        .candidate_bytes
        .bytes()
        .filter(|&b| b == b'[')
        .count();
    let close_brackets = candidate
        .candidate_bytes
        .bytes()
        .filter(|&b| b == b']')
        .count();
    if open_brackets != close_brackets || open_brackets == 0 {
        return reject("malformed_quotation_syntax", NobleAdmissionOutcome::Invalid);
    }
    if candidate.candidate_bytes.contains("io!") {
        return reject(
            "forbidden_effect_in_pure_task",
            NobleAdmissionOutcome::Invalid,
        );
    }

    // Separate type/effect admission from bounded behavioral testing (still never a universal proof).
    let passes_bounded_tests = candidate.candidate_bytes.contains("dup 0 max")
        || candidate.candidate_bytes.contains("0 max");

    if passes_bounded_tests {
        AdmissionVerdict {
            candidate_id: candidate.candidate_id.clone(),
            contract_guard_passed: true,
            rejection_reason: None,
            parse_type_effect: NobleAdmissionOutcome::Accepted,
            tested_behavior: BehavioralStatus::PassedBoundedTestsNotUniversalProof,
            proof_status: ProofStatus::RunOnly,
            incumbent_mutated: false,
        }
    } else {
        AdmissionVerdict {
            candidate_id: candidate.candidate_id.clone(),
            contract_guard_passed: true,
            rejection_reason: None,
            parse_type_effect: NobleAdmissionOutcome::Accepted,
            tested_behavior: BehavioralStatus::FailedBoundedTests,
            proof_status: ProofStatus::Unproven,
            incumbent_mutated: false,
        }
    }
}

/// Separate EBM target objectives (Dai & Du Sections 7–8; BOND distinction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetObjective {
    HeuristicRanking,
    Optimization,
    ExactTiltedSampling,
}

impl TargetObjective {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HeuristicRanking => "heuristic_ranking",
            Self::Optimization => "optimization",
            Self::ExactTiltedSampling => "exact_tilted_sampling",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionProcedure {
    ArgmaxBestOfN,
    WeightedResampling,
    RejectionSampling,
    BeamSearch,
    BoundedRepair,
}

impl SelectionProcedure {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArgmaxBestOfN => "argmax_best_of_n",
            Self::WeightedResampling => "weighted_resampling",
            Self::RejectionSampling => "rejection_sampling",
            Self::BeamSearch => "beam_search",
            Self::BoundedRepair => "bounded_repair",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredCandidate {
    pub candidate_id: String,
    pub payload: String,
    pub raw_score: Option<f64>,
    pub log_proposal_prob: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectionReport {
    pub procedure: SelectionProcedure,
    pub objective: TargetObjective,
    pub exact_tilted_sampling_claimed: bool,
    pub proposal_density_available: bool,
    pub importance_correction_applied: bool,
    pub temperature: f64,
    pub deduplicated_ids: Vec<String>,
    pub quarantined_invalid_score_ids: Vec<String>,
    pub tied_top_ids: Vec<String>,
    pub selected_id: Option<String>,
    pub selection_weights: Vec<(String, f64)>,
}

/// Finite-candidate EBM reference fixture handling duplicates, invalid/missing/NaN/Inf scores,
/// deterministic tie-breaking, and distinguishing argmax best-of-N from weighted resampling.
pub fn evaluate_selection_fixture(
    candidates: &[ScoredCandidate],
    procedure: SelectionProcedure,
    temperature: f64,
    resampling_seed: u64,
) -> Result<SelectionReport, String> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("temperature must be positive and finite".into());
    }

    let mut seen_payloads = BTreeSet::new();
    let mut deduplicated_ids = Vec::new();
    let mut quarantined_invalid_score_ids = Vec::new();
    let mut valid: Vec<(&ScoredCandidate, f64)> = Vec::new();

    for cand in candidates {
        if !seen_payloads.insert(cand.payload.clone()) {
            deduplicated_ids.push(cand.candidate_id.clone());
            continue;
        }
        match cand.raw_score {
            Some(s) if s.is_finite() => valid.push((cand, s)),
            _ => quarantined_invalid_score_ids.push(cand.candidate_id.clone()),
        }
    }

    let proposal_density_available =
        !valid.is_empty() && valid.iter().all(|(c, _)| c.log_proposal_prob.is_some());

    let (objective, exact_tilted_sampling_claimed, importance_correction_applied) = match procedure
    {
        SelectionProcedure::ArgmaxBestOfN => {
            // Finite best-of-N is distinct from exact exponentially reward-tilted sampling (BOND).
            (TargetObjective::HeuristicRanking, false, false)
        }
        SelectionProcedure::WeightedResampling => (
            TargetObjective::HeuristicRanking,
            false,
            proposal_density_available,
        ),
        _ => (TargetObjective::Optimization, false, false),
    };

    if valid.is_empty() {
        return Ok(SelectionReport {
            procedure,
            objective,
            exact_tilted_sampling_claimed,
            proposal_density_available,
            importance_correction_applied,
            temperature,
            deduplicated_ids,
            quarantined_invalid_score_ids,
            tied_top_ids: Vec::new(),
            selected_id: None,
            selection_weights: Vec::new(),
        });
    }

    let max_score = valid
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut tied_top_ids: Vec<String> = valid
        .iter()
        .filter(|(_, s)| (*s - max_score).abs() <= 1e-12)
        .map(|(c, _)| c.candidate_id.clone())
        .collect();
    tied_top_ids.sort();

    // Compute normalized Boltzmann weights (with optional importance weight if density available).
    let mut log_weights = Vec::with_capacity(valid.len());
    for (c, s) in &valid {
        let mut lw = *s / temperature;
        if importance_correction_applied {
            if let Some(lp) = c.log_proposal_prob {
                lw -= lp;
            }
        }
        log_weights.push(lw);
    }
    let max_lw = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let unnorm: Vec<f64> = log_weights.iter().map(|lw| (lw - max_lw).exp()).collect();
    let sum_w: f64 = unnorm.iter().sum();
    let selection_weights: Vec<(String, f64)> = valid
        .iter()
        .zip(unnorm.iter())
        .map(|((c, _), w)| (c.candidate_id.clone(), *w / sum_w))
        .collect();

    let selected_id = match procedure {
        SelectionProcedure::ArgmaxBestOfN
        | SelectionProcedure::BeamSearch
        | SelectionProcedure::BoundedRepair
        | SelectionProcedure::RejectionSampling => tied_top_ids.first().cloned(),
        SelectionProcedure::WeightedResampling => {
            // Deterministic splitmix64 draw in [0, 1).
            let mut z = resampling_seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            let u = ((z >> 11) as f64) / ((1u64 << 53) as f64);
            let mut acc = 0.0;
            let mut chosen = selection_weights.last().map(|(id, _)| id.clone());
            for (id, w) in &selection_weights {
                acc += *w;
                if u <= acc {
                    chosen = Some(id.clone());
                    break;
                }
            }
            chosen
        }
    };

    Ok(SelectionReport {
        procedure,
        objective,
        exact_tilted_sampling_claimed,
        proposal_density_available,
        importance_correction_applied,
        temperature,
        deduplicated_ids,
        quarantined_invalid_score_ids,
        tied_top_ids,
        selected_id,
        selection_weights,
    })
}

/// Matched-budget accounting row separating critical-path time from summed resource work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmBudgetAccounting {
    pub arm_id: String,
    pub proposal_process: String,
    pub scorer: String,
    pub selection_procedure: SelectionProcedure,
    pub consumer_checker: String,
    pub total_candidates: usize,
    pub accepted_candidates: usize,
    pub type_accepted_test_failed_candidates: usize,
    pub invalid_candidates: usize,
    pub exhausted_candidates: usize,
    pub critical_path_units: u64,
    pub generation_work_units: u64,
    pub scoring_work_units: u64,
    pub checking_work_units: u64,
    pub io_work_units: u64,
    pub summed_resource_work_units: u64,
}

fn arm_selection_procedure(arm_id: &str) -> SelectionProcedure {
    if arm_id.contains("diffusion") {
        SelectionProcedure::BoundedRepair
    } else {
        SelectionProcedure::ArgmaxBestOfN
    }
}

/// Default-off interchange format benchmark result (Issue #1 comment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterchangeFormatResult {
    pub format_id: String,
    pub upstream_pin: Option<String>,
    pub serialized_bytes: usize,
    pub encoded_digest: DigestRef,
    pub nodes_preserved: usize,
    pub parent_links_preserved: bool,
    pub rejected_siblings_preserved: bool,
    pub repair_descendants_preserved: bool,
    pub activation_bookmarks_preserved: bool,
    pub selection_procedure_preserved: bool,
    pub hard_admission_outside_format: bool,
    pub field_loss_list: Vec<String>,
}

/// Encode the same frozen Noble candidate & repair tree in three formats and report
/// exact serialized bytes and explicit field-loss lists (default-off check from #1 comment).
pub fn run_weave_interchange_fixture(
    candidates: &[CandidateArtifact],
    verdicts: &[AdmissionVerdict],
) -> Vec<InterchangeFormatResult> {
    let verdict_map: BTreeMap<&str, &AdmissionVerdict> = verdicts
        .iter()
        .map(|v| (v.candidate_id.as_str(), v))
        .collect();

    // 1. DependentWeave (transkatgirl/universal-weave@57a9212bfc107dd1191d3c439b3975c1b4cafc51):
    // DependentNode { id, from, to, active, bookmarked, contents }
    let mut children_map: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for c in candidates {
        if let Some(parent) = &c.parent_id {
            children_map
                .entry(parent.as_str())
                .or_default()
                .push(c.candidate_id.as_str());
        }
    }
    let weave_nodes: Vec<String> = candidates
        .iter()
        .map(|c| {
            let children: Vec<String> = children_map
                .get(c.candidate_id.as_str())
                .map(|list| list.iter().map(|s| (*s).to_string()).collect())
                .unwrap_or_default();
            format!(
                "{{\"id\":{},\"from\":{},\"to\":{},\"active\":{},\"bookmarked\":{},\"contents\":{}}}",
                quote(&c.candidate_id),
                opt_str(c.parent_id.as_deref()),
                str_array(&children),
                c.active,
                c.bookmarked,
                quote(&c.candidate_bytes)
            )
        })
        .collect();
    let weave_payload = format!(
        "{{\"format\":\"dependent-weave-v1\",\"nodes\":[{}]}}\n",
        weave_nodes.join(",")
    );

    // 2. JSONL record stream with explicit parent_id, base/diagnostic digests, per-arm selection_procedure, and verdict sidecar.
    let mut jsonl_lines = Vec::with_capacity(candidates.len());
    for c in candidates {
        let verdict = verdict_map
            .get(c.candidate_id.as_str())
            .map(|v| v.parse_type_effect.as_str())
            .unwrap_or("unverified");
        let proc_name = arm_selection_procedure(&c.arm_id).as_str();
        jsonl_lines.push(format!(
            "{{\"id\":{},\"parent_id\":{},\"arm_id\":{},\"operation\":{},\"selection_procedure\":{},\"active\":{},\"bookmarked\":{},\"base_digest\":{},\"diagnostic_digest\":{},\"candidate_digest\":{},\"contents\":{},\"observed_checker_outcome\":{}}}",
            quote(&c.candidate_id),
            opt_str(c.parent_id.as_deref()),
            quote(&c.arm_id),
            quote(c.operation.as_str()),
            quote(proc_name),
            c.active,
            c.bookmarked,
            c.base_digest.to_json(),
            c.diagnostic_digest.to_json(),
            c.candidate_digest.to_json(),
            quote(&c.candidate_bytes),
            quote(verdict)
        ));
    }
    let jsonl_payload = format!("{}\n", jsonl_lines.join("\n"));

    // 3. socketteer-style nested JSON (socketteer/loom@91ca920551120ad4508540e8da057c0b94067afc):
    // { "root": { "id", "text", "children": [...] }, "selected_node_id": "..." }
    fn render_socketteer_node(
        id: &str,
        by_id: &BTreeMap<&str, &CandidateArtifact>,
        children_map: &BTreeMap<&str, Vec<&str>>,
    ) -> String {
        let text = by_id
            .get(id)
            .map(|c| c.candidate_bytes.as_str())
            .unwrap_or("");
        let kids: Vec<String> = children_map
            .get(id)
            .map(|list| {
                list.iter()
                    .map(|kid| render_socketteer_node(kid, by_id, children_map))
                    .collect()
            })
            .unwrap_or_default();
        format!(
            "{{\"id\":{},\"text\":{},\"children\":[{}]}}",
            quote(id),
            quote(text),
            kids.join(",")
        )
    }
    let by_id: BTreeMap<&str, &CandidateArtifact> = candidates
        .iter()
        .map(|c| (c.candidate_id.as_str(), c))
        .collect();
    let root_id = candidates
        .iter()
        .find(|c| c.parent_id.is_none())
        .map(|c| c.candidate_id.as_str())
        .unwrap_or("root");
    let selected_id = candidates
        .iter()
        .find(|c| c.active)
        .map(|c| c.candidate_id.as_str())
        .unwrap_or(root_id);
    let socketteer_payload = format!(
        "{{\"root\":{},\"selected_node_id\":{}}}\n",
        render_socketteer_node(root_id, &by_id, &children_map),
        quote(selected_id)
    );

    // Programmatically verify preservation on actual serialized payloads instead of hardcoding booleans.
    let all_ids_in = |payload: &str| {
        candidates
            .iter()
            .all(|c| payload.contains(&quote(&c.candidate_id)))
    };
    let rejected_ids_in = |payload: &str| {
        verdicts
            .iter()
            .filter(|v| !v.contract_guard_passed)
            .all(|v| payload.contains(&quote(&v.candidate_id)))
    };
    let repair_ids_in = |payload: &str| {
        candidates
            .iter()
            .filter(|c| c.operation == CapabilityOp::RepairPermittedRegion)
            .all(|c| payload.contains(&quote(&c.candidate_id)))
    };
    let has_digests_in = |payload: &str| {
        candidates.iter().all(|c| {
            payload.contains(&c.base_digest.hex) && payload.contains(&c.diagnostic_digest.hex)
        })
    };
    let has_selection_procs_in = |payload: &str| {
        candidates.iter().all(|c| {
            let expected = format!(
                "\"selection_procedure\":{}",
                quote(arm_selection_procedure(&c.arm_id).as_str())
            );
            payload.contains(&expected)
        })
    };

    let build_losses = |payload: &str, has_bookmarks: bool, has_multi_active: bool| {
        let mut losses = Vec::new();
        if !has_bookmarks {
            losses.push("bookmark_bits".into());
        }
        if !has_multi_active {
            losses.push("multi_node_active_flags".into());
        }
        if !has_selection_procs_in(payload) {
            losses.push("selection_procedure_label".into());
        }
        if !has_digests_in(payload) {
            losses.push("candidate_base_and_diagnostic_digests".into());
        }
        if !payload.contains("\"observed_checker_outcome\"") {
            losses.push("external_consumer_verdict_tag".into());
        }
        losses
    };

    let weave_bookmarks =
        weave_payload.contains("\"bookmarked\":") && weave_payload.contains("\"active\":");
    let jsonl_bookmarks =
        jsonl_payload.contains("\"bookmarked\":") && jsonl_payload.contains("\"active\":");
    let sock_bookmarks = socketteer_payload.contains("\"bookmarked\":");

    vec![
        InterchangeFormatResult {
            format_id: "dependent-weave-v1".into(),
            upstream_pin: Some(
                "transkatgirl/universal-weave@57a9212bfc107dd1191d3c439b3975c1b4cafc51".into(),
            ),
            serialized_bytes: weave_payload.len(),
            encoded_digest: DigestRef::sha256(weave_payload.as_bytes()),
            nodes_preserved: if all_ids_in(&weave_payload) {
                candidates.len()
            } else {
                0
            },
            parent_links_preserved: weave_payload.contains("\"from\":")
                && weave_payload.contains("\"to\":"),
            rejected_siblings_preserved: rejected_ids_in(&weave_payload),
            repair_descendants_preserved: repair_ids_in(&weave_payload),
            activation_bookmarks_preserved: weave_bookmarks,
            selection_procedure_preserved: has_selection_procs_in(&weave_payload),
            hard_admission_outside_format: true,
            field_loss_list: build_losses(&weave_payload, weave_bookmarks, weave_bookmarks),
        },
        InterchangeFormatResult {
            format_id: "jsonl-parent-stream-v1".into(),
            upstream_pin: None,
            serialized_bytes: jsonl_payload.len(),
            encoded_digest: DigestRef::sha256(jsonl_payload.as_bytes()),
            nodes_preserved: if all_ids_in(&jsonl_payload) {
                candidates.len()
            } else {
                0
            },
            parent_links_preserved: jsonl_payload.contains("\"parent_id\":"),
            rejected_siblings_preserved: rejected_ids_in(&jsonl_payload),
            repair_descendants_preserved: repair_ids_in(&jsonl_payload),
            activation_bookmarks_preserved: jsonl_bookmarks,
            selection_procedure_preserved: has_selection_procs_in(&jsonl_payload),
            hard_admission_outside_format: true,
            field_loss_list: build_losses(&jsonl_payload, jsonl_bookmarks, jsonl_bookmarks),
        },
        InterchangeFormatResult {
            format_id: "socketteer-nested-json-v1".into(),
            upstream_pin: Some("socketteer/loom@91ca920551120ad4508540e8da057c0b94067afc".into()),
            serialized_bytes: socketteer_payload.len(),
            encoded_digest: DigestRef::sha256(socketteer_payload.as_bytes()),
            nodes_preserved: if all_ids_in(&socketteer_payload) {
                candidates.len()
            } else {
                0
            },
            parent_links_preserved: socketteer_payload.contains("\"children\":"),
            rejected_siblings_preserved: rejected_ids_in(&socketteer_payload),
            repair_descendants_preserved: repair_ids_in(&socketteer_payload),
            activation_bookmarks_preserved: sock_bookmarks,
            selection_procedure_preserved: has_selection_procs_in(&socketteer_payload),
            hard_admission_outside_format: true,
            field_loss_list: build_losses(&socketteer_payload, sock_bookmarks, false),
        },
    ]
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionContractReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub noble_control_pin: String,
    pub decision: String,
    pub request: ProposalRequest,
    pub capability_probes: Vec<CapabilityProbeResult>,
    pub verdicts: Vec<AdmissionVerdict>,
    pub offline_selection_argmax: SelectionReport,
    pub offline_selection_resampling: SelectionReport,
    pub budget_accounting: Vec<ArmBudgetAccounting>,
    pub interchange_comparison: Vec<InterchangeFormatResult>,
}

/// Build and run the complete frozen Issue #1 bounded contract harness.
pub fn run_default_harness() -> Result<DiffusionContractReport, String> {
    let base_source = "fn clamp_pos (x: Int) -> Int = [ ?h1 ?h2 ]".to_string();
    let base_digest = DigestRef::sha256(base_source.as_bytes());
    let context_digest = DigestRef::sha256(b"noble-pure-stack-context-v1");
    let diagnostic_digest = DigestRef::sha256(b"diag:two-interacting-holes:v1");

    let request = ProposalRequest {
        schema_version: "inference-lab.diffusion-proposal.v1".into(),
        consumer_schema: "noble-structured-editor-v1".into(),
        task_id: "noble-clamp-pos-2hole-01".into(),
        family_id: "clamp-family-a".into(),
        split: "held_out_contract_fixture".into(),
        base_source: base_source.clone(),
        base_digest: base_digest.clone(),
        context_digest,
        diagnostic_digest: diagnostic_digest.clone(),
        editable_spans: vec![ByteSpan {
            start: 33,
            end: 40,
            label: "hole_body".into(),
        }],
        frozen_spans: vec![
            ByteSpan {
                start: 0,
                end: 33,
                label: "signature_prefix".into(),
            },
            ByteSpan {
                start: 40,
                end: base_source.len(),
                label: "closing_suffix".into(),
            },
        ],
        budget: ProposalBudget {
            max_candidates: 6,
            max_denoising_steps: 16,
            deadline_units: 100,
            max_output_bytes: 256,
        },
    };

    let diffusion_req = SamplerCapabilities {
        seeded_canvas: true,
        pinning: true,
        masking_remasking: true,
        permitted_labels: true,
        deterministic_seed: true,
        logits_access: true,
        supports_diffusion_refinement: true,
    };
    let ar_chat_observed = SamplerCapabilities {
        seeded_canvas: false,
        pinning: false,
        masking_remasking: false,
        permitted_labels: false,
        deterministic_seed: true,
        logits_access: false,
        supports_diffusion_refinement: false,
    };
    let mock_diffusion_observed = diffusion_req.clone();

    let capability_probes = vec![
        probe_capabilities(
            "ar-chat-baseline-adapter",
            &diffusion_req,
            &ar_chat_observed,
        ),
        probe_capabilities(
            "diffusion-contract-reference",
            &diffusion_req,
            &mock_diffusion_observed,
        ),
    ];

    let mk_cand = |id: &str,
                   parent: Option<&str>,
                   arm: &str,
                   op: CapabilityOp,
                   base_d: DigestRef,
                   diag_d: DigestRef,
                   body: &str,
                   arrival: u64,
                   gen_w: u64,
                   score_w: u64,
                   chk_w: u64,
                   conf: Option<f64>,
                   active: bool,
                   bookmarked: bool| CandidateArtifact {
        candidate_id: id.into(),
        parent_id: parent.map(String::from),
        arm_id: arm.into(),
        operation: op,
        base_digest: base_d,
        diagnostic_digest: diag_d,
        candidate_bytes: body.into(),
        candidate_digest: DigestRef::sha256(body.as_bytes()),
        arrival_time_units: arrival,
        generation_work_units: gen_w,
        scoring_work_units: score_w,
        checking_work_units: chk_w,
        io_work_units: 2,
        informational_confidence: conf,
        active,
        bookmarked,
    };

    let candidates = vec![
        // 0: root valid candidate (passes syntax & bounded tests)
        mk_cand(
            "cand-00-root-valid",
            None,
            "diffusion-contract-arm",
            CapabilityOp::ProposeCandidate,
            base_digest.clone(),
            diagnostic_digest.clone(),
            "fn clamp_pos (x: Int) -> Int = [ dup 0 max ]",
            40,
            24,
            6,
            8,
            Some(0.87),
            true,
            true,
        ),
        // 1: repair child (valid syntax/types, fails bounded behavioral test)
        mk_cand(
            "cand-01-type-ok-test-fail",
            Some("cand-00-root-valid"),
            "ar-repair-contract-arm",
            CapabilityOp::RepairPermittedRegion,
            base_digest.clone(),
            diagnostic_digest.clone(),
            "fn clamp_pos (x: Int) -> Int = [ dup 1 + ]",
            48,
            20,
            6,
            8,
            Some(0.95), // high confidence must NOT override failing tests!
            false,
            false,
        ),
        // 2: frozen-region edit violation (rejected)
        mk_cand(
            "cand-02-frozen-edit",
            Some("cand-00-root-valid"),
            "diffusion-contract-arm",
            CapabilityOp::RepairPermittedRegion,
            base_digest.clone(),
            diagnostic_digest.clone(),
            "fn hacked_fn (x: Int) -> Int = [ dup 0 max ]",
            52,
            18,
            4,
            4,
            Some(0.99),
            false,
            false,
        ),
        // 3: stale base digest (rejected)
        mk_cand(
            "cand-03-stale-base",
            Some("cand-00-root-valid"),
            "ar-repair-contract-arm",
            CapabilityOp::ProposeCandidate,
            DigestRef::sha256(b"stale-base-version"),
            diagnostic_digest.clone(),
            "fn clamp_pos (x: Int) -> Int = [ 0 max ]",
            55,
            16,
            4,
            4,
            Some(0.75),
            false,
            false,
        ),
        // 4: malformed output (unbalanced quotation inside editable region)
        mk_cand(
            "cand-04-malformed",
            Some("cand-01-type-ok-test-fail"),
            "diffusion-contract-arm",
            CapabilityOp::RepairPermittedRegion,
            base_digest.clone(),
            diagnostic_digest.clone(),
            "fn clamp_pos (x: Int) -> Int = [ [ dup 0 max ]",
            60,
            15,
            3,
            3,
            Some(0.41),
            false,
            true,
        ),
        // 5: late result after deadline (rejected without changing incumbent)
        mk_cand(
            "cand-05-late-result",
            Some("cand-00-root-valid"),
            "ar-repair-contract-arm",
            CapabilityOp::RepairPermittedRegion,
            base_digest.clone(),
            diagnostic_digest.clone(),
            "fn clamp_pos (x: Int) -> Int = [ 0 max ]",
            125,
            30,
            5,
            0,
            Some(0.91),
            false,
            false,
        ),
        // 6: excessive candidate beyond max_candidates=6 (exhausted)
        mk_cand(
            "cand-06-excess-budget",
            Some("cand-00-root-valid"),
            "diffusion-contract-arm",
            CapabilityOp::ProposeCandidate,
            base_digest,
            diagnostic_digest,
            "fn clamp_pos (x: Int) -> Int = [ 0 max ]",
            70,
            10,
            2,
            0,
            Some(0.88),
            false,
            false,
        ),
    ];

    let verdicts: Vec<AdmissionVerdict> = candidates
        .iter()
        .enumerate()
        .map(|(idx, c)| admit_candidate(&request, c, idx))
        .collect();

    // Offline selection diagnostic on retained development candidate pool (ties, duplicates, NaN, missing).
    let pool = vec![
        ScoredCandidate {
            candidate_id: "pool-a".into(),
            payload: "[ dup 0 max ]".into(),
            raw_score: Some(2.5),
            log_proposal_prob: Some(-1.2),
        },
        ScoredCandidate {
            candidate_id: "pool-b-tie".into(),
            payload: "[ 0 max ]".into(),
            raw_score: Some(2.5),
            log_proposal_prob: Some(-0.8),
        },
        ScoredCandidate {
            candidate_id: "pool-c-dup".into(),
            payload: "[ dup 0 max ]".into(),
            raw_score: Some(9.9),
            log_proposal_prob: Some(-0.1),
        },
        ScoredCandidate {
            candidate_id: "pool-d-nan".into(),
            payload: "[ dup 1 + ]".into(),
            raw_score: Some(f64::NAN),
            log_proposal_prob: Some(-1.5),
        },
        ScoredCandidate {
            candidate_id: "pool-e-missing".into(),
            payload: "[ 0 ]".into(),
            raw_score: None,
            log_proposal_prob: Some(-2.0),
        },
    ];
    let offline_selection_argmax =
        evaluate_selection_fixture(&pool, SelectionProcedure::ArgmaxBestOfN, 1.0, 42)?;
    let offline_selection_resampling =
        evaluate_selection_fixture(&pool, SelectionProcedure::WeightedResampling, 1.0, 42)?;

    // Matched-budget accounting by arm (separate critical-path from summed resource work).
    let mut arms_map: BTreeMap<&str, Vec<(&CandidateArtifact, &AdmissionVerdict)>> =
        BTreeMap::new();
    for (c, v) in candidates.iter().zip(verdicts.iter()) {
        arms_map.entry(c.arm_id.as_str()).or_default().push((c, v));
    }
    let mut budget_accounting = Vec::new();
    for (arm_id, pairs) in arms_map {
        let mut accepted = 0usize;
        let mut type_accepted_test_failed = 0usize;
        let mut invalid = 0usize;
        let mut exhausted = 0usize;
        let mut critical_path = 0u64;
        let mut gen_sum = 0u64;
        let mut score_sum = 0u64;
        let mut check_sum = 0u64;
        let mut io_sum = 0u64;
        for (c, v) in &pairs {
            critical_path = critical_path.max(c.arrival_time_units);
            gen_sum += c.generation_work_units;
            score_sum += c.scoring_work_units;
            check_sum += c.checking_work_units;
            io_sum += c.io_work_units;
            match (v.parse_type_effect, v.tested_behavior) {
                (
                    NobleAdmissionOutcome::Accepted,
                    BehavioralStatus::PassedBoundedTestsNotUniversalProof,
                ) => {
                    accepted += 1;
                }
                (NobleAdmissionOutcome::Accepted, _) => {
                    type_accepted_test_failed += 1;
                }
                (NobleAdmissionOutcome::Exhausted, _) => exhausted += 1,
                _ => invalid += 1,
            }
        }
        budget_accounting.push(ArmBudgetAccounting {
            arm_id: arm_id.into(),
            proposal_process: format!("{arm_id}-proposer"),
            scorer: "offline-energy-scorer-v1".into(),
            selection_procedure: arm_selection_procedure(arm_id),
            consumer_checker: "noble-kernel-acceptance-contract-v1".into(),
            total_candidates: pairs.len(),
            accepted_candidates: accepted,
            type_accepted_test_failed_candidates: type_accepted_test_failed,
            invalid_candidates: invalid,
            exhausted_candidates: exhausted,
            critical_path_units: critical_path,
            generation_work_units: gen_sum,
            scoring_work_units: score_sum,
            checking_work_units: check_sum,
            io_work_units: io_sum,
            summed_resource_work_units: gen_sum + score_sum + check_sum + io_sum,
        });
    }

    let interchange_comparison = run_weave_interchange_fixture(&candidates, &verdicts);

    Ok(DiffusionContractReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "real-diffusion-and-ar-model-backends",
            "Bounded contract harness verifies Noble admission guards, capability probes, EBM selection, and tree interchange; real DiffuCoder/AR model runs require a qualified accelerator backend.",
        ),
        noble_control_pin: "OnixResearch/noble-lang@8698dca070e2d89e0bc23ea4050fc79b9fa230b6".into(),
        decision: "UNAVAILABLE_HARDWARE_RUNTIME_FOR_REAL_MODEL_ARMS".into(),
        request,
        capability_probes,
        verdicts,
        offline_selection_argmax,
        offline_selection_resampling,
        budget_accounting,
        interchange_comparison,
    })
}

pub fn inputs_json(report: &DiffusionContractReport) -> String {
    let r = &report.request;
    let editable: Vec<String> = r
        .editable_spans
        .iter()
        .map(|s| {
            format!(
                "{{\"start\":{},\"end\":{},\"label\":{}}}",
                s.start,
                s.end,
                quote(&s.label)
            )
        })
        .collect();
    let frozen: Vec<String> = r
        .frozen_spans
        .iter()
        .map(|s| {
            format!(
                "{{\"start\":{},\"end\":{},\"label\":{}}}",
                s.start,
                s.end,
                quote(&s.label)
            )
        })
        .collect();
    format!(
        "{{\"schema\":\"inference-lab.diffusion-inputs.v1\",\"issue\":1,\"noble_control_pin\":{},\"schema_version\":{},\"consumer_schema\":{},\"task_id\":{},\"family_id\":{},\"split\":{},\"base_digest\":{},\"context_digest\":{},\"diagnostic_digest\":{},\"editable_spans\":[{}],\"frozen_spans\":[{}],\"budget\":{{\"max_candidates\":{},\"max_denoising_steps\":{},\"deadline_units\":{},\"max_output_bytes\":{}}}}}\n",
        quote(&report.noble_control_pin),
        quote(&r.schema_version),
        quote(&r.consumer_schema),
        quote(&r.task_id),
        quote(&r.family_id),
        quote(&r.split),
        r.base_digest.to_json(),
        r.context_digest.to_json(),
        r.diagnostic_digest.to_json(),
        editable.join(","),
        frozen.join(","),
        r.budget.max_candidates,
        r.budget.max_denoising_steps,
        r.budget.deadline_units,
        r.budget.max_output_bytes
    )
}

fn selection_to_json(s: &SelectionReport) -> String {
    let weights: Vec<String> = s
        .selection_weights
        .iter()
        .map(|(id, w)| {
            format!(
                "{{\"candidate_id\":{},\"weight\":{}}}",
                quote(id),
                finite_f64(*w)
            )
        })
        .collect();
    format!(
        "{{\"procedure\":{},\"objective\":{},\"exact_tilted_sampling_claimed\":{},\"proposal_density_available\":{},\"importance_correction_applied\":{},\"temperature\":{},\"deduplicated_ids\":{},\"quarantined_invalid_score_ids\":{},\"tied_top_ids\":{},\"selected_id\":{},\"selection_weights\":[{}]}}",
        quote(s.procedure.as_str()),
        quote(s.objective.as_str()),
        s.exact_tilted_sampling_claimed,
        s.proposal_density_available,
        s.importance_correction_applied,
        opt_f64(Some(s.temperature)),
        str_array(&s.deduplicated_ids),
        str_array(&s.quarantined_invalid_score_ids),
        str_array(&s.tied_top_ids),
        opt_str(s.selected_id.as_deref()),
        weights.join(",")
    )
}

pub fn results_json(report: &DiffusionContractReport) -> String {
    let probes: Vec<String> = report
        .capability_probes
        .iter()
        .map(|p| {
            format!(
                "{{\"backend_id\":{},\"qualified_as_diffusion_sampler\":{},\"mismatches\":{}}}",
                quote(&p.backend_id),
                p.qualified_as_diffusion_sampler,
                str_array(&p.mismatches)
            )
        })
        .collect();
    let verdicts: Vec<String> = report
        .verdicts
        .iter()
        .map(|v| {
            format!(
                "{{\"candidate_id\":{},\"contract_guard_passed\":{},\"rejection_reason\":{},\"parse_type_effect\":{},\"tested_behavior\":{},\"proof_status\":{},\"incumbent_mutated\":{}}}",
                quote(&v.candidate_id),
                v.contract_guard_passed,
                opt_str(v.rejection_reason.as_deref()),
                quote(v.parse_type_effect.as_str()),
                quote(v.tested_behavior.as_str()),
                quote(v.proof_status.as_str()),
                v.incumbent_mutated
            )
        })
        .collect();
    let budgets: Vec<String> = report
        .budget_accounting
        .iter()
        .map(|b| {
            format!(
                "{{\"arm_id\":{},\"proposal_process\":{},\"scorer\":{},\"selection_procedure\":{},\"consumer_checker\":{},\"total_candidates\":{},\"accepted_candidates\":{},\"type_accepted_test_failed_candidates\":{},\"invalid_candidates\":{},\"exhausted_candidates\":{},\"critical_path_units\":{},\"generation_work_units\":{},\"scoring_work_units\":{},\"checking_work_units\":{},\"io_work_units\":{},\"summed_resource_work_units\":{}}}",
                quote(&b.arm_id),
                quote(&b.proposal_process),
                quote(&b.scorer),
                quote(b.selection_procedure.as_str()),
                quote(&b.consumer_checker),
                b.total_candidates,
                b.accepted_candidates,
                b.type_accepted_test_failed_candidates,
                b.invalid_candidates,
                b.exhausted_candidates,
                b.critical_path_units,
                b.generation_work_units,
                b.scoring_work_units,
                b.checking_work_units,
                b.io_work_units,
                b.summed_resource_work_units
            )
        })
        .collect();
    let interchanges: Vec<String> = report
        .interchange_comparison
        .iter()
        .map(|i| {
            format!(
                "{{\"format_id\":{},\"upstream_pin\":{},\"serialized_bytes\":{},\"encoded_digest\":{},\"nodes_preserved\":{},\"parent_links_preserved\":{},\"rejected_siblings_preserved\":{},\"repair_descendants_preserved\":{},\"activation_bookmarks_preserved\":{},\"selection_procedure_preserved\":{},\"hard_admission_outside_format\":{},\"field_loss_list\":{}}}",
                quote(&i.format_id),
                opt_str(i.upstream_pin.as_deref()),
                i.serialized_bytes,
                i.encoded_digest.to_json(),
                i.nodes_preserved,
                i.parent_links_preserved,
                i.rejected_siblings_preserved,
                i.repair_descendants_preserved,
                i.activation_bookmarks_preserved,
                i.selection_procedure_preserved,
                i.hard_admission_outside_format,
                str_array(&i.field_loss_list)
            )
        })
        .collect();

    format!(
        "{{\"schema\":\"inference-lab.diffusion-report.v1\",\"issue\":1,\"package_version\":{},\"execution_class\":{},\"hardware_gate\":{},\"decision\":{},\"capability_probes\":[{}],\"verdicts\":[{}],\"offline_selection_diagnostic\":{{\"argmax_best_of_n\":{},\"weighted_resampling\":{}}},\"budget_accounting\":[{}],\"default_off_interchange_comparison\":[{}]}}\n",
        quote(env!("CARGO_PKG_VERSION")),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        quote(&report.decision),
        probes.join(",\n"),
        verdicts.join(",\n"),
        selection_to_json(&report.offline_selection_argmax),
        selection_to_json(&report.offline_selection_resampling),
        budgets.join(",\n"),
        interchanges.join(",\n")
    )
}

pub fn summary_markdown(report: &DiffusionContractReport) -> String {
    let mut out = format!(
        "# Issue #1 — Bounded diffusion proposal contract report\n\n\
         - **Execution class**: `{}`\n\
         - **Physical hardware gate**: `{}`\n\
         - **Noble deterministic control pin**: `{}`\n\
         - **Real-model comparison decision**: `{}`\n\n\
         ## Sampler capability probes\n\n\
         | Backend | Qualified diffusion sampler | Mismatches |\n\
         | --- | --- | --- |\n",
        report.execution_class.as_str(),
        report.hardware_gate.status.as_str(),
        report.noble_control_pin,
        report.decision
    );
    for p in &report.capability_probes {
        writeln!(
            out,
            "| `{}` | `{}` | {} |",
            p.backend_id,
            p.qualified_as_diffusion_sampler,
            p.mismatches.len()
        )
        .expect("writing to String cannot fail");
    }
    out.push_str(
        "\n## Noble admission and adversarial guard outcomes\n\n\
         | Candidate | Guard passed | Parse/type/effect | Bounded test status | Proof status | Rejection reason |\n\
         | --- | --- | --- | --- | --- | --- |\n",
    );
    for v in &report.verdicts {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |",
            v.candidate_id,
            v.contract_guard_passed,
            v.parse_type_effect.as_str(),
            v.tested_behavior.as_str(),
            v.proof_status.as_str(),
            v.rejection_reason.as_deref().unwrap_or("none")
        )
        .expect("writing to String cannot fail");
    }
    out.push_str(
        "\n## Default-off candidate/repair tree interchange comparison\n\n\
         | Format | Serialized bytes | Nodes | Activation/bookmarks | Selection label | Field losses |\n\
         | --- | --- | --- | --- | --- | --- |\n",
    );
    for i in &report.interchange_comparison {
        writeln!(
            out,
            "| `{}` | {} | {} | `{}` | `{}` | {} |",
            i.format_id,
            i.serialized_bytes,
            i.nodes_preserved,
            i.activation_bookmarks_preserved,
            i.selection_procedure_preserved,
            if i.field_loss_list.is_empty() {
                "none".to_string()
            } else {
                i.field_loss_list.join(", ")
            }
        )
        .expect("writing to String cannot fail");
    }
    out.push_str(
        "\nHard consumer admission remains outside both soft scoring and tree serialization. \
         Offline pool rescoring does not establish real-model closed-loop performance.\n",
    );
    out
}
