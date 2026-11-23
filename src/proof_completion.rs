//! Issue #2: Masked proof completion through Compiler Lab's frozen acceptance gate.
//!
//! Second concrete consumer of Issue #1's bounded proposal envelope, keeping intake/checking/replay
//! ownership in Compiler Lab (`#75`, `#160`, `#218`, `#219`), Trellis proof recipes external, and
//! Octet evidence rails external.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_str, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofCheckerFamily {
    Lean4Pinned,
    VerusPinned,
}

impl ProofCheckerFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lean4Pinned => "lean-4.35.0-pinned",
            Self::VerusPinned => "verus-pinned",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProofPartition {
    TrainAdaptation,
    Validation,
    HeldOut,
}

impl ProofPartition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TrainAdaptation => "train_adaptation",
            Self::Validation => "validation",
            Self::HeldOut => "held_out",
        }
    }
}

/// Non-collapsing proof checker verdict required by Issue #2 (all 7 outcomes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofVerdict {
    Proved,
    Tested,
    Rejected,
    Unsupported,
    Unknown,
    Timeout,
    InternalFailure,
}

impl ProofVerdict {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Proved => "proved",
            Self::Tested => "tested",
            Self::Rejected => "rejected",
            Self::Unsupported => "unsupported",
            Self::Unknown => "unknown",
            Self::Timeout => "timeout",
            Self::InternalFailure => "internal_failure",
        }
    }
}

/// Independent fresh-checker outcome observed during obligation recheck.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshCheckerOutcome {
    Proved,
    TestedOnlyNotProved,
    Rejected,
    Unknown,
    Timeout,
    InfrastructureFailure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofObligation {
    pub task_id: String,
    pub theorem_family: String,
    pub partition: ProofPartition,
    pub checker_family: ProofCheckerFamily,
    pub profile_digest: DigestRef,
    pub allowed_imports: Vec<String>,
    pub frozen_prefix: String,
    pub editable_proof_body: String,
    pub frozen_suffix: String,
    pub executable_body_digest: DigestRef,
    pub contract_statement_digest: DigestRef,
    pub max_total_admissions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntakePath {
    GeneratorNeutral75,
    TemporarySavedResultAdapter,
}

impl IntakePath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GeneratorNeutral75 => "compiler-lab-75-generator-neutral",
            Self::TemporarySavedResultAdapter => "temporary-saved-result-adapter",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProofCandidate {
    pub candidate_id: String,
    pub task_id: String,
    pub arm_id: String,
    pub checker_family: ProofCheckerFamily,
    pub profile_digest: DigestRef,
    pub imports: Vec<String>,
    pub submitted_source: String,
    pub executable_body_digest: DigestRef,
    pub contract_statement_digest: DigestRef,
    pub attempt_index_including_repairs: usize,
    pub is_repair_attempt: bool,
    pub producer_claims_completed: bool,
    pub producer_confidence: f64,
    pub repl_empty_goals_observed: bool,
    pub optional_python_tool_used: bool,
    pub retrieval_mode: String,
    pub model_and_retrieval_cost_units: u64,
    pub checker_cost_units: u64,
    pub checker_outcome: FreshCheckerOutcome,
}

/// ExVerus-style temporary saved-result envelope translated before Compiler Lab `#75` lands.
/// Its `exverus_producer_status` is informational only and MUST NEVER override fresh checking.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporarySavedResultEnvelope {
    pub last_repaired_code_path: String,
    pub exverus_producer_status: String,
    pub candidate: ProofCandidate,
}

/// Translate a temporary saved-result envelope into the generator-neutral candidate record,
/// stripping producer status authority while preserving exact bytes and provenance.
pub fn translate_temporary_saved_result(
    envelope: &TemporarySavedResultEnvelope,
) -> Result<ProofCandidate, String> {
    if envelope.last_repaired_code_path.is_empty()
        || envelope.last_repaired_code_path.contains("/home/")
    {
        return Err("invalid_saved_result_path".into());
    }
    let mut translated = envelope.candidate.clone();
    // Producer-specific status ("Verified"/"Completed") is never authoritative; retain only informational flag.
    translated.producer_claims_completed =
        envelope.exverus_producer_status == "Verified" || translated.producer_claims_completed;
    Ok(translated)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofAdmissionResult {
    pub candidate_id: String,
    pub arm_id: String,
    pub retrieval_mode: String,
    pub is_repair_attempt: bool,
    pub model_and_retrieval_cost_units: u64,
    pub checker_cost_units: u64,
    pub intake_path: IntakePath,
    pub verdict: ProofVerdict,
    pub rejection_reason: Option<String>,
    pub frozen_context_preserved: bool,
    pub replay_digest: DigestRef,
}

/// Verify that no theorem family leaks across train/validation/held-out partitions.
pub fn validate_family_partitions(obligations: &[ProofObligation]) -> Vec<String> {
    let mut by_family: BTreeMap<&str, BTreeSet<ProofPartition>> = BTreeMap::new();
    for ob in obligations {
        by_family
            .entry(ob.theorem_family.as_str())
            .or_default()
            .insert(ob.partition);
    }
    let mut leaks = Vec::new();
    for (family, parts) in by_family {
        if parts.len() > 1 {
            let names: Vec<&str> = parts.iter().map(|p| p.as_str()).collect();
            leaks.push(format!("partition_leak:{family}:{}", names.join("+")));
        }
    }
    leaks
}

fn evaluate_core(
    obligation: &ProofObligation,
    candidate: &ProofCandidate,
    intake_path: IntakePath,
    common_tool_cohort_forbids_python: bool,
) -> ProofAdmissionResult {
    let replay_digest = DigestRef::sha256(
        format!(
            "{}:{}:{}",
            obligation.profile_digest.hex, candidate.task_id, candidate.submitted_source
        )
        .as_bytes(),
    );

    let result_with =
        |verdict: ProofVerdict, reason: Option<&str>, frozen_ok: bool| ProofAdmissionResult {
            candidate_id: candidate.candidate_id.clone(),
            arm_id: candidate.arm_id.clone(),
            retrieval_mode: candidate.retrieval_mode.clone(),
            is_repair_attempt: candidate.is_repair_attempt,
            model_and_retrieval_cost_units: candidate.model_and_retrieval_cost_units,
            checker_cost_units: candidate.checker_cost_units,
            intake_path,
            verdict,
            rejection_reason: reason.map(String::from),
            frozen_context_preserved: frozen_ok,
            replay_digest: replay_digest.clone(),
        };

    if candidate.checker_family != obligation.checker_family {
        return result_with(
            ProofVerdict::Unsupported,
            Some("mixed_language_or_checker_family_forbidden"),
            false,
        );
    }
    if candidate.profile_digest != obligation.profile_digest {
        return result_with(
            ProofVerdict::Rejected,
            Some("stale_profile_or_toolchain_identity"),
            false,
        );
    }
    if candidate.attempt_index_including_repairs >= obligation.max_total_admissions {
        return result_with(
            ProofVerdict::Rejected,
            Some("admission_budget_exceeded_including_repairs"),
            true,
        );
    }
    if common_tool_cohort_forbids_python && candidate.optional_python_tool_used {
        return result_with(
            ProofVerdict::Unsupported,
            Some("unmatched_tool_interface_optional_python_disabled"),
            true,
        );
    }
    if candidate.executable_body_digest != obligation.executable_body_digest {
        return result_with(
            ProofVerdict::Rejected,
            Some("frozen_executable_body_modified"),
            false,
        );
    }
    if candidate.contract_statement_digest != obligation.contract_statement_digest {
        return result_with(
            ProofVerdict::Rejected,
            Some("frozen_statement_or_contract_weakened"),
            false,
        );
    }
    let min_len = obligation
        .frozen_prefix
        .len()
        .saturating_add(obligation.frozen_suffix.len());
    if candidate.submitted_source.len() < min_len
        || !candidate
            .submitted_source
            .starts_with(&obligation.frozen_prefix)
        || !candidate
            .submitted_source
            .ends_with(&obligation.frozen_suffix)
    {
        return result_with(
            ProofVerdict::Rejected,
            Some("frozen_context_span_modified"),
            false,
        );
    }
    let allowed: BTreeSet<&str> = obligation
        .allowed_imports
        .iter()
        .map(String::as_str)
        .collect();
    for imp in &candidate.imports {
        if !allowed.contains(imp.as_str()) {
            return result_with(
                ProofVerdict::Rejected,
                Some("disallowed_import_outside_trust_closure"),
                false,
            );
        }
    }

    let body_start = obligation.frozen_prefix.len();
    let body_end = candidate.submitted_source.len() - obligation.frozen_suffix.len();
    let body = &candidate.submitted_source[body_start..body_end];

    for forbidden in ["sorry", "admit", "axiom ", "trusted_oracle"] {
        if body.contains(forbidden) {
            return result_with(
                ProofVerdict::Rejected,
                Some("new_axiom_or_unproved_hole_forbidden"),
                true,
            );
        }
    }
    for placeholder in ["leansearch?", "exact?", "apply?"] {
        if body.contains(placeholder) {
            return result_with(
                ProofVerdict::Rejected,
                Some("authoring_placeholder_not_replaced"),
                true,
            );
        }
    }

    match candidate.checker_outcome {
        FreshCheckerOutcome::Proved => result_with(ProofVerdict::Proved, None, true),
        FreshCheckerOutcome::TestedOnlyNotProved => result_with(
            ProofVerdict::Tested,
            Some("bounded_tests_passed_without_complete_proof"),
            true,
        ),
        FreshCheckerOutcome::Rejected => result_with(
            ProofVerdict::Rejected,
            Some("fresh_checker_refused_despite_producer_confidence"),
            true,
        ),
        FreshCheckerOutcome::Unknown => result_with(
            ProofVerdict::Unknown,
            Some("checker_returned_unknown_inconclusive"),
            true,
        ),
        FreshCheckerOutcome::Timeout => {
            result_with(ProofVerdict::Timeout, Some("checker_budget_timeout"), true)
        }
        FreshCheckerOutcome::InfrastructureFailure => result_with(
            ProofVerdict::InternalFailure,
            Some("checker_infrastructure_failure"),
            true,
        ),
    }
}

/// Independently validate a retained proof candidate through either generator-neutral `#75`
/// intake or the explicit temporary saved-result adapter (confirming producer status never overrides refusal).
pub fn evaluate_proof_candidate(
    obligation: &ProofObligation,
    candidate: &ProofCandidate,
    intake_path: IntakePath,
    common_tool_cohort_forbids_python: bool,
) -> ProofAdmissionResult {
    match intake_path {
        IntakePath::GeneratorNeutral75 => evaluate_core(
            obligation,
            candidate,
            intake_path,
            common_tool_cohort_forbids_python,
        ),
        IntakePath::TemporarySavedResultAdapter => {
            let envelope = TemporarySavedResultEnvelope {
                last_repaired_code_path: format!("artifacts/saved/{}.lean", candidate.candidate_id),
                // Adversarial: producer adapter claims "Verified" even on invalid candidates!
                exverus_producer_status: "Verified".into(),
                candidate: candidate.clone(),
            };
            match translate_temporary_saved_result(&envelope) {
                Ok(translated) => evaluate_core(
                    obligation,
                    &translated,
                    intake_path,
                    common_tool_cohort_forbids_python,
                ),
                Err(reason) => ProofAdmissionResult {
                    candidate_id: candidate.candidate_id.clone(),
                    arm_id: candidate.arm_id.clone(),
                    retrieval_mode: candidate.retrieval_mode.clone(),
                    is_repair_attempt: candidate.is_repair_attempt,
                    model_and_retrieval_cost_units: candidate.model_and_retrieval_cost_units,
                    checker_cost_units: candidate.checker_cost_units,
                    intake_path,
                    verdict: ProofVerdict::InternalFailure,
                    rejection_reason: Some(reason),
                    frozen_context_preserved: false,
                    replay_digest: DigestRef::sha256(candidate.submitted_source.as_bytes()),
                },
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProofArmBudgetSummary {
    pub arm_id: String,
    pub retrieval_mode: String,
    pub eligible_obligations: usize,
    pub attempted_candidates: usize,
    pub repair_attempts_charged: usize,
    pub proved: usize,
    pub tested: usize,
    pub rejected: usize,
    pub unsupported: usize,
    pub unknown: usize,
    pub timeout: usize,
    pub internal_failure: usize,
    pub model_and_retrieval_cost_units: u64,
    pub checker_cost_units: u64,
    pub adaptation_cost_separate_units: u64,
}

impl ProofArmBudgetSummary {
    pub fn accounted_candidates(&self) -> usize {
        self.proved
            + self.tested
            + self.rejected
            + self.unsupported
            + self.unknown
            + self.timeout
            + self.internal_failure
    }
}

/// Derive exact per-arm budget summaries directly from retained admission verdicts.
pub fn summarize_arm_budgets(admissions: &[ProofAdmissionResult]) -> Vec<ProofArmBudgetSummary> {
    let mut by_arm: BTreeMap<&str, ProofArmBudgetSummary> = BTreeMap::new();
    for adm in admissions {
        let entry = by_arm
            .entry(adm.arm_id.as_str())
            .or_insert_with(|| ProofArmBudgetSummary {
                arm_id: adm.arm_id.clone(),
                retrieval_mode: adm.retrieval_mode.clone(),
                eligible_obligations: 1,
                attempted_candidates: 0,
                repair_attempts_charged: 0,
                proved: 0,
                tested: 0,
                rejected: 0,
                unsupported: 0,
                unknown: 0,
                timeout: 0,
                internal_failure: 0,
                model_and_retrieval_cost_units: 0,
                checker_cost_units: 0,
                adaptation_cost_separate_units: 0,
            });
        entry.attempted_candidates += 1;
        if adm.is_repair_attempt {
            entry.repair_attempts_charged += 1;
        }
        entry.model_and_retrieval_cost_units = entry
            .model_and_retrieval_cost_units
            .saturating_add(adm.model_and_retrieval_cost_units);
        entry.checker_cost_units = entry
            .checker_cost_units
            .saturating_add(adm.checker_cost_units);
        match adm.verdict {
            ProofVerdict::Proved => entry.proved += 1,
            ProofVerdict::Tested => entry.tested += 1,
            ProofVerdict::Rejected => entry.rejected += 1,
            ProofVerdict::Unsupported => entry.unsupported += 1,
            ProofVerdict::Unknown => entry.unknown += 1,
            ProofVerdict::Timeout => entry.timeout += 1,
            ProofVerdict::InternalFailure => entry.internal_failure += 1,
        }
    }
    by_arm.into_values().collect()
}

#[derive(Debug, Clone)]
pub struct ProofCompletionReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub cohort_checker_family: ProofCheckerFamily,
    pub max_held_out_tasks: usize,
    pub max_admissions_per_task_arm: usize,
    pub generation_seeds: usize,
    pub clean_partition_leaks: Vec<String>,
    pub leaked_partition_negative_control: Vec<String>,
    pub intake_equivalence_verified: bool,
    pub offline_replay_identical: bool,
    pub admissions: Vec<ProofAdmissionResult>,
    pub arm_budgets: Vec<ProofArmBudgetSummary>,
    pub decision: String,
}

pub fn run_default_harness() -> ProofCompletionReport {
    let profile_digest = DigestRef::sha256(b"compiler-lab-160-lean-4.35.0-pinned-profile-v1");
    let exec_digest = DigestRef::sha256(b"def clampPos (x : Int) : Int := if x < 0 then 0 else x");
    let stmt_digest = DigestRef::sha256(b"theorem clampPos_nonneg (x : Int) : 0 <= clampPos x");

    let held_out_ob = ProofObligation {
        task_id: "lean-ob-01-clamp-nonneg".into(),
        theorem_family: "family-clamp-arith".into(),
        partition: ProofPartition::HeldOut,
        checker_family: ProofCheckerFamily::Lean4Pinned,
        profile_digest: profile_digest.clone(),
        allowed_imports: vec!["Init.Data.Int.Basic".into()],
        frozen_prefix: "theorem clampPos_nonneg (x : Int) : 0 <= clampPos x := by\n  ".into(),
        editable_proof_body: "/* MASK */".into(),
        frozen_suffix: "\n-- END_FROZEN_OBLIGATION".into(),
        executable_body_digest: exec_digest.clone(),
        contract_statement_digest: stmt_digest.clone(),
        max_total_admissions: 8,
    };
    let train_ob = ProofObligation {
        task_id: "lean-ob-train-add".into(),
        theorem_family: "family-add-comm".into(),
        partition: ProofPartition::TrainAdaptation,
        checker_family: ProofCheckerFamily::Lean4Pinned,
        profile_digest: profile_digest.clone(),
        allowed_imports: vec!["Init.Data.Int.Basic".into()],
        frozen_prefix: "theorem add_zero (x : Int) : x + 0 = x := by\n  ".into(),
        editable_proof_body: "/* MASK */".into(),
        frozen_suffix: "\n-- END_FROZEN_OBLIGATION".into(),
        executable_body_digest: exec_digest.clone(),
        contract_statement_digest: stmt_digest.clone(),
        max_total_admissions: 8,
    };
    let mut leaked_variant = held_out_ob.clone();
    leaked_variant.task_id = "lean-ob-leak-control".into();
    leaked_variant.partition = ProofPartition::TrainAdaptation;

    let clean_partition_leaks =
        validate_family_partitions(&[train_ob.clone(), held_out_ob.clone()]);
    let leaked_partition_negative_control =
        validate_family_partitions(&[train_ob, held_out_ob.clone(), leaked_variant]);

    let make_src = |body: &str| {
        format!(
            "{}{}{}",
            held_out_ob.frozen_prefix, body, held_out_ob.frozen_suffix
        )
    };

    let base_cand = |id: &str, arm: &str, body: &str| ProofCandidate {
        candidate_id: id.into(),
        task_id: held_out_ob.task_id.clone(),
        arm_id: arm.into(),
        checker_family: ProofCheckerFamily::Lean4Pinned,
        profile_digest: profile_digest.clone(),
        imports: vec!["Init.Data.Int.Basic".into()],
        submitted_source: make_src(body),
        executable_body_digest: exec_digest.clone(),
        contract_statement_digest: stmt_digest.clone(),
        attempt_index_including_repairs: 0,
        is_repair_attempt: false,
        producer_claims_completed: true,
        producer_confidence: 0.90,
        repl_empty_goals_observed: true,
        optional_python_tool_used: false,
        retrieval_mode: "retrieval_disabled".into(),
        model_and_retrieval_cost_units: 30,
        checker_cost_units: 15,
        checker_outcome: FreshCheckerOutcome::Proved,
    };

    let mut c01 = base_cand(
        "cand-01-valid-proof",
        "diffusion-plus-repair",
        "unfold clampPos; split <;> omega",
    );
    c01.attempt_index_including_repairs = 1;
    c01.is_repair_attempt = true;
    c01.retrieval_mode = "local_indexed_retrieval".into();

    let mut c02 = base_cand("cand-02-sorry-hole-high-conf", "ar-plus-repair", "sorry");
    c02.producer_confidence = 0.99;
    c02.checker_outcome = FreshCheckerOutcome::Rejected;

    let mut c03 = base_cand("cand-03-exec-body-modified", "ar-completion", "omega");
    c03.executable_body_digest = DigestRef::sha256(b"tampered-executable-impl");

    let mut c04 = base_cand("cand-04-weakened-contract", "ar-completion", "trivial");
    c04.contract_statement_digest = DigestRef::sha256(b"theorem clampPos_nonneg : True");
    c04.attempt_index_including_repairs = 1;

    let mut c05 = base_cand("cand-05-disallowed-import", "ar-plus-repair", "omega");
    c05.imports.push("Unsafe.ExternalAxiom".into());
    c05.attempt_index_including_repairs = 1;
    c05.is_repair_attempt = true;

    let mut c06 = base_cand(
        "cand-06-placeholder-leansearch",
        "aesop-duper-baseline",
        "leansearch? \"clamp nonneg\"",
    );
    c06.retrieval_mode = "external_service_retrieval".into();
    c06.checker_outcome = FreshCheckerOutcome::Rejected;

    let mut c07 = base_cand("cand-07-mixed-verus-cohort", "ar-completion", "omega");
    c07.checker_family = ProofCheckerFamily::VerusPinned;

    let mut c08 = base_cand(
        "cand-08-davinci-unmatched-python",
        "davinci-tool-agent-matched",
        "unfold clampPos; split <;> omega",
    );
    c08.optional_python_tool_used = true;
    c08.retrieval_mode = "local_indexed_retrieval".into();

    let mut c09 = base_cand(
        "cand-09-over-repair-budget",
        "diffusion-plus-repair",
        "unfold clampPos; split <;> omega",
    );
    c09.attempt_index_including_repairs = 8;
    c09.is_repair_attempt = true;
    c09.retrieval_mode = "local_indexed_retrieval".into();

    let mut c10 = base_cand("cand-10-checker-timeout", "ar-plus-repair", "aesop");
    c10.attempt_index_including_repairs = 2;
    c10.is_repair_attempt = true;
    c10.checker_outcome = FreshCheckerOutcome::Timeout;

    let mut c11 = base_cand(
        "cand-11-tested-not-proved",
        "deterministic-reuse-tactics",
        "decide",
    );
    c11.model_and_retrieval_cost_units = 0;
    c11.checker_outcome = FreshCheckerOutcome::TestedOnlyNotProved;

    let mut c12 = base_cand("cand-12-checker-unknown", "aesop-duper-baseline", "duper");
    c12.attempt_index_including_repairs = 1;
    c12.checker_outcome = FreshCheckerOutcome::Unknown;

    let mut c13 = base_cand("cand-13-checker-infra-failure", "ar-plus-repair", "omega");
    c13.attempt_index_including_repairs = 3;
    c13.checker_outcome = FreshCheckerOutcome::InfrastructureFailure;

    let mut c14 = base_cand("cand-14-stale-profile-identity", "ar-completion", "omega");
    c14.profile_digest = DigestRef::sha256(b"old-lean-4.34.0-profile");

    let mut c15 = base_cand("cand-15-frozen-span-modified", "ar-completion", "omega");
    c15.submitted_source = "theorem tampered_prefix := by omega".into();

    let candidates = vec![
        c01, c02, c03, c04, c05, c06, c07, c08, c09, c10, c11, c12, c13, c14, c15,
    ];

    let mut admissions = Vec::with_capacity(candidates.len());
    let mut intake_equivalence_verified = true;
    let mut offline_replay_identical = true;

    for cand in &candidates {
        let neutral =
            evaluate_proof_candidate(&held_out_ob, cand, IntakePath::GeneratorNeutral75, true);
        let temp = evaluate_proof_candidate(
            &held_out_ob,
            cand,
            IntakePath::TemporarySavedResultAdapter,
            true,
        );
        let replay =
            evaluate_proof_candidate(&held_out_ob, cand, IntakePath::GeneratorNeutral75, true);

        if neutral.verdict != temp.verdict || neutral.rejection_reason != temp.rejection_reason {
            intake_equivalence_verified = false;
        }
        if neutral != replay {
            offline_replay_identical = false;
        }
        admissions.push(neutral);
    }

    let arm_budgets = summarize_arm_budgets(&admissions);

    ProofCompletionReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "real_prover_model_and_compiler_lab_lean_toolchain",
            "Bounded contract guards, intake equivalence, and offline replay verified locally; real model cohort and live Compiler Lab Lean/Verus checker runs remain external gates.",
        ),
        cohort_checker_family: ProofCheckerFamily::Lean4Pinned,
        max_held_out_tasks: 24,
        max_admissions_per_task_arm: 8,
        generation_seeds: 3,
        clean_partition_leaks,
        leaked_partition_negative_control,
        intake_equivalence_verified,
        offline_replay_identical,
        admissions,
        arm_budgets,
        decision: "INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME".into(),
    }
}

pub fn inputs_json(report: &ProofCompletionReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-02-proof-completion.inputs.v1\",\n",
            "  \"issue\": 2,\n",
            "  \"execution_class\": {},\n",
            "  \"cohort_checker_family\": {},\n",
            "  \"bounds\": {{\"max_held_out_tasks\":{},\"max_admissions_per_task_arm\":{},\"generation_seeds\":{},\"optional_python_enabled\":false}},\n",
            "  \"external_owners\": {{\"intake\":\"compiler-lab#75\",\"lean_policy\":\"compiler-lab#160\",\"repl_sessions\":\"compiler-lab#218\",\"tactic_comparison\":\"compiler-lab#219\",\"compaction\":\"memory-lab#7\",\"training\":\"rsi-lab#84\"}},\n",
            "  \"non_gating_unverified_author_claims\": [\n",
            "    {{\"source\":\"Diffusion-Proof 2026.findings-emnlp.1367\",\"note\":\"pass@32 + 32 corrections (+4 MiniF2F, +0 ProofNet) is not a matched-budget win or Verus transfer\"}},\n",
            "    {{\"source\":\"daVinci-Prover arXiv:2611.16433v1 / GAIR-NLP/daVinci-prover\",\"note\":\"Reported figures are unverified external claims; targets Lean 4.35.0\"}}\n",
            "  ]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(report.cohort_checker_family.as_str()),
        report.max_held_out_tasks,
        report.max_admissions_per_task_arm,
        report.generation_seeds,
    )
}

pub fn results_json(report: &ProofCompletionReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-02-proof-completion.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 2,").unwrap();
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
    writeln!(
        out,
        "  \"clean_partition_leaks\": {},",
        str_array(&report.clean_partition_leaks)
    )
    .unwrap();
    writeln!(
        out,
        "  \"leaked_partition_negative_control\": {},",
        str_array(&report.leaked_partition_negative_control)
    )
    .unwrap();
    writeln!(
        out,
        "  \"intake_equivalence_verified\": {},",
        report.intake_equivalence_verified
    )
    .unwrap();
    writeln!(
        out,
        "  \"offline_replay_identical\": {},",
        report.offline_replay_identical
    )
    .unwrap();

    let adm: Vec<String> = report
        .admissions
        .iter()
        .map(|a| {
            format!(
                concat!(
                    "{{\"candidate_id\":{},\"arm_id\":{},\"intake_path\":{},\"verdict\":{},",
                    "\"rejection_reason\":{},\"frozen_context_preserved\":{},\"replay_digest\":{}}}"
                ),
                quote(&a.candidate_id),
                quote(&a.arm_id),
                quote(a.intake_path.as_str()),
                quote(a.verdict.as_str()),
                opt_str(a.rejection_reason.as_deref()),
                a.frozen_context_preserved,
                a.replay_digest.to_json(),
            )
        })
        .collect();
    writeln!(out, "  \"admissions\": [{}],", adm.join(",")).unwrap();

    let arms: Vec<String> = report
        .arm_budgets
        .iter()
        .map(|b| {
            let accounted = b.accounted_candidates();
            let rate = if b.attempted_candidates > 0 {
                b.proved as f64 / b.attempted_candidates as f64
            } else {
                0.0
            };
            format!(
                concat!(
                    "{{\"arm_id\":{},\"retrieval_mode\":{},\"eligible_obligations\":{},",
                    "\"attempted_candidates\":{},\"accounted_candidates\":{},\"repair_attempts_charged\":{},",
                    "\"proved\":{},\"tested\":{},\"rejected\":{},\"unsupported\":{},\"unknown\":{},",
                    "\"timeout\":{},\"internal_failure\":{},\"proved_fraction\":{},",
                    "\"model_and_retrieval_cost_units\":{},\"checker_cost_units\":{},",
                    "\"adaptation_cost_separate_units\":{}}}"
                ),
                quote(&b.arm_id),
                quote(&b.retrieval_mode),
                b.eligible_obligations,
                b.attempted_candidates,
                accounted,
                b.repair_attempts_charged,
                b.proved,
                b.tested,
                b.rejected,
                b.unsupported,
                b.unknown,
                b.timeout,
                b.internal_failure,
                finite_f64(rate),
                b.model_and_retrieval_cost_units,
                b.checker_cost_units,
                b.adaptation_cost_separate_units,
            )
        })
        .collect();
    writeln!(out, "  \"arm_budgets\": [{}]", arms.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &ProofCompletionReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #2: Masked Proof Completion through Compiler Lab's Frozen Gate"
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
        "- Cohort family: `{}`",
        report.cohort_checker_family.as_str()
    )
    .unwrap();
    writeln!(
        out,
        "- Generator-neutral (`#75`) vs temporary adapter equivalence: `{}`",
        report.intake_equivalence_verified
    )
    .unwrap();
    writeln!(
        out,
        "- Offline replay identical: `{}`",
        report.offline_replay_identical
    )
    .unwrap();
    writeln!(out, "- Decision: `{}`", report.decision).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Candidate Admissions & Frozen-Context Guards").unwrap();
    writeln!(out, "| Candidate | Arm | Verdict | Frozen OK | Reason |").unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- |").unwrap();
    for a in &report.admissions {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{}` | {} |",
            a.candidate_id,
            a.arm_id,
            a.verdict.as_str(),
            a.frozen_context_preserved,
            a.rejection_reason.as_deref().unwrap_or("none")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Bounded contract harness only. Compiler Lab retains proof intake/checking authority; no real model or Lean/Verus checker ran locally."
    )
    .unwrap();
    out
}
