use inference_lab::diffusion::{
    admit_candidate, evaluate_selection_fixture, probe_capabilities, run_default_harness,
    BehavioralStatus, CandidateArtifact, CapabilityOp, NobleAdmissionOutcome, ProofStatus,
    SamplerCapabilities, ScoredCandidate, SelectionProcedure,
};
use inference_lab::provenance::{DigestRef, ExecutionClass};

#[test]
fn capability_probe_exposes_ar_chat_lack_of_diffusion_support() {
    let report = run_default_harness().expect("diffusion harness runs");
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );

    let ar_probe = report
        .capability_probes
        .iter()
        .find(|p| p.backend_id == "ar-chat-baseline-adapter")
        .unwrap();
    assert!(!ar_probe.qualified_as_diffusion_sampler);
    assert!(!ar_probe.mismatches.is_empty());

    let diff_probe = report
        .capability_probes
        .iter()
        .find(|p| p.backend_id == "diffusion-contract-reference")
        .unwrap();
    assert!(diff_probe.qualified_as_diffusion_sampler);
    assert!(diff_probe.mismatches.is_empty());

    // Partial diffusion capability (missing pinning) must still fail qualification.
    let partial = SamplerCapabilities {
        seeded_canvas: true,
        pinning: false,
        masking_remasking: true,
        permitted_labels: true,
        deterministic_seed: true,
        logits_access: true,
        supports_diffusion_refinement: true,
    };
    let res = probe_capabilities("partial", &diff_probe.requested, &partial);
    assert!(!res.qualified_as_diffusion_sampler);
    assert!(res.mismatches.iter().any(|m| m.starts_with("pinning:")));
}

#[test]
fn noble_adversarial_guards_reject_without_mutating_incumbent() {
    let report = run_default_harness().unwrap();
    let find_v = |id: &str| {
        report
            .verdicts
            .iter()
            .find(|v| v.candidate_id == id)
            .unwrap()
    };

    // Valid candidate passes syntax and bounded tests (still RunOnly, never universal proof).
    let root = find_v("cand-00-root-valid");
    assert!(root.contract_guard_passed);
    assert_eq!(root.parse_type_effect, NobleAdmissionOutcome::Accepted);
    assert_eq!(
        root.tested_behavior,
        BehavioralStatus::PassedBoundedTestsNotUniversalProof
    );
    assert_eq!(root.proof_status, ProofStatus::RunOnly);

    // Type/syntax accepted candidate that fails bounded tests keeps both dimensions separate!
    let type_ok_fail = find_v("cand-01-type-ok-test-fail");
    assert_eq!(
        type_ok_fail.parse_type_effect,
        NobleAdmissionOutcome::Accepted
    );
    assert_eq!(
        type_ok_fail.tested_behavior,
        BehavioralStatus::FailedBoundedTests
    );
    assert_eq!(type_ok_fail.proof_status, ProofStatus::Unproven);

    // Frozen edit, stale base, malformed, late, excessive are all rejected without mutating incumbent.
    for (id, expected_reason, expected_outcome) in [
        (
            "cand-02-frozen-edit",
            "frozen_prefix_edited",
            NobleAdmissionOutcome::Invalid,
        ),
        (
            "cand-03-stale-base",
            "stale_base_digest",
            NobleAdmissionOutcome::Invalid,
        ),
        (
            "cand-04-malformed",
            "malformed_quotation_syntax",
            NobleAdmissionOutcome::Invalid,
        ),
        (
            "cand-05-late-result",
            "late_result_after_deadline",
            NobleAdmissionOutcome::Exhausted,
        ),
        (
            "cand-06-excess-budget",
            "excessive_candidates_budget_exceeded",
            NobleAdmissionOutcome::Exhausted,
        ),
    ] {
        let v = find_v(id);
        assert!(!v.contract_guard_passed, "{id} must be rejected");
        assert_eq!(v.rejection_reason.as_deref(), Some(expected_reason));
        assert_eq!(v.parse_type_effect, expected_outcome);
        assert!(!v.incumbent_mutated);
    }

    // Stale diagnostic digest is also rejected.
    let stale_diag = CandidateArtifact {
        candidate_id: "cand-stale-diag".into(),
        parent_id: None,
        arm_id: "test".into(),
        operation: CapabilityOp::RepairPermittedRegion,
        base_digest: report.request.base_digest.clone(),
        diagnostic_digest: DigestRef::sha256(b"old-diag"),
        candidate_bytes: "fn clamp_pos (x: Int) -> Int = [ dup 0 max ]".into(),
        candidate_digest: DigestRef::sha256(b"fn clamp_pos (x: Int) -> Int = [ dup 0 max ]"),
        arrival_time_units: 10,
        generation_work_units: 1,
        scoring_work_units: 1,
        checking_work_units: 1,
        io_work_units: 1,
        informational_confidence: Some(0.99),
        active: false,
        bookmarked: false,
    };
    let v = admit_candidate(&report.request, &stale_diag, 0);
    assert!(!v.contract_guard_passed);
    assert_eq!(
        v.rejection_reason.as_deref(),
        Some("stale_diagnostic_digest")
    );
}

#[test]
fn ebm_selection_fixture_handles_ties_duplicates_nan_missing_and_resampling() {
    let report = run_default_harness().unwrap();
    let argmax = &report.offline_selection_argmax;
    let resamp = &report.offline_selection_resampling;

    assert!(!argmax.exact_tilted_sampling_claimed);
    assert_eq!(argmax.deduplicated_ids, vec!["pool-c-dup"]);
    assert_eq!(
        argmax.quarantined_invalid_score_ids,
        vec!["pool-d-nan", "pool-e-missing"]
    );
    assert_eq!(argmax.tied_top_ids, vec!["pool-a", "pool-b-tie"]);
    assert_eq!(argmax.selected_id.as_deref(), Some("pool-a"));

    // Weighted resampling sees proposal density and applies importance correction without claiming exact sampling.
    assert!(resamp.proposal_density_available);
    assert!(resamp.importance_correction_applied);
    assert!(!resamp.exact_tilted_sampling_claimed);
    assert!(resamp.selected_id.is_some());

    // Infinite score must also be quarantined.
    let inf_pool = vec![ScoredCandidate {
        candidate_id: "c-inf".into(),
        payload: "x".into(),
        raw_score: Some(f64::INFINITY),
        log_proposal_prob: None,
    }];
    let inf_res =
        evaluate_selection_fixture(&inf_pool, SelectionProcedure::ArgmaxBestOfN, 1.0, 1).unwrap();
    assert_eq!(inf_res.quarantined_invalid_score_ids, vec!["c-inf"]);
    assert_eq!(inf_res.selected_id, None);
}

#[test]
fn default_off_interchange_comparison_reports_exact_field_losses() {
    let report = run_default_harness().unwrap();
    assert_eq!(report.interchange_comparison.len(), 3);

    // Budget accounting must keep type-accepted-test-failed separate from invalid with exact conservation.
    let mut saw_type_ok_test_fail = false;
    for arm in &report.budget_accounting {
        let sum = arm.accepted_candidates
            + arm.type_accepted_test_failed_candidates
            + arm.invalid_candidates
            + arm.exhausted_candidates;
        assert_eq!(sum, arm.total_candidates);
        if arm.arm_id == "ar-repair-contract-arm" {
            assert_eq!(arm.type_accepted_test_failed_candidates, 1);
            saw_type_ok_test_fail = true;
        }
    }
    assert!(saw_type_ok_test_fail);

    let weave = &report.interchange_comparison[0];
    let jsonl = &report.interchange_comparison[1];
    let sock = &report.interchange_comparison[2];

    assert_eq!(weave.format_id, "dependent-weave-v1");
    assert!(weave.activation_bookmarks_preserved);
    assert!(!weave.selection_procedure_preserved);
    assert!(weave.hard_admission_outside_format);
    assert!(weave
        .field_loss_list
        .contains(&"candidate_base_and_diagnostic_digests".to_string()));

    assert_eq!(jsonl.format_id, "jsonl-parent-stream-v1");
    assert!(jsonl.selection_procedure_preserved);
    assert!(jsonl.field_loss_list.is_empty());

    assert_eq!(sock.format_id, "socketteer-nested-json-v1");
    assert!(!sock.activation_bookmarks_preserved);
    assert!(!sock.selection_procedure_preserved);
    assert!(sock.field_loss_list.contains(&"bookmark_bits".to_string()));
}
