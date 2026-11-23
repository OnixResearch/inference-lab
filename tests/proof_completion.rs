use inference_lab::proof_completion::{
    evaluate_proof_candidate, run_default_harness, validate_family_partitions, FreshCheckerOutcome,
    IntakePath, ProofCandidate, ProofCheckerFamily, ProofObligation, ProofPartition, ProofVerdict,
};
use inference_lab::provenance::{DigestRef, ExecutionClass};

#[test]
fn proof_completion_enforces_frozen_context_partition_and_intake_equivalence() {
    let report = run_default_harness();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(report.clean_partition_leaks.is_empty());
    assert_eq!(report.leaked_partition_negative_control.len(), 1);
    assert!(report.intake_equivalence_verified);
    assert!(report.offline_replay_identical);

    let by_id = |id: &str| {
        report
            .admissions
            .iter()
            .find(|a| a.candidate_id == id)
            .unwrap()
    };

    assert_eq!(by_id("cand-01-valid-proof").verdict, ProofVerdict::Proved);
    assert_eq!(
        by_id("cand-02-sorry-hole-high-conf").verdict,
        ProofVerdict::Rejected
    );
    assert_eq!(
        by_id("cand-03-exec-body-modified").verdict,
        ProofVerdict::Rejected
    );
    assert!(!by_id("cand-03-exec-body-modified").frozen_context_preserved);
    assert_eq!(
        by_id("cand-04-weakened-contract").verdict,
        ProofVerdict::Rejected
    );
    assert_eq!(
        by_id("cand-05-disallowed-import").verdict,
        ProofVerdict::Rejected
    );
    assert_eq!(
        by_id("cand-06-placeholder-leansearch").verdict,
        ProofVerdict::Rejected
    );
    assert_eq!(
        by_id("cand-07-mixed-verus-cohort").verdict,
        ProofVerdict::Unsupported
    );
    assert_eq!(
        by_id("cand-08-davinci-unmatched-python").verdict,
        ProofVerdict::Unsupported
    );
    assert_eq!(
        by_id("cand-09-over-repair-budget").verdict,
        ProofVerdict::Rejected
    );
    assert_eq!(
        by_id("cand-10-checker-timeout").verdict,
        ProofVerdict::Timeout
    );
    assert_eq!(
        by_id("cand-11-tested-not-proved").verdict,
        ProofVerdict::Tested
    );
    assert_eq!(
        by_id("cand-12-checker-unknown").verdict,
        ProofVerdict::Unknown
    );
    assert_eq!(
        by_id("cand-13-checker-infra-failure").verdict,
        ProofVerdict::InternalFailure
    );
    assert_eq!(
        by_id("cand-14-stale-profile-identity").verdict,
        ProofVerdict::Rejected
    );
    assert_eq!(
        by_id("cand-15-frozen-span-modified").verdict,
        ProofVerdict::Rejected
    );

    // Exact denominator conservation across every arm AND exact sum across all admissions!
    let total_arm_attempts: usize = report
        .arm_budgets
        .iter()
        .map(|a| a.attempted_candidates)
        .sum();
    assert_eq!(total_arm_attempts, report.admissions.len());
    for arm in &report.arm_budgets {
        assert_eq!(arm.accounted_candidates(), arm.attempted_candidates);
    }
}

#[test]
fn overlapping_prefix_suffix_does_not_panic_and_producer_status_cannot_override() {
    let profile = DigestRef::sha256(b"p");
    let exec = DigestRef::sha256(b"e");
    let stmt = DigestRef::sha256(b"s");
    let ob = ProofObligation {
        task_id: "t1".into(),
        theorem_family: "fam1".into(),
        partition: ProofPartition::HeldOut,
        checker_family: ProofCheckerFamily::Lean4Pinned,
        profile_digest: profile.clone(),
        allowed_imports: vec!["Init".into()],
        frozen_prefix: "ABC".into(),
        editable_proof_body: "mask".into(),
        frozen_suffix: "BCD".into(),
        executable_body_digest: exec.clone(),
        contract_statement_digest: stmt.clone(),
        max_total_admissions: 4,
    };

    // Submitted source "ABCD" starts with "ABC" and ends with "BCD", but len(4) < len(3)+len(3): must NOT panic!
    let overlapping = ProofCandidate {
        candidate_id: "overlap-no-panic".into(),
        task_id: "t1".into(),
        arm_id: "ar".into(),
        checker_family: ProofCheckerFamily::Lean4Pinned,
        profile_digest: profile.clone(),
        imports: vec!["Init".into()],
        submitted_source: "ABCD".into(),
        executable_body_digest: exec.clone(),
        contract_statement_digest: stmt.clone(),
        attempt_index_including_repairs: 0,
        is_repair_attempt: false,
        producer_claims_completed: true,
        producer_confidence: 1.0,
        repl_empty_goals_observed: true,
        optional_python_tool_used: false,
        retrieval_mode: "retrieval_disabled".into(),
        model_and_retrieval_cost_units: 10,
        checker_cost_units: 5,
        checker_outcome: FreshCheckerOutcome::Proved,
    };
    let overlap_res =
        evaluate_proof_candidate(&ob, &overlapping, IntakePath::GeneratorNeutral75, true);
    assert_eq!(overlap_res.verdict, ProofVerdict::Rejected);
    assert_eq!(
        overlap_res.rejection_reason.as_deref(),
        Some("frozen_context_span_modified")
    );

    // Overconfident wrong proof rejected identically by generator-neutral and temporary saved-result adapter.
    let mut wrong = overlapping;
    wrong.candidate_id = "overconfident-wrong-proof".into();
    wrong.submitted_source = "ABCwrongBCD".into();
    wrong.checker_outcome = FreshCheckerOutcome::Rejected;

    let neutral = evaluate_proof_candidate(&ob, &wrong, IntakePath::GeneratorNeutral75, true);
    let temp = evaluate_proof_candidate(&ob, &wrong, IntakePath::TemporarySavedResultAdapter, true);
    assert_eq!(neutral.verdict, ProofVerdict::Rejected);
    assert_eq!(neutral.verdict, temp.verdict);
    assert_eq!(neutral.rejection_reason, temp.rejection_reason);
    assert!(validate_family_partitions(&[ob]).is_empty());
}
