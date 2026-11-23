use inference_lab::adaptive_spec::{
    choose_verify_length, run_default_harness as run_adaptive, SpecVerifyPolicy,
};
use inference_lab::drafting::{
    check_greedy_token_parity, evaluate_drafting_arm, run_default_harness as run_drafting,
    verify_draft_block, verify_stochastic_rejection_sampling, DraftingArmCondition,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};

#[test]
fn drafting_exact_rejection_sampling_restores_target_and_broken_sampler_fails() {
    let report = run_drafting().expect("drafting harness succeeds");
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );

    // Offline retained-output recheck reproduces recorded verdicts without model calls.
    assert!(report.offline_recheck.rechecked_matches_recorded);
    assert_eq!(report.offline_recheck.retained_tasks, 2);

    // Exact greedy match and mismatch retention.
    assert!(report.greedy_checks[0].exact_match);
    assert!(!report.greedy_checks[1].exact_match);
    assert_eq!(report.greedy_checks[1].first_mismatch_index, Some(2));
    assert_eq!(
        report.greedy_checks[1].first_mismatch_target_token,
        Some(305)
    );
    assert_eq!(
        report.greedy_checks[1].first_mismatch_draft_token,
        Some(999)
    );

    // Analytical stochastic rejection sampling parity: exact <= 1e-12 vs broken > 1e-12.
    let exact = &report.stochastic_checks[0];
    assert!(exact.restores_target_distribution);
    assert!(exact.max_marginal_abs_error.unwrap() <= 1e-12);

    let broken = &report.stochastic_checks[1];
    assert!(!broken.restores_target_distribution);
    assert!(broken.max_marginal_abs_error.unwrap() > 0.01);

    // Unnormalized and NaN distributions rejected.
    assert!(!report.stochastic_checks[2].distributions_valid);
    assert!(!report.stochastic_checks[3].distributions_valid);

    // Both single-stream (c1) and concurrent (c4) have valid baseline + draft-assisted pairs,
    // with derived metrics populated and unmeasured memory/energy explicitly None.
    let c4_base = report
        .arm_conditions
        .iter()
        .find(|a| a.arm_id == "no-draft-baseline-c4")
        .unwrap();
    let c4_draft = report
        .arm_conditions
        .iter()
        .find(|a| a.arm_id == "draft-assisted-contract-c4")
        .unwrap();
    assert_eq!(c4_base.outcome, OutcomeStatus::Accepted);
    assert_eq!(c4_draft.outcome, OutcomeStatus::Accepted);
    assert!(c4_draft.mean_accepted_tokens_per_verify.unwrap() > 1.5);
    assert!(c4_draft.drafter_time_share.unwrap() > 0.0);
    assert_eq!(c4_draft.peak_resident_memory_mib, None);
    assert_eq!(c4_draft.measured_energy_j, None);

    // Silent fallback and fast-but-zero-accepted-tasks disqualified.
    let silent = report
        .arm_conditions
        .iter()
        .find(|a| a.arm_id == "negative-silent-fallback-arm")
        .unwrap();
    assert_eq!(silent.outcome, OutcomeStatus::Invalid);
    assert!(silent
        .disqualification_reason
        .as_deref()
        .unwrap()
        .contains("silent_fallback"));

    // Invalid fallback > verify_passes or negative latency rejected.
    let mut bad_arm = c4_draft.clone();
    bad_arm.fallback_to_no_draft_passes = bad_arm.total_verify_passes + 1;
    assert_eq!(
        evaluate_drafting_arm(bad_arm).outcome,
        OutcomeStatus::Invalid
    );

    let mut neg_lat = DraftingArmCondition {
        draft_time_ms: -1.0,
        ..c4_draft.clone()
    };
    neg_lat.fallback_to_no_draft_passes = 0;
    assert_eq!(
        evaluate_drafting_arm(neg_lat).outcome,
        OutcomeStatus::Invalid
    );

    // External claims (DFlash2 6.6x, DNeo) remain strictly non-gating & unverified.
    for claim in &report.external_claims {
        assert!(!claim.verified_locally);
        assert!(!claim.gating_target);
    }

    // Extra asymmetric distribution check.
    let asym = verify_stochastic_rejection_sampling(
        "asym",
        &[0.7, 0.2, 0.08, 0.02],
        &[0.1, 0.5, 0.3, 0.1],
        true,
    );
    assert!(asym.restores_target_distribution);
    assert!(asym.max_marginal_abs_error.unwrap() <= 1e-12);

    let len_diff = check_greedy_token_parity("len", &[1, 2], &[1, 2, 3], &[1.0, 1.0, 0.5], false);
    assert!(!len_diff.exact_match);
    assert_eq!(len_diff.first_mismatch_index, Some(2));
}

#[test]
fn adaptive_spec_reuses_drafting_core_and_rejects_mislabeled_batching_and_sm100a() {
    let report = run_adaptive().expect("adaptive spec harness succeeds");
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.reused_drafting_controls.len(), 3);

    // Reused #3/#22 rollback checks.
    assert_eq!(report.reused_drafting_controls[0].accepted_draft_tokens, 0);
    assert_eq!(report.reused_drafting_controls[0].kv_rollback_len, 4);
    assert!(report.reused_drafting_controls[0].correction_token_emitted);
    assert!(report.reused_drafting_controls[2].bonus_token_emitted);
    assert_eq!(report.reused_drafting_controls[2].kv_rollback_len, 0);

    // Adaptive threshold shrinks block length when tail confidence drops.
    let policy = SpecVerifyPolicy::AdaptiveFrozenThreshold {
        min_len: 1,
        max_len: 6,
        confidence_threshold_milli: 800,
    };
    assert_eq!(choose_verify_length(policy, &[950, 900, 400, 300]), 2);
    assert_eq!(choose_verify_length(policy, &[200, 100]), 1);

    let mislabeled = report
        .conditions
        .iter()
        .find(|c| c.arm_id == "negative-cpu-overlap-mislabeled-continuous-batching")
        .unwrap();
    assert_eq!(mislabeled.outcome, OutcomeStatus::Invalid);

    let sm100a = report
        .conditions
        .iter()
        .find(|c| c.arm_id == "negative-sm100a-b300-extrapolation")
        .unwrap();
    assert_eq!(sm100a.outcome, OutcomeStatus::Unsupported);

    // Empty draft block rejected.
    assert!(verify_draft_block("empty", &[], &[], 1).is_err());
}
