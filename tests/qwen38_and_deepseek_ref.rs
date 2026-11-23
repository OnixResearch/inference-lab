use inference_lab::deepseek_ref::{
    apply_rope, evaluate_mtp_ablation, matvec, run_default_harness as run_deepseek_ref,
    run_layer_stack_ablation, sample_batch_windows, scheduled_eval_steps, TinyMlaParams,
    TinyMoeParams,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::qwen38_quant::{
    qwen38_27b_kv_planning_budget, run_default_harness as run_qwen38_quant,
    score_retained_attempts, spearman_rank_correlation, AttemptCategory, RetainedTaskAttempt,
};

#[test]
fn qwen38_quant_verifies_kv_budget_offline_rescoring_spearman_and_all_extension_gates() {
    let report = run_qwen38_quant().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );

    // Exact 64 KiB/token, 2 GiB @ 32768, 16 GiB @ 262144 calculated planning budget.
    let kv = qwen38_27b_kv_planning_budget();
    assert_eq!(kv.bytes_per_token_per_sequence, 65_536);
    assert_eq!(kv.bytes_at_32768_tokens, 2_u64 * 1024 * 1024 * 1024);
    assert_eq!(kv.bytes_at_262144_tokens, 16_u64 * 1024 * 1024 * 1024);
    assert!(kv.is_calculated_planning_not_observed_allocation);

    // Matched target passes; changed quant diverges; quality never converts parity failure.
    assert!(report.matched_target_parity_check.exact_match);
    assert!(!report.changed_quant_divergence_retained.exact_match);
    assert!(!report.quality_pass_converts_failed_parity);
    assert!(!report.overlapping_unified_ram_summed);
    assert!(!report.hidden_reasoning_leaked_into_animus_canonical);

    // Offline arm scoring: exact denominators & zero-accepted cost_per_accepted_task == None!
    assert_eq!(report.offline_arm_scores.len(), 2);
    assert!(report.offline_arm_scores[0].denominator_exact);
    assert!(report.offline_arm_scores[0]
        .cost_per_accepted_task
        .is_some());
    assert!(report.offline_arm_scores[1].denominator_exact);
    assert_eq!(report.offline_arm_scores[1].accepted, 0);
    assert!(report.offline_arm_scores[1]
        .cost_per_accepted_task
        .is_none());

    // QDA Spearman KL rank correlation: +1.0 aligned, -1.0 inverted control!
    assert!((report.qda_kl_spearman_aligned - 1.0).abs() < 1e-12);
    assert!((report.qda_kl_spearman_inverted_control + 1.0).abs() < 1e-12);
    assert!(spearman_rank_correlation(&[1.0, 1.0], &[1.0, 2.0]).is_err());

    // All extension arms stay default-off and non-gating (including ttf blocked_on_artifacts).
    assert_eq!(report.extension_arms.len(), 7);
    for arm in &report.extension_arms {
        assert!(!arm.default_enabled);
    }
    let ttf = report
        .extension_arms
        .iter()
        .find(|a| a.arm_id == "ttf-ternary-weights")
        .unwrap();
    assert_eq!(ttf.gate_status, ExecutionClass::BlockedOnArtifacts);

    // Incomplete or over-counted denominator must return acceptance_rate == None (never > 1.0!).
    let overcounted = score_retained_attempts(
        "overcounted",
        "changed_quant_task_quality",
        1,
        &[
            RetainedTaskAttempt {
                task_id: "a".into(),
                category: AttemptCategory::Accepted,
                wall_ms: 10.0,
                resource_cost_units: 1.0,
                prompt_tokens: Some(10),
                output_tokens: Some(5),
                reasoning_tokens: None,
            },
            RetainedTaskAttempt {
                task_id: "b".into(),
                category: AttemptCategory::Accepted,
                wall_ms: 10.0,
                resource_cost_units: 1.0,
                prompt_tokens: Some(10),
                output_tokens: Some(5),
                reasoning_tokens: None,
            },
        ],
    )
    .unwrap();
    assert!(!overcounted.denominator_exact);
    assert!(overcounted.acceptance_rate.is_none());

    // Conformance fixtures include Monoglot positive & misaligned negative.
    let by_id = |id: &str| {
        report
            .conformance_fixtures
            .iter()
            .find(|f| f.fixture_id == id)
            .unwrap()
    };
    assert_eq!(
        by_id("09-positive-monoglot-bit-identical-non-vocab-qualification").outcome,
        OutcomeStatus::Accepted
    );
    assert_eq!(
        by_id("10-negative-monoglot-misaligned-head-rejected-before-scoring").outcome,
        OutcomeStatus::Invalid
    );
    assert_eq!(
        by_id("11-negative-macaq-r-dense-27b-unsupported-see-issue-24").outcome,
        OutcomeStatus::Unsupported
    );
}

#[test]
fn deepseek_ref_reproduces_and_fixes_all_five_upstream_defects_with_exact_eq18_fixture() {
    let report = run_deepseek_ref().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.upstream_license, "MIT");
    assert!(report.checkpoint_roundtrip_exact);
    assert!(!report.checkpoint_includes_optimizer_rng_state);

    // Exact hand-derived Eq. 18 check: corrected = 0.017, buggy upstream = 0.003.
    assert!((report.eq18_hand_derived_expected - 0.017).abs() < 1e-12);
    assert!((report.eq18_corrected_observed - 0.017).abs() < 1e-12);
    assert!((report.eq18_upstream_buggy_observed - 0.003).abs() < 1e-12);

    assert_eq!(report.regressions.len(), 7);
    for r in &report.regressions {
        assert!(r.corrected_pass, "regression {} failed", r.defect_id);
        assert_eq!(r.outcome, OutcomeStatus::Accepted);
    }

    // Direct boundary tests on fixed helpers, real binary checkpoint round-trip, and NaN/theta guards.
    let (main_l, mtp_l, exec) = evaluate_mtp_ablation(&[1, 2, 3, 4], 0, false).unwrap();
    assert!(main_l.is_finite() && mtp_l == 0.0 && !exec);
    assert_eq!(scheduled_eval_steps(10, 3, 7), vec![3, 6, 9]);
    let w = sample_batch_windows(&[5, 6, 7], 2).unwrap();
    assert_eq!(w, vec![(vec![5, 6], vec![6, 7])]);
    assert!(sample_batch_windows(&[5, 6], 2).is_err());

    let p = TinyMlaParams::frozen_tiny();
    let mut raw = p.export_le_bytes();
    assert_eq!(TinyMlaParams::import_le_bytes(&raw).unwrap(), p);
    // Corrupt last f64 weight to NaN -> rejected!
    let n = raw.len();
    raw[n - 8..n].copy_from_slice(&f64::NAN.to_bits().to_le_bytes());
    assert!(TinyMlaParams::import_le_bytes(&raw).is_err());

    // matvec & apply_rope finiteness/theta guards.
    assert!(matvec(&[1.0], &[f64::NAN], 1, 1).is_err());
    assert!(apply_rope(&[1.0, f64::NAN], 0, 10_000.0).is_err());
    assert!(apply_rope(&[1.0, 2.0], 0, 0.0).is_err());

    // Real layer stack ablation: all-dense vs mixed dense+MoE.
    let moe_p = TinyMoeParams::frozen_tiny();
    let (_, aux_dense, moe_cnt_dense) =
        run_layer_stack_ablation(&moe_p, &[vec![1.0, 0.0, 0.0, 0.0]], &[0.0; 4], 2, 2, false)
            .unwrap();
    assert_eq!(aux_dense, 0.0);
    assert_eq!(moe_cnt_dense, 0);
    assert!(
        run_layer_stack_ablation(&moe_p, &[vec![1.0, 0.0, 0.0, 0.0]], &[0.0; 4], 2, 2, true)
            .is_err()
    );
}
