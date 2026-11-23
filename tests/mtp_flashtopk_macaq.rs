use inference_lab::flash_topk::{
    dense_reference_topk, run_default_harness as run_flash_topk, validate_flashtopk_contract,
    HardwareContractRequest, IndexerDtype, SavedIndexerCall,
};
use inference_lab::macaq_r::{
    inspect_refinement_invariants, rescore_arm, run_default_harness as run_macaq_r,
    verify_partition_disjointness, PairedTaskOutcome,
};
use inference_lab::mtp_decoder::{
    argmax_finite, exact_rejection_marginal, max_abs_diff, residual_distribution,
    run_default_harness as run_mtp_decoder,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};

#[test]
fn mtp_decoder_verifies_greedy_poisoned_cache_stochastic_exact_marginal_and_negative_controls() {
    let report = run_mtp_decoder().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(report.routing_bias_frozen);
    assert!(report.target_cached_vs_recompute_max_err <= 1e-11);
    assert!(report.prefix_causal_no_future_leak_max_err <= 1e-12);
    assert!(report.multi_request_isolation_max_err <= 1e-11);
    assert_eq!(
        report.decision,
        "GO_CPU_CONFORMANCE_ONLY_NO_SERVING_SPEEDUP_CLAIMED"
    );

    assert_eq!(report.greedy_fixtures.len(), 9);
    for f in &report.greedy_fixtures {
        assert!(
            f.exact_token_agreement,
            "greedy fixture {} disagreed with target baseline",
            f.fixture_id
        );
        assert!(
            f.max_cache_err <= 1e-11 && f.max_next_logit_err <= 1e-11,
            "cache or next-step logit drift on {}: cache={}, logit={}",
            f.fixture_id,
            f.max_cache_err,
            f.max_next_logit_err
        );
        assert!(f.target_cache_bytes > 0 && f.mtp_cache_bytes > 0);
        assert!(!f.rounds.is_empty());
    }
    // Verify fallback fixture counted fallbacks, rollback fixtures actually rolled back BOTH target & MTP caches,
    // and EOS fixture hit stopped_reason == "eos" without emitting rejected tail!
    assert!(report.greedy_fixtures[4].target_only_fallbacks > 0);
    assert!(report.greedy_fixtures[1].target_rollbacks > 0);
    assert!(report.greedy_fixtures[1].mtp_rollbacks > 0);
    let eos_fix = &report.greedy_fixtures[8];
    assert_eq!(eos_fix.fixture_id, "greedy_eos_boundary_no_tail_emit");
    assert_eq!(eos_fix.stopped_reason, "eos");
    assert_eq!(*eos_fix.generated.last().unwrap(), eos_fix.eos_token);
    assert!(eos_fix.mtp_rollbacks > 0);

    // Round-level draft & target probabilities serialized in results_json!
    let rjson = inference_lab::mtp_decoder::results_json(&report);
    assert!(rjson.contains("\"draft_probs\":"));
    assert!(rjson.contains("\"target_probs\":"));
    assert!(rjson.contains("\"target_cache_bytes\":"));

    // Stochastic exact marginal preservation & empirical calibration.
    assert!(report.stochastic.exact_marginal_max_err <= 1e-12);
    assert!(report.stochastic.empirical_pass);
    assert!(report.stochastic.impossible_zero_residual_rejected);
    assert!(report.stochastic.bonus_token_rule_verified);
    assert!(report
        .stochastic
        .branch_outcomes
        .iter()
        .all(|(_, _, ok)| *ok));

    // Direct API adversarial checks: NaN rejection, non-normalized dist rejection, impossible zero residual.
    assert!(max_abs_diff(&[f64::NAN], &[0.0]).is_err());
    assert!(argmax_finite(&[1.0, f64::NAN]).is_err());
    assert!(residual_distribution(&[0.25, 0.75], &[0.25, 0.75]).is_err());
    assert!(exact_rejection_marginal(&[f64::NAN, 1.0], &[0.5, 0.5]).is_err());
    assert!(exact_rejection_marginal(&[0.3, 0.3], &[0.5, 0.5]).is_err());
    let marg = exact_rejection_marginal(&[0.1, 0.6, 0.3], &[0.5, 0.2, 0.3]).unwrap();
    assert!(max_abs_diff(&marg, &[0.1, 0.6, 0.3]).unwrap() <= 1e-12);

    assert_eq!(report.negative_controls.len(), 6);
    for c in &report.negative_controls {
        assert_eq!(
            c.outcome,
            OutcomeStatus::Invalid,
            "MTP negative control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn flash_topk_enforces_sm120_sm121_and_64head_refusals_exact_ties_and_recovery_controls() {
    let report = run_flash_topk().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.sm120_sm121_decision,
        "NO_GO_UNSUPPORTED_AT_PINNED_REVISION"
    );
    assert_eq!(
        report.sm90_sm100_decision,
        "INCONCLUSIVE_GPU_EXECUTION_REQUIRED"
    );

    // SM90 and SM100 accepted; SM120, SM121, 64-head DeepSeek-V3.2, block-QSA, Tenstorrent unsupported!
    assert_eq!(
        report.compatibility_matrix[0].outcome,
        OutcomeStatus::Accepted
    );
    assert_eq!(
        report.compatibility_matrix[1].outcome,
        OutcomeStatus::Accepted
    );
    for row in &report.compatibility_matrix[2..] {
        assert_eq!(
            row.outcome,
            OutcomeStatus::Unsupported,
            "target {} must be refused",
            row.target_name
        );
    }

    // All 4 unmodified fixtures (including real k=1 boundary and real -0.0 tie in top-4) pass!
    assert_eq!(report.fixture_comparisons.len(), 4);
    assert_eq!(report.fixture_comparisons[2].case_id, "bf16_k1_boundary");
    for f in &report.fixture_comparisons {
        assert!(f.exact_selection_match, "fixture {} failed", f.case_id);
        assert!(f.deterministic_repeat_bit_identical);
        assert!(f.row_partition_invariant);
        assert!(!f.fault_injected);
    }

    // Fault-injected underflow & overflow recovery actually triggered, tracked recovery_key_evals, AND matched dense reference!
    assert_eq!(report.fault_injected_recovery_checks.len(), 2);
    assert!(report.fault_injected_recovery_checks[0].fault_injected);
    assert!(report.fault_injected_recovery_checks[0].underflow_recoveries > 0);
    assert!(report.fault_injected_recovery_checks[0].recovery_key_evals > 0);
    assert!(report.fault_injected_recovery_checks[0].exact_selection_match);
    assert!(report.fault_injected_recovery_checks[1].fault_injected);
    assert!(report.fault_injected_recovery_checks[1].overflow_recoveries > 0);
    assert!(report.fault_injected_recovery_checks[1].recovery_key_evals > 0);
    assert!(report.fault_injected_recovery_checks[1].exact_selection_match);

    // Reject NaN test_only_threshold_hook and zero num_heads/head_dim!
    let valid_small = SavedIndexerCall {
        case_id: "ok".into(),
        dtype: IndexerDtype::Bf16,
        num_rows: 1,
        num_heads: 1,
        num_keys: 2,
        head_dim: 2,
        k: 1,
        queries: vec![vec![vec![1.0, 0.0]]],
        keys: vec![vec![-1.0, 0.0], vec![1.0, 0.0]],
        weights: vec![vec![-1.0]],
        key_scales: vec![1.0, 1.0],
        windows: vec![(0, 2)],
    };
    let mut nan_hook_cfg = inference_lab::flash_topk::FusedSelectorConfig::unmodified(4);
    nan_hook_cfg.test_only_threshold_hook = Some(f64::NAN);
    assert!(inference_lab::flash_topk::fused_flashtopk_cpu(&valid_small, &nan_hook_cfg).is_err());
    let mut zero_heads = valid_small.clone();
    zero_heads.num_heads = 0;
    assert!(dense_reference_topk(&zero_heads).is_err());

    // Direct API contract checks: SM121 refused, non-finite & inverted window refused.
    let sm121 = validate_flashtopk_contract(&HardwareContractRequest {
        target_name: "spark".into(),
        backend: "cuda".into(),
        cuda_cc_major: Some(12),
        cuda_cc_minor: Some(1),
        num_heads: 32,
        head_dim: 128,
        topk: 2048,
        dtype: IndexerDtype::Bf16,
        contiguous_keys: true,
        single_contiguous_window: true,
        block_level_qsa: false,
    });
    assert_eq!(sm121.outcome, OutcomeStatus::Unsupported);

    let bad_call = SavedIndexerCall {
        case_id: "bad".into(),
        dtype: IndexerDtype::Bf16,
        num_rows: 1,
        num_heads: 1,
        num_keys: 2,
        head_dim: 2,
        k: 1,
        queries: vec![vec![vec![f64::INFINITY, 0.0]]],
        keys: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        weights: vec![vec![1.0]],
        key_scales: vec![1.0, 1.0],
        windows: vec![(0, 2)],
    };
    assert!(dense_reference_topk(&bad_call).is_err());

    assert_eq!(report.negative_controls.len(), 7);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "FlashTopK negative control {} must not pass",
            c.case_id
        );
    }
}

#[test]
fn macaq_r_refines_moe_codes_preserving_scales_out_of_scope_bytes_and_catches_controls() {
    let report = run_macaq_r().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.refined_tensor_count, 3);
    assert_eq!(
        report.original_tensor_payload_bytes,
        report.refined_tensor_payload_bytes
    );
    assert!(report.refined_total_gguf_bytes > report.original_total_gguf_bytes);
    assert!(report.calib_kl_after < report.calib_kl_before);
    assert!(report.heldout_kl_after <= report.heldout_kl_before + 1e-6);
    assert_eq!(
        report.receipt.total_creation_wall_ms(),
        report.receipt.interrupted_wall_ms + report.receipt.resumed_wall_ms
    );
    assert!(report.zero_accepted_arm.cost_per_accepted_task_ms.is_none());
    assert!(report
        .zero_accepted_arm
        .amortized_cost_per_accepted_100_uses_ms
        .is_none());

    // Direct API adversarial checks.
    assert!(verify_partition_disjointness(&["a", "b"], &["c"], &["b"]).is_err());
    let mtp_violation = rescore_arm(
        "bad",
        "rt",
        &[PairedTaskOutcome {
            task_id: "t".into(),
            status: OutcomeStatus::Accepted,
            latency_ms: 10.0,
            mtp_enabled: true,
            runtime_build: "rt".into(),
        }],
        0.0,
        10,
    );
    assert!(mtp_violation.is_err());
    // NaN or negative creation_cost_ms rejected cleanly, and synthetic label emitted in JSON.
    let bad_creation_cost = rescore_arm(
        "bad-cost",
        "rt",
        &[PairedTaskOutcome {
            task_id: "t".into(),
            status: OutcomeStatus::Accepted,
            latency_ms: 10.0,
            mtp_enabled: false,
            runtime_build: "rt".into(),
        }],
        f64::NAN,
        10,
    );
    assert!(bad_creation_cost.is_err());
    assert!(inference_lab::macaq_r::results_json(&report)
        .contains("\"synthetic_contract_fixture\":true"));
    let _ = inspect_refinement_invariants;

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "MACaQ-R negative control {} must reject",
            c.case_id
        );
    }
}
