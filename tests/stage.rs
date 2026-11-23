use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::stage::{
    frozen_dense_model, make_binding, run_default_harness, run_reset_reuse_sequence, BackendKind,
    StageAssignment,
};

#[test]
fn stage_reset_actually_clears_populated_kv_cache_before_gen2() {
    let weights = frozen_dense_model();
    let split2 = make_binding(
        &weights,
        vec![
            StageAssignment {
                stage_id: "stage-a-0-2".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 0,
                end_layer: 2,
                owns_embedding: true,
                owns_final_norm_and_logits: false,
                owns_sampler: false,
            },
            StageAssignment {
                stage_id: "stage-b-2-4".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 2,
                end_layer: 4,
                owns_embedding: false,
                owns_final_norm_and_logits: true,
                owns_sampler: true,
            },
        ],
    );
    let outcome =
        run_reset_reuse_sequence(&weights, &split2, &[5, 13, 1, 8], &[2, 7, 11, 4], &[9, 3])
            .expect("reset/reuse succeeds");
    assert_eq!(
        outcome.populated_kv_len, 4,
        "gen 1 must populate 4 KV entries"
    );
    assert_eq!(
        outcome.cleared_kv_len, 0,
        "reset(2) must clear populated KV entries"
    );
    assert_eq!(outcome.sampled_tokens.len(), 3);
}

#[test]
fn stage_split_placements_match_unsplit_reference_within_tolerance() {
    let report = run_default_harness().expect("stage harness executes cleanly");
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.hardware_gates.len(), 2);
    for gate in &report.hardware_gates {
        assert_eq!(gate.status, ExecutionClass::UnavailableHardwareRuntime);
    }

    // 4 parity cases: 2-stage full, 2-stage chunked 2+2, 3-stage full, 2-stage reset/reuse gen 2.
    assert_eq!(report.parity_checks.len(), 4);
    for p in &report.parity_checks {
        assert!(
            p.passed,
            "parity check {} ({}) failed: act_err={}, logit_err={}",
            p.placement_name, p.mode, p.max_boundary_abs_error, p.max_logit_abs_error
        );
        assert!(p.max_boundary_abs_error <= 1e-12);
        assert!(p.max_logit_abs_error <= 1e-12);
        assert!(p.matches_reference_tokens);
        assert_eq!(p.sampled_tokens, report.reference_sampled_tokens);
    }
}

#[test]
fn stage_all_negative_controls_reject_and_block_silent_cpu_fallback() {
    let report = run_default_harness().unwrap();
    assert_eq!(report.negative_controls.len(), 11);

    for neg in &report.negative_controls {
        assert!(
            neg.rejected,
            "negative control {} must be rejected",
            neg.control_name
        );
    }

    let layout_neg = report
        .negative_controls
        .iter()
        .find(|n| n.control_name == "neg-incompatible-stride-layout")
        .unwrap();
    assert_eq!(layout_neg.status, OutcomeStatus::Invalid);
    assert_eq!(layout_neg.reason, "incompatible_stride_layout");

    let digest_neg = report
        .negative_controls
        .iter()
        .find(|n| n.control_name == "neg-payload-digest-integrity-mismatch")
        .unwrap();
    assert_eq!(digest_neg.status, OutcomeStatus::Invalid);
    assert_eq!(digest_neg.reason, "payload_digest_integrity_mismatch");

    let hw_neg = report
        .negative_controls
        .iter()
        .find(|n| n.control_name == "neg-no-silent-cpu-fallback-for-apple-tt-hardware")
        .unwrap();
    assert_eq!(hw_neg.status, OutcomeStatus::UnavailableHardwareRuntime);
    assert!(hw_neg.reason.contains("no_silent_cpu_fallback"));

    let poisoned = report
        .negative_controls
        .iter()
        .find(|n| n.control_name == "neg-timeout-poisoned-buffer-and-stale-kv")
        .unwrap();
    assert_eq!(poisoned.status, OutcomeStatus::Invalid);
    assert!(poisoned.reason.contains("poisoned_buffer"));
}
