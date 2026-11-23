use inference_lab::learned_compiler::run_default_harness as run_learned;
use inference_lab::provenance::{DigestRef, ExecutionClass, OutcomeStatus};
use inference_lab::structured_read::{
    evaluate_extraction_gate, run_default_harness as run_structured, validate_structured_read,
    BackendGateCandidate, NormalizationScope, StructuredReadFixture, StructuredReadOp,
};

#[test]
fn learned_compiler_reports_blocked_on_artifacts_and_preserves_isolation_gate() {
    let report = run_learned();
    assert_eq!(report.execution_class, ExecutionClass::BlockedOnArtifacts);
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::BlockedOnArtifacts
    );
    assert_eq!(report.asset_inventory.len(), 4);
    assert_eq!(report.preserved_baselines.len(), 2);

    // Preserved #30 and #31 historical baseline details.
    assert!(report.preserved_baselines[0]
        .historical_claim
        .contains("27 evaluated outputs"));
    assert!(report.preserved_baselines[0]
        .historical_claim
        .contains("12-byte proved artifact"));
    assert!(report.preserved_baselines[1]
        .historical_claim
        .contains("900s timeout"));
    assert!(report.preserved_baselines[1]
        .historical_claim
        .contains("JAX/Orbax"));

    for b in &report.preserved_baselines {
        assert_eq!(b.current_replay_outcome, OutcomeStatus::BlockedOnArtifacts);
        assert_eq!(b.newly_measured_result, None);
    }

    let by_id = |id: &str| {
        report
            .integration_controls
            .iter()
            .find(|c| c.candidate_id == id)
            .unwrap()
    };

    assert_eq!(
        by_id("synthetic-positive-qwen-apple-container").outcome,
        OutcomeStatus::Accepted
    );
    assert_eq!(
        by_id("synthetic-negative-diffusion-failed-assembly").outcome,
        OutcomeStatus::Invalid
    );
    let linux = by_id("negative-linux-host-no-isolation-weakening");
    assert_eq!(linux.outcome, OutcomeStatus::UnavailableHardwareRuntime);
    assert!(!linux.isolation_weakened);
    assert_eq!(
        by_id("negative-frozen-checker-identity-mismatch").outcome,
        OutcomeStatus::Invalid
    );
    assert_eq!(
        by_id("negative-attempt-budget-exhausted").outcome,
        OutcomeStatus::Exhausted
    );
    assert_eq!(
        by_id("negative-hidden-teacher-compiler-repair").outcome,
        OutcomeStatus::Invalid
    );
    assert_eq!(
        by_id("negative-model-supplies-trusted-expected-output").outcome,
        OutcomeStatus::Invalid
    );
}

#[test]
fn structured_read_enforces_conditional_gate_and_all_adversarial_fixtures() {
    let report = run_structured();
    assert_eq!(report.execution_class, ExecutionClass::BlockedOnArtifacts);
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::BlockedOnArtifacts
    );

    for g in &report.gate_checks {
        assert!(
            !g.gate_satisfied,
            "{} must not satisfy gate",
            g.gate_case_id
        );
    }

    let positive_gate = evaluate_extraction_gate(
        "hypothetical-two-distinct-live-adapters",
        &[
            BackendGateCandidate {
                backend_id: "tt-qwen".into(),
                adapter_implementation_id: "tt-p150-ar-adapter".into(),
                is_live_real_backend: true,
                exposes_structured_read_not_chat_only: true,
            },
            BackendGateCandidate {
                backend_id: "diff-canvas".into(),
                adapter_implementation_id: "diffusion-canvas-adapter".into(),
                is_live_real_backend: true,
                exposes_structured_read_not_chat_only: true,
            },
        ],
        &["noble-1", "proof-2"],
    );
    assert!(positive_gate.gate_satisfied);

    // 3 operation-separated positive fixtures (AR, DiffusionCanvas, ClefJointHead) + 10 negative fixtures.
    assert_eq!(report.fixture_verdicts.len(), 13);
    assert_eq!(report.fixture_verdicts[0].op, StructuredReadOp::ArLabelRead);
    assert_eq!(
        report.fixture_verdicts[1].op,
        StructuredReadOp::DiffusionCanvasRead
    );
    assert_eq!(
        report.fixture_verdicts[2].op,
        StructuredReadOp::ClefJointHeadRead
    );
    for pos in &report.fixture_verdicts[0..3] {
        assert_eq!(pos.outcome, OutcomeStatus::Accepted);
        assert!(!pos.action_authorized_by_confidence);
    }
    for neg in &report.fixture_verdicts[3..] {
        assert_ne!(
            neg.outcome,
            OutcomeStatus::Accepted,
            "{} must be rejected",
            neg.fixture_id
        );
        assert!(!neg.action_authorized_by_confidence);
    }

    // Empty expected_ordered_labels must be rejected cleanly without indexing panic.
    let empty_labels = StructuredReadFixture {
        fixture_id: "empty-labels".into(),
        requested_op: StructuredReadOp::ArLabelRead,
        backend_supported_ops: vec![StructuredReadOp::ArLabelRead],
        expected_ordered_labels: vec![],
        observed_ordered_labels: vec![],
        label_token_counts: vec![],
        probabilities: vec![],
        normalization_scope: NormalizationScope::PermittedLabelsOnly,
        retained_label_mass: None,
        claimed_full_vocab_distribution: false,
        reads_used: 1,
        read_budget: 4,
        expected_calibration_digest: DigestRef::sha256(b"cal"),
        observed_calibration_digest: DigestRef::sha256(b"cal"),
        cancelled_or_late: false,
    };
    let empty_verdict = validate_structured_read(&empty_labels);
    assert_eq!(empty_verdict.outcome, OutcomeStatus::Invalid);
}
