use std::fs;
use std::process::Command;

use inference_lab::biollm_protocol::{
    compute_calibration_metrics, evaluate_falsifiable_bio_criterion,
    run_default_harness as run_biollm, synthesize_matched_signal,
    validate_external_benchmark_assumption, verify_synthetic_match, SignalMoments,
};
use inference_lab::intervention_replay::{
    run_default_harness as run_intervention, run_qwen_transformer_intervention,
    validate_intervention_spec, verify_replay_receipt, ActivationIntervention, InterventionSpec,
    ModelBackendArch,
};
use inference_lab::json::sha256_hex;
use inference_lab::provenance::{DigestRef, ExecutionClass, OutcomeStatus};
use inference_lab::quark_eval::{
    compare_quark_outputs, forward_full_prefix, run_default_harness as run_quark,
    validate_tool_turn, QuarkFaults, QuarkHybridCache, QuarkVariant, ToolTurnRecord,
};
use inference_lab::split_m2_p150::{
    derived_dense_boundary_payload_bytes, run_default_harness as run_split_m2_p150,
    verify_disjoint_moe_split,
};

#[test]
fn split_m2_p150_verifies_stage_reuse_disjoint_moe_payload_math_and_controls() {
    let report = run_split_m2_p150().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.stage_contract_reused_checks >= 4);
    assert!(report.dense_split_max_err <= 1.0e-12);
    assert!(report.moe_disjoint_split_max_err <= 1.0e-12);
    assert_eq!(report.logical_payload_1tok_bytes, 16 * 1024);
    assert_eq!(report.logical_payload_4096tok_bytes, 64 * 1024 * 1024);
    assert_eq!(derived_dense_boundary_payload_bytes(8192, 2, 1), 16384);
    assert!(report.derived_logical_payload_not_measured_throughput);
    assert_eq!(report.profile_verdicts.len(), 4);

    assert!(verify_disjoint_moe_split(&[0.1, 0.2, 0.3, 0.4], &[0, 1], &[1, 2], 3).is_err());

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Split M2+P150 control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn quark_eval_verifies_independent_full_prefix_rollover_tool_guard_and_controls() {
    let report = run_quark().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::BlockedOnArtifacts
    );
    assert!(!report.base_vs_zero_is_numerical_parity_assertion);
    assert_eq!(report.variants.len(), 2);
    for v in &report.variants {
        assert!(v.full_vs_cached_rollover_max_err <= 1.0e-12);
        assert_eq!(v.first_divergent_token_clean, None);
        assert!(v.changed_tool_result_max_err <= 1.0e-12);
        assert_eq!(v.heldout_planned, 5);
        assert_eq!(v.family_breakdown.len(), 5);
    }
    assert_eq!(report.variants[0].heldout_verified_success, 0);
    assert!(report.variants[0].cost_per_verified_success.is_none());
    assert_eq!(report.variants[1].heldout_verified_success, 4);
    assert!(report.variants[1].cost_per_verified_success.is_some());

    // Direct independent full_prefix vs broken rollover diverges precisely at token 4
    let seq = [2u32, 5, 1, 4, 7, 0];
    let full = forward_full_prefix(QuarkVariant::Zero, &seq).unwrap();
    let mut bad = QuarkHybridCache::fresh(QuarkVariant::Zero, 1, "s");
    let out_bad = bad
        .run_tokens(
            &seq,
            QuarkFaults {
                bad_local_window_rollover: true,
                ..QuarkFaults::default()
            },
        )
        .unwrap();
    let (e, div) = compare_quark_outputs(&full, &out_bad, 1.0e-6).unwrap();
    assert!(e > 1.0e-6 && div == Some(4));

    assert_eq!(
        validate_tool_turn(&ToolTurnRecord {
            emitted_tool_call: true,
            tool_actually_executed: false,
            generated_text_resembles_tool_result: true,
            assistant_stop_delimiter_present: true,
        }),
        OutcomeStatus::Invalid
    );

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Quark control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn intervention_replay_verifies_qwen_only_guard_task_outcomes_and_real_replay_receipt() {
    let report = run_intervention().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.paired_runs.len(), 4);
    assert!((report.inject_norm - report.random_control_norm).abs() <= 1.0e-12);
    assert!(!report.recurrent_hook_boundary.neuron_ids_portable_from_qwen);
    assert!(!report.looped_readout_descriptor.executed_in_first_slice);
    for r in &report.paired_runs {
        assert!(r.replay_bit_identical);
        assert!(r.baseline_task_pass);
        assert!(r.kl_vs_baseline.is_finite() && r.kl_vs_baseline >= 0.0);
        assert!(r.l1_vs_baseline.is_finite() && r.l1_vs_baseline >= 0.0);
        assert!(r
            .intervened_logprobs
            .iter()
            .all(|x| x.is_finite() && *x <= 0.0));
    }

    // Qwen backend refuses RecurrentStateful even when source_arch == target_arch!
    let rec_spec = InterventionSpec {
        experiment_id: "rec".into(),
        source_arch: ModelBackendArch::RecurrentStateful,
        target_arch: ModelBackendArch::RecurrentStateful,
        model_revision_digest: "pinned@sha256:1234".into(),
        seed: 1,
        prompt_tokens: vec![1, 2],
        target_layer: 0,
        target_token_pos: 0,
        intervention: ActivationIntervention::Ablate { neuron_idx: 0 },
    };
    assert!(run_qwen_transformer_intervention(&rec_spec).is_err());

    // Cross-architecture transfer rejected directly
    let bad = InterventionSpec {
        source_arch: ModelBackendArch::QwenTransformer,
        ..rec_spec
    };
    assert!(validate_intervention_spec(&bad).is_err());

    // Real replay verifier accepts exact digest and rejects tampered digest
    let valid_spec = InterventionSpec {
        experiment_id: "ok".into(),
        source_arch: ModelBackendArch::QwenTransformer,
        target_arch: ModelBackendArch::QwenTransformer,
        model_revision_digest: "pinned@sha256:1234".into(),
        seed: 7,
        prompt_tokens: vec![1, 2, 3],
        target_layer: 0,
        target_token_pos: 1,
        intervention: ActivationIntervention::Ablate { neuron_idx: 0 },
    };
    let out = run_qwen_transformer_intervention(&valid_spec).unwrap();
    assert_eq!(
        verify_replay_receipt(&valid_spec, &out.trace_digest).unwrap(),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        verify_replay_receipt(&valid_spec, &DigestRef::sha256(b"fake")).unwrap(),
        OutcomeStatus::Invalid
    );

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Intervention replay control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn biollm_protocol_verifies_nan_safe_moments_equal_budget_five_arms_and_controls() {
    let report = run_biollm().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::BlockedOnArtifacts
    );
    assert!(!report.assumes_biollm_8b_benchmarks_exist);
    assert_eq!(report.arms.len(), 5);
    assert_eq!(report.fixture_criterion_verdict, "FALSIFIED_OR_NULL");

    assert!((report.bio_signal_moments.mean - report.synth_signal_moments.mean).abs() <= 1.0e-9);
    assert!(
        (report.bio_signal_moments.variance - report.synth_signal_moments.variance).abs() <= 1.0e-9
    );

    // NaN moment & tolerance regression guards (zero NaN swallowing!)
    assert!(synthesize_matched_signal(
        SignalMoments {
            dim: 16,
            mean: 0.0,
            variance: 1.0,
            lag1_autocorr: f64::NAN,
        },
        1
    )
    .is_err());
    assert!(synthesize_matched_signal(
        SignalMoments {
            dim: 16,
            mean: f64::NAN,
            variance: 1.0,
            lag1_autocorr: 0.2,
        },
        1
    )
    .is_err());
    assert!(verify_synthetic_match(
        report.bio_signal_moments,
        SignalMoments {
            mean: f64::NAN,
            ..report.synth_signal_moments
        },
        1e-6,
        1e-6,
        0.5
    )
    .is_err());
    assert!(verify_synthetic_match(
        report.bio_signal_moments,
        report.synth_signal_moments,
        f64::NAN,
        1e-6,
        0.5
    )
    .is_err());

    // BaseModel zero-budget & NaN threshold guards
    assert!(evaluate_falsifiable_bio_criterion(&report.arms, f64::NAN, 0.85, 0.25).is_err());
    assert!(evaluate_falsifiable_bio_criterion(&report.arms[1..], 0.05, 0.85, 0.25).is_err());
    let mut bad_base = report.arms.clone();
    bad_base[0].training_budget.steps = 10;
    assert!(evaluate_falsifiable_bio_criterion(&bad_base, 0.05, 0.85, 0.25).is_err());

    let (verdict, st) = evaluate_falsifiable_bio_criterion(&report.arms, 0.05, 0.85, 0.25).unwrap();
    assert_eq!(verdict, "FALSIFIED_OR_NULL");
    assert_eq!(st, OutcomeStatus::Invalid);
    assert!(compute_calibration_metrics(&[(1.5, true)]).is_err());
    assert_eq!(
        validate_external_benchmark_assumption(true, false),
        OutcomeStatus::BlockedOnArtifacts
    );

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "BioLLM control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn batch5_suite_cli_is_two_run_byte_identical_refuses_overwrite_and_has_no_home_leaks() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!(
        "inference-lab-batch5-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run1 = base.join("run1");
    let run2 = base.join("run2");

    let out1 = Command::new(bin)
        .args(["batch5-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out1.status.success(),
        "run1 failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let overwrite = Command::new(bin)
        .args(["batch5-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!overwrite.status.success());

    let out2 = Command::new(bin)
        .args(["batch5-suite", "--out", run2.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out2.status.success(),
        "run2 failed: {}",
        String::from_utf8_lossy(&out2.stderr)
    );
    assert_eq!(out1.stdout, out2.stdout);

    let top_files = [
        "inputs.json",
        "results.json",
        "summary.md",
        "manifest.json",
        "COMPLETE",
    ];
    let manifest_text = fs::read_to_string(run1.join("manifest.json")).unwrap();
    assert!(!manifest_text.contains("/home/"));

    for f in top_files {
        let b1 = fs::read(run1.join(f)).unwrap();
        let b2 = fs::read(run2.join(f)).unwrap();
        assert_eq!(b1, b2, "top-level {f} differed across two runs");
        if f == "COMPLETE" {
            assert!(b1.is_empty(), "COMPLETE marker must be empty");
        } else {
            let txt = String::from_utf8_lossy(&b1);
            assert!(!txt.contains("/home/"), "leak in top-level {f}");
            if f != "manifest.json" {
                assert!(
                    manifest_text.contains(&sha256_hex(&b1)),
                    "manifest missing digest for {f}"
                );
            }
        }
    }

    let subdirs = [
        "issue-32-qsa-state-coherence",
        "issue-33-inkling-residuals",
        "issue-34-phoenix-nano",
        "issue-35-nemotron-spark",
        "issue-36-compact-hybrids",
        "issue-37-native-recurrent-bench",
        "issue-38-tiered-eval",
        "issue-39-cpu-contention",
        "issue-42-split-m2-p150",
        "issue-44-quark-eval",
        "issue-45-intervention-replay",
        "issue-46-biollm-protocol",
    ];
    for sub in subdirs {
        for f in ["inputs.json", "results.json", "summary.md", "COMPLETE"] {
            let b1 = fs::read(run1.join(sub).join(f)).unwrap();
            let b2 = fs::read(run2.join(sub).join(f)).unwrap();
            assert_eq!(b1, b2, "{sub}/{f} differed across two runs");
            if f == "COMPLETE" {
                assert!(b1.is_empty());
            } else {
                let txt = String::from_utf8_lossy(&b1);
                assert!(!txt.contains("/home/"), "leak in {sub}/{f}");
                assert!(
                    manifest_text.contains(&sha256_hex(&b1)),
                    "manifest missing digest for {sub}/{f}"
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&base);
}
