use std::fs;
use std::process::Command;

use inference_lab::disagg_kv_transfer::{
    compute_transfer_latency, qualify_disagg_hosts, run_default_harness as run_disagg_kv,
    PARITY_TOL as DISAGG_TOL,
};
use inference_lab::drafter_opd_replay::{
    gate_replay_verdict, repeat_mean_and_std, run_default_harness as run_drafter_opd, VerdictClass,
    DRIFT_TOL as OPD_DRIFT_TOL,
};
use inference_lab::hypertune_kv_adapters::{
    evaluate_concurrency_scaling, run_default_harness as run_hypertune,
    PARITY_TOL as HYPERTUNE_TOL, UPSTREAM_COMMIT as HYPERTUNE_COMMIT,
};
use inference_lab::json::sha256_hex;
use inference_lab::k2_uno_mova::{run_default_harness as run_k2_uno, softplus_safe, K2_PIN_STATUS};
use inference_lab::lfm3_d1_decision::{
    evaluate_typed_decisions, qualify_lfm3_adapter, run_default_harness as run_lfm3_d1,
    DecisionLabel, TypedDecisionPrediction, LFM3_PIN_STATUS, PARITY_TOL as LFM3_TOL,
};
use inference_lab::linquant_recurrent::{
    run_default_harness as run_linquant, CROSS_REPO_TT_ONIX_ISSUE, FP64_EQUIV_TOL,
    LINQUANT_PIN_STATUS,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::qwen38_longbook_gdn::{
    run_default_harness as run_longbook_gdn, verify_matched_serving, ServingMatchConfig,
    LONGBOOK_COMMIT,
};
use inference_lab::spark_copydrafter::{
    run_default_harness as run_copydrafter, STOCHASTIC_TOL as COPY_STOCH_TOL,
    UPSTREAM_COMMIT as DS41F_COMMIT,
};
use inference_lab::spark_kv_checkpoint::{
    run_default_harness as run_kv_ckpt, PARITY_TOL as CKPT_TOL, UPSTREAM_COMMIT as GLM_SPARK_COMMIT,
};
use inference_lab::spark_rdma_tp::{
    quantize_dequantize_4bit, run_default_harness as run_rdma_tp, validate_rlx_pin, RLX_PIN_STATUS,
    TP_PARITY_TOL,
};
use inference_lab::ttf_ternary_qwen::{
    pack_trits_5_per_byte, run_default_harness as run_ttf_ternary, unpack_trits_5_per_byte,
    PARITY_TOL as TERNARY_TOL, UPSTREAM_COMMIT as TTF_COMMIT,
};
use inference_lab::varloop_budget_protocol::{
    qualify_2dsd_host, run_default_harness as run_varloop, TwoDsdHostSpec, TWO_DSD_INKLING_BRANCH,
};

#[test]
fn batch7_issues_62_through_67_contracts_and_negative_controls() {
    // #62 disagg-kv-transfer
    let r62 = run_disagg_kv().unwrap();
    assert_eq!(r62.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r62.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r62.default_enabled);
    assert!(r62.synthetic_contract_fixture);
    for p in &r62.prompt_parity {
        assert!(p.token_exact_match);
        assert!(p.max_logit_abs_err <= DISAGG_TOL);
    }
    assert_eq!(
        qualify_disagg_hosts(false, false, 10.0, true),
        OutcomeStatus::Invalid
    );
    assert_eq!(
        qualify_disagg_hosts(false, true, 10.0, false),
        OutcomeStatus::UnavailableHardwareRuntime
    );
    let bad_lat = compute_transfer_latency("z", 1024, 10240, 0.0, 10.0, 2.0, 0.9, 1.0).unwrap();
    assert!(bad_lat.disagg_wire_transfer_s.is_none());
    assert_eq!(r62.negative_controls.len(), 8);
    for c in &r62.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#62 {}", c.case_id);
    }

    // #63 spark-rdma-tp
    let r63 = run_rdma_tp().unwrap();
    assert_eq!(r63.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r63.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r63.default_enabled);
    assert_eq!(r63.pin_status, RLX_PIN_STATUS);
    assert_eq!(
        validate_rlx_pin(RLX_PIN_STATUS, None),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        validate_rlx_pin(
            "commit_pinned",
            Some("1111111111111111111111111111111111111111")
        ),
        OutcomeStatus::Invalid
    );
    assert!(r63.fp64_tp2_max_abs_err <= TP_PARITY_TOL);
    assert!(quantize_dequantize_4bit(&[0.1, f64::NAN]).is_err());
    assert_eq!(r63.topology_matrix.len(), 9);
    assert_eq!(r63.negative_controls.len(), 8);
    for c in &r63.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#63 {}", c.case_id);
    }

    // #64 spark-kv-checkpoint
    let r64 = run_kv_ckpt().unwrap();
    assert_eq!(r64.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r64.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r64.default_enabled);
    assert_eq!(r64.upstream_commit, GLM_SPARK_COMMIT);
    assert_eq!(r64.upstream_commit.len(), 40);
    for p in &r64.resume_parity {
        assert!(p.token_exact_match);
        assert!(p.max_logit_abs_err <= CKPT_TOL);
    }
    assert_eq!(r64.negative_controls.len(), 8);
    for c in &r64.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#64 {}", c.case_id);
    }

    // #65 spark-copydrafter
    let r65 = run_copydrafter().unwrap();
    assert_eq!(r65.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r65.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r65.default_enabled);
    assert_eq!(r65.upstream_commit, DS41F_COMMIT);
    for p in &r65.prompt_checks {
        assert!(p.greedy_exact_match);
        assert!(p.stochastic_max_l1_err <= COPY_STOCH_TOL);
    }
    assert_eq!(r65.negative_controls.len(), 8);
    for c in &r65.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#65 {}", c.case_id);
    }

    // #66 hypertune-kv-adapters
    let r66 = run_hypertune().unwrap();
    assert_eq!(r66.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r66.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r66.default_enabled);
    assert_eq!(r66.upstream_commit, HYPERTUNE_COMMIT);
    assert!(r66.paged_vs_unpaged_max_abs_err <= HYPERTUNE_TOL);
    let oom66 = evaluate_concurrency_scaling(100, 16, 2048, 65_536, 48.0, 200.0, 1024.0).unwrap();
    assert_eq!(oom66.outcome, OutcomeStatus::Oom);
    assert_eq!(r66.negative_controls.len(), 8);
    for c in &r66.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#66 {}", c.case_id);
    }

    // #67 qwen38-longbook-gdn
    let r67 = run_longbook_gdn().unwrap();
    assert_eq!(r67.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r67.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r67.default_enabled);
    assert_eq!(r67.model_commit, LONGBOOK_COMMIT);
    assert!(r67.mini_all_gdn_state_elems < r67.mini_hybrid_kv_elems);
    assert!(r67.mini_all_gdn_recall_mae > r67.mini_hybrid_recall_mae);
    let cfg_a = ServingMatchConfig {
        runtime: "vllm".into(),
        quantization: "bf16".into(),
        tokenizer_digest: "tok1".into(),
        max_context: 262_144,
    };
    let mut cfg_b = cfg_a.clone();
    cfg_b.quantization = "q4".into();
    assert_eq!(
        verify_matched_serving(&cfg_a, &cfg_b),
        OutcomeStatus::Invalid
    );
    assert_eq!(r67.negative_controls.len(), 8);
    for c in &r67.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#67 {}", c.case_id);
    }
}

#[test]
fn batch7_issues_68_through_73_contracts_and_negative_controls() {
    // #68 linquant-recurrent
    let r68 = run_linquant().unwrap();
    assert_eq!(r68.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r68.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r68.default_enabled);
    assert!(!r68.grants_blackhole_acceptance);
    assert_eq!(r68.pin_status, LINQUANT_PIN_STATUS);
    assert_eq!(r68.cross_repo_tt_onix_issue, CROSS_REPO_TT_ONIX_ISSUE);
    assert!(r68.gdn_fp64_equiv_max_err <= FP64_EQUIV_TOL);
    assert!(r68.mamba2_fp64_equiv_max_err <= FP64_EQUIV_TOL);
    assert_eq!(r68.precision_comparisons.len(), 6);
    assert_eq!(r68.negative_controls.len(), 8);
    for c in &r68.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#68 {}", c.case_id);
    }

    // #69 varloop-budget-protocol
    let r69 = run_varloop().unwrap();
    assert_eq!(r69.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r69.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r69.default_enabled);
    assert_eq!(r69.decode_cells.len(), 6);
    for cell in &r69.decode_cells {
        assert_eq!(cell.mismatch_count_vs_ar, 0);
        assert!(cell.timing_permitted);
    }
    let spark_arm64 = TwoDsdHostSpec {
        host_label: "spark".into(),
        os: "linux".into(),
        arch: "aarch64".into(),
        cuda_major: 13,
        cuda_minor: 0,
        bf16_gpu: true,
        vram_gb: 128,
        two_dsd_branch: TWO_DSD_INKLING_BRANCH.into(),
        substitute_quantized_or_spark: false,
    };
    assert_eq!(
        qualify_2dsd_host(&spark_arm64),
        OutcomeStatus::UnavailableHardwareRuntime
    );
    assert_eq!(r69.negative_controls.len(), 8);
    for c in &r69.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#69 {}", c.case_id);
    }

    // #70 ttf-ternary-qwen
    let r70 = run_ttf_ternary().unwrap();
    assert_eq!(r70.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r70.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r70.default_enabled);
    assert_eq!(r70.upstream_commit, TTF_COMMIT);
    assert!(r70.ternary_pack_roundtrip_exact);
    assert!(r70.ternary_matvec_max_abs_err <= TERNARY_TOL);
    assert!((r70.ternary_trit_bpw - 1.6).abs() <= 1.0e-12);
    let trits = vec![-1, 0, 1, -1, 1, 0, 1, -1, 0, 0];
    let packed = pack_trits_5_per_byte(&trits).unwrap();
    assert_eq!(
        unpack_trits_5_per_byte(&packed, trits.len()).unwrap(),
        trits
    );
    assert!(unpack_trits_5_per_byte(&[243], 5).is_err());
    assert_eq!(r70.negative_controls.len(), 8);
    for c in &r70.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#70 {}", c.case_id);
    }

    // #71 drafter-opd-replay
    let r71 = run_drafter_opd().unwrap();
    assert_eq!(r71.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r71.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r71.default_enabled);
    assert!(r71.arms[2].mean_acceptance_length > r71.arms[0].mean_acceptance_length);
    assert!(r71.arms[2].max_target_drift_l1 <= OPD_DRIFT_TOL);
    assert!(repeat_mean_and_std(&[1.0, f64::NAN]).is_err());
    let (_, _, _, v_nf, o_nf) =
        gate_replay_verdict(&[100.0, 102.0, 98.0], &[100.5, 102.5, 98.5], 0.0, true).unwrap();
    assert_eq!(v_nf, VerdictClass::LosslessWithinNoiseFloor);
    assert_ne!(o_nf, OutcomeStatus::Accepted);
    assert_eq!(r71.negative_controls.len(), 8);
    for c in &r71.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#71 {}", c.case_id);
    }

    // #72 k2-uno-mova
    let r72 = run_k2_uno().unwrap();
    assert_eq!(r72.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r72.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r72.default_enabled);
    assert_eq!(r72.pin_status, K2_PIN_STATUS);
    assert_eq!(r72.uno_mismatches_vs_ar, 0);
    assert!(softplus_safe(f64::NAN).is_err());
    assert!((softplus_safe(50.0).unwrap() - 50.0).abs() <= 1.0e-12);
    assert_eq!(r72.negative_controls.len(), 8);
    for c in &r72.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#72 {}", c.case_id);
    }

    // #73 lfm3-d1-decision
    let r73 = run_lfm3_d1().unwrap();
    assert_eq!(r73.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r73.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r73.default_enabled);
    assert_eq!(r73.pin_status, LFM3_PIN_STATUS);
    assert!(r73.excludes_audio_model);
    assert_eq!(
        qualify_lfm3_adapter("LFM3-Audio-2B"),
        OutcomeStatus::Unsupported
    );
    assert!(r73.tri_hybrid_cached_max_abs_err <= LFM3_TOL);
    let bad_p = evaluate_typed_decisions(
        "wrong_hi_conf",
        &[TypedDecisionPrediction {
            fixture_id: "x".into(),
            schema_valid: true,
            predicted_label: Some(DecisionLabel::Allow),
            ground_truth_label: DecisionLabel::Deny,
            reported_prob: 0.99,
            latency_ms: 4.0,
        }],
    )
    .unwrap();
    assert_eq!(bad_p.consumer_correct_count, 0);
    assert_eq!(r73.negative_controls.len(), 8);
    for c in &r73.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#73 {}", c.case_id);
    }
}

#[test]
fn batch7_suite_cli_is_two_run_byte_identical_refuses_overwrite_and_has_no_home_leaks() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!(
        "inference-lab-batch7-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run1 = base.join("run1");
    let run2 = base.join("run2");

    let out1 = Command::new(bin)
        .args(["batch7-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out1.status.success(),
        "run1 failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let overwrite = Command::new(bin)
        .args(["batch7-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!overwrite.status.success());

    let out2 = Command::new(bin)
        .args(["batch7-suite", "--out", run2.to_str().unwrap()])
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
            assert!(b1.is_empty(), "COMPLETE marker must be 0 bytes");
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
        "issue-62-disagg-kv-transfer",
        "issue-63-spark-rdma-tp",
        "issue-64-spark-kv-checkpoint",
        "issue-65-spark-copydrafter",
        "issue-66-hypertune-kv-adapters",
        "issue-67-qwen38-longbook-gdn",
        "issue-68-linquant-recurrent",
        "issue-69-varloop-budget-protocol",
        "issue-70-ttf-ternary-qwen",
        "issue-71-drafter-opd-replay",
        "issue-72-k2-uno-mova",
        "issue-73-lfm3-d1-decision",
    ];
    for sub in subdirs {
        for f in ["inputs.json", "results.json", "summary.md", "COMPLETE"] {
            let b1 = fs::read(run1.join(sub).join(f)).unwrap();
            let b2 = fs::read(run2.join(sub).join(f)).unwrap();
            assert_eq!(b1, b2, "{sub}/{f} differed across two runs");
            if f == "COMPLETE" {
                assert!(b1.is_empty(), "{sub}/COMPLETE must be 0 bytes");
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
