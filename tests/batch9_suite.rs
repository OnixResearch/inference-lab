use std::fs;
use std::process::Command;

use inference_lab::alora_prefix_cache::{
    run_default_harness as run_alora, PARITY_TOL as ALORA_TOL,
};
use inference_lab::anchor_label_drafter::run_default_harness as run_anchor_drafter;
use inference_lab::drafter_opd_replay::DRIFT_TOL;
use inference_lab::json::sha256_hex;
use inference_lab::knowledgator_sys1_extract::run_default_harness as run_knowledgator;
use inference_lab::ling_fp8kv_alloc_cap::{
    quantize_fp8_e4m3, run_default_harness as run_ling, PARITY_TOL as LING_TOL, RECIPE_COMMIT,
};
use inference_lab::link_sparse_attention::{
    run_default_harness as run_link, PARITY_TOL as LINK_TOL,
};
use inference_lab::loop_kv_compression::{
    pack_int2_codes, run_default_harness as run_loop_kv, unpack_int2_codes,
    PARITY_TOL as LOOP_KV_TOL,
};
use inference_lab::mit_qda_quant::{
    run_default_harness as run_qda, spearman_rank_correlation, MATCHED_BPW_BUDGET,
};
use inference_lab::moe_spec_roofline::{
    expected_accepted_tokens, run_default_harness as run_moe_spec, PARITY_TOL as MOE_TOL,
};
use inference_lab::parasite_feature_control::{
    run_default_harness as run_parasite, validate_orthonormal_basis, MechanismClassification,
    REF_COMMIT as PARASITE_COMMIT,
};
use inference_lab::perseus_mpk_rdma::{
    qualify_two_node_rdma_host, run_default_harness as run_perseus, TwoNodeRdmaHostSpec,
    PARITY_TOL as PERSEUS_TOL,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::unlimited_ocr_longdoc::{
    levenshtein_distance, normalized_edit_distance, run_default_harness as run_ocr,
};

#[test]
fn batch9_issues_85_through_90_contracts_and_negative_controls() {
    // #85 link-sparse-attention
    let r85 = run_link().unwrap();
    assert_eq!(r85.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r85.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r85.default_enabled);
    assert!(r85.synthetic_contract_fixture);
    assert!(r85.sparse_arms[1].max_abs_err_vs_dense <= LINK_TOL);
    assert!(r85.sparse_arms[2].needle_block_recalled);
    assert_eq!(r85.negative_controls.len(), 8);
    for c in &r85.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#85 {}", c.case_id);
    }

    // #86 mit-qda-quant
    let r86 = run_qda().unwrap();
    assert_eq!(r86.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r86.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r86.default_enabled);
    assert!((r86.recipe_rows[3].avg_effective_bpw - MATCHED_BPW_BUDGET).abs() <= 1.0e-12);
    assert!(r86.recipe_rows[3].within_bit_budget);
    assert!(r86.recipe_rows[3].mean_logit_kl < r86.recipe_rows[1].mean_logit_kl);
    assert!(spearman_rank_correlation(&[1.0, f64::NAN], &[1.0, 2.0]).is_err());
    assert_eq!(r86.negative_controls.len(), 8);
    for c in &r86.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#86 {}", c.case_id);
    }

    // #87 knowledgator-sys1-extract
    let r87 = run_knowledgator().unwrap();
    assert_eq!(r87.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r87.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r87.default_enabled);
    assert_eq!(r87.arm_metrics.len(), 3);
    assert!(r87.arm_metrics[1].exact_field_f1.unwrap() > 0.999);
    assert_eq!(r87.negative_controls.len(), 8);
    for c in &r87.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#87 {}", c.case_id);
    }

    // #88 unlimited-ocr-longdoc
    let r88 = run_ocr().unwrap();
    assert_eq!(r88.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r88.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r88.default_enabled);
    assert_eq!(levenshtein_distance("abcd", "abef"), 2);
    assert!(normalized_edit_distance("", "").is_none());
    assert_eq!(r88.observations.len(), 8);
    assert_eq!(r88.negative_controls.len(), 8);
    for c in &r88.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#88 {}", c.case_id);
    }

    // #89 loop-kv-compression
    let r89 = run_loop_kv().unwrap();
    assert_eq!(r89.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r89.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r89.default_enabled);
    assert!(r89.zero_residual_identity_max_err <= LOOP_KV_TOL);
    assert!(r89.cross_loop_cosines[0].key_cosine.unwrap() > 0.95);
    let packed = pack_int2_codes(&[0, 1, 2, 3, 3, 2, 1, 0]).unwrap();
    assert_eq!(
        unpack_int2_codes(&packed, 8).unwrap(),
        vec![0, 1, 2, 3, 3, 2, 1, 0]
    );
    assert!((r89.methods[2].effective_bpw_payload - 0.5).abs() <= 1.0e-12);
    assert!((r89.methods[3].effective_bpw_payload - 0.25).abs() <= 1.0e-12);
    assert_eq!(r89.negative_controls.len(), 8);
    for c in &r89.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#89 {}", c.case_id);
    }

    // #90 alora-prefix-cache
    let r90 = run_alora().unwrap();
    assert_eq!(r90.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r90.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r90.default_enabled);
    assert!(!r90.arm_summaries[1].prefix_kv_cache_reusable);
    assert!(r90.arm_summaries[2].prefix_kv_cache_reusable);
    assert!(r90.arm_summaries[2].prefix_kv_max_diff_vs_base <= ALORA_TOL);
    assert!(r90.arm_summaries[2].cached_prefix_vs_full_max_err <= ALORA_TOL);
    assert!(r90.arm_summaries[2].suffix_adaptation_l2_delta > 1.0e-3);
    assert_eq!(r90.negative_controls.len(), 8);
    for c in &r90.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#90 {}", c.case_id);
    }
}

#[test]
fn batch9_issues_91_through_95_contracts_and_negative_controls() {
    // #91 perseus-mpk-rdma
    let r91 = run_perseus().unwrap();
    assert_eq!(r91.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r91.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r91.default_enabled);
    for m in &r91.mode_profiles {
        assert!(m.timing_admitted);
        assert!(m.collective_max_abs_err <= PERSEUS_TOL);
    }
    assert_eq!(
        qualify_two_node_rdma_host(&TwoNodeRdmaHostSpec {
            node_count: 2,
            rdma_nic_present: false,
            gpudirect_rdma_enabled: false,
            silent_tcp_fallback_allowed: true,
            upstream_artifacts_pinned: true,
        }),
        OutcomeStatus::Invalid
    );
    assert_eq!(r91.negative_controls.len(), 8);
    for c in &r91.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#91 {}", c.case_id);
    }

    // #92 moe-spec-roofline
    let r92 = run_moe_spec().unwrap();
    assert_eq!(r92.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r92.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r92.default_enabled);
    for o in &r92.overlap_observations {
        assert!(o.batched_vs_sequential_max_err <= MOE_TOL);
    }
    assert!(r92.overlap_observations[3].temporal_overlap_ratio > 0.25);
    assert!(expected_accepted_tokens(0, 0.8).is_err());
    assert!(expected_accepted_tokens(4, f64::NAN).is_err());
    assert_eq!(r92.negative_controls.len(), 8);
    for c in &r92.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#92 {}", c.case_id);
    }

    // #93 anchor-label-drafter
    let r93 = run_anchor_drafter().unwrap();
    assert_eq!(r93.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r93.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r93.default_enabled);
    for a in &r93.arm_results {
        assert!(a.lossless_verified);
        assert!(a.max_target_l1_drift <= DRIFT_TOL);
        assert_eq!(a.outcome, OutcomeStatus::Accepted);
    }
    assert!(
        r93.arm_results[1].mean_accepted_draft_length
            > r93.arm_results[0].mean_accepted_draft_length
    );
    assert_eq!(r93.negative_controls.len(), 8);
    for c in &r93.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#93 {}", c.case_id);
    }

    // #94 parasite-feature-control
    let r94 = run_parasite().unwrap();
    assert_eq!(r94.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r94.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r94.default_enabled);
    assert_eq!(PARASITE_COMMIT.len(), 40);
    assert!(r94.planted_parasite_eval.sufficiency_passed);
    assert!(!r94.planted_parasite_eval.necessity_passed);
    assert_eq!(
        r94.planted_parasite_eval.classification,
        MechanismClassification::SufficientNotNecessary
    );
    assert!(r94.positive_mechanism_eval.sufficiency_passed);
    assert!(r94.positive_mechanism_eval.necessity_passed);
    assert_eq!(
        r94.positive_mechanism_eval.classification,
        MechanismClassification::FoundMechanism
    );
    assert!(r94.qwen_interchange_replay_exact);
    assert!(r94.qwen_ablation_replay_exact);
    assert!(validate_orthonormal_basis(2, &[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).is_err());
    assert_eq!(r94.negative_controls.len(), 9);
    for c in &r94.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#94 {}", c.case_id);
    }

    // #95 ling-fp8kv-alloc-cap
    let r95 = run_ling().unwrap();
    assert_eq!(r95.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r95.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r95.default_enabled);
    assert_eq!(RECIPE_COMMIT.len(), 40);
    assert!(r95.digest_sensitivity_all_6_axes_verified);
    assert_eq!(r95.matrix_arms.len(), 4);
    assert!(r95.mla_parity.isolated_1stage_vs_2stage_bf16_max_logit_err <= LING_TOL);
    assert!(quantize_fp8_e4m3(f64::NAN).is_err());
    assert_eq!(r95.negative_controls.len(), 9);
    for c in &r95.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#95 {}", c.case_id);
    }
}

#[test]
fn batch9_suite_cli_is_two_run_byte_identical_refuses_overwrite_and_has_no_home_leaks() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!(
        "inference-lab-batch9-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run1 = base.join("run1");
    let run2 = base.join("run2");

    let out1 = Command::new(bin)
        .args(["batch9-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out1.status.success(),
        "run1 failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let overwrite = Command::new(bin)
        .args(["batch9-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!overwrite.status.success());

    let out2 = Command::new(bin)
        .args(["batch9-suite", "--out", run2.to_str().unwrap()])
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
        "issue-85-link-sparse-attention",
        "issue-86-mit-qda-quant",
        "issue-87-knowledgator-sys1-extract",
        "issue-88-unlimited-ocr-longdoc",
        "issue-89-loop-kv-compression",
        "issue-90-alora-prefix-cache",
        "issue-91-perseus-mpk-rdma",
        "issue-92-moe-spec-roofline",
        "issue-93-anchor-label-drafter",
        "issue-94-parasite-feature-control",
        "issue-95-ling-fp8kv-alloc-cap",
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
