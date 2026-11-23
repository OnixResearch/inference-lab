use std::fs;
use std::process::Command;

use inference_lab::adv_spec_reasoning::{
    argmax_checked, evaluate_policy, run_default_harness as run_adv_spec, softmax_checked,
    validate_matched_seed_comparison, DraftPolicy, VOCAB_SIZE as ADV_VOCAB,
};
use inference_lab::clairic_extraction::{
    run_default_harness as run_clairic, validate_architecture_path, verify_pinned_revision,
    ClairicArchitecturePath, CLAIRIC_FLASH_9B_COMMIT, CLAIRIC_FLASH_9B_REPO,
};
use inference_lab::decode_megakernels::{
    decode_multi_launch_reference, decode_persistent_megakernel,
    run_default_harness as run_megakernels, validate_pin_honesty, MegakernelFaults,
    MiniCausalTransformer,
};
use inference_lab::harness_lora_compare::{
    canonical_trigrams, run_default_harness as run_harness_lora,
};
use inference_lab::inkling_mini_serve::{
    forward_looped_3x_shared, forward_unrolled_48_reference, qualify_inkling_host_and_runtime,
    run_default_harness as run_inkling_mini, HostFloorClass, HostServeRequest,
    MiniatureInklingMini, INKLING_MINI_COMMIT, INKLING_MINI_REPO, UNROLLED_LAYER_APPLICATIONS,
};
use inference_lab::json::sha256_hex;
use inference_lab::latent_feedback::{
    run_default_harness as run_latent_feedback, validate_large_scale_claim,
    validate_matched_flop_budget,
};
use inference_lab::maya_ik2_serve::{
    qualify_maya_host, run_default_harness as run_maya_ik2, MayaHostSpec,
};
use inference_lab::nvme_coalescing::{
    parse_nvme_coalescing_feature, run_default_harness as run_nvme_coalescing, CoalescingState,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::serving_hill_climb::{
    checked_median, compute_practical_ceilings, run_default_harness as run_serving_hill_climb,
    HardwareCeilingInputs, SKILL_COMMIT,
};
use inference_lab::simplehc_mhc::{
    run_default_harness as run_simplehc_mhc, verify_doubly_stochastic,
};
use inference_lab::tufazip_online_codec::{
    decode_online_range, default_quantized_offline_prior, encode_online_range,
    pinned_small_text_prefix, run_default_harness as run_tufazip,
    UPSTREAM_COMMIT as TUFAZIP_COMMIT,
};

#[test]
fn batch6_issues_47_48_49_50_51_contracts_and_negative_controls() {
    // #47 adv-spec-reasoning
    let r47 = run_adv_spec().unwrap();
    assert_eq!(r47.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r47.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(r47.synthetic_contract_fixture);
    assert_eq!(r47.slot_noise[0].max_abs_err, 0.0);
    assert!(r47.slot_noise[3].max_abs_err > r47.slot_noise[1].max_abs_err);
    assert!(softmax_checked(&[f64::NAN; ADV_VOCAB]).is_err());
    assert!(argmax_checked(&[f64::NAN; ADV_VOCAB]).is_err());
    let empty47 = evaluate_policy(&[], DraftPolicy::UnbiasedDraft, 1, false).unwrap();
    assert!(empty47.accepted_task_rate.is_none());
    assert!(validate_matched_seed_comparison(&empty47, &empty47).is_err());
    assert_eq!(r47.negative_controls.len(), 8);
    for c in &r47.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#47 {}", c.case_id);
    }

    // #48 nvme-coalescing
    let r48 = run_nvme_coalescing().unwrap();
    assert_eq!(r48.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r48.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(r48.storage_latency_is_not_tok_s_claim);
    assert!(!r48.mutates_host_service);
    assert_eq!(
        parse_nvme_coalescing_feature("Aggregation Time (TIME): 0\nAggregation Threshold (THR): 0")
            .state,
        CoalescingState::Off
    );
    assert_eq!(
        parse_nvme_coalescing_feature("Aggregation Time (TIME): 2\nAggregation Threshold (THR): 8")
            .state,
        CoalescingState::On
    );
    assert_eq!(
        parse_nvme_coalescing_feature("unavailable").state,
        CoalescingState::Unknown
    );
    assert_eq!(r48.negative_controls.len(), 8);
    for c in &r48.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#48 {}", c.case_id);
    }

    // #49 harness-lora-compare
    let r49 = run_harness_lora().unwrap();
    assert_eq!(r49.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r49.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert_eq!(r49.arms.len(), 4);
    assert!(canonical_trigrams("a b c d").contains("a b c"));
    assert_eq!(
        r49.overlap_audit.iter().filter(|o| o.contaminated).count(),
        1
    );
    assert_eq!(r49.negative_controls.len(), 8);
    for c in &r49.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#49 {}", c.case_id);
    }

    // #50 latent-feedback
    let r50 = run_latent_feedback().unwrap();
    assert_eq!(r50.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r50.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert_eq!(r50.large_scale_stability_status, "open_untested");
    assert_eq!(validate_large_scale_claim("stable"), OutcomeStatus::Invalid);
    assert!(r50.grad_check_max_abs_err <= 1.0e-6);
    assert_eq!(r50.causal_invariance_max_err, 0.0);
    assert!(validate_matched_flop_budget(0, 100, 0.05).is_err());
    assert!(validate_matched_flop_budget(100, 100, f64::NAN).is_err());
    assert_eq!(r50.negative_controls.len(), 8);
    for c in &r50.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#50 {}", c.case_id);
    }

    // #51 simplehc-mhc
    let r51 = run_simplehc_mhc().unwrap();
    assert_eq!(r51.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r51.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert_eq!(
        r51.primary_source_pin_status,
        "post_only_unlocated_paper_repo"
    );
    assert!(r51.mhc_overhead_vs_simplehc_pct.unwrap() > 0.0);
    let non_ds = [[0.5; 4]; 4];
    let (_, ds_st) = verify_doubly_stochastic(&non_ds, 1.0e-6).unwrap();
    assert_eq!(ds_st, OutcomeStatus::Invalid);
    assert_eq!(r51.negative_controls.len(), 8);
    for c in &r51.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#51 {}", c.case_id);
    }
}

#[test]
fn batch6_issues_53_54_55_57_59_61_contracts_and_negative_controls() {
    // #53 serving-hill-climb
    let r53 = run_serving_hill_climb().unwrap();
    assert_eq!(r53.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r53.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r53.default_enabled);
    assert_eq!(r53.skill_commit, SKILL_COMMIT);
    assert!(checked_median(&[1.0, f64::NAN, 3.0]).is_err());
    let above = compute_practical_ceilings(&HardwareCeilingInputs {
        measured_mem_bw_gb_s: 200.0,
        measured_matmul_tflops: 100.0,
        bandwidth_from_measurement: true,
        flops_from_measurement: true,
        active_weight_bytes_per_token: 4.0e9,
        kv_bytes_per_token: 0.0,
        active_flops_per_token: 8.0e9,
        practical_efficiency: 0.8,
        requested_target_decode_tok_s: Some(500.0),
    })
    .unwrap();
    assert!(above.target_exceeds_practical_ceiling);
    assert_eq!(above.target_feasibility_outcome, OutcomeStatus::Unsupported);
    assert_eq!(r53.negative_controls.len(), 8);
    for c in &r53.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#53 {}", c.case_id);
    }

    // #54 tufazip-online-codec
    let r54 = run_tufazip().unwrap();
    assert_eq!(r54.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r54.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r54.default_enabled);
    assert!(!r54.enwik9_full_run_attempted);
    assert!(r54.online_codec.roundtrip_sha256_match);
    assert!(!r54.online_codec.online_weights_transferred);
    assert!(r54.conventional_lz77.roundtrip_sha256_match);
    assert_eq!(TUFAZIP_COMMIT, "b9d678db3a0f726e117c8cab51927975830395ac");
    assert_eq!(TUFAZIP_COMMIT.len(), 40);
    // Directly confirm skipping decoder online update breaks lossless round-trip
    let input = pinned_small_text_prefix();
    let art = encode_online_range(&input, &default_quantized_offline_prior()).unwrap();
    let ok = decode_online_range(&art, true).unwrap();
    let broken = decode_online_range(&art, false).unwrap();
    assert_eq!(sha256_hex(&ok), sha256_hex(&input));
    assert_ne!(sha256_hex(&broken), sha256_hex(&input));
    assert_eq!(r54.negative_controls.len(), 8);
    for c in &r54.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#54 {}", c.case_id);
    }

    // #55 clairic-extraction
    let r55 = run_clairic().unwrap();
    assert_eq!(r55.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r55.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r55.default_enabled);
    assert!(!r55.starts_issue_10_extraction);
    assert_eq!(
        validate_architecture_path(ClairicArchitecturePath::PreviewSingleForwardPass),
        OutcomeStatus::Unsupported
    );
    assert_eq!(
        verify_pinned_revision(CLAIRIC_FLASH_9B_REPO, CLAIRIC_FLASH_9B_COMMIT),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        verify_pinned_revision(CLAIRIC_FLASH_9B_REPO, "main"),
        OutcomeStatus::Invalid
    );
    assert_eq!(r55.negative_controls.len(), 8);
    for c in &r55.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#55 {}", c.case_id);
    }

    // #57 inkling-mini-serve
    let r57 = run_inkling_mini().unwrap();
    assert_eq!(r57.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r57.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r57.default_enabled);
    assert_eq!(UNROLLED_LAYER_APPLICATIONS, 48);
    let mini = MiniatureInklingMini::build_deterministic().unwrap();
    let x = [0.25, -0.5, 0.75, 0.125];
    let unrolled = forward_unrolled_48_reference(&mini, &x).unwrap();
    let looped = forward_looped_3x_shared(&mini, &x).unwrap();
    for (a, b) in unrolled.output.iter().zip(looped.output.iter()) {
        assert!((a - b).abs() <= 1.0e-12);
    }
    let maya_on_57 = qualify_inkling_host_and_runtime(&HostServeRequest {
        host_label: "spark".into(),
        platform: HostFloorClass::DgxSpark,
        runtime_name: "maya-ik2".into(),
        runtime_available_locally: false,
        checkpoint_repo: INKLING_MINI_REPO.into(),
        checkpoint_commit: INKLING_MINI_COMMIT.into(),
    });
    assert_eq!(maya_on_57.outcome, OutcomeStatus::Unsupported);
    assert_eq!(r57.negative_controls.len(), 8);
    for c in &r57.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#57 {}", c.case_id);
    }

    // #59 maya-ik2-serve
    let r59 = run_maya_ik2().unwrap();
    assert_eq!(r59.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r59.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r59.default_enabled);
    assert!(r59.distinct_from_issue_57);
    assert!(!r59.installs_gpu_unlock);
    let bad_arch = qualify_maya_host(&MayaHostSpec {
        host_label: "rtx4090".into(),
        os: "linux".into(),
        gpu_arch: "sm_89".into(),
        vram_gb: 24,
        cuda_major: 12,
        cuda_minor: 8,
        gguf_quant: "UD-Q4_K_XL".into(),
        gpu_unlock_present: false,
        binary_and_weights_available: false,
    });
    assert_eq!(bad_arch.outcome, OutcomeStatus::Unsupported);
    assert_eq!(r59.negative_controls.len(), 8);
    for c in &r59.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#59 {}", c.case_id);
    }

    // #61 decode-megakernels
    let r61 = run_megakernels().unwrap();
    assert_eq!(r61.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r61.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert_eq!(
        validate_pin_honesty("repo_url_only_commit_unpinned_offline", None),
        OutcomeStatus::Accepted
    );
    let model = MiniCausalTransformer::build_deterministic();
    let ref_trace = decode_multi_launch_reference(&model, &[1, 3], 4).unwrap();
    let mega_trace =
        decode_persistent_megakernel(&model, &[1, 3], 4, MegakernelFaults::default()).unwrap();
    assert_eq!(ref_trace.generated_tokens, mega_trace.generated_tokens);
    assert!(mega_trace.host_syncs < ref_trace.host_syncs);
    assert_eq!(r61.negative_controls.len(), 8);
    for c in &r61.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#61 {}", c.case_id);
    }
}

#[test]
fn batch6_suite_cli_is_two_run_byte_identical_refuses_overwrite_and_has_no_home_leaks() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!(
        "inference-lab-batch6-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run1 = base.join("run1");
    let run2 = base.join("run2");

    let out1 = Command::new(bin)
        .args(["batch6-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out1.status.success(),
        "run1 failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let overwrite = Command::new(bin)
        .args(["batch6-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!overwrite.status.success());

    let out2 = Command::new(bin)
        .args(["batch6-suite", "--out", run2.to_str().unwrap()])
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
        "issue-47-adv-spec-reasoning",
        "issue-48-nvme-coalescing",
        "issue-49-harness-lora-compare",
        "issue-50-latent-feedback",
        "issue-51-simplehc-mhc",
        "issue-53-serving-hill-climb",
        "issue-54-tufazip-online-codec",
        "issue-55-clairic-extraction",
        "issue-57-inkling-mini-serve",
        "issue-59-maya-ik2-serve",
        "issue-61-decode-megakernels",
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
