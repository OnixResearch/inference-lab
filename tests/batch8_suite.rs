use std::fs;
use std::process::Command;

use inference_lab::apple_nvidia_sidecar::{
    qualify_sidecar_hardware, run_default_harness as run_sidecar, SidecarLinkSpec,
    PARITY_TOL as SIDECAR_TOL, UPSTREAM_COMMIT as SIDECAR_COMMIT,
};
use inference_lab::blink_cpu_free_serving::{
    percentile_sorted, qualify_rdma_to_gpu, run_default_harness as run_blink, RdmaHostQualification,
};
use inference_lab::exl3_longctx_quant::{
    pack_exl3_tiles, quantize_exl3_tile, run_default_harness as run_exl3, unpack_exl3_tiles,
    EXACT_EXL3_BPW, PARITY_TOL as EXL3_TOL, UPSTREAM_COMMIT as STRIX_COMMIT,
};
use inference_lab::hnet_byte_modeling::{
    dynamic_chunk_bytes, run_default_harness as run_hnet, PARITY_TOL as HNET_TOL,
};
use inference_lab::json::sha256_hex;
use inference_lab::mimo_hysparse2_kv::{
    compute_ablation_row, run_default_harness as run_hysparse2, PARITY_TOL as HYSPARSE_TOL,
};
use inference_lab::pdl_breakable_graphs::{
    reduce_correlated_trace, run_default_harness as run_pdl, PARITY_TOL as PDL_TOL,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::qwen35_neuron_steering::{
    exact_two_neuron_delta, run_default_harness as run_steering,
};
use inference_lab::saber_dflash2_power::{
    evaluate_power_capped_arm, run_default_harness as run_saber, PowerSensorState,
    STOCHASTIC_TOL as SABER_TOL, UPSTREAM_COMMIT as SABER_COMMIT,
};
use inference_lab::silicon_codesign_audit::{
    evaluate_roofline_platform, run_default_harness as run_silicon_audit, HardwareEvidenceMaturity,
    RooflinePlatformInput,
};
use inference_lab::tensorlean_kernel_verify::{
    check_formal_obligations, run_default_harness as run_tensorlean, KernelIrSpec,
};
use inference_lab::tkcc_hillclimb_profile::{
    profile_hillclimb_cycles, run_default_harness as run_tkcc, HillClimbTrialTiming,
};

#[test]
fn batch8_issues_74_through_79_contracts_and_negative_controls() {
    // #74 pdl-breakable-graphs
    let r74 = run_pdl().unwrap();
    assert_eq!(r74.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r74.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r74.default_enabled);
    assert!(r74.synthetic_contract_fixture);
    assert_eq!(r74.arms.len(), 4);
    for a in &r74.arms {
        assert!(a.parity_passed);
        assert!(a.max_abs_err_vs_standard <= PDL_TOL);
        assert!(a.timing_permitted);
    }
    assert!(reduce_correlated_trace(&[], 0).is_err());
    assert_eq!(r74.negative_controls.len(), 8);
    for c in &r74.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#74 {}", c.case_id);
    }

    // #75 tensorlean-kernel-verify
    let r75 = run_tensorlean().unwrap();
    assert_eq!(r75.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r75.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r75.default_enabled);
    assert!((r75.formal_catch_rate.unwrap() - 1.0).abs() <= 1.0e-12);
    assert!(r75.aligned_diff_catch_rate.unwrap() < r75.formal_catch_rate.unwrap());
    let mut bad_ir = KernelIrSpec::reference();
    bad_ir.causal_mask_strict_le = false;
    assert!(!check_formal_obligations(&bad_ir).is_empty());
    assert_eq!(r75.negative_controls.len(), 8);
    for c in &r75.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#75 {}", c.case_id);
    }

    // #76 blink-cpu-free-serving
    let r76 = run_blink().unwrap();
    assert_eq!(r76.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r76.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r76.default_enabled);
    assert_eq!(r76.rdma_gate_outcome, OutcomeStatus::BlockedOnArtifacts);
    assert_eq!(
        qualify_rdma_to_gpu(&RdmaHostQualification {
            blink_artifacts_pinned: true,
            smartnic_gpudirect_rdma: false,
            same_pcie_root_complex: true,
            persistent_kernel_supported: true,
            silent_tcp_fallback_requested: true,
        }),
        OutcomeStatus::Invalid
    );
    assert!(percentile_sorted(&[], 99.0).unwrap().is_none());
    assert!(percentile_sorted(&[1.0, f64::NAN], 50.0).is_err());
    assert_eq!(r76.negative_controls.len(), 8);
    for c in &r76.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#76 {}", c.case_id);
    }

    // #77 qwen35-neuron-steering
    let r77 = run_steering().unwrap();
    assert_eq!(r77.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r77.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r77.default_enabled);
    assert_eq!(r77.dose_response.len(), 6);
    assert!(r77.dose_response[0].targeted_kl_vs_unsteered <= 1.0e-12);
    assert!(r77.dose_response[5].targeted_kl_vs_unsteered > 0.01);
    for row in &r77.dose_response {
        assert!(row.targeted_replay_exact);
        assert!(row.random_ctrl_replay_exact);
    }
    assert!(exact_two_neuron_delta(1.0, 1.0, 2.0, 0.0).is_err());
    assert_eq!(r77.negative_controls.len(), 8);
    for c in &r77.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#77 {}", c.case_id);
    }

    // #78 hnet-byte-modeling
    let r78 = run_hnet().unwrap();
    assert_eq!(r78.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r78.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r78.default_enabled);
    assert_eq!(r78.edge_cases.len(), 6);
    for e in &r78.edge_cases {
        assert!(e.exact_byte_roundtrip);
        assert!(e.upsample_max_abs_err <= HNET_TOL);
    }
    assert!(dynamic_chunk_bytes(&[0xff, 0xfe], 0.4).is_err());
    assert_eq!(r78.negative_controls.len(), 8);
    for c in &r78.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#78 {}", c.case_id);
    }

    // #79 tkcc-hillclimb-profile
    let r79 = run_tkcc().unwrap();
    assert_eq!(r79.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r79.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r79.default_enabled);
    assert_eq!(r79.profiles[0].dominant_bottleneck_phase, "compile");
    assert_eq!(r79.profiles[1].dominant_bottleneck_phase, "measurement");
    let skip_parity = HillClimbTrialTiming {
        candidate_id: "skip".into(),
        outcome: OutcomeStatus::Accepted,
        parity_gate_executed: false,
        compile_s: 1.0,
        warm_up_s: 1.0,
        parity_check_s: 0.0,
        measurement_s: 1.0,
        teardown_overhead_s: 0.5,
    };
    assert!(profile_hillclimb_cycles("skip", &[skip_parity]).is_err());
    assert_eq!(r79.negative_controls.len(), 8);
    for c in &r79.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#79 {}", c.case_id);
    }
}

#[test]
fn batch8_issues_80_through_84_contracts_and_negative_controls() {
    // #80 exl3-longctx-quant
    let r80 = run_exl3().unwrap();
    assert_eq!(r80.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r80.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r80.default_enabled);
    assert_eq!(r80.upstream_commit, STRIX_COMMIT);
    assert_eq!(r80.upstream_commit.len(), 40);
    assert!(r80.pack_roundtrip_exact);
    assert!((r80.exact_packed_bpw - EXACT_EXL3_BPW).abs() <= 1.0e-12);
    assert!(r80.quant_metrics[2].matvec_pack_parity_max_err <= EXL3_TOL);
    let t0 = quantize_exl3_tile(&[0.25; 16]).unwrap();
    let t1 = quantize_exl3_tile(&[-0.5; 16]).unwrap();
    let packed = pack_exl3_tiles(&[t0.clone(), t1.clone()]).unwrap();
    assert_eq!(packed.len(), 9);
    assert_eq!(unpack_exl3_tiles(&packed, 2).unwrap(), vec![t0, t1]);
    assert_eq!(r80.negative_controls.len(), 8);
    for c in &r80.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#80 {}", c.case_id);
    }

    // #81 saber-dflash2-power
    let r81 = run_saber().unwrap();
    assert_eq!(r81.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r81.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r81.default_enabled);
    assert_eq!(r81.upstream_commit, SABER_COMMIT);
    assert_eq!(r81.upstream_commit.len(), 40);
    assert!(r81.dflash2_verified_l1_err <= SABER_TOL);
    let no_sensor = evaluate_power_capped_arm(
        "no_sensor",
        true,
        true,
        0.0,
        2.5,
        50.0,
        150.0,
        210.0,
        PowerSensorState::MissingOrUncalibrated,
        None,
        true,
    )
    .unwrap();
    assert!(no_sensor.tok_per_sec_per_watt.is_none());
    assert_ne!(no_sensor.admission_outcome, OutcomeStatus::Accepted);
    assert_eq!(r81.negative_controls.len(), 8);
    for c in &r81.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#81 {}", c.case_id);
    }

    // #82 mimo-hysparse2-kv
    let r82 = run_hysparse2().unwrap();
    assert_eq!(r82.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r82.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r82.default_enabled);
    assert!(r82.cached_vs_full_max_abs_err <= HYSPARSE_TOL);
    assert!(r82.causal_non_leakage_max_err <= HYSPARSE_TOL);
    assert!(compute_ablation_row(8192, 49, 60, 2048, 22_528, 0.25, 0.9, 0.9).is_err());
    assert_eq!(r82.negative_controls.len(), 8);
    for c in &r82.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#82 {}", c.case_id);
    }

    // #83 apple-nvidia-sidecar
    let r83 = run_sidecar().unwrap();
    assert_eq!(r83.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(
        r83.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!r83.default_enabled);
    assert_eq!(r83.upstream_commit, SIDECAR_COMMIT);
    assert_eq!(r83.upstream_commit.len(), 40);
    assert!(r83.prefill_split_max_abs_err <= SIDECAR_TOL);
    assert!(r83.forced_decode_split_max_abs_err <= SIDECAR_TOL);
    assert!(r83.reset_reuse_max_abs_err <= SIDECAR_TOL);
    assert_eq!(
        qualify_sidecar_hardware(&SidecarLinkSpec {
            apple_metal_present: true,
            nvidia_egpu_present: false,
            thunderbolt_link_gbps: 26.0,
            silent_cpu_fallback_allowed: true,
        })
        .unwrap(),
        OutcomeStatus::Invalid
    );
    assert_eq!(r83.negative_controls.len(), 8);
    for c in &r83.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#83 {}", c.case_id);
    }

    // #84 silicon-codesign-audit
    let r84 = run_silicon_audit().unwrap();
    assert_eq!(r84.execution_class, ExecutionClass::BoundedContractHarness);
    assert_eq!(r84.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
    assert!(!r84.default_enabled);
    assert_eq!(r84.assumptions.len(), 5);
    assert!(evaluate_roofline_platform(&RooflinePlatformInput {
        platform_id: "bad_presil".into(),
        maturity: HardwareEvidenceMaturity::PreSiliconSimulationOnly,
        active_params_billions: 27.0,
        weight_bits: 2.0,
        memory_bandwidth_gb_s: 1000.0,
        peak_compute_tflops: 1000.0,
        board_power_watts: 100.0,
        physical_silicon_validated: true,
    })
    .is_err());
    assert_eq!(r84.negative_controls.len(), 8);
    for c in &r84.negative_controls {
        assert_ne!(c.outcome, OutcomeStatus::Accepted, "#84 {}", c.case_id);
    }
}

#[test]
fn batch8_suite_cli_is_two_run_byte_identical_refuses_overwrite_and_has_no_home_leaks() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!(
        "inference-lab-batch8-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run1 = base.join("run1");
    let run2 = base.join("run2");

    let out1 = Command::new(bin)
        .args(["batch8-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out1.status.success(),
        "run1 failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    let overwrite = Command::new(bin)
        .args(["batch8-suite", "--out", run1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!overwrite.status.success());

    let out2 = Command::new(bin)
        .args(["batch8-suite", "--out", run2.to_str().unwrap()])
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
        "issue-74-pdl-breakable-graphs",
        "issue-75-tensorlean-kernel-verify",
        "issue-76-blink-cpu-free-serving",
        "issue-77-qwen35-neuron-steering",
        "issue-78-hnet-byte-modeling",
        "issue-79-tkcc-hillclimb-profile",
        "issue-80-exl3-longctx-quant",
        "issue-81-saber-dflash2-power",
        "issue-82-mimo-hysparse2-kv",
        "issue-83-apple-nvidia-sidecar",
        "issue-84-silicon-codesign-audit",
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
