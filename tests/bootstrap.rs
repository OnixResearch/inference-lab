use std::fs;
use std::process::Command;

use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::seed;

#[test]
fn seed_preserves_all_non_collapsing_outcomes_and_denominators() {
    let scenario = seed::default_scenario();
    let report = seed::evaluate(&scenario).expect("seed evaluation succeeds");
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert_eq!(report.profiles.len(), 3);

    for profile in &report.profiles {
        assert_eq!(profile.planned, scenario.requests.len());
        assert_eq!(profile.outcomes.len(), scenario.requests.len());
        let sum = profile.accepted
            + profile.invalid
            + profile.oom
            + profile.timeout
            + profile.unsupported;
        assert_eq!(
            sum, profile.planned,
            "outcomes must partition planned population"
        );
        // Every profile must keep invalid, oom, and unsupported distinct.
        assert_eq!(profile.invalid, 1);
        assert!(profile.oom >= 1);
        assert_eq!(profile.unsupported, 1);
    }

    // Split profile meets tight deadline that static-batch misses.
    let static_p = &report.profiles[1];
    let split_p = &report.profiles[2];
    assert_eq!(static_p.timeout, 1);
    assert_eq!(split_p.timeout, 0);
    assert_eq!(split_p.accepted, 3);

    let req3 = split_p
        .outcomes
        .iter()
        .find(|o| o.request_id == "req-03-invalid-template")
        .unwrap();
    assert_eq!(req3.status, OutcomeStatus::Invalid);
    assert_eq!(req3.logical_completion_tick, None);
}

#[test]
fn seed_rejects_empty_or_zero_bound_scenarios() {
    let mut bad = seed::default_scenario();
    bad.requests.clear();
    assert!(seed::evaluate(&bad).is_err());

    let mut bad_kv = seed::default_scenario();
    bad_kv.kv_cache_token_limit = 0;
    assert!(seed::evaluate(&bad_kv).is_err());
}

#[test]
fn cli_publishes_byte_identical_runs_and_refuses_overwrite() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!("inference-lab-cli-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    for cmd in [
        "benchmark",
        "diffusion",
        "trace",
        "stage",
        "power",
        "tokenizer",
        "related-work",
    ] {
        let dir1 = base.join(format!("{cmd}-run1"));
        let dir2 = base.join(format!("{cmd}-run2"));

        let out1 = Command::new(bin)
            .args([cmd, "--out", dir1.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            out1.status.success(),
            "command {cmd} failed: {}",
            String::from_utf8_lossy(&out1.stderr)
        );

        let out2 = Command::new(bin)
            .args([cmd, "--out", dir2.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(out2.status.success());

        // Overwrite must fail cleanly without touching existing artifacts.
        let overwrite = Command::new(bin)
            .args([cmd, "--out", dir1.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!overwrite.status.success());

        for name in ["inputs.json", "results.json", "summary.md", "COMPLETE"] {
            let b1 = fs::read(dir1.join(name)).unwrap();
            let b2 = fs::read(dir2.join(name)).unwrap();
            assert_eq!(b1, b2, "{cmd}/{name} must be byte-identical across runs");
            if name == "COMPLETE" {
                assert!(b1.is_empty(), "COMPLETE marker must be empty");
            } else {
                let text = String::from_utf8(b1).unwrap();
                assert!(
                    !text.contains("/home/"),
                    "{cmd}/{name} must not leak /home/ paths"
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&base);
}
