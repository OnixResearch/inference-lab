use std::fs;
use std::process::Command;

use inference_lab::json::sha256_hex;
use inference_lab::lifecycle::run_default_harness as run_lifecycle;
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};

#[test]
fn lifecycle_verifies_exclusive_partition_unified_memory_and_12_controls() {
    let report = run_lifecycle();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.aba_phase_accounting.exclusive_partition_exact());
    assert!(report
        .rsi86_variant_phase_accounting
        .exclusive_partition_exact());
    assert!(
        report.aba_phase_accounting.raw_overlapping_sum_ms()
            > report.aba_phase_accounting.total_wall_ms
    );
    assert!(!report.overlapping_memory_counters_summed);

    // Missing page-cache counter remains None (unknown), never zero.
    assert!(report
        .unified_memory_observations
        .iter()
        .any(|m| m.counter_name == "page_cache_weight_residency_mib" && m.observed_mib.is_none()));
    for m in &report.unified_memory_observations {
        assert!(!m.represents_freed_physical_capacity);
    }

    assert_eq!(report.lifecycle_controls.len(), 13);
    let by_id = |id: &str| {
        report
            .lifecycle_controls
            .iter()
            .find(|c| c.case_id == id)
            .unwrap()
    };

    assert!(by_id("01-positive-aba-first-checked-completion").classified_ready);
    assert!(by_id("03-negative-animus17-model-listed-before-ready").false_ready_detected);
    assert!(!by_id("03-negative-animus17-model-listed-before-ready").classified_ready);
    assert!(by_id("04-negative-animus17-stale-healthy-after-shutdown").false_ready_detected);
    assert!(by_id("05-negative-late-completion-previous-generation").false_ready_detected);
    assert_eq!(
        by_id("06-negative-unsupported-native-sleep-no-restart-fallback").outcome,
        OutcomeStatus::Unsupported
    );
    assert_eq!(
        by_id("08-negative-interrupted-transition-timeout").outcome,
        OutcomeStatus::Timeout
    );
    assert!(by_id("10-negative-refuse-foreign-resource-cleanup").foreign_resource_untouched);
    assert_eq!(
        by_id("11-negative-animus17-two-clients-one-slot-rejected").outcome,
        OutcomeStatus::Exhausted
    );
    assert_eq!(
        by_id("12-control-unavailable-telemetry-stays-unknown").telemetry_state,
        "unknown_telemetry_unavailable"
    );
    let delayed = by_id("13-negative-delayed-readiness-unlisted-not-ready");
    assert!(!delayed.classified_ready);
    assert_ne!(delayed.outcome, OutcomeStatus::Accepted);

    // Residency policies conserve planned denominators using failed_requests (distinct from failed_transitions).
    for pol in &report.residency_policies {
        assert_eq!(pol.max_active_engines, 1);
        assert!(!pol.heartbeat_loss_counted_as_release);
        assert_eq!(
            pol.completed_requests + pol.rejected_or_expired + pol.failed_requests,
            pol.planned_requests
        );
    }
}

#[test]
fn batch2_cli_commands_and_suite_are_byte_identical_leak_free_and_refuse_overwrite() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!(
        "inference-lab-batch2-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    // 1. Run batch2-suite twice and verify byte-identical output + overwrite refusal + empty COMPLETE + zero /home/ leaks.
    let suite1 = base.join("suite-run1");
    let suite2 = base.join("suite-run2");

    for dir in [&suite1, &suite2] {
        let out = Command::new(bin)
            .args(["batch2-suite", "--out", dir.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "batch2-suite failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let overwrite = Command::new(bin)
        .args(["batch2-suite", "--out", suite1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!overwrite.status.success());

    let subdirs = [
        ("proof-completion", "issue-02-proof-completion"),
        ("drafting", "issue-03-drafting"),
        ("kernel-opt", "issue-04-kernel-opt"),
        ("kv-routing", "issue-05-kv-routing"),
        ("adaptive-spec", "issue-06-adaptive-spec"),
        ("serving-bench", "issue-07-serving-bench"),
        ("deepseek-spark", "issue-08-deepseek-spark"),
        ("learned-compiler", "issue-09-learned-compiler"),
        ("structured-read", "issue-10-structured-read"),
        ("lifecycle", "issue-11-lifecycle"),
    ];

    let manifest_text = fs::read_to_string(suite1.join("manifest.json")).unwrap();
    assert!(!manifest_text.contains("/home/"));

    for top in [
        "inputs.json",
        "results.json",
        "summary.md",
        "manifest.json",
        "COMPLETE",
    ] {
        let b1 = fs::read(suite1.join(top)).unwrap();
        let b2 = fs::read(suite2.join(top)).unwrap();
        assert_eq!(b1, b2, "batch2-suite/{top} must be byte-identical");
        if top == "COMPLETE" {
            assert!(b1.is_empty());
        } else {
            let s = String::from_utf8(b1.clone()).unwrap();
            assert!(!s.contains("/home/"), "batch2-suite/{top} leaked /home/");
            if top != "manifest.json" {
                let digest = sha256_hex(&b1);
                assert!(
                    manifest_text.contains(&digest),
                    "manifest.json missing top-level sha256 for {top}"
                );
            }
        }
    }

    // 2. Each standalone subcommand produces byte-identical files matching its batch2-suite subdir!
    for (cmd, subdir) in subdirs {
        let single = base.join(format!("single-{cmd}"));
        let out = Command::new(bin)
            .args([cmd, "--out", single.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "subcommand {cmd} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        for name in ["inputs.json", "results.json", "summary.md", "COMPLETE"] {
            let bs = fs::read(single.join(name)).unwrap();
            let b1 = fs::read(suite1.join(subdir).join(name)).unwrap();
            let b2 = fs::read(suite2.join(subdir).join(name)).unwrap();
            assert_eq!(
                bs, b1,
                "{cmd}/{name} must match batch2-suite/{subdir}/{name}"
            );
            assert_eq!(
                b1, b2,
                "batch2-suite/{subdir}/{name} must be byte-identical"
            );
            if name == "COMPLETE" {
                assert!(b1.is_empty());
            } else {
                let text = String::from_utf8(b1.clone()).unwrap();
                assert!(!text.contains("/home/"), "{subdir}/{name} leaked /home/");
                assert!(
                    manifest_text.contains(&sha256_hex(&b1)),
                    "manifest.json missing sha256 for {subdir}/{name}"
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&base);
}
