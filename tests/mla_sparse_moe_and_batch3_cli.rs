use std::fs;
use std::process::Command;

use inference_lab::json::sha256_hex;
use inference_lab::mla_comparison::{
    compare_batch_outputs, run_default_harness as run_mla_comparison,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::sparse_moe::{
    max_mat_diff, max_vec_diff, run_default_harness as run_sparse_moe,
};

#[test]
fn mla_comparison_proves_zero_expanded_kv_intermediates_poisoned_cache_safety_and_controls() {
    let report = run_mla_comparison().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(!report.element_count_reduction_is_measured_speedup);
    assert_eq!(report.default_dim_latent_elems, 144);
    assert_eq!(report.default_dim_expanded_elems, 480);
    assert!((report.default_dim_element_reduction_pct - 70.0).abs() < 1e-12);

    // Absorbed MLA matches uncached and explicit within 1e-11 AND constructs 0 per-head K/V intermediates!
    assert!(report.explicit_vs_uncached_max_err <= 1e-11);
    assert!(report.absorbed_vs_uncached_max_err <= 1e-11);
    assert!(report.absorbed_vs_explicit_max_err <= 1e-11);
    assert_eq!(report.absorbed_materialized_kv_elems, 0);
    assert!(report.explicit_materialized_kv_elems > 0);
    assert!(report.absorbed_prep_bytes > 0);
    assert!(report.gradient_fd_parity_max_err <= 1e-9);

    // compare_batch_outputs rejects NaN and shape mismatches (never swallows NaN).
    assert!(compare_batch_outputs(&[vec![vec![f64::NAN]]], &[vec![vec![0.0]]], 1e-6).is_err());
    assert!(compare_batch_outputs(&[vec![vec![1.0]]], &[vec![vec![1.0, 2.0]]], 1e-6).is_err());

    // Strategy accounting distinguishes recompilations across full vs valid-prefix vs bucketed.
    assert_eq!(report.strategy_accounting.len(), 3);
    assert_eq!(report.strategy_accounting[0].shape_recompilations, 1); // full_allocated_cache
    assert!(report.strategy_accounting[1].shape_recompilations > 1); // valid_prefix_only
    for s in &report.strategy_accounting {
        assert_eq!(s.absorbed_materialized_kv_elems, 0);
        assert!(s.explicit_materialized_kv_elems > 0);
    }

    // Positive control passes; all 6 negative controls reject with first divergent position!
    assert_eq!(report.controls[0].outcome, OutcomeStatus::Accepted);
    for neg in &report.controls[1..] {
        assert_eq!(
            neg.outcome,
            OutcomeStatus::Invalid,
            "MLA negative control {} must fail",
            neg.case_id
        );
        assert!(
            neg.first_divergent_pos.is_some(),
            "MLA negative control {} must report first divergent position",
            neg.case_id
        );
    }
}

#[test]
fn sparse_moe_matches_dense_and_tiny_forward_analytical_gradients_and_catches_all_controls() {
    let report = run_sparse_moe().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.dense_vs_tiny_max_err <= 1e-11);
    assert!(report.dense_vs_sparse_forward_max_err <= 1e-11);
    assert!(report.dense_vs_sparse_grad_max_err <= 1e-11);
    assert!(report.analytical_vs_fd_grad_max_err <= 1e-7);
    assert!(report.three_step_update_parity_max_err <= 1e-10);
    assert!(report.bf16_underflow_preserved_assignment_count);

    for fix in &report.differential_fixtures {
        assert_eq!(
            fix.outcome,
            OutcomeStatus::Accepted,
            "fixture {} failed",
            fix.case_id
        );
    }
    assert_eq!(report.negative_controls.len(), 7);
    for neg in &report.negative_controls {
        assert_eq!(
            neg.outcome,
            OutcomeStatus::Invalid,
            "negative control {} must fail",
            neg.case_id
        );
    }
    assert_eq!(report.sweep_rows.len(), 4);
    // max_mat_diff / max_vec_diff reject NaN and length mismatches cleanly.
    assert!(max_vec_diff(&[f64::NAN], &[0.0]).is_err());
    assert!(max_vec_diff(&[1.0], &[1.0, 2.0]).is_err());
    assert!(max_mat_diff(&[vec![f64::INFINITY]], &[vec![0.0]]).is_err());
}

#[test]
fn batch3_cli_commands_and_suite_are_byte_identical_leak_free_and_refuse_overwrite() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!(
        "inference-lab-batch3-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let suite1 = base.join("suite-run1");
    let suite2 = base.join("suite-run2");

    for dir in [&suite1, &suite2] {
        let out = Command::new(bin)
            .args(["batch3-suite", "--out", dir.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "batch3-suite failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // Overwrite refusal.
    let overwrite = Command::new(bin)
        .args(["batch3-suite", "--out", suite1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!overwrite.status.success());

    let subdirs = [
        ("rsi-bench", "issue-12-rsi-bench"),
        ("djev-protocol", "issue-13-djev-protocol"),
        ("rwkv-leaf", "issue-14-rwkv-leaf"),
        ("ggml-ops", "issue-15-ggml-ops"),
        ("rwkv-ggml", "issue-16-rwkv-ggml"),
        ("graphical-sampling", "issue-17-graphical-sampling"),
        ("qwen38-quant", "issue-18-qwen38-quant"),
        ("deepseek-ref", "issue-19-deepseek-ref"),
        ("mla-comparison", "issue-20-mla-comparison"),
        ("sparse-moe", "issue-21-sparse-moe"),
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
        assert_eq!(b1, b2, "batch3-suite/{top} must be byte-identical");
        if top == "COMPLETE" {
            assert!(b1.is_empty());
        } else {
            let s = String::from_utf8(b1.clone()).unwrap();
            assert!(!s.contains("/home/"), "batch3-suite/{top} leaked /home/");
            if top != "manifest.json" {
                let digest = sha256_hex(&b1);
                assert!(
                    manifest_text.contains(&digest),
                    "manifest.json missing top-level sha256 for {top}"
                );
            }
        }
    }

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
                "{cmd}/{name} must match batch3-suite/{subdir}/{name}"
            );
            assert_eq!(
                b1, b2,
                "batch3-suite/{subdir}/{name} must be byte-identical"
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
